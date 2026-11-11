# src/data/make_sydney_splits.py
# Sydney 메타데이터 CSV -> 슬라이드별 5개 항목 Grade(0~3) + K-fold split JSON (환자 단위)
#
# 라벨은 둘 중 하나로 준다.
#   --code_col <열>           "02202" 같은 Sydney CODE 한 열. 자리 순서 = --tasks (기본 hp neu mono atr im)
#   --grade_cols hp=<열> ...  항목별 Grade 열
# 0~3이 아닌 값(빈칸, N/A, X, - ...)은 결측(-1): 그 슬라이드의 그 항목만 loss·지표에서 빠진다.
# 5개 항목이 모두 결측인 슬라이드는 뺀다.
#
# split 방식은 make_splits.py와 같다 (fold마다 test 1개 그룹, 나머지에서 val_ratio만큼 val, 환자 단위).
# 층화 키는 --stratify_by 항목들의 Grade 조합이고, k개 미만인 조합은 'rare' 하나로 묶는다.
#
# 사용 예:
#   python -m src.data.make_sydney_splits --metadata_csv metadata/sydney_meta.csv \
#       --code_col sydney_code --patient_col patient_id --k 5 --seed 42 \
#       --feat_dir /path/to/sydney/embedding/20x_224px_0px_overlap/features_hoptimus0 \
#       --out splits/sydney_k5_seed42.json

import argparse
import json
import os
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from src.data.make_splits import stratified_group_split

# Updated Sydney System 5개 항목. 이 순서가 기본 CODE 자리 순서 (라벨 정의를 확인하고 다르면 --tasks로 바꾼다)
SYDNEY_TASKS = {
    "hp": "H. pylori",
    "neu": "Neutrophil activity",
    "mono": "Mononuclear cells (chronic inflammation)",
    "atr": "Glandular atrophy",
    "im": "Intestinal metaplasia",
}
NUM_GRADES = 4  # 0 none, 1 mild, 2 moderate, 3 marked
MISSING = -1
GRADE_WORDS = {"none": 0, "mild": 1, "moderate": 2, "marked": 3, "severe": 3}


def parse_args():
    p = argparse.ArgumentParser(description="Sydney grades (5 items x 0..3) + patient-level K-fold splits")
    p.add_argument("--metadata_csv", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--code_col", help="Sydney CODE column, one character per item in --tasks order (e.g. 02202)")
    src.add_argument("--grade_cols", nargs="+", metavar="TASK=COL",
                     help="Per-item grade columns, e.g. hp=HP neu=NEU mono=MONO atr=ATR im=IM")
    p.add_argument("--tasks", nargs="+", default=list(SYDNEY_TASKS), choices=list(SYDNEY_TASKS),
                   help="Item order = CODE character order (default: hp neu mono atr im)")
    p.add_argument("--slide_col", default="slide_id",
                   help="Slide id column = WSI filename without extension = TRIDENT .h5 stem")
    p.add_argument("--patient_col", default=None,
                   help="Group column; slides of one patient never cross train/val/test. Default: each slide is its own group")
    p.add_argument("--stratify_by", nargs="+", default=None, choices=list(SYDNEY_TASKS),
                   help="Items whose grade combination is the stratification key (default: all --tasks)")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--val_ratio", type=float, default=0.2, help="Fraction of the non-test part used for validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--feat_dir", default=None,
                   help="If given, slides without <feat_dir>/<slide_id>.h5 are dropped (and reported)")
    p.add_argument("--out", required=True)
    return p.parse_args()


def to_grade(value) -> int:
    """One grade cell -> 0..3. Anything else (blank, N/A, X, -, 4) -> MISSING."""
    if pd.isna(value):
        return MISSING
    s = str(value).strip().lower()
    if s in GRADE_WORDS:
        return GRADE_WORDS[s]
    try:
        f = float(s)
    except ValueError:
        return MISSING
    return int(f) if f in (0, 1, 2, 3) else MISSING


def parse_code(code: str, n_tasks: int):
    """'02202' -> [0, 2, 2, 0, 2]; non-0..3 characters (X, N, -) -> MISSING; wrong length -> None."""
    if code.isdigit() and len(code) < n_tasks:  # Excel을 거치면 앞자리 0이 빠진다 (02202 -> 2202)
        code = code.zfill(n_tasks)
    if len(code) != n_tasks:
        return None
    return [int(c) if c in "0123" else MISSING for c in code]


def grade_counts(grades: np.ndarray, tasks):
    return {t: {**{str(g): int((grades[:, i] == g).sum()) for g in range(NUM_GRADES)},
                "missing": int((grades[:, i] == MISSING).sum())}
            for i, t in enumerate(tasks)}


def main():
    args = parse_args()
    tasks = args.tasks
    if len(set(tasks)) != len(tasks):
        sys.exit(f"--tasks has duplicates: {tasks}")
    stratify_by = args.stratify_by or tasks
    if set(stratify_by) - set(tasks):
        sys.exit(f"--stratify_by {stratify_by} must be a subset of --tasks {tasks}")

    grade_cols = None
    if args.grade_cols:
        grade_cols = dict(kv.split("=", 1) for kv in args.grade_cols)
        if set(grade_cols) != set(tasks):
            sys.exit(f"--grade_cols must give exactly one column per task {tasks}, got {list(grade_cols)}")

    df = pd.read_csv(args.metadata_csv, dtype=str)  # 전부 문자열로 읽어야 CODE 앞자리 0이 보존된다
    label_cols = [args.code_col] if args.code_col else [grade_cols[t] for t in tasks]
    for col in [args.slide_col] + label_cols + ([args.patient_col] if args.patient_col else []):
        if col not in df.columns:
            sys.exit(f"Column '{col}' not in {args.metadata_csv}. Columns: {list(df.columns)}")

    dup = df[args.slide_col][df[args.slide_col].duplicated()].unique()
    if len(dup):
        sys.exit(f"Duplicated slide ids in metadata: {list(dup)[:10]}")

    if args.feat_dir:
        has_feat = df[args.slide_col].map(lambda s: os.path.exists(os.path.join(args.feat_dir, f"{s}.h5")))
        missing = df.loc[~has_feat, args.slide_col].tolist()
        if missing:
            print(f"[!] {len(missing)} slides have no features in {args.feat_dir} and are dropped: {missing[:10]}"
                  f"{' ...' if len(missing) > 10 else ''}")
        df = df[has_feat]

    if args.code_col:
        codes = df[args.code_col].fillna("").str.strip()
        padded = codes.str.isdigit() & (codes.str.len() < len(tasks))
        if padded.any():
            print(f"[!] {int(padded.sum())} codes are shorter than {len(tasks)} digits and were zero-padded "
                  f"(e.g. {codes[padded].tolist()[:5]}). Check that the leading 0s were lost, not something else.")
        parsed = codes.map(lambda c: parse_code(c, len(tasks)))
        bad = parsed.isna()
        if bad.any():
            print(f"[!] Dropped {int(bad.sum())} rows whose '{args.code_col}' is not a {len(tasks)}-character code: "
                  f"{codes[bad].tolist()[:10]}")
        df, parsed = df[~bad], parsed[~bad]
        grades = np.array(parsed.tolist(), dtype=int).reshape(-1, len(tasks))
    else:
        grades = np.stack([df[grade_cols[t]].map(to_grade).to_numpy(dtype=int) for t in tasks], axis=1)

    empty = (grades == MISSING).all(axis=1)
    if empty.any():
        print(f"[!] Dropped {int(empty.sum())} slides with no valid grade: {df[args.slide_col][empty].tolist()[:10]}")
        df, grades = df[~empty], grades[~empty]

    slide_ids = df[args.slide_col].to_numpy()
    groups = (df[args.patient_col] if args.patient_col else df[args.slide_col]).to_numpy()
    print(f"Slides: {len(df)}, groups: {len(set(groups))}, tasks (= CODE order): {tasks}")
    counts = grade_counts(grades, tasks)
    for t in tasks:
        c = counts[t]
        print(f"  {t:<5} " + " ".join(f"G{g}={c[str(g)]:<4}" for g in range(NUM_GRADES)) + f" missing={c['missing']}")
        rare = [g for g in range(NUM_GRADES) if 0 < c[str(g)] < args.k]
        if rare:
            print(f"  [!] {t}: grade {rare} has < k={args.k} slides; some test folds will miss it")

    # 층화 키: stratify_by 항목들의 Grade 조합 (결측 = x). 조합이 k개 미만이면 항목별 유무(0 / 1+) 조합으로,
    # 그것도 k개 미만이면 'rare' 하나로 묶는다
    s_grades = grades[:, [tasks.index(t) for t in stratify_by]]
    fine = ["".join("x" if g == MISSING else str(g) for g in row) for row in s_grades]
    coarse = ["".join("x" if g == MISSING else "+" if g > 0 else "0" for g in row) for row in s_grades]
    fine_n = Counter(fine)
    coarse_n = Counter(c for f, c in zip(fine, coarse) if fine_n[f] < args.k)
    keys = [f if fine_n[f] >= args.k else f"~{c}" if coarse_n[c] >= args.k else "rare" for f, c in zip(fine, coarse)]
    key_ids = {k: i for i, k in enumerate(sorted(set(keys)))}
    y = np.array([key_ids[k] for k in keys])
    n_coarse = sum(k.startswith("~") for k in keys)
    print(f"Stratify by {stratify_by}: {len(key_ids)} strata | slides by exact grades "
          f"{len(keys) - n_coarse - keys.count('rare')}, by 0/1+ {n_coarse}, 'rare' {keys.count('rare')}")

    warnings.filterwarnings("ignore", message="The least populated class")  # 작은 층은 당연히 n_splits보다 적다
    n_val_splits = max(2, int(round(1 / args.val_ratio)))
    all_idx = np.arange(len(df))
    folds = []
    for fold_i, (rest_idx, test_idx) in enumerate(stratified_group_split(all_idx, y, groups, args.k, args.seed), start=1):
        train_idx, val_idx = next(stratified_group_split(rest_idx, y, groups, n_val_splits, args.seed + fold_i))

        parts = {"train": train_idx, "val": val_idx, "test": test_idx}
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            assert not set(slide_ids[parts[a]]) & set(slide_ids[parts[b]]), f"fold {fold_i}: slide overlap {a}/{b}"
            assert not set(groups[parts[a]]) & set(groups[parts[b]]), f"fold {fold_i}: group overlap {a}/{b}"

        fold = {"fold": fold_i}
        for name, idx in parts.items():
            fold[name] = sorted(slide_ids[idx].tolist())
            fold[f"{name}_grade_counts"] = grade_counts(grades[idx], tasks)
        folds.append(fold)
        test_c = fold["test_grade_counts"]
        print(f"Fold {fold_i}: " + " | ".join(f"{n}={len(fold[n])}" for n in parts) + " | test G0/G1/G2/G3: "
              + " ".join(f"{t} " + "/".join(str(test_c[t][str(g)]) for g in range(NUM_GRADES)) for t in tasks))

    out = {
        "metadata_csv": os.path.abspath(args.metadata_csv),
        "slide_col": args.slide_col,
        "patient_col": args.patient_col,
        "label_source": {"code_col": args.code_col} if args.code_col else {"grade_cols": grade_cols},
        "tasks": tasks,
        "task_names": {t: SYDNEY_TASKS[t] for t in tasks},
        "num_grades": NUM_GRADES,
        "stratify_by": stratify_by,
        "k": args.k,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "labels": {s: [int(g) for g in row] for s, row in zip(slide_ids, grades)},
        "folds": folds,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[✓] Saved {args.out}")


if __name__ == "__main__":
    main()
