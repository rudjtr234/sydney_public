# src/data/make_splits.py
# 메타데이터 CSV -> K-fold split JSON (train/val/test, 환자 단위, 라벨 층화)
#
# 각 fold:  test = K개 그룹 중 1개,  val = 나머지에서 val_ratio 만큼,  train = 그 외
# split은 slide_id만 저장하므로 4개 인코더 임베딩에 같은 split을 그대로 쓸 수 있다.
#
# 사용 예:
#   python -m src.data.make_splits --metadata_csv meta.csv --label_col label \
#       --patient_col patient_id --k 5 --seed 42 \
#       --feat_dir /path/to/sydney/embedding/20x_224px_0px_overlap/features_hoptimus0 \
#       --out splits/label_k5_seed42.json

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def parse_args():
    p = argparse.ArgumentParser(description="Make patient-level stratified K-fold splits")
    p.add_argument("--metadata_csv", required=True)
    p.add_argument("--label_col", required=True, help="Label column (strings or ints)")
    p.add_argument("--slide_col", default="slide_id",
                   help="Slide id column = WSI filename without extension = TRIDENT .h5 stem")
    p.add_argument("--patient_col", default=None,
                   help="Group column; slides of one patient never cross train/val/test. Default: each slide is its own group")
    p.add_argument("--label_order", nargs="+", default=None,
                   help="Explicit class order, e.g. --label_order negative positive (index = class id). "
                        "Default: sorted unique values")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--val_ratio", type=float, default=0.2, help="Fraction of the non-test part used for validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--feat_dir", default=None,
                   help="If given, slides without <feat_dir>/<slide_id>.h5 are dropped (and reported)")
    p.add_argument("--out", required=True)
    return p.parse_args()


def stratified_group_split(idx, y, groups, n_splits, seed):
    """Yield (rest_idx, held_idx) pairs over idx using StratifiedGroupKFold."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for rest, held in sgkf.split(idx, y[idx], groups[idx]):
        yield idx[rest], idx[held]


def main():
    args = parse_args()
    df = pd.read_csv(args.metadata_csv, dtype={args.slide_col: str})

    for col in [args.slide_col, args.label_col] + ([args.patient_col] if args.patient_col else []):
        if col not in df.columns:
            sys.exit(f"Column '{col}' not in {args.metadata_csv}. Columns: {list(df.columns)}")

    n_before = len(df)
    df = df.dropna(subset=[args.label_col])
    if len(df) < n_before:
        print(f"[i] Dropped {n_before - len(df)} rows with empty '{args.label_col}'")

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

    raw_labels = df[args.label_col].astype(str)
    classes = args.label_order or sorted(raw_labels.unique())
    unknown = sorted(set(raw_labels) - set(classes))
    if unknown:
        sys.exit(f"Labels {unknown} are not in --label_order {classes}")
    label_map = {c: i for i, c in enumerate(classes)}

    slide_ids = df[args.slide_col].to_numpy()
    y = raw_labels.map(label_map).to_numpy()
    groups = (df[args.patient_col].astype(str) if args.patient_col else df[args.slide_col]).to_numpy()

    print(f"Slides: {len(df)}, groups: {len(set(groups))}, classes: {label_map}")
    print(f"Label counts: { {c: int((y == i).sum()) for c, i in label_map.items()} }")
    min_count = min(Counter(y).values())
    if min_count < args.k:
        print(f"[!] Smallest class has {min_count} slides < k={args.k}; some test folds will miss that class")

    n_val_splits = max(2, int(round(1 / args.val_ratio)))
    all_idx = np.arange(len(df))
    folds = []
    for fold_i, (rest_idx, test_idx) in enumerate(stratified_group_split(all_idx, y, groups, args.k, args.seed), start=1):
        train_idx, val_idx = next(stratified_group_split(rest_idx, y, groups, n_val_splits, args.seed + fold_i))

        # leakage check on both slide and group level
        parts = {"train": train_idx, "val": val_idx, "test": test_idx}
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            assert not set(slide_ids[parts[a]]) & set(slide_ids[parts[b]]), f"fold {fold_i}: slide overlap {a}/{b}"
            assert not set(groups[parts[a]]) & set(groups[parts[b]]), f"fold {fold_i}: group overlap {a}/{b}"

        fold = {"fold": fold_i}
        for name, idx in parts.items():
            fold[name] = sorted(slide_ids[idx].tolist())
            fold[f"{name}_label_counts"] = {c: int((y[idx] == i).sum()) for c, i in label_map.items()}
        folds.append(fold)
        print(f"Fold {fold_i}: " + " | ".join(f"{n}={len(fold[n])} {fold[f'{n}_label_counts']}" for n in parts))

    out = {
        "metadata_csv": os.path.abspath(args.metadata_csv),
        "label_col": args.label_col,
        "slide_col": args.slide_col,
        "patient_col": args.patient_col,
        "label_map": label_map,
        "k": args.k,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "labels": {s: int(l) for s, l in zip(slide_ids, y)},
        "folds": folds,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[✓] Saved {args.out}")


if __name__ == "__main__":
    main()
