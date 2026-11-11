# src/training/split_report.py
# 분할에 사례가 어떻게 들어갔는지: fold × (train / val / test) × 항목별 등급·생검 부위 건수
#
# 사용 (sydney env python, 프로젝트 폴더에서):
#   python -m src.training.split_report --run_dir outputs/sydney_mtl_hoptimus0_20x_v0.1.0   # 이미 돌린 실험의 splits.json
#   python -m src.training.split_report --split multilabel                                  # 학습 전 미리보기 (v0.3.0 분할)
#   python -m src.training.split_report --split code                                        # 학습 전 미리보기 (논문 분할)
# 화면: test·val에서 칸마다 fold 1~5 건수와 이상값(전체 × 비율), 드문 칸(전체 60건 이하)만. --all이면 전부
# 파일: --run_dir를 주면 그 폴더에 split_counts.csv (fold, part, n, 항목·등급 20칸, 부위 4칸). --out으로 다른 곳에
import argparse
import json
import os

import numpy as np
import pandas as pd

from src.training.train_sydney_mtl import (NUM_GRADES, SITES, TASKS, load_cases, make_folds,  # noqa: E402
                                           make_folds_multilabel)

COLS = [f"{t}{g}" for t in TASKS for g in range(NUM_GRADES)] + [f"site_{s}" for s in SITES]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run_dir", help="Training output folder with splits.json")
    src.add_argument("--split", choices=["code", "multilabel"], help="Preview a split without training")
    p.add_argument("--meta_csv", default="metadata/sydney_meta.csv")
    p.add_argument("--label_col", default="code_review")
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--val_size", type=float, default=0.125)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--all", action="store_true", help="Show every cell, not only the rare ones")
    p.add_argument("--out", help="CSV path (default: <run_dir>/split_counts.csv, none for a preview)")
    return p.parse_args()


def onehot(cases):
    Y = np.zeros((len(cases), len(COLS)), dtype=int)
    for i, c in enumerate(cases):
        for j, g in enumerate(c["grades"]):
            Y[i, j * NUM_GRADES + g] = 1
        Y[i, len(TASKS) * NUM_GRADES + c["site"] - 1] = 1
    return Y


def main():
    args = parse_args()
    if args.run_dir:
        with open(os.path.join(args.run_dir, "config.json")) as f:
            cfg = json.load(f)
        args.meta_csv, args.label_col = cfg["meta_csv"], cfg["label_col"]
        with open(os.path.join(args.run_dir, "splits.json")) as f:
            folds = json.load(f)["folds"]
        title = f"{args.run_dir} ({cfg.get('split_method', 'code_stratified')})"
    cases = load_cases(args.meta_csv, args.label_col)
    pos = {c["case_id"]: i for i, c in enumerate(cases)}
    if args.split:
        make = make_folds_multilabel(cases, args.kfold, args.val_size, args.seed) if args.split == "multilabel" \
            else make_folds([c["code"] for c in cases], args.kfold, args.val_size, args.seed)
        folds = [{"fold": fd["fold"], **{k: [cases[i]["case_id"] for i in fd[k]] for k in ("train", "val", "test")}}
                 for fd in make]
        title = f"preview --split {args.split} (seed {args.seed})"
    unknown = {c for fd in folds for k in ("train", "val", "test") for c in fd[k] if c not in pos}
    if unknown:
        raise SystemExit(f"{len(unknown)} cases in splits.json are not in {args.meta_csv} (labels changed?)")

    Y = onehot(cases)
    rows = []
    for fd in folds:
        for part in ("train", "val", "test"):
            idx = [pos[c] for c in fd[part]]
            rows.append({"fold": fd["fold"], "part": part, "n": len(idx), **dict(zip(COLS, Y[idx].sum(0).tolist()))})
    df = pd.DataFrame(rows)

    print(f"{title}\n{len(cases)} cases | train " + " ".join(str(n) for n in df[df.part == "train"].n)
          + " | val " + " ".join(str(n) for n in df[df.part == "val"].n)
          + " | test " + " ".join(str(n) for n in df[df.part == "test"].n))
    total = dict(zip(COLS, Y.sum(0).tolist()))
    for part in ("test", "val"):
        sub = df[df.part == part]
        share = sub.n.mean() / len(cases)
        print(f"\n[{part}] 칸 (전체) : fold 1~{len(folds)} 건수  | 이상값")
        devs = []
        for col in COLS:
            counts = sub[col].tolist()
            devs.append(np.abs(np.array(counts) - total[col] * share).mean())
            if args.all or total[col] <= 60:
                print(f"  {col:<12}({total[col]:>3}) : " + " ".join(f"{x:>3}" for x in counts)
                      + f"  | {total[col] * share:5.1f}")
        print(f"  평균 |건수 - 이상값| ({len(COLS)}칸): {np.mean(devs):.2f}")

    out = args.out or (os.path.join(args.run_dir, "split_counts.csv") if args.run_dir else None)
    if out:
        df.to_csv(out, index=False)
        print(f"\n-> {out}")


if __name__ == "__main__":
    main()
