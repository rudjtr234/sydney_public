# tests/make_synthetic_features.py
# 학습 코드 점검용 가짜 TRIDENT 임베딩 생성.
# 양성(또는 class c) 슬라이드는 일부 패치가 class별 방향으로 이동해 있어서, 코드가 맞으면 AUC가 0.5보다 높게 나온다.

import argparse
import os

import h5py
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_patients", type=int, default=40)
    p.add_argument("--n_classes", type=int, default=2)
    p.add_argument("--dim", type=int, default=1536)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    feat_dir = os.path.join(args.out_dir, "features_synthetic")
    os.makedirs(feat_dir, exist_ok=True)
    directions = rng.normal(size=(args.n_classes, args.dim)).astype(np.float32)

    rows = []
    for pid in range(args.n_patients):
        label = pid % args.n_classes
        for s in range(1 + (pid % 3 == 0)):  # every third patient has two slides
            slide_id = f"P{pid:03d}_S{s}"
            n = int(rng.integers(50, 400))
            feats = rng.normal(size=(n, args.dim)).astype(np.float32)
            hot = rng.random(n) < 0.2
            feats[hot] += 0.5 * directions[label]
            coords = np.stack([rng.integers(0, 50000, n), rng.integers(0, 50000, n)], axis=1).astype(np.int64)
            with h5py.File(os.path.join(feat_dir, f"{slide_id}.h5"), "w") as f:
                f.create_dataset("features", data=feats).attrs["encoder"] = "synthetic"
                f.create_dataset("coords", data=coords)
            rows.append({"slide_id": slide_id, "patient_id": f"P{pid:03d}", "label": f"class_{label}"})

    rows.append({"slide_id": "P999_S0", "patient_id": "P999", "label": "class_0"})  # no features -> must be dropped
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "metadata.csv"), index=False)
    print(f"Wrote {len(rows) - 1} slides to {feat_dir} and {args.out_dir}/metadata.csv")


if __name__ == "__main__":
    main()
