# tests/make_synthetic_sydney.py
# Sydney multi-task 학습 코드 점검용 가짜 TRIDENT 임베딩 + 메타데이터.
# 항목마다 방향 벡터가 있고, Grade g인 슬라이드는 패치의 약 8*g %가 그 방향으로 이동해 있다 -> 코드가 맞으면 QWK > 0.
# 실제 CRF에서 생길 수 있는 경우를 섞는다:
#   앞자리 0이 빠진 CODE(02202 -> 2202), Atrophy 평가불가(CODE의 X / 열의 N/A), 임베딩이 없는 슬라이드

import argparse
import os

import h5py
import numpy as np
import pandas as pd

TASKS = ["hp", "neu", "mono", "atr", "im"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_patients", type=int, default=90)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    feat_dir = os.path.join(args.out_dir, "features_synthetic")
    os.makedirs(feat_dir, exist_ok=True)
    directions = rng.normal(size=(len(TASKS), args.dim)).astype(np.float32)

    rows = []
    for pid in range(args.n_patients):
        for s in range(1 + (pid % 3 == 0)):  # every third patient has two slides
            slide_id = f"P{pid:03d}_S{s}"
            grades = rng.choice(4, size=len(TASKS), p=[0.4, 0.3, 0.2, 0.1])
            n = int(rng.integers(100, 500))
            feats = rng.normal(size=(n, args.dim)).astype(np.float32)
            for t, g in enumerate(grades):
                feats[rng.random(n) < 0.08 * g] += directions[t]
            coords = np.stack([rng.integers(0, 50000, n), rng.integers(0, 50000, n)], axis=1).astype(np.int64)
            with h5py.File(os.path.join(feat_dir, f"{slide_id}.h5"), "w") as f:
                f.create_dataset("features", data=feats).attrs["encoder"] = "synthetic"
                f.create_dataset("coords", data=coords)

            cols = {t.upper(): str(g) for t, g in zip(TASKS, grades)}
            code = "".join(map(str, grades))
            if rng.random() < 0.1:  # atrophy 평가불가
                code = code[:3] + "X" + code[4:]
                cols["ATR"] = "N/A"
            elif code.startswith("0") and len(rows) % 4 == 0:  # Excel이 앞자리 0을 지운 CODE
                code = code.lstrip("0") or "0"
            rows.append({"slide_id": slide_id, "patient_id": f"P{pid:03d}", "sydney_code": code, **cols})

    # no features -> must be dropped by --feat_dir
    rows.append({"slide_id": "P999_S0", "patient_id": "P999", "sydney_code": "01100",
                 **{t.upper(): "0" for t in TASKS}})
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "metadata.csv"), index=False)
    print(f"Wrote {len(rows) - 1} slides to {feat_dir} and {args.out_dir}/metadata.csv")


if __name__ == "__main__":
    main()
