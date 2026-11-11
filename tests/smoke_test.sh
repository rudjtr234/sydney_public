#!/usr/bin/env bash
# 가짜 임베딩으로 split 생성 -> 5개 MIL 모델 x (이진, 3클래스) 학습을 짧게 돌려 파이프라인 전체를 점검
#   bash tests/smoke_test.sh [work_dir] [gpu]
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-python}
WORK=${1:-$(mktemp -d)}
GPU=${2:-0}
cd "$PROJECT_DIR"

for n_classes in 2 3; do
    data="$WORK/c$n_classes"
    "$PYTHON" tests/make_synthetic_features.py --out_dir "$data" --n_classes "$n_classes" --dim 256
    "$PYTHON" -m src.data.make_splits --metadata_csv "$data/metadata.csv" --label_col label \
        --patient_col patient_id --k 3 --feat_dir "$data/features_synthetic" --out "$data/splits.json"
    for model in abmil clam_sb dsmil transmil acmil; do
        "$PYTHON" -m src.training.train --model_name "$model" --split_file "$data/splits.json" \
            --feat_dir "$data/features_synthetic" --out_dir "$data/out_$model" \
            --epochs 3 --patience 3 --bag_size 128 --num_workers 2 --gpu "$GPU" > "$data/out_$model.log" 2>&1 \
            || { echo "[FAIL] $model ($n_classes classes) -> $data/out_$model.log"; tail -20 "$data/out_$model.log"; exit 1; }
        echo "[OK] $model ($n_classes classes): $(tail -4 "$data/out_$model.log" | grep auc | sed 's/.*| *//')"
    done
done
echo "smoke test passed ($WORK)"
