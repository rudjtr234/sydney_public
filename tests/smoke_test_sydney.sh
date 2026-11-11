#!/usr/bin/env bash
# 가짜 임베딩으로 Sydney split 생성(--code_col, --grade_cols 두 방식이 같은 Grade를 내는지 확인)
# -> Multi-task ABMIL (ce, coral) 학습을 짧게 돌려 파이프라인 전체를 점검
#   bash tests/smoke_test_sydney.sh [work_dir] [gpu]
set -euo pipefail
export NUMPY_MADVISE_HUGEPAGE=0

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-python}
WORK=${1:-$(mktemp -d)}
GPU=${2:-0}
EPOCHS=${EPOCHS:-3}
cd "$PROJECT_DIR"

"$PYTHON" tests/make_synthetic_sydney.py --out_dir "$WORK"
common=(--metadata_csv "$WORK/metadata.csv" --patient_col patient_id --k 3 --feat_dir "$WORK/features_synthetic")
"$PYTHON" -m src.data.make_sydney_splits "${common[@]}" --code_col sydney_code --out "$WORK/splits_code.json"
"$PYTHON" -m src.data.make_sydney_splits "${common[@]}" --grade_cols hp=HP neu=NEU mono=MONO atr=ATR im=IM \
    --out "$WORK/splits_cols.json" > /dev/null
"$PYTHON" - "$WORK/splits_code.json" "$WORK/splits_cols.json" <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:])
assert a["labels"] == b["labels"], "--code_col and --grade_cols gave different grades"
assert a["folds"] == b["folds"], "--code_col and --grade_cols gave different folds"
n_missing = sum(g[3] < 0 for g in a["labels"].values())
assert n_missing > 0, "atrophy N/A was not parsed as missing"
print(f"[OK] --code_col and --grade_cols agree on {len(a['labels'])} slides ({n_missing} with atrophy missing)")
EOF

for head in ce coral; do
    out="$WORK/out_$head"
    "$PYTHON" -m src.training.train_sydney --split_file "$WORK/splits_code.json" \
        --feat_dir "$WORK/features_synthetic" --head "$head" --out_dir "$out" --class_weighted \
        --epochs "$EPOCHS" --patience "$EPOCHS" --bag_size 128 --num_workers 2 --gpu "$GPU" > "$out.log" 2>&1 \
        || { echo "[FAIL] $head -> $out.log"; tail -30 "$out.log"; exit 1; }
    for f in summary.csv fold1/test_predictions.csv fold1/best_model.pt; do
        [[ -f "$out/$f" ]] || { echo "[FAIL] $head: missing $out/$f"; exit 1; }
    done
    echo "[OK] $head: $(grep 'mean_qwk' "$out.log" | tail -1 | sed 's/.*|  *mean_qwk/mean_qwk/')"
done
echo "smoke test passed ($WORK)"
