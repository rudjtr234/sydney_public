#!/usr/bin/env bash
# 인코더 x MIL 모델 그리드 학습. GPU마다 한 실험씩 동시에 돌리고, GPU가 비면 다음 실험을 넣는다.
# 결과: $OUT_ROOT/<VERSION>/<encoder>_<model>/  (summary.csv에 fold별 test 지표 + mean/std)
#
# 사용 예:
#   SPLIT_FILE=splits/label_k5_seed42.json VERSION=v0.1.0 nohup bash scripts/run_train.sh > logs/train_v0.1.0.log 2>&1 &
#   ENCODERS="hoptimus0" MODELS="abmil transmil" GPUS="0 1" TRAIN_ARGS="--bag_size 5000 --lr 1e-5" bash scripts/run_train.sh
set -euo pipefail
umask 000  # NFS: 계정이 달라도 결과를 지우고 덮어쓸 수 있도록
export NUMPY_MADVISE_HUGEPAGE=0  # 큰 배열 할당이 THP compaction으로 멈추는 것 방지 (scripts/extract_embeddings.sh 참고)

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-python}
SPLIT_FILE=${SPLIT_FILE:?SPLIT_FILE=<make_splits.py output json> is required}
VERSION=${VERSION:?VERSION=<e.g. v0.1.0> is required}
FEAT_ROOT=${FEAT_ROOT:-/path/to/sydney/embedding/20x_224px_0px_overlap}
OUT_ROOT=${OUT_ROOT:-$PROJECT_DIR/outputs}
ENCODERS=${ENCODERS:-"hibou_l hoptimus0 virchow midnight12k"}
MODELS=${MODELS:-"abmil clam_sb dsmil transmil acmil"}
GPUS=${GPUS:-"0 1 2 3"}
TRAIN_ARGS=${TRAIN_ARGS:-}

read -r -a GPU_LIST <<< "$GPUS"
read -r -a EXTRA <<< "$TRAIN_ARGS"
declare -A GPU_PID=()

wait_for_free_gpu() {
    while true; do
        for g in "${GPU_LIST[@]}"; do
            pid=${GPU_PID[$g]:-}
            if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
                FREE_GPU=$g
                return
            fi
        done
        sleep 10
    done
}

cd "$PROJECT_DIR"
for enc in $ENCODERS; do
    for model in $MODELS; do
        out_dir="$OUT_ROOT/$VERSION/${enc}_${model}"
        mkdir -p "$out_dir"
        wait_for_free_gpu
        echo "[$(date '+%F %T')] GPU $FREE_GPU: $enc + $model -> $out_dir"
        "$PYTHON" -m src.training.train --model_name "$model" --split_file "$SPLIT_FILE" \
            --feat_dir "$FEAT_ROOT/features_$enc" --out_dir "$out_dir" --gpu "$FREE_GPU" "${EXTRA[@]}" \
            > "$out_dir/stdout.log" 2>&1 &
        GPU_PID[$FREE_GPU]=$!
    done
done
wait

echo "[$(date '+%F %T')] all done. Test AUC (mean ± std):"
for enc in $ENCODERS; do
    for model in $MODELS; do
        f="$OUT_ROOT/$VERSION/${enc}_${model}/summary.json"
        if [[ ! -f "$f" ]]; then
            echo "  ${enc}_${model}  FAILED -> see $(dirname "$f")/stdout.log"
            continue
        fi
        "$PYTHON" -c "import json,sys; s=json.load(open(sys.argv[1]))['summary']; print(f\"  {sys.argv[2]:<24} auc {s['auc']['mean']:.4f} ± {s['auc']['std']:.4f}  bacc {s['balanced_accuracy']['mean']:.4f}\")" "$f" "${enc}_${model}"
    done
done
