#!/usr/bin/env bash
# Sydney Multi-task ABMIL: 인코더 x head(ce, coral) 그리드. GPU마다 한 실험씩 (scripts/run_train.sh와 같은 방식)
# 결과: $OUT_ROOT/<VERSION>/<encoder>_<head>/  (summary.csv에 fold별 항목별 test 지표 + mean/std)
#
# 사용 예:
#   SPLIT_FILE=splits/sydney_k5_seed42.json VERSION=sydney_v0.1.0 \
#       nohup bash scripts/run_train_sydney.sh > logs/train_sydney_v0.1.0.log 2>&1 &
#   ENCODERS="hoptimus0" HEADS="ce" TRAIN_ARGS="--class_weighted" bash scripts/run_train_sydney.sh
set -euo pipefail
umask 000  # NFS: 계정이 달라도 결과를 지우고 덮어쓸 수 있도록
export NUMPY_MADVISE_HUGEPAGE=0  # 큰 배열 할당이 THP compaction으로 멈추는 것 방지 (scripts/extract_embeddings.sh 참고)

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-python}
SPLIT_FILE=${SPLIT_FILE:?SPLIT_FILE=<make_sydney_splits.py output json> is required}
VERSION=${VERSION:?VERSION=<e.g. sydney_v0.1.0> is required}
FEAT_ROOT=${FEAT_ROOT:-/path/to/sydney/embedding/20x_224px_0px_overlap}
OUT_ROOT=${OUT_ROOT:-$PROJECT_DIR/outputs}
ENCODERS=${ENCODERS:-"hibou_l hoptimus0 virchow midnight12k"}
HEADS=${HEADS:-"ce coral"}
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
    for head in $HEADS; do
        out_dir="$OUT_ROOT/$VERSION/${enc}_${head}"
        mkdir -p "$out_dir"
        wait_for_free_gpu
        echo "[$(date '+%F %T')] GPU $FREE_GPU: $enc + $head -> $out_dir"
        "$PYTHON" -m src.training.train_sydney --head "$head" --split_file "$SPLIT_FILE" \
            --feat_dir "$FEAT_ROOT/features_$enc" --out_dir "$out_dir" --gpu "$FREE_GPU" "${EXTRA[@]}" \
            > "$out_dir/stdout.log" 2>&1 &
        GPU_PID[$FREE_GPU]=$!
    done
done
wait

echo "[$(date '+%F %T')] all done. Test mean ± std:"
for enc in $ENCODERS; do
    for head in $HEADS; do
        f="$OUT_ROOT/$VERSION/${enc}_${head}/summary.json"
        if [[ ! -f "$f" ]]; then
            echo "  ${enc}_${head}  FAILED -> see $(dirname "$f")/stdout.log"
            continue
        fi
        "$PYTHON" - "$f" "${enc}_${head}" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))["summary"]
tasks = [c[:-4] for c in s if c.endswith("_qwk") and c != "mean_qwk"]
per_task = " ".join(f"{t} {s[t + '_qwk']['mean']:.3f}" for t in tasks)
print(f"  {sys.argv[2]:<22} mean_qwk {s['mean_qwk']['mean']:.4f} ± {s['mean_qwk']['std']:.4f}  "
      f"code_exact {s['code_exact']['mean']:.4f} | qwk {per_task}")
EOF
    done
done
