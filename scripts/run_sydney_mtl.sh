#!/usr/bin/env bash
# Sydney SydneyMTL 학습 실행: 학습을 백그라운드로 띄우고 바로 실시간 모니터(scripts/watch_sydney.sh)를 연다.
#   - 모니터에서 Ctrl+C를 눌러도 학습은 계속된다 (다시 보기: bash scripts/watch_sydney.sh)
#   - 인코더를 여러 개 주면 차례대로 학습한다 (같은 VERSION = 같은 설정으로 FM 비교)
#   - 인코더마다 학습이 끝나면 MLflow (experiment "sydney")에 자동으로 올라간다 (train_sydney_mtl.py)
#   - 결과: outputs/sydney_mtl_<encoder>_<MAG>x_v<VERSION>/, 실행 기록: outputs/launch_logs/
#
# 사용:
#   VERSION=0.1.0 ENCODERS=hoptimus0 GPU=0 bash scripts/run_sydney_mtl.sh
#   VERSION=0.1.0 ENCODERS="hoptimus0 midnight12k hibou_l virchow" MAG=10 GPU=5 bash scripts/run_sydney_mtl.sh   # 보조 서버 (10GB GPU)
#   VERSION=0.1.0 ENCODERS="hoptimus0 virchow midnight12k" GPU=0 bash scripts/run_sydney_mtl.sh
#   VERSION=0.1.0 ENCODERS=hibou_l MAG=40 GPU=0 bash scripts/run_sydney_mtl.sh   # 40x 임베딩
# 환경 변수 (기본값):
#   VERSION (필수, x.y.z)  ENCODERS (hibou_l)  MAG (20)  LABEL_COL (code_review)  LABEL_SOURCE ("CRF v2")
#   GPU (nvidia-smi 번호)  CPUS (taskset 코어 범위)  HOST_PROFILE (large|small: 서버별 GPU/CPUS 기본값)
#   EMB_ROOT (임베딩 루트)  PYTHON / MLFLOW_PYTHON (인터프리터)
#   LOSS (la)  OUTPUT_ROOT (outputs)  WATCH (1, 0이면 모니터 없이 띄우기만)
#   EXTRA_ARGS: train_sydney_mtl.py에 그대로 넘길 옵션 (예: "--epochs 50", "--no_mlflow")
# mlflow 가 없는 서버에서는 업로드를 끄고, 끝난 뒤 시작할 때 출력되는 명령으로 mlflow 가 있는 서버에서 올린다.
# 학습 중단: 시작할 때 출력되는 kill 명령 (프로세스 그룹 전체를 멈춤)
set -euo pipefail
umask 000  # 어느 계정이 돌려도 서로 결과를 정리할 수 있도록
export NUMPY_MADVISE_HUGEPAGE=0  # THP compaction 멈춤 방지
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # GPU 번호 = nvidia-smi 번호 (기본 순서는 빠른 GPU부터라 GPU 종류가 섞인 서버에서 어긋남)
cd "$(dirname "$0")/.."

VERSION=${VERSION:?VERSION=x.y.z is required, e.g. VERSION=0.1.0 bash scripts/run_sydney_mtl.sh}
ENCODERS=${ENCODERS:-hibou_l}
MAG=${MAG:-20}
# 서버마다 기본값이 다르다 (명령에 GPU / CPUS를 주면 그 값을 쓴다).
#   HOST_PROFILE=large : 48GB GPU 서버 (공용 서버라 CPU 30% 이내로 묶음)
#   HOST_PROFILE=small : 10GB GPU가 섞인 보조 서버
case "${HOST_PROFILE:-}" in
    large) DEF_GPU=0; DEF_CPUS=32-50 ;;
    small) DEF_GPU=5; DEF_CPUS=0-7 ;;
    *)  [[ -n "${GPU:-}" && -n "${CPUS:-}" ]] || { echo "Set HOST_PROFILE=large|small, or GPU and CPUS explicitly"; exit 1; }
        DEF_GPU=$GPU; DEF_CPUS=$CPUS ;;
esac
GPU=${GPU:-$DEF_GPU}
CPUS=${CPUS:-$DEF_CPUS}
LABEL_COL=${LABEL_COL:-code_review}
LABEL_SOURCE=${LABEL_SOURCE:-CRF v2}
LOSS=${LOSS:-la}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs}
WATCH=${WATCH:-1}
EXTRA_ARGS=${EXTRA_ARGS:-}
PYTHON=${PYTHON:-python}
"$PYTHON" -c "" 2>/dev/null || PYTHON=python3  # python 이 없으면 python3
MLFLOW_PYTHON=${MLFLOW_PYTHON:-$PYTHON}  # mlflow 가 다른 환경에 있으면 MLFLOW_PYTHON 으로 지정
MLFLOW_ARGS=()
"$MLFLOW_PYTHON" -c "import mlflow" 2>/dev/null || MLFLOW_ARGS=(--no_mlflow)
EMB_ROOT=${EMB_ROOT:-/path/to/sydney/embedding}
COORDS=${MAG}x_224px_0px_overlap
read -r -a EXTRA <<< "$EXTRA_ARGS"

if [[ "${1:-}" == "--worker" ]]; then  # 백그라운드에서 실제 학습을 차례대로 돌리는 부분
    for enc in $ENCODERS; do
        echo "[$(date '+%F %T')] start $enc ${MAG}x v$VERSION (host $(hostname), gpu $GPU, cpus $CPUS)"
        OMP_NUM_THREADS=4 taskset -c "$CPUS" "$PYTHON" -m src.training.train_sydney_mtl --version "$VERSION" \
            --output_root "$OUTPUT_ROOT" --meta_csv metadata/sydney_meta.csv \
            --label_col "$LABEL_COL" --label_source "$LABEL_SOURCE" --loss "$LOSS" --gpu "$GPU" \
            --feat_dir "$EMB_ROOT/$enc/$COORDS/features_$enc" "${MLFLOW_ARGS[@]}" "${EXTRA[@]}" \
            || echo "[$(date '+%F %T')] FAILED $enc (see $OUTPUT_ROOT/sydney_mtl_${enc}_${MAG}x_v$VERSION/train.log)"
        echo "[$(date '+%F %T')] end $enc ${MAG}x"
    done
    exit 0
fi

for enc in $ENCODERS; do
    [[ -d "$EMB_ROOT/$enc/$COORDS/features_$enc" ]] || { echo "No ${MAG}x embeddings for '$enc' in $EMB_ROOT"; exit 1; }
    if pgrep -f "train_sydney_mtl --version $VERSION .*$COORDS/features_$enc" > /dev/null; then
        echo "$enc ${MAG}x v$VERSION is already training. Watch it with: bash scripts/watch_sydney.sh"; exit 1
    fi
done

# 백그라운드로 띄우기 전에 학습 패키지 확인 (없으면 띄우자마자 실패하는데 화면에는 보이지 않는다)
if ! err=$("$PYTHON" -c "import torch, h5py, pandas, sklearn, matplotlib" 2>&1); then
    echo "학습 패키지가 없습니다 ($PYTHON): $(echo "$err" | tail -1)"
    echo "먼저 설치: $PYTHON -m pip install scikit-learn scipy matplotlib pillow"
    exit 1
fi

mkdir -p "$OUTPUT_ROOT/launch_logs"
LOG=$OUTPUT_ROOT/launch_logs/run_v${VERSION}_${MAG}x_gpu${GPU}_$(date +%Y%m%d_%H%M%S).log  # GPU별로 따로 (같은 초에 둘 띄워도 한 파일에 섞이지 않게)
export VERSION ENCODERS MAG GPU CPUS LABEL_COL LABEL_SOURCE LOSS OUTPUT_ROOT EXTRA_ARGS PYTHON MLFLOW_PYTHON EMB_ROOT
setsid nohup bash "$0" --worker > "$LOG" 2>&1 < /dev/null &
PGID=$!
echo "Started on $(hostname): ENCODERS=\"$ENCODERS\" MAG=${MAG}x VERSION=$VERSION GPU=$GPU CPUS=$CPUS (labels $LABEL_COL, loss $LOSS)"
echo "  python     : $PYTHON"
echo "  launch log : $LOG"
echo "  results    : $OUTPUT_ROOT/sydney_mtl_<encoder>_${MAG}x_v$VERSION/  (train.log, plots, MLflow upload at the end)"
echo "  stop       : kill -- -$PGID"
if [[ ${#MLFLOW_ARGS[@]} -gt 0 ]]; then
    echo "  MLflow     : 이 서버에는 mlflow 환경이 없어 업로드를 껐다. 끝난 뒤 mlflow 가 있는 서버에서 모델마다:"
    echo "               cd $(pwd) && $MLFLOW_PYTHON -m src.training.mlflow_sydney --out_dir $OUTPUT_ROOT/sydney_mtl_<encoder>_${MAG}x_v$VERSION"
fi
WATCH_ARGS=()
[[ "$OUTPUT_ROOT" == outputs ]] || WATCH_ARGS=(--output_root "$OUTPUT_ROOT")
echo "  watch again: bash scripts/watch_sydney.sh ${WATCH_ARGS[*]}"
if [[ "$WATCH" == "1" ]]; then
    sleep 3
    exec bash scripts/watch_sydney.sh "${WATCH_ARGS[@]}" --tail 8
fi
