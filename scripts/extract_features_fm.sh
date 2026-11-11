#!/usr/bin/env bash
# 모델 1개(또는 여러 개를 차례대로) × 배율 1개의 패치 특징만 추출 (TRIDENT --task feat). 조직 분할·패치 좌표는 다시 만들지 않는다.
# 추출을 백그라운드로 띄우고 바로 실시간 모니터(scripts/watch_embedding.sh)를 연다.
#   - 모니터에서 Ctrl+C를 눌러도 추출은 계속된다 (다시 보기: bash scripts/watch_embedding.sh)
#   - 같은 명령을 다시 돌리면 끝난 슬라이드는 건너뛴다 (TRIDENT resume). 같은 모델·배율이 이미 돌고 있으면 띄우지 않는다.
# 모델별 폴더 = $EMB_ROOT/<encoder>/ (TRIDENT job_dir), 배율마다 하위 폴더가 따로 생긴다:
#   $EMB_ROOT/<encoder>/20x_224px_0px_overlap/features_<encoder>/<slide>.h5   (features (N, dim), coords (N, 2))
#   $EMB_ROOT/<encoder>/40x_224px_0px_overlap/features_<encoder>/<slide>.h5
# 좌표가 없으면 COORDS_FROM 폴더에서 복사해 와서 모든 모델이 같은 패치 위치를 쓰게 한다.
# 그 배율의 좌표가 COORDS_FROM에도 없으면 먼저: MAG=<배율> bash scripts/make_coords.sh
#
# 사용 (gated 모델 토큰이 있는 계정에서):
#   ENC=hibou_l MAG=40 bash scripts/extract_features_fm.sh
#   ENC="hoptimus0 virchow" MAG=40 bash scripts/extract_features_fm.sh   # 여러 개면 앞 모델이 끝나는 대로 다음 모델 (로그는 하나)
# GPU 메모리가 작은 서버 (예: RTX 3080 10GB 0·1·5·6번 + V100 32GB 2·3·4번): 같은 명령에 HOST_PROFILE=small
#   → GPUS "0 1 5 6", CPUS 0-11, 배치 64
#   3080 10GB에서 ViT-g (hoptimus0, midnight12k)는 배치 256이면 슬라이드마다 CUDA OOM, V100은 hoptimus0 FlashAttention 미지원
# 환경 변수 (기본값 HOST_PROFILE=large / small, 프로필을 안 주면 GPUS 필수):
#   ENC (필수, 공백으로 여러 개)  MAG (20)  WATCH (1, 0이면 모니터 없이 띄우기만)
#   GPUS (nvidia-smi 번호, "0 1 2 3" / "0 1 5 6")  CPUS (taskset 코어 범위, 32-50 / 0-11)
#   EXTRA_ARGS ("--max_workers 4 --feat_batch_size 1024" / "... 64")
# 실행 기록: 실행 한 번에 하나, $EMB_ROOT/logs/<MAG>x/extract_<encoder>[+<encoder>...]_<MAG>x_<날짜>.log, 중지: 시작할 때 출력되는 kill 명령
set -euo pipefail
umask 000  # NFS: 계정이 달라도 결과를 지우고 덮어쓸 수 있도록
export NUMPY_MADVISE_HUGEPAGE=0  # THP compaction 멈춤 방지 (scripts/extract_embeddings.sh 참고)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # GPUS 번호 = nvidia-smi 번호 (기본 순서는 빠른 GPU부터라 종류가 섞인 서버에서 어긋남)

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
ENC=${ENC:?ENC=<hibou_l|hoptimus0|virchow|midnight12k> is required}
MAG=${MAG:-20}
WSI_DIR=${WSI_DIR:-/path/to/sydney/wsi_tiff}
EMB_ROOT=${EMB_ROOT:-/path/to/sydney/embedding}
COORDS_FROM=${COORDS_FROM:-$EMB_ROOT/midnight12k}
# 서버마다 기본값이 다르다 (명령에 GPUS / CPUS / EXTRA_ARGS를 주면 그 값을 쓴다).
# 10GB GPU에서 48GB용 배치를 쓰면 OOM, V100은 H-Optimus-0 FlashAttention 미지원이라 프로필별 기본값을 둔다.
case "${HOST_PROFILE:-}" in
    large) DEF_GPUS="0 1 2 3"; DEF_CPUS=32-50; DEF_BATCH=1024 ;;  # 48GB GPU (예: RTX 6000 Ada)
    small) DEF_GPUS="0 1 5 6"; DEF_CPUS=0-11; DEF_BATCH=64 ;;     # 10GB GPU만 (예: RTX 3080). V100은 H-Optimus-0
                                                                  # FlashAttention 미지원(Ampere 이상만). ViT-g는 배치 256이면 OOM
    *)  [[ -n "${GPUS:-}" ]] || { echo "Set HOST_PROFILE=large|small, or GPUS (and CPUS, EXTRA_ARGS) explicitly"; exit 1; }
        DEF_GPUS=$GPUS; DEF_CPUS=0-$(( $(nproc) - 1 )); DEF_BATCH=256 ;;   # 프로필 없음: 전체 코어, 보수적 배치
esac
GPUS=${GPUS:-$DEF_GPUS}
CPUS=${CPUS:-$DEF_CPUS}
WATCH=${WATCH:-1}
PYTHON=${PYTHON:-python}
"$PYTHON" -c "" 2>/dev/null || PYTHON=python3  # python 이 없으면 python3
EXTRA_ARGS=${EXTRA_ARGS:-"--max_workers 4 --feat_batch_size $DEF_BATCH"}

read -r -a ENCS <<< "$ENC"
COORDS=${MAG}x_224px_0px_overlap
read -r -a GPU_ARGS <<< "$GPUS"
read -r -a EXTRA <<< "$EXTRA_ARGS"

if [[ "${1:-}" == "--worker" ]]; then  # 백그라운드에서 실제 추출을 돌리는 부분 (모델 여러 개면 차례대로)
    cd "$PROJECT_DIR/TRIDENT"
    for enc in "${ENCS[@]}"; do
        job_dir=$EMB_ROOT/$enc
        echo "[$(date '+%F %T')] host=$(hostname) ENC=$enc MAG=$MAG WSI_DIR=$WSI_DIR JOB_DIR=$job_dir GPUS=$GPUS CPUS=$CPUS EXTRA_ARGS=$EXTRA_ARGS"
        echo "[$(date '+%F %T')] coords: $(ls "$job_dir/$COORDS/patches" | grep -c '_patches\.h5$') slides"
        taskset -c "$CPUS" "$PYTHON" run_batch_of_slides.py --task feat --patch_encoder "$enc" \
            --wsi_dir "$WSI_DIR" --job_dir "$job_dir" --mag "$MAG" --patch_size 224 --overlap 0 --skip_errors \
            --clear_dead_locks --gpus "${GPU_ARGS[@]}" "${EXTRA[@]}" \
            || { rc=$?; echo "[$(date '+%F %T')] $enc exited with code $rc (다음 모델로 넘어감)"; }
        echo "[$(date '+%F %T')] done: $(ls "$job_dir/$COORDS/features_$enc" 2>/dev/null | grep -c '\.h5$') feature files in $job_dir/$COORDS/features_$enc"
    done
    exit 0
fi

for enc in "${ENCS[@]}"; do  # 띄우기 전에 모델마다 확인: 이미 도는지, 좌표가 있는지
    job_dir=$EMB_ROOT/$enc
    if pgrep -f "run_batch_of_slides.py --task feat --patch_encoder $enc .*--job_dir $job_dir --mag $MAG " > /dev/null; then
        echo "$enc ${MAG}x is already running. Watch it with: bash scripts/watch_embedding.sh"; exit 1
    fi
    if [[ ! -d "$job_dir/$COORDS/patches" ]]; then
        [[ -d "$COORDS_FROM/$COORDS/patches" ]] || { echo "No ${MAG}x coords in $COORDS_FROM. Run first: MAG=$MAG bash scripts/make_coords.sh"; exit 1; }
        echo "[$(date '+%F %T')] $enc: copy segmentation + ${MAG}x coords from $COORDS_FROM"
        mkdir -p "$job_dir/$COORDS"
        [[ -d "$job_dir/contours_geojson" ]] || \
            cp -a "$COORDS_FROM"/{contours,contours_geojson,thumbnails,_config_segmentation.json,_logs_segmentation.txt} "$job_dir"/
        cp -a "$COORDS_FROM/$COORDS"/{patches,visualization,_config_coords.json,_logs_coords.txt} "$job_dir/$COORDS"/
    fi
done

mkdir -p "$EMB_ROOT/logs/${MAG}x"
LOG=$EMB_ROOT/logs/${MAG}x/extract_$(IFS=+; echo "${ENCS[*]}")_${MAG}x_$(date +%Y%m%d_%H%M).log
export ENC MAG WSI_DIR EMB_ROOT COORDS_FROM GPUS CPUS PYTHON EXTRA_ARGS
setsid nohup bash "$0" --worker > "$LOG" 2>&1 < /dev/null &
PGID=$!
echo "Started on $(hostname): ENC=\"$ENC\" MAG=${MAG}x GPUS=\"$GPUS\" CPUS=$CPUS"
for enc in "${ENCS[@]}"; do
    echo "  output     : $EMB_ROOT/$enc/$COORDS/features_$enc/"
done
echo "  log        : $LOG"
echo "  stop       : kill -- -$PGID"
echo "  watch again: bash scripts/watch_embedding.sh"
if [[ "$WATCH" == "1" && -t 1 ]]; then
    sleep 5
    exec bash "$PROJECT_DIR/scripts/watch_embedding.sh" --tail 6
fi
