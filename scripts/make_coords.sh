#!/usr/bin/env bash
# 배율 하나의 패치 좌표를 만든다 (TRIDENT --task coords, GPU 안 씀). 조직 분할(contours)은 기존 결과를 그대로 쓴다.
# 좌표는 $COORDS_FROM(midnight12k)/<MAG>x_224px_0px_overlap/에 만들고 다른 FM 폴더로 복사해서 4개 FM이 같은 패치 위치를 쓰게 한다.
# 배율마다 폴더가 따로라서 다른 배율 결과는 건드리지 않는다:
#   $EMB_ROOT/<fm>/20x_224px_0px_overlap/   패치 1장 = 40x 원본 448px 영역 (TIFF level 1에서 224px)
#   $EMB_ROOT/<fm>/40x_224px_0px_overlap/   패치 1장 = 40x 원본 224px 영역 (level 0 그대로)
#   $EMB_ROOT/<fm>/10x_224px_0px_overlap/   패치 1장 = 40x 원본 896px 영역 (level 2)
# 같은 명령을 다시 돌리면 끝난 슬라이드는 건너뛰고, 이미 복사된 파일은 덮어쓰지 않는다.
# 그다음 특징 추출: ENC=<fm> MAG=<MAG> bash scripts/extract_features_fm.sh
#
# 사용 (CPU 32-50번 코어, 작업자 4개):
#   MAG=40 bash scripts/make_coords.sh
set -euo pipefail
umask 000  # NFS: 계정이 달라도 결과를 지우고 덮어쓸 수 있도록
export NUMPY_MADVISE_HUGEPAGE=0  # THP compaction 멈춤 방지 (scripts/extract_embeddings.sh 참고)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
MAG=${MAG:?MAG=<40|20|10> is required}
WSI_DIR=${WSI_DIR:-/path/to/sydney/wsi_tiff}
EMB_ROOT=${EMB_ROOT:-/path/to/sydney/embedding}
COORDS_FROM=${COORDS_FROM:-$EMB_ROOT/midnight12k}
ENCODERS=${ENCODERS:-"hibou_l hoptimus0 virchow midnight12k"}
WORKERS=${WORKERS:-4}
PYTHON=${PYTHON:-python}

COORDS=${MAG}x_224px_0px_overlap
CPU_WORKERS=()
for _ in $(seq "$WORKERS"); do CPU_WORKERS+=(-1); done  # --gpus -1 -1 ...: GPU 없이 CPU 작업자 여러 개

[[ -d "$COORDS_FROM/contours_geojson" ]] || { echo "No segmentation in $COORDS_FROM (run scripts/extract_embeddings.sh first)"; exit 1; }
echo "[$(date '+%F %T')] coords ${MAG}x -> $COORDS_FROM/$COORDS ($WORKERS CPU workers)"
cd "$PROJECT_DIR/TRIDENT"
taskset -c 32-50 "$PYTHON" run_batch_of_slides.py --task coords --wsi_dir "$WSI_DIR" --job_dir "$COORDS_FROM" \
    --mag "$MAG" --patch_size 224 --overlap 0 --skip_errors --gpus "${CPU_WORKERS[@]}"
echo "[$(date '+%F %T')] coords: $(ls "$COORDS_FROM/$COORDS/patches" | grep -c '_patches\.h5$') slides"

for enc in $ENCODERS; do
    JOB_DIR=$EMB_ROOT/$enc
    [[ "$(realpath -m "$JOB_DIR")" == "$(realpath -m "$COORDS_FROM")" ]] && continue
    if [[ ! -d "$JOB_DIR/contours_geojson" ]]; then
        mkdir -p "$JOB_DIR"
        cp -a "$COORDS_FROM"/{contours,contours_geojson,thumbnails,_config_segmentation.json,_logs_segmentation.txt} "$JOB_DIR"/
    fi
    mkdir -p "$JOB_DIR/$COORDS"
    cp -an "$COORDS_FROM/$COORDS"/{patches,visualization} "$JOB_DIR/$COORDS"/
    cp -a "$COORDS_FROM/$COORDS"/{_config_coords.json,_logs_coords.txt} "$JOB_DIR/$COORDS"/
    echo "[$(date '+%F %T')] copied to $JOB_DIR/$COORDS ($(ls "$JOB_DIR/$COORDS/patches" | grep -c '_patches\.h5$') slides)"
done
