#!/usr/bin/env bash
# 새로 들어오거나 다시 스캔된 iSyntax 슬라이드만 골라 TIFF 변환 → 조직 분할·패치 좌표 → 모델별 특징까지 추가한다.
# 다른 슬라이드의 결과는 건드리지 않는다. 조직 분할·좌표는 $COORDS_ENC 폴더에서 1번 만들고 나머지 모델 폴더로 복사한다
# (extract_features_fm.sh와 같은 방식이라 모든 모델이 같은 패치를 쓴다).
# 배율: MAGS를 안 주면 $COORDS_ENC 폴더에 좌표 폴더(<배율>x_224px_0px_overlap)가 있는 배율 전부 (예: 20 40).
# 교체되는 기존 TIFF는 WSI 폴더의 _replaced_<날짜>/로 옮겨 둔다 (TRIDENT는 하위 폴더를 읽지 않음).
# 다시 실행해도 된다: TIFF가 원본보다 새것이면 변환을 건너뛰고, TIFF보다 나중에 만든 결과는 지우지 않는다.
#
# 사용:
#   SLIDES="CASE-0001" setsid nohup taskset -c 32-50 bash /path/to/sydney_project/scripts/add_slides.sh \
#     > /path/to/sydney/embedding/logs/add_slides_$(date +%Y%m%d_%H%M).log 2>&1 < /dev/null &
set -euo pipefail
umask 000  # NFS: 계정이 달라도 결과를 지우고 덮어쓸 수 있도록
export NUMPY_MADVISE_HUGEPAGE=0  # THP compaction 멈춤 방지
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
SLIDES=${SLIDES:?SLIDES="<slide> ..." is required}
SRC_DIR=${SRC_DIR:-"/path/to/sydney/wsi_isyntax"}
WSI_DIR=${WSI_DIR:-/path/to/sydney/wsi_tiff}
EMB_ROOT=${EMB_ROOT:-/path/to/sydney/embedding}
COORDS_ENC=${COORDS_ENC:-midnight12k}
ENCODERS=${ENCODERS:-"midnight12k hibou_l hoptimus0 virchow"}
GPU=${GPU:-0}
PYTHON=${PYTHON:-python}
J=$EMB_ROOT/$COORDS_ENC
MAGS=${MAGS:-$(cd "$J" && ls -d *x_224px_0px_overlap | sed 's/x_224px_0px_overlap$//' | tr '\n' ' ')}
read -r -a MAG_LIST <<< "$MAGS"
BACKUP=$WSI_DIR/_replaced_$(date +%Y%m%d)
COMMON=(--wsi_dir "$WSI_DIR" --patch_size 224 --overlap 0 --skip_errors --gpus "$GPU")
log() { echo "[$(date '+%F %T')] $*"; }

LIST=$(mktemp --suffix=.csv)
trap 'rm -f "$LIST"' EXIT
echo wsi > "$LIST"
log "SLIDES=$SLIDES ENCODERS=$ENCODERS MAGS=$MAGS GPU=$GPU"

# 1. TIFF 변환 (원본이 TIFF보다 새것일 때만)
for s in $SLIDES; do
    src=$(ls "$SRC_DIR/$s".isyntax "$SRC_DIR/$s".i2syntax 2>/dev/null | head -1 || true)
    [[ -n "$src" ]] || { log "source not found: $s"; exit 1; }
    tiff=$WSI_DIR/$s.tiff
    if [[ ! -e "$tiff" || "$src" -nt "$tiff" ]]; then
        if [[ -e "$tiff" ]]; then
            mkdir -p "$BACKUP"
            if [[ -e "$BACKUP/$s.tiff" ]]; then
                log "backup already exists, keeping it: $BACKUP/$s.tiff"
            else
                mv "$tiff" "$BACKUP"/
                log "old TIFF -> $BACKUP/$s.tiff"
            fi
        fi
        log "convert $src"
        "$PYTHON" "$PROJECT_DIR/scripts/convert_isyntax_to_tiff.py" "$src" "$tiff"
    else
        log "TIFF up to date: $tiff"
    fi
    echo "$s.tiff" >> "$LIST"
done

# 2. 새 TIFF보다 오래된 이 슬라이드의 결과 정리 (모델 폴더 × 배율마다)
for enc in $ENCODERS; do
    d=$EMB_ROOT/$enc
    for s in $SLIDES; do
        files=("$d/thumbnails/$s.jpg" "$d/contours/$s.jpg" "$d/contours_geojson/$s.geojson")
        for m in "${MAG_LIST[@]}"; do
            C=${m}x_224px_0px_overlap
            files+=("$d/$C/patches/${s}_patches.h5" "$d/$C/visualization/$s.jpg" "$d/$C/features_$enc/$s.h5")
        done
        for f in "${files[@]}"; do
            if [[ -e "$f" && ! "$f" -nt "$WSI_DIR/$s.tiff" ]]; then
                rm -f "$f"
                log "removed stale $f"
            fi
        done
    done
done

# 3. 조직 분할 (배율과 무관, 1번) · 배율별 패치 좌표 ($COORDS_ENC 폴더에서만)
cd "$PROJECT_DIR/TRIDENT"
log "segmentation + coords (${MAGS}) in $J"
"$PYTHON" run_batch_of_slides.py --task seg --job_dir "$J" --custom_list_of_wsis "$LIST" --mag "${MAG_LIST[0]}" \
    "${COMMON[@]}" --clear_dead_locks
for m in "${MAG_LIST[@]}"; do
    "$PYTHON" run_batch_of_slides.py --task coords --job_dir "$J" --custom_list_of_wsis "$LIST" --mag "$m" \
        "${COMMON[@]}" --clear_dead_locks
done

# 4. 나머지 모델 폴더로 복사 (모든 모델이 같은 좌표를 쓰도록)
for enc in $ENCODERS; do
    [[ "$enc" == "$COORDS_ENC" ]] && continue
    d=$EMB_ROOT/$enc
    for s in $SLIDES; do
        cp -a "$J/thumbnails/$s.jpg" "$d/thumbnails/"
        cp -a "$J/contours/$s.jpg" "$d/contours/"
        cp -a "$J/contours_geojson/$s.geojson" "$d/contours_geojson/"
        for m in "${MAG_LIST[@]}"; do
            C=${m}x_224px_0px_overlap
            if [[ -e "$J/$C/patches/${s}_patches.h5" ]]; then
                mkdir -p "$d/$C/patches" "$d/$C/visualization"
                cp -a "$J/$C/patches/${s}_patches.h5" "$d/$C/patches/"
                cp -a "$J/$C/visualization/$s.jpg" "$d/$C/visualization/"
            else
                log "no tissue patches: $s ${m}x"
            fi
        done
    done
    log "copied segmentation + coords -> $d"
done

# 5. 모델 × 배율별 특징 추출
for enc in $ENCODERS; do
    for m in "${MAG_LIST[@]}"; do
        log "features: $enc ${m}x"
        "$PYTHON" run_batch_of_slides.py --task feat --patch_encoder "$enc" --job_dir "$EMB_ROOT/$enc" \
            --custom_list_of_wsis "$LIST" --mag "$m" "${COMMON[@]}" --clear_dead_locks
    done
done

for enc in $ENCODERS; do
    for m in "${MAG_LIST[@]}"; do
        for s in $SLIDES; do
            f=$EMB_ROOT/$enc/${m}x_224px_0px_overlap/features_$enc/$s.h5
            if [[ -e "$f" ]]; then log "ok: $f"; else log "MISSING: $f"; fi
        done
    done
done
log "done"
