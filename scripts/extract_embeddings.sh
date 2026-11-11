#!/usr/bin/env bash
# Sydney WSI -> TRIDENT 패치 임베딩 (Hibou-L / H-Optimus-0 / Virchow / Midnight-12k)
#
# 4개 인코더 모두 20x / 224px 이므로 조직 분할·패치 좌표는 한 번만 만들고 인코더별로 특징만 추출한다.
# 결과: $JOB_DIR/20x_224px_0px_overlap/features_<encoder>/<slide>.h5   (features (N, dim), coords (N, 2))
# 같은 명령을 다시 돌리면 끝난 슬라이드는 건너뛴다 (TRIDENT resume).
#
# 사용 예:
#   nohup bash scripts/extract_embeddings.sh > logs/extract_$(date +%Y%m%d_%H%M).log 2>&1 &
#   GPUS="0 1" ENCODERS="hoptimus0" bash scripts/extract_embeddings.sh
#   EXTRA_ARGS="--search_nested --wsi_ext .svs" bash scripts/extract_embeddings.sh
set -euo pipefail

# NFS에서는 서버마다 uid 매핑이 달라 계정이 서로 다른 사용자로 보일 수 있다.
# 어느 쪽이 만든 결과든 다른 쪽이 지우거나 덮어쓸 수 있도록 권한을 연다.
umask 000

# 일부 서버는 THP=madvise이고 메모리 compaction 부하가 커서(kcompactd 100%), numpy가 큰 배열에 huge page를 요청하면
# 할당마다 수 분씩 멈춘다. TRIDENT 조직 분할이 슬라이드당 25초 -> 7분이 됐다. numpy의 요청을 끈다.
export NUMPY_MADVISE_HUGEPAGE=0

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
WSI_DIR=${WSI_DIR:-/path/to/sydney/wsi_tiff}
JOB_DIR=${JOB_DIR:-/path/to/sydney/embedding}
GPUS=${GPUS:-"0 1 2 3"}
ENCODERS=${ENCODERS:-"hibou_l hoptimus0 virchow midnight12k"}
PYTHON=${PYTHON:-python}
EXTRA_ARGS=${EXTRA_ARGS:-}   # 예: "--search_nested", "--wsi_ext .svs .ndpi", "--wsi_cache /local/ssd"

# reference: TRIDENT/.claude/skills/trident/reference.md -- 4개 인코더 모두 --patch_size 224 --mag 20
COMMON=(--wsi_dir "$WSI_DIR" --job_dir "$JOB_DIR" --mag 20 --patch_size 224 --overlap 0 --skip_errors)
read -r -a GPU_ARGS <<< "$GPUS"
read -r -a EXTRA <<< "$EXTRA_ARGS"

echo "[$(date '+%F %T')] WSI_DIR=$WSI_DIR JOB_DIR=$JOB_DIR GPUS=$GPUS ENCODERS=$ENCODERS"
cd "$PROJECT_DIR/TRIDENT"

echo "[$(date '+%F %T')] 1/3 tissue segmentation"
"$PYTHON" run_batch_of_slides.py --task seg "${COMMON[@]}" --gpus "${GPU_ARGS[@]}" "${EXTRA[@]}"

echo "[$(date '+%F %T')] 2/3 patch coordinates (20x, 224px)"
"$PYTHON" run_batch_of_slides.py --task coords "${COMMON[@]}" --gpus "${GPU_ARGS[@]}" "${EXTRA[@]}"

for enc in $ENCODERS; do
    echo "[$(date '+%F %T')] 3/3 features: $enc"
    "$PYTHON" run_batch_of_slides.py --task feat --patch_encoder "$enc" "${COMMON[@]}" --gpus "${GPU_ARGS[@]}" "${EXTRA[@]}"
done

echo "[$(date '+%F %T')] done -> check with: $PYTHON $PROJECT_DIR/scripts/check_embeddings.py --job_dir $JOB_DIR --wsi_dir $WSI_DIR"
