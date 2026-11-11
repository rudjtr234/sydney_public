#!/usr/bin/env bash
# TRIDENT 임베딩 추출 진행 상황 모니터 (src/data/watch_embedding.py). 추출에는 영향을 주지 않는다.
# 사용: bash scripts/watch_embedding.sh [--interval 30] [--once] [--tail 8]
cd "$(dirname "$0")/.." || exit 1
export NUMPY_MADVISE_HUGEPAGE=0
PYTHON=${PYTHON:-python}
"$PYTHON" -c "" 2>/dev/null || PYTHON=python3  # python 이 없으면 python3
exec "$PYTHON" -m src.data.watch_embedding "$@"
