#!/usr/bin/env bash
# Sydney SydneyMTL 학습 진행 상황 모니터 (src/training/watch_sydney_mtl.py). 학습에는 영향을 주지 않는다.
# 사용: bash scripts/watch_sydney.sh [--interval 10] [--once] [--tail 10] [--all]
cd "$(dirname "$0")/.." || exit 1
PYTHON=${PYTHON:-python}
"$PYTHON" -c "" 2>/dev/null || PYTHON=python3  # python 이 없으면 python3
exec "$PYTHON" -m src.training.watch_sydney_mtl "$@"
