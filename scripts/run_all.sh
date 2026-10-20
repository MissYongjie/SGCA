#!/usr/bin/env bash
# Reproduce the main SGCA results (paper Table 3) on HetData #1-#6.
# Usage: bash scripts/run_all.sh [extra args for tools/train.py, e.g. --device cpu]
set -e
cd "$(dirname "$0")/.."
for i in 1 2 3 4 5 6; do
  python -u tools/train.py --config configs/hetdata${i}.yaml "$@"
done
python tools/summarize.py work_dirs
