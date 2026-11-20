#!/usr/bin/env bash
# Part 2 experiments: sizes 1..7 for several worker counts and seeds -> ../../plots/plot-part2.csv
# Usage: ./run_experiments.sh [MAX_WORKERS] [TIMEOUT_SECONDS] [SEEDS]
set -euo pipefail
cd "$(dirname "$0")"
MAXW="${1:-$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) )}"
TO="${2:-120}"; SEEDS="${3:-5}"
CSV="../../plots/plot-part2.csv"; mkdir -p ../../plots; rm -f "$CSV"
WORKERS="1"; w=2; while [ "$w" -le "$MAXW" ]; do WORKERS="$WORKERS $w"; w=$((w*2)); done
for W in $WORKERS; do
  for S in $(seq 1 "$SEEDS"); do
    python3 backtrack_magic.py --min-n 1 --max-n 7 --workers "$W" --order random --seed "$S" --timeout "$TO" --csv "$CSV"
  done
done
python3 fit_part2.py --csv "$CSV"
