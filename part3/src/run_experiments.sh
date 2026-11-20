#!/usr/bin/env bash
# Usage: ./run_experiments.sh [WORKERS] [TIMEOUT_SECONDS]
set -euo pipefail
cd "$(dirname "$0")"
W="${1:-$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) )}"; TO="${2:-120}"
CSV="../../plots/plot-part3.csv"; mkdir -p ../../plots; rm -f "$CSV"
python3 cp_sat_magic.py --min-n 1 --max-n 20 --workers "$W" --timeout "$TO" --repeat 3 --csv "$CSV"
