#!/usr/bin/env bash
# Part 1 experiment driver. Usage: ./run_experiments.sh [THREADS] [N4_TIMEOUT_SECONDS]
# Builds magic_bf, runs all measurements, appends rows to ../../plots/plot-part1.csv, then fits the model.
set -euo pipefail
cd "$(dirname "$0")"
THREADS="${1:-$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1) )}"
N4_TIMEOUT="${2:-1800}"          # n=4 first solution needs ~8e10 permutations
CSV="../../plots/plot-part1.csv"
mkdir -p ../../plots; rm -f "$CSV"
make -s
R="./magic_bf --threads $THREADS --csv $CSV"

echo "== find one solution, n=1..3 (30 repeats each, small runs are noisy) =="
for n in 1 2 3; do $R --n $n --mode first --repeat 30 --quiet; done
$R --n 3 --mode first                         # show the solution
echo "== exhaustive n=3 (counts all 8 solutions) =="
$R --n 3 --mode full --repeat 10 --quiet; $R --n 3 --mode full
echo "== n=4: first solution by brute force (timeout ${N4_TIMEOUT}s) =="
$R --n 4 --mode first --timeout "$N4_TIMEOUT"
echo "== n=5: first solution by brute force (timeout 60s, expected to fail) =="
$R --n 5 --mode first --timeout 60
echo "== throughput / exact partial enumeration for the model fit =="
for K in 1 2 4 8 16 32; do $R --n 4 --mode full --prefix-depth 5 --max-tasks $K --quiet; done
for K in 1 2 4 8 16 32; do $R --n 5 --mode full --prefix-depth 14 --max-tasks $K --quiet; done
$R --n 4 --mode timed --seconds 15 --quiet
$R --n 5 --mode timed --seconds 15 --quiet
echo "== fit =="
python3 fit_model.py --csv "$CSV" --out-dir ../../plots
