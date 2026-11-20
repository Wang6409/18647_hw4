# HW4: Parallel Magic Square Solver

## Part 0: Predictions

I expect the relative runtime for nontrivial sizes to be roughly **Part 1 ≫ Part 2 > Part 3**, with Part 3 often fastest for moderate `n`. Brute force enumerates `(n²)!` permutations and will become impractical almost immediately. Backtracking with early pruning should be much faster than brute force but still exponential; parallel search helps but does not change the asymptotic wall. A CP/SMT/integer-optimization solver should benefit most from global constraints and propagation, though solver overhead may dominate for tiny cases.

A natural formulation uses variables `x[i][j] ∈ {1,...,n²}` for each cell, an **AllDifferent** constraint over all `n²` variables, and linear sum constraints for every row, column, and main diagonal equal to the magic constant `M = n(n²+1)/2`. For CP-SAT this is encoded directly with integer variables, distinctness, and equality constraints.

The raw search space is `(n²)!` for brute force. Backtracking and CP/SMT shrink it by rejecting partial assignments as soon as a row, column, or diagonal cannot reach `M`, and by enforcing AllDifferent early. Additional reductions include symmetry breaking, partial-sum bounds, and enforcing the magic constant. I expect `n=3,4` to be easy and `n=5` or maybe `n=6` feasible for smarter methods, while brute force will already be clearly impractical for `n ≥ 4` or `5`.

## Code Files

| File | One-sentence description |
|---|---|
| `part1/src/magic_bf.cpp` | Enumerates permutations of `1…n²` in lexicographic order, fills the square, checks rows/columns/diagonals, and supports `first`, `full`, and `timed` modes with OpenMP. |
| `part1/src/Makefile` | Builds `magic_bf` with `-O3 -fopenmp`. |
| `part1/src/run_experiments.sh` | Runs all Part 1 measurements and invokes the fitting script. |
| `part1/src/fit_model.py` | Fits `T = t0 + P/R`, extrapolates brute-force runtime to larger `n`, and produces a report and plots. |
| `part2/src/backtrack_magic.py` | Recursive backtracking solver with partial-sum pruning, direct completion of last row/column/diagonal cells, and process-pool parallelism. |
| `part2/src/run_experiments.sh` | Runs Part 2 experiments over multiple worker counts and random seeds, then invokes fitting. |
| `part2/src/fit_part2.py` | Fits an exponential model for Part 2, reports speedup/memory, and predicts the largest feasible `n`. |
| `part3/src/cp_sat_magic.py` | Models the problem with OR-Tools CP-SAT using integer variables, AllDifferent, sum constraints, and corner symmetry breaking. |
| `part3/src/requirements.txt` | Lists the Python dependency `ortools` for Part 3. |
| `part3/src/run_experiments.sh` | Runs CP-SAT experiments for `n=1…20`, repeats each size three times, and writes CSV results. |
| `plots/compare_plots.py` | Reads the three CSV files and plots first-solution time versus `n` for all three methods on one log-scale figure. |

## Running the Experiments (8-core machine)

### Part 1: Brute Force (C++/OpenMP)

Dependencies: `g++` with OpenMP, Python 3.9+, `numpy`, `matplotlib`.

```bash
cd part1/src
bash ./run_experiments.sh 8 1800
```

Single run example:
```bash
make && ./magic_bf --n 4 --threads 8 --timeout 1800 --csv /tmp/p1.csv
```

### Part 2: Backtracking Search (Python)

```bash
cd part2/src
bash ./run_experiments.sh 8 120 5
```

Single run example:
```bash
python3 backtrack_magic.py --n 5 --workers 4 --order random --seed 1
```

### Part 3: CP-SAT (OR-Tools)

Dependencies: `ortools`.

```bash
cd part3/src
pip install -r requirements.txt
python cp_sat_magic.py --min-n 1 --max-n 20 --workers 16 --timeout 120 --repeat 3 --csv ..\..\plots\plot-part3.csv
```

Single run example:
```bash
python cp_sat_magic.py --n 6
```

### Part 4: Plots and Analysis
```bash
cd /plots
python make_plots.py
