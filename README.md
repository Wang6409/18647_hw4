# 18-647 Homework 4: Parallel Magic Square Solver

## Environment

Use Python 3.11 or newer.

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -e ".[plot]"
```

OR-Tools is installed by the project dependencies. Matplotlib is optional for
later plot creation.

## Part 1: brute force

```powershell
.\\.venv\\Scripts\\python.exe part1/src/solver.py --n 3 --timeout 60
```

Benchmark a range:

```powershell
.\\.venv\\Scripts\\python.exe part1/src/solver.py --min-n 1 --max-n 4 --timeout 60
```

JSON is written to `part1/results.json` and benchmark rows are appended to
`plots/plot-part1.csv`.

## Part 2: parallel backtracking

```powershell
.\\.venv\\Scripts\\python.exe part2/src/solver.py --n 3 --timeout 60 --workers 4
```

The solver uses multiprocessing to parallelize independent fixed-first-row
branches. JSON is written to `part2/results.json` and benchmark rows are
appended to `plots/plot-part2.csv`.

## Part 3: OR-Tools CP-SAT

```powershell
.\\.venv\\Scripts\\python.exe part3/src/solver.py --n 4 --timeout 60 --workers 4
```

JSON is written to `part3/results.json` and benchmark rows are appended to
`plots/plot-part3.csv`.

## Benchmarks

```powershell
.\\.venv\\Scripts\\python.exe part1/src/solver.py --min-n 1 --max-n 3 --timeout 60
.\\.venv\\Scripts\\python.exe part2/src/solver.py --min-n 1 --max-n 4 --timeout 60 --workers 4
.\\.venv\\Scripts\\python.exe part3/src/solver.py --min-n 1 --max-n 5 --timeout 60 --workers 4
```
