#!/usr/bin/env python3
"""Part 3: magic square as a Constraint Program, solved with Google OR-Tools CP-SAT.

Model (n x n, numbers 1..n^2, T = n(n^2+1)/2):
  variables   x[r][c] in [1, n^2]
  constraints AllDifferent(all cells)
              sum of every row = T, every column = T, main diagonal = T, anti-diagonal = T
  symmetry breaking (n >= 3, valid under the 8 symmetries of the square):
              x[0][0] is the smallest corner, and x[0][n-1] < x[n-1][0]
The solver finds ONE solution (pure feasibility problem, no objective).

Usage
  python cp_sat_magic.py --n 6
  python cp_sat_magic.py --min-n 1 --max-n 12 --workers 8 --timeout 120 --csv ../../plots/plot-part3.csv
Dependencies: pip install -r requirements.txt   (ortools)
"""
import argparse
import csv
import os
import time

from ortools.sat.python import cp_model


def magic_constant(n):
    return n * (n * n + 1) // 2


def verify(g, n):
    T = magic_constant(n)
    flat = [v for row in g for v in row]
    return (sorted(flat) == list(range(1, n * n + 1))
            and all(sum(row) == T for row in g)
            and all(sum(g[r][c] for r in range(n)) == T for c in range(n))
            and sum(g[i][i] for i in range(n)) == T
            and sum(g[i][n - 1 - i] for i in range(n)) == T)


def build_model(n, symmetry=True):
    N, T = n * n, magic_constant(n)
    m = cp_model.CpModel()
    x = [[m.NewIntVar(1, N, f"x_{r}_{c}") for c in range(n)] for r in range(n)]
    m.AddAllDifferent([v for row in x for v in row])
    for r in range(n):
        m.Add(sum(x[r][c] for c in range(n)) == T)
    for c in range(n):
        m.Add(sum(x[r][c] for r in range(n)) == T)
    m.Add(sum(x[i][i] for i in range(n)) == T)
    m.Add(sum(x[i][n - 1 - i] for i in range(n)) == T)
    if symmetry and n >= 3:
        m.Add(x[0][0] < x[0][n - 1])
        m.Add(x[0][0] < x[n - 1][0])
        m.Add(x[0][0] < x[n - 1][n - 1])
        m.Add(x[0][n - 1] < x[n - 1][0])
    return m, x


def solve(n, workers=8, timeout=120.0, seed=0, symmetry=True):
    t0 = time.perf_counter()
    m, x = build_model(n, symmetry)
    t_build = time.perf_counter() - t0
    solver = cp_model.CpSolver()
    try:
        solver.parameters.num_workers = workers
    except AttributeError:                      # older OR-Tools
        solver.parameters.num_search_workers = workers
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.random_seed = seed
    t1 = time.perf_counter()
    status = solver.Solve(m)
    t_solve = time.perf_counter() - t1
    name = solver.StatusName(status)
    grid = None
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        grid = [[solver.Value(x[r][c]) for c in range(n)] for r in range(n)]
    return dict(n=n, status=name, grid=grid, build=t_build, solve=t_solve, total=time.perf_counter() - t0,
                branches=solver.NumBranches(), conflicts=solver.NumConflicts(), workers=workers,
                verified=bool(grid) and verify(grid, n))


FIELDS = ["n", "workers", "status", "build_seconds", "solve_seconds", "total_seconds",
          "branches", "conflicts", "verified"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int)
    ap.add_argument("--min-n", type=int, default=1)
    ap.add_argument("--max-n", type=int, default=10)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds per size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--no-symmetry", action="store_true")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    sizes = [a.n] if a.n else range(a.min_n, a.max_n + 1)
    for n in sizes:
        for rep in range(a.repeat):
            r = solve(n, a.workers, a.timeout, a.seed + rep, not a.no_symmetry)
            print(f"n={n} workers={r['workers']} status={r['status']} solve={r['solve']:.4f}s "
                  f"build={r['build']:.4f}s branches={r['branches']} conflicts={r['conflicts']} verified={r['verified']}",
                  flush=True)
            if r["grid"] and rep == 0 and n <= 12:
                for row in r["grid"]:
                    print("  " + " ".join(f"{v:3d}" for v in row))
            if a.csv:
                new = not os.path.exists(a.csv)
                os.makedirs(os.path.dirname(os.path.abspath(a.csv)), exist_ok=True)
                with open(a.csv, "a", newline="") as f:
                    w = csv.writer(f)
                    if new: w.writerow(FIELDS)
                    w.writerow([n, r["workers"], r["status"], f"{r['build']:.6f}", f"{r['solve']:.6f}",
                                f"{r['total']:.6f}", r["branches"], r["conflicts"], r["verified"]])
        if r["status"] == "UNKNOWN":
            print("  timeout reached; larger sizes will not be attempted"); break


if __name__ == "__main__":
    main()
