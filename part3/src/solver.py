"""Magic-square solver using Google OR-Tools CP-SAT.

CP-SAT is selected because it is a specialized constraint-programming solver,
supports AllDifferent and linear row/column constraints directly, and is much
better suited to this problem than a general continuous optimization library.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from ortools.sat.python import cp_model


def magic_square_target(n: int) -> int:
    if n < 1:
        raise ValueError("n must be at least 1")
    return n * (n * n + 1) // 2


def solve_cpsat(n: int, timeout_seconds: float, workers: int = 1) -> dict:
    if n < 1:
        raise ValueError("n must be at least 1")

    start = time.perf_counter()
    target = magic_square_target(n)
    model = cp_model.CpModel()
    square = [
        [model.new_int_var(1, n * n, f"x_{r}_{c}") for c in range(n)]
        for r in range(n)
    ]
    model.add_all_different([square[r][c] for r in range(n) for c in range(n)])

    for r in range(n):
        model.add(sum(square[r]) == target)
    for c in range(n):
        model.add(sum(square[r][c] for r in range(n)) == target)

    model.add(sum(square[i][i] for i in range(n)) == target)
    model.add(sum(square[i][n - i - 1] for i in range(n)) == target)

    # Remove equivalent rotational/reflectional solutions for faster search.
    if n > 1:
        model.add(square[0][0] < square[0][n - 1])
        model.add(square[0][0] < square[n - 1][0])

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_seconds
    solver.parameters.num_search_workers = max(1, workers)
    solver.parameters.log_search_progress = False

    status = solver.solve(model)
    elapsed = time.perf_counter() - start
    status_name = solver.status_name(status)
    solution = None

    if status in (cp_model.FEASIBLE, cp_model.UNKNOWN):
        if status == cp_model.FEASIBLE:
            solution = [
                [solver.value(square[r][c]) for c in range(n)]
                for r in range(n)
            ]
    elif status == cp_model.OPTIMAL:
        solution = [
            [solver.value(square[r][c]) for c in range(n)]
            for r in range(n)
        ]

    return {
        "part": 3,
        "method": "or_tools_cp_sat",
        "n": n,
        "target": target,
        "solution": solution,
        "elapsed_seconds": elapsed,
        "status": status_name,
        "timed_out": status == cp_model.UNKNOWN,
        "solver_workers": max(1, workers),
        "conflicts": solver.num_conflicts,
        "branches": solver.num_branches,
        "wall_time": solver.wall_time,
    }


def benchmark_sizes(
    start_n: int,
    end_n: int,
    timeout_seconds: float,
    workers: int,
) -> list[dict]:
    results = []
    for n in range(start_n, end_n + 1):
        print(f"Part 3: solving n={n} ...", flush=True)
        result = solve_cpsat(n, timeout_seconds, workers)
        results.append(result)
        if result["solution"]:
            print(
                f"  found solution in {result['elapsed_seconds']:.6f}s "
                f"(status={result['status']})"
            )
        else:
            print(
                f"  no solution in {result['elapsed_seconds']:.6f}s "
                f"(status={result['status']})"
            )
    return results


def append_csv(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "part", "method", "n", "target", "elapsed_seconds", "status",
        "timed_out", "solver_workers", "conflicts", "branches", "wall_time",
    ]
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow({key: result[key] for key in header})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 3 CP-SAT magic-square solver")
    parser.add_argument("--n", type=int, help="Solve one square size")
    parser.add_argument("--min-n", type=int, default=1)
    parser.add_argument("--max-n", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("part3/results.json"),
        help="JSON output path",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("plots/plot-part3.csv"),
        help="Benchmark CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n is not None:
        results = [solve_cpsat(args.n, args.timeout, args.workers)]
    else:
        results = benchmark_sizes(
            args.min_n,
            args.max_n,
            args.timeout,
            args.workers,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for result in results:
        append_csv(args.csv, result)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
