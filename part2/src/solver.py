"""Parallel greedy backtracking magic-square solver.

The first row is split across multiprocessing workers.  Each worker performs
row-major recursive backtracking and rejects partial rows, columns, and diagonals
as soon as they cannot reach the target sum.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable


def magic_square_target(n: int) -> int:
    if n < 1:
        raise ValueError("n must be at least 1")
    return n * (n * n + 1) // 2


def solve_with_fixed_first_row(
    n: int,
    first_row: tuple[int, ...],
    timeout_seconds: float,
) -> dict:
    target = magic_square_target(n)
    values = list(range(1, n * n + 1))
    square = [[0] * n for _ in range(n)]
    square[0] = list(first_row)
    used = set(first_row)
    row_sums = [sum(first_row)] + [0] * (n - 1)
    col_sums = [square[0][c] for c in range(n)]
    diag_main = square[0][0]
    diag_anti = square[0][n - 1]
    nodes = 0
    stop = False
    solution: list[list[int]] | None = None

    tracemalloc.start()
    start = time.perf_counter()

    def can_place(value: int, r: int, c: int) -> bool:
        if value in used:
            return False
        if row_sums[r] + value > target:
            return False
        if col_sums[c] + value > target:
            return False
        if r == c and diag_main + value > target:
            return False
        if c == n - r - 1 and diag_anti + value > target:
            return False
        return True

    def search(pos: int) -> bool:
        nonlocal nodes, stop, solution, diag_main, diag_anti
        nodes += 1

        elapsed = time.perf_counter() - start
        if elapsed >= timeout_seconds:
            stop = True
            return False

        if pos == n * n:
            if (
                all(sum(row) == target for row in square)
                and all(sum(square[r][c] for r in range(n)) == target for c in range(n))
                and sum(square[i][i] for i in range(n)) == target
                and sum(square[i][n - i - 1] for i in range(n)) == target
            ):
                solution = [row[:] for row in square]
                return True
            return False

        r, c = divmod(pos, n)
        if c == 0:
            if r > 0:
                row_sums[r] = 0

        for value in values:
            if not can_place(value, r, c):
                continue

            square[r][c] = value
            used.add(value)
            row_sums[r] += value
            col_sums[c] += value
            if r == c:
                diag_main += value
            if c == n - r - 1:
                diag_anti += value

            nr, nc = divmod(pos + 1, n)
            feasible = True
            if nc == 0 and pos + 1 < n * n:
                feasible = row_sums[nr - 1] <= target

            if feasible and search(pos + 1):
                return True
            if stop:
                return False

            if r == c:
                diag_main -= value
            if c == n - r - 1:
                diag_anti -= value
            col_sums[c] -= value
            row_sums[r] -= value
            used.remove(value)
            square[r][c] = 0

        return False

    found = search(n)
    elapsed = time.perf_counter() - start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "solution": solution,
        "found": found,
        "timed_out": stop,
        "elapsed_seconds": elapsed,
        "nodes": nodes,
        "peak_traced_bytes": peak_bytes,
        "first_row": list(first_row),
    }


def first_row_candidates(n: int) -> list[tuple[int, ...]]:
    target = magic_square_target(n)
    candidates = [
        permutation
        for permutation in itertools.permutations(range(1, n * n + 1), n)
        if sum(permutation) == target
    ]
    return candidates


def solve_parallel(n: int, workers: int, timeout_seconds: float) -> dict:
    if n < 1:
        raise ValueError("n must be at least 1")
    target = magic_square_target(n)
    start = time.perf_counter()

    if n == 1:
        return {
            "part": 2,
            "method": "parallel_row_major_backtracking",
            "n": n,
            "target": target,
            "solution": [[1]],
            "elapsed_seconds": time.perf_counter() - start,
            "nodes": 1,
            "peak_traced_bytes": 0,
            "workers": workers,
            "timed_out": False,
            "first_rows_tested": 1,
        }

    candidates = first_row_candidates(n)
    if not candidates:
        return {
            "part": 2,
            "method": "parallel_row_major_backtracking",
            "n": n,
            "target": target,
            "solution": None,
            "elapsed_seconds": time.perf_counter() - start,
            "nodes": 0,
            "peak_traced_bytes": 0,
            "workers": workers,
            "timed_out": False,
            "first_rows_tested": 0,
        }

    solution = None
    total_nodes = 0
    peak_bytes = 0
    timed_out = False
    first_rows_tested = 0

    executor_workers = max(1, min(workers, len(candidates)))
    deadline = start + timeout_seconds
    with ProcessPoolExecutor(max_workers=executor_workers) as executor:
        try:
            for batch_start in range(0, len(candidates), executor_workers):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    timed_out = True
                    break

                batch = candidates[batch_start:batch_start + executor_workers]
                futures = [
                    executor.submit(
                        solve_with_fixed_first_row,
                        n,
                        candidate,
                        remaining,
                    )
                    for candidate in batch
                ]
                for future in as_completed(futures):
                    result = future.result()
                    first_rows_tested += 1
                    total_nodes += result["nodes"]
                    peak_bytes = max(peak_bytes, result["peak_traced_bytes"])
                    timed_out = timed_out or result["timed_out"]
                    if result["found"] and solution is None:
                        solution = result["solution"]

                if solution is not None:
                    break
        except KeyboardInterrupt:
            timed_out = True

    elapsed = time.perf_counter() - start
    if elapsed >= timeout_seconds:
        timed_out = True

    return {
        "part": 2,
        "method": "parallel_row_major_backtracking",
        "n": n,
        "target": target,
        "solution": solution,
        "elapsed_seconds": elapsed,
        "nodes": total_nodes,
        "peak_traced_bytes": peak_bytes,
        "workers": executor_workers,
        "timed_out": timed_out,
        "first_rows_tested": first_rows_tested,
    }


def benchmark_sizes(
    start_n: int,
    end_n: int,
    workers: int,
    timeout_seconds: float,
) -> list[dict]:
    results = []
    for n in range(start_n, end_n + 1):
        print(f"Part 2: solving n={n} ...", flush=True)
        result = solve_parallel(n, workers, timeout_seconds)
        results.append(result)
        if result["solution"]:
            print(
                f"  found solution in {result['elapsed_seconds']:.6f}s "
                f"({result['nodes']} nodes)"
            )
        else:
            print(
                f"  no solution in {result['elapsed_seconds']:.6f}s "
                f"({result['nodes']} nodes)"
            )
    return results


def append_csv(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "part", "method", "n", "target", "elapsed_seconds", "nodes",
        "peak_traced_bytes", "workers", "timed_out", "first_rows_tested",
    ]
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        import csv
        writer = csv.DictWriter(file, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow({key: result[key] for key in header})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 2 parallel backtracking solver")
    parser.add_argument("--n", type=int, help="Solve one square size")
    parser.add_argument("--min-n", type=int, default=1)
    parser.add_argument("--max-n", type=int, default=5)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("part2/results.json"),
        help="JSON output path",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("plots/plot-part2.csv"),
        help="Benchmark CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n is not None:
        results = [solve_parallel(args.n, args.workers, args.timeout)]
    else:
        results = benchmark_sizes(
            args.min_n,
            args.max_n,
            args.workers,
            args.timeout,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for result in results:
        append_csv(args.csv, result)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    mp.freeze_support()
    main()
