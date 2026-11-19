"""Brute-force magic-square solver using permutations.

This implementation intentionally uses no square-sum pruning: every permutation
is written to a flat square and then checked.  It models the requested brute-force
algorithm directly.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Iterable


def magic_square_target(n: int) -> int:
    if n < 1:
        raise ValueError("n must be at least 1")
    return n * (n * n + 1) // 2


def is_magic_square(square: list[list[int]], target: int) -> bool:
    n = len(square)
    if n == 1:
        return square[0][0] == 1

    if any(sum(row) != target for row in square):
        return False
    if any(sum(square[r][c] for r in range(n)) != target for c in range(n)):
        return False
    if sum(square[i][i] for i in range(n)) != target:
        return False
    if sum(square[i][n - i - 1] for i in range(n)) != target:
        return False
    return True


def solve_brute_force(n: int, timeout_seconds: float | None = None) -> dict:
    if n < 1:
        raise ValueError("n must be at least 1")

    target = magic_square_target(n)
    total_permutations = math.factorial(n * n)
    start = time.perf_counter()
    checked = 0
    solution: list[list[int]] | None = None
    timed_out = False

    for permutation in itertools.permutations(range(1, n * n + 1)):
        checked += 1
        square = [
            list(permutation[r * n:(r + 1) * n])
            for r in range(n)
        ]
        if is_magic_square(square, target):
            solution = square
            break

        if timeout_seconds is not None:
            elapsed = time.perf_counter() - start
            if elapsed >= timeout_seconds:
                timed_out = True
                break

    elapsed = time.perf_counter() - start
    return {
        "part": 1,
        "method": "brute_force_permutations",
        "n": n,
        "target": target,
        "solution": solution,
        "elapsed_seconds": elapsed,
        "permutations_checked": checked,
        "total_permutations": total_permutations,
        "timed_out": timed_out,
    }


def benchmark_sizes(
    start_n: int,
    end_n: int,
    timeout_seconds: float,
) -> list[dict]:
    results = []
    for n in range(start_n, end_n + 1):
        print(f"Part 1: solving n={n} ...", flush=True)
        result = solve_brute_force(n, timeout_seconds)
        results.append(result)
        if result["solution"]:
            print(f"  found solution in {result['elapsed_seconds']:.6f}s")
        elif result["timed_out"]:
            print(
                f"  timed out after {result['elapsed_seconds']:.6f}s "
                f"({result['permutations_checked']} permutations)"
            )
        else:
            print(
                f"  no solution after {result['elapsed_seconds']:.6f}s "
                f"({result['permutations_checked']} permutations)"
            )
    return results


def append_csv(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "part", "method", "n", "target", "elapsed_seconds",
        "permutations_checked", "total_permutations", "timed_out",
    ]
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        import csv
        writer = csv.DictWriter(file, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow({key: result[key] for key in header})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 1 brute-force magic-square solver")
    parser.add_argument("--n", type=int, help="Solve one square size")
    parser.add_argument("--min-n", type=int, default=1)
    parser.add_argument("--max-n", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("part1/results.json"),
        help="JSON output path",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("plots/plot-part1.csv"),
        help="Benchmark CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n is not None:
        results = [solve_brute_force(args.n, args.timeout)]
    else:
        results = benchmark_sizes(args.min_n, args.max_n, args.timeout)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for result in results:
        append_csv(args.csv, result)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
