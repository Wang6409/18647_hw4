#!/usr/bin/env python3
"""Part 2: recursive backtracking magic-square solver (pure Python, parallel).

Finds ONE magic square of size n (numbers 1..n^2, all rows, columns and both main
diagonals sum to T = n(n^2+1)/2).

Search: cells are filled in row-major order; every choice forks the search tree
(recursive function `rec`).  Pruning (early rejection):
  * a number is used at most once;
  * the last cell of a row is *forced* (T - row_sum); the last cell of a column
    (last row) is forced (T - col_sum) -> no branching there;
  * for every line (row / column / diagonal) through the new cell, the sum of the
    still-empty cells k must be reachable:  k(k+1)/2 <= remaining <= k*N - k(k-1)/2
    (sum of the k smallest / k largest possible distinct values);
  * completed diagonals must equal T.

Parallelism: the tree is cut into independent sub-trees by enumerating all valid
prefixes of the first `depth` cells (in lexicographic order); a pool of workers
searches sub-trees; the first solution found stops everybody (shared Event).
  --backend process : multiprocessing pool (real parallel speed-up; Python's GIL
                      prevents CPU speed-up with threads)           [default]
  --backend thread  : thread pool (same code, GIL-limited; for comparison)

Usage examples
  python backtrack_magic.py --n 4
  python backtrack_magic.py --n 5 --workers 8 --order random --seed 1
  python backtrack_magic.py --min-n 1 --max-n 7 --workers 8 --timeout 120 --csv ../../plots/plot-part2.csv
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import random
import resource
import sys
import time
from concurrent.futures import (FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait)


class _Abort(Exception):
    pass


def magic_constant(n: int) -> int:
    return n * (n * n + 1) // 2


def search(n, prefix=(), collect_depth=None, stop_check=None, order="asc", seed=0):
    """Backtracking search below `prefix`.

    collect_depth=None : return (solution_list_or_None, nodes_visited)
    collect_depth=d    : return (list_of_valid_prefixes_of_length_d, nodes_visited)
    """
    N = n * n
    T = magic_constant(n)
    grid = [0] * N
    used = [False] * (N + 2)
    rows = [0] * n
    cols = [0] * n
    diag = [0, 0]  # main, anti
    nodes = 0
    out = []
    rng = random.Random(seed)
    lo = [k * (k + 1) // 2 for k in range(n + 1)]                # smallest possible sum of k distinct values
    hi = [k * N - k * (k - 1) // 2 for k in range(n + 1)]        # largest possible sum of k distinct values

    def feasible(pos, v):
        """Can value v be put at pos without violating a line bound?"""
        r, c = divmod(pos, n)
        k = n - 1 - c                       # empty cells left in row after this one
        rem = T - rows[r] - v
        if rem < lo[k] or rem > hi[k]:
            return False
        k = n - 1 - r                       # empty cells left in column
        rem = T - cols[c] - v
        if rem < lo[k] or rem > hi[k]:
            return False
        if r == c:
            k = n - 1 - r
            rem = T - diag[0] - v
            if rem < lo[k] or rem > hi[k]:
                return False
        if c == n - 1 - r:
            k = n - 1 - r
            rem = T - diag[1] - v
            if rem < lo[k] or rem > hi[k]:
                return False
        return True

    def place(pos, v):
        r, c = divmod(pos, n)
        grid[pos] = v; used[v] = True
        rows[r] += v; cols[c] += v
        if r == c: diag[0] += v
        if c == n - 1 - r: diag[1] += v

    def unplace(pos, v):
        r, c = divmod(pos, n)
        grid[pos] = 0; used[v] = False
        rows[r] -= v; cols[c] -= v
        if r == c: diag[0] -= v
        if c == n - 1 - r: diag[1] -= v

    def rec(pos, stop_depth):
        """Fill cell `pos` and recurse. Returns True when a full solution is in `grid`."""
        nonlocal nodes
        if pos == stop_depth:
            if collect_depth is not None:
                out.append(tuple(grid[:pos]))
                return False
            return True                      # pos == N: all cells filled and all lines verified
        nodes += 1
        if stop_check is not None and (nodes & 0x7FF) == 0 and stop_check():
            raise _Abort
        r, c = divmod(pos, n)
        if c == n - 1:                       # forced by row sum
            cand = [T - rows[r]]
        elif r == n - 1:                     # forced by column sum
            cand = [T - cols[c]]
        else:
            cand = range(1, N + 1)
            if order != "asc":
                cand = list(cand)
                if order == "random": rng.shuffle(cand)
                else: cand.reverse()
        for v in cand:
            if v < 1 or v > N or used[v]:
                continue
            if not feasible(pos, v):
                continue
            place(pos, v)
            # completed diagonals must hit T exactly (bound check above gives rem==0 -> lo[0]=hi[0]=0)
            if rec(pos + 1, stop_depth):
                return True
            unplace(pos, v)
        return False

    target_depth = N if collect_depth is None else collect_depth
    # apply the given prefix (assumed valid)
    for i, v in enumerate(prefix):
        place(i, v)
    try:
        found = rec(len(prefix), target_depth)
    except _Abort:
        return ("aborted", nodes)
    if collect_depth is not None:
        return out, nodes
    return (grid[:] if found else None), nodes


# ----------------------------------------------------------------------------- parallel driver
_EVENT = None


def _init_worker(ev):
    global _EVENT
    _EVENT = ev


def _run_task(args):
    n, prefix, order, seed, deadline = args
    def stop_check():
        return _EVENT.is_set() or (deadline is not None and time.time() >= deadline)
    if _EVENT.is_set():
        return None, 0, False
    res, nodes = search(n, prefix, None, stop_check, order, seed)
    if res == "aborted":
        return None, nodes, False
    return res, nodes, True          # (solution or None, nodes, finished_subtree)


def peak_rss_mb():
    scale = 1024 * 1024 if sys.platform == "darwin" else 1024
    me = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale
    ch = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / scale
    return me, ch


def verify(sq, n):
    T = magic_constant(n)
    g = [sq[r * n:(r + 1) * n] for r in range(n)]
    ok = sorted(sq) == list(range(1, n * n + 1))
    ok &= all(sum(row) == T for row in g)
    ok &= all(sum(g[r][c] for r in range(n)) == T for c in range(n))
    ok &= sum(g[i][i] for i in range(n)) == T and sum(g[i][n - 1 - i] for i in range(n)) == T
    return bool(ok)


def solve(n, workers=1, backend="process", order="asc", seed=0, timeout=None, tasks_per_worker=16):
    t0 = time.time()
    deadline = None if timeout is None else t0 + timeout
    if workers <= 1:
        stop = (lambda: time.time() >= deadline) if deadline else None
        res, nodes = search(n, (), None, stop, order, seed)
        status = "timeout" if res == "aborted" else ("found" if res else "no_solution")
        sol = res if isinstance(res, list) else None
        return dict(n=n, workers=1, backend="serial", status=status, solution=sol, nodes=nodes,
                    elapsed=time.time() - t0, tasks=1)

    # split into sub-trees: smallest prefix depth giving >= tasks_per_worker*workers tasks
    nodes = 0
    depth = 1
    while True:
        prefixes, cnt = search(n, (), depth, None, "asc", 0)
        nodes += cnt
        if len(prefixes) >= tasks_per_worker * workers or depth >= n * n - 1:
            break
        depth += 1
    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    ev = threading_event = None
    if backend == "process":
        ev = ctx.Event()
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(ev,))
    else:
        import threading
        ev = threading.Event()
        _init_worker(ev)
        pool = ThreadPoolExecutor(max_workers=workers)
    futures = [pool.submit(_run_task, (n, p, order, seed + i, deadline)) for i, p in enumerate(prefixes)]
    solution, pending, status = None, set(futures), "no_solution"
    try:
        while pending:
            done, pending = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
            for f in done:
                sol, nd, _fin = f.result()
                nodes += nd
                if sol is not None and solution is None:
                    solution, status = sol, "found"
                    ev.set()
            if solution is not None:
                break
            if deadline is not None and time.time() >= deadline:
                ev.set(); status = "timeout"; break
    finally:
        ev.set()
        for f in pending:
            f.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
    for f in futures:                      # count nodes of tasks that finished while shutting down
        if f in pending and f.done() and not f.cancelled():
            try: nodes += f.result()[1]
            except Exception: pass
    return dict(n=n, workers=workers, backend=backend, status=status, solution=solution, nodes=nodes,
                elapsed=time.time() - t0, tasks=len(prefixes))


CSV_FIELDS = ["n", "workers", "backend", "order", "status", "elapsed_seconds", "nodes_visited",
              "nodes_per_second", "tasks", "peak_rss_parent_mb", "peak_rss_worker_mb", "verified"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int)
    ap.add_argument("--min-n", type=int, default=1)
    ap.add_argument("--max-n", type=int, default=6)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--backend", choices=["process", "thread"], default="process")
    ap.add_argument("--order", choices=["asc", "desc", "random"], default="asc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds per size (0 = none)")
    ap.add_argument("--csv", default=None, help="append results to this CSV file")
    a = ap.parse_args()
    sizes = [a.n] if a.n else range(a.min_n, a.max_n + 1)
    timeout = a.timeout or None
    for n in sizes:
        r = solve(n, a.workers, a.backend, a.order, a.seed, timeout)
        sol = r["solution"]
        ok = bool(sol) and verify(sol, n)
        me, ch = peak_rss_mb()
        rate = r["nodes"] / r["elapsed"] if r["elapsed"] > 0 else 0
        print(f"n={n} workers={r['workers']} backend={r['backend']} status={r['status']} "
              f"time={r['elapsed']:.4f}s nodes={r['nodes']} ({rate:.3g}/s) tasks={r['tasks']} "
              f"rss_parent={me:.0f}MB rss_worker_max={ch:.0f}MB verified={ok}", flush=True)
        if sol:
            for i in range(n):
                print("  " + " ".join(f"{v:3d}" for v in sol[i * n:(i + 1) * n]))
        if a.csv:
            new = not os.path.exists(a.csv)
            os.makedirs(os.path.dirname(os.path.abspath(a.csv)), exist_ok=True)
            with open(a.csv, "a", newline="") as f:
                w = csv.writer(f)
                if new: w.writerow(CSV_FIELDS)
                w.writerow([n, r["workers"], r["backend"], a.order, r["status"], f"{r['elapsed']:.6f}",
                            r["nodes"], f"{rate:.1f}", r["tasks"], f"{me:.1f}", f"{ch:.1f}", ok])
        if r["status"] == "timeout":
            print("  timeout reached; larger sizes will not be attempted"); break


if __name__ == "__main__":
    main()
