#!/usr/bin/env python3
"""Fit runtime / node / memory models for Part 2 (exponential growth in n) and print speed-ups.
Model: log10(median time to first solution) = a + b*n  fitted on sizes n>=4 that were solved.
Usage: python3 fit_part2.py --csv ../../plots/plot-part2.csv
"""
import argparse, csv, math, statistics as st
from collections import defaultdict

ap = argparse.ArgumentParser(); ap.add_argument("--csv", default="../../plots/plot-part2.csv")
ap.add_argument("--budget", type=float, default=3600.0, help="'reasonable' time budget in seconds")
a = ap.parse_args()
rows = list(csv.DictReader(open(a.csv)))
by = defaultdict(list)                       # (workers, n) -> list of rows
for r in rows: by[(int(r["workers"]), int(r["n"]))].append(r)
workers = sorted({w for w, _ in by})

print("median time to first solution [s] (solved/attempts) and median nodes")
for w in workers:
    for n in sorted({n for ww, n in by if ww == w}):
        rs = by[(w, n)]
        ok = [r for r in rs if r["status"] == "found"]
        if ok:
            print(f"  workers={w} n={n}: T={st.median(float(r['elapsed_seconds']) for r in ok):.4g}s "
                  f"nodes={st.median(int(r['nodes_visited']) for r in ok):.3g} ({len(ok)}/{len(rs)}) "
                  f"rss_worker={max(float(r['peak_rss_worker_mb']) for r in ok):.0f}MB")
        else:
            print(f"  workers={w} n={n}: unsolved in budget (0/{len(rs)})")

def linfit(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a0 = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys); ss_res = sum((y - (a0 + b * x)) ** 2 for x, y in zip(xs, ys))
    return a0, b, (1 - ss_res / ss_tot) if ss_tot else float("nan")

w1 = workers[0]
pts = [(n, math.log10(st.median(float(r["elapsed_seconds"]) for r in by[(w1, n)] if r["status"] == "found")))
       for n in sorted({n for ww, n in by if ww == w1}) if n >= 4 and any(r["status"] == "found" for r in by[(w1, n)])]
if len(pts) >= 2:
    a0, b, r2 = linfit([p[0] for p in pts], [p[1] for p in pts])
    print(f"\nfit (workers={w1}): log10 T = {a0:.3f} + {b:.3f} n   R^2={r2:.3f}   (growth x{10**b:.1f} per +1 in n)")
    n_star = (math.log10(a.budget) - a0) / b
    if len(pts) < 3: print("  WARNING: <3 solved sizes -> R^2 is meaningless; treat the extrapolation as a rough guess")
    print(f"predicted largest n within {a.budget:.0f}s: n ~ {n_star:.1f}")
else:
    print("\nnot enough solved sizes (n>=4) for an exponential fit")

print("\nspeed-up vs 1 worker (median time ratio):")
for n in sorted({n for _, n in by}):
    base = [float(r["elapsed_seconds"]) for r in by.get((w1, n), []) if r["status"] == "found"]
    if len(base) < 1 or n < 4: continue
    line = f"  n={n}: "
    for w in workers[1:]:
        t = [float(r["elapsed_seconds"]) for r in by.get((w, n), []) if r["status"] == "found"]
        if t: line += f"w={w}: x{st.median(base)/st.median(t):.2f}  "
    print(line)
