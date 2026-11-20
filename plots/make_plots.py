#!/usr/bin/env python3
"""Generate the three plots for HW4 (PDF + PNG each) from the experiment CSVs.

Run from the plots/ directory after the experiment scripts:
    python make_plots.py              # all available parts
    python make_plots.py --parts 1 3  # selected parts
Outputs: plot-part1.{pdf,png}, plot-part2.{pdf,png}, plot-part3.{pdf,png}, plot-compare.{pdf,png}
Inputs : plot-part1.csv (magic_bf), plot-part2.csv (backtrack_magic.py), plot-part3.csv (cp_sat_magic.py)
Timeouts are drawn as hollow markers with an up-arrow (= lower bound), never as exact runtimes.
Requires: numpy, matplotlib.
"""
import argparse, csv, math, os, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

YEAR = 365.25 * 86400
AGE_UNIVERSE = 13.8e9 * YEAR
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8.5,
                     "figure.dpi": 100, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3})
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def read(path):
    if not os.path.exists(path):
        print(f"[skip] {path} not found"); return []
    with open(path) as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.pdf / {name}.png")


def arrow_marker(ax, x, y, color, label=None):
    """hollow marker + up arrow = timeout (lower bound)"""
    ax.plot([x], [y], marker="o", mfc="white", mec=color, color=color, ls="none", ms=7, label=label, zorder=5)
    ax.annotate("", xy=(x, y), xycoords="data", xytext=(0, 24), textcoords="offset points",
                arrowprops=dict(arrowstyle="<-", color=color, lw=1.5, shrinkA=0, shrinkB=5))


def headroom(ax, factor=30):
    """leave room above the data so timeout arrows are not clipped (log axis)"""
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi * factor)


def l10fact(m):
    return math.lgamma(m + 1) / math.log(10)


def fmt(s):
    for lim, div, u in [(1e-3, 1e-6, "µs"), (1, 1e-3, "ms"), (60, 1, "s"), (3600, 60, "min"), (86400, 3600, "h"), (YEAR, 86400, "d")]:
        if s < lim: return f"{s/div:.3g} {u}"
    return f"{s/YEAR:.2g} yr" if s / YEAR < 1e6 else f"1e{math.log10(s/YEAR):.0f} yr"


# --------------------------------------------------------------------------------------------- Part 1
def plot_part1(path="plot-part1.csv"):
    rows = read(path)
    if not rows: return
    th = max(int(r["threads"]) for r in rows)
    rows = [r for r in rows if int(r["threads"]) == th and float(r["elapsed_seconds"]) > 0]
    P = np.array([float(r["permutations_checked"]) for r in rows]); T = np.array([float(r["elapsed_seconds"]) for r in rows])
    ns = np.array([int(r["n"]) for r in rows])
    # fit T = t0 + P/R, relative-error weighted least squares
    w = 1 / T
    A = np.vstack([np.ones_like(P), P]).T
    (t0, b), *_ = np.linalg.lstsq(A * w[:, None], T * w, rcond=None)
    R = 1 / b; pred = t0 + b * P
    r2 = 1 - np.sum((np.log(T) - np.log(pred)) ** 2) / np.sum((np.log(T) - np.log(T).mean()) ** 2)
    thr = {n: float(np.median([p / t for p, t, k in zip(P, T, ns) if k == n and p >= 1e7])) for n in set(ns) if any(p >= 1e7 for p, k in zip(P, ns) if k == n)}
    R_ext = thr.get(5) or thr.get(4) or R

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    # (a) runtime vs permutations
    mk = {1: "o", 2: "s", 3: "^", 4: "D", 5: "v"}
    for n in sorted(set(ns)):
        sel = ns == n
        ax[0].loglog(P[sel], T[sel], mk.get(n, "o"), color=COLORS[(n - 1) % 6], alpha=.75, label=f"measured, n={n}")
    xx = np.logspace(0, math.log10(P.max()) + .3, 100)
    ax[0].loglog(xx, t0 + b * xx, "k--", label=f"fit T = t0 + P/R\nR = {R:.2e} perms/s, R² = {r2:.4f}\n(log scale; threads = {th})")
    ax[0].set_xlabel("permutations enumerated P"); ax[0].set_ylabel("wall-clock time [s]")
    ax[0].set_title("(a) Part 1 brute force: measured runtime vs. work"); ax[0].legend(loc="upper left")
    # (b) per-n picture: measured first-solution, timeout, extrapolated full-space time
    first = [r for r in read(path) if r["mode"] == "first" and int(r["threads"]) == th]
    found = {int(r["n"]): float(r["elapsed_seconds"]) for r in first if r["status"] == "found"}
    tout = {int(r["n"]): float(r["elapsed_seconds"]) for r in first if r["status"] == "timeout"}
    nn = np.arange(2, 9)
    Tfull = [10 ** (l10fact(n * n) - math.log10(R_ext)) for n in nn]
    ax[1].semilogy(nn, Tfull, "--", color="gray", marker=".", label=f"extrapolated full enumeration (n²)!/R,\nR = {R_ext:.2e}/s")
    ns_f = sorted(found)
    ax[1].semilogy(ns_f, [max(found[n], 1e-7) for n in ns_f], "o-", color=COLORS[0], label="measured: first solution found")
    for i, (n, t) in enumerate(sorted(tout.items())):
        arrow_marker(ax[1], n, t, COLORS[1], "timeout (lower bound)" if i == 0 else None)
    for name, secs in [("1 hour", 3600), ("1 year", YEAR), ("age of universe", AGE_UNIVERSE)]:
        ax[1].axhline(secs, ls=":", color="k", lw=.8); ax[1].text(2.05, secs * 1.5, name, fontsize=8)
    ax[1].set_ylim(1e-7, 1e40); ax[1].set_xlabel("square size n"); ax[1].set_ylabel("time [s] (log scale)")
    ax[1].set_title("(b) Part 1: first-solution time and extrapolation"); ax[1].legend(loc="upper left")
    save(fig, "plot-part1")


# --------------------------------------------------------------------------------------------- Part 2
def plot_part2(path="plot-part2.csv"):
    rows = read(path)
    if not rows: return
    workers = sorted({int(r["workers"]) for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    solved_med = {}
    for i, w in enumerate(workers):
        col = COLORS[i % 6]; xs, ys, lo, hi = [], [], [], []
        for n in sorted({int(r["n"]) for r in rows if int(r["workers"]) == w}):
            rs = [r for r in rows if int(r["workers"]) == w and int(r["n"]) == n]
            ok = [float(r["elapsed_seconds"]) for r in rs if r["status"] == "found"]
            if ok:
                xs.append(n); ys.append(max(st.median(ok), 1e-5)); lo.append(max(min(ok), 1e-5)); hi.append(max(ok))
                solved_med[(w, n)] = st.median(ok)
            if not ok and any(r["status"] == "timeout" for r in rs):
                arrow_marker(ax[0], n, max(float(r["elapsed_seconds"]) for r in rs), col)
            elif ok and any(r["status"] == "timeout" for r in rs):   # some seeds timed out
                ax[0].plot([n], [max(float(r["elapsed_seconds"]) for r in rs if r["status"] == "timeout")], "x", color=col, ms=6)
        label = "1 process (serial)" if w == 1 else f"{w} processes"
        ax[0].semilogy(xs, ys, "o-", color=col, label=label)
        ax[0].fill_between(xs, lo, hi, color=col, alpha=.15)
    ax[0].plot([], [], "o", mfc="white", mec="k", label="all seeds timed out (lower bound)")
    ax[0].plot([], [], "x", color="k", label="some seeds timed out")
    ax[0].set_xlabel("square size n"); ax[0].set_ylabel("median time to first solution [s] (log scale)")
    ax[0].set_title("(a) Part 2: runtime vs. n (band = min..max over seeds)"); headroom(ax[0])
    ax[0].legend(loc="upper left")
    # (b) speed-up vs workers for largest n solved by every worker count
    common = [n for n in sorted({n for _, n in solved_med}) if n >= 4 and all((w, n) in solved_med for w in workers)]
    for i, n in enumerate(common[-3:]):
        base = solved_med[(workers[0], n)]
        ax[1].plot(workers, [base / solved_med[(w, n)] for w in workers], "o-", color=COLORS[i], label=f"n = {n}")
    ax[1].plot(workers, [w / workers[0] for w in workers], "k--", label="ideal linear")
    ax[1].set_xscale("log", base=2); ax[1].set_xticks(workers); ax[1].set_xticklabels(workers)
    ax[1].set_xlabel("worker processes"); ax[1].set_ylabel("speed-up (median time ratio)")
    ax[1].set_title("(b) Part 2: speed-up of the process pool"); ax[1].legend()
    save(fig, "plot-part2")


# --------------------------------------------------------------------------------------------- Part 3
def plot_part3(path="plot-part3.csv"):
    rows = read(path)
    if not rows: return
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    by = {}
    for r in rows: by.setdefault(int(r["n"]), []).append(r)
    xs, solve, build, lo, hi = [], [], [], [], []
    for n in sorted(by):
        ok = [r for r in by[n] if r["status"] in ("OPTIMAL", "FEASIBLE")]
        if ok:
            s = [max(float(r["solve_seconds"]), 1e-5) for r in ok]
            xs.append(n); solve.append(st.median(s)); lo.append(min(s)); hi.append(max(s))
            build.append(max(st.median(float(r["build_seconds"]) for r in ok), 1e-5))
        elif any(r["status"] == "INFEASIBLE" for r in by[n]):
            t = max(st.median(float(r["solve_seconds"]) for r in by[n]), 1e-5)
            ax.plot([n], [t], "x", color="k", ms=9, mew=2, label="proved infeasible (n = 2)")
        elif any(r["status"] == "UNKNOWN" for r in by[n]):
            arrow_marker(ax, n, max(float(r["solve_seconds"]) for r in by[n]), COLORS[1], "timeout (lower bound)")
    ax.semilogy(xs, solve, "o-", color=COLORS[0], label="CP-SAT solve time (median)")
    ax.fill_between(xs, lo, hi, color=COLORS[0], alpha=.15, label="min..max over repeats")
    ax.semilogy(xs, build, "s--", color=COLORS[2], label="model build time (Python)")
    w = rows[0]["workers"]
    ax.set_xlabel("square size n"); ax.set_ylabel("time to first solution [s] (log scale)")
    ax.set_title(f"Part 3 OR-Tools CP-SAT ({w} workers): runtime vs. n")
    headroom(ax); ax.legend(loc="upper left")
    save(fig, "plot-part3")


# --------------------------------------------------------------------------------------------- comparison
def plot_compare(p1="plot-part1.csv", p2="plot-part2.csv", p3="plot-part3.csv"):
    r1 = [r for r in read(p1) if r["mode"] == "first"]; r2 = read(p2); r3 = read(p3)
    if not (r1 or r2 or r3): return
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    def med(rows, ok, key, filt=lambda r: True):
        d = {}
        for r in rows:
            if filt(r) and r["status"] in ok: d.setdefault(int(r["n"]), []).append(max(float(r[key]), 1e-6))
        return {n: st.median(v) for n, v in sorted(d.items())}
    def tos(rows, bad, key, filt=lambda r: True):
        d = {}
        for r in rows:
            if filt(r) and r["status"] in bad: d[int(r["n"])] = max(d.get(int(r["n"]), 0), float(r[key]))
        return d
    th1 = max((int(r["threads"]) for r in r1), default=0)
    s1 = med(r1, ("found",), "elapsed_seconds", lambda r: int(r["threads"]) == th1); t1 = tos(r1, ("timeout",), "elapsed_seconds")
    wmax = max((int(r["workers"]) for r in r2), default=0)
    s2 = med(r2, ("found",), "elapsed_seconds", lambda r: int(r["workers"]) == wmax)
    t2 = {n: t for n, t in tos(r2, ("timeout",), "elapsed_seconds", lambda r: int(r["workers"]) == wmax).items() if n not in s2}
    s3 = med(r3, ("OPTIMAL", "FEASIBLE"), "solve_seconds"); t3 = tos(r3, ("UNKNOWN",), "solve_seconds")
    for (name, s, t), c, m in zip([(f"Part 1 brute force (C++/OpenMP, {th1} threads)", s1, t1),
                                   (f"Part 2 backtracking (Python, {wmax} processes)", s2, t2),
                                   ("Part 3 CP-SAT (OR-Tools)", s3, t3)], COLORS, "osv"):
        if s: ax.semilogy(list(s), list(s.values()), m + "-", color=c, label=name)
        for n, v in t.items(): arrow_marker(ax, n, v, c)
    ax.plot([], [], "o", mfc="white", mec="k", label="timeout (lower bound)")
    ax.set_xlabel("square size n"); ax.set_ylabel("median time to first solution [s] (log scale)")
    ax.set_title("Time to find ONE magic square: Parts 1-3"); headroom(ax); ax.legend(loc="lower right", fontsize=7.5)
    save(fig, "plot-compare")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--parts", nargs="*", type=int, default=[1, 2, 3, 4],
                                                    help="1,2,3 = single plots, 4 = comparison plot")
    a = ap.parse_args()
    if 1 in a.parts: plot_part1()
    if 2 in a.parts: plot_part2()
    if 3 in a.parts: plot_part3()
    if 4 in a.parts: plot_compare()
