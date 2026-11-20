#!/usr/bin/env python3
"""Fit and extrapolate the runtime model for the Part 1 brute-force solver.

Model:   T(P) = t0 + P / R        (P = permutations enumerated, R = throughput)
Brute force for size n has P = (n^2)!, hence T(n) = t0 + (n^2)! / R.
Also fits the log-log slope (should be ~1 if runtime is linear in P).

Usage: python3 fit_model.py --csv ../../plots/plot-part1.csv --out-dir ../../plots
Requires: numpy, matplotlib (optional, for the PDF plot).
"""
import argparse, csv, math, os
import numpy as np

AGE_UNIVERSE_S = 13.8e9 * 365.25 * 86400
YEAR = 365.25 * 86400


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(dict(n=int(r["n"]), threads=int(r["threads"]),
                             P=float(r["permutations_checked"]),
                             T=float(r["elapsed_seconds"]),
                             mode=r["mode"], solutions=int(r["solutions_found"]),
                             rank=int(r["first_solution_rank"]), status=r["status"],
                             whole=int(r["status"] == "exhausted")))
    return rows


def fmt_time(s):
    if s < 1: return f"{s*1e3:.3g} ms"
    if s < 60: return f"{s:.3g} s"
    if s < 3600: return f"{s/60:.3g} min"
    if s < 86400: return f"{s/3600:.3g} h"
    if s < YEAR: return f"{s/86400:.3g} days"
    if s < 1e6 * YEAR: return f"{s/YEAR:.3g} years"
    return f"10^{math.log10(s/YEAR):.1f} years"


def log10_fact(m):
    return math.lgamma(m + 1) / math.log(10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../../plots/plot-part1.csv")
    ap.add_argument("--out-dir", default="../../plots")
    ap.add_argument("--threads", type=int, default=0, help="thread count to fit (default: max in data)")
    a = ap.parse_args()

    rows = load(a.csv)
    th = a.threads or max(r["threads"] for r in rows)
    rows = [r for r in rows if r["threads"] == th and r["T"] > 0]
    # tiny runs (<1e6 perms) are dominated by thread start-up; keep them for t0 but not for R
    P = np.array([r["P"] for r in rows]); T = np.array([r["T"] for r in rows])

    # --- Model 1: T = t0 + P/R  (least squares, relative-error weighted) ---
    w = 1.0 / T
    A = np.vstack([np.ones_like(P), P]).T
    coef, *_ = np.linalg.lstsq(A * w[:, None], T * w, rcond=None)
    t0, b = coef; R = 1.0 / b
    pred = t0 + b * P
    ss_res = np.sum(((np.log(T) - np.log(np.maximum(pred, 1e-12))) ** 2))
    ss_tot = np.sum((np.log(T) - np.log(T).mean()) ** 2)
    r2_log = 1 - ss_res / ss_tot
    ss_res_lin = np.sum((T - pred) ** 2); ss_tot_lin = np.sum((T - T.mean()) ** 2)
    r2_lin = 1 - ss_res_lin / ss_tot_lin
    mape = np.mean(np.abs(pred - T) / T) * 100

    # --- Model 2: log-log slope ---
    big = P >= 1e5
    slope, icpt = np.polyfit(np.log(P[big]), np.log(T[big]), 1)

    # --- per-n throughput (runs with >= 1e7 perms) ---
    per_n = {}
    for n in sorted({r["n"] for r in rows}):
        thr = [r["P"] / r["T"] for r in rows if r["n"] == n and r["P"] >= 1e7]
        if thr: per_n[n] = float(np.median(thr))

    # sanity: full n<=3 runs, solution counts
    sol = {r["n"]: r["solutions"] for r in rows if r["mode"] == "full" and r["whole"]}
    first = sorted([r for r in rows if r["mode"] == "first" and r["status"] in ("found", "timeout")],
                   key=lambda r: (r["n"], r["T"]))
    # number of normal magic squares (all 8 symmetries counted): n=3: 8, n=4: 7040, n=5: 2202441792
    S = {3: 8, 4: 7040, 5: 2202441792}

    # --- extrapolation ---
    R_use = per_n.get(5) or per_n.get(4) or R
    ext = []
    for n in range(1, 11):
        N = n * n
        l10P = log10_fact(N)
        l10T = l10P - math.log10(R_use)  # seconds (log10)
        ext.append((n, l10P, l10T))
    thresholds = {"1 hour": 3600, "1 day": 86400, "1 year": YEAR, "age of universe": AGE_UNIVERSE_S}

    lines = []
    lines.append(f"threads used for fit        : {th}")
    lines.append(f"data points                 : {len(rows)}")
    lines.append(f"Model T = t0 + P/R")
    lines.append(f"  t0 (start-up overhead)    : {t0*1e6:.1f} us")
    lines.append(f"  R  (throughput)           : {R:.3e} permutations/s")
    lines.append(f"  R^2 (log scale)           : {r2_log:.6f}")
    lines.append(f"  R^2 (linear scale)        : {r2_lin:.6f}")
    lines.append(f"  mean abs rel. error       : {mape:.2f} %")
    lines.append(f"log-log slope (P>=1e5)      : {slope:.4f}  (1.0 = perfectly linear in P)")
    lines.append("median throughput per n (runs >= 1e7 perms): " +
                 ", ".join(f"n={n}: {v:.3e}/s" for n, v in per_n.items()))
    lines.append("solutions found in exhaustive runs: " + ", ".join(f"n={n}: {s}" for n, s in sol.items()))
    lines.append(f"throughput used for extrapolation: {R_use:.3e}/s")
    lines.append("")
    lines.append(f"{'n':>3} {'(n^2)!':>12} {'T = (n^2)!/R':>22}")
    for n, l10P, l10T in ext:
        Ps = f"{10**l10P:.3e}" if l10P < 300 else f"1e{l10P:.1f}"
        Ts = fmt_time(10 ** l10T) if l10T < 300 else f"1e{l10T:.1f} s"
        if l10T - math.log10(YEAR) > 6: Ts = f"10^{l10T - math.log10(YEAR):.1f} years"
        lines.append(f"{n:>3} {Ps:>12} {Ts:>22}")
    lines.append("")
    lines.append("time to first solution (lexicographic brute force, measured):")
    for r in first:
        lines.append(f"  n={r['n']} status={r['status']} T={fmt_time(r['T'])} perms_checked={r['P']:.3e} "
                     f"rank={r['rank'] if r['status']=='found' else 'n/a'}")
    lines.append("expected perms until first solution if solutions were uniformly spread: (n^2)!/S_n")
    for n, Sn in S.items():
        l10 = log10_fact(n * n) - math.log10(Sn)
        lines.append(f"  n={n}: ~10^{l10:.1f} perms -> {fmt_time(10 ** (l10 - math.log10(R_use))) if l10 < 300 else 'inf'}"
                     f" (at {R_use:.2e}/s, {th} thread(s))")
    lines.append("")
    for name, secs in thresholds.items():
        for n, l10P, l10T in ext:
            if l10T > math.log10(secs):
                lines.append(f"first n with predicted time > {name:16s}: n={n}")
                break
    txt = "\n".join(lines)
    print(txt)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "part1-fit-results.txt"), "w") as f:
        f.write(txt + "\n")

    # data for the Excel template (aggregated)
    with open(os.path.join(a.out_dir, "plot-part1-extrapolation.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["n", "log10_permutations", "log10_time_seconds", "time_human"])
        for n, l10P, l10T in ext:
            wr.writerow([n, f"{l10P:.4f}", f"{l10T:.4f}", fmt_time(10 ** l10T) if l10T < 300 else f"1e{l10T:.1f} s"])

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available: skipping PDF plot"); return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    marks = {1: "o", 2: "s", 3: "^", 4: "D", 5: "v"}
    for n in sorted({r["n"] for r in rows}):
        xs = [r["P"] for r in rows if r["n"] == n]; ys = [r["T"] for r in rows if r["n"] == n]
        ax[0].loglog(xs, ys, marks.get(n, "o"), label=f"measured n={n}", alpha=.8)
    xx = np.logspace(0, 10.5, 100)
    ax[0].loglog(xx, t0 + b * xx, "k--", label=f"fit: T = t0 + P/R\nR={R:.2e}/s, R²(log)={r2_log:.4f}")
    ax[0].set_xlabel("permutations enumerated P"); ax[0].set_ylabel("runtime [s]")
    ax[0].set_title(f"Part 1 measured runtime ({th} threads)"); ax[0].legend(fontsize=7); ax[0].grid(True, which="both", alpha=.3)
    ns = [e[0] for e in ext]; lt = [e[2] for e in ext]
    ax[1].plot(ns, lt, "o-", label="predicted T(n) = (n²)!/R")
    for name, secs in thresholds.items():
        ax[1].axhline(math.log10(secs), ls=":", color="gray"); ax[1].text(1.05, math.log10(secs) + 2, name, fontsize=7)
    ax[1].set_xlabel("n"); ax[1].set_ylabel("log10(time [s])"); ax[1].set_title("Extrapolated brute-force runtime")
    ax[1].legend(fontsize=7); ax[1].grid(True, alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "plot-part1.pdf"))
    print("wrote plot-part1.pdf")


if __name__ == "__main__":
    main()
