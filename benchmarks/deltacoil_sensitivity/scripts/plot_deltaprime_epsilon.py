#!/usr/bin/env python3
# plot_deltaprime_epsilon.py -- Phase C benchmark for the single-surface TJ study: GPEC's tearing
# Delta-prime versus inverse aspect ratio epsilon = a/R0, at fixed cylindrical q-profile, for each
# single-rational-surface case. As epsilon -> 0 the toroidal Delta-prime must converge to the cylindrical
# (large-aspect) value; the dotted lines mark the analytic vacuum limit r_s*Delta' = -2m from the
# independent Newcomb solver (exact when the current gradient at the surface is negligible).
#
# Reads scratch eps_<lab>_<eps>/gpec_eps_<lab>_<eps>.h5 written by the eps-scan runner.
# Usage: python3 plot_deltaprime_epsilon.py <scratch_task_single_dir> <out_figdir>

import sys, os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SC, FIGDIR = sys.argv[1], sys.argv[2]
os.makedirs(FIGDIR, exist_ok=True)
LABS = ["q1", "q2", "q3", "q4", "q5"]
MOF = {"q1":1,"q2":2,"q3":3,"q4":4,"q5":5}
EPS = [0.2, 0.1, 0.05]

def read_dp(h5):
    with h5py.File(h5, "r") as f:
        if "galerkin/pest3_Delta" not in f:
            return None, None
        dp = np.atleast_2d(f["galerkin/pest3_Delta"][()]).ravel()
        q = np.atleast_1d(f["singular/q"][()])
        return float(np.real(dp[0])), float(q[0]) if len(q) else None

data = {}   # lab -> list of (eps, Dprime)
print("=== GPEC tearing Delta-prime vs aspect ratio epsilon (fixed q-profile) ===")
print(f"  {'case':5s} {'m':>2s} " + "".join(f"eps={e:<7}" for e in EPS) + "  vacuum(-2m)  converged?")
for lab in LABS:
    m = MOF[lab]; row = []
    for e in EPS:
        h5 = os.path.join(SC, f"eps_{lab}_{e}", f"gpec_eps_{lab}_{e}.h5")
        if os.path.exists(h5):
            dp, q = read_dp(h5)
            row.append((e, dp))
        else:
            row.append((e, None))
    data[lab] = row
    vals = [d for _, d in row if d is not None]
    conv = (abs(vals[-1] - vals[-2]) if len(vals) >= 2 else float("nan"))
    cells = "".join(f"{(d if d is not None else float('nan')):<+11.3f}" for _, d in row)
    print(f"  {lab:5s} {m:>2d} {cells}  {-2*m:<+11d}  d(last2)={conv:.3f}")

# ---- plot ----
colors = plt.cm.viridis(np.linspace(0.08, 0.82, len(LABS)))
fig, ax = plt.subplots(figsize=(9.0, 5.6))
for k, lab in enumerate(LABS):
    m = MOF[lab]
    es = [e for e, d in data[lab] if d is not None]
    ds = [d for e, d in data[lab] if d is not None]
    if not ds:
        continue
    ax.plot(es, ds, "o-", color=colors[k], lw=1.9, ms=6, label=f"{lab}: q={m}/1")
    ax.axhline(-2*m, ls=":", color=colors[k], lw=1.1, alpha=0.7)
    ax.text(0.205, -2*m, f" -2m={-2*m}", color=colors[k], fontsize=7.5, va="center", ha="left")
ax.set_xscale("log")
ax.invert_xaxis()   # epsilon decreasing to the right = more cylindrical
ax.set_xlabel(r"inverse aspect ratio  $\epsilon = a/R_0$   (decreasing to the right = more cylindrical)", fontsize=11)
ax.set_ylabel(r"tearing $\Delta'$  (PEST3, $= r_s\Delta'$)", fontsize=11)
ax.set_title("Single-surface TJ study: GPEC tearing $\\Delta'$ vs aspect ratio\n"
             "dotted = analytic Newcomb vacuum limit $r_s\\Delta'=-2m$ (exact); "
             "convergence as $\\epsilon\\to0$ validates the cylindrical reduction", fontsize=10.5)
ax.grid(alpha=0.25, which="both")
ax.legend(fontsize=9, title="rational surface", title_fontsize=9, loc="best")
fig.tight_layout()
out = os.path.join(FIGDIR, "single_surface_deltaprime_epsilon.png")
fig.savefig(out, dpi=150)
print("wrote:", out)
