#!/usr/bin/env python3
# plot_marginal_q2.py -- Phase D marginal-stability scan for the single q=2 TJ surface. Varies the
# edge safety factor qa (fixed qc=1.5, so the current-peaking nu=qa/qc changes) while keeping exactly
# one rational surface (q=2). Tracks the tearing Delta-prime; the classic single-surface tearing test
# is Delta-prime passing through zero (marginal stability) as the current gradient at the surface changes.
# Reads scratch marg_q2_qa<...>/gpec_marg_q2_qa<...>.h5.
# Usage: python3 plot_marginal_q2.py <scratch_task_single_dir> <out_figdir>

import sys, os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SC, FIGDIR = sys.argv[1], sys.argv[2]
os.makedirs(FIGDIR, exist_ok=True)
QC = 1.5
QAS = [2.1, 2.2, 2.3, 2.4, 2.5, 2.7, 2.9]

rows = []
for qa in QAS:
    tag = f"qa{qa}".replace(".", "p")
    h5 = os.path.join(SC, f"marg_q2_{tag}", f"gpec_marg_q2_{tag}.h5")
    if not os.path.exists(h5):
        continue
    with h5py.File(h5, "r") as f:
        if "galerkin/pest3_Delta" not in f:
            continue
        dp = float(np.real(np.atleast_2d(f["galerkin/pest3_Delta"][()]).ravel()[0]))
        q = np.atleast_1d(f["singular/q"][()])
        rows.append((qa, qa / QC, dp, len(q)))

print("=== q2 marginal-stability scan (qc=1.5 fixed) ===")
print(f"  {'qa':>5s} {'nu=qa/qc':>9s} {'msing':>5s} {'Delta_prime':>12s}")
for qa, nu, dp, ms in rows:
    print(f"  {qa:>5.2f} {nu:>9.3f} {ms:>5d} {dp:>+12.3f}")

qas = np.array([r[0] for r in rows]); dps = np.array([r[2] for r in rows])
# locate zero crossing by linear interpolation between sign changes
qa_marg = None
for i in range(len(qas) - 1):
    if dps[i] == 0 or (dps[i] < 0) != (dps[i+1] < 0):
        qa_marg = qas[i] + (0 - dps[i]) * (qas[i+1] - qas[i]) / (dps[i+1] - dps[i])
        break
print(f"  marginal (Delta_prime=0) at qa ~ {qa_marg:.3f} (nu ~ {qa_marg/QC:.3f})" if qa_marg else "  no sign change in scanned range")

fig, ax = plt.subplots(figsize=(8.2, 5.2))
ax.plot(qas, dps, "o-", color="#1f5fbf", lw=2, ms=7)
ax.axhline(0, color="k", lw=0.9, ls="--")
if qa_marg:
    ax.axvline(qa_marg, color="#c0392b", lw=1.2, ls=":")
    ax.plot([qa_marg], [0], "s", color="#c0392b", ms=9,
            label=f"marginal: qa~{qa_marg:.3f} (nu~{qa_marg/QC:.3f})")
    ax.legend(fontsize=9, loc="best")
ax.text(0.02, 0.96, "unstable (Delta' > 0)", transform=ax.transAxes, fontsize=9, color="#a33", va="top")
ax.text(0.02, 0.05, "stable (Delta' < 0)", transform=ax.transAxes, fontsize=9, color="#264", va="bottom")
ax.set_xlabel("edge safety factor  qa   (current peaking nu = qa/qc, qc=1.5 fixed)", fontsize=11)
ax.set_ylabel("tearing $\\Delta'$  (PEST3)", fontsize=11)
ax.set_title("Single q=2 TJ surface: marginal-stability scan\n$\\Delta'$ vs current peaking; zero crossing = tearing marginal point", fontsize=11)
ax.grid(alpha=0.25)
fig.tight_layout()
out = os.path.join(FIGDIR, "single_surface_q2_marginal.png")
fig.savefig(out, dpi=150)
print("wrote:", out)
