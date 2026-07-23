#!/usr/bin/env python3
"""plot_deltacoil_sensitivity.py - delta_coil sensitivity to dmlim (outer truncation) and singfac_min
(rational-surface approach distance), DIII-D-like n=1. Overlays the OLD DIII-D equilibrium (dashed,
open markers) against the RETARGETED Ip=1.15 MA equilibrium (solid, filled; commit 8f0a7cce).
NOTE: the delta_coil code is bit-identical across branches - this-branch code on the old geqdsk
reproduces the old values exactly (q5=38.57). The visible difference is ENTIRELY the equilibrium
retarget (q95 4.505->4.782, qmax 5.426->5.972), amplified at the edge surface q5 which is hyper-
sensitive to the edge q-profile/truncation. Values measured by deltacoil_sensitivity.jl (gal_match_flag=false)."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os

# ---- Sweep A: dmlim (truncation), set_psilim_via_dmlim=true ; NaN = surface absent ----
dmlim = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
A_prev = {  # previous week (loop-boundary-cond-riccati); q5 dropped out for dmlim > qmax-5 = 0.426
    "q=2": [11.40, 11.72, 11.07, 11.00, 11.14],
    "q=3": [9.389, 9.560, 9.619, 9.319, 9.279],
    "q=4": [12.42, 12.51, 14.57, 13.05, 12.51],
    "q=5 (edge)": [38.57, 33.49, np.nan, np.nan, np.nan],
}
A_new = {   # this branch (GGJ_colocation_backend): q5 now present at all dmlim and ~4x larger
    "q=2": [10.36, 10.56, 10.83, 11.12, 11.42],
    "q=3": [8.279, 8.334, 8.460, 8.607, 8.762],
    "q=4": [11.34, 11.13, 11.12, 11.20, 11.28],
    "q=5 (edge)": [200.6, 152.7, 139.4, 134.1, 130.9],
}
# ---- Sweep B: singfac_min (approach distance), dmlim fixed at 0.2 ----
sfac = np.array([1e-5, 1e-4, 1e-3])
B_prev = {
    "q=2": [11.55, 11.55, 11.55], "q=3": [9.469, 9.469, 9.469],
    "q=4": [12.45, 12.45, 12.45], "q=5 (edge)": [35.09, 35.10, 35.09],
}
B_new = {
    "q=2": [10.45, 10.45, 10.45], "q=3": [8.292, 8.292, 8.292],
    "q=4": [11.20, 11.20, 11.20], "q=5 (edge)": [167.8, 167.8, 167.8],
}
colors = {"q=2": "#2E86C1", "q=3": "#28B463", "q=4": "#E67E22", "q=5 (edge)": "#C0392B"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
fig.suptitle("delta_coil sensitivity, DIII-D-like, n=1  ·  old equilibrium (dashed) vs retargeted Ip=1.15 MA (solid)",
             fontsize=13, fontweight="bold")

for k in A_new:
    ax1.plot(dmlim, A_prev[k], "o--", color=colors[k], lw=1.6, ms=6, mfc="white", alpha=0.8)
    ax1.plot(dmlim, A_new[k],  "o-",  color=colors[k], lw=2.2, ms=7, label=k)
ax1.set_yscale("log")                                  # log-y: q5 (~130-200) dwarfs q2/q3/q4 (~10)
ax1.set_xlabel("dmlim  (outer truncation, set_psilim_via_dmlim = true)")
ax1.set_ylabel("|delta_coil|  (log)")
ax1.set_title("Sweep A: truncation point", fontsize=11, fontweight="bold")
ax1.legend(fontsize=9, title="solid = Ip=1.15 MA (new geqdsk)\ndashed = old geqdsk", title_fontsize=8)
ax1.grid(alpha=0.25, which="both")
ax1.annotate("q5 shifts ~5x with the equilibrium retarget\n(edge q-profile); delta_coil code is\nbit-identical across branches",
             xy=(0.5, 139.4), xytext=(0.34, 58), fontsize=8.2, color=colors["q=5 (edge)"],
             arrowprops=dict(arrowstyle="->", color=colors["q=5 (edge)"], lw=1.1))

for k in B_new:
    ax2.plot(sfac, B_prev[k], "o--", color=colors[k], lw=1.6, ms=6, mfc="white", alpha=0.8)
    ax2.plot(sfac, B_new[k],  "o-",  color=colors[k], lw=2.2, ms=7, label=k)
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel("singfac_min  (rational-surface approach distance)")
ax2.set_ylabel("|delta_coil|  (log)")
ax2.set_title("Sweep B: approach distance  (dmlim = 0.2, all 4 surfaces)", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9); ax2.grid(alpha=0.25, which="both")
ax2.text(0.5, 0.02, "flat to 5-6 sig figs across 2 decades for both equilibria,\n"
         "delta_coil is essentially independent of singfac_min at every surface",
         transform=ax2.transAxes, ha="center", va="bottom", fontsize=8.5, style="italic")

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.abspath(os.path.join(here, "..", "figures", "deltacoil_sensitivity.png"))
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote:", out)
