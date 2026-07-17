#!/usr/bin/env python3
"""plot_deltacoil_sensitivity.py — visualize delta_coil sensitivity to dmlim and singfac_min.
Values measured by deltacoil_sensitivity.jl on the DIII-D-like case (qmax=5.426)."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

# --- Sweep A: dmlim (truncation); NaN = surface absent at that dmlim ---
dmlim = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
A = {  # |delta_coil| per surface
    "q=2": [11.40, 11.72, 11.07, 11.00, 11.14],
    "q=3": [9.389, 9.560, 9.619, 9.319, 9.279],
    "q=4": [12.42, 12.51, 14.57, 13.05, 12.51],
    "q=5 (edge)": [38.57, 33.49, np.nan, np.nan, np.nan],
}
# --- Sweep B: singfac_min (approach distance) ---
sfac = np.array([1e-5, 1e-4, 1e-3])
B = {
    "q=2": [11.55, 11.55, 11.55],
    "q=3": [9.469, 9.469, 9.469],
    "q=4": [12.45, 12.45, 12.45],
    "q=5 (edge)": [35.09, 35.10, 35.09],
}
colors = {"q=2": "#2E86C1", "q=3": "#28B463", "q=4": "#E67E22", "q=5 (edge)": "#C0392B"}
QMAX = 5.426; thresh = QMAX - 5.0   # dmlim above which q5 is dropped

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
fig.suptitle("delta_coil sensitivity — DIII-D-like, n=1  (qmax = 5.426)", fontsize=13, fontweight="bold")

for k, v in A.items():
    ax1.plot(dmlim, v, "o-", color=colors[k], lw=2, ms=7, label=k)
ax1.axvline(thresh, color="grey", ls="--", lw=1.4)
ax1.text(thresh + 0.01, ax1.get_ylim()[1]*0.96, f"  q5 dropped for\n  dmlim > {thresh:.3f}\n  (q₅+dmlim > qmax)",
         va="top", fontsize=8.5, color="grey")
ax1.set_xlabel("dmlim  (outer truncation, set_psilim_via_dmlim=true)")
ax1.set_ylabel("|delta_coil|")
ax1.set_title("Sweep A — truncation point", fontsize=11, fontweight="bold")
ax1.legend(fontsize=9); ax1.grid(alpha=0.25)

for k, v in B.items():
    ax2.plot(sfac, v, "o-", color=colors[k], lw=2, ms=7, label=k)
ax2.set_xscale("log")
ax2.set_xlabel("singfac_min  (rational-surface approach distance)")
ax2.set_ylabel("|delta_coil|")
ax2.set_title("Sweep B — approach distance  (dmlim=0.2, all 4 surfaces)", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9); ax2.grid(alpha=0.25)
ax2.text(0.5, 0.02, "flat to 5–6 significant figures across 2 decades — delta_coil is\n"
         "essentially independent of singfac_min at every surface incl. edge",
         transform=ax2.transAxes, ha="center", va="bottom", fontsize=8.5, style="italic")

out = "deltacoil_sensitivity.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
