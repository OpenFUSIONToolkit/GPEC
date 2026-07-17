#!/usr/bin/env python3
"""plot_deltacoil_dmlim_fine.py — fine dmlim sweep (0.30-0.50) through the q5-drop threshold (0.426).
Shows the delta_coil pattern collapse is a STEP at the threshold, not a smooth rotation.
Values from deltacoil_metrics.jl (SKIP_SINGFAC=1) on the DIII-D-like case, qmax=5.426."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

QMAX = 5.426; thr = QMAX - 5.0  # 0.426
lo = np.array([0.30, 0.35, 0.40, 0.42])          # q5 present
hi = np.array([0.44, 0.46, 0.48, 0.50])          # q5 dropped
cos_lo = {  # cosine vs nominal, below threshold
    "q=2": [0.996348, 0.998666, 0.999843, 0.999991],
    "q=3": [0.992477, 0.997249, 0.999675, 0.999982],
    "q=4": [0.981228, 0.993110, 0.999184, 0.999955],
    "q=5 (edge)": [0.942364, 0.978476, 0.997414, 0.999857],
}
cos_hi = {  # above threshold (q5 absent)
    "q=2": [0.775613, 0.784651, 0.793553, 0.802348],
    "q=3": [0.551682, 0.568847, 0.585903, 0.602900],
    "q=4": [0.104129, 0.125729, 0.149245, 0.174318],
}
colors = {"q=2": "#2E86C1", "q=3": "#28B463", "q=4": "#E67E22", "q=5 (edge)": "#C0392B"}

fig, ax = plt.subplots(figsize=(9.2, 6.0), constrained_layout=True)
for k in cos_lo:
    ax.plot(lo, cos_lo[k], "o-", color=colors[k], lw=2, ms=7, label=k)
for k in cos_hi:
    ax.plot(hi, cos_hi[k], "s--", color=colors[k], lw=2, ms=7)  # dashed above threshold
ax.axvline(thr, color="grey", ls=":", lw=1.6)
ax.axvspan(thr, 0.51, color="grey", alpha=0.07)
ax.text(thr + 0.004, 0.05, f"q5 dropped\n(dmlim > {thr:.3f})", fontsize=9, color="dimgrey")
ax.text(0.315, 0.05, "q5 retained\n(4 surfaces)", fontsize=9, color="dimgrey")
ax.annotate("q4 pattern nearly orthogonal\nto nominal (cos ≈ 0.10)",
            xy=(0.44, 0.104), xytext=(0.455, 0.30), fontsize=8.5, color=colors["q=4"],
            arrowprops=dict(arrowstyle="->", color=colors["q=4"], lw=1.2))
ax.set_xlabel("dmlim  (outer truncation, set_psilim_via_dmlim=true)")
ax.set_ylabel("cosine similarity of delta_coil with nominal")
ax.set_title("Fine dmlim sweep through the q5-drop threshold\n"
             "solid = q5 present;  dashed = q5 dropped — the pattern collapses as a STEP at 0.426",
             fontsize=11.5, fontweight="bold")
ax.set_ylim(0, 1.03); ax.set_xlim(0.29, 0.51)
ax.legend(fontsize=9, loc="center right"); ax.grid(alpha=0.25)
out = "benchmarks/deltacoil_sensitivity/figures/deltacoil_dmlim_fine.png"
import os; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=140, bbox_inches="tight"); print(f"wrote {out}")
