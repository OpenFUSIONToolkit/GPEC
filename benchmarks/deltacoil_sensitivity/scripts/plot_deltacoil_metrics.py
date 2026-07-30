#!/usr/bin/env python3
"""plot_deltacoil_metrics.py - cosine similarity of delta_coil with the nominal case, per surface.
The dot-product metric that the norm hides: shows the response PATTERN rotating under truncation."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

dmlim = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
cosA = {  # cosine similarity with nominal (set_psilim_via_dmlim=false baseline)
    "q=2": [0.975948, 0.996348, 0.802348, 0.882046, 0.938423],
    "q=3": [0.950813, 0.992477, 0.602900, 0.762331, 0.875203],
    "q=4": [0.879623, 0.981228, 0.174318, 0.464867, 0.705142],
    "q=5 (edge)": [0.593958, 0.942364, np.nan, np.nan, np.nan],
}
sfac = np.array([1e-5, 1e-4, 1e-3])
cosB = {  # identical across singfac (delta_coil independent of singfac_min)
    "q=2": [0.988340]*3, "q=3": [0.976060]*3, "q=4": [0.940802]*3, "q=5 (edge)": [0.818224]*3,
}
colors = {"q=2": "#2E86C1", "q=3": "#28B463", "q=4": "#E67E22", "q=5 (edge)": "#C0392B"}
QMAX = 5.426; thresh = QMAX - 5.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
fig.suptitle("delta_coil: normalized dot product with nominal (cosine similarity), per surface\n"
             "1.0 = same response pattern;  <1 = pattern ROTATED (invisible to the norm)",
             fontsize=12.5, fontweight="bold")

for k, v in cosA.items():
    ax1.plot(dmlim, v, "o-", color=colors[k], lw=2, ms=7, label=k)
ax1.axvline(thresh, color="grey", ls="--", lw=1.3)
ax1.text(thresh + 0.012, 0.30, f"q5 dropped\n(dmlim > {thresh:.3f})", fontsize=8.5, color="grey")
ax1.set_xlabel("dmlim  (outer truncation)"); ax1.set_ylabel("cosine similarity with nominal")
ax1.set_title("Sweep A - truncation: pattern rotates hard once q5 drops", fontsize=10.5, fontweight="bold")
ax1.set_ylim(0, 1.03); ax1.legend(fontsize=9, loc="lower right"); ax1.grid(alpha=0.25)

for k, v in cosB.items():
    ax2.plot(sfac, v, "o-", color=colors[k], lw=2, ms=7, label=k)
ax2.set_xscale("log"); ax2.set_ylim(0, 1.03)
ax2.set_xlabel("singfac_min  (rational-surface approach distance)")
ax2.set_ylabel("cosine similarity with nominal")
ax2.set_title("Sweep B - approach distance: perfectly flat (no effect)", fontsize=10.5, fontweight="bold")
ax2.legend(fontsize=9, loc="lower left"); ax2.grid(alpha=0.25)
ax2.text(0.5, 0.55, "cosine identical to 6 decimals across 2 decades of singfac_min\n"
         "(offset from 1.0 is the truncation-flag on/off, not singfac)",
         transform=ax2.transAxes, ha="center", fontsize=8.5, style="italic")

out = "deltacoil_metrics.png"; fig.savefig(out, dpi=140, bbox_inches="tight"); print(f"wrote {out}")
