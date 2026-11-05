"""M1 figures: guiding-center orbits on the DIII-D-like example and their zero-orbit-width (ZOW) limit.

Reads results/m1_orbits/{orbits_rz,surfaces_rz,zow,lar}.csv (written by m1_checks.jl) and writes
results/m1_orbits/fig_m1_orbits.{svg,png} (orbits in R, Z at three energies) and fig_m1_zow.{svg,png}
(relative frequency error versus energy against BounceAveraging, and the large-aspect-ratio pendulum check).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "results" / "m1_orbits"
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "axes.linewidth": 0.6})


def fig_orbits():
    rz, sf = pd.read_csv(OUT / "orbits_rz.csv"), pd.read_csv(OUT / "surfaces_rz.csv")
    xs = sorted(rz.x.unique())
    fig, axes = plt.subplots(1, len(xs), figsize=(7.2, 3.4), sharey=True)
    styles = {(0.95, 1): ("k", "-", "trapped"), (0.3, 1): ("0.45", "--", "passing, σ = +1 (counter-current)"),
              (0.3, -1): ("0.45", ":", "passing, σ = −1 (co-current)")}
    for ax, x in zip(axes, xs):
        for psi, d in sf.groupby("psi"):
            ax.plot(d.R, d.Z, color="0.75", lw=0.6 if psi < 0.99 else 1.0)
        for (lam, sig), (c, ls, lab) in styles.items():
            d = rz[(rz.x == x) & (rz["lambda"] == lam) & (rz.sigma == sig)]
            if len(d):
                ax.plot(d.R, d.Z, color=c, ls=ls, lw=0.9, label=f"{lab}" + ("" if d.status.iloc[0] in lab else f" [{d.status.iloc[0]}]"))
        ax.set_aspect("equal")
        ax.set_title(f"x = E/T = {x:g}  ({x * 2.14:.0f} keV D)", fontsize=8.5)
        ax.set_xlabel("R (m)")
    axes[0].set_ylabel("Z (m)")
    axes[0].legend(loc="lower left", fontsize=6, frameon=False)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"fig_m1_orbits.{ext}", dpi=140)


def fig_zow():
    z, lar = pd.read_csv(OUT / "zow.csv"), pd.read_csv(OUT / "lar.csv")
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.2, 2.7))
    d = z[(z.psi0 == 0.5)]
    for (lam, cs), g in d.groupby(["lambda", "code_sigma"]):
        g = g[g.status == ("trapped" if cs == 0 else "passing")]
        p, m = g[g.sigma == 1].set_index("x"), g[g.sigma == -1].set_index("x")
        common = p.index.intersection(m.index)
        if len(common) < 3:
            continue
        ls = "-" if cs == 0 else "--"
        a1.loglog(common, np.abs(p.loc[common].wb_rel_err), ls, color="k", lw=0.8)
        a1.loglog(common, np.abs((p.loc[common].wb_rel_err + m.loc[common].wb_rel_err) / 2), ls, color="0.6", lw=0.8)
        if cs == 0:
            a2.loglog(common, np.abs(p.loc[common].prec_over_wd_x + 1), "-", color="k", lw=0.8)
            a2.loglog(common, np.abs((p.loc[common].prec_over_wd_x + m.loc[common].prec_over_wd_x) / 2 + 1), "-", color="0.6", lw=0.8)
    xx = np.array([1e-4, 1])
    for a in (a1, a2):
        a.loglog(xx, 1e-3 * np.sqrt(xx / 1e-2), ":", color="0.3", lw=0.6)
        a.set_xlabel("x = E/T")
    a1.set_title("ω_b vs BounceAveraging", fontsize=8.5)
    a1.set_ylabel("|relative error|")
    a2.set_title("trapped ⟨φ̇⟩ vs s·ω_D·x", fontsize=8.5)
    a1.text(2e-4, 3e-1, "black: σ = +1\ngrey: σ-average\nsolid trapped, dashed passing\ndotted: ∝ x^{1/2}", fontsize=6, va="top")
    for psi, g in lar.groupby("psi0"):
        a3.plot(g.kappa2, g.wb / g.wb_pendulum - 1, "o-", ms=2.5, lw=0.7, label=f"ψ = {psi:g}")
    a3.axvline(1, color="0.6", lw=0.6, ls=":")
    a3.set_xlabel("trapping parameter κ²")
    a3.set_title("LAR: ω_b vs pendulum (Antonenas 4.6–4.7)", fontsize=8.5)
    a3.set_ylabel("relative difference")
    a3.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"fig_m1_zow.{ext}", dpi=140)


if __name__ == "__main__":
    fig_orbits()
    fig_zow()
