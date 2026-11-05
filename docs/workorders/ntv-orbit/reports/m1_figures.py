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

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "results" / "m1_orbits"


def read(name):
    """CSV with a header row as a dict of numpy arrays (strings left as str)."""
    d = np.genfromtxt(OUT / name, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return {k: d[k] for k in d.dtype.names}


plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "axes.linewidth": 0.6})


def fig_orbits():
    rz, sf = read("orbits_rz.csv"), read("surfaces_rz.csv")
    xs = sorted(set(rz["x"]))
    fig, axes = plt.subplots(1, len(xs), figsize=(7.2, 3.4), sharey=True)
    styles = {(0.95, 1): ("k", "-", "trapped"), (0.3, 1): ("0.45", "--", "passing, σ = +1 (counter-current)"),
              (0.3, -1): ("0.45", ":", "passing, σ = −1 (co-current)")}
    for ax, x in zip(axes, xs):
        for psi in sorted(set(sf["psi"])):
            m = sf["psi"] == psi
            ax.plot(sf["R"][m], sf["Z"][m], color="0.75", lw=0.6 if psi < 0.99 else 1.0)
        for (lam, sig), (c, ls, lab) in styles.items():
            m = (rz["x"] == x) & (rz["lambda"] == lam) & (rz["sigma"] == sig)
            if m.any():
                st = rz["status"][m][0]
                ax.plot(rz["R"][m], rz["Z"][m], color=c, ls=ls, lw=0.9, label=lab + ("" if st in lab else f" [{st}]"))
        ax.set_aspect("equal")
        ax.set_title(f"x = E/T = {x:g}  ({x * 2.14:.0f} keV D)", fontsize=8.5)
        ax.set_xlabel("R (m)")
    axes[0].set_ylabel("Z (m)")
    axes[0].legend(loc="lower left", fontsize=6, frameon=False)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"fig_m1_orbits.{ext}", dpi=140)


def fig_zow():
    z, lar = read("zow.csv"), read("lar.csv")
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for lam in sorted(set(z["lambda"][z["psi0"] == 0.5])):
        base = (z["psi0"] == 0.5) & (z["lambda"] == lam)
        cs = z["code_sigma"][base][0]
        want = "trapped" if cs == 0 else "passing"
        rows = {}
        for sv in (1, -1):
            m = base & (z["sigma"] == sv) & (z["status"] == want)
            rows[sv] = dict(zip(z["x"][m], zip(z["wb_rel_err"][m], z["prec_over_wd_x"][m])))
        common = sorted(set(rows[1]) & set(rows[-1]))
        if len(common) < 3:
            continue
        x = np.array(common)
        wp, wm = np.array([rows[1][c][0] for c in common]), np.array([rows[-1][c][0] for c in common])
        pp, pm = np.array([rows[1][c][1] for c in common]), np.array([rows[-1][c][1] for c in common])
        ls = "-" if cs == 0 else "--"
        a1.loglog(x, np.abs(wp), ls, color="k", lw=0.8)
        a1.loglog(x, np.abs((wp + wm) / 2), ls, color="0.6", lw=0.8)
        if cs == 0:
            a2.loglog(x, np.abs(pp + 1), "-", color="k", lw=0.8)
            a2.loglog(x, np.abs((pp + pm) / 2 + 1), "-", color="0.6", lw=0.8)
    xx = np.array([1e-4, 1])
    for a in (a1, a2):
        a.loglog(xx, 1e-3 * np.sqrt(xx / 1e-2), ":", color="0.3", lw=0.6)
        a.set_xlabel("x = E/T")
    a1.set_title("ω_b vs BounceAveraging", fontsize=8.5)
    a1.set_ylabel("|relative error|")
    a2.set_title("trapped ⟨φ̇⟩ vs s·ω_D·x", fontsize=8.5)
    a1.text(0.97, 0.03, "black σ = +1, grey σ-average\nsolid trapped, dashed passing\ndotted ∝ x$^{1/2}$", transform=a1.transAxes,
            fontsize=6, ha="right", va="bottom")
    for psi, (c, mk) in zip(sorted(set(lar["psi0"])), (("k", "o"), ("0.45", "s"), ("0.7", "^"))):
        m = lar["psi0"] == psi
        a3.plot(lar["kappa2"][m], lar["wb"][m] / lar["wb_pendulum"][m] - 1, "-", marker=mk, color=c, ms=2.5, lw=0.7, label=f"ψ = {psi:g}")
    a3.axvline(1, color="0.6", lw=0.6, ls=":")
    a3.set_xlabel("trapping parameter κ²")
    a3.set_title("LAR ω_b vs pendulum form", fontsize=8.5)
    a3.set_ylabel("relative difference")
    a3.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"fig_m1_zow.{ext}", dpi=140)


if __name__ == "__main__":
    fig_orbits()
    fig_zow()
