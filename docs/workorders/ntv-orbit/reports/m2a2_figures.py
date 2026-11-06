"""M2 attempt-2 figures. fig_ures: orbit quantities against energy at fixed pitch, fine scan against the 12 table nodes.

Reads results/m2a2_dev/ures.csv (m2a2_ures.jl); writes results/m2a2_dev/fig_m2a2_ures.{svg,png,pdf}.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[5]
DEV = ROOT / "results" / "m2a2_dev"
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "axes.linewidth": 0.6})
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8984"
CLASS = {1: ("trapped", BLUE), 2: ("passing", ORANGE)}


def save(fig, name):
    for ext in ("svg", "png", "pdf"):
        fig.savefig(DEV / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def rows(R, psi, lam, sig, grid):
    sel = [r for r in R if r["grid"] == grid and abs(float(r["psi"]) - psi) < 1e-9 and abs(float(r["lambda"]) - lam) < 1e-4
           and int(float(r["sigma"])) == sig]
    return sorted(sel, key=lambda r: float(r["u"]))


def series(rs, q):
    st = np.array([int(float(r["status"])) for r in rs])
    u = np.array([float(r["u"]) for r in rs])
    if q == "dj":
        v = np.array([sum(float(r[f"dj{l}"]) for l in range(-6, 7)) for r in rs])
    else:
        v = np.array([float(r[q]) / float(r["u"]) for r in rs])
    return u, v, st


def fig_ures():
    R = list(csv.DictReader(open(DEV / "ures.csv")))
    cases = [(0.70, 1.05, 1), (0.70, 0.75, -1), (0.70, 0.45, 1), (0.96, 0.6551, -1)]
    fig, ax = plt.subplots(2, len(cases), figsize=(8.6, 4.4), sharex=True)
    for j, (psi, lam, sig) in enumerate(cases):
        for i, (q, lab) in enumerate((("wb", "ω_b/√x (rad/s)"), ("dj", "Σ_ℓ|δJ_ℓ|² (table units)"))):
            a = ax[i, j]
            uf, vf, sf = series(rows(R, psi, lam, sig, "fine"), q)
            un, vn, sn = series(rows(R, psi, lam, sig, "node"), q)
            for c, (name, col) in CLASS.items():
                m, mn = sf == c, sn == c
                if m.sum() < 2:
                    continue
                a.plot(uf[m], vf[m], "-", color=col, lw=1.1, label=f"{name}, 150-point scan")
                if mn.sum() >= 4:
                    us = np.linspace(un[mn].min(), un[mn].max(), 300)
                    a.plot(us, CubicSpline(un[mn], vn[mn])(us), ":", color=INK, lw=0.9, label="spline of table nodes")
                a.plot(un[mn], vn[mn], "o", color=INK, ms=2.8)
            for e in uf[1:][sf[1:] != sf[:-1]]:
                a.axvline(e, color=MUTED, lw=0.6, ls="--")
            if i == 0:
                a.set_title(f"ψ = {psi}, λ = {lam}, σ = {sig:+d}", fontsize=7.5)
            if j == 0:
                a.set_ylabel(lab)
            if i == 1:
                a.set_xlabel("u = √x")
    h, l = ax[1, 0].get_legend_handles_labels()
    seen = dict(zip(l, h))
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig_m2a2_ures")


if __name__ == "__main__":
    fig_ures()
