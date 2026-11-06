"""M2 attempt-2 figures. fig_m2a2_ures: orbit quantities against energy at fixed pitch, fine scan against the 12 table nodes.

Reads results/m2a2_dev/ures.csv (m2a2_ures.jl); writes results/m2a2_dev/fig_m2a2_ures.{pdf,png,svg}.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline

from ntvstyle import ONEHALF, PASSING, ROOT, TRAPPED, panel, save

DEV = ROOT / "results" / "m2a2_dev"
CLASS = {1: TRAPPED, 2: PASSING}


def rows(R, psi, lam, sig, grid):
    sel = [r for r in R if r["grid"] == grid and abs(float(r["psi"]) - psi) < 1e-9 and abs(float(r["lambda"]) - lam) < 1e-4
           and int(float(r["sigma"])) == sig]
    return sorted(sel, key=lambda r: float(r["u"]))


def series(rs, q):
    st = np.array([int(float(r["status"])) for r in rs])
    u = np.array([float(r["u"]) for r in rs])
    if q == "dj":
        v = np.array([sum(float(r[f"dj{l}"]) for l in range(-6, 7)) for r in rs]) * 1e4
    else:
        v = np.array([float(r[q]) / float(r["u"]) for r in rs]) * 1e-3
    return u, v, st


def fig_ures():
    R = list(csv.DictReader(open(DEV / "ures.csv")))
    cases = [(0.70, 0.75, -1), (0.70, 1.05, 1)]
    fig, ax = plt.subplots(2, 2, figsize=(ONEHALF, 3.9), sharex=True)
    for j, (psi, lam, sig) in enumerate(cases):
        for i, q in enumerate(("wb", "dj")):
            a = ax[i, j]
            uf, vf, sf = series(rows(R, psi, lam, sig, "fine"), q)
            un, vn, sn = series(rows(R, psi, lam, sig, "node"), q)
            for c, col in CLASS.items():
                m, mn = sf == c, sn == c
                if m.sum() < 2:
                    continue
                a.plot(uf[m], vf[m], "-", color=col)
                if mn.sum() >= 4:
                    us = np.linspace(un[mn].min(), un[mn].max(), 300)
                    a.plot(us, CubicSpline(un[mn], vn[mn])(us), ":", color="k", lw=1.2)
                a.plot(un[mn], vn[mn], "o", color="k", ms=3.5)
            for e in uf[1:][sf[1:] != sf[:-1]]:
                a.axvline(e, color="0.6", lw=0.8, ls="--")
            panel(a, "abcd"[2 * i + j])
            if i == 1:
                a.set_xlabel("u = √x")
    ax[0, 0].set_ylabel("ω$_b$/√x (krad/s)")
    ax[1, 0].set_ylabel("Σ|δJ$_ℓ$|² (×10$^{-4}$)")
    handles = [Line2D([], [], color=TRAPPED), Line2D([], [], color=PASSING), Line2D([], [], color="k", ls=":", marker="o", ms=3.5)]
    fig.legend(handles, ["trapped", "passing", "table"], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    save(fig, DEV, "fig_m2a2_ures")


if __name__ == "__main__":
    fig_ures()
