"""M2 figures: the zero-orbit-width (ZOW) limit of the orbit-averaged torque (forb) against the surface path (fgar).

Reads results/m2_zow/{zow_scan,profiles}.csv (written by m2_checks.jl) and writes results/m2_zow/fig_m2_zow.{svg,png}
(relative difference in total torque against orbit width, from a charge scan) and fig_m2_profile.{svg,png}
(torque density on the fixed ψ nodes, thermal deuterium and the near-ZOW charge).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "results" / "m2_zow"
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "axes.linewidth": 0.6})


def read(name):
    d = np.genfromtxt(OUT / name, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return {k: d[k] for k in d.dtype.names}


def fig_zow():
    s = read("zow_scan.csv")
    w, rel, rre, zi = s["width_over_a"], s["rel"], s["rel_re"], s["zi"]
    fit = zi >= 16
    p, c = np.polyfit(np.log(w[fit]), np.log(rel[fit]), 1)
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    ax.loglog(w, rel, "o-", color="k", ms=3.5, lw=0.8, label="|T_forb / T_fgar − 1| (complex)")
    ax.loglog(w, rre, "s--", color="0.55", ms=3, lw=0.7, label="real part only")
    ww = np.array([w[fit].min() / 2, w.max()])
    ax.loglog(ww, np.exp(c) * ww**p, ":", color="0.3", lw=0.7, label=f"fit, slope {p:.2f}")
    for x, y, z in zip(w, rel, zi):
        ax.annotate(f"Z={z}", (x, y), textcoords="offset points", xytext=(4, -9), fontsize=6, color="0.35")
    ax.set_xlabel("orbit width Δ_b / a  (∝ 1/Z)")
    ax.set_ylabel("relative difference in total torque")
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("svg", "png", "pdf"):
        fig.savefig(OUT / f"fig_m2_zow.{ext}", dpi=150)
    return p


def fig_profile():
    d = read("profiles.csv")
    psi = d["psi"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7), sharex=True)
    a1.plot(psi, d["fgar_z1_re"], "-", color="0.55", lw=1.0, label="fgar (surface)")
    a1.plot(psi, d["forb_z1_re"], "-", color="k", lw=1.0, label="forb (orbit)")
    a1.set_title("thermal D (Z = 1): Re dT/dψ_N", fontsize=8.5)
    a1.set_ylabel("N·m per unit ψ_N")
    a1.legend(fontsize=6.5, frameon=False)
    top = 20.0
    a1.set_ylim(-1, top)
    j = np.argmax(d["fgar_z1_re"])
    if d["fgar_z1_re"][j] > top:
        a1.annotate(f"fgar peak {d['fgar_z1_re'][j]:.0f}", (psi[j], top), xytext=(-60, -12), textcoords="offset points", fontsize=6.5, color="0.4")
    a2.plot(psi, d["fgar_z1024_re"], "-", color="0.55", lw=1.0)
    a2.plot(psi, d["forb_z1024_re"], "--", color="k", lw=1.0)
    a2.set_title("Z = 1024 (width / 1024): the two coincide", fontsize=8.5)
    a2.set_ylabel("N·m per unit ψ_N")
    for a in (a1, a2):
        a.set_xlim(0.8, 1.0)
        a.set_xlabel("ψ_N")
        a.axhline(0, color="0.8", lw=0.5)
    fig.tight_layout()
    for ext in ("svg", "png", "pdf"):
        fig.savefig(OUT / f"fig_m2_profile.{ext}", dpi=150)


if __name__ == "__main__":
    print(f"slope {fig_zow():.3f}")
    fig_profile()
