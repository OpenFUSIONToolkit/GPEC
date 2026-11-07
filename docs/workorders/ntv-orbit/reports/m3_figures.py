"""M3 figures, from results/m3_zow2, results/m3_z1phi2, results/m3_arc and (for comparison) the M2 attempt-3 results; each
writes <name>.{pdf,png,svg} beside its data.

fig_m3_zow      check 1 with the M3 drive: forb against the same-grid replay versus orbit width, M3 and M2 attempt 3
fig_m3_profile  Z = 1 DIII-D-like torque density with Φ on: fgar, forb at M2 attempt 3 and at M3
fig_m3_moments  ARC alpha moments versus ψ: density and pressure of the slowing-down model and the gacode fast He
fig_m3_alpha    ARC alpha torque density versus ψ, fgar and forb, response from q = 1.02 outward (run_fix, alpha_fix2)
fig_m3_lost     ARC alpha lost fraction versus ψ (alpha_fix2)
fig_m3_alpha_q1 ARC with q = 1 inside the response (run_qlow): forb at 32×8 and 64×12 pitch × energy nodes, and fgar

Usage (from the project root): python3 gpec/docs/workorders/ntv-orbit/reports/m3_figures.py [name ...]
"""

import csv
import glob
import sys

import numpy as np
from m2a3_figures import total, width, zow_rows
from ntvstyle import LOST, ONEHALF, ORBIT, ROOT, SINGLE, SURFACE, panel, plt, save

ZOW, ZOW_M2 = ROOT / "results" / "m3_zow2", ROOT / "results" / "m2a3_zow512"
Z1, Z1_M2 = ROOT / "results" / "m3_z1phi2", ROOT / "results" / "m2a3_z1phi"
ARC = ROOT / "results" / "m3_arc"
M2COL = "#8fb3e0"


def fig_zow():
    fig, ax = plt.subplots(figsize=(SINGLE, 2.9))
    for path, col, lab in ((ZOW_M2, M2COL, "M2"), (ZOW, ORBIT, "M3")):
        Z = zow_rows(path)
        zs = sorted(z for z in Z if z >= 16)
        w = np.array([width(z) for z in zs])
        a = np.array([abs(total(Z[z], "forb") / total(Z[z], "replay") - 1) for z in zs])
        b = np.array([abs(total(Z[z], "forbB") / total(Z[z], "replayB") - 1) for z in zs])
        ax.loglog(w, a, "o-", color=col, ms=5, label=f"A, {lab}")
        ax.loglog(w, b, "s--", color=col, ms=4, mfc="white", label=f"B, {lab}")
        s, _ = np.polyfit(np.log(w), np.log(a), 1)
        print(f"fig_m3_zow {lab}: A slope {s:.3f}")
        for z, wi, ai, bi in zip(zs, w, a, b):
            print(f"  Z={z:6d} Δb/a={wi:.3e} A={ai:.3e} B={bi:.3e}")
    ax.set_xlabel("Δ$_b$/a")
    ax.set_ylabel("|T$_{forb}$/T$_{replay}$ − 1|")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, ZOW, "fig_m3_zow")
    for z, R in sorted(zow_rows(ZOW).items()):
        _status(R, f"  Z={z:6d}")


def _status(R, tag):
    """Dropped (slot 6) and one-sided (slot 7) ∂ψ̄/∂ψ₀ samples against the samples kept, over all forb tables of R."""
    c = np.sum([[int(x) for x in r["nstatus"].split(";")] for r in R], axis=0)
    ns = sum(int(r["samples"]) for r in R)
    print(f"{tag} kept {ns} dropped {c[5]} one-sided {c[6]} ({c[6] / ns:.1%}) status {list(c)}")


def fig_profile():
    m2 = sorted(zow_rows(Z1_M2)[1], key=lambda r: float(r["psi"]))
    m3 = sorted(zow_rows(Z1)[1], key=lambda r: float(r["psi"]))
    fig, ax = plt.subplots(1, 2, figsize=(ONEHALF, 2.6))
    for a, lim in zip(ax, ((0, 1), (0.88, 1))):
        for R, key, col, ls, lab in ((m3, "fgar", SURFACE, "-", "fgar"), (m2, "forb", M2COL, "--", "forb, M2"), (m3, "forb", ORBIT, "-", "forb, M3")):
            psi = np.array([float(r["psi"]) for r in R])
            env = np.array([float(r["envelope"]) for r in R])
            a.plot(psi, env * np.array([float(r[key + "_re"]) for r in R]), ls, color=col, marker=".", ms=3, label=lab)
        a.set_xlim(*lim)
        a.set_xlabel("ψ")
    ax[0].set_ylabel("Re dT/dψ (N m)")
    ax[0].legend(loc="upper left", fontsize=8)
    panel(ax[0], "a")
    panel(ax[1], "b")
    save(fig, Z1, "fig_m3_profile")
    for R, key, lab in ((m3, "fgar", "fgar"), (m2, "forb", "forb M2"), (m3, "forb", "forb M3")):
        print(f"fig_m3_profile: {lab} total {total(R, key):.4g} over {len(R)} nodes")
    _status(m3, "fig_m3_profile forb M3")


def _csv(path):
    R = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k]) for r in R]) for k in R[0]}


def fig_moments():
    a, g = _csv(ARC / "alpha_moments.csv"), _csv(ARC / "gacode_alpha.csv")
    fig, ax = plt.subplots(1, 2, figsize=(ONEHALF, 2.6))
    ax[0].semilogy(a["psi"], a["n_alpha"], "-", color=ORBIT, label="model")
    ax[0].semilogy(g["psi"], g["n_He_fast"], "--", color=SURFACE, label="gacode")
    ax[1].semilogy(a["psi"], a["p_alpha_Pa"] / 1e3, "-", color=ORBIT)
    ax[1].semilogy(g["psi"], g["p_He_fast_Pa"] / 1e3, "--", color=SURFACE)
    ax[0].set_ylabel("n$_α$ (m$^{-3}$)")
    ax[1].set_ylabel("p$_α$ (kPa)")
    for x, letter in zip(ax, "ab"):
        x.set_xlabel("ψ")
        panel(x, letter)
    ax[0].legend(loc="lower left", fontsize=8)
    save(fig, ARC, "fig_m3_moments")


def arc_rows(sub):
    R = [r for f in glob.glob(str(ARC / sub / "arc_alpha_c*.csv")) for r in csv.DictReader(open(f))]
    return sorted(R, key=lambda r: float(r["psi"]))


def _arc_totals(R, tag):
    psi = np.array([float(r["psi"]) for r in R])
    env = np.array([float(r["envelope"]) for r in R])
    w = np.array([float(r["weight"]) for r in R])
    for key, e in (("fgar", env), ("forb", 1.0)):
        T = np.sum(w * e * np.array([complex(float(r[key + "_re"]), float(r[key + "_im"])) for r in R]))
        print(f"{tag}: {key} total {T:.4g} over {len(R)} nodes, ψ {psi.min():.3f}..{psi.max():.3f}")
    tb, ti = np.array([float(r["table_s"]) for r in R]), np.array([float(r["integrals_s"]) for r in R])
    print(f"{tag} cost: table median {np.median(tb):.0f} s, max {tb.max():.0f} s, sum {tb.sum() / 3600:.2f} h; "
          f"integrals median {np.median(ti):.0f} s, sum {ti.sum() / 3600:.2f} h")
    lf = np.array([float(r["lost_fraction"]) for r in R])
    "samples" in R[0] and _status(R, tag)
    print(f"{tag} lost fraction monotone:", bool(np.all(np.diff(lf) >= -1e-12)), "last", lf[-1])


def fig_alpha():
    """Main ARC figure: the qlow = 1.02 response (q = 1 excluded), 64 pitch × 12 energy nodes."""
    R = arc_rows("alpha_fix2")
    psi = np.array([float(r["psi"]) for r in R])
    env = np.array([float(r["envelope"]) for r in R])
    fig, ax = plt.subplots(1, 2, figsize=(ONEHALF, 2.6))
    for a, part, lab in ((ax[0], "re", "Re dT/dψ (N m)"), (ax[1], "im", "Im dT/dψ (N m)")):
        a.plot(psi, env * np.array([float(r["fgar_" + part]) for r in R]), "-", color=SURFACE, marker=".", ms=3, label="fgar")
        a.plot(psi, [float(r["forb_" + part]) for r in R], "-", color=ORBIT, marker=".", ms=3, label="forb")
        a.set_ylabel(lab)
        a.set_xlabel("ψ")
    ax[0].legend(loc="upper left", fontsize=8)
    panel(ax[0], "a")
    panel(ax[1], "b")
    save(fig, ARC, "fig_m3_alpha")
    _arc_totals(R, "fig_m3_alpha")


def fig_lost():
    R = arc_rows("alpha_fix2")
    fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
    ax.plot([float(r["psi"]) for r in R], [float(r["lost_fraction"]) for r in R], "-", color=LOST, marker=".", ms=3)
    ax.axhline(0.5, color="0.6", ls=":", lw=1)
    ax.set_xlabel("ψ")
    ax.set_ylabel("lost fraction")
    ax.set_ylim(0, 1.02)
    save(fig, ARC, "fig_m3_lost")


def fig_alpha_q1():
    """The qlow = 0.8 response (q = 1 at ψ = 0.313 inside): forb at two resolutions against fgar."""
    lo, hi = arc_rows("alpha_qlow"), arc_rows("alpha_qlow_hi")
    fig, ax = plt.subplots(1, 2, figsize=(ONEHALF, 2.6))
    for a, part, lab in ((ax[0], "re", "Re dT/dψ (N m)"), (ax[1], "im", "Im dT/dψ (N m)")):
        for R, key, col, ls, name in ((hi, "fgar", SURFACE, "-", "fgar"), (lo, "forb", M2COL, "--", "forb, 32×8"), (hi, "forb", ORBIT, "-", "forb, 64×12")):
            psi = np.array([float(r["psi"]) for r in R])
            e = np.array([float(r["envelope"]) for r in R]) if key == "fgar" else 1.0
            a.plot(psi, e * np.array([float(r[key + "_" + part]) for r in R]), ls, color=col, marker=".", ms=3, label=name)
        a.axvline(0.313, color="0.6", ls=":", lw=1)
        a.set_xlim(0, 0.8)
        a.set_ylabel(lab)
        a.set_xlabel("ψ")
    ax[1].legend(loc="lower right", fontsize=8)
    panel(ax[0], "a")
    panel(ax[1], "b")
    save(fig, ARC, "fig_m3_alpha_q1")
    _arc_totals(lo, "q1 32x8")
    _arc_totals(hi, "q1 64x12")


FIGS = {"zow": fig_zow, "profile": fig_profile, "moments": fig_moments, "alpha": fig_alpha, "lost": fig_lost, "alpha_q1": fig_alpha_q1}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGS:
        FIGS[name]()
