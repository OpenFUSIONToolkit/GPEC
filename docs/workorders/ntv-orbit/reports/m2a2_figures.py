"""M2 attempt-2 figures, all from results/m2a2_dev and results/m2a2_zow; each writes <name>.{pdf,png,svg} beside its data.

fig_m2a2_ures      orbit quantities against energy at fixed pitch: fine scan, attempt 1's 12 nodes, attempt 2's refined nodes (ures*.csv)
fig_m2a2_table     one per-column forb table in the (λ, u) plane: class segments and their sample nodes (table_p070_z1.csv)
fig_m2a2_pitch     pitch-integral checks: replay error against pitch nodes, and the 512-node reference (pitch_*.csv)
fig_m2a2_converge  check 4: torque change on doubling the resolution, and the cost per surface (converge.csv)
fig_m2a2_zow       check 1: forb against the same-grid replay versus orbit width, both variants (m2a2_zow/*.csv)
fig_m2a2_floor     check 1 by ψ node: each node's share of |T_forb − T_replay| in both variants (m2a2_zow/*.csv)
fig_m2a2_classes   variant B by orbit class (m2a2_zow_classes/*.csv)
fig_m2a2_wd        passing-orbit precession over the surface ω_D at Z = 1024 (wd_ratio_z1024.csv)
fig_m2a2_profile   Z = 1 torque density: forb, its same-grid replay, and fgar (m2a2_zow/*.csv)
"""

import csv
import glob
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from scipy.interpolate import CubicSpline

from ntvstyle import DOUBLE, ONEHALF, ORBIT, PASSING, ROOT, SINGLE, SURFACE, TRAPPED, panel, save

DEV = ROOT / "results" / "m2a2_dev"
ZOW = ROOT / "results" / "m2a2_zow"
ZOWC = ROOT / "results" / "m2a2_zow_classes"
CLASS = {1: TRAPPED, 2: PASSING}
PSI_MARK = {0.5: "o", 0.7: "s", 0.96: "^"}


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
    R = list(csv.DictReader(open(DEV / "ures.csv"))) + list(csv.DictReader(open(DEV / "ures_table.csv")))
    cases = [(0.70, 0.75, -1), (0.70, 1.05, 1)]
    fig, ax = plt.subplots(2, 2, figsize=(ONEHALF, 3.9), sharex=True)
    for j, (psi, lam, sig) in enumerate(cases):
        for i, q in enumerate(("wb", "dj")):
            a = ax[i, j]
            uf, vf, sf = series(rows(R, psi, lam, sig, "fine"), q)
            un, vn, sn = series(rows(R, psi, lam, sig, "node"), q)
            ur, vr, sr = series(rows(R, psi, lam, sig, "refined"), q)
            for c, col in CLASS.items():
                m, mn, mr = sf == c, sn == c, sr == c
                if m.sum() < 2:
                    continue
                a.plot(uf[m], vf[m], "-", color=col)
                if mn.sum() >= 4:
                    us = np.linspace(un[mn].min(), un[mn].max(), 300)
                    a.plot(us, CubicSpline(un[mn], vn[mn])(us), ":", color=SURFACE, lw=1.2)
                a.plot(un[mn], vn[mn], "s", color=SURFACE, ms=3.5)
                if mr.sum() >= 4:
                    us = np.linspace(ur[mr].min(), ur[mr].max(), 300)
                    a.plot(us, CubicSpline(ur[mr], vr[mr])(us), "--", color="k", lw=1.0)
                a.plot(ur[mr], vr[mr], "o", color="k", ms=2.5)
            for e in uf[1:][sf[1:] != sf[:-1]]:
                a.axvline(e, color="0.6", lw=0.8, ls="--")
            panel(a, "abcd"[2 * i + j])
            if i == 1:
                a.set_xlabel("u = √x")
    ax[0, 0].set_ylabel("ω$_b$/√x (krad/s)")
    ax[1, 0].set_ylabel("Σ|δJ$_ℓ$|² (×10$^{-4}$)")
    handles = [Line2D([], [], color=TRAPPED), Line2D([], [], color=PASSING), Line2D([], [], color=SURFACE, ls=":", marker="s", ms=3.5),
               Line2D([], [], color="k", ls="--", marker="o", ms=2.5)]
    fig.legend(handles, ["trapped", "passing", "attempt 1", "attempt 2"], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07),
               columnspacing=0.9, handletextpad=0.4)
    save(fig, DEV, "fig_m2a2_ures")


def fig_table():
    R = list(csv.DictReader(open(DEV / "table_p070_z1.csv")))
    cols = defaultdict(lambda: defaultdict(list))
    for r in R:
        cols[int(r["il"])][int(r["seg"])].append(r)
    fig, ax = plt.subplots(1, 2, figsize=(ONEHALF, 3.0), sharey=True)
    lams = sorted({float(r["lambda"]) for r in R})
    for a in ax:
        for lam in lams:
            a.plot([lam, lam], [0, 6.5], "-", color="#cfcfcb", lw=2.2, solid_capstyle="butt")
    for il, segs in cols.items():
        sign, last_ub = 0, -1.0
        for k in sorted(segs):
            rs = segs[k]
            ua, ub, lam = float(rs[0]["ua"]), float(rs[0]["ub"]), float(rs[0]["lambda"])
            if ua < last_ub - 1e-9:
                sign = 1
            last_ub = ub
            col = PASSING if rs[0]["passing"] == "true" else TRAPPED
            a = ax[sign]
            a.plot([lam, lam], [ua, ub], "-", color=col, lw=2.2, solid_capstyle="butt")
            a.plot([lam] * len(rs), [float(r["u"]) for r in rs], ".", color="k", ms=1.6)
    for a, letter in zip(ax, "ab"):
        a.set_xlabel("λ")
        panel(a, letter)
    ax[0].set_ylabel("u = √x")
    handles = [Line2D([], [], color=c, lw=2.2) for c in (PASSING, TRAPPED)] + [Line2D([], [], color="#cfcfcb", lw=2.2),
                                                                            Line2D([], [], color="k", marker=".", ls="")]
    fig.legend(handles, ["passing", "trapped", "lost", "nodes"], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06))
    save(fig, DEV, "fig_m2a2_table")


def fig_pitch():
    S = list(csv.DictReader(open(DEV / "pitch_scheme.csv")))
    F = list(csv.DictReader(open(DEV / "pitch_ref512.csv")))
    fig, ax = plt.subplots(1, 2, figsize=(ONEHALF, 2.6))
    for psi, mk in PSI_MARK.items():
        for scheme, ls in (("E-spline", "--"), ("input-spline", "-")):
            rs = sorted((r for r in S if r["scheme"] == scheme and abs(float(r["psi"]) - psi) < 1e-9), key=lambda r: int(r["nodes"]))
            x = [int(r["nodes"]) for r in rs]
            y = [float(r["total_rel_to_fgar128"]) for r in rs]
            ax[0].loglog(x, y, ls, marker=mk, color=ORBIT, ms=4)
    ax[0].set_xlabel("pitch nodes")
    ax[0].set_ylabel("|T/T$_{fgar,128}$ − 1|")
    ax[0].set_xticks([16, 32, 64, 128], ["16", "32", "64", "128"])
    ax[0].xaxis.set_minor_formatter(NullFormatter())
    cases = ["fgar 128", "replay 128", "replay 512"]
    colors = [SURFACE, ORBIT, ORBIT]
    for j, (case, col) in enumerate(zip(cases, colors)):
        for k, psi in enumerate(PSI_MARK):
            v = [float(r["total_rel_to_fgar512"]) for r in F if r["case"] == case and abs(float(r["psi"]) - psi) < 1e-9][0]
            ax[1].semilogy(j + (k - 1) * 0.18, v, PSI_MARK[psi], color=col, ms=5)
    ax[1].set_xticks(range(3), ["fgar\n128", "replay\n128", "replay\n512"])
    ax[1].set_xlim(-0.5, 2.5)
    ax[1].set_ylabel("|T/T$_{fgar,512}$ − 1|")
    panel(ax[0], "a")
    panel(ax[1], "b")
    handles = [Line2D([], [], color="k", marker=m, ls="") for m in PSI_MARK.values()]
    fig.legend(handles, [f"ψ = {p}" for p in PSI_MARK], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    save(fig, DEV, "fig_m2a2_pitch")


def fig_converge():
    R = list(csv.DictReader(open(DEV / "converge.csv")))
    fig, ax = plt.subplots(1, 2, figsize=(ONEHALF, 2.6))
    for psi, mk in PSI_MARK.items():
        rs = sorted((r for r in R if abs(float(r["psi"]) - psi) < 1e-9), key=lambda r: int(r["nlambda"]))
        T = [complex(float(r["To_re"]), float(r["To_im"])) for r in rs]
        nl = [int(r["nlambda"]) for r in rs]
        d = [abs(T[i + 1] - T[i]) / abs(T[i + 1]) * 100 for i in range(len(T) - 1)]
        ax[0].plot([f"{nl[i]}→{nl[i + 1]}" for i in range(len(d))], d, "-", marker=mk, color=ORBIT, ms=5)
        ax[1].loglog(nl, [float(r["build_s"]) / 60 for r in rs], "-", marker=mk, color=ORBIT, ms=5)
        ax[1].loglog(nl, [float(r["integral_s"]) / 60 for r in rs], ":", marker=mk, color=SURFACE, ms=5)
    ax[0].axhline(5, color="0.5", ls="--", lw=1)
    ax[0].set_ylabel("|ΔT|/|T| (%)")
    ax[0].set_xlabel("pitch nodes")
    ax[1].set_xlabel("pitch nodes")
    ax[1].set_ylabel("minutes per surface")
    ax[1].set_xticks([32, 64, 128], ["32", "64", "128"])
    ax[1].xaxis.set_minor_formatter(NullFormatter())
    ax[1].set_yticks([0.2, 0.5, 1, 2, 5], ["0.2", "0.5", "1", "2", "5"])
    ax[1].yaxis.set_minor_formatter(NullFormatter())
    panel(ax[0], "a")
    panel(ax[1], "b")
    handles = [Line2D([], [], color="k", marker=m, ls="") for m in PSI_MARK.values()]
    handles += [Line2D([], [], color=ORBIT), Line2D([], [], color=SURFACE, ls=":")]
    fig.legend(handles, [f"ψ = {p}" for p in PSI_MARK] + ["table", "integrals"], loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08),
               columnspacing=0.8, handletextpad=0.3)
    save(fig, DEV, "fig_m2a2_converge")


def zow_rows(path):
    out = defaultdict(list)
    for f in glob.glob(str(path / "zow_z*.csv")):
        for r in csv.DictReader(open(f)):
            out[int(r["zi"])].append(r)
    return out


def total(R, key):
    return sum(float(r["weight"]) * float(r["envelope"]) * complex(float(r[key + "_re"]), float(r[key + "_im"])) for r in R)


def width(zi):
    """Trapped-orbit width over a at ψ = 0.5 (attempt-1 scan, m2_zow/zow_scan.csv); ∝ 1/Z beyond its last charge."""
    W = {int(r["zi"]): float(r["width_over_a"]) for r in csv.DictReader(open(ROOT / "results" / "m2_zow" / "zow_scan.csv"))}
    return W[zi] if zi in W else W[4096] * 4096 / zi


def fig_zow():
    Z = zow_rows(ZOW)
    zs = sorted(z for z in Z if z > 1)
    w = np.array([width(z) for z in zs])
    a = np.array([abs(total(Z[z], "forb") / total(Z[z], "replay") - 1) for z in zs])
    b = np.array([abs(total(Z[z], "forbB") / total(Z[z], "replayB") - 1) for z in zs])
    old = {int(r["zi"]): float(r["rel"]) for r in csv.DictReader(open(ROOT / "results" / "m2_zow" / "zow_scan.csv"))}
    fig, ax = plt.subplots(figsize=(SINGLE, 2.8))
    ax.loglog(w, a, "o-", color=ORBIT, ms=5, label="A")
    ax.loglog(w, b, "s--", color=ORBIT, ms=4, label="B")
    zo = sorted(z for z in old if z > 1)
    ax.loglog([width(z) for z in zo], [old[z] for z in zo], "x:", color=SURFACE, ms=5, label="attempt 1")
    m = (w > 3e-5) & (w < 4e-3)
    if m.sum() >= 2:
        s, c = np.polyfit(np.log(w[m]), np.log(a[m]), 1)
        ax.loglog(w[m], np.exp(c) * w[m] ** s, "-", color="0.6", lw=0.8)
        print(f"fig_m2a2_zow: slope {s:.3f} over Δb/a {w[m].min():.2e}..{w[m].max():.2e}")
    ax.set_xlabel("Δ$_b$/a")
    ax.set_ylabel("|T$_{forb}$/T$_{replay}$ − 1|")
    ax.legend(loc="lower right")
    save(fig, ZOW, "fig_m2a2_zow")
    for z, wi, ai, bi in zip(zs, w, a, b):
        print(f"  Z={z:6d} Δb/a={wi:.3e} A={ai:.3e} B={bi:.3e}")


def fig_floor():
    Z = zow_rows(ZOW)
    zs = [z for z in (16, 256, 4096, 65536) if z in Z]
    shades = dict(zip(zs, ("#b9b8b3", "#8a8984", "#4d4c48", "#0b0b0b")))
    fig, ax = plt.subplots(1, 2, figsize=(ONEHALF, 2.6), sharey=True)
    for z in zs:
        R = sorted(Z[z], key=lambda r: float(r["psi"]))
        psi = np.array([float(r["psi"]) for r in R])
        for a, v in zip(ax, ("", "B")):
            T = total(R, "replay" + v)
            d = [abs(total([r], "forb" + v) - total([r], "replay" + v)) / abs(T) for r in R]
            a.semilogy(psi, d, ".-", color=shades[z], ms=4, lw=1, label=f"Z = {z}")
    for a, letter in zip(ax, "ab"):
        a.set_xlabel("ψ")
        a.set_ylim(1e-9, 1e-1)
        panel(a, letter)
    ax[0].set_ylabel("node share of |ΔT| / |T|")
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(zs), bbox_to_anchor=(0.5, 1.07), handlelength=1.2, columnspacing=1.0)
    save(fig, ZOW, "fig_m2a2_floor")


def fig_classes():
    Z = zow_rows(ZOWC)
    zs = sorted(Z)
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    for v, mk, ls in (("", "o", "-"), ("B", "s", "--")):
        for c, col in (("passing", PASSING), ("trapped", TRAPPED)):
            rel = [abs(total(Z[z], f"cls_forb{v}_{c}") / total(Z[z], f"cls_replay{v}_{c}") - 1) for z in zs]
            ax.semilogy([str(z) for z in zs], rel, mk + ls, color=col, ms=5)
            print(f"fig_m2a2_classes: variant {v or 'A'} {c}: " + ", ".join(f"Z={z} {r:.2e}" for z, r in zip(zs, rel)))
    ax.set_xlabel("Z")
    ax.set_ylabel("|T$_{forb}$/T$_{replay}$ − 1|")
    handles = [Line2D([], [], color=PASSING, lw=2), Line2D([], [], color=TRAPPED, lw=2),
               Line2D([], [], color="k", marker="o", ls="-"), Line2D([], [], color="k", marker="s", ls="--")]
    ax.legend(handles, ["passing", "trapped", "A", "B"], ncol=2, loc="best")
    save(fig, ZOWC, "fig_m2a2_classes")


def fig_wd():
    R = list(csv.DictReader(open(DEV / "wd_ratio_z1024.csv")))
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    for cls, d, col, mk in (("pass", "1", PASSING, "^"), ("pass", "-1", PASSING, "v"), ("trap", "0", TRAPPED, "o")):
        rs = [r for r in R if r["cls"] == cls and r["dir"] == d]
        lam = np.array([float(r["lambda"]) for r in rs])
        mid = np.array([(float(r["wd_lo"]) + float(r["wd_hi"])) / 2 for r in rs])
        ax.plot(lam, mid, mk, color=col, ms=4)
    ax.axhline(1, color="0.5", ls="--", lw=1)
    ax.axhline(0, color="0.8", lw=0.6)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("λ")
    ax.set_ylabel("orbit precession / surface ω$_D$")
    handles = [Line2D([], [], color=PASSING, marker="^", ls=""), Line2D([], [], color=PASSING, marker="v", ls=""),
               Line2D([], [], color=TRAPPED, marker="o", ls="")]
    ax.legend(handles, ["co-passing", "counter", "trapped"], loc="lower left")
    save(fig, DEV, "fig_m2a2_wd")


def fig_profile():
    Z = zow_rows(ZOW)
    R = sorted(Z[1], key=lambda r: float(r["psi"]))
    psi = np.array([float(r["psi"]) for r in R])
    env = np.array([float(r["envelope"]) for r in R])
    fig, ax = plt.subplots(1, 2, figsize=(ONEHALF, 2.6))
    for a, lim in zip(ax, ((0, 1), (0.88, 1))):
        for key, col, ls in (("fgar", SURFACE, "-"), ("replay", SURFACE, ":"), ("forb", ORBIT, "-")):
            a.plot(psi, env * np.array([float(r[key + "_re"]) for r in R]), ls, color=col, marker=".", ms=3, label=key)
        a.set_xlim(*lim)
        a.set_xlabel("ψ")
    ax[0].set_ylabel("Re dT/dψ (N m)")
    ax[0].legend(loc="upper left")
    panel(ax[0], "a")
    panel(ax[1], "b")
    save(fig, ZOW, "fig_m2a2_profile")
    for key in ("fgar", "replay", "forb"):
        print(f"fig_m2a2_profile: Z=1 {key} integrated {total(R, key):.4g} over {len(R)} nodes")


FIGS = {"ures": fig_ures, "table": fig_table, "pitch": fig_pitch, "converge": fig_converge, "zow": fig_zow, "floor": fig_floor, "classes": fig_classes,
        "wd": fig_wd, "profile": fig_profile}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGS:
        FIGS[name]()
