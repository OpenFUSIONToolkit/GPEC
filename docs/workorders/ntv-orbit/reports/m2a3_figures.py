"""M2 attempt-3 figures, from results/m2a3_dev, results/m2a3_zow and (for comparison) results/m2a2_*; each writes
<name>.{pdf,png,svg} beside its data.

fig_m2a3_phi_orbits  Φ-on orbits the old classifier left unclosed, and the class counts before and after (phi_orbits.jl)
fig_m2a3_jac         the orbit field's rebuilt Jacobian against rzphi_jac in θ (field_compare.jl)
fig_m2a3_zow         check 1: forb against the same-grid replay versus orbit width; attempt 2, attempt 3 with the replay at
                     fgar's 128 θ points (m2a2_zow.jl) and at 512 (m2a3_zow.jl)
fig_m2a3_floor       check 1 by ψ node, attempt 3 against the 512-point replay (m2a3_zow512/*.csv)
fig_m2a3_converge    check 4 with Φ off and on: change on doubling and cost per surface (converge.jl)
fig_m2a3_profile     Z = 1 torque density: fgar, and forb with Φ off (m2a3_zow) and on (m2a3_z1phi, z1_phi.jl)

Usage (from the project root): python3 gpec/docs/workorders/ntv-orbit/reports/m2a3_figures.py [name ...]
"""

import csv
import glob
import sys
from collections import Counter, defaultdict

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from ntvstyle import DOUBLE, ONEHALF, ORBIT, PASSING, ROOT, SINGLE, SURFACE, TRAPPED, UNCLOSED, panel, plt, save

DEV = ROOT / "results" / "m2a3_dev"
ZOW, ZOW128, ZOW2 = ROOT / "results" / "m2a3_zow512", ROOT / "results" / "m2a3_zow", ROOT / "results" / "m2a2_zow"
PSI_MARK = {0.5: "o", 0.7: "s", 0.96: "^"}


def _read(name):
    return np.genfromtxt(DEV / name, delimiter=",", names=True, dtype=None, encoding=None)


def _orbit(traj, lam, sigma, u):
    return traj[np.isclose(traj["lambda"], lam, atol=1e-4) & (traj["sigma"] == sigma) & np.isclose(traj["u"], u) & (traj["phi"] == 1)]


def fig_phi_orbits():
    run, tj = _read("phi_orbits_run_traj.csv"), _read("phi_orbits_tj_traj.csv")
    fig, axs = plt.subplots(1, 4, figsize=(DOUBLE, 2.4), layout="constrained", gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
    a = _orbit(run, 0.78896, -1, 0.5)
    k = a["tau_n"] <= a["tau_n"].min() + 60
    axs[0].plot(a["theta"][k], a["psi"][k], color=TRAPPED)
    axs[0].set_xlabel("θ (turns)")
    axs[0].set_ylabel("ψ")
    axs[1].plot(a["theta"][k], a["w"][k], color=TRAPPED)
    axs[1].axhline(0, color="0.6", lw=0.8)
    axs[1].set_xlabel("θ (turns)")
    axs[1].set_ylabel(r"$v_\parallel/v$")
    c = _orbit(tj, 1.0618, 1, 1.0)
    k = c["tau_n"] <= c["tau_n"].min() + 60
    axs[2].plot(np.mod(c["theta"][k], 1.0), c["w"][k], ".", ms=1.5, color=PASSING)
    axs[2].axhline(0, color="0.6", lw=0.8)
    axs[2].set_xlabel("θ mod 1")
    axs[2].set_ylabel(r"$v_\parallel/v$")
    cases = (("run", "DIII-D"), ("tj", "TJ"))
    bars = (("phi_orbits_{}_summary.csv", 0, "Φ off"), ("phi_orbits_{}_summary.csv", 1, "Φ on, old"), ("phi_orbits_{}_fix0.5_summary.csv", 1, "Φ on, new"))
    width = 0.27
    for i, (case, _) in enumerate(cases):
        for j, (fname, phi, _) in enumerate(bars):
            s = _read(fname.format(case))
            cnt = Counter(s["status"][s["phi"] == phi])
            x = i + (j - 1) * width * 1.1
            b = 0
            for st, col in (("trapped", TRAPPED), ("passing", PASSING), ("unclosed", UNCLOSED)):
                axs[3].bar(x, cnt[st], width, bottom=b, color=col, label=st if (i, j) == (0, 0) else None, edgecolor="white", lw=0.5)
                b += cnt[st]
            axs[3].text(x, b + 3, "off old new".split()[j], ha="center", va="bottom", fontsize=8, rotation=90)
    axs[3].set_xticks([0, 1], [c[1] for c in cases])
    axs[3].set_ylim(0, 200)
    axs[3].set_ylabel("orbits")
    axs[3].legend(loc="upper center", fontsize=8, frameon=False, ncol=3, handlelength=0.8, columnspacing=0.6, bbox_to_anchor=(0.5, 1.0))
    for ax, l in zip(axs, "abcd"):
        panel(ax, l)
    save(fig, DEV, "fig_m2a3_phi_orbits")


def fig_jac():
    d = _read("field_compare.csv")
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    for psi, col in ((0.97, "#b9b8b3"), (0.918, "#0b0b0b"), (0.5, TRAPPED)):
        s = d[np.isclose(d["psi"], psi)]
        ax.plot(s["theta"], -2 * np.pi * s["djac_orb"] / s["djac_eq"], "-", color=col, lw=1, label=f"ψ = {psi}")
    ax.set_ylim(0, 2.5)
    ax.set_xlabel("θ (turns)")
    ax.set_ylabel(r"$\partial_\psi\mathcal{J}$ rebuilt / rzphi")
    ax.legend(loc="upper right", fontsize=8, ncol=3, handlelength=1.0, columnspacing=0.8)
    save(fig, DEV, "fig_m2a3_jac")


def zow_rows(path):
    out = defaultdict(list)
    for f in glob.glob(str(path / "zow_z*_c*.csv")):
        if "classes" in f:
            continue
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
    fig, ax = plt.subplots(figsize=(SINGLE, 2.9))
    for path, col, lab, show in ((ZOW2, SURFACE, "attempt 2", "AB"), (ZOW128, "#8fb3e0", "θ: 128", "B"), (ZOW, ORBIT, "attempt 3", "AB")):
        Z = zow_rows(path)
        zs = sorted(z for z in Z if z >= 16)
        w = np.array([width(z) for z in zs])
        a = np.array([abs(total(Z[z], "forb") / total(Z[z], "replay") - 1) for z in zs])
        b = np.array([abs(total(Z[z], "forbB") / total(Z[z], "replayB") - 1) for z in zs])
        if "A" in show:
            ax.loglog(w, a, "o-", color=col, ms=5, label=f"A, {lab}")
        ax.loglog(w, b, "s--", color=col, ms=4, mfc="white", label=f"B, {lab}")
        s, _ = np.polyfit(np.log(w), np.log(a), 1)
        loc = np.diff(np.log(a)) / np.diff(np.log(w))
        print(f"fig_m2a3_zow {lab}: A slope {s:.3f} over Δb/a {w.min():.2e}..{w.max():.2e}; local slopes " + " ".join(f"{x:.2f}" for x in loc))
        for z, wi, ai, bi in zip(zs, w, a, b):
            print(f"  Z={z:6d} Δb/a={wi:.3e} A={ai:.3e} B={bi:.3e}")
    ax.set_xlabel("Δ$_b$/a")
    ax.set_ylabel("|T$_{forb}$/T$_{replay}$ − 1|")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, ZOW, "fig_m2a3_zow")


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
        a.set_ylim(1e-10, 1e-1)
        panel(a, letter)
    ax[0].set_ylabel("node share of |ΔT| / |T|")
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(zs), bbox_to_anchor=(0.5, 1.07), handlelength=1.2, columnspacing=1.0)
    save(fig, ZOW, "fig_m2a3_floor")


def fig_converge():
    R = list(csv.DictReader(open(DEV / "converge.csv")))
    fig, ax = plt.subplots(1, 2, figsize=(ONEHALF, 2.6))
    for pot, col, ls in (("false", SURFACE, "--"), ("true", ORBIT, "-")):
        for psi, mk in PSI_MARK.items():
            rs = sorted((r for r in R if r["potential"] == pot and abs(float(r["psi"]) - psi) < 1e-9), key=lambda r: int(r["nlambda"]))
            T = [complex(float(r["T_re"]), float(r["T_im"])) for r in rs]
            nl = [int(r["nlambda"]) for r in rs]
            d = [abs(T[i + 1] - T[i]) / abs(T[i + 1]) * 100 for i in range(len(T) - 1)]
            ax[0].plot([f"{nl[i]}→{nl[i + 1]}" for i in range(len(d))], d, ls, marker=mk, color=col, ms=5)
            ax[1].loglog(nl, [float(r["build_s"]) / 60 for r in rs], ls, marker=mk, color=col, ms=5)
    ax[0].axhline(5, color="0.5", ls=":", lw=1)
    ax[0].set_ylabel("|ΔT|/|T| (%)")
    ax[0].set_xlabel("pitch nodes")
    ax[1].set_xlabel("pitch nodes")
    ax[1].set_ylabel("table minutes per surface")
    ax[1].set_xticks([32, 64, 128], ["32", "64", "128"])
    ax[1].xaxis.set_minor_formatter(NullFormatter())
    ax[1].set_yticks([0.5, 1, 2, 4], ["0.5", "1", "2", "4"])
    ax[1].yaxis.set_minor_formatter(NullFormatter())
    panel(ax[0], "a")
    panel(ax[1], "b")
    handles = [Line2D([], [], color="k", marker=m, ls="") for m in PSI_MARK.values()]
    handles += [Line2D([], [], color=SURFACE, ls="--"), Line2D([], [], color=ORBIT)]
    fig.legend(handles, [f"ψ = {p}" for p in PSI_MARK] + ["Φ off", "Φ on"], loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08),
               columnspacing=0.8, handletextpad=0.3)
    save(fig, DEV, "fig_m2a3_converge")


def fig_profile():
    off = sorted(zow_rows(ZOW128)[1], key=lambda r: float(r["psi"]))
    on = sorted(zow_rows(ROOT / "results" / "m2a3_z1phi")[1], key=lambda r: float(r["psi"]))
    fig, ax = plt.subplots(1, 2, figsize=(ONEHALF, 2.6))
    for a, lim in zip(ax, ((0, 1), (0.88, 1))):
        for R, key, col, ls, lab in ((off, "fgar", SURFACE, "-", "fgar"), (off, "forb", ORBIT, "--", "forb, Φ off"), (on, "forb", ORBIT, "-", "forb, Φ on")):
            psi = np.array([float(r["psi"]) for r in R])
            env = np.array([float(r["envelope"]) for r in R])
            a.plot(psi, env * np.array([float(r[key + "_re"]) for r in R]), ls, color=col, marker=".", ms=3, label=lab)
        a.set_xlim(*lim)
        a.set_xlabel("ψ")
    ax[0].set_ylabel("Re dT/dψ (N m)")
    ax[0].legend(loc="upper left", fontsize=8)
    panel(ax[0], "a")
    panel(ax[1], "b")
    save(fig, ROOT / "results" / "m2a3_z1phi", "fig_m2a3_profile")
    for R, key, lab in ((off, "fgar", "fgar"), (off, "forb", "forb Φ off"), (on, "forb", "forb Φ on")):
        inner = [r for r in R if float(r["psi"]) < 0.9]
        outer = [r for r in R if float(r["psi"]) >= 0.9]
        print(f"fig_m2a3_profile: {lab} total {total(R, key):.4g}, ψ<0.9 {total(inner, key):.4g}, ψ≥0.9 {total(outer, key):.4g}; table {sum(float(r['table_s']) for r in R) / 3600:.2f} h")


FIGS = {"phi_orbits": fig_phi_orbits, "jac": fig_jac, "zow": fig_zow, "floor": fig_floor, "converge": fig_converge, "profile": fig_profile}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGS:
        FIGS[name]()
