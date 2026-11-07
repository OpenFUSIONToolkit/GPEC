"""M4 figures and tables, from results/m4_{arc,iter,sparc}; each figure writes <name>.{pdf,png,svg} into results/m4_arc.

table           per device, variant and λ: Δ_b/a of a 3.5 MeV trapped alpha at r/a = 0.8, fgar and forb totals, ratio, nodes
exclusion       ARC: the ratio with 0..3 orbit widths excluded from the inner edge of the response (the cheap A-vs-B check)
fig_alpha_scan  KR1: Re(forb)/Re(fgar) total alpha torque versus Δ_b/a, field scan (variants B and A) and energy scan
fig_m4_profiles ARC torque density versus ψ for each λ (variant B), fgar and forb
fig_m4_coils    plasma boundary and the n = 1 forcing coils of ARC, ITER and SPARC

Usage (from the project root): python3 gpec/docs/workorders/ntv-orbit/reports/m4_figures.py [name ...]
"""

import csv
import re
import glob
import sys

import numpy as np
from ntvstyle import ORBIT, ROOT, SINGLE, SURFACE, panel, plt, save

RES = ROOT / "results"
DEVICES = {"ARC": RES / "m4_arc", "ITER": RES / "m4_iter", "SPARC": RES / "m4_sparc"}
LAMS = (1, 2, 4, 8, 16)
MARK = {"ARC": "o", "ITER": "s", "SPARC": "D"}
# Energy-scan runs at λ = 1: directory and the meta.csv width column for that birth energy.
ESCAN = {"E_875": ("width_0p88MeV", 0.875), "E_219": ("width_0p22MeV", 0.21875)}


def rows(path):
    """Per-surface rows of a run; surfaces rerun after the refinement-spacing fix (path/fix, weight 0) replace their values."""
    R = [r for f in glob.glob(str(path / "arc_alpha_c*.csv")) for r in csv.DictReader(open(f))]
    fix = {r["psi"]: r for f in glob.glob(str(path / "fix" / "arc_alpha_c*.csv")) for r in csv.DictReader(open(f))}
    R = [{**r, **{k: v for k, v in fix[r["psi"]].items() if k not in ("psi", "weight")}} if r["psi"] in fix else r for r in R]
    return sorted(R, key=lambda r: float(r["psi"]))


def meta(dev, lam, eq=""):
    f = DEVICES[dev] / f"{eq}lam{lam:02d}" / "meta.csv"
    return next(csv.DictReader(open(f))) if f.exists() else None


def run_dir(dev, variant, lam, dense=True, eq=""):
    """Collisions: fixed ν (dirs A_, DA_) and ν/λ (B_, DB_) coincide at λ = 1. Dense runs (D*, uniform nodes over the
    resonance steps) are preferred; the 39-node Gauss runs are the fallback and the dense=False comparison. eq = "q110_"
    selects the q₀ = 1.1 family."""
    if variant == "A" and lam > 1:
        pre = ("DA", "A") if dense else ("A",)
    elif lam == 1:
        pre = ("DA", "DB", "B") if dense else ("B",)
    else:
        pre = ("DB", "B") if dense else ("B",)
    for p in pre:
        d = DEVICES[dev] / f"{eq}{p}_lam{lam:02d}"
        if rows(d):
            return d
    return DEVICES[dev] / f"{eq}{pre[-1]}_lam{lam:02d}"


def totals(R, psimin=-np.inf):
    """Weighted ψ sums of the torque: fgar times the axis-validity envelope, forb as is; nodes below psimin dropped."""
    R = [r for r in R if float(r["psi"]) >= psimin]
    w = np.array([float(r["weight"]) for r in R])
    env = np.array([float(r["envelope"]) for r in R])
    tf = np.sum(w * env * np.array([complex(float(r["fgar_re"]), float(r["fgar_im"])) for r in R]))
    to = np.sum(w * np.array([complex(float(r["forb_re"]), float(r["forb_im"])) for r in R]))
    return tf, to, len(R)


def scan():
    """List of dicts: dev, eq ("" published q, "q110_"), kind (fixed, nulam, E), lam, width, fgar, forb, nodes, src."""
    out = []
    for dev in DEVICES:
        for eq in ("", "q110_"):
            for variant, kind in (("A", "fixed"), ("B", "nulam")):
                for lam in LAMS:
                    d = run_dir(dev, variant, lam, eq=eq)
                    m, R = meta(dev, lam, eq), rows(d)
                    if m is None or not R:
                        continue
                    tf, to, n = totals(R)
                    out.append(dict(dev=dev, eq=eq, kind=kind, lam=lam, width=float(m["width_3p5MeV"]), fgar=tf, forb=to, nodes=n,
                                    src=d.name, lost=float(R[-1]["lost_fraction"])))
            for sub, (col, e) in ESCAN.items():
                d = DEVICES[dev] / (eq + sub if eq == "" else f"{eq}E{int(e * 1000)}")
                m, R = meta(dev, 1, eq), rows(d)
                if m is None or not R:
                    continue
                tf, to, n = totals(R)
                out.append(dict(dev=dev, eq=eq, kind="E", lam=1, energy=e, width=float(m[col]), fgar=tf, forb=to, nodes=n, src=d.name,
                                lost=float(R[-1]["lost_fraction"])))
    return out


def table():
    for s in scan():
        print(f"{s['dev']:5s} {s['eq'] or 'pub_':5s} {s['kind']:5s} λ={s['lam']:2d} {s.get('energy', 3.5):6.3f} MeV  Δb/a={s['width']:.4f}  "
              f"fgar {s['fgar']:.4g}  forb {s['forb']:.4g}  ratio {s['forb'].real / s['fgar'].real:.4f}  "
              f"nodes {s['nodes']} {s.get('src', '')}  edge lost {s['lost']:.3f}")


def exclusion(dev="ARC"):
    """Ratio with k = 0..3 orbit widths (in ψ, from the trapped-orbit ψ range at r/a = 0.8) dropped above the response's inner edge."""
    for variant in ("B", "A"):
        for lam in LAMS:
            m, R = meta(dev, lam), rows(run_dir(dev, variant, lam))
            if m is None or not R or "dpsi_3p5MeV" not in m:
                continue
            p0, dpsi = float(m["psilow_response"]), float(m["dpsi_3p5MeV"])
            r = []
            for k in range(4):
                tf, to, n = totals(R, p0 + k * dpsi)
                r.append(f"{to.real / tf.real:.4f} ({n})")
            print(f"{dev} {variant} λ={lam:2d} Δψ={dpsi:.4f}: k=0..3 ratio " + "  ".join(r))


def fig_alpha_scan():
    """(a) published q, (b) q₀ = 1.1: ratio versus Δ_b/a; fixed ν (filled, solid), ν/λ (open, dotted), energy scan (grey)."""
    S = scan()
    full = {}
    for s in S:
        full[s["dev"], s["eq"]] = max(full.get((s["dev"], s["eq"]), 0), s["nodes"] if s["src"].startswith(s["eq"] + "D") else 0)
    S = [s for s in S if not s["src"].startswith(s["eq"] + "D") or s["nodes"] == full[s["dev"], s["eq"]]]
    fig, axs = plt.subplots(1, 2, figsize=(SINGLE, 2.5), sharey=True, sharex=True)
    for ax, eq, letter in zip(axs, ("", "q110_"), "ab"):
        for dev in DEVICES:
            for kind, face, ls, color in (("fixed", "full", "-", ORBIT), ("nulam", "none", ":", ORBIT), ("E", "none", "", SURFACE)):
                P = sorted((s for s in S if s["dev"] == dev and s["eq"] == eq and s["kind"] == kind), key=lambda s: s["width"])
                if not P:
                    continue
                ax.plot([s["width"] for s in P], [s["forb"].real / s["fgar"].real for s in P], marker=MARK[dev], fillstyle=face, ls=ls,
                        color=color, ms=5, label=dev if kind == "fixed" else None)
        ax.axhline(1.0, color=SURFACE, lw=0.8)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel(r"$\Delta_b/a$")
        panel(ax, letter)
    axs[0].set_ylabel(r"$T_\mathrm{forb}/T_\mathrm{fgar}$")
    axs[0].legend(frameon=False, fontsize=9)
    save(fig, DEVICES["ARC"], "fig_alpha_scan")


def fig_m4_profiles(dev="ARC"):
    fig, axs = plt.subplots(len(LAMS), 1, figsize=(SINGLE, 1.1 * len(LAMS) + 0.6), sharex=True)
    for ax, lam, letter in zip(axs, LAMS, "abcde"):
        R, m = rows(run_dir(dev, "B", lam)), meta(dev, lam)
        if not R:
            continue
        psi = np.array([float(r["psi"]) for r in R])
        env = np.array([float(r["envelope"]) for r in R])
        ax.plot(psi, env * np.array([float(r["fgar_re"]) for r in R]), color=SURFACE, marker=".", label="fgar")
        ax.plot(psi, [float(r["forb_re"]) for r in R], color=ORBIT, marker=".", label="forb")
        ax.text(0.98, 0.85, rf"$\lambda = {lam}$, $\Delta_b/a = {float(m['width_3p5MeV']):.3f}$", transform=ax.transAxes, ha="right", va="top")
        panel(ax, letter)
    axs[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.12, 1.0))
    axs[-1].set_xlabel(r"$\psi_N$")
    axs[len(LAMS) // 2].set_ylabel(r"$dT/d\psi_N$ (N m)")
    save(fig, DEVICES[dev], f"fig_m4_profiles_{dev.lower()}")


def boundary(geqdsk):
    """(R, Z) of the gEQDSK plasma boundary."""
    num = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?")
    lines = open(geqdsk).read().split("\n")
    rest = "\n".join(lines[1:])
    m = re.search(r"\n\s*(\d+)\s+(\d+)\s*\n", rest)
    t = [float(x) for x in num.findall(rest[m.end():])][: 2 * int(m.group(1))]
    return np.array(t[0::2]), np.array(t[1::2])


def fig_m4_coils():
    """Plasma boundary and the n = 1 forcing coils' poloidal cross-sections for the three devices."""
    eq = {"ARC": DEVICES["ARC"] / "inputs" / "geqdsk-ARCv3a-x1", "ITER": DEVICES["ITER"] / "inputs" / "geqdsk-ITER-Hmode",
          "SPARC": DEVICES["SPARC"] / "inputs" / "SPARC_DN_PRD_freegs_513_20221206"}
    a = np.loadtxt(ROOT / "gpec" / "src" / "ForcingTerms" / "coil_geometries" / "iter_efcm.dat", skiprows=1)
    coils = {"ARC": [((6.59, 8.07), (5.34, 4.00)), ((6.59, 8.07), (-5.34, -4.00))],
             "ITER": [((np.hypot(a[:, 0], a[:, 1]).mean(),) * 2, (a[:, 2].min(), a[:, 2].max()))],
             "SPARC": [((2.82, 2.82), (-0.75, 0.75))]}
    fig, axs = plt.subplots(1, 3, figsize=(SINGLE, 2.4))
    for ax, dev, letter in zip(axs, DEVICES, "abc"):
        R, Z = boundary(eq[dev])
        ax.plot(R, Z, color=SURFACE)
        for r, z in coils[dev]:
            ax.plot(r, z, color=ORBIT, lw=3, solid_capstyle="butt")
        ax.set_aspect("equal")
        ax.set_xlabel("R (m)")
        ax.set_title(dev, loc="right")
        panel(ax, letter)
    axs[0].set_ylabel("Z (m)")
    save(fig, DEVICES["ARC"], "fig_m4_coils")


def _surface(dev, sub, psi):
    for r in rows(DEVICES[dev] / sub):
        if abs(float(r["psi"]) - psi) < 5e-5:
            return float(r["envelope"]) * float(r["fgar_re"]), float(r["forb_re"])
    return None


def fig_m4_nu():
    """Per-surface |forb − fgar|/|fgar| versus λ on three ARC surfaces, fixed ν and ν/λ (39-node runs and single surfaces)."""
    fig, axs = plt.subplots(1, 2, figsize=(SINGLE, 2.3), sharey=True)
    for ax, (v, title), letter in zip(axs, (("A", "fixed ν"), ("B", "ν/λ")), "ab"):
        for psi, mk in ((0.6113, "o"), (0.6500, "s"), (0.6887, "D")):
            x, y = [], []
            for lam in (2, 4, 8, 16, 64, 256):
                c = 8 + [0.5500, 0.5887, 0.6113, 0.6500, 0.6887].index(psi) if lam > 16 else None
                sub = f"{v}_lam{lam:02d}" if lam <= 16 else f"{v}_lam{lam}_c{c}"
                sf = _surface("ARC", sub, psi)
                if sf:
                    x.append(lam)
                    y.append(abs(sf[1] - sf[0]) / abs(sf[0]))
            ax.plot(x, y, marker=mk, color=ORBIT, ms=4, label=f"ψ = {psi:.2f}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\lambda$")
        ax.set_title(title, loc="right")
        panel(ax, letter)
    axs[0].set_ylabel(r"$|T_\mathrm{forb} - T_\mathrm{fgar}|/|T_\mathrm{fgar}|$")
    axs[1].legend(frameon=False, fontsize=9)
    save(fig, DEVICES["ARC"], "fig_m4_nu")


def fig_m4_inner():
    """SPARC λ = 1, published q: forb on the six innermost surfaces with three treatments below the response's first ψ."""
    fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
    base = rows(DEVICES["SPARC"] / "B_lam01")[:8]
    psi = [float(r["psi"]) for r in base]
    ax.plot(psi, [float(r["envelope"]) * float(r["fgar_re"]) for r in base], color=SURFACE, marker="o", ms=4, label="fgar")
    ax.plot(psi, [float(r["forb_re"]) for r in base], color=ORBIT, marker="o", ms=4, label="forb, cubic")
    for sub, ls, lab in (("Iflat_lam01", "--", "forb, flat"), ("Izero_lam01", ":", "forb, zero")):
        R = rows(DEVICES["SPARC"] / sub)
        ax.plot([float(r["psi"]) for r in R], [float(r["forb_re"]) for r in R], color=ORBIT, ls=ls, marker=".", label=lab)
    m = meta("SPARC", 1)
    ax.axvline(float(m["psilow_response"]), color=SURFACE, lw=0.8)
    ax.set_xlabel(r"$\psi_N$")
    ax.set_ylabel(r"$dT/d\psi_N$ (N m)")
    ax.legend(frameon=False, fontsize=9)
    save(fig, DEVICES["SPARC"], "fig_m4_inner")


def fig_m4_nodes():
    """ARC λ = 16, ν/λ: dense fgar sweep with the 39 Gauss nodes and the dense forb, on ψ ≥ 0.55."""
    fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
    d = np.genfromtxt(DEVICES["ARC"] / "sweep" / "fgar_lam16_nu0.0625.csv", delimiter=",", names=True)
    k = d["psi"] > 0.55
    ax.plot(d["psi"][k], d["envelope"][k] * d["fgar_re"][k], color=SURFACE, lw=1, label="fgar, 600 points")
    R = [r for r in rows(DEVICES["ARC"] / "B_lam16") if float(r["psi"]) > 0.55]
    ax.plot([float(r["psi"]) for r in R], [float(r["forb_re"]) for r in R], "o", fillstyle="none", color=ORBIT, ms=4, label="forb, Gauss nodes")
    R = [r for r in rows(DEVICES["ARC"] / "DB_lam16") if float(r["psi"]) > 0.55]
    ax.plot([float(r["psi"]) for r in R], [float(r["forb_re"]) for r in R], "-", color=ORBIT, lw=0.8, label="forb, dense")
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_xlabel(r"$\psi_N$")
    ax.set_ylabel(r"$dT/d\psi_N$ (N m)")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    save(fig, DEVICES["ARC"], "fig_m4_nodes")


def fig_m4_defect():
    """ARC λ = 16, ψ = 0.99275: (a) |δJ₀|² samples of one segment and a cubic through them; (b) the ℓ = 0 pitch kernel."""
    from scipy.interpolate import CubicSpline

    E = DEVICES["ARC"] / "edge_diag"
    fig, axs = plt.subplots(2, 1, figsize=(SINGLE, 3.8))
    R = sorted((r for r in csv.DictReader(open(E / "lam16_exact.csv")) if r["il"] == "100" and float(r["ub"]) < 3), key=lambda r: float(r["u"]))
    u = np.array([float(r["u"]) for r in R])
    y = np.array([float(r["dJdJmax"]) for r in R])
    t = np.linspace(u[0], u[-1], 4001)
    axs[0].semilogy(t, np.abs(CubicSpline(u, y)(t)), color=SURFACE, lw=1, label="cubic")
    axs[0].semilogy(u, y, ".", color=ORBIT, ms=4, label="samples")
    axs[0].set_xlim(1.6, 2.16)
    axs[0].set_xlabel(r"$u = \sqrt{E/T}$")
    axs[0].set_ylabel(r"$|\delta J|^2$")
    axs[0].legend(frameon=False, fontsize=9)
    panel(axs[0], "a")
    for f, ls, lab in (("kernel_exact.csv", "-", "before"), ("kernel_near.csv", "--", "neighbour")):
        d = np.genfromtxt(E / f, delimiter=",", names=True)
        axs[1].plot(d["lambda"], np.abs(d["re"]), ls, color=ORBIT if ls == "-" else SURFACE, lw=1, label=lab)
    axs[1].set_yscale("log")
    axs[1].set_xlabel(r"pitch $\lambda$")
    axs[1].set_ylabel(r"$|K_0(\lambda)|$")
    axs[1].legend(frameon=False, fontsize=9)
    panel(axs[1], "b")
    save(fig, DEVICES["ARC"], "fig_m4_defect")


def fig_m4_shift():
    """ARC published q, fixed ν, dense: fgar and forb torque density across the ψ ≈ 0.80–0.84 resonance step, λ = 2, 4, 8, 16."""
    fig, axs = plt.subplots(4, 1, figsize=(SINGLE, 4.6), sharex=True)
    for ax, lam, letter in zip(axs, (2, 4, 8, 16), "abcd"):
        R = [r for r in rows(run_dir("ARC", "A", lam)) if 0.78 < float(r["psi"]) < 0.85]
        psi = [float(r["psi"]) for r in R]
        ax.plot(psi, [float(r["envelope"]) * float(r["fgar_re"]) * 1e4 for r in R], color=SURFACE, marker=".", label="fgar")
        ax.plot(psi, [float(r["forb_re"]) * 1e4 for r in R], color=ORBIT, marker=".", label="forb")
        ax.text(0.03, 0.8, rf"$\lambda = {lam}$", transform=ax.transAxes)
        panel(ax, letter)
    axs[0].legend(frameon=False, fontsize=9, loc="upper right")
    axs[-1].set_xlabel(r"$\psi_N$")
    fig.supylabel(r"$dT/d\psi_N$ ($10^{-4}$ N m)")
    save(fig, DEVICES["ARC"], "fig_m4_shift")


def fig_m4_q0():
    """ARC λ = 1, 600-point fgar alpha torque density for the published q (response from q = 1.02) and q₀ = 1.05, 1.1, 1.2."""
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    for f, color, ls, lab in (("lam01", SURFACE, "-", "published"), ("q105", ORBIT, "--", "1.05"), ("q110", ORBIT, "-", "1.1"),
                              ("q120", ORBIT, ":", "1.2")):
        d = np.genfromtxt(DEVICES["ARC"] / "sweep" / f"fgar_{f}_nu1.csv", delimiter=",", names=True)
        ax.plot(d["psi"], d["envelope"] * d["fgar_re"], color=color, ls=ls, lw=1, label=lab)
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_xlabel(r"$\psi_N$")
    ax.set_ylabel(r"$dT/d\psi_N$ (N m)")
    ax.legend(frameon=False, fontsize=9, title=r"$q_0$", title_fontsize=9, loc="lower left")
    save(fig, DEVICES["ARC"], "fig_m4_q0")


if __name__ == "__main__":
    names = sys.argv[1:] or ["table", "fig_alpha_scan", "fig_m4_coils", "fig_m4_nu", "fig_m4_inner", "fig_m4_nodes"]
    for name in names:
        globals()[name]()
