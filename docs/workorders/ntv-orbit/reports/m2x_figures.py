"""M2 explainer figures: what the charge scan changes, the three grids, the separatrix-layer fill and the proposed fix.

Reads results/m2_dev/explain_*.csv (m2x_diag.jl), results/m2_dev/layer_diag_psi96.csv and results/m2_zow/zow_scan.csv;
writes results/m2_zow/fig_m2x_{charge,grids,layer,dj,fix}.{svg,png}.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[5]
DEV, OUT = ROOT / "results" / "m2_dev", ROOT / "results" / "m2_zow"
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "axes.linewidth": 0.6})
BLUE, ORANGE, AQUA, YELLOW, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#8a8984"
CLASS = {1: ("trapped", BLUE), 2: ("passing", ORANGE), 3: ("lost (counts zero)", "#3b3b38"), 4: ("hit axis grid", "#3b3b38"),
         0: ("unclosed", YELLOW), -1: ("no orbit (w² < 0)", "#ffffff")}
LAYER_FACTOR = 10.0


def read(path):
    d = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return {k: d[k] for k in d.dtype.names}


def save(fig, name):
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_charge():
    o, s = read(DEV / "explain_orbits.csv"), read(OUT / "zow_scan.csv")
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    shades = {1: INK, 4: "#52514e", 16: "#8a8984", 64: "#b9b8b3"}
    for z, c in shades.items():
        m = (o["case"] == "mid") & (o["zi"] == z)
        ax[0].plot(o["psi"][m], o["Z"][m], color=c, lw=1.0 if z == 1 else 0.8, label=f"Z = {z}")
    ax[0].axvline(0.5, color=MUTED, lw=0.5, ls=":")
    ax[0].set_xlabel("ψ_N along the orbit")
    ax[0].set_ylabel("height z (m)")
    ax[0].set_title("(a) one trapped D orbit, x = 1, from ψ = 0.5", fontsize=7.5)
    ax[0].legend(fontsize=6, frameon=False, loc="center right")
    zi, w = s["zi"], s["width_over_a"]
    ax[1].loglog(zi, w, "o-", color=INK, ms=3.5, lw=0.8)
    ax[1].loglog(zi, w[0] / zi, ":", color=MUTED, lw=0.7, label="∝ 1/Z")
    ax[1].annotate("physical\n(thermal D)", (zi[0], w[0]), textcoords="offset points", xytext=(6, -14), fontsize=6)
    ax[1].set_xlabel("charge number Z (mass fixed at 2 m_p)")
    ax[1].set_ylabel("orbit width Δ_b / a")
    ax[1].set_title("(b) width scales as 1/Z", fontsize=7.5)
    ax[1].legend(fontsize=6, frameon=False)
    ax[2].semilogx(zi, s["Tf_re"], "o-", color=BLUE, ms=3.5, lw=0.8, label="Re T_fgar (torque)")
    ax[2].semilogx(zi, s["Tf_im"], "s--", color=ORANGE, ms=3, lw=0.8, label="Im T_fgar (2n δW_K)")
    ax[2].axhline(0, color=MUTED, lw=0.5)
    ax[2].set_xlabel("charge number Z")
    ax[2].set_ylabel("surface-path T (N·m)")
    ax[2].set_title("(c) the reference changes too", fontsize=7.5)
    ax[2].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    save(fig, "fig_m2x_charge")


def fig_grids():
    d, p = read(DEV / "layer_diag_psi96.csv"), read(DEV / "explain_psinodes.csv")
    b = read(DEV / "explain_boundary.csv")
    lam = np.unique(d["lambda"])
    tr = np.array([d["trapped"][d["lambda"] == x][0] for x in lam])
    m = (b["psi"] == 0.96) & (b["zi"] == 1)
    ltp, lmax = b["lambda_tp"][m][0], b["lambda_max"][m][0]
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
    mid, dl = (lam[1:] + lam[:-1]) / 2, np.diff(lam)
    ax[0].semilogy(mid, dl, "-", color=MUTED, lw=0.6)
    ax[0].semilogy(mid[tr[1:] == 0], dl[tr[1:] == 0], "o", color=ORANGE, ms=2.2, label="passing nodes")
    ax[0].semilogy(mid[tr[:-1] == 1], dl[tr[:-1] == 1], "o", color=BLUE, ms=2.2, label="trapped nodes")
    ax[0].axvline(ltp, color=INK, lw=0.6)
    ax[0].annotate("λtp = B₀/Bmax", (ltp, dl.min()), textcoords="offset points", xytext=(4, 2), fontsize=6)
    ax[0].axvline(lmax, color=MUTED, lw=0.6, ls=":")
    ax[0].set_xlabel("pitch λ = μB₀/E")
    ax[0].set_ylabel("node spacing Δλ")
    ax[0].set_title(f"(a) pitch grid, ψ = 0.96: {len(lam)} nodes", fontsize=7.5)
    ax[0].legend(fontsize=6, frameon=False, loc="lower left")
    us = np.unique(d["u"])
    uu = np.linspace(0, 6.2, 400)
    wgt = 2 * uu**6 * np.exp(-(uu**2))
    ax[1].plot(uu, wgt / wgt.max(), color=MUTED, lw=0.8, label="weight x^{5/2} e^{−x} dx, in u")
    ax[1].vlines(us, 0, 1.05, color=INK, lw=0.6)
    ax[1].plot([], [], "|", color=INK, label=f"{len(us)} table nodes u_k")
    ax[1].set_xlabel("u = √x = √(E/T)")
    ax[1].set_ylabel("normalized weight")
    ax[1].set_title("(b) energy nodes u_k = 6((k−½)/12)^{3/2}", fontsize=7.5)
    ax[1].legend(fontsize=6, frameon=False, loc="upper right")
    for i, (key, lab, y) in enumerate((("gl3", "3-point (36 nodes)", 1), ("gl6", "6-point (72 nodes)", 0))):
        x = p["psi"][p["set"] == key]
        ax[2].plot(x, np.full_like(x, y), "|", color=INK, ms=9, mew=0.8)
        ax[2].annotate(lab, (0.08, y + 0.18), fontsize=6)
    for x in p["psi"][p["set"] == "rational"]:
        ax[2].axvline(x, color=ORANGE, lw=0.6, ls="--")
    for x in p["psi"][p["set"] == "panel"]:
        ax[2].plot([x, x], [-0.4, 1.4], color=MUTED, lw=0.4)
    ax[2].set_ylim(-0.6, 1.8)
    ax[2].set_yticks([])
    ax[2].set_xlabel("ψ_N")
    ax[2].set_title("(c) ψ nodes (Gauss–Legendre per panel)", fontsize=7.5)
    ax[2].legend(handles=[Line2D([], [], color=ORANGE, ls="--", lw=0.6, label="rational q"),
                          Line2D([], [], color=MUTED, lw=0.4, label="panel edges")], fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_m2x_grids")


def class_map(ax, c, b, psi, zi, slot, band=True, grid="table", nodes=True):
    m = (c["psi"] == psi) & (c["zi"] == zi) & (c["slot"] == slot)
    mb = (b["psi"] == psi) & (b["zi"] == zi)
    ltp, lmax = b["lambda_tp"][mb][0], b["lambda_max"][mb][0]
    us = np.unique(c["u"][m])
    edges = np.concatenate([[0], (us[1:] + us[:-1]) / 2, [us[-1] + (us[-1] - us[-2]) / 2]])
    fm = m & (c["grid"] == "fine")
    lf = np.unique(c["lambda"][fm])
    dlf = lf[1] - lf[0]
    for k, u in enumerate(us):
        r = fm & (c["u"] == u)
        order = np.argsort(c["lambda"][r])
        for x, st in zip(c["lambda"][r][order], c["status"][r][order]):
            ax.add_patch(plt.Rectangle((x - dlf / 2, edges[k]), dlf, edges[k + 1] - edges[k], color=CLASS[int(st)][1], lw=0))
    shift = np.array([np.nanmax(np.abs(b["lambda_c"][mb & (b["k"] == k + 1)] - ltp)) for k in range(len(us))])
    if band:
        for k in range(len(us)):
            wdt = LAYER_FACTOR * shift[k]
            ax.add_patch(plt.Rectangle((ltp - wdt, edges[k]), 2 * wdt, edges[k + 1] - edges[k], facecolor=(1, 1, 1, 0.55),
                                       edgecolor=(0, 0, 0, 0.35), hatch="////", lw=0))
            ax.plot([ltp - wdt, ltp - wdt], [edges[k], edges[k + 1]], color=INK, lw=0.6)
            ax.plot([ltp + wdt, ltp + wdt], [edges[k], edges[k + 1]], color=INK, lw=0.6)
    lc = np.array([b["lambda_c"][mb & (b["k"] == k + 1) & (b["slot"] == slot)][0] for k in range(len(us))])
    ok = np.isfinite(lc)
    ax.plot(lc[ok], us[ok], "o-", color="white", mec=INK, mew=0.7, ms=3.5, lw=1.0, zorder=5)
    if nodes:
        tm = m & (c["grid"] == "table")
        lt = np.unique(c["lambda"][tm])
        ax.plot(lt, np.full_like(lt, -0.12), "|", color=INK, ms=5, mew=0.5, clip_on=False)
    ax.axvline(ltp, color=INK, lw=0.5, ls=":")
    ax.set_xlim(0, lmax)
    ax.set_ylim(0, edges[-1])
    ax.set_xlabel("pitch λ")
    ax.set_ylabel("energy node u = √x")
    return ltp, lmax, lc, us, shift


def legend_classes(fig, extra=()):
    h = [Patch(color=CLASS[k][1], label=CLASS[k][0]) for k in (1, 2, 3, 0)]
    h += [Line2D([], [], marker="o", color="white", mec=INK, label="λc: orbit's own trapped–passing edge"),
          Line2D([], [], color=INK, ls=":", lw=0.5, label="λtp: surface edge")] + list(extra)
    fig.legend(handles=h, fontsize=6, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.07))


def fig_layer():
    c, b = read(DEV / "explain_classes.csv"), read(DEV / "explain_boundary.csv")
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True)
    for a, (psi, zi, slot, t) in zip(ax, ((0.96, 1, 1, "(a) ψ = 0.96, Z = 1, σ = +1"), (0.96, 1, 2, "(b) ψ = 0.96, Z = 1, σ = −1"),
                                          (0.96, 16, 1, "(c) ψ = 0.96, Z = 16, σ = +1"))):
        class_map(a, c, b, psi, zi, slot)
        a.set_title(t, fontsize=7.5)
    for a in ax[1:]:
        a.set_ylabel("")
    legend_classes(fig, [Patch(facecolor="white", edgecolor=INK, hatch="////", label="layer fill: surface value written")])
    fig.tight_layout()
    save(fig, "fig_m2x_layer")


def fig_dj():
    d, b, n = read(DEV / "layer_diag_psi96.csv"), read(DEV / "explain_boundary.csv"), read(DEV / "explain_dj_lc.csv")
    mb = (b["psi"] == 0.96) & (b["zi"] == 1)
    ltp = b["lambda_tp"][mb][0]
    ks = sorted(set(n["k"].tolist()))
    fig, ax = plt.subplots(1, len(ks), figsize=(7.4, 2.7), squeeze=False)
    for a, k in zip(ax[0], ks):
        slot = int(n["slot"][n["k"] == k][0])
        # σ = −1: trapped entries sit in slot 2; its passing orbits run in −θ, so they sit in slot 1
        r = (d["k"] == k) & (((d["trapped"] == 1) & (d["slot"] == slot)) | ((d["trapped"] == 0) & (d["slot"] == 1)))
        lam = d["lambda"][r]
        o = np.argsort(lam)
        lam, surf, v, dj = lam[o], d["surf"][r][o], d["valid10"][r][o], d["dj10"][r][o]
        u = d["u"][r][0]
        shift = np.nanmax(np.abs(b["lambda_c"][mb & (b["k"] == k)] - ltp))
        wdt = LAYER_FACTOR * shift
        a.axvspan(ltp - wdt, ltp + wdt, facecolor="white", edgecolor=MUTED, hatch="////", lw=0.4, label="layer (surface value used)")
        a.plot(lam, surf, "--", color=MUTED, lw=0.8, label="surface path $\Sigma_\ell|\delta J_\ell|^2$")
        a.plot(lam[v == 1], dj[v == 1], "o", color=INK, ms=2.5, label="table: orbit value")
        q = (n["k"] == k) & (n["slot"] == slot)
        o2 = np.argsort(n["lambda"][q])
        ln, djn, st = n["lambda"][q][o2], n["dj"][q][o2], n["status"][q][o2]
        lc = n["lambda_c"][q][0]
        for cls_, col in ((1, BLUE), (2, ORANGE)):
            mm = st == cls_
            a.plot(ln[mm], djn[mm], ".", color=col, ms=4, label=f"new points at λc: {CLASS[cls_][0]}")
        a.plot(ln[st == 0], np.zeros((st == 0).sum()), "d", color=YELLOW, ms=3, label="new points: unclosed within τ_max")
        a.axvline(lc, color=INK, lw=0.6, label="λc")
        a.axvline(ltp, color=INK, lw=0.5, ls=":", label="λtp")
        lo, hi = min(ln.min(), ltp - 1.5 * wdt), max(ln.max(), ltp + 1.5 * wdt)
        a.set_xlim(lo, hi)
        sel = (lam > lo) & (lam < hi)
        top = max(np.nanmax(surf[sel]) if sel.any() else 0, np.nanmax(djn) if djn.size else 0, np.nanmax(dj[sel]) if sel.any() else 0)
        a.set_ylim(-0.05 * top, 1.15 * top)
        a.set_xlabel("pitch λ")
        a.set_ylabel("$\Sigma_\ell|\delta J_\ell|^2$ (table units)")
        a.set_title(f"u = √x = {u:.2f}, σ = {'+1' if slot == 1 else '−1'}, ψ = 0.96, Z = 1", fontsize=7.5)
    h, l = ax[0][0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "fig_m2x_dj")


def row_edges(lf, st, lc):
    """Class edges along one energy row of the fine scan; the trapped–passing edge is the bisected λc where found."""
    keep = st >= 0
    lf, st = lf[keep], st[keep]
    edges = []
    for i in np.nonzero(st[1:] != st[:-1])[0]:
        e = 0.5 * (lf[i] + lf[i + 1])
        if {int(st[i]), int(st[i + 1])} == {1, 2} and np.isfinite(lc):
            e = lc
        edges.append(e)
    bounds = np.concatenate([[lf[0] - 0.5 * (lf[1] - lf[0])], edges, [lf[-1] + 0.5 * (lf[1] - lf[0])]])
    seg = [int(st[np.searchsorted(lf, 0.5 * (x + y))]) if np.searchsorted(lf, 0.5 * (x + y)) < len(lf) else int(st[-1])
           for x, y in zip(bounds[:-1], bounds[1:])]
    return bounds, seg


def fix_nodes(bounds, seg, total=64):
    """Nodes clustered at both ends of every closed-orbit segment; none on lost segments (they count zero)."""
    lens = np.array([y - x if c in (1, 2) else 0.0 for x, y, c in zip(bounds[:-1], bounds[1:], seg)])
    out = []
    for x, y, ln in zip(bounds[:-1], bounds[1:], lens):
        if ln == 0:
            continue
        m = max(4, int(round(total * ln / lens.sum())))
        t = np.linspace(0, 1, m + 2)[1:-1]
        out.append(x + (y - x) * (1 - np.cos(np.pi * t)) / 2)
    return np.concatenate(out) if out else np.array([])


def fig_fix():
    c, b = read(DEV / "explain_classes.csv"), read(DEV / "explain_boundary.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for a, title, new in ((ax[0], "(a) now: one grid for every row, clustered at λtp", False),
                          (ax[1], "(b) proposed: per-row grid, clustered at the row's class edges", True)):
        ltp, lmax, lc, us, _ = class_map(a, c, b, 0.96, 1, 1, band=not new, nodes=False)
        base = (c["psi"] == 0.96) & (c["zi"] == 1) & (c["slot"] == 1)
        lt = np.unique(c["lambda"][base & (c["grid"] == "table")])
        for k, u in enumerate(us):
            if new:
                r = base & (c["grid"] == "fine") & (c["u"] == u)
                o = np.argsort(c["lambda"][r])
                bounds, seg = row_edges(c["lambda"][r][o], c["status"][r][o], lc[k])
                lam = fix_nodes(bounds, seg)
            else:
                lam = lt
            a.plot(lam, np.full_like(lam, u), "|", color=INK, ms=4, mew=0.4)
        a.set_title(title, fontsize=7.5)
    ax[1].set_ylabel("")
    legend_classes(fig, [Line2D([], [], marker="|", ls="", color=INK, label="pitch nodes (64 per row in b)")])
    fig.tight_layout()
    save(fig, "fig_m2x_fix")


if __name__ == "__main__":
    fig_charge()
    fig_grids()
    fig_layer()
    fig_dj()
    fig_fix()
