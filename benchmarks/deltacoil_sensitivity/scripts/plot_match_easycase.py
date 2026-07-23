#!/usr/bin/env python3
# Task 4 (matching easy case) validation figures. Reads the three LAR resistive-match runs and
# produces:
#   match_easycase_cout.png   - matched |cout| per surface (STRIDE vs galerkin) + ideal=0, and the
#                               STRIDE-vs-galerkin cout correlation per surface.
#   match_easycase_prec.png   - solve residual and reconnected-flux self-consistency (log scale).
# Usage: plot_match_easycase.py <driven.h5> <ideal.h5> <single.h5> <out_figdir>
import sys, numpy as np, h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DRIVEN, IDEAL, SINGLE, FIGDIR = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

def cread(fid, path):
    """Read a Julia HDF5 complex/real dataset as a numpy array, restored to Julia (column-major)
    orientation: h5py exposes 2D datasets transposed, so we transpose back."""
    if path not in fid:
        return None
    a = fid[path][()]
    if a.dtype.names and set(a.dtype.names) >= {"r", "i"}:
        a = a["r"] + 1j * a["i"]
    if a.ndim == 2:
        a = a.T
    return a

def surf_metrics(h5):
    """Per-surface matched-cout norms + STRIDE-vs-galerkin correlation, plus residuals/self-consistency."""
    with h5py.File(h5, "r") as fid:
        scout = cread(fid, "singular/resonant_match/cout")
        cin   = cread(fid, "singular/resonant_match/cin")
        dr    = cread(fid, "singular/resonant_match/deltar")
        flux  = cread(fid, "singular/resonant_match/reconnected_flux")
        sres  = float(np.array(fid["singular/resonant_match/residual"][()]))
        gcout = cread(fid, "galerkin/match/cout")
        gres  = float(np.array(fid["galerkin/match/residual"][()]))
    msing = dr.shape[0]
    # per-surface (two rows per surface) norms and complex correlation
    s_norm, g_norm, corr = [], [], []
    for i in range(msing):
        rows = slice(2 * i, 2 * i + 2)
        s = np.asarray(scout[rows]).ravel().astype(complex)
        g = np.asarray(gcout[rows]).ravel().astype(complex)
        s_norm.append(np.linalg.norm(s))
        g_norm.append(np.linalg.norm(g))
        denom = np.linalg.norm(s) * np.linalg.norm(g) + 1e-300
        corr.append(abs(np.vdot(s, g)) / denom)
    # reconnected-flux self-consistency: reconnected_flux ?= -InnerBlock*cin
    inner = np.zeros_like(flux, dtype=complex)
    for i in range(msing):
        d1, d2 = dr[i, 0], dr[i, 1]
        r1, r2 = 2 * i, 2 * i + 1
        inner[r1] = -(-d1 * cin[r1] + d2 * cin[r2])
        inner[r2] = -(-d1 * cin[r1] - d2 * cin[r2])
    fden = np.linalg.norm(flux) + 1e-300
    selfc = np.linalg.norm(inner - flux) / fden
    return dict(msing=msing, s_norm=s_norm, g_norm=g_norm, corr=corr,
               sres=sres, gres=gres, selfc=selfc,
               scout_tot=float(np.linalg.norm(scout.astype(complex))),
               gcout_tot=float(np.linalg.norm(gcout.astype(complex))))

d = surf_metrics(DRIVEN)
si = surf_metrics(SINGLE)
with h5py.File(IDEAL, "r") as fid:
    ideal_scout = float(np.linalg.norm(cread(fid, "singular/resonant_match/cout").astype(complex)))
    ideal_gcout = float(np.linalg.norm(cread(fid, "galerkin/match/cout").astype(complex)))
    ideal_sres  = float(np.array(fid["singular/resonant_match/residual"][()]))

BLUE, ORANGE, GREY = "#1f5fbf", "#e07b1a", "#666666"

# ---------- Figure 1: matched cout (per surface) + correlation ----------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.2))

# Panel A: per-surface matched |cout|, STRIDE vs galerkin, + ideal=0
labels = ["driven q=2", "driven q=3", "single q=2", "ideal q=2", "ideal q=3"]
stride_vals = d["s_norm"] + si["s_norm"] + [ideal_scout, ideal_scout]
galerk_vals = d["g_norm"] + si["g_norm"] + [ideal_gcout, ideal_gcout]
x = np.arange(len(labels)); w = 0.38
axA.bar(x - w/2, stride_vals, w, label="STRIDE (Riccati outer)", color=BLUE)
axA.bar(x + w/2, galerk_vals, w, label="galerkin (RDCON outer)", color=ORANGE)
axA.set_xticks(x); axA.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
axA.set_ylabel("matched  |cout|  per surface")
axA.set_title("(a) Matched outer coefficient: driven vs ideal", fontsize=10)
axA.legend(fontsize=8, loc="upper right")
axA.annotate("ideal limit:\ncout = 0 exactly\n(perfect shielding)", xy=(3.5, 0.0),
             xytext=(2.7, max(stride_vals)*0.55), fontsize=8, color=GREY,
             ha="left", arrowprops=dict(arrowstyle="->", color=GREY))

# Panel B: STRIDE-vs-galerkin cout correlation per surface
clab = ["driven q=2\n(inner, coupled)", "driven q=3\n(edge, coupled)", "single q=2\n(isolated pair)"]
cvals = [d["corr"][0], d["corr"][1], si["corr"][0]]
cx = np.arange(len(clab))
bars = axB.bar(cx, cvals, 0.5, color=[GREY, GREY, BLUE])
axB.axhline(1.0, color="k", lw=0.7, ls=":")
axB.set_ylim(0, 1.08); axB.set_xticks(cx); axB.set_xticklabels(clab, fontsize=8.3)
axB.set_ylabel("|<STRIDE, galerkin>| / (norms)")
axB.set_title("(b) STRIDE vs galerkin matched-cout correlation", fontsize=10)
for b, v in zip(bars, cvals):
    axB.text(b.get_x()+b.get_width()/2, v+0.015, f"{v:.4f}", ha="center", fontsize=8.5)

fig.tight_layout()
f1 = f"{FIGDIR}/match_easycase_cout.png"
fig.savefig(f1, dpi=150); print("WROTE:", f1)

# ---------- Figure 2: residual + self-consistency (log) ----------
fig2, ax = plt.subplots(figsize=(8.6, 4.0))
groups = ["driven", "single-surface", "ideal"]
sres = [d["sres"], si["sres"], ideal_sres]
gres = [d["gres"], si["gres"], 0.0]
selfc = [d["selfc"], si["selfc"], np.nan]  # ideal self-consistency is N/A (no inner layer)
floor = 1e-18
def clamp(v): return [max(x, floor) if np.isfinite(x) else np.nan for x in v]
gx = np.arange(len(groups)); w = 0.26
ax.bar(gx - w, clamp(sres), w, label="STRIDE solve residual", color=BLUE)
ax.bar(gx,      clamp(gres), w, label="galerkin solve residual", color=ORANGE)
ax.bar(gx + w, clamp(selfc), w, label="reconnected-flux self-consistency", color="#2a9d4a")
ax.axhline(1e-16, color="k", lw=0.7, ls=":"); ax.text(2.35, 1.3e-16, "1e-16", fontsize=8, color=GREY)
ax.set_yscale("log"); ax.set_ylim(1e-18, 1e-10)
ax.set_xticks(gx); ax.set_xticklabels(groups)
ax.set_ylabel("relative error  (log scale)")
ax.set_title("Match solve residual and reconnected-flux self-consistency", fontsize=10.5)
ax.legend(fontsize=8, loc="upper left")
ax.text(1.62, 3e-18, "ideal: cout=0, no inner layer\n(self-consistency N/A)", fontsize=7.6, color=GREY)
fig2.tight_layout()
f2 = f"{FIGDIR}/match_easycase_prec.png"
fig2.savefig(f2, dpi=150); print("WROTE:", f2)

# ---------- console summary for the PDF text ----------
print("\n=== SUMMARY ===")
print(f"driven : msing={d['msing']} sres={d['sres']:.3e} gres={d['gres']:.3e} selfc={d['selfc']:.3e} "
      f"corr={['%.4f'%c for c in d['corr']]}")
print(f"single : msing={si['msing']} sres={si['sres']:.3e} gres={si['gres']:.3e} selfc={si['selfc']:.3e} "
      f"corr={['%.4f'%c for c in si['corr']]}")
print(f"ideal  : ||cout|| STRIDE={ideal_scout:.3e} galerkin={ideal_gcout:.3e} sres={ideal_sres:.3e}")
