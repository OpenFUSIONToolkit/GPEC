#!/usr/bin/env python3
# plot_deltacoil_psihigh_panels.py -- per-surface paneled view of the psihigh truncation scan, in the
# "Slide 7" edge-truncation style: ONE subplot per rational surface (its own y-scale), with
# Riccati/STRIDE (orange, solid, circles) vs Galerkin/RDCON (blue, dashed, squares) overlaid, gray
# dotted vertical lines at each rational-surface location, and all panels combined into a single PNG.
# Reads results/deltacoil_psihigh_rg_<case>.csv from deltacoil_psihigh_riccati_galerkin.jl.
#
# Usage: plot_deltacoil_psihigh_panels.py <csv> [CASE label] [--qmax=Q] [--quantity=norm|cos]

import sys, os, csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

argv = [a for a in sys.argv[1:]]
QMAX = None; SUFFIX = ""; QUANT = "norm"; PDFDIR = None
_kept = []
for a in argv:
    if a.startswith("--qmax="):
        QMAX = float(a.split("=", 1)[1]); SUFFIX = "_core"
    elif a.startswith("--quantity="):
        QUANT = a.split("=", 1)[1]
    elif a.startswith("--pdfdir="):
        PDFDIR = os.path.expanduser(a.split("=", 1)[1])
    else:
        _kept.append(a)
argv = _kept
CSVP = os.path.abspath(argv[0])
here = os.path.dirname(os.path.abspath(__file__))
_tag = os.path.splitext(os.path.basename(CSVP))[0].replace("deltacoil_psihigh_rg_", "")
CASE = argv[1] if len(argv) > 1 else {
    "LARresistivematchtest": "LAR (large-aspect-ratio)",
    "DIIIDlikegalresistivepeexample": "DIII-D-like",
}.get(_tag, _tag)

with open(CSVP) as f:
    rows = list(csv.reader(f))
header = rows[0]
qs = sorted({c.split("_q")[1] for c in header if c.startswith("norm_ric_q")}, key=float)
if QMAX is not None:
    qs = [q for q in qs if float(q) <= QMAX + 1e-9]
idx = {c: i for i, c in enumerate(header)}
psihigh = np.array([float(r[0]) for r in rows[1:] if r and r[0] != ""])

def col(name):
    j = idx.get(name); out = []
    for r in rows[1:]:
        if not r or r[0] == "":
            continue
        v = r[j] if (j is not None and j < len(r)) else ""
        try:
            out.append(float(v) if v not in ("", "FAIL", "NaN") else np.nan)
        except ValueError:
            out.append(np.nan)
    return np.array(out)

# entry psihigh per surface (approx psi_s) = smallest psihigh at which the surface has data
def entry_ph(q):
    y = col(f"norm_ric_q{q}"); m = np.isfinite(y)
    return psihigh[m].min() if m.any() else np.nan
PSI_S = {q: entry_ph(q) for q in qs}

ORANGE = "#ff7f0e"; BLUE = "#1f77b4"
prefix, ylabel, ylog = ("norm", "|delta_coil|", True) if QUANT == "norm" \
    else ("cos", "cosine sim. with nominal", False)

# common x-range (0.9 -> 0.9999) for every panel so surfaces are compared on the same axis
GX0 = psihigh.min(); GXMAX = psihigh.max()
def edge_log_x(ax):
    fwd = lambda p: -np.log10(np.clip(1.0 - np.asarray(p, float), 1e-9, None))
    inv = lambda t: 1.0 - 10.0 ** (-np.asarray(t, float))
    ax.set_xscale("function", functions=(fwd, inv))
    cand = np.array([0.9, 0.99, 0.999, 0.9999])
    tk = cand[(cand >= GX0 - 1e-9) & (cand <= GXMAX + 1e-9)]
    ax.set_xticks(tk); ax.set_xticklabels([f"{t:g}" for t in tk]); ax.minorticks_off()
    ax.set_xlim(GX0 - (1 - GX0) * 0.06, GXMAX + (1 - GXMAX) * 0.4)

N = len(qs)
# balanced grid: a single row for <=3 surfaces, otherwise two rows (2x2 for 4, 2x4 for 7-8), so
# panels stay roughly square and legible rather than a long thin strip
if N <= 3:
    ncols = N; nrows = 1
else:
    ncols = math.ceil(N / 2); nrows = 2
fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 4.1 * nrows), squeeze=False)

for i, q in enumerate(qs):
    ax = axes[i // ncols][i % ncols]
    yr = col(f"{prefix}_ric_q{q}"); yg = col(f"{prefix}_gal_q{q}")
    mr = np.isfinite(yr); mg = np.isfinite(yg)
    ax.plot(psihigh[mr], yr[mr], "o-", color=ORANGE, lw=1.6, ms=4)
    ax.plot(psihigh[mg], yg[mg], "s--", color=BLUE, lw=1.5, ms=4)
    if ylog:
        ax.set_yscale("log")
    else:
        ax.set_ylim(-0.05, 1.06)
    edge_log_x(ax)
    # gray dotted vertical lines only for rational surfaces that ENTER within the plotted psihigh window.
    # A surface already present at the leftmost psihigh (entry == psihigh.min()) lies interior to the
    # window (psi_s < psihigh_min), i.e. off the left edge, so drawing it at the boundary would falsely
    # imply a surface at psihigh_min. Those are skipped (e.g. LAR q=2,3 both sit at psi<0.9).
    ytop = ax.get_ylim()[1]
    for qq in qs:
        ps = PSI_S[qq]
        if np.isfinite(ps) and ps > GX0 + 1e-9:
            ax.axvline(ps, ls=":", color="#999", lw=0.9)
            ax.text(ps, ytop, f" q={float(qq):g}", color="#666", fontsize=7, rotation=90, va="top", ha="left")
    ax.set_title(f"q = {float(q):g}/1", fontsize=11)
    ax.set_xlabel(r"$\psi_{high}$", fontsize=10)
    if i % ncols == 0:
        ax.set_ylabel(ylabel + ("  (log)" if ylog else ""), fontsize=10)
    ax.grid(alpha=0.25, which="both")
    if i == 0:
        handles = [Line2D([0], [0], color=ORANGE, ls="-", marker="o", ms=4, label="Riccati (STRIDE)"),
                   Line2D([0], [0], color=BLUE, ls="--", marker="s", ms=4, label="Galerkin (RDCON)"),
                   Line2D([0], [0], color="#999", ls=":", label="rational surface")]
        ax.legend(handles=handles, fontsize=8, loc="best", framealpha=0.9)

for j in range(N, nrows * ncols):
    axes[j // ncols][j % ncols].axis("off")

qtxt = "|delta_coil|" if QUANT == "norm" else "delta_coil shape cosine"
fig.suptitle(f"Edge truncation: {qtxt} vs psihigh (log to psi=1, no wall)  |  {CASE}, n=1\n"
             "per rational surface: Riccati (STRIDE, orange) vs Galerkin (RDCON, blue)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
stem = os.path.splitext(os.path.basename(CSVP))[0] + SUFFIX + f"_panels_{prefix}"
out = os.path.abspath(os.path.join(here, "..", "figures", stem + ".png"))
fig.savefig(out, dpi=150); print("wrote:", out)

# optional vector PDF with a human-readable, title-based filename for easy navigation
if PDFDIR is not None:
    os.makedirs(PDFDIR, exist_ok=True)
    case_short = "DIII-D" if "DIII" in CASE else ("LAR" if "LAR" in CASE else CASE)
    qrange = f"q{int(float(qs[0]))}-{int(float(qs[-1]))}"
    qtxt_file = "norm (delta_coil magnitude)" if QUANT == "norm" else "shape cosine"
    fname = f"{case_short} - delta_coil edge truncation - {qrange} - Riccati vs Galerkin - {qtxt_file}.pdf"
    pdfout = os.path.join(PDFDIR, fname)
    fig.savefig(pdfout); print("wrote PDF:", pdfout)
