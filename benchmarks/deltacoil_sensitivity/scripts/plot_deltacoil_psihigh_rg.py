#!/usr/bin/env python3
# plot_deltacoil_psihigh_rg.py -- plot the psihigh truncation scan with BOTH methods overlaid:
# Riccati/STRIDE (solid circles) and Galerkin/RDCON (dashed squares), one color per rational surface.
# Reads results/deltacoil_psihigh_rg_<case>.csv from deltacoil_psihigh_riccati_galerkin.jl and writes
# three figures per case:
#   <stem>_norm.png       |delta_coil| per surface vs psihigh (log-y)
#   <stem>_relchange.png  relative change vs the outermost psihigh (convergence view)
#   <stem>_metric.png     cosine shape metric vs the full-domain nominal (shape rotation view)
# x-axis is log distance from the edge (1 - psihigh); edge at right.
#
# Usage: python3 plot_deltacoil_psihigh_rg.py <deltacoil_psihigh_rg_<case>.csv> [CASE label]

import sys, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

# positional: CSV [CASE label]; optional --qmax=<q> keeps only surfaces with q <= qmax and appends
# "_core" to the output names (so the full-range figure is not overwritten). Used to keep the DIII-D
# comparison legible when near-edge surfaces (q>=6) diverge by many decades.
argv = [a for a in sys.argv[1:]]
QMAX = None; SUFFIX = ""
_kept = []
for a in argv:
    if a.startswith("--qmax="):
        QMAX = float(a.split("=", 1)[1]); SUFFIX = "_core"
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
# surfaces present, from the norm_ric_q<q> columns
qs = sorted({c.split("_q")[1] for c in header if c.startswith("norm_ric_q")}, key=float)
if QMAX is not None:
    qs = [q for q in qs if float(q) <= QMAX + 1e-9]
idx = {c: i for i, c in enumerate(header)}

def col(name):
    j = idx.get(name)
    out = []
    for r in rows[1:]:
        if not r or r[0] == "":
            continue
        v = r[j] if (j is not None and j < len(r)) else ""
        try:
            out.append(float(v)) if v not in ("", "FAIL", "NaN") else out.append(np.nan)
        except ValueError:
            out.append(np.nan)
    return np.array(out)

psihigh = np.array([float(r[0]) for r in rows[1:] if r and r[0] != ""])
# Riccati/STRIDE keeps the cool viridis palette; Galerkin/RDCON uses warm tones (orange -> dark red),
# so the two methods are distinguished by COLOR FAMILY (cool vs warm) and surface by shade within it.
colors_ric = plt.cm.viridis(np.linspace(0.08, 0.82, len(qs)))
colors_gal = plt.cm.YlOrRd(np.linspace(0.45, 0.95, len(qs)))

# x-axis: distance from edge (1 - psihigh) on a log scale, edge at right. Anchor to leftmost psihigh
# that has any data so empty near-axis points do not push the range.
def edge_log_x(ax, x0, xmax):
    fwd = lambda p: -np.log10(np.clip(1.0 - np.asarray(p, float), 1e-9, None))
    inv = lambda t: 1.0 - 10.0 ** (-np.asarray(t, float))
    ax.set_xscale("function", functions=(fwd, inv))
    cand = np.array([0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    tk = cand[(cand >= x0 - 1e-9) & (cand <= xmax + 1e-9)]
    ax.set_xticks(tk); ax.set_xticklabels([f"{t:g}" for t in tk]); ax.minorticks_off()
    ax.set_xlim(x0 - (1 - x0) * 0.06, xmax + (1 - xmax) * 0.4)

def method_legend(ax, loc="lower left", bbox=(1.015, 0.0)):
    # representative mid-palette swatches so the cool=Riccati / warm=Galerkin mapping is explicit;
    # default placement is OUTSIDE the axes (lower right) so it never covers data points
    handles = [Line2D([0], [0], color=plt.cm.viridis(0.45), ls="-", marker="o", ms=5, lw=2, label="Riccati (STRIDE), cool"),
               Line2D([0], [0], color=plt.cm.YlOrRd(0.75), ls="--", marker="s", ms=5, lw=2, label="Galerkin (RDCON), warm")]
    leg = ax.legend(handles=handles, fontsize=8.5, loc=loc, bbox_to_anchor=bbox,
                    framealpha=0.95, title="method (color family)", title_fontsize=8.5)
    ax.add_artist(leg)

def surface_legend(ax, loc="upper left", bbox=(1.015, 1.0)):
    # each surface shows BOTH its Riccati (cool) and Galerkin (warm) swatch as a paired handle;
    # default placement is OUTSIDE the axes (upper right) so all points stay visible
    handles = [(Line2D([0], [0], color=colors_ric[k], ls="-", marker="o", ms=5, lw=2),
                Line2D([0], [0], color=colors_gal[k], ls="--", marker="s", ms=5, lw=2)) for k in range(len(qs))]
    labels = [f"q = {q}" for q in qs]
    ax.legend(handles=handles, labels=labels, fontsize=8.5, loc=loc, bbox_to_anchor=bbox,
              title="rational surface  (Riccati | Galerkin)", title_fontsize=8.5, framealpha=0.95,
              handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=3.0)

def x0_of(mask_cols):
    has = np.zeros(len(psihigh), bool)
    for c in mask_cols:
        y = col(c); has |= np.isfinite(y)
    return psihigh[has].min() if has.any() else psihigh.min()

stem = os.path.splitext(os.path.basename(CSVP))[0] + SUFFIX
figdir = os.path.abspath(os.path.join(here, "..", "figures")); os.makedirs(figdir, exist_ok=True)

# ---------- (1) norm ----------
_panel = "norm"
fig, ax = plt.subplots(figsize=(11.4, 5.4)); ax.set_yscale("log")
for k, q in enumerate(qs):
    yr = col(f"norm_ric_q{q}"); yg = col(f"norm_gal_q{q}")
    mr = np.isfinite(yr); mg = np.isfinite(yg)
    ax.plot(psihigh[mr], yr[mr], "o-", color=colors_ric[k], lw=1.8, ms=5)
    ax.plot(psihigh[mg], yg[mg], "s--", color=colors_gal[k], lw=1.5, ms=4.5, alpha=0.95)
x0 = x0_of([f"norm_ric_q{q}" for q in qs] + [f"norm_gal_q{q}" for q in qs])
edge_log_x(ax, x0, psihigh.max())
ax.set_xlabel(r"outer truncation  $\psi_{high}$   (log distance from edge, 1 - $\psi_{high}$; edge at right)", fontsize=11)
ax.set_ylabel(r"|delta_coil|  per surface  (log)", fontsize=11)
ax.set_title(f"delta_coil vs outer truncation: Riccati (STRIDE) vs Galerkin (RDCON)\n{CASE}, n=1", fontsize=11)
ax.grid(alpha=0.25, which="both")
# legends INSIDE the upper-left empty region (curves stay in a flat band until Galerkin blows up on the right)
method_legend(ax, loc="upper left", bbox=(0.012, 0.52))
surface_legend(ax, loc="upper left", bbox=(0.012, 0.985))
fig.subplots_adjust(left=0.08, bottom=0.13, right=0.97, top=0.9)
p1 = os.path.join(figdir, stem + "_norm.png"); fig.savefig(p1, dpi=150); print("wrote:", p1)

# ---------- (2) relchange vs outermost psihigh ----------
_panel = "rel"
fig, ax = plt.subplots(figsize=(11.4, 5.4)); ax.set_yscale("log")
for k, q in enumerate(qs):
    for meth, cmap, style in (("ric", colors_ric, ("o-", 1.8, 5, 1.0)), ("gal", colors_gal, ("s--", 1.5, 4.5, 0.95))):
        y = col(f"norm_{meth}_q{q}"); m = np.isfinite(y)
        if m.sum() < 2:
            continue
        ref = y[m][-1]
        ax.plot(psihigh[m], np.abs(y[m] - ref) / max(abs(ref), 1e-300),
                style[0], color=cmap[k], lw=style[1], ms=style[2], alpha=style[3])
x0 = x0_of([f"norm_ric_q{q}" for q in qs] + [f"norm_gal_q{q}" for q in qs])
edge_log_x(ax, x0, psihigh.max())
ax.set_xlabel(r"outer truncation  $\psi_{high}$   (log distance from edge, 1 - $\psi_{high}$; edge at right)", fontsize=11)
ax.set_ylabel(r"|$\Delta$ delta_coil| / |delta_coil($\psi_{high}$=max)|", fontsize=11)
ax.set_title(f"Relative sensitivity of delta_coil to the truncation boundary\n{CASE}: Riccati vs Galerkin (lower = more converged)", fontsize=11)
ax.grid(alpha=0.25, which="both")
# legends INSIDE the lower-left empty region (curves live in the upper-left / lower-right)
method_legend(ax, loc="lower left", bbox=(0.012, 0.30))
surface_legend(ax, loc="lower left", bbox=(0.012, 0.015))
fig.subplots_adjust(left=0.09, bottom=0.13, right=0.97, top=0.9)
p2 = os.path.join(figdir, stem + "_relchange.png"); fig.savefig(p2, dpi=150); print("wrote:", p2)

# ---------- (3) cosine shape metric ----------
_panel = "metric"
fig, ax = plt.subplots(figsize=(11.4, 5.4))
for k, q in enumerate(qs):
    yr = col(f"cos_ric_q{q}"); yg = col(f"cos_gal_q{q}")
    mr = np.isfinite(yr); mg = np.isfinite(yg)
    ax.plot(psihigh[mr], yr[mr], "o-", color=colors_ric[k], lw=1.9, ms=5)
    ax.plot(psihigh[mg], yg[mg], "s--", color=colors_gal[k], lw=1.5, ms=4.5, alpha=0.95)
ax.axhline(1.0, ls=":", color="#888", lw=1.0)
x0 = x0_of([f"cos_ric_q{q}" for q in qs] + [f"cos_gal_q{q}" for q in qs])
edge_log_x(ax, x0, psihigh.max())
ax.set_ylim(-0.05, 1.06)
ax.set_xlabel(r"outer truncation  $\psi_{high}$   (log distance from edge, 1 - $\psi_{high}$; edge at right)", fontsize=11)
ax.set_ylabel("cosine similarity of delta_coil with full-domain nominal", fontsize=11)
ax.set_title(f"Shape (cosine) metric vs psihigh truncation: Riccati vs Galerkin\n{CASE}, n=1  (1.0 = shape unchanged; lower = pattern rotated)", fontsize=11)
ax.grid(alpha=0.25)
method_legend(ax); surface_legend(ax)
fig.subplots_adjust(left=0.08, bottom=0.13, right=0.76, top=0.88)
p3 = os.path.join(figdir, stem + "_metric.png"); fig.savefig(p3, dpi=150); print("wrote:", p3)
