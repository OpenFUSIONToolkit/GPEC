"""Shared figure helpers for the mre-islands reports and paper figures.

Fonts, sizes, weights and line widths come from the user's matplotlibrc (paper_bold); nothing here overrides them.
"""
import os
import string

import matplotlib.pyplot as plt

# Print widths in inches: single column, 1.5 column, two column.
WIDTH = {"single": 3.4, "onehalf": 5.0, "double": 7.0}

# One fixed color per physical category for the whole project (Okabe-Ito, colorblind safe).
PALETTE = {
    "sparc": "#0072B2",     # machine: SPARC
    "arc": "#D55E00",       # machine: ARC
    "riccati": "#000000",   # Δ′ formalism: Riccati
    "galerkin": "#009E73",  # Δ′ formalism: Galerkin
    "fortran": "#CC79A7",   # Fortran GPEC 1.6 reference
    "seed": "#999999",      # seed-width bands
    "neutral": "#555555",
}
# Second channel so series also differ without color.
DASH = {"vac": "-", "wall": "--"}
MARKER = {1: "o", 2: "s", 3: "^"}


def figure(width="single", aspect=0.75, nrows=1, ncols=1, **kw):
    """Figure at true print width; height = width x aspect."""
    w = WIDTH.get(width, width)
    return plt.subplots(nrows, ncols, figsize=(w, w * aspect), **kw)


def panel_letters(axes, x=0.03, y=0.97, start=0):
    """Write (a), (b), ... in the upper-left corner of each axis."""
    for k, ax in enumerate(getattr(axes, "flat", axes)):
        ax.text(x, y, f"({string.ascii_lowercase[start + k]})", transform=ax.transAxes, ha="left", va="top")


def save(fig, base):
    """Write base.pdf, base.png (300 dpi) and base.svg; returns the three paths."""
    os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
    paths = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300}), ("svg", {})):
        p = f"{base}.{ext}"
        fig.savefig(p, **kw)
        paths.append(p)
    return paths
