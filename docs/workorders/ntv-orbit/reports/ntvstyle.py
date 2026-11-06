"""Shared figure conventions for the ntv-orbit reports, on top of Nik's matplotlibrc (paper_bold: bold serif, 10 pt).

Never override the rc fonts, sizes, weights or line widths in a figure script. Widths: 3.4 in (single column, default),
5.0 in (1.5 column) or 7.0 in (two column), same font sizes. Explanations go in the report caption, not on the plot.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[5]
SINGLE, ONEHALF, DOUBLE = 3.4, 5.0, 7.0

# One meaning per color across the project.
TRAPPED, PASSING, LOST, UNCLOSED = "#2a78d6", "#eb6834", "#3b3b38", "#eda100"
SURFACE, ORBIT = "#8a8984", "#0b0b0b"  # fgar (surface path), forb (orbit path)


def save(fig, out_dir, name):
    """Write <name>.pdf (papers, Overleaf), <name>.png at 300 dpi (slides) and <name>.svg (HTML reports)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    print(out_dir / f"{name}.pdf")


def panel(ax, letter):
    """Panel letter (a), (b), ... above the upper-left corner, where it cannot cover data."""
    ax.set_title(f"({letter})", loc="left")
