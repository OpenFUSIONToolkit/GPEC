#!/usr/bin/env python3
"""
Replots the existing reg_plotting/scan_results.csv in the style of
comp_min.ipynb (1×2 layout: in-axis identification error + cross-axis
leakage, linear axes, "1% of true" reference line). One pair of figures
per Hall grid size.

Outputs go to similar_plotting/.

Run after reg_plot.py has produced the CSV. No Julia required.
"""
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "reg_plotting" / "scan_results.csv"
OUTPUT_DIR = ROOT / "similar_plotting"
OUTPUT_DIR.mkdir(exist_ok=True)

METHODS = ["new", "old", "plus"]
# Markers chosen to match comp_min.ipynb's convention (circle=new, square=old)
# with plus → triangle to round out the three-method comparison.
COLORS = {"old": "#d62728", "new": "#1f77b4", "plus": "#2ca02c"}
MARKERS = {"old": "s", "new": "o", "plus": "^"}
ERR_FLOOR = 1e-12

# Slice the larger CSV to a single "representative" noise level so these
# figures stay comparable to the original notebook plots.
DEFAULT_REL = 1.0e-3
DEFAULT_FLOOR = 1.0e-5


def safe_abs(v):
    return max(abs(v), ERR_FLOOR)


def parse_float(s):
    if s is None or s == "" or s in ("NaN", "nan"):
        return float("nan")
    return float(s)


def load_rows(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["n_probes"] = int(r["n_probes"])
            for k in (
                "dx_true_mm", "dtx_true_deg",
                "dx_rec_mm", "dy_rec_mm", "dz_rec_mm",
                "dtx_rec_deg", "dty_rec_deg",
                "runtime_s",
            ):
                r[k] = parse_float(r[k])
            # New columns may be present after the 3D-crossing rewrite;
            # be lenient with the older CSV format.
            r["noise_rel_frac"] = parse_float(r.get("noise_rel_frac", ""))
            r["noise_floor_T"] = parse_float(r.get("noise_floor_T", ""))
            r["error_message"] = r.get("error_message", "") or ""
            r["failed"] = bool(r["error_message"]) or any(
                math.isnan(r[k]) for k in ("dx_rec_mm", "dtx_rec_deg")
            )
            rows.append(r)
    return rows


def select_default_noise(rows):
    """If the CSV has noise columns, slice to the default cell. Otherwise
    (older single-noise CSV) return all non-failed rows."""
    has_noise = any(not math.isnan(r["noise_rel_frac"]) for r in rows)
    if not has_noise:
        return [r for r in rows if not r["failed"]]
    out = []
    for r in rows:
        if r["failed"]:
            continue
        if not math.isclose(r["noise_rel_frac"], DEFAULT_REL,
                            rel_tol=1e-9, abs_tol=1e-15):
            continue
        if not math.isclose(r["noise_floor_T"], DEFAULT_FLOOR,
                            rel_tol=1e-9, abs_tol=1e-15):
            continue
        out.append(r)
    return out


def select(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def plot_shift_scan(rows, n_probes, outpath):
    """Mirrors comp_min.ipynb shift-plot cell (1×2 layout)."""
    fig, (ax_in, ax_cross) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.13, top=0.88, wspace=0.28)

    # ─── In-axis Δx error ───────────────────────────────────────────────
    xs_any = None
    for m in METHODS:
        sub = sorted(
            select(rows, n_probes=n_probes, scan_kind="shift", method=m),
            key=lambda r: r["dx_true_mm"],
        )
        xs = [r["dx_true_mm"] for r in sub]
        ys = [safe_abs(r["dx_rec_mm"] - r["dx_true_mm"]) for r in sub]
        ax_in.plot(xs, ys, marker=MARKERS[m], color=COLORS[m], lw=2,
                   markersize=8, label=f"{m} |Δx error|")
        xs_any = xs
    if xs_any:
        ax_in.plot(xs_any, [x / 100 for x in xs_any], ls="--", color="gray",
                     lw=1.5, label="1% of true")
    ax_in.set_xlabel("true Δx [mm]")
    ax_in.set_ylabel("|recovered − true| [mm]")
    ax_in.set_title("Shift scan: in-axis (Δx) error")
    ax_in.grid(True, which="both", alpha=0.3)
    ax_in.legend(loc="lower right", fontsize=9)

    # ─── Cross-axis leakage: |Δy,Δz| [mm] + |tilt| [deg] ────────────────
    for m in METHODS:
        sub = sorted(
            select(rows, n_probes=n_probes, scan_kind="shift", method=m),
            key=lambda r: r["dx_true_mm"],
        )
        xs = [r["dx_true_mm"] for r in sub]
        ys_pos = [safe_abs(np.hypot(r["dy_rec_mm"], r["dz_rec_mm"])) for r in sub]
        ys_tilt = [safe_abs(np.hypot(r["dtx_rec_deg"], r["dty_rec_deg"])) for r in sub]
        ax_cross.plot(xs, ys_pos, marker=MARKERS[m], color=COLORS[m], lw=2,
                        markersize=8, label=f"{m} |Δy,Δz| [mm]")
        ax_cross.plot(xs, ys_tilt, marker=MARKERS[m], color=COLORS[m], lw=1.5,
                        ls=":", markersize=7, label=f"{m} |tilt| [deg]")
    ax_cross.set_xlabel("true Δx [mm]")
    ax_cross.set_ylabel("|spurious component|")
    ax_cross.set_title("Shift scan: cross-axis leakage")
    ax_cross.grid(True, which="both", alpha=0.3)
    ax_cross.legend(loc="lower right", fontsize=8, ncol=2)

    fig.suptitle(f"Shift scan @ {n_probes} probes", fontsize=13)
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"  wrote {outpath}")


def plot_tilt_scan(rows, n_probes, outpath):
    """Mirrors comp_min.ipynb tilt-plot cell (1×2 layout)."""
    fig, (ax_in, ax_cross) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.13, top=0.88, wspace=0.28)

    # ─── In-axis Δtx error ──────────────────────────────────────────────
    xs_any = None
    for m in METHODS:
        sub = sorted(
            select(rows, n_probes=n_probes, scan_kind="tilt", method=m),
            key=lambda r: r["dtx_true_deg"],
        )
        xs = [r["dtx_true_deg"] for r in sub]
        ys = [safe_abs(r["dtx_rec_deg"] - r["dtx_true_deg"]) for r in sub]
        ax_in.plot(xs, ys, marker=MARKERS[m], color=COLORS[m], lw=2,
                     markersize=8, label=f"{m} |Δtx error|")
        xs_any = xs
    if xs_any:
        ax_in.plot(xs_any, [x / 100 for x in xs_any], ls="--", color="gray",
                     lw=1.5, label="1% of true")
    ax_in.set_xlabel("true Δtx [deg]")
    ax_in.set_ylabel("|recovered − true| [deg]")
    ax_in.set_title("Tilt scan: in-axis (Δtx) error")
    ax_in.grid(True, which="both", alpha=0.3)
    ax_in.legend(loc="lower right", fontsize=9)

    # ─── Cross-component leakage: |Δty| [deg] + |shift| [mm] ────────────
    for m in METHODS:
        sub = sorted(
            select(rows, n_probes=n_probes, scan_kind="tilt", method=m),
            key=lambda r: r["dtx_true_deg"],
        )
        xs = [r["dtx_true_deg"] for r in sub]
        ys_dty = [safe_abs(r["dty_rec_deg"]) for r in sub]
        ys_shift = [
            safe_abs(np.sqrt(r["dx_rec_mm"] ** 2 + r["dy_rec_mm"] ** 2 + r["dz_rec_mm"] ** 2))
            for r in sub
        ]
        ax_cross.plot(xs, ys_dty, marker=MARKERS[m], color=COLORS[m], lw=2,
                        markersize=8, label=f"{m} |Δty| [deg]")
        ax_cross.plot(xs, ys_shift, marker=MARKERS[m], color=COLORS[m], lw=1.5,
                        ls=":", markersize=7, label=f"{m} |shift| [mm]")
    ax_cross.set_xlabel("true Δtx [deg]")
    ax_cross.set_ylabel("|spurious component|")
    ax_cross.set_title("Tilt scan: cross-component leakage")
    ax_cross.grid(True, which="both", alpha=0.3)
    ax_cross.legend(loc="lower right", fontsize=8, ncol=2)

    fig.suptitle(f"Tilt scan @ {n_probes} probes", fontsize=13)
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"  wrote {outpath}")


def main():
    if not CSV_PATH.exists():
        sys.exit(f"CSV not found: {CSV_PATH}. Run reg_plot.py first.")
    all_rows = load_rows(CSV_PATH)
    rows = select_default_noise(all_rows)
    grids = sorted(set(r["n_probes"] for r in rows))
    print(f"Loaded {len(all_rows)} rows, sliced to {len(rows)} at "
          f"rel={DEFAULT_REL:g}/floor={DEFAULT_FLOOR:g}, grids: {grids}")
    for g in grids:
        plot_shift_scan(rows, g, OUTPUT_DIR / f"shift_scan_{g:04d}_probes.png")
        plot_tilt_scan(rows, g, OUTPUT_DIR / f"tilt_scan_{g:04d}_probes.png")
    print(f"\nDone. Plots in {OUTPUT_DIR}")
    for p in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
