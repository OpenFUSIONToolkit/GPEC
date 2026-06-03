#!/usr/bin/env python3
"""
Linear-axis re-plots from reg_plotting/scan_results.csv into new_plotting/.

Covers four scan dimensions:
  - shift identification error (Δx)
  - tilt identification error (Δtx)
  - Hall-probe grid refinement
  - measurement noise (relative + absolute floor sweeps)

All x and y axes are linear (no log scales). Default noise slice
(rel=1e-3, floor=1e-5) is used for the shift / tilt / grid plots.

Run after reg_plot.py has produced the CSV; no Julia required.
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
OUTPUT_DIR = ROOT / "new_plotting"
OUTPUT_DIR.mkdir(exist_ok=True)

METHODS = ["old", "new", "plus"]
COLORS = {"old": "#d62728", "new": "#1f77b4", "plus": "#2ca02c"}
MARKERS = {"old": "s", "new": "o", "plus": "^"}

DEFAULT_REL = 1.0e-3
DEFAULT_FLOOR = 1.0e-5

# Reference perturbations for the noise sweeps (one in-axis magnitude per scan)
SWEEP_DX_MM = 1.0
SWEEP_DTX_DEG = 0.1


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
                "noise_rel_frac", "noise_floor_T",
                "dx_rec_mm", "dy_rec_mm", "dz_rec_mm",
                "dtx_rec_deg", "dty_rec_deg",
                "runtime_s",
            ):
                r[k] = parse_float(r[k])
            r["error_message"] = r.get("error_message", "") or ""
            r["failed"] = bool(r["error_message"]) or any(
                math.isnan(r[k]) for k in ("dx_rec_mm", "dtx_rec_deg")
            )
            rows.append(r)
    return rows


def at_default_noise(rows):
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
    out = []
    for r in rows:
        ok = True
        for k, v in kw.items():
            if isinstance(v, float):
                if not (abs(r.get(k, float("nan")) - v) < 1e-12):
                    ok = False
                    break
            elif r.get(k) != v:
                ok = False
                break
        if ok:
            out.append(r)
    return out


# ───────────────────────────────────────────────────────────────────────────
# Shift / tilt scans — linear axes
# ───────────────────────────────────────────────────────────────────────────

def _format_tick(v):
    """Compact label for a numeric x tick (e.g. '0.01', '1', '100')."""
    if v == 0:
        return "0"
    if abs(v) >= 1:
        return f"{v:g}"
    return f"{v:g}"


def _categorical_x(ax, values, rotate=0):
    """Place ticks at 0..N-1 and label them with the actual values."""
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([_format_tick(v) for v in values], rotation=rotate)


def plot_inaxis_vs_perturb(rows, kind, perturb_key, rec_key,
                           x_label, y_label, title, outname):
    """One subplot per Hall grid size; signed error on linear axes,
    perturbation magnitudes shown as evenly-spaced categorical ticks."""
    grids = sorted(set(r["n_probes"] for r in rows))
    fig, axes = plt.subplots(1, len(grids), figsize=(3.6 * len(grids), 4.4),
                             sharey=True)
    if len(grids) == 1:
        axes = [axes]
    perturb_vals = sorted({r[perturb_key] for r in rows if r["scan_kind"] == kind})
    for ax, g in zip(axes, grids):
        for m in METHODS:
            sub = sorted(
                select(rows, n_probes=g, scan_kind=kind, method=m),
                key=lambda r: r[perturb_key],
            )
            xs = [perturb_vals.index(r[perturb_key]) for r in sub]
            ys = [r[rec_key] - r[perturb_key] for r in sub]
            ax.plot(xs, ys, marker=MARKERS[m], color=COLORS[m], label=m,
                    lw=2, markersize=7)
        ax.axhline(0.0, color="gray", lw=0.8, ls=":")
        _categorical_x(ax, perturb_vals)
        ax.set_xlabel(x_label)
        ax.set_title(f"{g} probes")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(y_label)
    axes[-1].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    path = OUTPUT_DIR / outname
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_crossaxis_vs_perturb(rows, kind, perturb_key, leak_components,
                              x_label, y_label, title, outname):
    """Cross-axis leakage magnitude (positive by construction)."""
    grids = sorted(set(r["n_probes"] for r in rows))
    fig, axes = plt.subplots(1, len(grids), figsize=(3.6 * len(grids), 4.4),
                             sharey=True)
    if len(grids) == 1:
        axes = [axes]
    perturb_vals = sorted({r[perturb_key] for r in rows if r["scan_kind"] == kind})
    for ax, g in zip(axes, grids):
        for m in METHODS:
            sub = sorted(
                select(rows, n_probes=g, scan_kind=kind, method=m),
                key=lambda r: r[perturb_key],
            )
            xs = [perturb_vals.index(r[perturb_key]) for r in sub]
            ys = [math.sqrt(sum(r[c] ** 2 for c in leak_components)) for r in sub]
            ax.plot(xs, ys, marker=MARKERS[m], color=COLORS[m], label=m,
                    lw=2, markersize=7)
        _categorical_x(ax, perturb_vals)
        ax.set_xlabel(x_label)
        ax.set_title(f"{g} probes")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel(y_label)
    axes[-1].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    path = OUTPUT_DIR / outname
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


# ───────────────────────────────────────────────────────────────────────────
# Grid scans — linear axes
# ───────────────────────────────────────────────────────────────────────────

def plot_error_vs_grid(rows, kind, perturb_key, rec_key, perturb_values,
                       perturb_label_fmt, x_label, y_label, title, outname):
    """One subplot per fixed perturbation magnitude."""
    fig, axes = plt.subplots(1, len(perturb_values),
                             figsize=(3.6 * len(perturb_values), 4.4),
                             sharey=True)
    if len(perturb_values) == 1:
        axes = [axes]
    grids = sorted({r["n_probes"] for r in rows})
    for ax, pv in zip(axes, perturb_values):
        for m in METHODS:
            sub = sorted(
                [r for r in rows
                 if r["scan_kind"] == kind
                 and r["method"] == m
                 and abs(r[perturb_key] - pv) < 1e-9],
                key=lambda r: r["n_probes"],
            )
            xs = [grids.index(r["n_probes"]) for r in sub]
            ys = [abs(r[rec_key] - r[perturb_key]) for r in sub]
            ax.plot(xs, ys, marker=MARKERS[m], color=COLORS[m], label=m,
                    lw=2, markersize=7)
        _categorical_x(ax, grids)
        ax.set_xlabel(x_label)
        ax.set_title(perturb_label_fmt.format(pv))
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel(y_label)
    axes[-1].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    path = OUTPUT_DIR / outname
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


# ───────────────────────────────────────────────────────────────────────────
# Noise scans — linear axes
# ───────────────────────────────────────────────────────────────────────────

def plot_error_vs_noise(rows, kind, perturb_key, rec_key, perturb_value,
                        noise_axis, fixed_axis, fixed_value,
                        x_label, y_label, title, outname):
    """One subplot per Hall grid size. x = noise_axis (categorical), y = |err|."""
    grids = sorted(set(r["n_probes"] for r in rows))
    fig, axes = plt.subplots(1, len(grids), figsize=(3.6 * len(grids), 4.4),
                             sharey=True)
    if len(grids) == 1:
        axes = [axes]
    noise_vals = sorted({r[noise_axis] for r in rows})
    for ax, g in zip(axes, grids):
        for m in METHODS:
            sub = sorted(
                [r for r in rows
                 if not r["failed"]
                 and r["n_probes"] == g
                 and r["scan_kind"] == kind
                 and r["method"] == m
                 and abs(r[perturb_key] - perturb_value) < 1e-9
                 and abs(r[fixed_axis] - fixed_value) < 1e-15],
                key=lambda r: r[noise_axis],
            )
            xs = [noise_vals.index(r[noise_axis]) for r in sub]
            ys = [abs(r[rec_key] - r[perturb_key]) for r in sub]
            ax.plot(xs, ys, marker=MARKERS[m], color=COLORS[m], label=m,
                    lw=2, markersize=7)
        _categorical_x(ax, noise_vals, rotate=30)
        ax.set_xlabel(x_label)
        ax.set_title(f"{g} probes")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel(y_label)
    axes[-1].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    path = OUTPUT_DIR / outname
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_runtime_vs_grid(rows, outname="runtime_vs_grid.png"):
    grids = sorted(set(r["n_probes"] for r in rows))
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.5))
    for m in METHODS:
        ts = []
        for g in grids:
            sub = [r for r in rows
                   if r["n_probes"] == g and r["method"] == m and not r["failed"]]
            ts.append(np.mean([r["runtime_s"] for r in sub]) if sub else float("nan"))
        ax.plot(range(len(grids)), ts, marker=MARKERS[m], color=COLORS[m],
                label=m, lw=2, markersize=8)
    _categorical_x(ax, grids)
    ax.set_xlabel("n_probes")
    ax.set_ylabel("mean runtime per fit [s]")
    ax.set_title("Method runtime vs. Hall grid size (linear)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    path = OUTPUT_DIR / outname
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main():
    if not CSV_PATH.exists():
        sys.exit(f"CSV not found: {CSV_PATH}. Run reg_plot.py first.")
    rows = load_rows(CSV_PATH)
    slice_rows = at_default_noise(rows)
    print(f"Loaded {len(rows)} rows; default-noise slice = {len(slice_rows)} rows "
          f"(rel={DEFAULT_REL:g}, floor={DEFAULT_FLOOR:g})")

    # — Shift scan —
    plot_inaxis_vs_perturb(
        slice_rows, kind="shift",
        perturb_key="dx_true_mm", rec_key="dx_rec_mm",
        x_label="true Δx [mm]",
        y_label="Δx_recovered − Δx_true [mm]",
        title="Shift scan: in-axis (Δx) error",
        outname="shift_inaxis_error.png",
    )
    plot_crossaxis_vs_perturb(
        slice_rows, kind="shift",
        perturb_key="dx_true_mm",
        leak_components=["dy_rec_mm", "dz_rec_mm"],
        x_label="true Δx [mm]",
        y_label="|spurious (Δy, Δz)| [mm]",
        title="Shift scan: cross-axis position leakage",
        outname="shift_position_leakage.png",
    )
    plot_crossaxis_vs_perturb(
        slice_rows, kind="shift",
        perturb_key="dx_true_mm",
        leak_components=["dtx_rec_deg", "dty_rec_deg"],
        x_label="true Δx [mm]",
        y_label="|spurious tilt| [deg]",
        title="Shift scan: spurious tilt leakage",
        outname="shift_tilt_leakage.png",
    )

    # — Tilt scan —
    plot_inaxis_vs_perturb(
        slice_rows, kind="tilt",
        perturb_key="dtx_true_deg", rec_key="dtx_rec_deg",
        x_label="true Δtx [deg]",
        y_label="Δtx_recovered − Δtx_true [deg]",
        title="Tilt scan: in-axis (Δtx) error",
        outname="tilt_inaxis_error.png",
    )
    plot_crossaxis_vs_perturb(
        slice_rows, kind="tilt",
        perturb_key="dtx_true_deg",
        leak_components=["dty_rec_deg"],
        x_label="true Δtx [deg]",
        y_label="|spurious Δty| [deg]",
        title="Tilt scan: cross-component tilt leakage",
        outname="tilt_tilt_leakage.png",
    )
    plot_crossaxis_vs_perturb(
        slice_rows, kind="tilt",
        perturb_key="dtx_true_deg",
        leak_components=["dx_rec_mm", "dy_rec_mm", "dz_rec_mm"],
        x_label="true Δtx [deg]",
        y_label="|spurious shift| [mm]",
        title="Tilt scan: spurious shift leakage",
        outname="tilt_position_leakage.png",
    )

    # — Grid scans —
    shift_values = sorted({r["dx_true_mm"] for r in slice_rows
                           if r["scan_kind"] == "shift"})
    tilt_values = sorted({r["dtx_true_deg"] for r in slice_rows
                          if r["scan_kind"] == "tilt"})
    plot_error_vs_grid(
        slice_rows, kind="shift",
        perturb_key="dx_true_mm", rec_key="dx_rec_mm",
        perturb_values=shift_values,
        perturb_label_fmt="Δx = {:g} mm",
        x_label="n_probes",
        y_label="|Δx_recovered − Δx_true| [mm]",
        title="Shift id error vs. Hall grid size",
        outname="shift_error_vs_grid.png",
    )
    plot_error_vs_grid(
        slice_rows, kind="tilt",
        perturb_key="dtx_true_deg", rec_key="dtx_rec_deg",
        perturb_values=tilt_values,
        perturb_label_fmt="Δtx = {:g} deg",
        x_label="n_probes",
        y_label="|Δtx_recovered − Δtx_true| [deg]",
        title="Tilt id error vs. Hall grid size",
        outname="tilt_error_vs_grid.png",
    )

    # — Noise scans (use full CSV, not the default-noise slice) —
    plot_error_vs_noise(
        rows, kind="shift",
        perturb_key="dx_true_mm", rec_key="dx_rec_mm",
        perturb_value=SWEEP_DX_MM,
        noise_axis="noise_rel_frac",
        fixed_axis="noise_floor_T", fixed_value=DEFAULT_FLOOR,
        x_label="noise_rel_frac",
        y_label=f"|Δx err| [mm]  (Δx={SWEEP_DX_MM} mm)",
        title=f"Shift id error vs. relative noise  (floor={DEFAULT_FLOOR:g} T)",
        outname="noise_rel_vs_shift_error.png",
    )
    plot_error_vs_noise(
        rows, kind="tilt",
        perturb_key="dtx_true_deg", rec_key="dtx_rec_deg",
        perturb_value=SWEEP_DTX_DEG,
        noise_axis="noise_rel_frac",
        fixed_axis="noise_floor_T", fixed_value=DEFAULT_FLOOR,
        x_label="noise_rel_frac",
        y_label=f"|Δtx err| [deg]  (Δtx={SWEEP_DTX_DEG}°)",
        title=f"Tilt id error vs. relative noise  (floor={DEFAULT_FLOOR:g} T)",
        outname="noise_rel_vs_tilt_error.png",
    )
    plot_error_vs_noise(
        rows, kind="shift",
        perturb_key="dx_true_mm", rec_key="dx_rec_mm",
        perturb_value=SWEEP_DX_MM,
        noise_axis="noise_floor_T",
        fixed_axis="noise_rel_frac", fixed_value=DEFAULT_REL,
        x_label="noise_floor_T [T]",
        y_label=f"|Δx err| [mm]  (Δx={SWEEP_DX_MM} mm)",
        title=f"Shift id error vs. noise floor  (rel={DEFAULT_REL:g})",
        outname="noise_floor_vs_shift_error.png",
    )
    plot_error_vs_noise(
        rows, kind="tilt",
        perturb_key="dtx_true_deg", rec_key="dtx_rec_deg",
        perturb_value=SWEEP_DTX_DEG,
        noise_axis="noise_floor_T",
        fixed_axis="noise_rel_frac", fixed_value=DEFAULT_REL,
        x_label="noise_floor_T [T]",
        y_label=f"|Δtx err| [deg]  (Δtx={SWEEP_DTX_DEG}°)",
        title=f"Tilt id error vs. noise floor  (rel={DEFAULT_REL:g})",
        outname="noise_floor_vs_tilt_error.png",
    )

    plot_runtime_vs_grid(slice_rows)

    print(f"\nDone. Plots in {OUTPUT_DIR}")
    for p in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
