#!/usr/bin/env python3
"""
Diagnostic comparison of three coil-centering methods (old / new / plus)
across Hall grid sizes, perturbation magnitudes, and a 2-D noise sweep
(noise_rel_frac × noise_floor_T).

Workflow:
  1. Spawns examples/Br_3D_example/reg_plotting_scan.jl which writes
     reg_plotting/scan_results.csv (full 3D crossing, ~3240 rows).
  2. Loads the CSV.
  3. Regenerates the original suite of plots from a "default noise" slice
     (rel=1e-3, floor=1e-5) so they're comparable to the previous scan.
  4. Adds new noise-sweep plots showing how each method degrades with
     increasing relative noise and increasing noise floor.

Usage:
    python3 reg_plot.py             # full scan + plots
    python3 reg_plot.py --no-scan   # skip Julia, just re-plot from CSV
"""
import csv
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
EXAMPLE_DIR = ROOT / "examples" / "Br_3D_example"
OUTPUT_DIR = ROOT / "reg_plotting"
OUTPUT_DIR.mkdir(exist_ok=True)

CSV_PATH = OUTPUT_DIR / "scan_results.csv"
JULIA_SCAN = EXAMPLE_DIR / "reg_plotting_scan.jl"

METHODS = ["old", "new", "plus"]
COLORS = {"old": "tab:red", "new": "tab:blue", "plus": "tab:green"}
MARKERS = {"old": "s", "new": "o", "plus": "^"}
ERR_FLOOR = 1e-12

# "Representative" noise slice for the regenerated original plots — matches
# the previous scan so cross-comparison is meaningful.
DEFAULT_REL = 1.0e-3
DEFAULT_FLOOR = 1.0e-5

# Heatmap reference cell (grid + perturbation magnitude held fixed)
HEATMAP_GRID = 200
HEATMAP_DX_MM = 1.0
HEATMAP_DTX_DEG = 0.1

# Noise-sweep reference (one perturbation per plot, grid scan in facets)
SWEEP_DX_MM = 1.0
SWEEP_DTX_DEG = 0.1


def run_julia_scan():
    print(f"[reg_plot] Julia scan -> {CSV_PATH}")
    proc = subprocess.run(
        ["julia", f"--project={ROOT}", str(JULIA_SCAN), str(CSV_PATH)],
        cwd=ROOT,
    )
    if proc.returncode != 0:
        sys.exit(f"Julia scan failed (returncode={proc.returncode})")


def parse_float(s):
    if s is None or s == "" or s == "NaN" or s == "nan":
        return float("nan")
    return float(s)


def load_rows(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["n_probes"] = int(r["n_probes"])
            r["seed"] = int(r["seed"])
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


def select(rows, **kw):
    out = []
    for r in rows:
        ok = True
        for k, v in kw.items():
            if isinstance(v, float):
                if not (abs(r.get(k, float("nan")) - v) < 1e-12 or r.get(k) == v):
                    ok = False
                    break
            else:
                if r.get(k) != v:
                    ok = False
                    break
        if ok:
            out.append(r)
    return out


def select_default_noise(rows):
    """Slice to the default-noise rows, drop failed cells."""
    out = []
    for r in rows:
        if r["failed"]:
            continue
        if not math.isclose(r["noise_rel_frac"], DEFAULT_REL, rel_tol=1e-9, abs_tol=1e-15):
            continue
        if not math.isclose(r["noise_floor_T"], DEFAULT_FLOOR, rel_tol=1e-9, abs_tol=1e-15):
            continue
        out.append(r)
    return out


def safe_abs(v):
    if math.isnan(v):
        return float("nan")
    return max(abs(v), ERR_FLOOR)


def report_failures(rows):
    print("[reg_plot] Convergence-failure tally:")
    for m in METHODS:
        subs = [r for r in rows if r["method"] == m]
        fails = [r for r in subs if r["failed"]]
        if fails:
            print(f"  {m:5s}: {len(fails):4d} / {len(subs):4d} cells failed")
            # show a sample
            for r in fails[:3]:
                print(f"        e.g. grid={r['n_probes']} kind={r['scan_kind']} "
                      f"rel={r['noise_rel_frac']:g} floor={r['noise_floor_T']:g}: "
                      f"{r['error_message'][:80]}")
        else:
            print(f"  {m:5s}: 0 failures / {len(subs)} cells")


# ───────────────────────────────────────────────────────────────────────────
# Plots that reuse the DEFAULT_NOISE slice (mirror previous reg_plotting set)
# ───────────────────────────────────────────────────────────────────────────

def plot_inaxis_vs_perturb(rows, kind, perturb_key, rec_key, x_label, y_label,
                           title, outname):
    grids = sorted(set(r["n_probes"] for r in rows))
    fig, axes = plt.subplots(1, len(grids), figsize=(3.6 * len(grids), 4.0),
                             sharey=True)
    if len(grids) == 1:
        axes = [axes]
    for ax, g in zip(axes, grids):
        for m in METHODS:
            sub = sorted(
                select(rows, n_probes=g, scan_kind=kind, method=m),
                key=lambda r: r[perturb_key],
            )
            xs = [r[perturb_key] for r in sub]
            ys = [safe_abs(r[rec_key] - r[perturb_key]) for r in sub]
            ax.loglog(xs, ys, marker=MARKERS[m], color=COLORS[m], label=m,
                      lw=1.5, markersize=7)
        ax.set_xlabel(x_label)
        ax.set_title(f"{g} probes")
        ax.grid(True, which="both", alpha=0.3)
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
    grids = sorted(set(r["n_probes"] for r in rows))
    fig, axes = plt.subplots(1, len(grids), figsize=(3.6 * len(grids), 4.0),
                             sharey=True)
    if len(grids) == 1:
        axes = [axes]
    for ax, g in zip(axes, grids):
        for m in METHODS:
            sub = sorted(
                select(rows, n_probes=g, scan_kind=kind, method=m),
                key=lambda r: r[perturb_key],
            )
            xs = [r[perturb_key] for r in sub]
            ys = []
            for r in sub:
                leak = math.sqrt(sum(r[c] ** 2 for c in leak_components))
                ys.append(safe_abs(leak))
            ax.loglog(xs, ys, marker=MARKERS[m], color=COLORS[m], label=m,
                      lw=1.5, markersize=7)
        ax.set_xlabel(x_label)
        ax.set_title(f"{g} probes")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(y_label)
    axes[-1].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    path = OUTPUT_DIR / outname
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_error_vs_grid(rows, kind, perturb_key, rec_key, perturb_values,
                       perturb_label_fmt, x_label, y_label, title, outname):
    fig, axes = plt.subplots(1, len(perturb_values),
                             figsize=(3.6 * len(perturb_values), 4.0),
                             sharey=True)
    if len(perturb_values) == 1:
        axes = [axes]
    for ax, pv in zip(axes, perturb_values):
        for m in METHODS:
            sub = sorted(
                [r for r in rows
                 if r["scan_kind"] == kind
                 and r["method"] == m
                 and abs(r[perturb_key] - pv) < 1e-9],
                key=lambda r: r["n_probes"],
            )
            xs = [r["n_probes"] for r in sub]
            ys = [safe_abs(r[rec_key] - r[perturb_key]) for r in sub]
            ax.loglog(xs, ys, marker=MARKERS[m], color=COLORS[m], label=m,
                      lw=1.5, markersize=7)
        ax.set_xlabel(x_label)
        ax.set_title(perturb_label_fmt.format(pv))
        ax.grid(True, which="both", alpha=0.3)
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
            sub = [r for r in rows if r["n_probes"] == g and r["method"] == m
                   and not r["failed"]]
            ts.append(np.mean([r["runtime_s"] for r in sub]) if sub else float("nan"))
        ax.loglog(grids, ts, marker=MARKERS[m], color=COLORS[m], label=m,
                  lw=2.0, markersize=8)
    ax.set_xlabel("n_probes")
    ax.set_ylabel("mean runtime per fit [s]")
    ax.set_title("Method runtime vs. Hall grid size (default-noise slice)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = OUTPUT_DIR / outname
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_heatmap_grid_vs_perturb(rows, kind, perturb_key, rec_key, perturb_label,
                                 outname):
    grids = sorted(set(r["n_probes"] for r in rows))
    perturbs = sorted({r[perturb_key] for r in rows if r["scan_kind"] == kind})
    fig, axes = plt.subplots(1, len(METHODS), figsize=(5.0 * len(METHODS), 4.5),
                             sharey=True)
    all_err = []
    for m in METHODS:
        for g in grids:
            for pv in perturbs:
                sub = [r for r in rows
                       if r["scan_kind"] == kind and r["method"] == m
                       and r["n_probes"] == g
                       and abs(r[perturb_key] - pv) < 1e-9]
                if sub:
                    e = safe_abs(sub[0][rec_key] - sub[0][perturb_key])
                    if not math.isnan(e):
                        all_err.append(e)
    if not all_err:
        print(f"  skipping {outname}: no data")
        return
    vmin = max(min(all_err), 1e-12)
    vmax = max(all_err)
    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    for ax, m in zip(axes, METHODS):
        Z = np.full((len(perturbs), len(grids)), np.nan)
        for i, pv in enumerate(perturbs):
            for j, g in enumerate(grids):
                sub = [r for r in rows
                       if r["scan_kind"] == kind and r["method"] == m
                       and r["n_probes"] == g
                       and abs(r[perturb_key] - pv) < 1e-9]
                if sub:
                    Z[i, j] = safe_abs(sub[0][rec_key] - sub[0][perturb_key])
        im = ax.pcolormesh(grids, perturbs, Z, norm=norm,
                           cmap="viridis", shading="nearest")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n_probes")
        ax.set_title(m)
    axes[0].set_ylabel(f"true {perturb_label}")
    fig.colorbar(im, ax=axes, label=f"|err in {perturb_label}|", shrink=0.85)
    fig.suptitle(f"Per-method id-error heatmap ({perturb_label})")
    path = OUTPUT_DIR / outname
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ───────────────────────────────────────────────────────────────────────────
# New noise-sweep plots (use the full CSV)
# ───────────────────────────────────────────────────────────────────────────

def plot_error_vs_noise_axis(rows, kind, perturb_key, rec_key, perturb_value,
                             noise_axis, fixed_axis, fixed_value,
                             x_label, y_label, title, outname):
    """
    Facet per grid size. x = noise_axis values (linear/log auto), y = |err|.
    `noise_axis` is "noise_rel_frac" or "noise_floor_T".
    `fixed_axis` is the other one, held at `fixed_value`.
    `perturb_value` selects a single in-axis perturbation magnitude.
    Excludes failed rows. Drops noise_axis == 0 points from log plots.
    """
    grids = sorted(set(r["n_probes"] for r in rows))
    fig, axes = plt.subplots(1, len(grids), figsize=(3.6 * len(grids), 4.0),
                             sharey=True)
    if len(grids) == 1:
        axes = [axes]
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
            xs = [r[noise_axis] for r in sub]
            ys = [safe_abs(r[rec_key] - r[perturb_key]) for r in sub]
            # Replace any zero x with a small floor so log axis works
            xs_plot = [max(x, 1e-7) for x in xs]
            ax.loglog(xs_plot, ys, marker=MARKERS[m], color=COLORS[m], label=m,
                      lw=1.5, markersize=7)
        ax.set_xlabel(x_label)
        ax.set_title(f"{g} probes")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(y_label)
    axes[-1].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    path = OUTPUT_DIR / outname
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_heatmap_noise(rows, kind, perturb_key, rec_key, perturb_value,
                       perturb_label, outname,
                       grid_value=HEATMAP_GRID):
    """
    Per method, 2D heatmap of |err| in (noise_rel × noise_floor) at fixed
    grid + perturbation magnitude. Log color scale.
    """
    rel_vals = sorted({r["noise_rel_frac"] for r in rows})
    floor_vals = sorted({r["noise_floor_T"] for r in rows})
    # Replace 0.0 with a small floor so log axis renders
    rel_plot = [max(v, 1e-7) for v in rel_vals]
    floor_plot = [max(v, 1e-8) for v in floor_vals]

    fig, axes = plt.subplots(1, len(METHODS), figsize=(5.0 * len(METHODS), 4.5),
                             sharey=True)
    all_err = []
    for m in METHODS:
        for nr in rel_vals:
            for nf in floor_vals:
                sub = [r for r in rows
                       if not r["failed"]
                       and r["n_probes"] == grid_value
                       and r["scan_kind"] == kind
                       and r["method"] == m
                       and abs(r[perturb_key] - perturb_value) < 1e-9
                       and abs(r["noise_rel_frac"] - nr) < 1e-15
                       and abs(r["noise_floor_T"] - nf) < 1e-15]
                if sub:
                    e = safe_abs(sub[0][rec_key] - sub[0][perturb_key])
                    if not math.isnan(e):
                        all_err.append(e)
    if not all_err:
        print(f"  skipping {outname}: no data")
        return
    vmin = max(min(all_err), 1e-12)
    vmax = max(all_err)
    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    for ax, m in zip(axes, METHODS):
        Z = np.full((len(floor_vals), len(rel_vals)), np.nan)
        for i, nf in enumerate(floor_vals):
            for j, nr in enumerate(rel_vals):
                sub = [r for r in rows
                       if not r["failed"]
                       and r["n_probes"] == grid_value
                       and r["scan_kind"] == kind
                       and r["method"] == m
                       and abs(r[perturb_key] - perturb_value) < 1e-9
                       and abs(r["noise_rel_frac"] - nr) < 1e-15
                       and abs(r["noise_floor_T"] - nf) < 1e-15]
                if sub:
                    Z[i, j] = safe_abs(sub[0][rec_key] - sub[0][perturb_key])
        im = ax.pcolormesh(rel_plot, floor_plot, Z, norm=norm,
                           cmap="viridis", shading="nearest")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("noise_rel_frac")
        ax.set_title(m)
    axes[0].set_ylabel("noise_floor_T [T]")
    fig.colorbar(im, ax=axes,
                 label=f"|err in {perturb_label}|", shrink=0.85)
    fig.suptitle(f"|err| in (noise_rel × noise_floor)  @ {grid_value} probes, true {perturb_label} = {perturb_value:g}")
    path = OUTPUT_DIR / outname
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main():
    do_scan = "--no-scan" not in sys.argv
    if do_scan:
        run_julia_scan()
    else:
        if not CSV_PATH.exists():
            sys.exit(f"--no-scan requested but {CSV_PATH} does not exist")

    rows = load_rows(CSV_PATH)
    print(f"[reg_plot] Loaded {len(rows)} rows from {CSV_PATH}")
    report_failures(rows)

    slice_rows = select_default_noise(rows)
    print(f"[reg_plot] Default-noise slice (rel={DEFAULT_REL}, floor={DEFAULT_FLOOR}): "
          f"{len(slice_rows)} rows")

    print("[reg_plot] Writing default-noise-slice plots ...")
    plot_inaxis_vs_perturb(
        slice_rows, kind="shift",
        perturb_key="dx_true_mm", rec_key="dx_rec_mm",
        x_label="true Δx [mm]",
        y_label="|Δx_recovered − Δx_true| [mm]",
        title="In-axis shift identification error vs. true shift",
        outname="shift_inaxis_error_vs_shift.png",
    )
    plot_inaxis_vs_perturb(
        slice_rows, kind="tilt",
        perturb_key="dtx_true_deg", rec_key="dtx_rec_deg",
        x_label="true Δtx [deg]",
        y_label="|Δtx_recovered − Δtx_true| [deg]",
        title="In-axis tilt identification error vs. true tilt",
        outname="tilt_inaxis_error_vs_tilt.png",
    )
    plot_crossaxis_vs_perturb(
        slice_rows, kind="shift",
        perturb_key="dx_true_mm",
        leak_components=["dy_rec_mm", "dz_rec_mm"],
        x_label="true Δx [mm]",
        y_label="|spurious (Δy,Δz)| [mm]",
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
    plot_crossaxis_vs_perturb(
        slice_rows, kind="tilt",
        perturb_key="dtx_true_deg",
        leak_components=["dty_rec_deg"],
        x_label="true Δtx [deg]",
        y_label="|spurious Δty| [deg]",
        title="Tilt scan: cross-axis tilt leakage",
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

    plot_heatmap_grid_vs_perturb(
        slice_rows, kind="shift",
        perturb_key="dx_true_mm", rec_key="dx_rec_mm",
        perturb_label="Δx [mm]",
        outname="heatmap_shift_error.png",
    )
    plot_heatmap_grid_vs_perturb(
        slice_rows, kind="tilt",
        perturb_key="dtx_true_deg", rec_key="dtx_rec_deg",
        perturb_label="Δtx [deg]",
        outname="heatmap_tilt_error.png",
    )

    plot_runtime_vs_grid(slice_rows)

    print("[reg_plot] Writing noise-sweep plots ...")
    plot_error_vs_noise_axis(
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
    plot_error_vs_noise_axis(
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
    plot_error_vs_noise_axis(
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
    plot_error_vs_noise_axis(
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
    plot_heatmap_noise(
        rows, kind="shift",
        perturb_key="dx_true_mm", rec_key="dx_rec_mm",
        perturb_value=HEATMAP_DX_MM,
        perturb_label="Δx [mm]",
        outname="heatmap_shift_noise.png",
    )
    plot_heatmap_noise(
        rows, kind="tilt",
        perturb_key="dtx_true_deg", rec_key="dtx_rec_deg",
        perturb_value=HEATMAP_DTX_DEG,
        perturb_label="Δtx [deg]",
        outname="heatmap_tilt_noise.png",
    )

    print("\n[reg_plot] Done. Plots in:", OUTPUT_DIR)
    for p in sorted(OUTPUT_DIR.glob("*.png")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
