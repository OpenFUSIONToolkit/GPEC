"""Heatmap diagnostics for MWV/min_centering.py at I=100 A,
noise_floor_T = 1e-5 T, scanning (Δx_input, noise_rel_frac).

For each cell the coil is given a pure x-shift, synthetic Hall probes are
generated with noise, and the fit is run. Heatmaps plot recovery error in
(rel, true-Δx) space.
"""
import csv
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from min_centering import B, fit, load_dat, place

HERE = os.path.dirname(os.path.abspath(__file__))
COIL_PATH = os.path.join(HERE, "..", "examples", "Br_3D_example", "sparc_pf1u.dat")
OUT_DIR = os.path.join(HERE, "diag_100A_shift_noise")
os.makedirs(OUT_DIR, exist_ok=True)

CURRENT_A     = 100.0
NOISE_FLOOR_T = 1.0e-5
SHIFTS_MM     = [0.01, 0.1, 1.0, 10.0, 100.0]
NOISE_RELS    = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5]
SEED_BASE     = 91011


def add_noise(B_clean, floor, rel, rng):
    mag = np.linalg.norm(B_clean, axis=1, keepdims=True)
    sigma = np.sqrt(floor ** 2 + (rel * mag) ** 2)
    return B_clean + rng.standard_normal(B_clean.shape) * sigma


def build_probes(c0, R_inner, R_outer):
    Z_inner = np.linspace(c0[2] - 0.6, c0[2] + 0.6, 14)
    Z_outer = np.linspace(c0[2] - 0.4, c0[2] + 0.4, 8)
    phi_i = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    phi_o = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    PpI, ZzI = np.meshgrid(phi_i, Z_inner, indexing="ij")
    PpO, ZzO = np.meshgrid(phi_o, Z_outer, indexing="ij")
    return np.vstack([
        np.column_stack([R_inner * np.cos(PpI.ravel()) + c0[0],
                         R_inner * np.sin(PpI.ravel()) + c0[1], ZzI.ravel()]),
        np.column_stack([R_outer * np.cos(PpO.ravel()) + c0[0],
                         R_outer * np.sin(PpO.ravel()) + c0[1], ZzO.ravel()]),
    ])


def heatmap(Z, x_vals, y_vals, x_label, y_label, title, outpath,
            fmt="{:.2g}", log_color=False):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    Zsafe = np.where(np.isfinite(Z) & (Z > 0), Z, np.nan) if log_color else Z
    if log_color:
        vmax = np.nanmax(Zsafe)
        vmin = max(np.nanmin(Zsafe), vmax * 1e-6) if np.isfinite(vmax) else 1e-12
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        im = ax.imshow(Zsafe, origin="lower", aspect="auto", cmap="viridis", norm=norm)
    else:
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([f"{v:g}" for v in x_vals], rotation=30)
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([f"{v:g}" for v in y_vals])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    Zmax = np.nanmax(Z) if np.isfinite(np.nanmax(Z)) else 1.0
    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            v = Z[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    color="white" if v < Zmax / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label=title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"  wrote {outpath}")


if __name__ == "__main__":
    coil = load_dat(COIL_PATH, stride=20)
    c0 = coil.mean(axis=0)
    Rext = np.hypot(coil[:, 0] - c0[0], coil[:, 1] - c0[1])
    probes = build_probes(c0, 0.40 * Rext.min(), 0.50 * Rext.max())
    print(f"coil: {coil.shape[0]} segs, centroid={c0}")
    print(f"probes: {probes.shape[0]}")

    init = np.concatenate([c0, [0.0, 0.0]])
    results = []
    print(f"\nScan: {len(SHIFTS_MM)} shifts × {len(NOISE_RELS)} rel-noise levels "
          f"(I={CURRENT_A} A, floor={NOISE_FLOOR_T:g} T)")
    for i, dx_mm in enumerate(SHIFTS_MM):
        true_pose = np.concatenate([c0 + np.array([dx_mm * 1e-3, 0.0, 0.0]),
                                    [0.0, 0.0]])
        B_clean = CURRENT_A * B(place(coil, true_pose), probes)
        for j, rel in enumerate(NOISE_RELS):
            rng = np.random.default_rng(SEED_BASE + i * 100 + j)
            B_meas = add_noise(B_clean, NOISE_FLOOR_T, rel, rng)
            # Fit on 1-A model; rescale measurement accordingly.
            init_with_dx = init.copy()
            init_with_dx[0] = c0[0] + dx_mm * 1e-3 / 2  # warm-start halfway
            rec = fit(coil, probes, B_meas / CURRENT_A, init=init)
            err = rec - true_pose
            results.append({
                "dx_true_mm": dx_mm, "rel": rel,
                "err_dx_mm":   err[0] * 1e3,
                "err_dy_mm":   err[1] * 1e3,
                "err_dz_mm":   err[2] * 1e3,
                "err_tx_deg":  err[3],
                "err_ty_deg":  err[4],
                "err_shift_mag_mm":  float(np.linalg.norm(err[:3]) * 1e3),
                "err_tilt_mag_deg":  float(np.linalg.norm(err[3:])),
            })
            print(f"  Δx={dx_mm:7.3f} mm  rel={rel:g}:  "
                  f"|shift_err|={np.linalg.norm(err[:3]) * 1e3:.3e} mm,  "
                  f"|tilt_err|={np.linalg.norm(err[3:]):.3e} deg")

    csv_path = os.path.join(OUT_DIR, "diag_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {csv_path}")

    def Z_of(key):
        Z = np.zeros((len(SHIFTS_MM), len(NOISE_RELS)))
        for k, r in enumerate(results):
            i = SHIFTS_MM.index(r["dx_true_mm"])
            j = NOISE_RELS.index(r["rel"])
            Z[i, j] = r[key]
        return Z

    heatmap(
        np.abs(Z_of("err_dx_mm")),
        NOISE_RELS, SHIFTS_MM,
        "noise_rel_frac", "true Δx [mm]",
        "|Δx recovery error| [mm]",
        os.path.join(OUT_DIR, "heatmap_dx_error.png"),
        log_color=True,
    )
    heatmap(
        Z_of("err_shift_mag_mm"),
        NOISE_RELS, SHIFTS_MM,
        "noise_rel_frac", "true Δx [mm]",
        "|total shift error| [mm]",
        os.path.join(OUT_DIR, "heatmap_shift_error.png"),
        log_color=True,
    )
    heatmap(
        np.sqrt(Z_of("err_dy_mm") ** 2 + Z_of("err_dz_mm") ** 2),
        NOISE_RELS, SHIFTS_MM,
        "noise_rel_frac", "true Δx [mm]",
        "|spurious (Δy, Δz)| [mm]",
        os.path.join(OUT_DIR, "heatmap_position_leakage.png"),
        log_color=True,
    )
    heatmap(
        Z_of("err_tilt_mag_deg"),
        NOISE_RELS, SHIFTS_MM,
        "noise_rel_frac", "true Δx [mm]",
        "|spurious tilt| [deg]",
        os.path.join(OUT_DIR, "heatmap_tilt_leakage.png"),
        log_color=True,
    )

    print(f"\nAll outputs in {OUT_DIR}")
