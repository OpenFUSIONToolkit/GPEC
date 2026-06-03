"""Diagnostics for MWV/min_centering.py at I=100 A and noise_floor_T=1e-5,
scanning noise_rel_frac. Writes diag_100A/ outputs.

Noise model per Hall-probe component:
    sigma_i = sqrt(NOISE_FLOOR_T**2 + (rel * |B|)**2)
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
OUT_DIR = os.path.join(HERE, "diag_100A")
os.makedirs(OUT_DIR, exist_ok=True)

CURRENT_A     = 100.0
NOISE_FLOOR_T = 1.0e-5
NOISE_RELS    = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5]
SEED_BASE     = 54321


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


if __name__ == "__main__":
    coil = load_dat(COIL_PATH, stride=20)
    c0 = coil.mean(axis=0)
    Rext = np.hypot(coil[:, 0] - c0[0], coil[:, 1] - c0[1])
    probes = build_probes(c0, 0.40 * Rext.min(), 0.50 * Rext.max())
    print(f"coil: {coil.shape[0]} segs, centroid={c0}")
    print(f"probes: {probes.shape[0]}")

    true_pose = np.concatenate([c0 + np.array([0.005, -0.002, 0.001]),
                                [0.30, -0.10]])
    # Scale B-field to current = 100 A (the B() helper assumes 1 A).
    B_clean = CURRENT_A * B(place(coil, true_pose), probes)
    Bmag = np.linalg.norm(B_clean, axis=1)
    print(f"I = {CURRENT_A} A, |B| range: {Bmag.min():.3e}-{Bmag.max():.3e} T  "
          f"(median {np.median(Bmag):.3e} T)")
    print(f"Noise floor: {NOISE_FLOOR_T:g} T  "
          f"(= {NOISE_FLOOR_T / np.median(Bmag) * 100:.2f}% of median |B|)")
    init = np.concatenate([c0, [0.0, 0.0]])

    results = []
    print("\nScanning noise_rel_frac:")
    for j, rel in enumerate(NOISE_RELS):
        rng = np.random.default_rng(SEED_BASE + j)
        B_meas = add_noise(B_clean, NOISE_FLOOR_T, rel, rng)

        # Wrap forward model so the fit sees the same 100 A scaling.
        def B100(coils, probes):
            return CURRENT_A * B(coils, probes)

        # Monkey-patch B inside min_centering's namespace would be intrusive;
        # easier: fit on the 1 A model, but rescale the measurement.
        rec = fit(coil, probes, B_meas / CURRENT_A, init=init)
        err = rec - true_pose
        results.append({
            "rel": rel,
            "err_dx_m":   err[0], "err_dy_m":  err[1], "err_dz_m": err[2],
            "err_tx_deg": err[3], "err_ty_deg": err[4],
            "err_shift_mag_m":  float(np.linalg.norm(err[:3])),
            "err_tilt_mag_deg": float(np.linalg.norm(err[3:])),
        })
        print(f"  rel={rel:g}: "
              f"|shift_err|={np.linalg.norm(err[:3]) * 1e3:.3e} mm, "
              f"|tilt_err|={np.linalg.norm(err[3:]):.3e} deg")

    csv_path = os.path.join(OUT_DIR, "diag_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {csv_path}")

    # 2-subplot summary
    rels = [r["rel"] for r in results]
    shift_err_mm  = [r["err_shift_mag_m"] * 1e3 for r in results]
    tilt_err_deg  = [r["err_tilt_mag_deg"]      for r in results]
    xs = list(range(len(rels)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ax1.plot(xs, shift_err_mm, "o-", color="#1f77b4", lw=2, markersize=8)
    ax1.set_xticks(xs); ax1.set_xticklabels([f"{r:g}" for r in rels], rotation=30)
    ax1.set_xlabel("noise_rel_frac")
    ax1.set_ylabel("|shift error| [mm]")
    ax1.set_title(f"Shift error vs. relative noise  "
                  f"(I={CURRENT_A} A, floor={NOISE_FLOOR_T:g} T)")
    ax1.grid(True, alpha=0.3); ax1.set_ylim(bottom=0)

    ax2.plot(xs, tilt_err_deg, "s-", color="#d62728", lw=2, markersize=8)
    ax2.set_xticks(xs); ax2.set_xticklabels([f"{r:g}" for r in rels], rotation=30)
    ax2.set_xlabel("noise_rel_frac")
    ax2.set_ylabel("|tilt error| [deg]")
    ax2.set_title(f"Tilt error vs. relative noise  "
                  f"(I={CURRENT_A} A, floor={NOISE_FLOOR_T:g} T)")
    ax2.grid(True, alpha=0.3); ax2.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "rel_scan_summary.png"), dpi=140)
    plt.close(fig)
    print(f"  wrote {os.path.join(OUT_DIR, 'rel_scan_summary.png')}")

    # Per-component subplot
    components = [
        ("err_dx_m",   "Δx [mm]",      1e3),
        ("err_dy_m",   "Δy [mm]",      1e3),
        ("err_dz_m",   "Δz [mm]",      1e3),
        ("err_tx_deg", "tilt_x [deg]", 1.0),
        ("err_ty_deg", "tilt_y [deg]", 1.0),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), sharex=True)
    for ax, (key, label, scale) in zip(axes, components):
        ys = [r[key] * scale for r in results]
        ax.plot(xs, ys, "o-", lw=2, markersize=7)
        ax.axhline(0.0, color="gray", lw=0.8, ls=":")
        ax.set_xticks(xs); ax.set_xticklabels([f"{r:g}" for r in rels], rotation=30)
        ax.set_xlabel("noise_rel_frac")
        ax.set_ylabel(f"error in {label}")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Per-component recovery error  "
                 f"(I={CURRENT_A} A, floor={NOISE_FLOOR_T:g} T, seed_base={SEED_BASE})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "rel_scan_components.png"), dpi=140)
    plt.close(fig)
    print(f"  wrote {os.path.join(OUT_DIR, 'rel_scan_components.png')}")

    print(f"\nAll outputs in {OUT_DIR}")
