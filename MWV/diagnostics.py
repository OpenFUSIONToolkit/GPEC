"""Diagnostics for MWV/min_centering.py on sparc_pf1u.dat:
scan recovery error across (noise_floor_T, noise_rel_frac).

Noise model per Hall-probe component:
    sigma_i = sqrt(noise_floor_T**2 + (noise_rel_frac * |B|)**2)

Writes diag_plots/diag_results.csv and a set of heatmap PNGs.
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
OUT_DIR = os.path.join(HERE, "diag_plots")
os.makedirs(OUT_DIR, exist_ok=True)

NOISE_FLOORS = [0.0, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3]      # Tesla
NOISE_RELS   = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5]
SEED_BASE    = 12345


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


def heatmap(values, label, fname, fmt="{:.2g}"):
    Z = np.array(values).reshape(len(NOISE_FLOORS), len(NOISE_RELS))
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(NOISE_RELS)))
    ax.set_xticklabels([f"{r:g}" for r in NOISE_RELS], rotation=30)
    ax.set_yticks(range(len(NOISE_FLOORS)))
    ax.set_yticklabels([f"{f:g}" for f in NOISE_FLOORS])
    ax.set_xlabel("noise_rel_frac")
    ax.set_ylabel("noise_floor_T [T]")
    ax.set_title(label)
    Zmax = max(Z.max(), 1e-30)
    for i in range(len(NOISE_FLOORS)):
        for j in range(len(NOISE_RELS)):
            ax.text(j, i, fmt.format(Z[i, j]), ha="center", va="center",
                    color="white" if Z[i, j] < Zmax / 2 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, label=label)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


if __name__ == "__main__":
    coil = load_dat(COIL_PATH, stride=20)
    c0 = coil.mean(axis=0)
    Rext = np.hypot(coil[:, 0] - c0[0], coil[:, 1] - c0[1])
    probes = build_probes(c0, 0.40 * Rext.min(), 0.50 * Rext.max())
    print(f"coil: {coil.shape[0]} segs, centroid={c0}")
    print(f"probes: {probes.shape[0]}")

    true_pose = np.concatenate([c0 + np.array([0.005, -0.002, 0.001]),
                                [0.30, -0.10]])
    B_clean = B(place(coil, true_pose), probes)
    Bmag = np.linalg.norm(B_clean, axis=1)
    print(f"|B| range: {Bmag.min():.3e}-{Bmag.max():.3e} T  "
          f"(median {np.median(Bmag):.3e} T)")
    init = np.concatenate([c0, [0.0, 0.0]])

    results = []
    print("\nScanning noise (floor T × rel frac):")
    for i, floor in enumerate(NOISE_FLOORS):
        for j, rel in enumerate(NOISE_RELS):
            rng = np.random.default_rng(SEED_BASE + i * 100 + j)
            B_meas = add_noise(B_clean, floor, rel, rng)
            rec = fit(coil, probes, B_meas, init=init)
            err = rec - true_pose
            results.append({
                "floor_T": floor, "rel": rel,
                "err_dx_m":   err[0], "err_dy_m":  err[1], "err_dz_m": err[2],
                "err_tx_deg": err[3], "err_ty_deg": err[4],
                "err_shift_mag_m":   float(np.linalg.norm(err[:3])),
                "err_tilt_mag_deg":  float(np.linalg.norm(err[3:])),
            })
            print(f"  floor={floor:g}  rel={rel:g}:  "
                  f"|shift_err|={np.linalg.norm(err[:3]) * 1e3:.3e} mm,  "
                  f"|tilt_err|={np.linalg.norm(err[3:]):.3e} deg")

    csv_path = os.path.join(OUT_DIR, "diag_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {csv_path}")

    heatmap([r["err_shift_mag_m"] * 1e3 for r in results],
            "|shift error| [mm]", "heatmap_shift_error.png")
    heatmap([r["err_tilt_mag_deg"] for r in results],
            "|tilt error| [deg]", "heatmap_tilt_error.png", fmt="{:.2e}")
    for key, label, fn, scale in [
        ("err_dx_m",   "|Δx error| [mm]",      "heatmap_err_dx.png", 1e3),
        ("err_dy_m",   "|Δy error| [mm]",      "heatmap_err_dy.png", 1e3),
        ("err_dz_m",   "|Δz error| [mm]",      "heatmap_err_dz.png", 1e3),
        ("err_tx_deg", "|tilt_x error| [deg]", "heatmap_err_tx.png", 1.0),
        ("err_ty_deg", "|tilt_y error| [deg]", "heatmap_err_ty.png", 1.0),
    ]:
        heatmap([abs(r[key]) * scale for r in results], label, fn,
                fmt="{:.2e}")

    print(f"\nAll outputs in {OUT_DIR}")
