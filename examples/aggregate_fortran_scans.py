#!/usr/bin/env python3
"""
Aggregate Fortran STRIDE scan NetCDF outputs into the summary CSV format
expected by `LAR_beta_scan/run_scan.jl --compare`, `LAR_epsilon_scan/run_scan.jl --compare`,
and `plot_scan_comparison.py`.

Source: ~/Desktop/plasma/GPEC/docs/scans/outputs/{beta_scan,epsilon_scan}/stride_output_*.nc

Output columns (Julia run_scan.jl format):
    <param>, delta_prime_21_real, delta_prime_21_imag,
             delta_prime_31_real, delta_prime_31_imag,
             delta_W_plasma, delta_W_vacuum, delta_W_total, q_edge
where <param> is `pressure_factor` (beta) or `epsilon` (epsilon).
"""

from pathlib import Path
import re
import csv
import sys
import numpy as np
from netCDF4 import Dataset

FORTRAN_ROOT = Path.home() / "Desktop" / "plasma" / "GPEC" / "docs" / "scans" / "outputs"
JULIA_ROOT = Path(__file__).resolve().parent

SCANS = {
    "beta": {
        "src_dir": FORTRAN_ROOT / "beta_scan",
        "prefix": "stride_output_pf_",
        "param_name": "pressure_factor",
        "out_csv": JULIA_ROOT / "LAR_beta_scan" / "reference" / "pressure_factor_scan_summary.csv",
    },
    "epsilon": {
        "src_dir": FORTRAN_ROOT / "epsilon_scan",
        "prefix": "stride_output_eps_",
        "param_name": "epsilon",
        "out_csv": JULIA_ROOT / "LAR_epsilon_scan" / "reference" / "epsilon_scan_summary.csv",
    },
}


def extract_row(nc_path: Path, param_value: float) -> dict:
    """Pull Δ'(2/1), Δ'(3/1), δW components, and q_edge from one STRIDE .nc file."""
    with Dataset(nc_path) as d:
        # Δ' matrix: shape (i=2, r_prime, r); i=0 real, i=1 imag
        dp = d.variables["Delta_prime"][:]
        q_rat = d.variables["q_rational"][:]

        # Locate indices of 2/1 and 3/1 rational surfaces (q_rational values 2.0 and 3.0)
        idx_21 = np.argmin(np.abs(q_rat - 2.0))
        idx_31 = np.argmin(np.abs(q_rat - 3.0))

        dp_21_re = float(dp[0, idx_21, idx_21]) if np.isclose(q_rat[idx_21], 2.0, atol=0.05) else np.nan
        dp_21_im = float(dp[1, idx_21, idx_21]) if np.isclose(q_rat[idx_21], 2.0, atol=0.05) else np.nan
        dp_31_re = float(dp[0, idx_31, idx_31]) if np.isclose(q_rat[idx_31], 3.0, atol=0.05) else np.nan
        dp_31_im = float(dp[1, idx_31, idx_31]) if np.isclose(q_rat[idx_31], 3.0, atol=0.05) else np.nan

        # Least-stable eigenvalue (first entry, i=0 = real part); matches Julia extract
        dw_p = float(d.variables["W_p_eigenvalue"][0, 0])
        dw_v = float(d.variables["W_v_eigenvalue"][0, 0])
        dw_t = float(d.variables["W_t_eigenvalue"][0, 0])

        # q_edge: use qmax attribute (≈ Julia info/qlim for monotonic profiles)
        q_edge = float(d.qmax) if hasattr(d, "qmax") else np.nan

    return {
        "param":                param_value,
        "delta_prime_21_real":  dp_21_re,
        "delta_prime_21_imag":  dp_21_im,
        "delta_prime_31_real":  dp_31_re,
        "delta_prime_31_imag":  dp_31_im,
        "delta_W_plasma":       dw_p,
        "delta_W_vacuum":       dw_v,
        "delta_W_total":        dw_t,
        "q_edge":               q_edge,
    }


def aggregate(scan_key: str) -> Path:
    cfg = SCANS[scan_key]
    src_dir: Path = cfg["src_dir"]
    prefix: str   = cfg["prefix"]
    out_csv: Path = cfg["out_csv"]

    if not src_dir.is_dir():
        print(f"[skip] {src_dir} not found", file=sys.stderr)
        return None

    # Collect and sort .nc files by their parameter value.
    pat = re.compile(rf"^{re.escape(prefix)}(-?\d+(?:\.\d+)?)\.nc$")
    entries = []
    for p in src_dir.iterdir():
        m = pat.match(p.name)
        if m:
            entries.append((float(m.group(1)), p))
    entries.sort()

    if not entries:
        print(f"[skip] no files matching {prefix}*.nc in {src_dir}", file=sys.stderr)
        return None

    rows = [extract_row(path, val) for val, path in entries]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            cfg["param_name"],
            "delta_prime_21_real", "delta_prime_21_imag",
            "delta_prime_31_real", "delta_prime_31_imag",
            "delta_W_plasma", "delta_W_vacuum", "delta_W_total",
            "q_edge",
        ])
        for r in rows:
            w.writerow([
                f"{r['param']:.10g}",
                f"{r['delta_prime_21_real']:.10e}", f"{r['delta_prime_21_imag']:.10e}",
                f"{r['delta_prime_31_real']:.10e}", f"{r['delta_prime_31_imag']:.10e}",
                f"{r['delta_W_plasma']:.10e}", f"{r['delta_W_vacuum']:.10e}", f"{r['delta_W_total']:.10e}",
                f"{r['q_edge']:.10e}",
            ])

    print(f"[ok] wrote {len(rows):3d} rows -> {out_csv}")
    return out_csv


if __name__ == "__main__":
    for key in ("beta", "epsilon"):
        aggregate(key)
