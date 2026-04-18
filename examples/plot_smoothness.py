#!/usr/bin/env python3
"""
Assess and compare the smoothness of Fortran vs Julia Δ' scan results.

Uses a local polynomial residual: fit a quadratic through a sliding window
of 5 points, measure the deviation of the center point from the fit.
A smooth curve has near-zero residuals; erratic jumps produce large residuals.

Usage:
    python3 plot_smoothness.py
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / 'scan_comparison_plots'

def load(path):
    return np.genfromtxt(str(path), delimiter=',', names=True)

def local_poly_residual(x, y, window=5):
    """Compute local polynomial fit residuals.

    For each interior point, fit a quadratic through the surrounding `window`
    points and return the absolute residual of the center point from the fit.
    Points near the edges use asymmetric windows.
    """
    n = len(x)
    half = window // 2
    residuals = np.full(n, np.nan)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        if hi - lo < 3:
            continue
        xi = x[lo:hi]
        yi = y[lo:hi]
        mask = np.isfinite(yi)
        if np.sum(mask) < 3:
            continue
        # Fit polynomial: degree = min(2, npts-2) so there's always at least 1 residual DOF
        deg = min(2, np.sum(mask) - 2)
        if deg < 1:
            continue
        coeffs = np.polyfit(xi[mask], yi[mask], deg)
        y_fit = np.polyval(coeffs, x[i])
        residuals[i] = abs(y[i] - y_fit)

    return residuals

def relative_residual(x, y, window=5):
    """Local polynomial residual normalized by local |y| scale."""
    res = local_poly_residual(x, y, window)
    # Normalize by local absolute value (rolling max over window)
    half = window // 2
    n = len(y)
    scale = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        vals = np.abs(y[lo:hi])
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            scale[i] = max(np.median(vals), 1e-10)
    return res / scale

def plot_smoothness(window=3):
    # Load data
    eps_j = load(ROOT / 'LAR_epsilon_scan' / 'outputs' / 'julia_epsilon_scan_summary.csv')
    eps_f = load(ROOT / 'LAR_epsilon_scan' / 'reference' / 'epsilon_scan_summary.csv')
    beta_j = load(ROOT / 'LAR_beta_scan' / 'outputs' / 'julia_beta_scan_summary.csv')
    beta_f = load(ROOT / 'LAR_beta_scan' / 'reference' / 'pressure_factor_scan_summary.csv')

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    datasets = [
        ('Pressure factor', beta_f, beta_j, 'pressure_factor'),
        (r'$\varepsilon = a/R_0$', eps_f, eps_j, 'epsilon'),
    ]

    for col, (xlabel, fd, jd, pcol) in enumerate(datasets):
        xf = fd[pcol]
        xj = jd[pcol]

        for row, (qty_name, qty_key) in enumerate([
            (r"$\Delta'_{2/1}$", 'delta_prime_21_real'),
            (r"$\Delta'_{3/1}$", 'delta_prime_31_real'),
            (r"$\delta W_{total}$", 'delta_W_total'),
        ]):
            ax = axes[row, col]

            yf = fd[qty_key]
            yj = jd[qty_key]

            # Compute relative residuals
            mask_f = np.isfinite(yf)
            mask_j = np.isfinite(yj)

            res_f = relative_residual(xf[mask_f], yf[mask_f], window=window)
            res_j = relative_residual(xj[mask_j], yj[mask_j], window=window)

            # Plot
            ax.semilogy(xf[mask_f], res_f * 100, 'o-', color='C0', ms=3, lw=1,
                        label='Fortran', alpha=0.8)
            ax.semilogy(xj[mask_j], res_j * 100, 's-', color='C1', ms=3, lw=1,
                        label='Julia', alpha=0.8)

            ax.set_xlabel(xlabel)
            ax.set_ylabel('Local fit residual (%)')
            ax.set_title(f'{qty_name} smoothness')
            ax.legend(frameon=False, fontsize=8)
            ax.grid(True, alpha=0.3, which='both')
            ax.set_ylim(1e-4, 200)
            ax.axhline(1, color='grey', ls=':', lw=0.5, alpha=0.5)

            # Compute and annotate median residual
            med_f = np.nanmedian(res_f) * 100
            med_j = np.nanmedian(res_j) * 100
            ax.text(0.98, 0.95, f'median: F={med_f:.3f}%, J={med_j:.3f}%',
                    transform=ax.transAxes, ha='right', va='top', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    fig.suptitle(f'Scan Smoothness: Local Quadratic Fit Residual ({window}-point window)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()

    outfile = OUTPUT_DIR / f'smoothness_comparison_w{window}.png'
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Saved: {outfile}')

    # Also print summary table
    print('\nSmoothness Summary (median local fit residual %):')
    print(f'{"Scan":>10s} {"Quantity":>12s} {"Fortran":>10s} {"Julia":>10s} {"Winner":>10s}')
    print('-' * 55)
    for scan_name, fd, jd, pcol in [
        ('Beta', beta_f, beta_j, 'pressure_factor'),
        ('Epsilon', eps_f, eps_j, 'epsilon'),
    ]:
        for qty_name, qty_key in [
            ("dp21", 'delta_prime_21_real'),
            ("dp31", 'delta_prime_31_real'),
            ("dW_total", 'delta_W_total'),
        ]:
            xf = fd[pcol]; yf = fd[qty_key]
            xj = jd[pcol]; yj = jd[qty_key]
            mf = np.isfinite(yf); mj = np.isfinite(yj)
            res_f = np.nanmedian(relative_residual(xf[mf], yf[mf], window=window) * 100)
            res_j = np.nanmedian(relative_residual(xj[mj], yj[mj], window=window) * 100)
            winner = 'Julia' if res_j < res_f else 'Fortran'
            print(f'{scan_name:>10s} {qty_name:>12s} {res_f:9.4f}% {res_j:9.4f}% {winner:>10s}')


if __name__ == '__main__':
    for w in [3, 4, 5]:
        plot_smoothness(window=w)
