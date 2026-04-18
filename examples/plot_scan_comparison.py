#!/usr/bin/env python3
"""
Plot Julia vs Fortran STRIDE scan results: Delta' and delta-W.

Produces three figures matching the Fortran circular_test_scans branch style:
  1. Beta (pressure factor) scan — 3-panel (Delta', delta-W, q_edge)
  2. Epsilon scan — 3-panel (Delta', delta-W, q_edge)
  3. Both scans — 2x2 (Delta' and delta-W, side by side)

Fortran data shown as connected lines+markers; Julia overlaid as scatter points.

Usage:
    python plot_scan_comparison.py
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
JULIA_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = JULIA_ROOT / 'scan_comparison_plots'

JULIA_CSV = {
    'beta':    JULIA_ROOT / 'LAR_beta_scan'    / 'outputs' / 'julia_beta_scan_summary.csv',
    'epsilon': JULIA_ROOT / 'LAR_epsilon_scan' / 'outputs' / 'julia_epsilon_scan_summary.csv',
}
# Fortran reference CSVs aggregated from STRIDE .nc outputs by
# examples/aggregate_fortran_scans.py (run it once to populate):
FORTRAN_CSV = {
    'beta':    JULIA_ROOT / 'LAR_beta_scan'    / 'reference' / 'pressure_factor_scan_summary.csv',
    'epsilon': JULIA_ROOT / 'LAR_epsilon_scan' / 'reference' / 'epsilon_scan_summary.csv',
}

PARAM_COL   = {'beta': 'pressure_factor', 'epsilon': 'epsilon'}
PARAM_LABEL = {'beta': r'Pressure factor', 'epsilon': r'$\varepsilon = a/R_0$'}


def load(path):
    return np.genfromtxt(str(path), delimiter=',', names=True)


def plot_single_scan(fortran, julia, scan_type, save=True):
    """3-panel figure: |Delta'|, delta-W, q_edge — Fortran lines + Julia scatter."""
    xf = fortran[PARAM_COL[scan_type]]
    xj = julia[PARAM_COL[scan_type]]
    xlabel = PARAM_LABEL[scan_type]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel 1: |Delta'| (log) ───────────────────────────────────────────
    ax = axes[0]
    # Fortran lines
    ax.semilogy(xf, np.abs(fortran['delta_prime_21_real']),
                'o-', color='C0', ms=3, lw=1.2, label="2/1 Fortran")
    ax.semilogy(xf, np.abs(fortran['delta_prime_31_real']),
                's-', color='C1', ms=3, lw=1.2, label="3/1 Fortran")
    # Julia scatter
    mask_j21 = np.isfinite(julia['delta_prime_21_real'])
    mask_j31 = np.isfinite(julia['delta_prime_31_real'])
    ax.semilogy(xj[mask_j21], np.abs(julia['delta_prime_21_real'][mask_j21]),
                'D', color='#1f77b4', ms=5, mfc='none', mew=1.2, label="2/1 Julia", zorder=5)
    ax.semilogy(xj[mask_j31], np.abs(julia['delta_prime_31_real'][mask_j31]),
                'D', color='#ff7f0e', ms=5, mfc='none', mew=1.2, label="3/1 Julia", zorder=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$|\Delta'|$")
    ax.set_title(r"$\Delta'$ (real)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # ── Panel 2: delta-W ──────────────────────────────────────────────────
    ax = axes[1]
    # Fortran lines
    ax.plot(xf, fortran['delta_W_plasma'], 's-', color='C1', ms=3, lw=1.2, label=r'$\delta W_p$ Fortran')
    ax.plot(xf, fortran['delta_W_vacuum'], '^-', color='C2', ms=3, lw=1.2, label=r'$\delta W_v$ Fortran')
    ax.plot(xf, fortran['delta_W_total'],  'o-', color='C3', ms=3, lw=1.2, label=r'$\delta W_t$ Fortran')
    # Julia scatter
    mask_wp = np.isfinite(julia['delta_W_plasma'])
    ax.plot(xj[mask_wp], julia['delta_W_plasma'][mask_wp],
            'D', color='#ff7f0e', ms=5, mfc='none', mew=1.0, label=r'$\delta W_p$ Julia', zorder=5)
    ax.plot(xj[mask_wp], julia['delta_W_vacuum'][mask_wp],
            'D', color='#2ca02c', ms=5, mfc='none', mew=1.0, label=r'$\delta W_v$ Julia', zorder=5)
    ax.plot(xj[mask_wp], julia['delta_W_total'][mask_wp],
            'D', color='#d62728', ms=5, mfc='none', mew=1.0, label=r'$\delta W_t$ Julia', zorder=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$\delta W$')
    ax.set_title('Energy components')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: q_edge ───────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(xf, fortran['q_edge'], 'D-', color='C4', ms=3, lw=1.2, label='Fortran')
    mask_q = np.isfinite(julia['q_edge'])
    ax.plot(xj[mask_q], julia['q_edge'][mask_q],
            'D', color='#9467bd', ms=5, mfc='none', mew=1.2, label='Julia', zorder=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$q_{\mathrm{edge}}$')
    ax.set_title('Edge safety factor')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)

    scan_name = 'Pressure Factor' if scan_type == 'beta' else 'Epsilon'
    fig.suptitle(f'STRIDE Benchmark — {scan_name} Scan: Fortran vs Julia',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()

    if save:
        outfile = OUTPUT_DIR / f'{scan_type}_scan_comparison.png'
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f'Saved: {outfile}')
    return fig


def plot_both(f_beta, j_beta, f_eps, j_eps, save=True):
    """2x2 figure matching Fortran both_scans_results.png, with Julia overlaid."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    datasets = [
        ('beta',    f_beta,  j_beta,  f_beta['pressure_factor'],  j_beta['pressure_factor']),
        ('epsilon', f_eps,   j_eps,   f_eps['epsilon'],           j_eps['epsilon']),
    ]

    for col, (stype, fd, jd, xf, xj) in enumerate(datasets):
        xlabel = PARAM_LABEL[stype]

        # ── Top row: |Delta'| (log) ──────────────────────────────────────
        ax = axes[0, col]
        ax.semilogy(xf, np.abs(fd['delta_prime_21_real']),
                    'o-', color='C0', ms=3, lw=1.2, label='2/1 Fortran')
        ax.semilogy(xf, np.abs(fd['delta_prime_31_real']),
                    's-', color='C1', ms=3, lw=1.2, label='3/1 Fortran')
        m21 = np.isfinite(jd['delta_prime_21_real'])
        m31 = np.isfinite(jd['delta_prime_31_real'])
        ax.semilogy(xj[m21], np.abs(jd['delta_prime_21_real'][m21]),
                    'D', color='#1f77b4', ms=5, mfc='none', mew=1.2, label='2/1 Julia', zorder=5)
        ax.semilogy(xj[m31], np.abs(jd['delta_prime_31_real'][m31]),
                    'D', color='#ff7f0e', ms=5, mfc='none', mew=1.2, label='3/1 Julia', zorder=5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$|\Delta'|$")
        ax.set_title(r"$\Delta'$ (real)")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(True, alpha=0.3, which='both')

        # ── Bottom row: delta-W ──────────────────────────────────────────
        ax = axes[1, col]
        ax.plot(xf, fd['delta_W_plasma'], 's-', color='C1', ms=3, lw=1.2, label=r'$\delta W_p$ Fortran')
        ax.plot(xf, fd['delta_W_vacuum'], '^-', color='C2', ms=3, lw=1.2, label=r'$\delta W_v$ Fortran')
        ax.plot(xf, fd['delta_W_total'],  'o-', color='C3', ms=3, lw=1.2, label=r'$\delta W_t$ Fortran')
        mw = np.isfinite(jd['delta_W_plasma'])
        ax.plot(xj[mw], jd['delta_W_plasma'][mw],
                'D', color='#ff7f0e', ms=5, mfc='none', mew=1.0, label=r'$\delta W_p$ Julia', zorder=5)
        ax.plot(xj[mw], jd['delta_W_vacuum'][mw],
                'D', color='#2ca02c', ms=5, mfc='none', mew=1.0, label=r'$\delta W_v$ Julia', zorder=5)
        ax.plot(xj[mw], jd['delta_W_total'][mw],
                'D', color='#d62728', ms=5, mfc='none', mew=1.0, label=r'$\delta W_t$ Julia', zorder=5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$\delta W$')
        ax.set_title('Energy components')
        ax.axhline(0, color='k', lw=0.5, ls='--')
        ax.legend(frameon=False, fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle('STRIDE Benchmark Scans: Fortran vs Julia',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()

    if save:
        outfile = OUTPUT_DIR / 'both_scans_comparison.png'
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f'Saved: {outfile}')
    return fig


def plot_error_summary(f_beta, j_beta, f_eps, j_eps, save=True):
    """2x2 figure: percentage error in dp21, dp31 for both scans."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    datasets = [
        ('beta',    f_beta,  j_beta,  'pressure_factor'),
        ('epsilon', f_eps,   j_eps,   'epsilon'),
    ]

    for col, (stype, fd, jd, pcol) in enumerate(datasets):
        xf = fd[pcol]
        xj = jd[pcol]
        xlabel = PARAM_LABEL[stype]

        # Match points
        errs_21 = []
        errs_31 = []
        errs_dw = []
        xs = []
        for i, xjv in enumerate(xj):
            idx = np.argmin(np.abs(xf - xjv))
            if abs(xf[idx] - xjv) > 1e-6:
                continue
            j21 = jd['delta_prime_21_real'][i]
            f21 = fd['delta_prime_21_real'][idx]
            j31 = jd['delta_prime_31_real'][i]
            f31 = fd['delta_prime_31_real'][idx]
            jdw = jd['delta_W_total'][i]
            fdw = fd['delta_W_total'][idx]
            if not (np.isfinite(j21) and np.isfinite(f21)):
                continue
            xs.append(xjv)
            errs_21.append(100 * abs(j21 - f21) / max(abs(f21), 1e-30))
            errs_31.append(100 * abs(j31 - f31) / max(abs(f31), 1e-30) if np.isfinite(j31) and np.isfinite(f31) else np.nan)
            errs_dw.append(100 * abs(jdw - fdw) / max(abs(fdw), 1e-30) if np.isfinite(jdw) and np.isfinite(fdw) else np.nan)

        xs = np.array(xs)
        errs_21 = np.array(errs_21)
        errs_31 = np.array(errs_31)
        errs_dw = np.array(errs_dw)

        # Top: dp21 and dp31 error
        ax = axes[0, col]
        ax.semilogy(xs, errs_21, 'o-', color='C0', ms=4, lw=1, label=r"$\Delta'_{2/1}$")
        m31 = np.isfinite(errs_31)
        ax.semilogy(xs[m31], errs_31[m31], 's-', color='C1', ms=4, lw=1, label=r"$\Delta'_{3/1}$")
        ax.axhline(1, color='k', lw=0.5, ls='--', alpha=0.5)
        ax.axhline(5, color='grey', lw=0.5, ls=':', alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Relative error (%)')
        ax.set_title(r"$\Delta'$ error")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim(0.01, 500)

        # Bottom: dW error
        ax = axes[1, col]
        mdw = np.isfinite(errs_dw)
        ax.semilogy(xs[mdw], errs_dw[mdw], 'o-', color='C3', ms=4, lw=1, label=r'$\delta W_t$')
        ax.axhline(1, color='k', lw=0.5, ls='--', alpha=0.5)
        ax.axhline(5, color='grey', lw=0.5, ls=':', alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Relative error (%)')
        ax.set_title(r'$\delta W$ error')
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim(0.01, 500)

    fig.suptitle('Julia vs Fortran: Percentage Error',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()

    if save:
        outfile = OUTPUT_DIR / 'error_summary.png'
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f'Saved: {outfile}')
    return fig


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    f_beta = load(FORTRAN_CSV['beta'])
    j_beta = load(JULIA_CSV['beta'])
    f_eps  = load(FORTRAN_CSV['epsilon'])
    j_eps  = load(JULIA_CSV['epsilon'])

    plot_single_scan(f_beta, j_beta, 'beta')
    plot_single_scan(f_eps,  j_eps,  'epsilon')
    plot_both(f_beta, j_beta, f_eps, j_eps)
    plot_error_summary(f_beta, j_beta, f_eps, j_eps)

    print(f'\nAll figures saved to: {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
