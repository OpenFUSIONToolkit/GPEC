#!/usr/bin/env python3
"""
Plot DIII-D STRIDE comparison: Fortran vs Julia Δ' per surface.

Filters out unreliable surfaces (|Im/Re| > 2 in BOTH codes, or m > 8).
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / 'outputs'


def load_fortran():
    with open(OUTPUT_DIR / 'fortran_stride_results.json') as f:
        return {(r['shot'], r['n']): r for r in json.load(f)}


def load_julia():
    results = {}
    if not HAS_H5PY:
        return results
    for d in OUTPUT_DIR.iterdir():
        if not d.is_dir():
            continue
        parts = d.name.rsplit('_n', 1)
        if len(parts) != 2:
            continue
        shot = parts[0]
        try:
            n = int(parts[1])
        except ValueError:
            continue
        h5_path = d / 'gpec.h5'
        if not h5_path.exists():
            continue
        try:
            with h5py.File(str(h5_path), 'r') as f:
                msing = int(f['singular/msing'][()])
                m_sing = f['singular/m'][:]
                q_sing = f['singular/q'][:]
                dp_mat = f['singular/delta_prime_matrix'][:] if 'singular/delta_prime_matrix' in f else None
                ep = f['vacuum/ep'][:]; ev = f['vacuum/ev'][:]; et = f['vacuum/et'][:]
                surfaces = []
                for s in range(msing):
                    m_val = int(m_sing[0, s]) if m_sing.shape[0] < m_sing.shape[1] else int(m_sing[s, 0])
                    dp = dp_mat[s, s] if dp_mat is not None and s < dp_mat.shape[0] else float('nan') + 1j * float('nan')
                    surfaces.append({'m': m_val, 'n': n, 'q': float(q_sing[s]),
                                     'dp_real': float(dp.real), 'dp_imag': float(dp.imag)})
                results[(shot, n)] = {
                    'surfaces': surfaces,
                    'dW_plasma': float(ep[0].real), 'dW_vacuum': float(ev[0].real), 'dW_total': float(et[0].real),
                }
        except Exception as e:
            print(f'Warning: {h5_path}: {e}')
    return results


def is_reliable(f_re, f_im, j_re, j_im, m, max_abs=500):
    """Filter: keep surface if m <= 8, at least one code has |Im/Re| < 2, and |Re| < max_abs."""
    if m > 8:
        return False
    if not (np.isfinite(f_re) and np.isfinite(j_re)):
        return False
    if abs(f_re) > max_abs or abs(j_re) > max_abs:
        return False
    f_ratio = abs(f_im / f_re) if abs(f_re) > 1e-10 else 999
    j_ratio = abs(j_im / j_re) if abs(j_re) > 1e-10 else 999
    return min(f_ratio, j_ratio) < 2


def main():
    fortran = load_fortran()
    julia = load_julia()

    shots = ['147131', '153833', '204441']
    ns = [1, 2]

    fig, axes = plt.subplots(len(shots), len(ns), figsize=(14, 12))
    if len(shots) == 1:
        axes = axes.reshape(1, -1)

    for row, shot in enumerate(shots):
        for col, n in enumerate(ns):
            ax = axes[row, col]
            fk = (shot, n)
            jk = (shot, n)
            fr = fortran.get(fk)
            jr = julia.get(jk)

            if fr is None or jr is None:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{shot} n={n}')
                continue

            # Match surfaces by m
            f_surfs = {s['m']: s for s in fr['surfaces']}
            j_surfs = {s['m']: s for s in jr['surfaces']}
            common = sorted(set(f_surfs) & set(j_surfs))

            ms_plot = []
            f_vals = []
            j_vals = []
            f_im_re = []
            j_im_re = []
            colors = []

            for m in common:
                fs = f_surfs[m]; js = j_surfs[m]
                if not is_reliable(fs['dp_real'], fs['dp_imag'], js['dp_real'], js['dp_imag'], m):
                    continue
                ms_plot.append(m)
                f_vals.append(fs['dp_real'])
                j_vals.append(js['dp_real'])
                fr_ratio = abs(fs['dp_imag'] / fs['dp_real']) if abs(fs['dp_real']) > 1e-10 else 0
                jr_ratio = abs(js['dp_imag'] / js['dp_real']) if abs(js['dp_real']) > 1e-10 else 0
                f_im_re.append(fr_ratio)
                j_im_re.append(jr_ratio)

            if not ms_plot:
                ax.text(0.5, 0.5, 'No reliable surfaces', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{shot} n={n}')
                continue

            ms_plot = np.array(ms_plot)
            f_vals = np.array(f_vals)
            j_vals = np.array(j_vals)

            x = np.arange(len(ms_plot))
            w = 0.35
            bars_f = ax.bar(x - w/2, f_vals, w, label='Fortran', color='C0', alpha=0.8, edgecolor='black', linewidth=0.5)
            bars_j = ax.bar(x + w/2, j_vals, w, label='Julia', color='C1', alpha=0.8, edgecolor='black', linewidth=0.5)

            ax.set_xticks(x)
            ax.set_xticklabels([f'{m}/{n}' for m in ms_plot], fontsize=8)
            ax.set_xlabel('m/n')
            ax.set_ylabel(r"Re($\Delta'$)")
            ax.axhline(0, color='k', lw=0.5, ls='-')
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.2, axis='y')

            # Add percentage error annotations
            for i in range(len(ms_plot)):
                err = 100 * abs(j_vals[i] - f_vals[i]) / max(abs(f_vals[i]), 1e-10)
                ypos = max(f_vals[i], j_vals[i]) if max(f_vals[i], j_vals[i]) > 0 else min(f_vals[i], j_vals[i])
                offset = 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]) if ypos >= 0 else -0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                color = 'green' if err < 2 else ('orange' if err < 5 else 'red')
                ax.annotate(f'{err:.1f}%', (x[i], ypos), textcoords='offset points',
                           xytext=(0, 8 if ypos >= 0 else -12), ha='center', fontsize=6, color=color, weight='bold')

            # Title with dW info
            f_dW = fr.get('dW_total', float('nan'))
            j_dW = jr.get('dW_total', float('nan'))
            dW_err = 100 * abs(j_dW - f_dW) / max(abs(f_dW), 1e-10) if np.isfinite(f_dW) and np.isfinite(j_dW) else float('nan')
            title = f'{shot} n={n}'
            if np.isfinite(dW_err) and dW_err < 50:
                title += f'  (δW err={dW_err:.1f}%)'
            ax.set_title(title, fontsize=10, fontweight='bold')

    fig.suptitle(r"DIII-D $\Delta'$ Comparison: Fortran STRIDE vs Julia GPEC" + "\n(reliable surfaces only: m ≤ 8, min |Im/Re| < 2)",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()

    outfile = OUTPUT_DIR / 'diiid_comparison_bars.png'
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Saved: {outfile}')

    # Scatter plot with inset showing the Fortran outlier
    fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5.5),
                               gridspec_kw={'width_ratios': [1.2, 0.6, 1]})
    ax_main, ax_outlier, ax_hist = axes2

    all_f = []; all_j = []
    color_map = {'147131': 'C0', '153833': 'C2', '204441': 'C3'}
    marker_map = {1: 'o', 2: 's'}

    # Collect reliable points
    for shot in shots:
        for n in ns:
            fr = fortran.get((shot, n))
            jr = julia.get((shot, n))
            if fr is None or jr is None:
                continue
            f_surfs = {s['m']: s for s in fr['surfaces']}
            j_surfs = {s['m']: s for s in jr['surfaces']}
            fv = []; jv = []
            for m in sorted(set(f_surfs) & set(j_surfs)):
                fs = f_surfs[m]; js = j_surfs[m]
                if not is_reliable(fs['dp_real'], fs['dp_imag'], js['dp_real'], js['dp_imag'], m):
                    continue
                fv.append(fs['dp_real']); jv.append(js['dp_real'])
                all_f.append(fs['dp_real']); all_j.append(js['dp_real'])
            if fv:
                ax_main.scatter(fv, jv, c=color_map.get(shot, 'grey'), marker=marker_map.get(n, 'o'),
                               s=50, alpha=0.8, edgecolors='black', linewidths=0.5,
                               label=f'{shot} n={n}', zorder=5)

    all_f = np.array(all_f); all_j = np.array(all_j)
    lims = [min(all_f.min(), all_j.min()) * 1.15, max(all_f.max(), all_j.max()) * 1.15]
    ax_main.plot(lims, lims, 'k--', lw=0.8, alpha=0.5, label='1:1')
    ax_main.set_xlabel(r"Fortran Re($\Delta'$)", fontsize=11)
    ax_main.set_ylabel(r"Julia Re($\Delta'$)", fontsize=11)
    ax_main.set_title(r"$\Delta'$ parity (reliable surfaces)", fontsize=11)
    ax_main.legend(fontsize=7, loc='upper left')
    ax_main.grid(True, alpha=0.3)
    ax_main.set_aspect('equal')
    ax_main.set_xlim(lims); ax_main.set_ylim(lims)

    # Middle panel: Fortran outlier — 204441 n=2 3/2 surface
    # Fortran: +1227, Julia: +2.7 — show both on a bar chart with broken axis feel
    fr_204_n2 = fortran.get(('204441', 2))
    jr_204_n2 = julia.get(('204441', 2))
    if fr_204_n2 and jr_204_n2:
        f32 = next((s for s in fr_204_n2['surfaces'] if s['m'] == 3), None)
        j32 = next((s for s in jr_204_n2['surfaces'] if s['m'] == 3), None)
        if f32 and j32:
            f_val = f32['dp_real']
            j_val = j32['dp_real']
            f_im_re = abs(f32['dp_imag'] / f32['dp_real']) if abs(f32['dp_real']) > 0 else 0
            j_im_re = abs(j32['dp_imag'] / j32['dp_real']) if abs(j32['dp_real']) > 0 else 0

            bars = ax_outlier.bar([0, 1], [f_val, j_val], color=['C0', 'C1'],
                                  alpha=0.8, edgecolor='black', linewidth=0.8, width=0.6)
            ax_outlier.set_xticks([0, 1])
            ax_outlier.set_xticklabels(['Fortran', 'Julia'], fontsize=9)
            ax_outlier.set_ylabel(r"Re($\Delta'_{3/2}$)", fontsize=10)
            ax_outlier.set_title('204441 n=2, 3/2 surface\n(Fortran diverged)', fontsize=9, fontweight='bold', color='red')

            # Annotate values
            ax_outlier.annotate(f'{f_val:+.0f}\n|Im/Re|={f_im_re:.2f}',
                               (0, f_val), textcoords='offset points', xytext=(0, 8),
                               ha='center', fontsize=8, color='C0', weight='bold')
            ax_outlier.annotate(f'{j_val:+.1f}\n|Im/Re|={j_im_re:.3f}',
                               (1, j_val), textcoords='offset points', xytext=(0, 8),
                               ha='center', fontsize=8, color='C1', weight='bold')

            # Add Fortran dW annotation
            ax_outlier.text(0.5, 0.02,
                           f'Fortran δW = {fr_204_n2["dW_total"]:+.0f}\nJulia δW = {jr_204_n2["dW_total"]:+.3f}',
                           transform=ax_outlier.transAxes, ha='center', va='bottom', fontsize=7,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))
            ax_outlier.grid(True, alpha=0.2, axis='y')

    # Right panel: error histogram
    errs = 100 * np.abs(all_j - all_f) / np.maximum(np.abs(all_f), 1e-10)
    ax_hist.hist(errs, bins=np.logspace(-1, 2, 25), color='C0', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax_hist.set_xscale('log')
    ax_hist.axvline(np.median(errs), color='red', ls='--', lw=1.5, label=f'median = {np.median(errs):.1f}%')
    ax_hist.axvline(2, color='green', ls=':', lw=1, alpha=0.7, label='2% target')
    ax_hist.set_xlabel('Relative error (%)', fontsize=11)
    ax_hist.set_ylabel('Count', fontsize=11)
    ax_hist.set_title(f'Error distribution\n({len(errs)} reliable surfaces)', fontsize=11)
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    fig2.suptitle("DIII-D Δ' Comparison: Julia GPEC vs Fortran STRIDE",
                  fontsize=13, fontweight='bold')
    fig2.tight_layout()

    outfile2 = OUTPUT_DIR / 'diiid_comparison_scatter.png'
    fig2.savefig(outfile2, dpi=150, bbox_inches='tight')
    print(f'Saved: {outfile2}')


if __name__ == '__main__':
    main()
