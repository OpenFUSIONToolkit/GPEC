#!/usr/bin/env python3
# plot_robustness_multipanel.py -- one clean 1x3 "Numerical Robustness" figure built entirely from CSVs:
#   (A) radial truncation      : delta_coil convergence metric vs psihigh (STRIDE value -> edge value)
#   (B) singular-value trunc.  : SVD spectrum sigma_i, nominal vs dmlim/tol perturbations (subspace invariant)
#   (C) resistivity            : matched penetrated field ||bpen|| vs eta (response saturates to ideal limit)
# STRIDE-only self-convergence (not a STRIDE-vs-Galerkin benchmark): each panel rebuts a different
# "is this just a numerical artifact?" objection.
#
# Usage: python3 plot_robustness_multipanel.py            # LAR defaults
#        python3 plot_robustness_multipanel.py <metric.csv> <svd_tol_result.txt> <resistivity_scan.csv>

import sys, os, csv, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
res  = os.path.join(here, '..', 'results')
figd = os.path.join(here, '..', 'figures')
os.makedirs(figd, exist_ok=True)

metric_csv = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(res, 'deltacoil_psihigh_metric_lar.csv')
svd_txt    = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(res, 'deltacoil_svd_tol_result.txt')
resist_csv = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else os.path.join(res, 'resistivity_scan_LARresistivematchtest_f1p0.csv')
# optional Fourier-truncation panel (deltacoil_mhigh_scan.jl output); folded in as panel (B) when present
mhigh_csv  = os.path.abspath(sys.argv[4]) if len(sys.argv) > 4 else os.path.join(res, 'deltacoil_mhigh_LARresistivematchtest.csv')

COLORS = plt.cm.viridis(np.linspace(0.10, 0.80, 6))


def read_metric(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    head = rows[0]
    qcols = head[1:]
    x, ys = [], {c: [] for c in qcols}
    for r in rows[1:]:
        if not r or r[0] == '':
            continue
        x.append(float(r[0]))
        for i, c in enumerate(qcols):
            v = r[i + 1]
            ys[c].append(float(v) if v not in ('', 'FAIL') else np.nan)
    return np.array(x), {c: np.array(v, float) for c, v in ys.items()}


def read_svd_spectrum(path):
    # parse the "singular-value spectrum sigma_i (first 6)" table: run | s1 | s2 | ...
    runs = {}
    in_block = False
    with open(path) as f:
        for line in f:
            if 'singular-value spectrum' in line:
                in_block = True
                continue
            if in_block:
                if '|' not in line:
                    continue
                parts = [p.strip() for p in line.split('|')]
                name = parts[0]
                if name.lower().startswith('run') or name == '':
                    continue
                vals = []
                for p in parts[1:]:
                    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', p)
                    vals.append(float(m.group()) if m else np.nan)
                if vals:
                    runs[name] = np.array(vals, float)
    return runs


def read_resist(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    head = rows[0]
    idx = {c: i for i, c in enumerate(head)}
    qvals = sorted({float(c.split('_q')[1]) for c in head[1:] if '_q' in c})
    eta, bpen = [], {q: [] for q in qvals}
    for r in rows[1:]:
        if not r or r[0] == '':
            continue
        eta.append(float(r[0]))
        for q in qvals:
            col = f'gal_bpen_q{q}'
            v = r[idx[col]] if col in idx else ''
            bpen[q].append(float(v) if v not in ('', 'FAIL') else np.nan)
    return np.array(eta), qvals, {q: np.array(v, float) for q, v in bpen.items()}


fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.5, 5.2))

# ---- Panel A: radial truncation (psihigh) ----
xm, ym = read_metric(metric_csv)
for i, (c, y) in enumerate(ym.items()):
    q = c.replace('q', '').rstrip('0').rstrip('.')
    m = ~np.isnan(y)
    axA.plot(xm[m], y[m], 'o-', color=COLORS[i * 2], lw=2, ms=5, label=f'q = {q}')
axA.axhline(1.0, color='0.6', ls=':', lw=1.2)
axA.set_xlabel('edge truncation  psihigh', fontsize=11)
axA.set_ylabel('delta_coil metric  (normalized to edge value)', fontsize=11)
axA.set_title('(A) Radial truncation', fontsize=12, fontweight='bold')
axA.legend(fontsize=9, loc='lower right')
axA.grid(alpha=0.25)

# ---- Panel B: singular-value truncation (SVD) ----
spec = read_svd_spectrum(svd_txt)
order = ['nominal', 'tol=1.0e-8', 'tol=1.0e-10', 'tol=1.0e-12', 'dmlim=0.42', 'dmlim=0.3', 'dmlim=0.5']
styles = {'nominal': dict(color='k', lw=3, ms=8, marker='o', zorder=1, label='nominal'),
          'dmlim=0.5': dict(color='crimson', lw=1.8, ls='--', ms=6, marker='v', zorder=3,
                            label='dmlim=0.5 (drops a mode)')}
overlay = dict(color='tab:cyan', lw=0, ms=6, marker='+', mew=1.6, zorder=4)
labeled_overlay = False
for name in order:
    if name not in spec:
        continue
    s = spec[name]
    xi = np.arange(1, len(s) + 1)
    if name in styles:
        axB.plot(xi, s, **styles[name])
    else:
        kw = dict(overlay)
        if not labeled_overlay:
            kw['label'] = 'tol 1e-8..1e-12, dmlim=0.42 (overlaid)'
            labeled_overlay = True
        axB.plot(xi, s, **kw)
axB.set_yscale('log')
axB.set_xlabel('singular-value index  i', fontsize=11)
axB.set_ylabel('sigma_i  of delta_coil matrix  (log)', fontsize=11)
axB.set_title('(B) Singular-value truncation', fontsize=12, fontweight='bold')
axB.legend(fontsize=9, loc='upper right')
axB.grid(alpha=0.25, which='both')

# ---- Panel C: resistivity (eta) ----
eta, qvals, bpen = read_resist(resist_csv)
for i, q in enumerate(qvals):
    y = bpen[q]
    m = ~np.isnan(y)
    qs = str(q).rstrip('0').rstrip('.')
    axC.plot(eta[m], y[m], 'o-', color=COLORS[i * 2], lw=2, ms=5, label=f'q = {qs}')
axC.set_xscale('log'); axC.set_yscale('log')
axC.invert_xaxis()
axC.set_xlabel('resistivity  eta   (low eta = high |Q|, to the right)', fontsize=11)
axC.set_ylabel('matched penetrated field  ||bpen||  (log)', fontsize=11)
axC.set_title('(C) Resistivity', fontsize=12, fontweight='bold')
axC.legend(fontsize=9, loc='best')
axC.grid(alpha=0.25, which='both')

fig.suptitle('Numerical robustness of STRIDE delta_coil: truncation & resistivity scans  (LAR, n=1)',
             fontsize=13.5, fontweight='bold')
fig.subplots_adjust(left=0.055, right=0.985, bottom=0.135, top=0.86, wspace=0.28)

out = os.path.join(figd, 'robustness_multipanel_LAR.png')
fig.savefig(out, dpi=150)
print('wrote:', os.path.abspath(out))
