#!/usr/bin/env python3
# plot_psihigh_fig1style_rg.py -- Task-1 / Fig-1a-b style view of the Riccati-vs-Galerkin edge scan:
# all surfaces overlaid on ONE magnitude axis (left) plus the relative-change convergence view (right),
# from the 100-point deltacoil_psihigh_rg_<case>.csv (psihigh 0.9 -> 0.9999). Color = surface;
# STRIDE (Riccati) solid+circles, Galerkin (RDCON) dashed+squares. Surfaces that ENTER within the
# window get dotted grid-entry + dashed true-psi_s markers; surfaces already present at the left edge
# are interior (psi_s < 0.9, off-screen left) and their psi_s is noted in the legend.
# Usage: plot_psihigh_fig1style_rg.py <rg_csv> [--qmax=Q] [--psi_s="2:0.49,3:0.886,..."]

import sys, os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

QMAX = None; TRUE = {}; pos = []
for a in sys.argv[1:]:
    if a.startswith('--qmax='):
        QMAX = float(a.split('=', 1)[1])
    elif a.startswith('--psi_s='):
        for kv in a.split('=', 1)[1].split(','):
            if ':' in kv:
                k, v = kv.split(':'); TRUE[float(k)] = float(v)
    else:
        pos.append(a)
CSVP = os.path.abspath(pos[0]); here = os.path.dirname(os.path.abspath(__file__))
_tag = os.path.splitext(os.path.basename(CSVP))[0].replace('deltacoil_psihigh_rg_', '')
CASE = {'LARresistivematchtest': 'LAR (large-aspect-ratio)',
        'DIIIDlikegalresistivepeexample': 'DIII-D-like'}.get(_tag, _tag)

rows = list(csv.reader(open(CSVP))); h = rows[0]; idx = {c: i for i, c in enumerate(h)}
ph = np.array([float(r[0]) for r in rows[1:] if r and r[0]])
qs = sorted({c.split('_q')[1] for c in h if c.startswith('norm_ric_q')}, key=float)
if QMAX is not None:
    qs = [q for q in qs if float(q) <= QMAX + 1e-9]

def col(n):
    j = idx.get(n); out = []
    for r in rows[1:]:
        if not r or not r[0]:
            continue
        v = r[j] if (j is not None and j < len(r)) else ''
        out.append(float(v) if v not in ('', 'FAIL', 'NaN') else np.nan)
    return np.array(out)

GX0, GXM = ph.min(), ph.max()
def edge_log_x(ax):
    fwd = lambda p: -np.log10(np.clip(1.0 - np.asarray(p, float), 1e-9, None))
    inv = lambda t: 1.0 - 10.0 ** (-np.asarray(t, float))
    ax.set_xscale('function', functions=(fwd, inv))
    cand = np.array([0.9, 0.99, 0.999, 0.9999])
    tk = cand[(cand >= GX0 - 1e-9) & (cand <= GXM + 1e-9)]
    ax.set_xticks(tk); ax.set_xticklabels([f'{t:g}' for t in tk]); ax.minorticks_off()
    ax.set_xlim(GX0 - (1 - GX0) * 0.06, GXM + (1 - GXM) * 0.4)

colors = plt.cm.viridis(np.linspace(0.08, 0.82, len(qs)))
fig, (axM, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4))

for q, c in zip(qs, colors):
    qf = float(q)
    yr = col(f'norm_ric_q{q}'); yg = col(f'norm_gal_q{q}')
    interior = np.isfinite(yr[np.argmin(ph)])                      # present at the leftmost psihigh?
    lab = f'q = {qf:g}'
    if interior and qf in TRUE:
        lab += f'  (ψ$_s$≈{TRUE[qf]:.2f}, interior)'
    mR = np.isfinite(yr); mG = np.isfinite(yg)
    axM.plot(ph[mR], yr[mR], '-', color=c, lw=1.8, label=lab)
    axM.plot(ph[mG], yg[mG], '--', color=c, lw=1.2, alpha=0.85)
    # markers only for surfaces that enter within the window (not interior/off-left)
    if not interior:
        ent = ph[mR].min()
        axM.axvline(ent, ls=':', color=c, lw=1.0, alpha=0.7)
        t = f' q={qf:g} enters ≈{ent:.4f}'
        if qf in TRUE:
            axM.axvline(TRUE[qf], ls='--', color=c, lw=1.0, alpha=0.5)
            t += f' (ψ$_s$≈{TRUE[qf]:.4f})'
        axM.text(ent, axM.get_ylim()[1], t, color=c, fontsize=7, rotation=90, va='top', ha='left')
    # relative change vs the outermost (edge-most) value, STRIDE
    if mR.any():
        ref = yr[mR][-1]
        axR.plot(ph[mR], np.abs(yr[mR] - ref) / max(abs(ref), 1e-300), '-', color=c, lw=1.8, label=f'q = {qf:g}')

for ax in (axM, axR):
    ax.set_yscale('log'); edge_log_x(ax); ax.grid(alpha=0.25, which='both')
    ax.set_xlabel('outer truncation  ψ$_{high}$   (log distance from edge; edge at right)', fontsize=10)
axM.set_ylabel('|delta_coil|  per surface  (log)', fontsize=11)
axR.set_ylabel('relative change vs ψ$_{high}$=max  (log; lower = converged)', fontsize=10)
axM.set_title('|delta_coil| vs ψ$_{high}$  (all surfaces, one axis)', fontsize=11)
axR.set_title('Convergence: relative sensitivity to the boundary (Riccati)', fontsize=11)
# legends
h1 = axM.get_legend_handles_labels()
axM.legend(*h1, fontsize=8, title='rational surface', title_fontsize=8, loc='lower right')
style = [Line2D([0], [0], color='#444', ls='-', label='Riccati (STRIDE)'),
         Line2D([0], [0], color='#444', ls='--', label='Galerkin (RDCON)')]
axR.legend(handles=style + [Line2D([0], [0], color=c, ls='-', label=f'q = {float(q):g}') for q, c in zip(qs, colors)],
           fontsize=8, loc='lower left', ncol=2)
fig.suptitle(f'Edge truncation, Fig-1a/1b style: {CASE}, n=1  ·  100 points, ψ$_{{high}}$ = 0.9 to 0.9999\n'
             'color = surface;  Riccati (STRIDE) solid,  Galerkin (RDCON) dashed', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
stem = 'deltacoil_psihigh_fig1style_' + _tag
out = os.path.abspath(os.path.join(here, '..', 'figures', stem + '.png'))
fig.savefig(out, dpi=150); print('wrote:', out)
