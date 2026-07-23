#!/usr/bin/env python3
# plot_deltacoil_psihigh_metric.py - plot the plot-2 dot-product (cosine) shape metric vs the psihigh
# truncation. Reads results/deltacoil_psihigh_metric.csv from deltacoil_psihigh_metric.jl.
# Each curve is one rational surface: cosine similarity of its delta_coil with the full-domain nominal.
# cos = 1 means shape unchanged (only rescaled); cos < 1 means the coil-response pattern rotated.
#
# Usage: python3 plot_deltacoil_psihigh_metric.py [results/deltacoil_psihigh_metric.csv]

import sys, os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, '..', 'results', 'deltacoil_psihigh_metric.csv')
csv_path = os.path.abspath(csv_path)

with open(csv_path) as f:
    rows = list(csv.reader(f))
header = rows[0]
qvals = [float(c[1:]) for c in header[1:]]
psihigh, data = [], {q: [] for q in qvals}
for r in rows[1:]:
    if not r or r[0] == '':
        continue
    psihigh.append(float(r[0]))
    for q, cell in zip(qvals, r[1:]):
        data[q].append(float(cell) if cell not in ('', 'FAIL') else np.nan)
psihigh = np.array(psihigh)

_case = os.path.splitext(os.path.basename(csv_path))[0].replace('_metric', '').replace('deltacoil_psihigh', '').strip('_').upper()
CASE = {'': 'DIII-D-like', 'DIIID': 'DIII-D-like', 'LAR': 'LAR (large-aspect-ratio)'}.get(_case, _case)

colors = plt.cm.viridis(np.linspace(0.08, 0.82, len(qvals)))

# leftmost psihigh that actually has data (first surface entry); ticks/annotations anchor here so the
# empty low-psihigh points do not push labels off the plotted range.
_has = np.array([any(not np.isnan(data[q][i]) for q in qvals) for i in range(len(psihigh))])
x0 = psihigh[_has].min() if _has.any() else psihigh.min()

fig, ax = plt.subplots(figsize=(8.4, 5.1))
for q, c in zip(qvals, colors):
    y = np.array(data[q], float)
    m = ~np.isnan(y)
    ax.plot(psihigh[m], y[m], 'o-', color=c, lw=1.9, ms=6, label=f'q = {q:g}')

ax.axhline(1.0, ls=':', color='#888', lw=1.0)
ax.text(x0, 1.03, '1.0 = shape unchanged (only rescaled)', fontsize=8, color='#555', va='bottom', ha='left')
ax.text(x0, 0.06, 'low = pattern rotated\n(shape change the norm hides)',
        fontsize=8, color='#a33', va='bottom', ha='left')
# x-axis = distance from the edge (1 - psihigh) on a log scale (edge at right), so the log-packed
# near-edge points spread out and the curves approach convergence horizontally, not vertically.
_fwd = lambda p: -np.log10(np.clip(1.0 - np.asarray(p, float), 1e-9, None))
_inv = lambda t: 1.0 - 10.0 ** (-np.asarray(t, float))
ax.set_xscale('function', functions=(_fwd, _inv))
_cand = np.array([0.6, 0.75, 0.85, 0.9, 0.95, 0.98, 0.99])
_ticks = _cand[(_cand >= x0 - 1e-9) & (_cand <= psihigh.max() + 1e-9)]
ax.set_xticks(_ticks); ax.set_xticklabels([f'{t:g}' for t in _ticks])
ax.minorticks_off()
ax.set_xlim(x0 - (1 - x0) * 0.06, psihigh.max() + (1 - psihigh.max()) * 0.4)
ax.set_xlabel('outer truncation  ψ$_{high}$   (log distance from edge, 1 - ψ$_{high}$; edge at right)', fontsize=11)
ax.set_ylabel('cosine similarity of delta_coil with full-domain nominal', fontsize=11)
ax.set_title('Dot-product (shape) metric vs psihigh truncation\n'
             f'{CASE}, n=1  ·  per surface, aligned by poloidal mode m  ·  nominal = largest ψ$_{{high}}$',
             fontsize=11)
ax.set_ylim(-0.05, 1.06)
ax.grid(alpha=0.25)
ax.legend(title='rational surface', fontsize=9, title_fontsize=9, loc='lower right')
fig.subplots_adjust(left=0.12, bottom=0.13, right=0.97, top=0.86)

stem = os.path.splitext(os.path.basename(csv_path))[0]
out = os.path.abspath(os.path.join(here, '..', 'figures', stem + '.png'))
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print('wrote:', out)
