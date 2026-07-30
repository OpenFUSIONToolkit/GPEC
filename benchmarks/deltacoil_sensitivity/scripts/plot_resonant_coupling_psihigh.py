#!/usr/bin/env python3
# plot_resonant_coupling_psihigh.py -- matched resonant coupling ||cout|| per rational surface vs the
# outer truncation psihigh, comparing the two match paths: STRIDE (singular/resonant_match/cout, solid)
# and galerkin (galerkin/match/cout, dashed). Reads resonant_coupling_psihigh_<case>.csv.
# x-axis = log distance from edge (1 - psihigh), edge at right, matching the other psihigh figures.
#
# Usage: python3 plot_resonant_coupling_psihigh.py results/resonant_coupling_psihigh_<case>.csv

import sys, os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
    os.path.join(here, '..', 'results', 'resonant_coupling_psihigh.csv')

with open(csv_path) as f:
    rows = list(csv.reader(f))
header = rows[0]
scols = [c for c in header[1:] if c.startswith('stride_q')]
gcols = [c for c in header[1:] if c.startswith('gal_q')]
qvals = [float(c.split('q')[1]) for c in scols]
idx = {c: i + 1 for i, c in enumerate(header[1:])}
psihigh, S, G = [], {q: [] for q in qvals}, {q: [] for q in qvals}
for r in rows[1:]:
    if not r or r[0] == '':
        continue
    psihigh.append(float(r[0]))
    for q in qvals:
        sv = r[idx[f'stride_q{q}']]; gv = r[idx[f'gal_q{q}']]
        S[q].append(float(sv) if sv not in ('', 'FAIL') else np.nan)
        G[q].append(float(gv) if gv not in ('', 'FAIL') else np.nan)
psihigh = np.array(psihigh)

_case = os.path.splitext(os.path.basename(csv_path))[0].replace('resonant_coupling_psihigh', '').strip('_').upper()
CASE = 'LAR (large-aspect-ratio)' if 'LAR' in _case else ('DIII-D-like' if ('DIIID' in _case or _case == '') else _case)

_hasany = np.array([any(not np.isnan(S[q][i]) for q in qvals) for i in range(len(psihigh))])
x0 = psihigh[_hasany].min() if _hasany.any() else psihigh.min()
def edge_log_x(ax):
    fwd = lambda p: -np.log10(np.clip(1.0 - np.asarray(p, float), 1e-9, None))
    inv = lambda t: 1.0 - 10.0 ** (-np.asarray(t, float))
    ax.set_xscale('function', functions=(fwd, inv))
    cand = np.array([0.5, 0.6, 0.75, 0.85, 0.9, 0.95, 0.98, 0.99])
    tk = cand[(cand >= x0 - 1e-9) & (cand <= psihigh.max() + 1e-9)]
    ax.set_xticks(tk); ax.set_xticklabels([f'{t:g}' for t in tk]); ax.minorticks_off()
    ax.set_xlim(x0 - (1 - x0) * 0.06, psihigh.max() + (1 - psihigh.max()) * 0.4)

colors = plt.cm.viridis(np.linspace(0.08, 0.82, len(qvals)))
fig, ax = plt.subplots(figsize=(8.6, 5.2))
ax.set_yscale('log')
for q, c in zip(qvals, colors):
    ys = np.array(S[q], float); m = ~np.isnan(ys)
    ax.plot(psihigh[m], ys[m], 'o-', color=c, lw=2.0, ms=6, label=f'q = {q:g}, STRIDE')
    yg = np.array(G[q], float); mg = ~np.isnan(yg)
    ax.plot(psihigh[mg], yg[mg], 's--', color=c, lw=1.5, ms=5, mfc='white', label=f'q = {q:g}, galerkin')
edge_log_x(ax)
ax.set_xlabel('outer truncation  ψ$_{high}$   (log distance from edge, 1 - ψ$_{high}$; edge at right)', fontsize=11)
ax.set_ylabel('matched resonant coupling\n||cout|| per surface  (log)', fontsize=11)
ax.set_title('Matched resonant coupling vs psihigh truncation: STRIDE (solid) vs galerkin (dashed)\n'
             f'{CASE}, n=1  ·  cout from Wang 2020 Eq. 11 match', fontsize=11)
ax.grid(alpha=0.25, which='both')
ax.legend(fontsize=7.6, ncol=2, loc='best')
fig.subplots_adjust(left=0.13, bottom=0.13, right=0.97, top=0.88)

stem = os.path.splitext(os.path.basename(csv_path))[0].replace('resonant_coupling', 'rescoup')
out = os.path.abspath(os.path.join(here, '..', 'figures', stem + '.png'))
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print('wrote:', out)
