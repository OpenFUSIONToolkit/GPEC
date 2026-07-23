#!/usr/bin/env python3
# plot_resistivity_scan.py -- resistivity (eta) scan comparing the two inner-layer backends (ray vs
# galerkin) in the galerkin match. Two panels: left = inner-layer |deltar| (Delta(Q), where the backends
# differ), right = matched penetrated field ||bpen|| (the shielding family). ray solid, galerkin dashed,
# one color per rational surface. |Q| grows as eta falls (roughly |Q| ~ eta^-1/3), toward the left.
# Reads resistivity_scan_<case>.csv.
#
# Usage: python3 plot_resistivity_scan.py results/resistivity_scan_<case>.csv

import sys, os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

csv_path = os.path.abspath(sys.argv[1])
here = os.path.dirname(os.path.abspath(__file__))
with open(csv_path) as f:
    rows = list(csv.reader(f))
header = rows[0]
qvals = sorted({float(c.split('_q')[1]) for c in header[1:]})
idx = {c: i for i, c in enumerate(header)}
eta = []
data = {(m, q): [] for m in ('ray_deltar', 'gal_deltar', 'ray_bpen', 'gal_bpen') for q in qvals}
for r in rows[1:]:
    if not r or r[0] == '':
        continue
    eta.append(float(r[0]))
    for (m, q) in data:
        col = f'{m}_q{q}'
        v = r[idx[col]] if col in idx else ''
        data[(m, q)].append(float(v) if v not in ('', 'FAIL') else np.nan)
eta = np.array(eta)

_case = os.path.splitext(os.path.basename(csv_path))[0].replace('resistivity_scan', '').strip('_').upper()
CASE = 'LAR (large-aspect-ratio)' if 'LAR' in _case else 'DIII-D-like'
colors = plt.cm.viridis(np.linspace(0.08, 0.82, len(qvals)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
def panel(ax, ray_key, gal_key, ylab, ttl):
    for q, c in zip(qvals, colors):
        yr = np.array(data[(ray_key, q)], float); mr = ~np.isnan(yr)
        ax.plot(eta[mr], yr[mr], 'o-', color=c, lw=2.0, ms=6, label=f'q = {q:g}, ray')
        yg = np.array(data[(gal_key, q)], float); mg = ~np.isnan(yg)
        ax.plot(eta[mg], yg[mg], 's--', color=c, lw=1.5, ms=5, mfc='white', label=f'q = {q:g}, galerkin')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.invert_xaxis()   # low eta (high |Q|, where galerkin drifts) on the RIGHT
    ax.set_xlabel('resistivity  eta   (low eta = high |Q|, to the right)', fontsize=10.5)
    ax.set_ylabel(ylab, fontsize=10.5)
    ax.set_title(ttl, fontsize=10.5, fontweight='bold')
    ax.grid(alpha=0.25, which='both')

panel(ax1, 'ray_deltar', 'gal_deltar', 'inner-layer |deltar|  (log)', 'Inner-layer Delta(Q): ray vs galerkin backend')
panel(ax2, 'ray_bpen', 'gal_bpen', 'matched penetrated field  ||bpen||  (log)', 'Penetrated (shielding) field: ray vs galerkin backend')
ax1.legend(fontsize=7.4, ncol=2, loc='best')
fig.suptitle(f'Resistivity scan: ray (solid) vs galerkin (dashed) inner-layer backend  ·  {CASE}, n=1',
             fontsize=12, fontweight='bold')
fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.88, wspace=0.22)

stem = os.path.splitext(os.path.basename(csv_path))[0].replace('resistivity_scan', 'resist')
out = os.path.abspath(os.path.join(here, '..', 'figures', stem + '.png'))
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print('wrote:', out)
