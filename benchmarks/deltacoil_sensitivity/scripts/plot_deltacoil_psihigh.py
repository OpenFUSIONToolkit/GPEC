#!/usr/bin/env python3
# plot_deltacoil_psihigh.py - plot |delta_coil| per rational surface vs the outer truncation psihigh
# (log-packed toward the edge). Reads results/deltacoil_psihigh_result.csv written by
# deltacoil_psihigh_scan.jl. Each surface's curve begins where psihigh first exceeds its psi_s.
#
# Usage: python3 plot_deltacoil_psihigh.py [results/deltacoil_psihigh_result.csv]

import sys, os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
# optional --psi_s="2:0.518,3:0.770,..." gives the TRUE rational-surface psi locations (from the
# equilibrium's singular/psi), annotated alongside the grid-entry proxy. Non-flag arg = csv path.
TRUE_PSIS = {}
_pos = []
for a in sys.argv[1:]:
    if a.startswith('--psi_s='):
        for kv in a.split('=', 1)[1].split(','):
            if ':' in kv:
                k, v = kv.split(':'); TRUE_PSIS[float(k)] = float(v)
    else:
        _pos.append(a)
csv_path = _pos[0] if _pos else os.path.join(here, '..', 'results', 'deltacoil_psihigh_result.csv')
csv_path = os.path.abspath(csv_path)

# --- read CSV ---
with open(csv_path) as f:
    rows = list(csv.reader(f))
header = rows[0]
qcols = header[1:]                                   # e.g. ['q2.0','q3.0','q4.0','q5.0']
qvals = [float(c[1:]) for c in qcols]
psihigh, data = [], {q: [] for q in qvals}
for r in rows[1:]:
    if not r or r[0] == '':
        continue
    psihigh.append(float(r[0]))
    for q, cell in zip(qvals, r[1:]):
        data[q].append(float(cell) if cell not in ('', 'FAIL') else np.nan)
psihigh = np.array(psihigh)

# case label derived from the CSV name, so titles are correct for DIII-D / LAR / etc.
_case = os.path.splitext(os.path.basename(csv_path))[0].replace('_result', '').replace('deltacoil_psihigh', '').strip('_').upper()
CASE = {'': 'DIII-D-like', 'DIIID': 'DIII-D-like', 'LAR': 'LAR (large-aspect-ratio)'}.get(_case, _case)

# entry psihigh per surface (first psihigh at which it appears) - a case-agnostic proxy for psi_s,
# so the same plotter works for DIII-D, LAR, or any case without hardcoded surface locations.
def entry_psihigh(q):
    y = np.array(data[q], float)
    idx = np.where(~np.isnan(y))[0]
    return psihigh[idx[0]] if len(idx) else np.nan
PSI_S = {q: entry_psihigh(q) for q in qvals}
colors = plt.cm.viridis(np.linspace(0.08, 0.82, len(qvals)))

# x-axis = distance from the edge (1 - psihigh) on a log scale (edge at right), so the log-packed
# near-edge points spread out and curves approach convergence horizontally, not vertically. Anchor to
# the leftmost psihigh that has data so empty low-psihigh points do not push labels off the range.
_has = np.array([any(not np.isnan(data[q][i]) for q in qvals) for i in range(len(psihigh))])
x0 = psihigh[_has].min() if _has.any() else psihigh.min()
def edge_log_x(ax):
    fwd = lambda p: -np.log10(np.clip(1.0 - np.asarray(p, float), 1e-9, None))
    inv = lambda t: 1.0 - 10.0 ** (-np.asarray(t, float))
    ax.set_xscale('function', functions=(fwd, inv))
    cand = np.array([0.6, 0.75, 0.85, 0.9, 0.95, 0.98, 0.99])
    tk = cand[(cand >= x0 - 1e-9) & (cand <= psihigh.max() + 1e-9)]
    ax.set_xticks(tk); ax.set_xticklabels([f'{t:g}' for t in tk]); ax.minorticks_off()
    ax.set_xlim(x0 - (1 - x0) * 0.06, psihigh.max() + (1 - psihigh.max()) * 0.4)

fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax.set_yscale('log')                                  # log-y so the edge surface (q5) does not flatten q2/q3/q4
for q, c in zip(qvals, colors):
    y = np.array(data[q], float)
    m = ~np.isnan(y)
    ax.plot(psihigh[m], y[m], 'o-', color=c, lw=1.8, ms=6, label=f'q = {q:g}')
for q, c in zip(qvals, colors):                        # vlines after data so ylim is settled on the log scale
    if q in PSI_S and np.isfinite(PSI_S[q]):
        ax.axvline(PSI_S[q], ls=':', color=c, lw=1.0, alpha=0.7)          # grid-entry (first psihigh with data)
        lbl = f' q={q:g} enters ≈{PSI_S[q]:.3f}'
        if q in TRUE_PSIS:                                                # true rational-surface location
            ax.axvline(TRUE_PSIS[q], ls='--', color=c, lw=1.0, alpha=0.55)
            lbl += f'  (ψ$_s$≈{TRUE_PSIS[q]:.3f})'
        ax.text(PSI_S[q], ax.get_ylim()[1], lbl, color=c,
                fontsize=7.5, rotation=90, va='top', ha='left')

edge_log_x(ax)
ax.set_xlabel('outer truncation  ψ$_{high}$   (log distance from edge, 1 - ψ$_{high}$; edge at right)', fontsize=11)
ax.set_ylabel('|delta_coil|  per surface  (log)', fontsize=11)
_sub = 'each curve begins where ψ$_{high}$ passes that surface'
if TRUE_PSIS:
    _sub += '  ·  dotted = grid entry, dashed = true rational surface ψ$_s$'
ax.set_title('delta_coil vs outer truncation ψ$_{high}$ (log-packed toward edge)\n'
             f'{CASE}, n=1  ·  {_sub}', fontsize=10.5)
ax.grid(alpha=0.25)
ax.legend(title='rational surface', fontsize=9, title_fontsize=9)
fig.subplots_adjust(left=0.12, bottom=0.13, right=0.97, top=0.88)

# derive output name from the CSV stem so different cases (diiid/lar) get distinct figures
stem = os.path.splitext(os.path.basename(csv_path))[0].replace('_result', '')
out = os.path.join(here, '..', 'figures', stem + '.png')
out = os.path.abspath(out); os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print('wrote:', out)

# second panel: relative change vs the edge-most (last) psihigh - how much truncation still moves each surface
fig2, ax2 = plt.subplots(figsize=(8.2, 5.0))
for q, c in zip(qvals, colors):
    y = np.array(data[q], float)
    m = ~np.isnan(y)
    if m.sum() < 2:
        continue
    ref = y[m][-1]                                    # reference = value at largest psihigh where present
    ax2.plot(psihigh[m], np.abs(y[m] - ref) / max(abs(ref), 1e-300), 'o-', color=c, lw=1.8, ms=6, label=f'q = {q:g}')
ax2.set_yscale('log')
edge_log_x(ax2)
ax2.set_xlabel('outer truncation  ψ$_{high}$   (log distance from edge, 1 - ψ$_{high}$; edge at right)', fontsize=11)
ax2.set_ylabel('|Δ delta_coil| / |delta_coil(ψ$_{high}$=max)|', fontsize=11)
ax2.set_title(f'Relative sensitivity of delta_coil to the truncation boundary, {CASE}\n'
              '(vs the outermost ψ$_{high}$ reference; lower = more converged)', fontsize=11)
ax2.grid(alpha=0.25, which='both')
ax2.legend(title='rational surface', fontsize=9, title_fontsize=9)
fig2.subplots_adjust(left=0.13, bottom=0.13, right=0.97, top=0.88)
out2 = os.path.join(here, '..', 'figures', stem + '_relchange.png')
out2 = os.path.abspath(out2)
fig2.savefig(out2, dpi=150)
print('wrote:', out2)
