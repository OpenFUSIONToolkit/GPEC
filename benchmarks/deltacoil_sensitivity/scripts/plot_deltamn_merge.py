#!/usr/bin/env python3
# Task 5: plot the merged resonant response vs resistivity from deltamn_resistivity_merge.jl.
#   (a) full |Delta_mn| vs eta, one curve per rational surface (the capstone result)
#   (b) decomposition per surface: the eta-independent coil term vs the eta-dependent plasma term.
# The coil term must be FLAT in eta; its relative spread is printed as an internal consistency check.
# Usage: plot_deltamn_merge.py <deltamn_merge_*.csv> <out.png> [title]
import sys, csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV, OUT = sys.argv[1], sys.argv[2]
TITLE = sys.argv[3] if len(sys.argv) > 3 else ""

rows = list(csv.DictReader(open(CSV)))
if not rows:
    raise SystemExit(f"no rows in {CSV}")
cols = rows[0].keys()
qs = sorted({c.split("_q")[1] for c in cols if c.startswith("dmn_q")}, key=float)

def col(name):
    out = []
    for r in rows:
        v = r.get(name, "")
        try:
            out.append(float(v)) if v not in ("", None) else out.append(np.nan)
        except ValueError:
            out.append(np.nan)
    return np.array(out)

eta = col("eta")
order = np.argsort(-eta)          # high eta (weakly driven) -> low eta (strongly driven)
eta = eta[order]

COLORS = ["#1f5fbf", "#e07b1a", "#2a9d4a", "#b3306b", "#6a3fb5"]
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.4))

print(f"\n=== {CSV} ===")
for k, q in enumerate(qs):
    c = COLORS[k % len(COLORS)]
    tot = col(f"dmn_q{q}")[order]
    coil = col(f"dmn_coil_q{q}")[order]
    plas = col(f"dmn_plasma_q{q}")[order]
    fitr = col(f"fitres_q{q}")[order]

    axA.plot(eta, tot, "o-", color=c, lw=1.8, ms=4, label=f"q = {q}")
    axB.plot(eta, plas, "o-", color=c, lw=1.8, ms=4, label=f"q = {q}  plasma (eta-dep)")
    axB.plot(eta, coil, "s--", color=c, lw=1.4, ms=3.5, alpha=0.75, label=f"q = {q}  coil (eta-indep)")

    good = np.isfinite(coil) & (coil > 0)
    spread = (np.nanmax(coil[good]) / np.nanmin(coil[good]) - 1.0) if good.sum() > 1 else float("nan")
    swing = (np.nanmax(tot[np.isfinite(tot)]) / np.nanmin(tot[np.isfinite(tot)])) if np.isfinite(tot).sum() > 1 else float("nan")
    print(f"  q={q}: |Delta_mn| swings {swing:.3g}x over eta | coil-term relative spread {spread:.2e} "
          f"(should be ~0) | max fit residual {np.nanmax(fitr):.2e}")

for ax in (axA, axB):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.invert_xaxis()                      # left = high eta (weak drive), right = low eta (strong drive)
    ax.set_xlabel("resistivity  eta   (decreasing to the right = more strongly driven layer)")
    ax.grid(True, which="major", alpha=0.25)
axA.set_ylabel("|Delta_mn|  (merged resonant response)")
axA.set_title("(a) Full merged resonant response vs resistivity", fontsize=10.5)
axA.legend(fontsize=8.5)
axB.set_ylabel("contribution to |Delta_mn|")
axB.set_title("(b) Decomposition: eta-independent coil vs eta-dependent plasma", fontsize=10.5)
axB.legend(fontsize=7.4, ncol=1)

if TITLE:
    fig.suptitle(TITLE, fontsize=11.5, y=1.00)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("WROTE:", OUT)
