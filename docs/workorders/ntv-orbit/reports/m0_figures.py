"""M0 current-behavior figure: ZOW torque density on the DIII-D-like example, n = 1 and n = 3.

Reads results/m0_diiid_n{1,3}[_w005]/gpec.h5 (feature/ntv-orbit; KineticForces code = develop 4376e717)
and writes results/m0_report/fig_m0_torque.{svg,png} plus m0_numbers.json with the numbers the report quotes.
"""

import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "results" / "m0_report"
TITLE = {1: "n = 1, C-coil, 1 kA", 3: "n = 3, I-coil, 1 kA, even parity"}
SIGMAS = {"": 0.05, "_w005": 0.005}  # PerturbedEquilibrium regularization_width per run suffix


def load(path):
    with h5py.File(path, "r") as f:
        g = f["KineticForces/fgar"]
        o = np.argsort(g["psi"][()])
        d = dict(psi=g["psi"][()][o], dT=g["dTdpsi"][()][o].real, T=g["T"][()][o].real, total_torque=float(g["total_torque"][()]),
                 total_energy=float(g["total_energy"][()]), res_psi=g["resonance_psi"][()])
        v = f["KineticForces/Validity"]
        d["val"] = {k: v[k][()] for k in v.keys()}
        ss = f["SingularSurfaces"]
        d["rat_psi"], d["rat_q"] = ss["rational_psi"][()], ss["rational_q"][()]
    return d


def banana_psi_width(val, psi0):
    """Thermal-ion banana width at psi0 as Δψ_N, using the local dψ_N/dr."""
    dpsidr = np.gradient(val["psi"], val["r_minor"])
    return float(np.interp(psi0, val["psi"], val["rho_banana"] * dpsidr))


def fwhm(psi, y, k):
    """Full width at half maximum in ψ_N of the peak of |y| at index k (NaN if not bounded)."""
    a, half = np.abs(y), np.abs(y[k]) / 2
    lo, hi = k, k
    while lo > 0 and a[lo] > half:
        lo -= 1
    while hi < len(a) - 1 and a[hi] > half:
        hi += 1
    return float(psi[hi] - psi[lo]) if 0 < lo and hi < len(a) - 1 else float("nan")


def nearest(x, arr):
    arr = np.asarray(arr)
    return (float(arr[np.argmin(np.abs(arr - x))]), float(np.min(np.abs(arr - x)))) if arr.size else (np.nan, np.inf)


def main():
    runs = {}
    for n in (1, 3):
        for suf in SIGMAS:
            p = ROOT / "results" / f"m0_diiid_n{n}{suf}" / "gpec.h5"
            try:
                runs[(n, suf)] = load(p)
            except (OSError, KeyError) as err:
                print(f"skipping {p}: {err}")
    plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "axes.linewidth": 0.6})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.9), sharex=True, gridspec_kw=dict(height_ratios=[1.5, 1]))
    summary = {}
    for j, n in enumerate((1, 3)):
        if (n, "") not in runs:
            continue
        d = runs[(n, "")]
        rats = [(p, q) for p, q in zip(d["rat_psi"], d["rat_q"]) if abs(q * n - round(q * n)) < 1e-6]
        ax, ax2 = axes[0, j], axes[1, j]
        for p, q in rats:
            w = banana_psi_width(d["val"], p)
            ax.axvspan(p - w, p + w, color="0.9", lw=0)
            for a in (ax, ax2):
                a.axvline(p, color="0.55", ls=":", lw=0.7)
            if (n == 1 and q < 5) or (n == 3 and q * n < 10 and int(round(q * n)) % 2 == 0):
                ax.text(p, 1.01, f"{int(round(q * n))}/{n}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5)
        for p in d["res_psi"]:
            ax.plot([p], [1.0], marker="v", ms=3.5, color="k", transform=ax.get_xaxis_transform(), clip_on=False)
        psic = float(np.ravel(d["val"]["psi_c"])[0])
        for a in (ax, ax2):
            a.axvspan(0, psic, color="0.7", lw=0)
        ax.semilogy(d["psi"], np.abs(d["dT"]), "k-", lw=0.9, label=f"σ = {SIGMAS['']}")
        ax2.plot(d["psi"], d["T"], "k-", lw=0.9)
        if (n, "_w005") in runs:
            e = runs[(n, "_w005")]
            ax.semilogy(e["psi"], np.abs(e["dT"]), color="0.45", ls="--", lw=0.8, label=f"σ = {SIGMAS['_w005']}")
            ax2.plot(e["psi"], e["T"], color="0.45", ls="--", lw=0.8)
        ax.set_ylim(1e-3, None)
        ax.set_title(TITLE[n], fontsize=8.5, pad=13)
        ax.legend(loc="upper left", fontsize=7, frameon=False)
        ax2.set_xlabel("ψ$_N$")
        ax2.set_xlim(0, 1)
        if j == 0:
            ax.set_ylabel("|dT$_φ$/dψ$_N$|  (N m)")
            ax2.set_ylabel("T$_φ$(ψ$_N$)  (N m)")
        # numbers quoted in the report
        for suf in SIGMAS:
            if (n, suf) not in runs:
                continue
            r = runs[(n, suf)]
            pk, _ = find_peaks(np.abs(r["dT"]), prominence=0.05 * np.max(np.abs(r["dT"])))
            peaks = []
            for k in pk:
                prat, drat = nearest(r["psi"][k], [p for p, _ in rats])
                pres, dres = nearest(r["psi"][k], r["res_psi"])
                peaks.append(dict(psi=float(r["psi"][k]), dTdpsi=float(r["dT"][k]), fwhm=fwhm(r["psi"], r["dT"], k),
                                  banana_dpsi=banana_psi_width(r["val"], r["psi"][k]), nearest_rational=prat, d_rational=drat,
                                  nearest_resonance=pres, d_resonance=dres))
            T = r["T"]
            summary[f"n{n}{suf}"] = dict(sigma=SIGMAS[suf], total_torque=r["total_torque"], total_energy=r["total_energy"], npts=int(len(r["psi"])),
                                         edge_share_psi_gt_0p9=float((T[-1] - np.interp(0.9, r["psi"], T)) / T[-1]), psi_c=psic,
                                         rationals=[float(p) for p, _ in rats], resonances=[float(p) for p in r["res_psi"]], peaks=peaks)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_m0_torque.svg")
    fig.savefig(OUT / "fig_m0_torque.png", dpi=140)
    (OUT / "m0_numbers.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
