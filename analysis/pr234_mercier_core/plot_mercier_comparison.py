import csv
from pathlib import Path

import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "mercier_comparison.csv"


def load_rows():
    rows = []
    with CSV_PATH.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in row.items()})
    return rows


def subset(rows, lo, hi):
    return [row for row in rows if lo <= row["psi"] <= hi]


def plot_core(rows):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    psi = [row["psi"] for row in rows]

    ax = axes[0, 0]
    ax.semilogx(psi, [row["old_mercier_di"] for row in rows], marker="o", label="restored Mercier.jl D_I")
    ax.semilogx(psi, [row["current_resistive_di"] for row in rows], marker="x", linestyle="--", label="Bal.jl resistive_interchange D_I")
    ax.semilogx(psi, [row["det_d0bar"] for row in rows], marker="s", markersize=4, label="Bal.jl det(d0bar) D_I")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("psi_N")
    ax.set_ylabel("D_I")
    ax.set_title("Core D_I behavior, psi_N = 1e-4 ... 1e-1")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.semilogx(psi, [row["qprime"] for row in rows], marker="o")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("psi_N")
    ax.set_ylabel("dq/dpsi_N")
    ax.set_title("q' is small/sign-changing in the core")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1, 0]
    ax.semilogx(psi, [row["old_current_di_rel_diff"] for row in rows], marker="o", label="restored vs Bal.jl resistive")
    ax.semilogx(psi, [row["det_old_di_rel_diff"] for row in rows], marker="s", label="det(d0bar) vs restored")
    ax.set_xlabel("psi_N")
    ax.set_ylabel("relative difference")
    ax.set_title("Agreement and det(d0bar) mismatch")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.semilogx(psi, [row["ballooning_delta_prime"] for row in rows], marker="o", color="tab:green")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("psi_N")
    ax.set_ylabel("ballooning Delta'")
    ax.set_title("Ballooning Delta' in the same core region")
    ax.grid(True, which="both", alpha=0.3)

    out = BASE / "mercier_core_1e-4_to_1e-1.png"
    fig.savefig(out, dpi=180)
    return out


def plot_outer(rows):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    psi = [row["psi"] for row in rows]

    ax = axes[0]
    ax.plot(psi, [row["old_mercier_di"] for row in rows], label="restored Mercier.jl D_I", linewidth=1.8)
    ax.plot(psi, [row["current_resistive_di"] for row in rows], linestyle="--", label="Bal.jl resistive_interchange D_I", linewidth=1.4)
    ax.plot(psi, [row["det_d0bar"] for row in rows], linestyle=":", label="Bal.jl det(d0bar) D_I", linewidth=1.4)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("psi_N")
    ax.set_ylabel("D_I")
    ax.set_title("Outer-region D_I behavior, psi_N = 1e-1 ... 0.99")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.semilogy(psi, [row["old_current_di_rel_diff"] for row in rows], label="restored vs Bal.jl resistive")
    ax.semilogy(psi, [row["det_old_di_rel_diff"] for row in rows], label="det(d0bar) vs restored")
    ax.set_xlabel("psi_N")
    ax.set_ylabel("relative difference")
    ax.set_title("Relative differences outside the ill-conditioned core")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    out = BASE / "mercier_outer_1e-1_to_0p99.png"
    fig.savefig(out, dpi=180)
    return out


def main():
    rows = load_rows()
    core = subset(rows, 1e-4, 1e-1)
    outer = subset(rows, 1e-1, 0.99)
    print(plot_core(core))
    print(plot_outer(outer))


if __name__ == "__main__":
    main()
