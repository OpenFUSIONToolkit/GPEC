"""
Diagnostic script: compare ξ-space and power-normalized flux eigenvalues across the edge scan.

Usage:
    julia --project=. scripts/compare_eigenvalues_edge_scan.jl [path/to/gpec.h5]

Default: examples/DIIID-like_ideal_example/gpec.h5
"""

using HDF5
using Plots
using Statistics

h5_path = length(ARGS) >= 1 ? ARGS[1] : "examples/DIIID-like_ideal_example/gpec.h5"

if !isfile(h5_path)
    error("File not found: $h5_path\nRun the example first, then point this script at the output.")
end

h5open(h5_path, "r") do f
    if !haskey(f, "edge_scan/psi")
        error("No edge_scan data in $h5_path")
    end

    psi = read(f, "edge_scan/psi")
    q = read(f, "edge_scan/q")

    # ξ-space eigenvalues
    et = real.(read(f, "edge_scan/total_energy"))
    ep = real.(read(f, "edge_scan/plasma_energy"))
    ev = real.(read(f, "edge_scan/vacuum_energy"))

    # Power-normalized flux eigenvalues
    pn_et = real.(read(f, "edge_scan/pn_total_energy"))
    pn_ep = real.(read(f, "edge_scan/pn_plasma_energy"))
    pn_ev = real.(read(f, "edge_scan/pn_vacuum_energy"))

    # Filter NaN
    valid = .!isnan.(et) .& .!isnan.(pn_et)

    # Symmetric ylims keyed on 3× median |value| — a robust trend scale that ignores
    # rational-surface spikes. ylim = (-ymax*1.1, ymax*1.1).
    function symmetric_ylims(datasets)
        all_vals = vcat([filter(!isnan, d) for d in datasets]...)
        isempty(all_vals) && return (-1.0, 1.0)
        ymax = 3.0 * median(abs.(all_vals))
        ymax <= 0 && return (-1.0, 1.0)
        return (-ymax * 1.1, ymax * 1.1)
    end

    # Plot 1: ξ-space eigenvalues
    p1 = plot(psi[valid], et[valid]; label="total (et)", lw=2, xlabel="ψ", ylabel="δW (ξ-space)", title="ξ-space eigenvalues",
        ylims=(-1.0, 1.0))
    plot!(p1, psi[valid], ep[valid]; label="plasma (ep)", lw=1.5, ls=:dash)
    plot!(p1, psi[valid], ev[valid]; label="vacuum (ev)", lw=1.5, ls=:dash)
    hline!(p1, [0.0]; label="", color=:black, ls=:dot, alpha=0.5)

    # Plot 2: Power-normalized flux eigenvalues
    p2 = plot(psi[valid], pn_et[valid]; label="total (pn_et)", lw=2, xlabel="ψ", ylabel="δW (power-norm flux)", title="Power-norm flux eigenvalues",
        ylims=symmetric_ylims([pn_et[valid], pn_ep[valid], pn_ev[valid]]))
    plot!(p2, psi[valid], pn_ep[valid]; label="plasma (pn_ep)", lw=1.5, ls=:dash)
    plot!(p2, psi[valid], pn_ev[valid]; label="vacuum (pn_ev)", lw=1.5, ls=:dash)
    hline!(p2, [0.0]; label="", color=:black, ls=:dot, alpha=0.5)

    # Combined plot
    fig = plot(p1, p2; layout=(2, 1), size=(800, 600), left_margin=12Plots.mm, bottom_margin=4Plots.mm)

    outfile = replace(h5_path, ".h5" => "_edge_eigenvalues.png")
    savefig(fig, outfile)
    println("Saved: $(abspath(outfile))")
end
