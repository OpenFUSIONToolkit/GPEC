"""
    _plot_cond_fbar.jl

Shared helper used by the kinetic-DCON benchmark scripts to visualise the
cond(F̄) scan that the Julia side performs when searching for kinetic
singular surfaces (zeros of det(F̄)).

The benchmarks `include(...)` this file and then call
`plot_cond_fbar_scan(h5path, png_path; title)` after `GPE.main()` finishes.
The HDF5 data is populated by `ForceFreeStates.find_kinetic_singular_surfaces!`
and written by `GeneralizedPerturbedEquilibrium.write_outputs_to_HDF5` under
the `singular/kinetic/` group.
"""

using HDF5
using Plots
using Printf

"""
    plot_cond_fbar_scan(h5path, png_path; title="") → png_path or nothing

Load `singular/kinetic/{scan_psi, scan_cond, scan_threshold, psi}` from the
GPEC HDF5 output and save a PNG plot of cond(F̄) vs ψ. The accepted
kinetic singular surfaces and the ideal rational surfaces are overlaid as
vertical markers so it is obvious whether any cond(F̄) peak crossed the
threshold. Returns `nothing` if no scan data is present (kinetic_factor==0).
"""
function plot_cond_fbar_scan(h5path::String, png_path::String; title::String="")
    scan_psi, scan_cond, thr, k_psi, i_psi, i_q, kmsing = h5open(h5path, "r") do h5
        haskey(h5, "singular") && haskey(h5["singular"], "kinetic") || return (Float64[], Float64[], 0.0, Float64[], Float64[], Float64[], 0)
        kg = h5["singular/kinetic"]
        (read(kg["scan_psi"]),
            read(kg["scan_cond"]),
            read(kg["scan_threshold"]),
            read(kg["psi"]),
            read(h5["singular/psi"]),
            read(h5["singular/q"]),
            read(kg["kmsing"]))
    end

    isempty(scan_psi) && return nothing

    p = plot(scan_psi, scan_cond;
        yscale=:log10,
        lw=2,
        color=:steelblue,
        label="cond(F̄)",
        xlabel="ψ",
        ylabel="cond(F̄)",
        title=title,
        left_margin=12Plots.mm,
        bottom_margin=4Plots.mm,
        size=(900, 500))

    if thr > 0
        hline!(p, [thr]; color=:red, linestyle=:dash, lw=1.5,
            label=@sprintf("threshold = %.0e", thr))
    end

    for (idx, ps) in enumerate(i_psi)
        vline!(p, [ps]; color=:gray, linestyle=:dot, lw=1,
            label=idx == 1 ? "ideal rational (q=$(round(i_q[idx]; digits=2)))" : nothing)
    end

    for (idx, ps) in enumerate(k_psi)
        vline!(p, [ps]; color=:crimson, lw=1.5,
            label=idx == 1 ? "accepted kinsing" : nothing)
    end

    annotate!(p, [(scan_psi[end],
        maximum(filter(isfinite, scan_cond)),
        text(@sprintf("kmsing = %d", kmsing), :right, 9, :black))])

    savefig(p, png_path)
    println(stderr, "  cond(F̄) plot: ", abspath(png_path))
    return abspath(png_path)
end
