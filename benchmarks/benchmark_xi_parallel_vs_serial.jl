#!/usr/bin/env julia
# benchmark_xi_parallel_vs_serial.jl — compare DCON ξ-function storage
# between `use_parallel=false` (EulerLagrange serial path) and
# `use_parallel=true` (Riccati parallel-FM path).
#
# Background: with `use_parallel=true`, the propagator-based FM phase
# stores u_store only at chunk endpoints and leaves ud_store as ZEROS
# for the inter-surface FM chunks (see Riccati.jl:1497 docstring
# caveat).  Only the outer-plasma re-integration (past the last
# rational) populates ud densely.  Since ud_store[:,:,1,:] is the
# perturbed-equilibrium input dξ_ψ/dψ and ud_store[:,:,2,:] is ξ_s,
# this is a real gap.
#
# This benchmark runs the Solovev_ideal_example twice (serial vs
# parallel), reads the saved HDF5 ξ-function arrays, and overlays them
# on one figure for each of:
#     integration/xi_psi   = u_store[:,:,1,:]
#     integration/dxi_psi  = ud_store[:,:,1,:]
#     integration/xi_s     = ud_store[:,:,2,:]
#
# The figure pdfs land in `benchmarks/figures/`.
#
# Usage:
#     julia --project=.. benchmark_xi_parallel_vs_serial.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GeneralizedPerturbedEquilibrium
using HDF5
using Plots
using TOML
using Printf

EXAMPLE_DIR = joinpath(@__DIR__, "..", "examples", "Solovev_ideal_example")
FIG_DIR     = joinpath(@__DIR__, "figures")
mkpath(FIG_DIR)


function run_with_use_parallel(use_parallel::Bool)
    tag = use_parallel ? "parallel" : "serial"
    run_dir = mktempdir(prefix = "gpec_xi_$(tag)_")
    @info "Running Solovev with use_parallel=$use_parallel  → $run_dir"

    # Copy example files into the run dir, then patch gpec.toml.
    for f in readdir(EXAMPLE_DIR)
        src = joinpath(EXAMPLE_DIR, f)
        # Don't copy the example's pre-saved gpec.h5
        if isfile(src) && f != "gpec.h5"
            cp(src, joinpath(run_dir, f); force = true)
        end
    end

    config = TOML.parsefile(joinpath(run_dir, "gpec.toml"))
    config["ForceFreeStates"]["use_parallel"] = use_parallel
    config["ForceFreeStates"]["force_termination"] = true   # skip perturbed-equilibrium phase
    config["ForceFreeStates"]["write_outputs_to_HDF5"] = true
    config["ForceFreeStates"]["HDF5_filename"] = "gpec.h5"
    open(joinpath(run_dir, "gpec.toml"), "w") do io
        TOML.print(io, config)
    end

    GeneralizedPerturbedEquilibrium.main([run_dir])
    return joinpath(run_dir, "gpec.h5")
end


function read_xi(h5_path::AbstractString)
    h5open(h5_path, "r") do f
        return (
            psi     = read(f, "integration/psi"),
            q       = read(f, "integration/q"),
            xi_psi  = read(f, "integration/xi_psi"),
            dxi_psi = read(f, "integration/dxi_psi"),
            xi_s    = read(f, "integration/xi_s"),
            mlow    = read(f, "info/mlow"),
            mpert   = read(f, "info/mpert"),
        )
    end
end


function plot_channel(label::String, data_serial, data_parallel, channel_key::Symbol,
                       fname::String; m_index::Int = 1, sol_index::Int = 1)
    psi_s  = data_serial.psi
    psi_p  = data_parallel.psi
    arr_s  = getproperty(data_serial,   channel_key)
    arr_p  = getproperty(data_parallel, channel_key)

    # arr is (numpert, numpert, 2, nstep) — but data flattened to (numpert, numpert, nstep)
    # because xi_psi etc. were saved as u_store[:,:,1,:] (i.e. one solution component).
    # So arr_s[m_index, sol_index, :] is one m-mode of one ξ basis solution.
    ys = abs.(arr_s[m_index, sol_index, :])
    yp = abs.(arr_p[m_index, sol_index, :])

    plot(psi_s, ys, label = "serial (use_parallel=false)",
         lw = 2, color = :blue, marker = :circle, ms = 2, mz = nothing,
         xlabel = "ψ_N", ylabel = "|$label|",
         title = "$label  (m_index=$m_index, sol_index=$sol_index)",
         legend = :topleft, size = (900, 400))
    plot!(psi_p, yp, label = "parallel (use_parallel=true)",
          lw = 2, color = :red, ls = :dash, marker = :diamond, ms = 2)

    out_png = joinpath(FIG_DIR, fname * ".png")
    out_pdf = joinpath(FIG_DIR, fname * ".pdf")
    savefig(out_png)
    savefig(out_pdf)
    @info "  → $out_png"
end


function plot_overlay(data_serial, data_parallel)
    # Sum |·|² across the IC (sol_index) dimension to get a basis-
    # invariant magnitude per (mode, ψ) — this avoids picking arbitrary
    # IC columns and gives a cleaner physical comparison.  Then take
    # the first m-mode in the band for a representative trace.
    m_idx = 1
    norm_s_xi   = vec(sqrt.(sum(abs2.(view(data_serial.xi_psi,   m_idx, :, :)), dims = 1)))
    norm_p_xi   = vec(sqrt.(sum(abs2.(view(data_parallel.xi_psi, m_idx, :, :)), dims = 1)))
    norm_s_dxi  = vec(sqrt.(sum(abs2.(view(data_serial.dxi_psi,  m_idx, :, :)), dims = 1)))
    norm_p_dxi  = vec(sqrt.(sum(abs2.(view(data_parallel.dxi_psi, m_idx, :, :)), dims = 1)))
    norm_s_xis  = vec(sqrt.(sum(abs2.(view(data_serial.xi_s,     m_idx, :, :)), dims = 1)))
    norm_p_xis  = vec(sqrt.(sum(abs2.(view(data_parallel.xi_s,   m_idx, :, :)), dims = 1)))

    psi_s = data_serial.psi
    psi_p = data_parallel.psi

    title_suffix = @sprintf("\n(serial: %d saved ψ; parallel: %d saved ψ)",
                            length(psi_s), length(psi_p))

    common_kw = (legend = :topright,
                 left_margin = 12Plots.mm, bottom_margin = 4Plots.mm)

    p1 = plot(psi_s, norm_s_xi, label = "serial", lw = 2, color = :blue,
              marker = :circle, ms = 2.5,
              xlabel = "ψ_N", ylabel = "‖ξ_ψ(m=$m_idx, ·)‖₂",
              title = "ξ_ψ   u_store[m,:,1,:]" * title_suffix; common_kw...)
    plot!(p1, psi_p, norm_p_xi, label = "parallel", lw = 2, color = :red,
          ls = :dash, marker = :diamond, ms = 3.5)

    p2 = plot(psi_s, norm_s_dxi, label = "serial", lw = 2, color = :blue,
              marker = :circle, ms = 2.5,
              xlabel = "ψ_N", ylabel = "‖dξ_ψ/dψ(m=$m_idx, ·)‖₂",
              title = "dξ_ψ/dψ   ud_store[m,:,1,:]"; common_kw...)
    plot!(p2, psi_p, norm_p_dxi, label = "parallel", lw = 2, color = :red,
          ls = :dash, marker = :diamond, ms = 3.5)

    p3 = plot(psi_s, norm_s_xis, label = "serial", lw = 2, color = :blue,
              marker = :circle, ms = 2.5,
              xlabel = "ψ_N", ylabel = "‖ξ_s(m=$m_idx, ·)‖₂",
              title = "ξ_s   ud_store[m,:,2,:]"; common_kw...)
    plot!(p3, psi_p, norm_p_xis, label = "parallel", lw = 2, color = :red,
          ls = :dash, marker = :diamond, ms = 3.5)

    fig = plot(p1, p2, p3; layout = (3, 1), size = (1000, 1300),
               left_margin = 14Plots.mm, bottom_margin = 4Plots.mm,
               plot_title = "Solovev_ideal_example: DCON ξ-function storage (parallel vs serial)")
    out_png = joinpath(FIG_DIR, "xi_benchmark_solovev.png")
    out_pdf = joinpath(FIG_DIR, "xi_benchmark_solovev.pdf")
    savefig(fig, out_png)
    savefig(fig, out_pdf)
    @info "  → $out_png"
    @info "  → $out_pdf"
    return fig
end


function summarize(data_serial, data_parallel)
    println("=" ^ 72)
    println("ξ-function array shapes:")
    println("=" ^ 72)
    for (lab, d) in (("serial", data_serial), ("parallel", data_parallel))
        @printf("  %s:\n", lab)
        @printf("    psi:     %s\n", size(d.psi))
        @printf("    xi_psi:  %s\n", size(d.xi_psi))
        @printf("    dxi_psi: %s\n", size(d.dxi_psi))
        @printf("    xi_s:    %s\n", size(d.xi_s))
    end
    println()
    println("=" ^ 72)
    println("Zero-fraction in ud_store channels  (ud=zeros for FM chunks in parallel):")
    println("=" ^ 72)
    for (lab, d) in (("serial", data_serial), ("parallel", data_parallel))
        n_total_dx = length(d.dxi_psi)
        n_total_xs = length(d.xi_s)
        n_zero_dx = count(==(0), d.dxi_psi)
        n_zero_xs = count(==(0), d.xi_s)
        @printf("  %-9s dxi_psi zeros: %6d / %d  (%.1f%%)\n",
                lab, n_zero_dx, n_total_dx, 100.0 * n_zero_dx / n_total_dx)
        @printf("  %-9s xi_s    zeros: %6d / %d  (%.1f%%)\n",
                lab, n_zero_xs, n_total_xs, 100.0 * n_zero_xs / n_total_xs)
    end
    println()
end


function main()
    h5_serial   = run_with_use_parallel(false)
    h5_parallel = run_with_use_parallel(true)

    @info "Reading ξ functions from both HDF5 outputs"
    data_serial   = read_xi(h5_serial)
    data_parallel = read_xi(h5_parallel)

    summarize(data_serial, data_parallel)
    plot_overlay(data_serial, data_parallel)
    @info "Done."
end


main()
