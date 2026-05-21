#!/usr/bin/env julia
# benchmark_xi_parallel_vs_serial.jl — compare DCON ξ-function storage
# between `use_parallel=false` (serial EulerLagrange path) and
# `use_parallel=true` (parallel propagator BVP with the appended serial-EL
# dense pass that populates HDF5 integration/xi_* in axis basis).
#
# Background: with `use_parallel=true`, the propagator-based FM phase
# stores u_store only at chunk endpoints in Riccati S form, and leaves
# ud_store as ZEROS for the inter-surface FM chunks.  Since u_store[:,:,1,:]
# is ξ_ψ, ud_store[:,:,1,:] is dξ_ψ/dψ, and ud_store[:,:,2,:] is ξ_s,
# downstream PerturbedEquilibrium reconstruction cannot read this sparse
# storage.  The `populate_dense_xi = true` (default) flag appends a serial
# EulerLagrange pass that replaces odet so the HDF5 outputs match what the
# pure serial path produces — same dense ψ grid, same axis basis.
#
# Runs the same gpec.toml twice (serial vs parallel) on each requested
# example, reads the saved HDF5 ξ-function arrays, and overlays them for
# every RESONANT mode (m such that q = m/n falls inside the integration
# range).  Per-example figure pdfs/pngs land in `benchmarks/figures/`.
#
# Usage:
#     julia --project=.. benchmark_xi_parallel_vs_serial.jl
#     julia --project=.. benchmark_xi_parallel_vs_serial.jl Solovev_ideal_example DIIID-like_ideal_example

using GeneralizedPerturbedEquilibrium
using HDF5
using Plots
using TOML
using Printf

const EXAMPLES_ROOT = joinpath(@__DIR__, "..", "examples")
const FIG_DIR       = joinpath(@__DIR__, "figures")
mkpath(FIG_DIR)


function run_with_use_parallel(example_dir::AbstractString, use_parallel::Bool)
    tag = use_parallel ? "parallel" : "serial"
    ex_tag = basename(rstrip(example_dir, '/'))
    run_dir = mktempdir(prefix = "gpec_xi_$(ex_tag)_$(tag)_")
    @info "Running $ex_tag with use_parallel=$use_parallel  → $run_dir"

    # Copy example files into the run dir, then patch gpec.toml.
    for f in readdir(example_dir)
        src = joinpath(example_dir, f)
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
        # singular/m is shape (msing, max_modes); take the first column
        # (dominant resonant m per surface)
        m_matrix = read(f, "singular/m")
        msing    = read(f, "singular/msing")
        resonant_m = msing > 0 ?
            Int[m_matrix[s, 1] for s in 1:msing] :
            Int[]
        return (
            psi      = read(f, "integration/psi"),
            q        = read(f, "integration/q"),
            xi_psi   = read(f, "integration/xi_psi"),
            dxi_psi  = read(f, "integration/dxi_psi"),
            xi_s     = read(f, "integration/xi_s"),
            sing_psi = read(f, "singular/psi"),
            sing_q   = read(f, "singular/q"),
            mlow     = read(f, "info/mlow"),
            mpert    = read(f, "info/mpert"),
            msing    = msing,
            resonant_m = resonant_m,
        )
    end
end


"""
    mode_norm_over_ICs(arr, m_idx) -> Vector{Float64}

For arr of shape (mpert, numpert_total, nstep), pick the m-row `m_idx` and
return the per-ψ L2 norm over the IC index (numpert_total dimension).  This
gives a basis-invariant magnitude per (m, ψ).
"""
mode_norm_over_ICs(arr::AbstractArray, m_idx::Int) =
    vec(sqrt.(sum(abs2.(view(arr, m_idx, :, :)), dims = 1)))


function plot_overlay(example_name::AbstractString, data_serial, data_parallel)
    @assert data_serial.mlow == data_parallel.mlow
    @assert data_serial.resonant_m == data_parallel.resonant_m
    mlow       = data_serial.mlow
    resonant_m = data_serial.resonant_m
    @assert !isempty(resonant_m) "No resonant surfaces found in $example_name"

    psi_s   = data_serial.psi
    psi_p   = data_parallel.psi
    sing_ψ  = data_serial.sing_psi

    title_suffix = @sprintf("\n(serial: %d saved ψ; parallel: %d saved ψ; resonant m = %s)",
                            length(psi_s), length(psi_p), join(resonant_m, ", "))

    common_kw = (legend = :topleft,
                 left_margin = 14Plots.mm, bottom_margin = 4Plots.mm)

    # One color per resonant m
    palette = [:dodgerblue, :crimson, :forestgreen, :purple, :orange, :darkgoldenrod,
               :teal, :brown, :magenta, :olive]

    # Log-y handles the orders-of-magnitude spread between non-resonant and
    # near-resonant amplitudes (mode spikes at q = m/n can be 6+ decades
    # above the bulk).  Setting the lower y-limit from the actual minimum
    # of the data (rather than a fixed N-decade clamp) prevents cropping
    # the long radial tails of low-amplitude modes in stiff equilibria.
    function make_overlay_panel(field_sym, ylabel, title_text; show_legend::Bool = true)
        kw = (; common_kw...)
        if !show_legend
            kw = merge(kw, (; legend = false))
        end
        p = plot(; xlabel = "ψ_N", ylabel = ylabel, title = title_text,
                 yscale = :log10, kw...)
        ymin_global = Inf
        ymax_global = -Inf
        for (k, m) in enumerate(resonant_m)
            m_idx = m - mlow + 1   # 1-based index into mpert-sized mode dim
            color = palette[mod1(k, length(palette))]
            arr_s = getproperty(data_serial,   field_sym)
            arr_p = getproperty(data_parallel, field_sym)
            ys = mode_norm_over_ICs(arr_s, m_idx)
            yp = mode_norm_over_ICs(arr_p, m_idx)
            for v in ys; v > 0 && (ymin_global = min(ymin_global, v); ymax_global = max(ymax_global, v)); end
            for v in yp; v > 0 && (ymin_global = min(ymin_global, v); ymax_global = max(ymax_global, v)); end
            plot!(p, psi_s, ys; label = "serial   m=$m",
                  lw = 2, color = color, ls = :solid)
            plot!(p, psi_p, yp; label = "parallel m=$m",
                  lw = 1.5, color = color, ls = :dash, marker = :diamond, ms = 2.5,
                  markerstrokewidth = 0)
        end
        if isfinite(ymax_global)
            ylims!(p, ymin_global * 0.5, ymax_global * 2)
        end
        for ψr in sing_ψ
            vline!(p, [ψr]; ls = :dot, color = :gray, lw = 1, label = "")
        end
        return p
    end

    # Residual panel: |serial − parallel| per resonant mode.  When the dense
    # EL pass faithfully reproduces the standalone serial run, this is zero
    # to machine precision; we floor the log at eps() so the plot is finite
    # and a single horizontal line at the floor reads as "bit-identical".
    function make_residual_panel(field_sym, ylabel, title_text; show_legend::Bool = false)
        kw = (; common_kw...)
        if !show_legend
            kw = merge(kw, (; legend = false))
        end
        p = plot(; xlabel = "ψ_N", ylabel = ylabel, title = title_text,
                 yscale = :log10, kw...)
        floor_val = eps(Float64)
        ymax_global = floor_val
        for (k, m) in enumerate(resonant_m)
            m_idx = m - mlow + 1
            color = palette[mod1(k, length(palette))]
            ys = mode_norm_over_ICs(getproperty(data_serial,   field_sym), m_idx)
            yp = mode_norm_over_ICs(getproperty(data_parallel, field_sym), m_idx)
            # The two paths share the same ψ grid (verified by `summarize`)
            @assert length(ys) == length(yp) "serial/parallel ψ-grid lengths differ"
            resid = max.(abs.(ys .- yp), floor_val)
            for v in resid; v > ymax_global && (ymax_global = v); end
            plot!(p, psi_s, resid; label = "m=$m", lw = 1.6, color = color,
                  marker = :circle, ms = 2.0, markerstrokewidth = 0)
        end
        ylims!(p, floor_val * 0.5, max(ymax_global * 5, floor_val * 10))
        for ψr in sing_ψ
            vline!(p, [ψr]; ls = :dot, color = :gray, lw = 1, label = "")
        end
        return p
    end

    p1 = make_overlay_panel(:xi_psi,  "‖ξ_ψ(m, ·)‖₂",    "ξ_ψ" * title_suffix; show_legend = true)
    p2 = make_overlay_panel(:dxi_psi, "‖dξ_ψ/dψ(m, ·)‖₂", "dξ_ψ/dψ";              show_legend = false)
    p3 = make_overlay_panel(:xi_s,    "‖ξ_s(m, ·)‖₂",    "ξ_s";                  show_legend = false)
    r1 = make_residual_panel(:xi_psi,  "|Δ ξ_ψ|",        "ξ_ψ  residual"          ; show_legend = true)
    r2 = make_residual_panel(:dxi_psi, "|Δ dξ_ψ/dψ|",    "dξ_ψ/dψ  residual"      ; show_legend = false)
    r3 = make_residual_panel(:xi_s,    "|Δ ξ_s|",        "ξ_s  residual"          ; show_legend = false)

    fig = plot(p1, r1, p2, r2, p3, r3; layout = (3, 2), size = (1600, 1300),
               left_margin = 16Plots.mm, bottom_margin = 4Plots.mm,
               plot_title = "$example_name: resonant-mode ξ comparison (use_parallel vs serial)")
    base = lowercase(replace(example_name, r"[^A-Za-z0-9_]" => "_"))
    out_png = joinpath(FIG_DIR, "xi_benchmark_$(base).png")
    out_pdf = joinpath(FIG_DIR, "xi_benchmark_$(base).pdf")
    savefig(fig, out_png)
    savefig(fig, out_pdf)
    @info "  → $out_png"
    @info "  → $out_pdf"
    return fig
end


function summarize(example_name::AbstractString, data_serial, data_parallel)
    println("=" ^ 72)
    println("[$example_name]  ξ-function array shapes:")
    println("=" ^ 72)
    for (lab, d) in (("serial", data_serial), ("parallel", data_parallel))
        @printf("  %s:\n", lab)
        @printf("    psi:        %s\n", size(d.psi))
        @printf("    xi_psi:     %s\n", size(d.xi_psi))
        @printf("    dxi_psi:    %s\n", size(d.dxi_psi))
        @printf("    xi_s:       %s\n", size(d.xi_s))
        @printf("    msing:      %d\n", d.msing)
        @printf("    resonant m: %s\n", join(d.resonant_m, ", "))
    end
    println()
    println("=" ^ 72)
    println("Zero-fraction in ud_store channels  (was 100% for FM chunks before fix):")
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
    println("=" ^ 72)
    println("Resonant-mode max |·| over ψ  (serial vs parallel):")
    println("=" ^ 72)
    mlow = data_serial.mlow
    @printf("  %-4s  %-12s  %-14s  %-14s  %-14s  %-14s\n",
            "m", "channel", "max|serial|", "max|parallel|", "max|Δ|", "max|Δ|/max|·|")
    for m in data_serial.resonant_m
        m_idx = m - mlow + 1
        for (label, field) in (("xi_psi", :xi_psi), ("dxi_psi", :dxi_psi), ("xi_s", :xi_s))
            ys = mode_norm_over_ICs(getproperty(data_serial,   field), m_idx)
            yp = mode_norm_over_ICs(getproperty(data_parallel, field), m_idx)
            denom = max(maximum(ys), maximum(yp), eps())
            absdiff = maximum(abs.(ys .- yp))
            rel = absdiff / denom
            @printf("  %-4d  %-12s  %-14.6e  %-14.6e  %-14.6e  %-14.6e\n",
                    m, label, maximum(ys), maximum(yp), absdiff, rel)
        end
    end
    println()

    # ψ-grid check: are the two paths literally on the same ψ snapshots?
    if length(data_serial.psi) == length(data_parallel.psi)
        max_dpsi = maximum(abs.(data_serial.psi .- data_parallel.psi))
        @printf("  ψ-grid:  same length (%d), max|Δψ| = %.6e\n",
                length(data_serial.psi), max_dpsi)
    else
        @printf("  ψ-grid:  DIFFERENT lengths — serial %d, parallel %d\n",
                length(data_serial.psi), length(data_parallel.psi))
    end
    println()
end


function benchmark_example(example_name::AbstractString)
    example_dir = joinpath(EXAMPLES_ROOT, example_name)
    isdir(example_dir) || error("example directory not found: $example_dir")
    @info ""
    @info "════════════════════════════════════════════════════════════════"
    @info "  Benchmarking example: $example_name"
    @info "════════════════════════════════════════════════════════════════"
    h5_serial   = run_with_use_parallel(example_dir, false)
    h5_parallel = run_with_use_parallel(example_dir, true)

    @info "Reading ξ functions from both HDF5 outputs"
    data_serial   = read_xi(h5_serial)
    data_parallel = read_xi(h5_parallel)

    summarize(example_name, data_serial, data_parallel)
    plot_overlay(example_name, data_serial, data_parallel)
end


function main()
    # Default: benchmark both the Solovev analytic case and the DIII-D-like
    # geqdsk case.  Override by passing one or more example dir names on the
    # command line.
    examples = isempty(ARGS) ?
        ["Solovev_ideal_example", "DIIID-like_ideal_example"] :
        ARGS
    for ex in examples
        benchmark_example(ex)
    end
    @info "Done."
end


main()
