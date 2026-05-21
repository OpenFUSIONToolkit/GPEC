#!/usr/bin/env julia
# benchmark_parallel_u_store.jl — verify that the single-pass parallel-Riccati
# integration recovers the same DCON eigenmode displacement as the standard
# Euler-Lagrange path.
#
# The parallel-FM path reconstructs a dense `u_store` directly from the per-chunk
# propagator histories (see assemble_parallel_dense! / apply_edge_gauge! in
# ForceFreeStates/Riccati.jl) — no second serial-EL pass. This script runs each
# example twice, with `use_parallel = false` (standard EL) and `use_parallel = true`
# (single-pass parallel-Riccati), projects the least-stable boundary energy
# eigenmode onto u_store to get the radial displacement ξ_m(ψ), overlays the two,
# and prints a quantitative comparison.
#
# Usage:
#     julia --project=. benchmarks/benchmark_parallel_u_store.jl
#     julia --project=. benchmarks/benchmark_parallel_u_store.jl Solovev_ideal_example

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GeneralizedPerturbedEquilibrium
using LinearAlgebra
using Plots
using Printf
using TOML

const FFS = GeneralizedPerturbedEquilibrium.ForceFreeStates
const Eq  = GeneralizedPerturbedEquilibrium.Equilibrium
const Vac = GeneralizedPerturbedEquilibrium.Vacuum
const FT  = GeneralizedPerturbedEquilibrium.ForcingTerms
const PE  = GeneralizedPerturbedEquilibrium.PerturbedEquilibrium

const EXAMPLES_ROOT = joinpath(@__DIR__, "..", "examples")
const FIG_DIR       = joinpath(@__DIR__, "figures")

"""
Run one example end-to-end (equilibrium → fixed-boundary integration → free-boundary
energies) with the chosen integration path `mode ∈ {:standard, :riccati, :parallel}`.
Returns `(odet, vac, intr, elapsed)`.
"""
function run_case(example_dir::AbstractString, mode::Symbol)
    inputs = TOML.parsefile(joinpath(example_dir, "gpec.toml"))
    inputs["ForceFreeStates"]["verbose"] = false
    inputs["ForceFreeStates"]["use_parallel"] = (mode === :parallel)
    inputs["ForceFreeStates"]["use_riccati"] = (mode === :riccati)
    inputs["ForceFreeStates"]["write_outputs_to_HDF5"] = false

    intr = FFS.ForceFreeStatesInternal(; dir_path=example_dir)
    ctrl = FFS.ForceFreeStatesControl(; (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)

    eq_config = Eq.EquilibriumConfig(inputs["Equilibrium"], example_dir)
    additional = nothing
    if eq_config.eq_type == "sol" && haskey(inputs, "SOL_INPUT")
        additional = Eq.SolovevConfig(inputs["SOL_INPUT"])
    elseif eq_config.eq_type in ("tj_analytic", "tj_analytic_direct") && haskey(inputs, "TJ_ANALYTIC_INPUT")
        additional = Eq.TJAnalyticConfig(inputs["TJ_ANALYTIC_INPUT"])
    elseif eq_config.eq_type == "lar" && haskey(inputs, "LAR_INPUT")
        additional = Eq.LargeAspectRatioConfig(inputs["LAR_INPUT"])
    end
    equil = additional === nothing ? Eq.setup_equilibrium(eq_config) :
            Eq.setup_equilibrium(eq_config, additional)

    intr.wall_settings = haskey(inputs, "Wall") ?
        Vac.WallShapeSettings(; (Symbol(k) => v for (k, v) in inputs["Wall"])...) :
        Vac.WallShapeSettings()

    FFS.sing_lim!(intr, ctrl, equil)
    intr.nlow = ctrl.nn_low
    intr.nhigh = ctrl.nn_high
    intr.npert = 1
    FFS.sing_find!(intr, equil)
    intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
    intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
    intr.mpert = intr.mhigh - intr.mlow + 1
    intr.mband = intr.mpert - 1
    intr.numpert_total = intr.mpert * intr.npert

    metric = FFS.make_metric(equil; mband=intr.mband, fft_flag=ctrl.fft_flag)
    ffit = FFS.make_matrix(equil, intr, metric)

    local odet, vac, fm_propagators, fm_chunks, fm_S_left
    elapsed = @elapsed begin
        odet, fm_propagators, fm_chunks, fm_S_left =
            FFS.eulerlagrange_integration(ctrl, equil, ffit, intr)
        vac = FFS.free_run!(odet, ctrl, equil, ffit, intr)
    end

    pe_state = nothing
    if haskey(inputs, "PerturbedEquilibrium")
        forcing_raw = get(inputs, "ForcingTerms", Dict{String,Any}())
        coil_sets_raw = Vector{Dict{String,Any}}(get(forcing_raw, "coil_set", Dict{String,Any}[]))
        scalar_forcing = filter(p -> p.first != "coil_set", forcing_raw)
        ft_ctrl = FT.ForcingTermsControl(; (Symbol(k) => v for (k, v) in scalar_forcing)...)
        ft_ctrl.coil_sets_raw = coil_sets_raw

        pe_ctrl_dict = copy(inputs["PerturbedEquilibrium"])
        pe_ctrl_dict["verbose"] = false
        pe_ctrl_dict["write_outputs_to_HDF5"] = false
        pe_ctrl = PE.PerturbedEquilibriumControl(; (Symbol(k) => v for (k, v) in pe_ctrl_dict)...)
        pe_intr = PE.PerturbedEquilibriumInternal(; dir_path=example_dir)

        pe_state = PE.compute_perturbed_equilibrium(
            equil, odet, vac, intr, ft_ctrl, pe_ctrl, pe_intr, metric, ffit
        )
    end

    return odet, vac, intr, elapsed, pe_state
end

"""
Project the least-stable boundary-energy eigenmode onto a path's `u_store` to get the
radial displacement ξ_m(ψ). Returns `(psi, xi, evals)`; `xi` of shape (numpert_total, npsi),
`evals` the sorted real eigenvalues (for a degeneracy check). Each path is projected with
its own eigenvector — the physical eigenmode is gauge-/order-invariant when `u_store` and
`e` come from the same path.
"""
function eigenmode_displacement(odet, vac)
    F = eigen(vac.wt)
    order = sortperm(real.(F.values))
    e = F.vectors[:, order[1]]
    npsi = length(odet.psi_store)
    N = size(odet.u_store, 1)
    psi = collect(odet.psi_store[1:npsi])
    xi = zeros(ComplexF64, N, npsi)
    for k in 1:npsi
        @views xi[:, k] .= odet.u_store[:, :, 1, k] * e
    end
    return psi, xi, real.(F.values[order])
end

"""
Compare PerturbedEquilibrium outputs across the three integration paths and emit a
TOML-serializable snapshot. Primary metric is `||Φ_res||`; per-surface arrays are
included for trend tracking and diagnosing where paths diverge. Returns `nothing`
when none of the paths ran PE (no [PerturbedEquilibrium] in gpec.toml).
"""
function compare_pe_outputs(pe_states::AbstractDict, example_name::AbstractString)
    have = filter(p -> p.second !== nothing, pe_states)
    isempty(have) && return nothing

    println("\n  PerturbedEquilibrium outputs (per-path):")
    println("  " * "-"^68)
    @printf("  %-9s  %-14s  %s\n", "path", "||Φ_res||", "rational q values")
    function _phi_norm(st)
        rf = getfield(st, :resonant_flux)
        return isempty(rf) ? NaN : norm(rf)
    end
    qs_of(st) = collect(Float64.(getfield(st, :rational_q)))

    qs = nothing
    for mode in (:standard, :riccati, :parallel)
        haskey(have, mode) || continue
        st = have[mode]
        qs = qs_of(st)
        qs_str = isempty(qs) ? "(none)" : join((@sprintf("%.3f", q) for q in qs), ", ")
        @printf("  %-9s  %.6e  [%s]\n", String(mode), _phi_norm(st), qs_str)
    end

    function _diff_norm(a, b)
        nb = sum(abs2, b)
        nb == 0 && return NaN
        return sqrt(sum(abs2, a .- b) / nb)
    end
    if haskey(have, :standard)
        st_std = have[:standard]
        rf_std = collect(getfield(st_std, :resonant_flux))
        dp_std = collect(getfield(st_std, :delta_prime))
        println("  " * "-"^68)
        println("  cross-path relative differences vs :standard")
        for mode in (:parallel, :riccati)
            haskey(have, mode) || continue
            st = have[mode]
            rf = collect(getfield(st, :resonant_flux))
            dp = collect(getfield(st, :delta_prime))
            @printf("    %-9s  ||Δrf||/||rf||=%.3e   ||Δdp||/||dp||=%.3e\n",
                    String(mode), _diff_norm(rf, rf_std), _diff_norm(dp, dp_std))
        end
    end

    snapshot = Dict{String,Any}(
        "example" => example_name,
        "rational_q" => qs === nothing ? Float64[] : qs,
    )
    for mode in (:standard, :riccati, :parallel)
        haskey(have, mode) || continue
        st = have[mode]
        snapshot[String(mode)] = Dict{String,Any}(
            "phi_res_norm" => _phi_norm(st),
            "resonant_flux_re" => real.(getfield(st, :resonant_flux)),
            "resonant_flux_im" => imag.(getfield(st, :resonant_flux)),
            "delta_prime_re" => real.(getfield(st, :delta_prime)),
            "delta_prime_im" => imag.(getfield(st, :delta_prime)),
            "island_half_width" => collect(Float64.(getfield(st, :island_half_width))),
            "chirikov_parameter" => collect(Float64.(getfield(st, :chirikov_parameter))),
        )
    end
    return snapshot
end

"""Relative L2 difference of `a` vs `b` minimized over a global complex scalar α — the
right metric for an eigenmode, which is defined only up to such a scalar."""
function scalar_invariant_reldiff(a, b)
    α = sum(conj.(b) .* a) / (sum(abs2, b) + eps())
    return sqrt(sum(abs2, a .- α .* b) / (sum(abs2, b) + eps()))
end

"""Linear interpolation of `y` sampled at `x` onto the grid `xq` (x assumed sorted)."""
function interp1(x::Vector{Float64}, y::Vector{ComplexF64}, xq::AbstractVector{Float64})
    out = zeros(ComplexF64, length(xq))
    for (i, xi) in enumerate(xq)
        if xi <= x[1]
            out[i] = y[1]
        elseif xi >= x[end]
            out[i] = y[end]
        else
            j = searchsortedlast(x, xi)
            t = (xi - x[j]) / (x[j+1] - x[j])
            out[i] = (1 - t) * y[j] + t * y[j+1]
        end
    end
    return out
end

function benchmark_example(example_name::AbstractString)
    example_dir = joinpath(EXAMPLES_ROOT, example_name)
    isdir(example_dir) || error("example directory not found: $example_dir")

    # Copy inputs to a temp directory so the working tree stays clean (equilibrium
    # setup may refresh side-car cache files).
    println("\n", "="^70)
    println("Example: $example_name")
    println("="^70)

    odet, vac, intr_std, telapsed, pe_states =
        mktempdir() do tmp
            work = joinpath(tmp, example_name)
            cp(example_dir, work)
            odet = Dict{Symbol,Any}()
            vac = Dict{Symbol,Any}()
            telapsed = Dict{Symbol,Float64}()
            pe = Dict{Symbol,Any}()
            intr_ref = nothing
            for mode in (:standard, :riccati, :parallel)
                println("  running $mode ...")
                o, v, i, t, p = run_case(work, mode)
                odet[mode] = o
                vac[mode] = v
                telapsed[mode] = t
                pe[mode] = p
                intr_ref = i
            end
            return (odet, vac, intr_ref, telapsed, pe)
        end

    # Eigenmode radial displacement ξ_m(ψ) for each path (projected with its own
    # eigenvector), interpolated to a common ψ grid.
    raw = Dict(m => eigenmode_displacement(odet[m], vac[m]) for m in (:standard, :riccati, :parallel))
    N = size(raw[:standard][2], 1)

    # Pick poloidal harmonics that carry the plasma helicity (m = n·q across the
    # rational surfaces) plus one neighbour on each side. Far-negative-m solutions
    # are tiny on tokamak equilibria and aren't what we care about for PE coupling.
    resonant_ms = sort!(unique(Int[s.m[1] for s in intr_std.sing if !isempty(s.m)]))
    plot_ms = if isempty(resonant_ms)
        # Fallback: middle of the mode range
        mmid = intr_std.mlow + N ÷ 2
        collect((mmid - 2):(mmid + 2))
    else
        m_lo = max(intr_std.mlow, minimum(resonant_ms) - 1)
        m_hi = min(intr_std.mlow + N - 1, maximum(resonant_ms) + 1)
        collect(m_lo:m_hi)
    end
    plot_idx = [m - intr_std.mlow + 1 for m in plot_ms]
    nplot = length(plot_ms)

    lo = maximum(minimum(raw[m][1]) for m in keys(raw))
    hi = minimum(maximum(raw[m][1]) for m in keys(raw))
    psg = collect(range(lo, hi; length=400))

    xi = Dict{Symbol,Matrix{ComplexF64}}()
    for mode in (:standard, :riccati, :parallel)
        psi_m, xi_m, _ = raw[mode]
        d = zeros(ComplexF64, nplot, length(psg))
        for (k, idx) in enumerate(plot_idx)
            d[k, :] .= interp1(psi_m, ComplexF64.(xi_m[idx, :]), psg)
        end
        xi[mode] = d
    end
    # The eigenmode is defined only up to a global complex scalar. Align riccati/parallel
    # to the standard path by the optimal α = ⟨std,mode⟩/⟨mode,mode⟩; after this,
    # reldiff(xi[mode], xi[standard]) is the scalar-invariant relative difference.
    for mode in (:riccati, :parallel)
        α = sum(conj.(xi[mode]) .* xi[:standard]) / (sum(abs2, xi[mode]) + eps())
        xi[mode] .*= α
    end

    reldiff(a, b) = sum(abs.(a .- b)) / (sum(abs.(b)) + eps())
    println()
    for mode in (:standard, :riccati, :parallel)
        @printf("  %-9s  saved steps %4d   runtime %5.1fs   ifix %d   max|u_store| %.2e\n",
                mode, length(raw[mode][1]), telapsed[mode], odet[mode].ifix,
                maximum(abs, odet[mode].u_store))
        evs = raw[mode][3]
        @printf("            least-stable eigenvalues: %s\n",
                join((@sprintf("%+.5e", v) for v in evs[1:min(5, length(evs))]), "  "))
    end
    # Direct u_store[:,:,1,:] comparison on the helicity m-row diagonal, interpolated
    # to psg — gauge/basis-independent only if both paths share the canonical axis basis.
    let
        ud = Dict{Symbol,Matrix{ComplexF64}}()
        for mode in (:standard, :parallel)
            o = odet[mode]
            np = length(o.psi_store)
            pm = collect(o.psi_store[1:np])
            d = zeros(ComplexF64, nplot, length(psg))
            for (k, idx) in enumerate(plot_idx)
                d[k, :] .= interp1(pm, ComplexF64.(o.u_store[idx, idx, 1, 1:np]), psg)
            end
            ud[mode] = d
        end
        @printf("  u_store[:,:,1] diagonal rel-L1 (parallel vs standard): %.3e\n",
                reldiff(ud[:parallel], ud[:standard]))
    end
    # Gauge-invariant Riccati matrix S(ψ) = U₁·U₂⁻¹ diagonal — independent of the
    # axis-basis gauge, so it must agree if the parallel path recovers the right physics.
    let
        Sd = Dict{Symbol,Matrix{ComplexF64}}()
        for mode in (:standard, :parallel)
            o = odet[mode]
            np = length(o.psi_store)
            pm = collect(o.psi_store[1:np])
            d = zeros(ComplexF64, nplot, length(psg))
            for (k, idx) in enumerate(plot_idx)
                Sdiag = ComplexF64[]
                for j in 1:np
                    U1 = @view o.u_store[:, :, 1, j]
                    U2 = @view o.u_store[:, :, 2, j]
                    push!(Sdiag, try (U1 / U2)[idx, idx] catch; ComplexF64(NaN) end)
                end
                d[k, :] .= interp1(pm, Sdiag, psg)
            end
            Sd[mode] = d
        end
        @printf("  S(ψ)=U₁U₂⁻¹ diagonal rel-L1 (parallel vs standard): %.3e\n",
                reldiff(Sd[:parallel], Sd[:standard]))
    end
    @printf("  ξ_m(ψ) rel-L1:   parallel vs standard %.3e   serial-riccati vs standard %.3e\n",
            reldiff(xi[:parallel], xi[:standard]), reldiff(xi[:riccati], xi[:standard]))
    println("  per-mode ξ_m(ψ) rel-L1  (parallel vs standard | serial-riccati vs standard):")
    for (k, m) in enumerate(plot_ms)
        @printf("    m = %+d :  %.3e   |   %.3e\n", m,
                reldiff(xi[:parallel][k:k, :], xi[:standard][k:k, :]),
                reldiff(xi[:riccati][k:k, :], xi[:standard][k:k, :]))
    end

    pe_snapshot = compare_pe_outputs(pe_states, example_name)
    if pe_snapshot !== nothing
        mkpath(FIG_DIR)
        snap_path = abspath(joinpath(FIG_DIR, "parallel_u_store_pe_$(example_name).toml"))
        open(snap_path, "w") do io
            TOML.print(io, pe_snapshot)
        end
        println("\n  PE snapshot saved: $snap_path")
        pe_fig_path = abspath(joinpath(FIG_DIR, "parallel_u_store_pe_$(example_name).png"))
        plot_pe_comparison(pe_snapshot, example_name, pe_fig_path)
        println("  PE comparison figure: $pe_fig_path")
    end

    # Overlay plot: Re and Im of the least-stable eigenmode displacement ξ_m(ψ) for the
    # helicity-range poloidal harmonics (m ≈ n·q at the rational surfaces) —
    # parallel (solid), serial-riccati (dotted), standard (dashed). With the canonical
    # u_store all three should overlay.
    mkpath(FIG_DIR)
    palette = distinguishable_colors(nplot)
    preal = plot(; title="$example_name — Re ξ_m(ψ), least-stable mode",
                 xlabel="ψ_n", ylabel="Re ξ_m  (normalized)",
                 left_margin=12Plots.mm, bottom_margin=4Plots.mm, legend=:outertopright)
    pimag = plot(; title="$example_name — Im ξ_m(ψ), least-stable mode",
                 xlabel="ψ_n", ylabel="Im ξ_m  (normalized)",
                 left_margin=12Plots.mm, bottom_margin=4Plots.mm, legend=:outertopright)
    for (k, m) in enumerate(plot_ms)
        plot!(preal, psg, real(xi[:parallel][k, :]); color=palette[k], lw=2, label="m=$m parallel")
        plot!(preal, psg, real(xi[:riccati][k, :]); color=palette[k], lw=1, ls=:dot, label="")
        plot!(preal, psg, real(xi[:standard][k, :]); color=palette[k], lw=1, ls=:dash, label="m=$m standard")
        plot!(pimag, psg, imag(xi[:parallel][k, :]); color=palette[k], lw=2, label="m=$m parallel")
        plot!(pimag, psg, imag(xi[:riccati][k, :]); color=palette[k], lw=1, ls=:dot, label="")
        plot!(pimag, psg, imag(xi[:standard][k, :]); color=palette[k], lw=1, ls=:dash, label="m=$m standard")
    end
    fig = plot(preal, pimag; layout=(2, 1), size=(900, 800))
    outpath = abspath(joinpath(FIG_DIR, "parallel_u_store_$(example_name).png"))
    savefig(fig, outpath)
    println("\n  ξ_m figure saved: $outpath")
    return nothing
end

"""
Render a 2x2 panel comparing per-rational-surface PE outputs across the three paths:
- top-left:  |Φ_res|     (resonant flux magnitude per surface)
- top-right: arg(Φ_res)  (phase, since cancellations matter as much as magnitude)
- bot-left:  |Δ'|        (tearing parameter magnitude per surface)
- bot-right: w_island    (island half-width per surface)

Each panel groups three side-by-side bars per rational surface (one per path), so the
eye can compare per-surface agreement between :standard / :riccati / :parallel directly.
"""
function plot_pe_comparison(snapshot::AbstractDict, example_name::AbstractString, outpath::AbstractString)
    paths = [m for m in ("standard", "riccati", "parallel") if haskey(snapshot, m)]
    isempty(paths) && return
    qs = get(snapshot, "rational_q", Float64[])
    isempty(qs) && return
    n_surf = length(qs)
    surf_labels = ["q=" * (@sprintf("%.2f", q)) for q in qs]
    surf_x = collect(1.0:n_surf)
    offsets = Dict("standard" => -0.18, "riccati" => 0.0, "parallel" => +0.18)
    colors  = Dict("standard" => :tomato, "riccati" => :forestgreen, "parallel" => :steelblue)
    markers = Dict("standard" => :circle, "riccati" => :diamond, "parallel" => :utriangle)

    function _get(path, key)
        d = snapshot[path]
        v = collect(Float64.(d[key]))
        length(v) < n_surf ? vcat(v, fill(NaN, n_surf - length(v))) : v[1:n_surf]
    end
    function _cvec(path, base)
        re = _get(path, base * "_re")
        im = _get(path, base * "_im")
        return complex.(re, im)
    end

    common = (; left_margin=10Plots.mm, bottom_margin=4Plots.mm,
              xticks=(surf_x, surf_labels), legend=:outertopright,
              xlims=(0.5, n_surf + 0.5))

    p_phi_mag = plot(; title="|Φ_res| per surface", ylabel="|Φ_res|",
                     yscale=:log10, common...)
    p_phi_arg = plot(; title="arg(Φ_res) per surface", ylabel="arg(Φ_res) [rad]",
                     ylims=(-π - 0.1, π + 0.1), common...)
    p_dp_mag  = plot(; title="|Δ'| per surface", ylabel="|Δ'|", yscale=:log10, common...)
    p_w       = plot(; title="island half-width per surface", ylabel="w_island",
                     yscale=:log10, common...)

    # Stem-style markers; log-scale axes handle wide dynamic ranges (DIIID standard
    # q=2 |Φ_res| can be 6 OOM larger than other surfaces). One marker per path per
    # surface — eye-friendly when paths disagree by orders of magnitude.
    function _scatter_stem!(p, xs, ys, color, marker, label; logy=false)
        valid = isfinite.(ys)
        if logy
            # Drop non-positive values (log axis can't render them); also drop tiny
            # values below a relative floor so the y-axis doesn't get distorted.
            ys = copy(ys)
            ys[.!valid .| (ys .<= 0)] .= NaN
        end
        scatter!(p, xs, ys; color=color, marker=marker, ms=7, lw=0, label=label)
    end

    for path in paths
        phi = _cvec(path, "resonant_flux")
        dp  = _cvec(path, "delta_prime")
        w   = _get(path, "island_half_width")
        xshift = surf_x .+ offsets[path]
        _scatter_stem!(p_phi_mag, xshift, abs.(phi), colors[path], markers[path], path; logy=true)
        _scatter_stem!(p_phi_arg, xshift, angle.(phi), colors[path], markers[path], path)
        _scatter_stem!(p_dp_mag,  xshift, abs.(dp),  colors[path], markers[path], path; logy=true)
        _scatter_stem!(p_w,       xshift, w,        colors[path], markers[path], path; logy=true)
    end

    fig = plot(p_phi_mag, p_phi_arg, p_dp_mag, p_w;
               layout=(2, 2), size=(1200, 800),
               plot_title="$example_name — PE outputs across integration paths")
    savefig(fig, outpath)
    return outpath
end

function main()
    examples = isempty(ARGS) ?
        ["Solovev_ideal_example", "DIIID-like_ideal_example"] : ARGS
    for ex in examples
        benchmark_example(ex)
    end
    println()
end

main()
