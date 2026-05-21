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

    local odet, vac
    elapsed = @elapsed begin
        odet, _, _, _ = FFS.eulerlagrange_integration(ctrl, equil, ffit, intr)
        vac = FFS.free_run!(odet, ctrl, equil, ffit, intr)
    end
    return odet, vac, intr, elapsed
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

    odet, vac, intr_std, telapsed =
        mktempdir() do tmp
            work = joinpath(tmp, example_name)
            cp(example_dir, work)
            odet = Dict{Symbol,Any}()
            vac = Dict{Symbol,Any}()
            telapsed = Dict{Symbol,Float64}()
            intr_ref = nothing
            for mode in (:standard, :riccati, :parallel)
                println("  running $mode ...")
                o, v, i, t = run_case(work, mode)
                odet[mode] = o
                vac[mode] = v
                telapsed[mode] = t
                intr_ref = i
            end
            return (odet, vac, intr_ref, telapsed)
        end

    # Eigenmode radial displacement ξ_m(ψ) for each path (projected with its own
    # eigenvector), interpolated to a common ψ grid.
    raw = Dict(m => eigenmode_displacement(odet[m], vac[m]) for m in (:standard, :riccati, :parallel))
    N = size(raw[:standard][2], 1)
    nplot = min(5, N)
    lo = maximum(minimum(raw[m][1]) for m in keys(raw))
    hi = minimum(maximum(raw[m][1]) for m in keys(raw))
    psg = collect(range(lo, hi; length=400))

    xi = Dict{Symbol,Matrix{ComplexF64}}()
    for mode in (:standard, :riccati, :parallel)
        psi_m, xi_m, _ = raw[mode]
        d = zeros(ComplexF64, nplot, length(psg))
        for m in 1:nplot
            d[m, :] .= interp1(psi_m, ComplexF64.(xi_m[m, :]), psg)
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
    # Direct u_store[:,:,1,:] comparison (diagonal m=j), interpolated to psg — gauge/
    # basis-independent only if both paths share the canonical axis basis.
    let
        ud = Dict{Symbol,Matrix{ComplexF64}}()
        for mode in (:standard, :parallel)
            o = odet[mode]
            np = length(o.psi_store)
            pm = collect(o.psi_store[1:np])
            d = zeros(ComplexF64, nplot, length(psg))
            for m in 1:nplot
                d[m, :] .= interp1(pm, ComplexF64.(o.u_store[m, m, 1, 1:np]), psg)
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
            for m in 1:nplot
                Sdiag = ComplexF64[]
                for k in 1:np
                    U1 = @view o.u_store[:, :, 1, k]
                    U2 = @view o.u_store[:, :, 2, k]
                    push!(Sdiag, try (U1 / U2)[m, m] catch; ComplexF64(NaN) end)
                end
                d[m, :] .= interp1(pm, Sdiag, psg)
            end
            Sd[mode] = d
        end
        @printf("  S(ψ)=U₁U₂⁻¹ diagonal rel-L1 (parallel vs standard): %.3e\n",
                reldiff(Sd[:parallel], Sd[:standard]))
    end
    @printf("  ξ_m(ψ) rel-L1:   parallel vs standard %.3e   serial-riccati vs standard %.3e\n",
            reldiff(xi[:parallel], xi[:standard]), reldiff(xi[:riccati], xi[:standard]))
    println("  per-mode ξ_m(ψ) rel-L1  (parallel vs standard | serial-riccati vs standard):")
    for m in 1:nplot
        pm = intr_std.mlow + (m - 1)
        @printf("    m = %+d :  %.3e   |   %.3e\n", pm,
                reldiff(xi[:parallel][m:m, :], xi[:standard][m:m, :]),
                reldiff(xi[:riccati][m:m, :], xi[:standard][m:m, :]))
    end

    # Overlay plot: Re and Im of the least-stable eigenmode displacement ξ_m(ψ) for the
    # first five poloidal harmonics — parallel (solid), serial-riccati (dotted),
    # standard (dashed). With the canonical u_store all three should overlay.
    mkpath(FIG_DIR)
    palette = distinguishable_colors(nplot)
    preal = plot(; title="$example_name — Re ξ_m(ψ), least-stable mode",
                 xlabel="ψ_n", ylabel="Re ξ_m  (normalized)",
                 left_margin=12Plots.mm, bottom_margin=4Plots.mm, legend=:outertopright)
    pimag = plot(; title="$example_name — Im ξ_m(ψ), least-stable mode",
                 xlabel="ψ_n", ylabel="Im ξ_m  (normalized)",
                 left_margin=12Plots.mm, bottom_margin=4Plots.mm, legend=:outertopright)
    for m in 1:nplot
        pm = intr_std.mlow + (m - 1)
        plot!(preal, psg, real(xi[:parallel][m, :]); color=palette[m], lw=2, label="m=$pm parallel")
        plot!(preal, psg, real(xi[:riccati][m, :]); color=palette[m], lw=1, ls=:dot, label="")
        plot!(preal, psg, real(xi[:standard][m, :]); color=palette[m], lw=1, ls=:dash, label="m=$pm standard")
        plot!(pimag, psg, imag(xi[:parallel][m, :]); color=palette[m], lw=2, label="m=$pm parallel")
        plot!(pimag, psg, imag(xi[:riccati][m, :]); color=palette[m], lw=1, ls=:dot, label="")
        plot!(pimag, psg, imag(xi[:standard][m, :]); color=palette[m], lw=1, ls=:dash, label="m=$pm standard")
    end
    fig = plot(preal, pimag; layout=(2, 1), size=(900, 800))
    outpath = abspath(joinpath(FIG_DIR, "parallel_u_store_$(example_name).png"))
    savefig(fig, outpath)
    println("\n  figure saved: $outpath")
    return nothing
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
