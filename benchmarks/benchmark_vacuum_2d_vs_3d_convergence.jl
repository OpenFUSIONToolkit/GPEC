#!/usr/bin/env julia
#
# Convergence test of the 3D vacuum solver toward the 2D reference for the
# axisymmetric Solovev equilibrium used by `examples/Solovev_ideal_example_3D`.
#
# For n = 1 and m = -15:15, we:
#   1. Compute the vacuum response matrix Wv in 2D (nzeta = 1) at high
#      poloidal resolution. This serves as the reference.
#   2. Compute Wv in 3D for an increasing sequence of (mtheta, nzeta) grids,
#      keeping the ratio mtheta/nzeta approximately constant (matching the
#      mthvac/nzvac ratio used in the example's gpec.toml).
#   3. Report and plot the relative Frobenius-norm error
#          ||Wv_3D - Wv_2D|| / ||Wv_2D||
#      for both `nowall` and `conformal` wall configurations.
#
# Usage:
#   julia --project=. benchmarks/benchmark_vacuum_2d_vs_3d_convergence.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using LinearAlgebra
using Printf
using TOML
using JLD2
using Plots
using LaTeXStrings

using GeneralizedPerturbedEquilibrium
const GPEC = GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Vacuum
using GeneralizedPerturbedEquilibrium.Equilibrium

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

"""
Construct the Solovev equilibrium used by the 3D example.
"""
function load_solovev_equilibrium()
    example_dir = joinpath(@__DIR__, "..", "examples", "Solovev_ideal_example_3D")
    inputs = TOML.parsefile(joinpath(example_dir, "gpec.toml"))
    eq_config = Equilibrium.EquilibriumConfig(inputs["Equilibrium"], example_dir)
    equil = Equilibrium.setup_equilibrium(eq_config)
    return equil, inputs
end

"""
Load example TOML inputs without building equilibrium.
"""
function load_solovev_example_inputs()
    example_dir = joinpath(@__DIR__, "..", "examples", "Solovev_ideal_example_3D")
    return TOML.parsefile(joinpath(example_dir, "gpec.toml"))
end

"""
Build VacuumInput for a given (mtheta, nzeta), holding all mode parameters fixed.
"""
function make_vac_inputs(equil, psi; mtheta::Int, nzeta::Int, mpert::Int, mlow::Int, nlow::Int, npert::Int=1)
    return Vacuum.VacuumInput(
        equil, psi,
        mtheta, nzeta,
        mpert, mlow,
        npert, nlow;
        force_wv_symmetry=true
    )
end

"""
Run a single vacuum response solve and return the Wv matrix.
"""
function run_wv(vac_inputs::VacuumInput, wall_settings::WallShapeSettings)
    wv, _, _, _, _ = Vacuum.compute_vacuum_response(vac_inputs, wall_settings)
    return Matrix(wv)
end

"""
Build a deterministic cache tag from benchmark physics/settings (excluding sweep points).
"""
function make_cache_tag(; mlow, mhigh, nlow, npert, mtheta_2D, aspect_mt_over_nz, a_conformal)
    aspect_str = replace(@sprintf("%.6f", aspect_mt_over_nz), "." => "p")
    a_str = replace(@sprintf("%.6f", a_conformal), "." => "p")
    return "mlow$(mlow)_mhigh$(mhigh)_n$(nlow)_npert$(npert)_m2d$(mtheta_2D)_aspect$(aspect_str)_a$(a_str)"
end

const CACHE_SCHEMA_VERSION = 2

cache_point_key(mtheta::Int, nzeta::Int) = "mtheta=$(mtheta)_nzeta=$(nzeta)"

"""
Initialize a cache container for this benchmark run.
"""
function init_cache_data(a_conformal)
    return Dict(
        "schema_version" => CACHE_SCHEMA_VERSION,
        "a_conformal" => a_conformal,
        "wv_2D_ref" => Dict{String,Matrix{ComplexF64}}(),
        "wv_3D" => Dict{String,Dict{String,Matrix{ComplexF64}}}(
            "nowall" => Dict{String,Matrix{ComplexF64}}(),
            "conformal" => Dict{String,Matrix{ComplexF64}}()
        ),
        "timings_2D" => Dict{String,Float64}(),
        "timings_3D" => Dict{String,Dict{String,Float64}}(
            "nowall" => Dict{String,Float64}(),
            "conformal" => Dict{String,Float64}()
        )
    )
end

"""
Whether the cache file matches the current schema.
"""
function cache_is_compatible(cache_data)
    return get(cache_data, "schema_version", 0) == CACHE_SCHEMA_VERSION
end

"""
Check if all required points are already in cache.
"""
function cache_is_complete(cache_data, mtheta_values, nzeta_for)
    has_2d = all(haskey(cache_data["wv_2D_ref"], w) for w in ("nowall", "conformal"))
    has_3d = all(
        haskey(cache_data["wv_3D"][w], cache_point_key(mtheta, nzeta_for(mtheta)))
        for w in ("nowall", "conformal"), mtheta in mtheta_values
    )
    return has_2d && has_3d
end

"""
Build results arrays from cached 2D/3D Wv matrices.
"""
function results_from_cache(cache_data, mtheta_values, nzeta_for)
    wv_2D_ref = cache_data["wv_2D_ref"]
    results = Dict{String,Vector{NamedTuple{(:mtheta, :nzeta, :relerr, :t),Tuple{Int,Int,Float64,Float64}}}}(
        "nowall" => NamedTuple{(:mtheta, :nzeta, :relerr, :t),Tuple{Int,Int,Float64,Float64}}[],
        "conformal" => NamedTuple{(:mtheta, :nzeta, :relerr, :t),Tuple{Int,Int,Float64,Float64}}[]
    )

    for mtheta in mtheta_values
        nzeta = nzeta_for(mtheta)
        pkey = cache_point_key(mtheta, nzeta)
        for wname in ("nowall", "conformal")
            wv = cache_data["wv_3D"][wname][pkey]
            wv_ref = wv_2D_ref[wname]
            relerr = norm(wv .- wv_ref) / norm(wv_ref)
            t = get(cache_data["timings_3D"][wname], pkey, NaN)
            push!(results[wname], (; mtheta, nzeta, relerr, t))
        end
    end
    return wv_2D_ref, results
end

"""
Render and save the convergence plot from precomputed results.
"""
function save_convergence_plot(results, a_conformal)
    plt = plot(;
        xscale=:log10, yscale=:log10,
        xlabel=L"\mathbf{N_\theta \times N_\zeta}",
        # xlabel="# of grid points",

        ylabel=L"\|\| \mathbf{W}_V^{3D} - \mathbf{W}_V^{2D} \|\| / \|\| \mathbf{W}_V^{2D} \|\|",
        title="Solovev Tokamak\nn = 1, m = [-15, 15]",
        legend=:bottomleft,
        size=(325, 500),
        dpi=300,
        legendfontsize=11,
        guidefontsize=16,
        tickfontsize=14,
        titlefontsize=16,
        left_margin=4Plots.mm,
        bottom_margin=0Plots.mm,
        right_margin=3Plots.mm,
        minorgrid=true,
        foreground_color_legend=:black,
        background_color_legend=RGBA(1, 1, 1, 0.85),
        xlims=(1e2, 1e5)
    )
    for wname in ("nowall", "conformal")
        xs = [r.mtheta * r.nzeta for r in results[wname]]
        ys = [r.relerr for r in results[wname]]
        label_str = wname == "nowall" ?
                    L"\mathbf{No\ wall}" :
                    L"\mathbf{Wall}"
        line_style = wname == "conformal" ? :dot : :solid
        color = wname == "nowall" ? :blue : :orange
        plot!(plt, xs, ys; lw=3, linestyle=line_style, label=label_str, color=color)
        plot!(plt, xs, ys; marker=:circle, ms=5, lw=0, label="", color=color)
    end


    outpath = joinpath(@__DIR__, "vacuum_2d_vs_3d_convergence.png")
    savefig(plt, outpath)
    @info "Saved convergence plot to $outpath"
    savefig(plt, joinpath(@__DIR__, "vacuum_2d_vs_3d_convergence.pdf"))
    return outpath
end

# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

function run_convergence(;
    # Mode parameters: n = 1, m in [-15, 15]
    mlow::Int=-15,
    mhigh::Int=15,
    nlow::Int=1,
    npert::Int=1,
    # 2D reference resolution (npol only; nzeta = 1)
    mtheta_2D::Int=2048,
    # 3D sweep (stay <= 96 per user request)
    mtheta_values::Vector{Int}=[16, 24, 32, 48, 64, 96],
    # Cache controls for quick plot-only reruns
    use_cache::Bool=true,
    force_recompute::Bool=false
)
    mpert = mhigh - mlow + 1

    # Read TOML first so cache hits can skip equilibrium setup.
    toml_inputs = load_solovev_example_inputs()
    a_conformal = toml_inputs["Wall"]["a"]

    # Grid aspect ratio: hold mtheta/nzeta fixed at the example's mthvac/nzvac.
    mthvac = toml_inputs["ForceFreeStates"]["mthvac"]
    nzvac = toml_inputs["ForceFreeStates"]["nzvac"]
    aspect_mt_over_nz = mthvac / nzvac   # = 96/128 = 0.75 for this example
    nzeta_for(mtheta::Int) = max(8, Int(round(mtheta / aspect_mt_over_nz)))

    wall_configs = [
        ("nowall", WallShapeSettings(; shape="nowall")),
        ("conformal", WallShapeSettings(; shape="conformal", a=a_conformal,
            equal_arc_wall=toml_inputs["Wall"]["equal_arc_wall"]))
    ]

    @info @sprintf("Convergence test: n=%d, m=%d..%d (mpert=%d); aspect mtheta/nzeta=%.3f",
        nlow, mlow, mhigh, mpert, aspect_mt_over_nz)

    cache_dir = joinpath(@__DIR__, ".cache")
    mkpath(cache_dir)
    cache_tag = make_cache_tag(;
        mlow, mhigh, nlow, npert, mtheta_2D,
        aspect_mt_over_nz, a_conformal
    )
    cache_file = joinpath(cache_dir, "vacuum_2d_vs_3d_convergence_" * cache_tag * ".jld2")

    cache_data = init_cache_data(a_conformal)
    if use_cache && !force_recompute && isfile(cache_file)
        @info "Loading cached convergence data from $cache_file"
        loaded_cache = JLD2.load(cache_file, "cache_data")
        if cache_is_compatible(loaded_cache)
            cache_data = loaded_cache
            if cache_is_complete(cache_data, mtheta_values, nzeta_for)
                @info "Cache is complete for this run; regenerating plot only."
                wv_2D_ref, results = results_from_cache(cache_data, mtheta_values, nzeta_for)
                outpath = save_convergence_plot(results, cache_data["a_conformal"])
                return (; wv_2D_ref, results, outpath, cache_file)
            end
            @info "Cache is partial; only missing points will be computed."
        else
            @warn "Cache schema mismatch (old format). Rebuilding cache for this run."
        end
    end

    # --- Equilibrium setup ---
    equil, _ = load_solovev_equilibrium()
    psi = equil.config.psihigh

    # --- 2D reference ---
    println("\n=== 2D reference (mtheta=$mtheta_2D, nzeta=1) ===")
    wv_2D_ref = cache_data["wv_2D_ref"]
    for (wname, wset) in wall_configs
        if haskey(wv_2D_ref, wname)
            dt = get(cache_data["timings_2D"], wname, NaN)
            @printf("  %-10s : ||Wv||_F = %.6e  (cached; %.2f s)\n", wname, norm(wv_2D_ref[wname]), dt)
        else
            inputs_2D = make_vac_inputs(equil, psi;
                mtheta=mtheta_2D, nzeta=1,
                mpert=mpert, mlow=mlow, nlow=nlow, npert=npert)
            t0 = time()
            wv = run_wv(inputs_2D, wset)
            dt = time() - t0
            @printf("  %-10s : ||Wv||_F = %.6e  (%.2f s)\n", wname, norm(wv), dt)
            wv_2D_ref[wname] = wv
            cache_data["timings_2D"][wname] = dt
        end
    end

    # --- 3D sweep ---
    println("\n=== 3D sweep ===")
    results = Dict{String,Vector{NamedTuple{(:mtheta, :nzeta, :relerr, :t),Tuple{Int,Int,Float64,Float64}}}}()
    for (wname, _) in wall_configs
        results[wname] = NamedTuple{(:mtheta, :nzeta, :relerr, :t),Tuple{Int,Int,Float64,Float64}}[]
    end

    for mtheta in mtheta_values
        nzeta = nzeta_for(mtheta)
        pkey = cache_point_key(mtheta, nzeta)
        @printf("\n-- mtheta=%d, nzeta=%d (ratio=%.3f) --\n", mtheta, nzeta, mtheta/nzeta)
        for (wname, wset) in wall_configs
            if haskey(cache_data["wv_3D"][wname], pkey)
                wv = cache_data["wv_3D"][wname][pkey]
                dt = get(cache_data["timings_3D"][wname], pkey, NaN)
                cached_str = @sprintf("cached; %.2f s", dt)
            else
                inputs_3D = make_vac_inputs(equil, psi;
                    mtheta=mtheta, nzeta=nzeta,
                    mpert=mpert, mlow=mlow, nlow=nlow, npert=npert)
                t0 = time()
                wv = run_wv(inputs_3D, wset)
                dt = time() - t0
                cache_data["wv_3D"][wname][pkey] = wv
                cache_data["timings_3D"][wname][pkey] = dt
                cached_str = @sprintf("%.2f s", dt)
            end
            wv_ref = wv_2D_ref[wname]
            relerr = norm(wv .- wv_ref) / norm(wv_ref)
            @printf("  %-10s : relerr=%.3e  (%s)\n", wname, relerr, cached_str)
            push!(results[wname], (; mtheta, nzeta, relerr, t=dt))
        end
    end

    if use_cache
        JLD2.save(cache_file, "cache_data", cache_data)
        @info "Saved convergence cache to $cache_file"
    end

    outpath = save_convergence_plot(results, a_conformal)
    return (; wv_2D_ref, results, outpath, cache_file)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_convergence()
end
