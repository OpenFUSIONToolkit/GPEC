#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using LinearAlgebra
using Printf
using TOML
using Statistics
using BenchmarkTools
using Plots
using LaTeXStrings
using JPEC

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")
const EXAMPLE_TOML = joinpath(EXAMPLE_DIR, "jpec.toml")
# `mscvac` reads `vac.in` from this directory; keep it consistent with `EXAMPLE_DIR` physics.
const BENCHMARK_DIR = joinpath(@__DIR__, "DIIID_ideal_example")
const CACHE_DIR = joinpath(@__DIR__, ".cache")
const CACHE_FILE = joinpath(CACHE_DIR, "vacuum_fortran_vs_julia_runtime_cache.toml")

function load_runtime_cache(path::String=CACHE_FILE)
    if !isfile(path)
        return Dict{String,Any}("entries" => Dict{String,Any}())
    end
    data = TOML.parsefile(path)
    haskey(data, "entries") || (data["entries"] = Dict{String,Any}())
    return data
end

function save_runtime_cache(cache::Dict{String,Any}, path::String=CACHE_FILE)
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, cache)
    end
end

function runtime_cache_key(
    impl::String,
    n::Int,
    mtheta::Int;
    mlow::Int,
    mhigh::Int,
    complex_flag::Bool,
    kernelsign::Float64,
    wall_flag::Bool,
    farwall_flag::Bool,
    ahg_file::Union{Nothing,String},
    seconds::Float64
)
    ahg_str = something(ahg_file, "none")
    return join(
        [
            "impl=$(impl)",
            "n=$(n)",
            "mtheta=$(mtheta)",
            "mlow=$(mlow)",
            "mhigh=$(mhigh)",
            "complex=$(complex_flag)",
            "kernelsign=$(kernelsign)",
            "wall=$(wall_flag)",
            "farwall=$(farwall_flag)",
            "ahg=$(ahg_str)",
            "seconds=$(seconds)"
        ], "|")
end

"""
Run the Fortran vacuum response matrix calculation.
"""
function run_fortran_wv!(
    scratch_wv::Matrix{ComplexF64},
    mlow::Int,
    mhigh::Int,
    mtheta_vac::Int,
    n::Int,
    qa::Float64,
    r::Vector{Float64},
    z::Vector{Float64},
    delta::Vector{Float64};
    complex_flag::Bool=true,
    kernelsign::Float64=-1.0,
    wall_flag::Bool=false,
    farwall_flag::Bool=true,
    ahg_file::Union{Nothing,String}=nothing
)
    mpert = mhigh - mlow + 1
    JPEC.Vacuum.set_surface_params(
        mtheta_vac,
        mlow,
        mhigh,
        n,
        qa,
        r,
        z,
        delta
    )
    grri = zeros(Float64, 2 * (mtheta_vac + 5), 2 * mpert)
    xzpts = zeros(Float64, mtheta_vac + 5, 4)
    try
        JPEC.Vacuum.mscvac(
            scratch_wv,
            mpert,
            mtheta_vac,
            mtheta_vac,
            complex_flag,
            kernelsign,
            wall_flag,
            farwall_flag,
            grri,
            xzpts,
            ahg_file,
            BENCHMARK_DIR
        )
    finally
        JPEC.Vacuum.unset_surface_params()
    end
    return scratch_wv
end

"""
Run the Julia vacuum response matrix calculation.
"""
function run_julia_wv(vac_inputs, wall_settings; n::Int, mtheta::Int)
    mpert = vac_inputs.mhigh - vac_inputs.mlow + 1
    vi = JPEC.Vacuum.VacuumInput(
        vac_inputs.equil,
        vac_inputs.psi,
        mtheta,
        mpert,
        vac_inputs.mlow,
        n;
        force_wv_symmetry=vac_inputs.force_wv_symmetry
    )
    wv, _, _, _ = JPEC.Vacuum.compute_vacuum_response(vi, wall_settings)
    return Matrix(wv)
end

benchmark_seconds(times_ns::Vector{Float64}) = times_ns ./ 1e9

"""
Benchmark runtime using only `@benchmark`.

Runs with a time budget and enforces a minimum sample count for stable std estimates.
"""
function benchmark_runtime_stats(f::Function; seconds::Float64=1.0, min_samples::Int=5)
    local_f = f
    b = @benchmark $local_f() seconds=seconds samples=min_samples evals=1
    ts = benchmark_seconds(Float64.(b.times))
    t_mean = mean(ts)
    t_std = length(ts) >= 2 ? std(ts) : 0.0
    return t_mean, t_std
end

"""
Run one benchmark point for a given (n, mtheta_vac).
"""
function run_point!(
    scratch_wv::Matrix{ComplexF64},
    mlow::Int,
    mhigh::Int,
    n::Int,
    mtheta_vac::Int,
    vac_inputs,
    wall_settings,
    complex_flag::Bool,
    kernelsign::Float64,
    wall_flag::Bool,
    farwall_flag::Bool,
    ahg_file::Union{Nothing,String};
    seconds::Float64=1.0,
    min_samples::Int=5,
    cache_entries::Union{Nothing,Dict{String,Any}}=nothing
)
    mpert = mhigh - mlow + 1
    vi_fortran = JPEC.Vacuum.VacuumInput(
        vac_inputs.equil,
        vac_inputs.psi,
        mtheta_vac,
        mpert,
        mlow,
        n;
        force_wv_symmetry=vac_inputs.force_wv_symmetry
    )
    qa = vac_inputs.qa
    r_dcon, z_dcon, delta_dcon = JPEC.Vacuum.dcon_surface_arrays_for_fortran(vi_fortran, qa)
    fortran_call =
        () -> begin
            fill!(scratch_wv, 0.0 + 0.0im)
            run_fortran_wv!(
                scratch_wv,
                mlow,
                mhigh,
                mtheta_vac,
                n,
                qa,
                r_dcon,
                z_dcon,
                delta_dcon;
                complex_flag=complex_flag,
                kernelsign=kernelsign,
                wall_flag=wall_flag,
                farwall_flag=farwall_flag,
                ahg_file=ahg_file
            )
        end

    julia_call = () -> run_julia_wv(vac_inputs, wall_settings; n=n, mtheta=mtheta_vac)
    fortran_call() # warmup
    julia_call()   # warmup

    fortran_key = runtime_cache_key(
        "fortran", n, mtheta_vac;
        mlow=mlow, mhigh=mhigh, complex_flag=complex_flag, kernelsign=kernelsign,
        wall_flag=wall_flag, farwall_flag=farwall_flag, ahg_file=ahg_file, seconds=seconds
    )
    julia_key = runtime_cache_key(
        "julia", n, mtheta_vac;
        mlow=mlow, mhigh=mhigh, complex_flag=complex_flag, kernelsign=kernelsign,
        wall_flag=wall_flag, farwall_flag=farwall_flag, ahg_file=ahg_file, seconds=seconds
    )

    t_fortran_mean::Float64 = NaN
    t_fortran_std::Float64 = NaN
    if cache_entries !== nothing && haskey(cache_entries, fortran_key)
        entry = cache_entries[fortran_key]
        t_fortran_mean = Float64(entry["mean"])
        t_fortran_std = Float64(entry["std"])
        @printf("  [cache] fortran n=%d mtheta=%d\n", n, mtheta_vac)
    else
        t_fortran_mean, t_fortran_std = benchmark_runtime_stats(fortran_call; seconds=seconds, min_samples=min_samples)
        if cache_entries !== nothing
            cache_entries[fortran_key] = Dict("mean" => t_fortran_mean, "std" => t_fortran_std)
        end
    end

    t_julia_mean::Float64 = NaN
    t_julia_std::Float64 = NaN
    if cache_entries !== nothing && haskey(cache_entries, julia_key)
        entry = cache_entries[julia_key]
        t_julia_mean = Float64(entry["mean"])
        t_julia_std = Float64(entry["std"])
        @printf("  [cache] julia   n=%d mtheta=%d\n", n, mtheta_vac)
    else
        t_julia_mean, t_julia_std = benchmark_runtime_stats(julia_call; seconds=seconds, min_samples=min_samples)
        if cache_entries !== nothing
            cache_entries[julia_key] = Dict("mean" => t_julia_mean, "std" => t_julia_std)
        end
    end

    return (; t_fortran_mean, t_fortran_std, t_julia_mean, t_julia_std)
end

function load_benchmark_setup(; mlow::Int=-15, mhigh::Int=15, force_wv_symmetry::Bool=true)
    isfile(EXAMPLE_TOML) || error("Missing benchmark config: $(EXAMPLE_TOML)")
    inputs = TOML.parsefile(EXAMPLE_TOML)
    eq_config = JPEC.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], EXAMPLE_DIR)
    equil = JPEC.Equilibrium.setup_equilibrium(eq_config)
    wall_settings = JPEC.Vacuum.WallShapeSettings(; (Symbol(k) => v for (k, v) in inputs["Wall"])...)
    psi = equil.config.psihigh
    qa = equil.params.qa
    mpert = mhigh - mlow + 1
    vac_inputs = (;
        equil,
        psi,
        qa,
        mlow,
        mhigh,
        mpert,
        force_wv_symmetry
    )
    return vac_inputs, wall_settings
end

"""
Create poster-quality runtime plot with error bars.
"""
function save_runtime_plot(
    xs::AbstractVector{<:Real},
    fortran_times::AbstractVector{<:Real},
    fortran_errs::AbstractVector{<:Real},
    julia_times::AbstractVector{<:Real},
    julia_errs::AbstractVector{<:Real};
    xlabel::Union{String,LaTeXString},
    title::String,
    xscale=:identity,
    yscale=:identity,
    outname::String
)
    finite_mask = isfinite.(fortran_times) .& isfinite.(fortran_errs) .&
                  isfinite.(julia_times) .& isfinite.(julia_errs)
    xs_plot = xs[finite_mask]
    fortran_plot = fortran_times[finite_mask]
    fortran_err_plot = fortran_errs[finite_mask]
    julia_plot = julia_times[finite_mask]
    julia_err_plot = julia_errs[finite_mask]
    isempty(xs_plot) && error("All points are non-finite; cannot render plot $(outname).")

    plt = plot(
        xs_plot,
        fortran_plot;
        lw=4,
        marker=:circle,
        ms=7,
        color=:red,
        linestyle=:solid,
        xscale=xscale,
        yscale=yscale,
        xlabel=xlabel,
        ylabel=L"\mathbf{Runtime\ (s)}",
        title=title,
        # legend=:inside,
        size=(550, 400),
        dpi=300,
        legendfontsize=14,
        guidefontsize=16,
        tickfontsize=14,
        titlefontsize=16,
        minorgrid=true,
        foreground_color_legend=:black,
        background_color_legend=RGBA(1, 1, 1, 0.85),
        yerror=fortran_err_plot,
        label=L"\mathbf{Fortran}",
        show=false
    );
    plot!(
        xs_plot,
        julia_plot;
        lw=4,
        marker=:circle,
        ms=7,
        color=:green,
        linestyle=:dot,
        yerror=julia_err_plot,
        label=L"\mathbf{Julia}",
        show=false
    );

    outpath = joinpath(@__DIR__, outname)
    savefig(plt, outpath * ".png")
    savefig(plt, outpath * ".pdf")
    @info "Saved plot to $outpath"
    return outpath
end

function run_benchmark(;
    n_values::Vector{Int}=collect(1:2:15),
    mtheta_values::Vector{Int}=[2^k for k in 6:11],
    fixed_n_for_mtheta_scan::Int=1,
    mtheta_ref::Int=512,
    mlow::Int=-15,
    mhigh::Int=15,
    force_wv_symmetry::Bool=true,
    complex_flag::Bool=true,
    kernelsign::Float64=1.0,
    wall_flag::Bool=false,
    farwall_flag::Bool=true,
    ahg_file::Union{Nothing,String}=nothing,
    seconds::Float64=20.0,
    min_samples::Int=10,
    use_cache::Bool=true
)
    cache = use_cache ? load_runtime_cache() : Dict{String,Any}("entries" => Dict{String,Any}())
    cache_entries = cache["entries"]

    vac_inputs, wall_settings = load_benchmark_setup(;
        mlow=mlow,
        mhigh=mhigh,
        force_wv_symmetry=force_wv_symmetry
    )
    mpert = vac_inputs.mpert

    scratch_wv = zeros(ComplexF64, mpert, mpert)

    println("\n" * "="^72)
    println("Vacuum runtime benchmark: Fortran mscvac vs Julia compute_vacuum_response")
    println("="^72)
    @printf("n sweep values: %s\n", repr(n_values))
    @printf("mtheta sweep values: %s\n", repr(mtheta_values))
    @printf("fixed n for mtheta sweep: %d\n\n", fixed_n_for_mtheta_scan)

    # -----------------------------
    # Sweep 1: varying n
    # -----------------------------
    fortran_times_n = Float64[]
    fortran_errs_n = Float64[]
    julia_times_n = Float64[]
    julia_errs_n = Float64[]
    println("=== Sweep 1: varying toroidal mode number n ===")
    @printf("Using fixed mtheta = %d\n", mtheta_ref)
    for n in n_values
        r = run_point!(
            scratch_wv,
            mlow,
            mhigh,
            n,
            mtheta_ref,
            vac_inputs,
            wall_settings,
            complex_flag,
            kernelsign,
            wall_flag,
            farwall_flag,
            ahg_file;
            seconds=seconds,
            min_samples=min_samples,
            cache_entries=cache_entries
        )
        push!(fortran_times_n, r.t_fortran_mean)
        push!(fortran_errs_n, r.t_fortran_std)
        push!(julia_times_n, r.t_julia_mean)
        push!(julia_errs_n, r.t_julia_std)
        @printf(
            "n=%2d  fortran=%.4fs ± %.4fs  julia=%.4fs ± %.4fs\n",
            n, r.t_fortran_mean, r.t_fortran_std, r.t_julia_mean, r.t_julia_std
        )
    end

    out_n = save_runtime_plot(
        n_values,
        fortran_times_n,
        fortran_errs_n,
        julia_times_n,
        julia_errs_n;
        xlabel=L"\mathbf{n\ (toroidal\ mode\ number)}",
        title="", #"Runtime vs mode number (fixed N_θ = $(mtheta_ref))",
        xscale=:identity,
        yscale=:identity,
        outname="vacuum_wv_fortran_vs_julia_n_scan"
    );

    # -----------------------------
    # Sweep 2: varying mtheta
    # -----------------------------
    fortran_times_mtheta = Float64[]
    fortran_errs_mtheta = Float64[]
    julia_times_mtheta = Float64[]
    julia_errs_mtheta = Float64[]

    println("\n=== Sweep 2: varying mtheta ===")
    @printf("Using fixed n = %d\n", fixed_n_for_mtheta_scan)
    for mtheta_vac in mtheta_values
        r = run_point!(
            scratch_wv,
            mlow,
            mhigh,
            fixed_n_for_mtheta_scan,
            mtheta_vac,
            vac_inputs,
            wall_settings,
            complex_flag,
            kernelsign,
            wall_flag,
            farwall_flag,
            ahg_file;
            seconds=seconds,
            min_samples=min_samples,
            cache_entries=cache_entries
        )
        push!(fortran_times_mtheta, r.t_fortran_mean)
        push!(fortran_errs_mtheta, r.t_fortran_std)
        push!(julia_times_mtheta, r.t_julia_mean)
        push!(julia_errs_mtheta, r.t_julia_std)
        @printf(
            "mtheta=%4d  fortran=%.4fs ± %.4fs  julia=%.4fs ± %.4fs\n",
            mtheta_vac, r.t_fortran_mean, r.t_fortran_std, r.t_julia_mean, r.t_julia_std
        )
    end

    out_mtheta = save_runtime_plot(
        mtheta_values,
        fortran_times_mtheta,
        fortran_errs_mtheta,
        julia_times_mtheta,
        julia_errs_mtheta;
        xlabel=L"\mathbf{N_\theta}\ (\mathbf{number\ of\ grid\ points})",
        title="", #"Runtime vs grid resolution (fixed n = $(fixed_n_for_mtheta_scan))",
        xscale=:log10,
        yscale=:log10,
        outname="vacuum_wv_fortran_vs_julia_mtheta_scan"
    );

    if use_cache
        save_runtime_cache(cache)
    end
    return (
        n_scan=(n=n_values, t_fortran=fortran_times_n, t_fortran_std=fortran_errs_n, t_julia=julia_times_n, t_julia_std=julia_errs_n, outpath=out_n),
        mtheta_scan=(
            mtheta=mtheta_values,
            t_fortran=fortran_times_mtheta,
            t_fortran_std=fortran_errs_mtheta,
            t_julia=julia_times_mtheta,
            t_julia_std=julia_errs_mtheta,
            outpath=out_mtheta
        )
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
