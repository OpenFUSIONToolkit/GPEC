#!/usr/bin/env julia
#
# Scan delta_mlow = delta_mhigh for regression_solovev_ideal_example on two git
# branches and plot minimum total energy eigenvalue Re(min et) vs delta.
#
# Each branch uses one Julia subprocess that loads GPEC once and loops over δ.
#
# Usage:
#   julia --project=. benchmarks/benchmark_delta_m_scan.jl
#   julia --project=. benchmarks/benchmark_delta_m_scan.jl --quick
#   julia --project=. benchmarks/benchmark_delta_m_scan.jl --branch1 develop --branch2 jmh/vacuum_galerkin
#   julia --project=. benchmarks/benchmark_delta_m_scan.jl --deltas 0,4,8,12,16

using Dates
using HDF5
using Printf
using TOML

const PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))
const EXAMPLE_DIR = joinpath(PROJECT_ROOT, "examples", "DIIID-like_ideal_example_copy")
const OUTPUT_DIR = joinpath(@__DIR__, "delta_m_scan")
const GPEC_BASE = TOML.parsefile(joinpath(EXAMPLE_DIR, "gpec.toml"))

function prepare_run_dir(base::Dict, example_dir::AbstractString, delta::Int)
    raw = deepcopy(base)
    raw["ForceFreeStates"]["delta_mlow"] = delta
    raw["ForceFreeStates"]["delta_mhigh"] = delta
    raw["ForceFreeStates"]["verbose"] = false

    run_dir = mktempdir(; prefix="gpec_delta_m_$(delta)_")
    open(joinpath(run_dir, "gpec.toml"), "w") do io
        TOML.print(io, raw)
    end
    for f in readdir(example_dir)
        src = abspath(joinpath(example_dir, f))
        isfile(src) && f != "gpec.toml" && symlink(src, joinpath(run_dir, f))
    end
    return run_dir
end

function read_et_min(h5_path::String)
    h5open(h5_path, "r") do h5
        et = if haskey(h5, "vacuum/et")
            read(h5["vacuum/et"])
        elseif haskey(h5, "FreeBoundaryStability/XiNorm/eigenmode_energies")
            read(h5["FreeBoundaryStability/XiNorm/eigenmode_energies"])
        else
            keys_str = join(collect(keys(h5)), ", ")
            error("No eigenvalue dataset in $h5_path (top-level keys: $keys_str)")
        end
        return minimum(real.(et))
    end
end

function worker_scan(example_dir::AbstractString, base_toml::AbstractString, deltas, output_csv::AbstractString)
    base = TOML.parsefile(base_toml)
    et_min = Float64[]
    for (i, delta) in enumerate(deltas)
        @info "delta=$delta ($i/$(length(deltas)))"
        run_dir = prepare_run_dir(base, example_dir, delta)
        try
            GeneralizedPerturbedEquilibrium.main([run_dir])
            h5_path = joinpath(run_dir, "gpec.h5")
            isfile(h5_path) || error("Missing output: $h5_path")
            val = read_et_min(h5_path)
            push!(et_min, val)
            @printf("  Re(min et) = %+.6f\n", val)
        catch err
            @warn "Run failed" delta exception=(err, catch_backtrace())
            push!(et_min, NaN)
        finally
            rm(run_dir; force=true, recursive=true)
        end
    end
    open(output_csv, "w") do io
        println(io, "delta,et_min_real")
        for (delta, et) in zip(deltas, et_min)
            println(io, "$delta,$et")
        end
    end
end

function parse_options(args)
    opts = Dict(
        "branch1" => "develop",
        "branch2" => strip(readchomp(`git -C $PROJECT_ROOT branch --show-current`)),
        "deltas" => nothing,
        "quick" => false
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--quick"
            opts["quick"] = true
            i += 1
        elseif arg == "--branch1" && i < length(args)
            opts["branch1"] = String(args[i+1])
            i += 2
        elseif arg == "--branch2" && i < length(args)
            opts["branch2"] = String(args[i+1])
            i += 2
        elseif arg == "--deltas" && i < length(args)
            opts["deltas"] = [parse(Int, strip(s)) for s in split(args[i+1], ',')]
            i += 2
        else
            error("Unknown argument: $arg")
        end
    end
    if opts["deltas"] === nothing
        opts["deltas"] = opts["quick"] ? [0, 8, 16, 24] : collect(0:4:32)
    end
    return opts
end

function git_short_ref(ref)
    return readchomp(`git -C $PROJECT_ROOT rev-parse --short $ref`)
end

function checkout_ref(branch::AbstractString)
    @info "Checking out $branch"
    run(`git -C $PROJECT_ROOT checkout $branch`)
end

function read_worker_csv(path::AbstractString, deltas)
    rows = Dict{Int,Float64}()
    open(path, "r") do io
        readline(io)  # header
        for line in eachline(io)
            isempty(strip(line)) && continue
            parts = split(strip(line), ',')
            rows[parse(Int, parts[1])] = parse(Float64, parts[2])
        end
    end
    return [get(rows, delta, NaN) for delta in deltas]
end

function scan_branch(branch::AbstractString, deltas::AbstractVector{<:Integer})
    commit = git_short_ref("HEAD")
    @info "[$branch @ $commit] scanning $(length(deltas)) δ values in one Julia session"

    base_toml = tempname() * ".toml"
    output_csv = tempname() * ".csv"
    try
        open(base_toml, "w") do io
            TOML.print(io, GPEC_BASE)
        end
        delta_str = join(string.(deltas), ',')
        run(
            `julia --project=$PROJECT_ROOT $(@__FILE__) --worker $PROJECT_ROOT $EXAMPLE_DIR $base_toml $delta_str $output_csv`
        )
        et_min = read_worker_csv(output_csv, deltas)
        return (branch=branch, commit=commit, et_min=et_min)
    finally
        isfile(base_toml) && rm(base_toml; force=true)
        isfile(output_csv) && rm(output_csv; force=true)
    end
end

function save_results_csv(path::String, deltas, results)
    open(path, "w") do io
        println(io, "delta,branch,commit,et_min_real")
        for r in results
            for (delta, et) in zip(deltas, r.et_min)
                println(io, "$delta,$(r.branch),$(r.commit),$et")
            end
        end
    end
end

function plot_results(path::String, deltas, results)
    default(; lw=2, marker=:circle, markersize=5, legend=:best, grid=true)
    p = plot(;
        xlabel="δ (delta_mlow = delta_mhigh)",
        ylabel="Re(min et)",
        yscale=:log10,
        size=(700, 450),
        left_margin=12Plots.mm,
        bottom_margin=4Plots.mm
    )
    markers = [:circle, :rect, :utriangle, :diamond]
    for (i, r) in enumerate(results)
        et_log = [v > 0 ? v : NaN for v in r.et_min]
        plot!(
            p,
            deltas,
            et_log;
            marker=markers[mod1(i, length(markers))],
            label="$(r.branch) @ $(r.commit)"
        )
    end
    savefig(p, path)
    println("Saved plot: $path")
end

function main()
    opts = parse_options(ARGS)
    deltas = opts["deltas"]
    branch1 = opts["branch1"]
    branch2 = opts["branch2"]

    mkpath(OUTPUT_DIR)
    original_branch = strip(readchomp(`git -C $PROJECT_ROOT branch --show-current`))

    println("="^70)
    println("delta_m scan — regression_solovev_ideal_example")
    println("="^70)
    @printf("Example:  %s\n", EXAMPLE_DIR)
    @printf("Deltas:   %s\n", join(string.(deltas), ", "))
    @printf("Branches: %s vs %s\n", branch1, branch2)
    println()

    results = Dict{String,Any}()
    try
        checkout_ref(branch1)
        results["branch1"] = scan_branch(branch1, deltas)

        checkout_ref(branch2)
        results["branch2"] = scan_branch(branch2, deltas)
    finally
        @info "Restoring branch $original_branch"
        checkout_ref(original_branch)
    end

    scan_results = [results["branch1"], results["branch2"]]
    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    csv_path = joinpath(OUTPUT_DIR, "delta_m_scan_$(stamp).csv")
    png_path = joinpath(OUTPUT_DIR, "delta_m_scan_$(stamp).png")
    save_results_csv(csv_path, deltas, scan_results)
    plot_results(png_path, deltas, scan_results)

    println()
    println("="^70)
    @printf("%6s", "delta")
    for r in scan_results
        @printf("  %20s", "$(r.branch)")
    end
    println()
    println("-"^70)
    for (j, delta) in enumerate(deltas)
        @printf("%6d", delta)
        for r in scan_results
            val = r.et_min[j]
            if isnan(val)
                @printf("  %20s", "FAIL")
            else
                @printf("  %20.6f", val)
            end
        end
        println()
    end
    println("="^70)
    println("CSV:  $csv_path")
    println("Plot: $png_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 1 && ARGS[1] == "--worker"
        using Pkg
        Pkg.activate(ARGS[2])
        Pkg.instantiate()
        using GeneralizedPerturbedEquilibrium
        deltas = [parse(Int, strip(s)) for s in split(ARGS[5], ',')]
        worker_scan(ARGS[3], ARGS[4], deltas, ARGS[6])
    else
        using Plots
        main()
    end
end
