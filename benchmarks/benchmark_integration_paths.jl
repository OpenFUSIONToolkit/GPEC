#!/usr/bin/env julia
"""
Benchmark the three integration paths (standard, riccati, parallel) on Solovev and DIIID examples.
Runs in a single Julia process to avoid measuring compilation overhead.
Produces accuracy and performance tables similar to PR #178.

Usage:
    julia --project=. -t4 benchmarks/benchmark_integration_paths.jl
"""

using GeneralizedPerturbedEquilibrium
using HDF5, Printf, TOML

const PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))

struct BenchResult
    example::String
    path::String
    et1::Float64
    nsteps::Int
    runtime::Float64
end

function run_one(example_dir::String, path_name::String; num_warm::Int=2)
    abs_dir = abspath(example_dir)
    gpec_toml = joinpath(abs_dir, "gpec.toml")

    # Read and modify config
    config = TOML.parsefile(gpec_toml)
    ffs = get(config, "ForceFreeStates", Dict{String,Any}())
    if path_name == "standard"
        ffs["use_riccati"] = false
        ffs["use_parallel"] = false
    elseif path_name == "riccati"
        ffs["use_riccati"] = true
        ffs["use_parallel"] = false
    elseif path_name == "parallel"
        ffs["use_riccati"] = false
        ffs["use_parallel"] = true
    end
    config["ForceFreeStates"] = ffs

    # Write modified config in-place, restore after
    original_toml = read(gpec_toml, String)

    try
        open(gpec_toml, "w") do f
            TOML.print(f, config)
        end

        # JIT warmup
        println("  [$path_name] JIT warmup...")
        GeneralizedPerturbedEquilibrium.main([abs_dir])

        # Timed runs
        runtimes = Float64[]
        for i in 1:num_warm
            println("  [$path_name] Warm run $i/$num_warm...")
            t0 = time()
            GeneralizedPerturbedEquilibrium.main([abs_dir])
            push!(runtimes, time() - t0)
            @printf("    %.2f s\n", runtimes[end])
        end

        # Read results
        gpec_h5 = joinpath(abs_dir, "gpec.h5")
        et1, nsteps = h5open(gpec_h5, "r") do h5
            et = read(h5["vacuum/et"])
            ns = read(h5["integration/nstep"])
            (real(et[1]), ns)
        end

        avg_t = sum(runtimes) / length(runtimes)
        return BenchResult(basename(example_dir), path_name, et1, nsteps, avg_t)
    finally
        write(gpec_toml, original_toml)
    end
end

function main()
    examples = [
        joinpath(PROJECT_ROOT, "examples", "Solovev_ideal_example"),
        joinpath(PROJECT_ROOT, "examples", "DIIID-like_ideal_example"),
    ]
    paths = ["standard", "riccati", "parallel"]

    results = BenchResult[]
    for ex in examples
        println("\n" * "="^60)
        println("Example: $(basename(ex))")
        println("="^60)
        for p in paths
            r = run_one(ex, p)
            push!(results, r)
            @printf("  → et[1]=%.5f  steps=%d  time=%.2fs\n", r.et1, r.nsteps, r.runtime)
        end
    end

    # Print Accuracy table
    println("\n\n## Accuracy\n")
    println("| Example | Path | et[1] | Error vs std |")
    println("|---------|------|-------|--------------|")
    for ex in unique(r.example for r in results)
        group = filter(r -> r.example == ex, results)
        std_et1 = group[1].et1
        N = 0
        toml_path = joinpath(PROJECT_ROOT, "examples", ex, "gpec.toml")
        if isfile(toml_path)
            cfg = TOML.parsefile(toml_path)
            ffs_cfg = get(cfg, "ForceFreeStates", Dict())
            mlow = get(ffs_cfg, "delta_mlow", 8)
            mhigh = get(ffs_cfg, "delta_mhigh", 8)
            N = mlow + mhigh
        end
        for r in group
            err_str = r.path == "standard" ? "—" : @sprintf("%.3f%%", 100*abs(r.et1 - std_et1)/abs(std_et1))
            short_ex = startswith(r.example, "Solovev") ? "Solovev N=$N" : "DIIID N=$N"
            @printf("| %s | %s | %.5f | %s |\n", short_ex, r.path, r.et1, err_str)
        end
    end

    # Print Performance table
    nthreads = Threads.nthreads()
    println("\n## Performance ($nthreads threads)\n")
    println("| Example | Path | Time | Speedup |")
    println("|---------|------|------|---------|")
    for ex in unique(r.example for r in results)
        group = filter(r -> r.example == ex, results)
        std_time = group[1].runtime
        N = 0
        toml_path = joinpath(PROJECT_ROOT, "examples", ex, "gpec.toml")
        if isfile(toml_path)
            cfg = TOML.parsefile(toml_path)
            ffs_cfg = get(cfg, "ForceFreeStates", Dict())
            mlow = get(ffs_cfg, "delta_mlow", 8)
            mhigh = get(ffs_cfg, "delta_mhigh", 8)
            N = mlow + mhigh
        end
        for r in group
            speedup = std_time / r.runtime
            short_ex = startswith(r.example, "Solovev") ? "Solovev N=$N" : "DIIID N=$N"
            speedup_str = r.path == "standard" ? "1.00×" : @sprintf("**%.2f×**", speedup)
            @printf("| %s | %s | %.2fs | %s |\n", short_ex, r.path, r.runtime, speedup_str)
        end
    end
end

main()
