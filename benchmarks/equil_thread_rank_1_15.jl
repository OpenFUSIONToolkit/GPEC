#!/usr/bin/env julia
#=
Run setup_equilibrium in fresh subprocesses with -t 1 … -t 15, collect wall times,
then print a table ranked slowest → fastest (highest runtime first).

Child processes use --startup-file=no and BLAS/OpenMP env vars set to 1 thread so
Julia's -t n scaling is not fought by nested linear-algebra threading.

Usage (from JPEC package root):
    julia --project=. benchmarks/equil_thread_rank_1_15.jl
    julia --project=. benchmarks/equil_thread_rank_1_15.jl path/to/example_dir

WSL (Ubuntu, etc.):
    cd /mnt/c/Users/<you>/PlasmaLab/JPEC   (or clone under ~/ for better I/O)
    julia --project=. benchmarks/equil_thread_rank_1_15.jl

Requires benchmarks/equil_thread_worker.jl (same directory).
=#
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Printf

function main()
    example_dir = abspath(get(ARGS, 1, joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")))
    proj = abspath(joinpath(@__DIR__, ".."))
    worker = abspath(joinpath(@__DIR__, "equil_thread_worker.jl"))
    isdir(example_dir) || error("Example directory not found: $example_dir")
    isfile(joinpath(example_dir, "gpec.toml")) || error("Missing gpec.toml in $example_dir")

    println("="^60)
    println("Thread sweep: 1–15 (subprocess per count, warmup+timed in worker)")
    println("Example: $example_dir")
    println("="^60)

    results = Tuple{Int,Float64}[]
    for n in 1:15
        cmd = addenv(
            `$(Base.julia_cmd()) --startup-file=no -t $n --project=$proj $worker $example_dir`,
            "OPENBLAS_NUM_THREADS" => "1",
            "MKL_NUM_THREADS" => "1",
            "OMP_NUM_THREADS" => "1",
            "VECLIB_MAXIMUM_THREADS" => "1",
            "NUMEXPR_NUM_THREADS" => "1",
        )
        @printf("  Running -t %2d ... ", n)
        flush(stdout)
        out = read(cmd, String)
        lines = filter(!isempty, strip.(split(out, '\n')))
        t = nothing
        for line in reverse(lines)
            v = tryparse(Float64, line)
            if v !== nothing
                t = v
                break
            end
        end
        t === nothing && error("Could not parse timing from worker stdout:\n$out")
        t = t::Float64
        push!(results, (n, t))
        @printf("%.4f s\n", t)
    end

    sorted = sort(results; by=x -> x[2], rev=true)

    println()
    println("="^60)
    println("Ranked by runtime (slowest → fastest, i.e. highest s first)")
    println("="^60)
    @printf("%-6s %-10s %s\n", "Rank", "Threads", "Seconds")
    println("-"^60)
    for (rank, (nt, t)) in enumerate(sorted)
        @printf("%-6d %-10d %.6f\n", rank, nt, t)
    end
    println("="^60)

    best = argmin(x -> x[2], results)
    @printf("\nFastest: %d threads (%.6f s)\n", best[1], best[2])
end

main()
