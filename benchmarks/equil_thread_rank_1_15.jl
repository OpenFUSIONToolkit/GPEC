#!/usr/bin/env julia
"""
Run `setup_equilibrium` in fresh subprocesses with `-t 1` … `-t 15`, collect
wall times, then print a table ranked **slowest → fastest** (highest runtime first).

Usage (from JPEC root):
    julia --project=. benchmarks/equil_thread_rank_1_15.jl
    julia --project=. benchmarks/equil_thread_rank_1_15.jl path/to/example_dir

Requires `benchmarks/equil_thread_worker.jl` (same directory).
"""
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Printf

function main()
    example_dir = get(ARGS, 1, joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example"))
    proj = joinpath(@__DIR__, "..")
    worker = joinpath(@__DIR__, "equil_thread_worker.jl")

    println("="^60)
    println("Thread sweep: 1–15 (subprocess per count, warmup+timed in worker)")
    println("Example: $example_dir")
    println("="^60)

    results = Tuple{Int,Float64}[]
    for n in 1:15
        cmd = `$(Base.julia_cmd()) -t $n --project=$proj $worker $example_dir`
        @printf("  Running -t %2d ... ", n)
        flush(stdout)
        out = read(cmd, String)
        lines = filter(!isempty, strip.(split(out, '\n')))
        # Last non-empty line should be the timing float (worker may print nothing else on stdout)
        t = parse(Float64, lines[end])
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
