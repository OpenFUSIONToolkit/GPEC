#!/usr/bin/env julia
#
# Sweep thread counts 1…NMAX with fresh Julia processes (thread count is fixed at startup).
#
# Usage (from JPEC repo root):
#   julia --project=. benchmarks/equilibrium_threads_sweep.jl [output.csv] [nmax]
#
# Optional env (passed to child `runone` processes):
#   GPEC_EQUIL_BENCH_TOML — path to gpec.toml (default: benchmarks/DIIID_ideal_example/gpec.toml)
#   GPEC_EQUIL_BENCH_SAMPLES — BenchmarkTools samples per -t N (default: 10)
#
# Then plot:
#   julia --project=. benchmarks/plot_equilibrium_threads.jl [output.csv]

function main()
    root = dirname(dirname(abspath(@__FILE__)))
    runone = joinpath(root, "benchmarks", "equilibrium_threads_runone.jl")
    isfile(runone) || error("missing $runone")

    outcsv = get(ARGS, 1, joinpath(root, "benchmarks", "equilibrium_threads_results.csv"))
    nmax = parse(Int, get(ARGS, 2, "20"))

    outcsv = abspath(outcsv)
    mkpath(dirname(outcsv))

    open(outcsv, "w") do io
        println(io, "nthreads,median_s,min_s,max_s,samples")
    end

    @info "Writing $outcsv (threads 1:$nmax); each line is @benchmark median (evals=1; samples from GPEC_EQUIL_BENCH_SAMPLES or 10)"
    jc = Base.julia_cmd()
    for n in 1:nmax
        cmd = `$jc -t $n --project=$root $runone $outcsv`
        @info "Running: $cmd"
        run(cmd)
    end
    @info "Done. Plot with: julia --project=. benchmarks/plot_equilibrium_threads.jl $outcsv"
end

main()
