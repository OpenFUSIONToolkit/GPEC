#!/usr/bin/env julia
#
# Run once per Julia process (fixed thread count from -t N).
# Usage:
#   julia -t 4 --project=. benchmarks/equilibrium_threads_runone.jl <output.csv>
#
# Appends one CSV row: nthreads,median_s,min_s,max_s,samples

using BenchmarkTools
using Logging
using TOML
using GeneralizedPerturbedEquilibrium.Equilibrium as EQ

function load_prepared_direct_input(gpec_path::String)
    inputs = TOML.parsefile(gpec_path)
    base = dirname(abspath(gpec_path))
    cfg = EQ.EquilibriumConfig(inputs["Equilibrium"], base)
    raw = EQ.read_efit(cfg)
    psihigh_safe, adjusted = EQ.clamp_psihigh_to_separatrix(raw)
    if adjusted
        raw.config.psihigh = psihigh_safe
    end
    return raw
end

function main()
    length(ARGS) >= 1 || error("usage: julia -t N --project=. equilibrium_threads_runone.jl <output.csv>")
    outcsv = ARGS[1]

    bench_dir = dirname(abspath(@__FILE__))
    default_toml = joinpath(bench_dir, "DIIID_ideal_example", "gpec.toml")
    gpec_path = get(ENV, "GPEC_EQUIL_BENCH_TOML", default_toml)
    gpec_path = abspath(gpec_path)
    isfile(gpec_path) || error("gpec.toml not found: $gpec_path")

    nt = Threads.nthreads()
    toml_lit = gpec_path
    nsamples = parse(Int, get(ENV, "GPEC_EQUIL_BENCH_SAMPLES", "10"))

    # Fresh DirectRunInput each sample — equilibrium_solver mutates raw (direct_position!, etc.)
    b = with_logger(NullLogger()) do
        @benchmark EQ.equilibrium_solver(raw; benchmark = false) setup = (raw = load_prepared_direct_input($toml_lit)) evals = 1 samples = $nsamples
    end

    med = median(b.times) * 1e-9
    tmin = minimum(b.times) * 1e-9
    tmax = maximum(b.times) * 1e-9
    nsamp = length(b.times)

    open(outcsv, "a") do io
        println(io, "$nt,$med,$tmin,$tmax,$nsamp")
    end

    @info "threads=$nt  median=$(round(med; digits=4)) s  (min=$tmin, max=$tmax, samples=$nsamp)  → appended to $outcsv"
end

main()
