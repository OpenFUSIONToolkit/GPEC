#!/usr/bin/env julia
#
# Sweep GPEC `main` over thread counts 1…NMAX. For each N, runs exactly 2 times in a
# fresh Julia process and keeps only run #2 (run #1 is warmup/JIT).
#
# Parses stage timings from @info lines:
#   Equilibrium construction | Force-Free States | Perturbed Equilibrium | full GPEC
#
# Usage (from JPEC repo root):
#   julia --project=. benchmarks/gpec_thread_sweep.jl [case_dir] [nthread_max] [n_runs]
#
# Defaults:
#   case_dir      = examples/DIIID-like_ideal_example
#   nthread_max   = 20
#   n_runs        = 2   (fixed: run #1 discarded, keep run #2 only)
#
# Optional env:
#   GPEC_THREAD_SWEEP_CSV — if set, also write this CSV path with averages per N.

using Printf
using Statistics

const ROOT = dirname(dirname(abspath(@__FILE__)))
const RUNNER = joinpath(ROOT, "benchmarks", "gpec_thread_sweep_runner.jl")

function parse_timings(log::AbstractString)
    rx_equil = r"Equilibrium construction completed in ([0-9.]+) s"
    rx_ffs = r"Force-Free States completed in ([0-9.]+) s"
    rx_pe = r"Perturbed Equilibrium completed in ([0-9.]+) s"
    rx_tot = r"GPEC completed successfully in ([0-9.]+) s"
    function grab(rx)
        m = match(rx, log)
        m === nothing ? nothing : parse(Float64, m.captures[1])
    end
    (
        equil = grab(rx_equil),
        force_free = grab(rx_ffs),
        perturbed = grab(rx_pe),
        total = grab(rx_tot),
    )
end

function run_once(julia_exe, nthreads::Int, case_dir::AbstractString)::NamedTuple
    isfile(RUNNER) || error("missing runner: $RUNNER")
    cmd = `$julia_exe -t $nthreads --project=$ROOT $RUNNER $case_dir`
    io = IOBuffer()
    try
        run(pipeline(cmd; stdout = io, stderr = io))
    catch e
        @error "Child process failed" exception = e cmd
        rethrow()
    end
    text = String(take!(io))
    parse_timings(text)
end

function sweep(;
    case_dir::AbstractString,
    nthread_max::Int,
    n_runs::Int,
    drop_first::Bool,
)
    n_runs == 2 || error("This sweep is configured for exactly 2 runs per thread count")
    drop_first || error("This sweep is configured to discard run #1 and keep run #2")
    julia_exe = Base.julia_cmd()
    results = []

    @info "GPEC thread sweep" case_dir nthread_max n_runs drop_first ROOT

    for nt in 1:nthread_max
        samples = []
        for r in 1:n_runs
            @info "threads=$nt  run $r/$n_runs"
            push!(samples, run_once(julia_exe, nt, case_dir))
        end
        use = samples[2:2]

        function avg(field::Symbol)
            xs = Float64[]
            for s in use
                v = getproperty(s, field)
                v === nothing || push!(xs, v)
            end
            isempty(xs) ? NaN : mean(xs)
        end

        push!(
            results,
            (
                nthreads = nt,
                equil_s = avg(:equil),
                ffs_s = avg(:force_free),
                pe_s = avg(:perturbed),
                total_s = avg(:total),
            ),
        )
    end
    return results
end

function print_table(results)
    wn = 4
    we = 12
    wf = 14
    wp = 14
    wt = 12
    sep = repeat("-", wn + we + wf + wp + wt + 9)
    println()
    println(sep)
    @printf("| %*s | %*s | %*s | %*s | %*s |\n", wn, "Nt", we, "equil", wf, "force-free", wp, "perturbed", wt, "total")
    println(sep)
    for row in results
        @printf(
            "| %*d | %*.3f | %*.3f | %*.3f | %*.3f |\n",
            wn,
            row.nthreads,
            we,
            row.equil_s,
            wf,
            row.ffs_s,
            wp,
            row.pe_s,
            wt,
            row.total_s,
        )
    end
    println(sep)
    println()
    println("Times are seconds from run #2 only (run #1 per Nt discarded as warmup).")
    println()
end

function write_csv(path::AbstractString, results)
    open(path, "w") do io
        println(io, "nthreads,equil_mean_s,force_free_mean_s,perturbed_mean_s,total_mean_s")
        for row in results
            println(
                io,
                row.nthreads,
                ",",
                row.equil_s,
                ",",
                row.ffs_s,
                ",",
                row.pe_s,
                ",",
                row.total_s,
            )
        end
    end
    @info "Wrote CSV" path
end

function main()
    case_rel = get(ARGS, 1, joinpath("examples", "DIIID-like_ideal_example"))
    case_dir = abspath(joinpath(ROOT, case_rel))
    isdir(case_dir) || error("case directory not found: $case_dir")
    isfile(joinpath(case_dir, "gpec.toml")) || error("missing gpec.toml in $case_dir")

    nthread_max = parse(Int, get(ARGS, 2, "20"))
    n_runs = parse(Int, get(ARGS, 3, "2"))

    results = sweep(; case_dir, nthread_max, n_runs, drop_first = true)
    print_table(results)

    csv_env = get(ENV, "GPEC_THREAD_SWEEP_CSV", "")
    if !isempty(csv_env)
        write_csv(abspath(csv_env), results)
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
