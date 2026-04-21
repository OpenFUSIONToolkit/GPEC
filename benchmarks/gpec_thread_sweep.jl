#!/usr/bin/env julia
#
# Sweep GPEC `main` over thread counts 1…NMAX.  For each N, spawns ONE fresh Julia
# process and runs GPEC n_runs times inside it.  Run #1 is JIT warmup and is
# discarded; the remaining runs are averaged.
#
# Paying startup cost once per thread count (not once per run) keeps total wall
# time manageable even on slow-startup systems (WSL, etc.).
#
# Usage (from JPEC repo root):
#   julia --project=. benchmarks/gpec_thread_sweep.jl [case_dir] [nthread_max] [n_runs]
#
# Defaults:
#   case_dir      = examples/DIIID-like_ideal_example
#   nthread_max   = 8
#   n_runs        = 3   (run #1 = JIT warmup; runs 2..n_runs averaged)
#
# Output:
#   - Table with equil / ffs / perturb / total times, speedup, and efficiency
#   - CSV written to benchmarks/gpec_thread_sweep_results.csv (or GPEC_THREAD_SWEEP_CSV env)

using Printf
using Statistics

const ROOT = dirname(dirname(abspath(@__FILE__)))
const RUNNER = joinpath(ROOT, "benchmarks", "gpec_thread_sweep_runner.jl")

function parse_timing_line(line::AbstractString)
    m_eq  = match(r"equil=([0-9.NaN]+)", line)
    m_ffs = match(r"ffs=([0-9.NaN]+)", line)
    m_pe  = match(r"pe=([0-9.NaN]+)", line)
    m_tot = match(r"total=([0-9.NaN]+)", line)
    parse_f(m) = (m === nothing) ? NaN : parse(Float64, m.captures[1])
    (equil=parse_f(m_eq), force_free=parse_f(m_ffs), perturbed=parse_f(m_pe), total=parse_f(m_tot))
end

function run_all(julia_exe, nthreads::Int, case_dir::AbstractString, n_runs::Int)
    isfile(RUNNER) || error("missing runner: $RUNNER")
    cmd = `$julia_exe -t $nthreads --project=$ROOT $RUNNER $case_dir $n_runs`
    io = IOBuffer()
    try
        run(pipeline(cmd; stdout=io, stderr=io))
    catch e
        @error "Child process failed" exception=e cmd
        rethrow()
    end
    text = String(take!(io))
    samples = NamedTuple[]
    for line in split(text, '\n')
        startswith(strip(line), "GPEC_TIMING") && push!(samples, parse_timing_line(line))
    end
    return samples
end

function sweep(; case_dir::AbstractString, nthread_max::Int, n_runs::Int)
    n_runs >= 2 || error("n_runs must be ≥ 2 (run #1 is discarded as JIT warmup)")
    julia_exe = Base.julia_cmd()
    results = []

    @info "GPEC thread sweep" case_dir nthread_max n_runs ROOT

    for nt in 1:nthread_max
        @info "threads=$nt — spawning subprocess for $n_runs runs (run #1 = JIT warmup)"
        samples = run_all(julia_exe, nt, case_dir, n_runs)

        if isempty(samples)
            @warn "No GPEC_TIMING lines found for threads=$nt — check runner output"
        end
        use = length(samples) >= 2 ? samples[2:end] : samples

        function avg(field::Symbol)
            xs = [getproperty(s, field) for s in use]
            filter!(!isnan, xs)
            isempty(xs) ? NaN : mean(xs)
        end

        push!(results, (
            nthreads  = nt,
            equil_s   = avg(:equil),
            ffs_s     = avg(:force_free),
            pe_s      = avg(:perturbed),
            total_s   = avg(:total),
            n_samples = length(use),
        ))
    end
    return results
end

function print_table(results)
    t1 = results[1].total_s

    wn = 4; we = 10; wf = 12; wp = 12; wt = 10; ws = 8; weff = 8
    sep = repeat("-", wn + we + wf + wp + wt + ws + weff + 16)

    println()
    println(sep)
    @printf("| %*s | %*s | %*s | %*s | %*s | %*s | %*s |\n",
        wn, "Nt", we, "equil (s)", wf, "ffs (s)", wp, "perturb (s)",
        wt, "total (s)", ws, "speedup", weff, "effic.")
    println(sep)
    for row in results
        speedup    = isnan(t1) ? NaN : t1 / row.total_s
        efficiency = speedup / row.nthreads
        @printf("| %*d | %*.3f | %*.3f | %*.3f | %*.3f | %*.2fx | %*.0f%% |\n",
            wn, row.nthreads, we, row.equil_s, wf, row.ffs_s, wp, row.pe_s,
            wt, row.total_s, ws, speedup, weff, efficiency * 100)
    end
    println(sep)
    println()
    n_kept = results[1].n_samples
    println("Averaged over runs 2–$(n_kept + 1) per thread count (run #1 = JIT warmup, discarded).")
    println("Speedup = total(1 thread) / total(N threads).  Efficiency = speedup / N.")
    println()
end

function write_csv(path::AbstractString, results)
    open(path, "w") do io
        println(io, "nthreads,equil_mean_s,force_free_mean_s,perturbed_mean_s,total_mean_s,n_samples")
        for row in results
            @printf(io, "%d,%.6f,%.6f,%.6f,%.6f,%d\n",
                row.nthreads, row.equil_s, row.ffs_s, row.pe_s, row.total_s, row.n_samples)
        end
    end
    @info "Wrote CSV → $path"
end

function main()
    case_rel = get(ARGS, 1, joinpath("examples", "DIIID-like_ideal_example"))
    case_dir = abspath(joinpath(ROOT, case_rel))
    isdir(case_dir)                            || error("case directory not found: $case_dir")
    isfile(joinpath(case_dir, "gpec.toml"))    || error("missing gpec.toml in $case_dir")

    nthread_max = parse(Int, get(ARGS, 2, "8"))
    n_runs      = parse(Int, get(ARGS, 3, "3"))

    results  = sweep(; case_dir, nthread_max, n_runs)
    print_table(results)

    csv_path = let env = get(ENV, "GPEC_THREAD_SWEEP_CSV", "")
        isempty(env) ? joinpath(ROOT, "benchmarks", "gpec_thread_sweep_results.csv") : abspath(env)
    end
    write_csv(csv_path, results)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
