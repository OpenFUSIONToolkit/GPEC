#!/usr/bin/env julia
# benchmark_ffs_thread_scaling.jl — ForceFreeStates integration runtime vs thread
# count for the three integration paths (standard EL, serial Riccati, parallel FM).
#
# Julia's thread count is fixed at process start, so this script orchestrates one
# worker subprocess per thread count (`julia -t N`). Each worker uses BenchmarkTools
# to measure `eulerlagrange_integration` — the ForceFreeStates stage whose cost the
# parallel-FM path parallelises — for the DIII-D-like example at n=1 and n=4
# (n=4 has more rational surfaces to cross).
#
# The standard-EL and serial-Riccati paths contain no `Threads.@threads`, so they
# are thread-independent: they are measured once (in the 1-thread worker) and drawn
# as horizontal reference bands. Only the parallel-FM path is swept across 1..N
# threads.
#
# Usage:
#   julia --project=. benchmarks/benchmark_ffs_thread_scaling.jl              # full sweep 1..8
#   julia --project=. benchmarks/benchmark_ffs_thread_scaling.jl 4            # sweep 1..4
#   julia --project=. benchmarks/benchmark_ffs_thread_scaling.jl --samples 8  # 8 BenchmarkTools samples
#   julia --project=. benchmarks/benchmark_ffs_thread_scaling.jl --plot-only  # re-plot cached results
#   julia --project=. benchmarks/benchmark_ffs_thread_scaling.jl --force      # ignore the cache
#
# Per-thread results cache to benchmarks/ffs_thread_scaling_results/threads_<N>.toml
# (resumable across machines; not committed). Figure written to
# benchmarks/figures/ffs_thread_scaling_DIIID.png.
#
# The actual measurement code lives in benchmark_ffs_thread_scaling_worker.jl; it is
# `include`d only inside the worker subprocess (it uses the `@benchmarkable` macro,
# which must not be parsed by the orchestrator process).
#
# BenchmarkTools is a benchmark-only tool and is intentionally NOT a GPEC dependency.
# Install it once in your global Julia environment before running this script:
#     julia -e 'using Pkg; Pkg.add("BenchmarkTools")'
# Worker subprocesses resolve it via the global-environment fallback on the load path.

const ROOT      = abspath(joinpath(@__DIR__, ".."))
const IS_WORKER = "--worker" in ARGS

using Pkg
Pkg.activate(ROOT)

using TOML, Printf

const EXAMPLE_DIR = joinpath(ROOT, "examples", "DIIID-like_ideal_example")
const RESULTS_DIR = joinpath(@__DIR__, "ffs_thread_scaling_results")
const FIG_DIR     = joinpath(@__DIR__, "figures")
const N_VALUES    = (1, 4)
const MAX_THREADS_DEFAULT = 8
const SAMPLES_DEFAULT     = 5

if !IS_WORKER
    using Plots
end

# ---------------------------------------------------------------------------
# Orchestrator — spawns one worker per thread count, then plots.
# ---------------------------------------------------------------------------

function spawn_worker(nth, outfile, samples)
    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    thisfile = abspath(@__FILE__)
    cmd = `$julia -t $nth --project=$ROOT $thisfile --worker $outfile --samples $samples`
    println("\n", "="^70)
    println("Spawning worker: $nth thread(s)")
    println("="^70)
    run(cmd)
end

"""Load every cached `threads_<N>.toml` keyed by thread count."""
function load_results()
    data = Dict{Int,Any}()
    isdir(RESULTS_DIR) || return data
    for f in readdir(RESULTS_DIR; join=true)
        endswith(f, ".toml") || continue
        d = TOML.parsefile(f)
        data[Int(d["threads"])] = d
    end
    return data
end

"""Extract `(threads, median_s, min_s, std_s, memory_MiB, allocs)` for one
`(n, mode)`, sorted by thread count. Paths absent at a thread count are skipped."""
function series(data, nkey, mode)
    xs = Int[]; med = Float64[]; mn = Float64[]; sd = Float64[]
    mem = Float64[]; al = Float64[]
    for t in sort(collect(keys(data)))
        res = get(data[t]["results"], nkey, nothing)
        res === nothing && continue
        r = get(res, mode, nothing)
        r === nothing && continue
        push!(xs, t)
        push!(med, r["median_s"])
        push!(mn, r["min_s"])
        push!(sd, r["std_s"])
        push!(mem, r["memory_bytes"] / 2^20)
        push!(al, Float64(r["allocs"]))
    end
    return xs, med, mn, sd, mem, al
end

function print_summary(data)
    println("\n", "="^94)
    println("ForceFreeStates integration runtime — DIII-D-like example")
    println("="^94)
    @printf("%-8s  %-4s  %-16s  %10s  %10s  %10s  %10s  %9s\n",
            "threads", "n", "path", "median[s]", "min[s]", "std[s]", "mem[MiB]", "speedup")
    println("-"^94)
    for n in N_VALUES
        nkey = "n$n"
        par_t1 = nothing
        for mode in ("standard", "riccati", "parallel")
            xs, med, mn, sd, mem, _ = series(data, nkey, mode)
            isempty(xs) && continue
            if mode == "parallel"
                j1 = findfirst(==(1), xs)
                par_t1 = j1 === nothing ? nothing : med[j1]
            end
            for (i, t) in enumerate(xs)
                sp = (mode == "parallel" && par_t1 !== nothing) ?
                     @sprintf("%.2fx", par_t1 / med[i]) : "--"
                @printf("%-8d  %-4d  %-16s  %10.3f  %10.3f  %10.3f  %10.1f  %9s\n",
                        t, n, mode, med[i], mn[i], sd[i], mem[i], sp)
            end
        end
        println("-"^94)
    end
end

function plot_results(max_threads)
    data = load_results()
    if isempty(data)
        @warn "no cached results in $RESULTS_DIR — nothing to plot"
        return nothing
    end
    mkpath(FIG_DIR)
    samples = maximum(Int(d["samples"]) for d in values(data))

    panels = Plots.Plot[]
    for what in (:runtime, :memory), n in N_VALUES
        nkey = "n$n"
        ylabel = what === :runtime ? "eulerlagrange_integration  [s]" : "allocated memory  [MiB]"
        p = plot(; title="DIII-D-like,  n=$n", xlabel="threads", ylabel=ylabel,
                 left_margin=12Plots.mm, bottom_margin=4Plots.mm, legend=:topright,
                 xticks=1:max_threads, xlims=(0.5, max_threads + 0.5))

        xs, med, _, sd, mem, _ = series(data, nkey, "parallel")
        if !isempty(xs)
            y = what === :runtime ? med : mem
            err = what === :runtime ? sd : zeros(length(y))
            plot!(p, xs, y; yerror=err, marker=:circle, ms=5, lw=2,
                  color=:steelblue, label="parallel FM")
            if what === :runtime
                # Ideal strong scaling anchored at the 1-thread parallel point.
                plot!(p, xs, y[1] .* xs[1] ./ xs; ls=:dot, lw=1, color=:gray,
                      label="ideal 1/N")
            end
        end

        # standard / serial-Riccati: thread-independent, drawn as flat bands.
        for (mode, col, lab) in (("standard", :tomato, "standard EL (serial)"),
                                 ("riccati", :forestgreen, "serial Riccati"))
            xs0, med0, _, sd0, mem0, _ = series(data, nkey, mode)
            isempty(xs0) && continue
            y0 = what === :runtime ? med0[1] : mem0[1]
            e0 = what === :runtime ? sd0[1] : 0.0
            xline = collect(1:max_threads)
            plot!(p, xline, fill(y0, length(xline)); ribbon=fill(e0, length(xline)),
                  ls=:dash, lw=2, color=col, fillalpha=0.15, label=lab)
        end
        push!(panels, p)
    end

    fig = plot(panels...; layout=(2, 2), size=(1200, 900),
               plot_title="ForceFreeStates thread scaling — DIII-D-like  ($samples BenchmarkTools samples)")
    out = abspath(joinpath(FIG_DIR, "ffs_thread_scaling_DIIID.png"))
    savefig(fig, out)
    print_summary(data)
    println("\nFigure saved: $out")
    return out
end

function parse_orchestrator_args()
    max_threads = MAX_THREADS_DEFAULT
    samples     = SAMPLES_DEFAULT
    force       = false
    plot_only   = false
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--samples"
            samples = parse(Int, ARGS[i + 1]); i += 2
        elseif a == "--max-threads"
            max_threads = parse(Int, ARGS[i + 1]); i += 2
        elseif a == "--force"
            force = true; i += 1
        elseif a == "--plot-only"
            plot_only = true; i += 1
        elseif tryparse(Int, a) !== nothing
            max_threads = parse(Int, a); i += 1
        else
            @warn "ignoring unrecognized argument: $a"
            i += 1
        end
    end
    return max_threads, samples, force, plot_only
end

function run_orchestrator()
    max_threads, samples, force, plot_only = parse_orchestrator_args()
    mkpath(RESULTS_DIR)
    if !plot_only
        # BenchmarkTools is not a GPEC dependency — fail fast if the workers won't
        # be able to resolve it, rather than after spawning the first subprocess.
        if Base.find_package("BenchmarkTools") === nothing
            error("BenchmarkTools is required by the benchmark workers but is not " *
                  "installed. Add it to your global environment:\n" *
                  "    julia -e 'using Pkg; Pkg.add(\"BenchmarkTools\")'")
        end
        for nth in 1:max_threads
            outfile = joinpath(RESULTS_DIR, "threads_$nth.toml")
            if isfile(outfile) && !force
                @info "cached: threads_$nth.toml (use --force to rerun)"
                continue
            end
            spawn_worker(nth, outfile, samples)
        end
    end
    plot_results(max_threads)
end

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if IS_WORKER
    include(joinpath(@__DIR__, "benchmark_ffs_thread_scaling_worker.jl"))
    wi = findfirst(==("--worker"), ARGS)
    outfile = ARGS[wi + 1]
    si = findfirst(==("--samples"), ARGS)
    samples = si === nothing ? SAMPLES_DEFAULT : parse(Int, ARGS[si + 1])
    run_worker(outfile, samples)
else
    run_orchestrator()
end
