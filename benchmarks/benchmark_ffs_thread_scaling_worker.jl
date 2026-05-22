# benchmark_ffs_thread_scaling_worker.jl — measurement code for one thread count.
#
# `include`d by benchmark_ffs_thread_scaling.jl only inside a worker subprocess.
# Uses the BenchmarkTools `@benchmarkable` macro (which must not be parsed by the
# orchestrator process, where BenchmarkTools is not loaded). Relies on the constants
# EXAMPLE_DIR / N_VALUES / SAMPLES_DEFAULT defined by the including script.

using GeneralizedPerturbedEquilibrium
using BenchmarkTools  # benchmark-only tool; resolved from the global env, not a GPEC dep
using Statistics

const FFS = GeneralizedPerturbedEquilibrium.ForceFreeStates
const Eq  = GeneralizedPerturbedEquilibrium.Equilibrium
const Vac = GeneralizedPerturbedEquilibrium.Vacuum

"""Build the n-independent plasma equilibrium once per worker."""
function build_equilibrium(inputs)
    eq_config = Eq.EquilibriumConfig(inputs["Equilibrium"], EXAMPLE_DIR)
    return Eq.setup_equilibrium(eq_config)
end

"""ForceFreeStatesControl for toroidal mode `n`, integration path `mode`, and a
`parallel_threads` cap of `nth` (only the parallel-FM path reads it)."""
function make_ctrl(inputs, n, mode, nth)
    ffs = copy(inputs["ForceFreeStates"])
    ffs["verbose"]               = false
    ffs["write_outputs_to_HDF5"] = false
    ffs["nn_low"]                = n
    ffs["nn_high"]               = n
    ffs["use_parallel"]          = (mode === :parallel)
    ffs["use_riccati"]           = (mode === :riccati)
    ffs["parallel_threads"]      = max(nth, 1)
    return FFS.ForceFreeStatesControl(; (Symbol(k) => v for (k, v) in ffs)...)
end

"""Fresh integration state `(ctrl, ffit, intr)` for one benchmark sample. The
equilibrium is reused (read-only); everything `eulerlagrange_integration` mutates
is rebuilt here so each sample starts from a clean slate."""
function make_state(inputs, n, mode, nth, equil)
    ctrl = make_ctrl(inputs, n, mode, nth)
    intr = FFS.ForceFreeStatesInternal(; dir_path=EXAMPLE_DIR)
    intr.wall_settings = haskey(inputs, "Wall") ?
        Vac.WallShapeSettings(; (Symbol(k) => v for (k, v) in inputs["Wall"])...) :
        Vac.WallShapeSettings()
    FFS.sing_lim!(intr, ctrl, equil)
    intr.nlow = ctrl.nn_low
    intr.nhigh = ctrl.nn_high
    intr.npert = 1
    FFS.sing_find!(intr, equil)
    intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
    intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
    intr.mpert = intr.mhigh - intr.mlow + 1
    intr.mband = intr.mpert - 1
    intr.numpert_total = intr.mpert * intr.npert
    metric = FFS.make_metric(equil; mband=intr.mband, fft_flag=ctrl.fft_flag)
    ffit = FFS.make_matrix(equil, intr, metric)
    return ctrl, ffit, intr
end

"""Benchmark `eulerlagrange_integration` for one `(n, mode)` at the worker's thread
count. Setup (untimed) rebuilds the mutable state every sample."""
function bench_path(inputs, n, mode, nth, equil, samples)
    builder = () -> make_state(inputs, n, mode, nth, equil)
    # Untimed warmup so the timed samples are free of JIT compilation.
    let
        ctrl, ffit, intr = builder()
        FFS.eulerlagrange_integration(ctrl, equil, ffit, intr)
    end
    b = @benchmarkable FFS.eulerlagrange_integration(c, $equil, f, i) setup = ((c, f, i) = ($builder)()) samples = samples evals = 1 seconds = 3600
    return run(b)
end

"""Reduce a BenchmarkTools `Trial` to a TOML-serializable summary (seconds)."""
function summarize(trial)
    t = trial.times ./ 1e9
    return Dict{String,Any}(
        "min_s"        => minimum(t),
        "median_s"     => median(t),
        "mean_s"       => mean(t),
        "std_s"        => length(t) > 1 ? std(t) : 0.0,
        "n_samples"    => length(t),
        "memory_bytes" => trial.memory,
        "allocs"       => trial.allocs
    )
end

function run_worker(outfile, samples)
    nth = Threads.nthreads()
    inputs = TOML.parsefile(joinpath(EXAMPLE_DIR, "gpec.toml"))
    equil = build_equilibrium(inputs)
    # standard / serial-Riccati are thread-independent — measure them once.
    modes = nth == 1 ? (:standard, :riccati, :parallel) : (:parallel,)
    out = Dict{String,Any}("threads" => nth, "samples" => samples,
                           "results" => Dict{String,Any}())
    for n in N_VALUES
        nkey = "n$n"
        out["results"][nkey] = Dict{String,Any}()
        for mode in modes
            @info "worker nth=$nth  n=$n  mode=$mode  (samples=$samples)"
            trial = bench_path(inputs, n, mode, nth, equil, samples)
            r = summarize(trial)
            out["results"][nkey][String(mode)] = r
            @printf("  -> median %.3f s   min %.3f s   std %.3f s   mem %.1f MiB   allocs %d\n",
                    r["median_s"], r["min_s"], r["std_s"], r["memory_bytes"] / 2^20, r["allocs"])
        end
    end
    mkpath(dirname(outfile))
    open(outfile, "w") do io
        TOML.print(io, out)
    end
    @info "worker wrote $outfile"
end
