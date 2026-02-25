"""
Benchmark script for profiling JPEC workloads with Julia's Profile module and PProf.

This script:
1. Runs a chosen workload (full pipeline, equilibrium-only, or stability-only)
2. Uses Profile.@profile to collect samples
3. Exports results to pprof format via PProf.jl for inspection (flame graph, etc.)

Test data: Default is the DIIID-like ideal example (realistic EFIT + ForceFreeStates).
Use Solovev_ideal_example for a faster, smaller profile.

Usage:
    julia --project=. benchmarks/benchmark_profile.jl [method] [path]

Arguments:
    method  "full" | "equilibrium" | "stability"  (default: full)
    path    Path to example directory with jpec.toml (default: examples/DIIID-like_ideal_example)

Examples:
    julia --project=. benchmarks/benchmark_profile.jl
    julia --project=. benchmarks/benchmark_profile.jl full
    julia --project=. benchmarks/benchmark_profile.jl stability examples/DIIID-like_ideal_example
    julia --project=. benchmarks/benchmark_profile.jl equilibrium examples/Solovev_ideal_example
"""

using Printf
using Pkg
using Profile
using PProf
using TOML

Pkg.activate(joinpath(@__DIR__, ".."))
using JPEC
using FastInterpolations: cubic_interp, CubicFit
import FastInterpolations

# --- Defaults ---
const DEFAULT_METHOD = "full"
const DEFAULT_PATH = joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")

# --- Output suppression ---
function run_quietly(f)
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            return f()
        end
    end
end

# --- Workload: full JPEC pipeline ---
function workload_full(path::String)
    JPEC.main([path])
end

# --- Workload: equilibrium setup only ---
function workload_equilibrium(path::String)
    Eq = JPEC.Equilibrium
    inputs = TOML.parsefile(joinpath(path, "jpec.toml"))
    eq_config = Eq.EquilibriumConfig(inputs["Equilibrium"], path)
    eq_input, _ = Eq.read_efit(eq_config)
    pe, _ = Eq.equilibrium_solver(eq_input)
    Eq.equilibrium_global_parameters!(pe)
    Eq.equilibrium_qfind!(pe)
    Eq.equilibrium_gse!(pe)
    pe
end

# --- Workload: stability (ForceFreeStates) after equilibrium ---
function run_dcon_phases(path::String, equil)
    FFS = JPEC.ForceFreeStates
    Vacuum = JPEC.Vacuum

    intr = FFS.ForceFreeStatesInternal(; dir_path=path)
    inputs = TOML.parsefile(joinpath(path, "jpec.toml"))
    ctrl = FFS.ForceFreeStatesControl(; (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
    ctrl.verbose = false
    ctrl.write_outputs_to_HDF5 = false

    if "Wall" in keys(inputs)
        intr.wall_settings = Vacuum.WallShapeSettings(; (Symbol(k) => v for (k, v) in inputs["Wall"])...)
    end
    if "DEBUG" in keys(inputs)
        intr.debug_settings = FFS.DebugSettings(; (Symbol(k) => v for (k, v) in inputs["DEBUG"])...)
    end
    ctrl.delta_mhigh *= 2

    FFS.sing_lim!(intr, ctrl, equil)
    if ctrl.set_psilim_via_dmlim && ctrl.psiedge < intr.psilim
        ctrl.psiedge = 1.0
    end

    locstab_fs = zeros(Float64, length(equil.profiles.xs), 5)
    if ctrl.mer_flag
        FFS.mercier_scan!(locstab_fs, equil)
    end
    if ctrl.bal_flag
        FFS.compute_ballooning_stability!(ctrl, locstab_fs, equil)
    end
    intr.locstab = cubic_interp(equil.profiles.xs, locstab_fs; bc=CubicFit(), extrap=FastInterpolations.ExtendExtrap())

    if ctrl.nn_low == 0
        ctrl.nn_low = ctrl.nn_high
    end
    if ctrl.nn_high == 0
        ctrl.nn_high = ctrl.nn_low
    end
    intr.nlow = ctrl.nn_low
    intr.nhigh = ctrl.nn_high
    intr.npert = intr.nhigh - intr.nlow + 1

    FFS.sing_find!(intr, equil)

    if ctrl.delta_mlow < 0 || ctrl.delta_mhigh < 0
        error("Negative delta_mlow or delta_mhigh not allowed")
    end
    if ctrl.cyl_flag
        intr.mlow = ctrl.delta_mlow
        intr.mhigh = ctrl.delta_mhigh
    elseif ctrl.sing_start == 0
        intr.mlow = trunc(Int, min(intr.nlow * equil.params.qmin, 0)) - 4 - ctrl.delta_mlow
        intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
    else
        mmin = Inf
        for ising in Int(ctrl.sing_start):intr.msing
            mmin = min(mmin, minimum(intr.sing[ising].m))
        end
        intr.mlow = trunc(Int, mmin) - ctrl.delta_mlow
        intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
    end
    intr.mpert = intr.mhigh - intr.mlow + 1
    if ctrl.delta_mband >= intr.mpert
        ctrl.delta_mband = 0
    end
    intr.mband = intr.mpert - 1 - ctrl.delta_mband
    intr.mband = min(max(intr.mband, 0), intr.mpert - 1)
    intr.numpert_total = intr.mpert * intr.npert

    metric = nothing
    ffit = nothing
    odet = nothing
    vac_data = nothing

    if ctrl.mat_flag || ctrl.ode_flag
        metric = FFS.make_metric(equil; mband=intr.mband, fft_flag=ctrl.fft_flag)
        ffit = FFS.make_matrix(equil, intr, metric)
    end

    if ctrl.ode_flag
        odet = FFS.eulerlagrange_integration(ctrl, equil, ffit, intr)
    end

    if ctrl.vac_flag && !(ctrl.ksing > 0 && ctrl.ksing <= intr.msing + 1)
        vac_data = FFS.free_run!(odet, ctrl, equil, ffit, intr)
    end

    (intr, odet, vac_data)
end

function workload_stability(path::String)
    equil = workload_equilibrium(path)
    run_dcon_phases(path, equil)
end

# --- Parse args ---
function parse_args()
    method = DEFAULT_METHOD
    path = DEFAULT_PATH
    if length(ARGS) >= 1
        method = ARGS[1]
    end
    if length(ARGS) >= 2
        path = abspath(ARGS[2])
    else
        path = abspath(path)
    end
    (method, path)
end

# --- Main ---
function main()
    method, path = parse_args()

    if !isdir(path)
        error("Example path is not a directory: $path")
    end
    if !isfile(joinpath(path, "jpec.toml"))
        error("No jpec.toml found in $path")
    end

    workload = nothing
    if method == "full"
        workload = () -> run_quietly(() -> workload_full(path))
    elseif method == "equilibrium"
        workload = () -> run_quietly(() -> workload_equilibrium(path))
    elseif method == "stability"
        workload = () -> run_quietly(() -> workload_stability(path))
    else
        error("Unknown method: $method. Use full, equilibrium, or stability.")
    end

    println("="^80)
    println("JPEC Profile Benchmark (Profile + PProf)")
    println("="^80)
    println(@sprintf("Method: %s", method))
    println(@sprintf("Path:   %s", path))
    println("="^80)

    println("\n1. WARMUP")
    println("-"^80)
    println("Running workload once (output suppressed)...")
    workload()
    println("Warmup done.")

    println("\n2. PROFILING")
    println("-"^80)
    Profile.clear()
    n_runs = method == "full" ? 1 : 2
    println(@sprintf("Collecting profile over %d run(s)...", n_runs))
    for _ in 1:n_runs
        @profile workload()
    end
    println("Profile collection done.")

    println("\n3. EXPORT TO PPROF")
    println("-"^80)
    outfile = joinpath(@__DIR__, "..", "profile_$(method).pb.gz")
    pprof(; outfile=outfile)
    println(@sprintf("Written: %s", outfile))
    println("View with: pprof -http=:8080 \"$(outfile)\"")
    println("Or use PProf's web UI if it opened automatically.")

    println("\n" * "="^80)
    println("Profile benchmark complete.")
    println("="^80)
end

main()
