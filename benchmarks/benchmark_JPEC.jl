# Master benchmark script: runs full ForceFreeStates (stability) walkthrough and benchmarks each method within each step.
# Reports runtime (mean ± std) and memory allocations per method, sorted by runtime.
# Output is suppressed during benchmarks; final run output is shown for verification.
# MUST change path in the config to whichever example want to benchmark
# Usage: julia --project=. benchmarks/benchmark_JPEC.jl

using Printf
using Pkg
using Statistics
using TOML

Pkg.activate(joinpath(@__DIR__, ".."))
using JPEC
using FastInterpolations: cubic_interp, CubicFit
import FastInterpolations

# --- Config ---
# CHANGE THIS TO THE PATH YOU WANT!
const BENCHMARK_PATH = joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")
const N_RUNS = 5  # Number of runs for averaging (methods called once per run get N_RUNS samples)

# Reference values for verification. Update these from a known-good run (e.g. copy from verification output).
const EXPECTED = (
    et1_real = 1.701,
    et1_imag = 0.00001875,
    nstep = 910,
    ep_real = -3.949e-2,
    ep_imag = 2.014e-5,
    ev_real = 1.740,
    ev_imag = -1.389e-6,
)
const TOL_REL = 0.01   # 1% relative tolerance for checks
const TOL_ABS = 1e-4   # Absolute tolerance for small values

# --- Output suppression ---
function run_quietly(f)
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            return f()
        end
    end
end

# --- ForceFreeStates phases (replicates JPEC.main flow with per-phase timing). Pass precomputed `equil`. ---
function run_dcon_phases(path, equil)
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

    phases = Tuple{String,Float64,Int64}[]

    _, t, b, _, _ = @timed FFS.sing_lim!(intr, ctrl, equil)
    push!(phases, ("ForceFreeStates.sing_lim!", t, b))

    if ctrl.set_psilim_via_dmlim && ctrl.psiedge < intr.psilim
        ctrl.psiedge = 1.0
    end

    locstab_fs = zeros(Float64, length(equil.profiles.xs), 5)
    if ctrl.mer_flag
        _, t, b, _, _ = @timed FFS.mercier_scan!(locstab_fs, equil)
        push!(phases, ("ForceFreeStates.mercier_scan!", t, b))
    end
    if ctrl.bal_flag
        _, t, b, _, _ = @timed FFS.compute_ballooning_stability!(ctrl, locstab_fs, equil)
        push!(phases, ("ForceFreeStates.compute_ballooning_stability!", t, b))
    end

    _, t, b, _, _ = @timed intr.locstab = cubic_interp(equil.profiles.xs, locstab_fs; bc=CubicFit(), extrap=FastInterpolations.ExtendExtrap())
    push!(phases, ("ForceFreeStates.locstab_spline", t, b))

    if ctrl.nn_low == 0
        ctrl.nn_low = ctrl.nn_high
    end
    if ctrl.nn_high == 0
        ctrl.nn_high = ctrl.nn_low
    end
    intr.nlow = ctrl.nn_low
    intr.nhigh = ctrl.nn_high
    intr.npert = intr.nhigh - intr.nlow + 1

    _, t, b, _, _ = @timed FFS.sing_find!(intr, equil)
    push!(phases, ("ForceFreeStates.sing_find!", t, b))

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
        metric, t, b, _, _ = @timed FFS.make_metric(equil; mband=intr.mband, fft_flag=ctrl.fft_flag)
        push!(phases, ("ForceFreeStates.make_metric", t, b))

        ffit, t, b, _, _ = @timed FFS.make_matrix(equil, intr, metric)
        push!(phases, ("ForceFreeStates.make_matrix", t, b))
    end

    if ctrl.ode_flag
        odet, t, b, _, _ = @timed FFS.eulerlagrange_integration(ctrl, equil, ffit, intr)
        push!(phases, ("ForceFreeStates.eulerlagrange_integration", t, b))
    end

    if ctrl.vac_flag && !(ctrl.ksing > 0 && ctrl.ksing <= intr.msing + 1)
        vac_data, t, b, _, _ = @timed FFS.free_run!(odet, ctrl, equil, ffit, intr)
        push!(phases, ("ForceFreeStates.free_run!", t, b))
    end

    (phases, intr, ctrl, equil, ffit, odet, vac_data)
end

# --- Run benchmarks N_RUNS times, collect stats ---
function collect_benchmark_stats(path)
    Eq = JPEC.Equilibrium
    inputs = TOML.parsefile(joinpath(path, "jpec.toml"))
    eq_config = Eq.EquilibriumConfig(inputs["Equilibrium"], path)

    all_phase_times = Dict{String,Vector{Float64}}()
    all_phase_allocs = Dict{String,Vector{Int64}}()
    last_vac = nothing
    last_odet = nothing

    for run in 1:N_RUNS
        run_quietly() do
            eq_input, t, b, _, _ = @timed Eq.read_efit(eq_config)
            record_phase!(all_phase_times, all_phase_allocs, "Equilibrium.read_efit", t, b)

            pe, t, b, _, _ = @timed Eq.equilibrium_solver(eq_input)
            record_phase!(all_phase_times, all_phase_allocs, "Equilibrium.equilibrium_solver", t, b)

            _, t, b, _, _ = @timed Eq.equilibrium_global_parameters!(pe)
            record_phase!(all_phase_times, all_phase_allocs, "Equilibrium.equilibrium_global_parameters!", t, b)

            _, t, b, _, _ = @timed Eq.equilibrium_qfind!(pe)
            record_phase!(all_phase_times, all_phase_allocs, "Equilibrium.equilibrium_qfind!", t, b)

            _, t, b, _, _ = @timed Eq.equilibrium_gse!(pe)
            record_phase!(all_phase_times, all_phase_allocs, "Equilibrium.equilibrium_gse!", t, b)

            phases, _, _, _, _, odet, vac = run_dcon_phases(path, pe)
            last_odet = odet
            last_vac = vac
            for (name, t, b) in phases
                record_phase!(all_phase_times, all_phase_allocs, name, t, b)
            end
        end
    end

    results = []
    for name in sort(collect(keys(all_phase_times)))
        ts = all_phase_times[name]
        bs = all_phase_allocs[name]
        push!(results, (
            method = name,
            time_avg = mean(ts),
            time_std = length(ts) > 1 ? std(ts) : 0.0,
            n_runs = length(ts),
            alloc_bytes_avg = mean(bs),
        ))
    end
    (results, last_odet, last_vac)
end

function record_phase!(times, allocs, name, t, b)
    if !haskey(times, name)
        times[name] = Float64[]
        allocs[name] = Int64[]
    end
    push!(times[name], t)
    push!(allocs[name], b)
end

# --- Verification ---
function verify_results(odet, vac_data)
    checks = []
    if vac_data !== nothing
        et1 = vac_data.et[1]
        ok_et = (abs(real(et1) - EXPECTED.et1_real) / max(abs(EXPECTED.et1_real), 1e-10) <= TOL_REL) &&
                (abs(imag(et1) - EXPECTED.et1_imag) <= TOL_ABS)
        push!(checks, ("et[1] (least stable eigenmode energy)", et1, complex(EXPECTED.et1_real, EXPECTED.et1_imag), ok_et))

        ep1 = vac_data.ep[1]
        ok_ep = (abs(real(ep1) - EXPECTED.ep_real) / max(abs(EXPECTED.ep_real), 1e-10) <= TOL_REL) &&
                (abs(imag(ep1) - EXPECTED.ep_imag) <= TOL_ABS)
        push!(checks, ("ep[1] (plasma energy)", ep1, complex(EXPECTED.ep_real, EXPECTED.ep_imag), ok_ep))

        ev1 = vac_data.ev[1]
        ok_ev = (abs(real(ev1) - EXPECTED.ev_real) / max(abs(EXPECTED.ev_real), 1e-10) <= TOL_REL) &&
                (abs(imag(ev1) - EXPECTED.ev_imag) <= TOL_ABS)
        push!(checks, ("ev[1] (vacuum energy)", ev1, complex(EXPECTED.ev_real, EXPECTED.ev_imag), ok_ev))
    end
    if odet !== nothing
        ok_nstep = abs(odet.step - EXPECTED.nstep) <= max(2, EXPECTED.nstep ÷ 100)
        push!(checks, ("ODE steps", odet.step, EXPECTED.nstep, ok_nstep))
    end
    checks
end

# --- Print table sorted by runtime ---
function print_results_table(results)
    sorted = sort(results, by = r -> r.time_avg, rev = true)
    println("\n")
    println("="^100)
    println("BENCHMARK RESULTS (sorted by runtime descending)")
    println("="^100)
    println(@sprintf("%-45s | %10s | %12s | %6s | %12s", "Method", "Avg Time (s)", "Std Dev (s)", "Runs", "Alloc (MB)"))
    println("-"^100)
    for r in sorted
        t_str = @sprintf("%.4f", r.time_avg)
        t_std_str = r.time_std > 0 ? @sprintf("%.4f", r.time_std) : "-"
        alloc_mb = r.alloc_bytes_avg / 1e6
        alloc_str = alloc_mb > 0 ? @sprintf("%.2f", alloc_mb) : "-"
        println(@sprintf("%-45s | %10s | %12s | %6s | %12s", r.method, t_str, t_std_str, r.n_runs, alloc_str))
    end
    println("="^100)
end

# --- Main ---
function main()
    println("="^60)
    println("JPEC Benchmark: $(BENCHMARK_PATH)")
    println("Running $(N_RUNS) samples per phase (output suppressed)")
    println("="^60)

    results, odet, vac_data = collect_benchmark_stats(BENCHMARK_PATH)

    print_results_table(results)

    # Verification
    println("\n")
    println("="^60)
    println("RESULT VERIFICATION (vs expected reference values)")
    println("="^60)
    checks = verify_results(odet, vac_data)
    for (name, got, exp, ok) in checks
        status = ok ? "✓ PASS" : "✗ FAIL"
        println(@sprintf("%-35s | got: %-25s | expected: %-25s | %s", name, repr(got), repr(exp), status))
    end
    println("="^60)

    # Final run with output for verification
    println("\n")
    println("FINAL RUN (output visible):")
    println("-"^60)
    JPEC.main([BENCHMARK_PATH])
    println("-"^60)
    println("Benchmark complete.")
end

main()
