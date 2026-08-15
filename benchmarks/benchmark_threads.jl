# Thread-scaling benchmark for the Riccati chunked propagator integration.
# Runs the Solovev (N=8) and DIIID-like (N=26) examples with integrator="riccati"
# across 1, 2, 4, 8 threads and compares against the forward path. The chunk list is
# thread-independent, so et[1] must not move with the thread count — only wall-clock.
#
# Usage (from JPEC_main root):
#   for t in 1 2 4 8; do julia -t $t --project=. benchmarks/benchmark_threads.jl; done

using GeneralizedPerturbedEquilibrium, TOML, Printf, Statistics

function run_ffs(ex; integrator)
    inputs = TOML.parsefile(joinpath(ex, "gpec.toml"))
    inputs["ForceFreeStates"]["verbose"] = false
    inputs["ForceFreeStates"]["integrator"] = integrator
    inputs["ForceFreeStates"]["write_outputs_to_HDF5"] = false
    intr = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesInternal(; dir_path=ex)
    ctrl = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesControl(;
        (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
    eq_config = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], ex)
    equil = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(eq_config)
    intr.wall_settings = GeneralizedPerturbedEquilibrium.Vacuum.WallShapeSettings(;
        (Symbol(k) => v for (k, v) in inputs["Wall"])...)
    GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_lim!(intr, ctrl, equil)
    intr.nlow = ctrl.nn_low;
    intr.nhigh = ctrl.nn_high;
    intr.npert = 1
    GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_find!(intr, equil)
    intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
    intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
    intr.mpert = intr.mhigh - intr.mlow + 1
    intr.numpert_total = intr.mpert * intr.npert
    metric = GeneralizedPerturbedEquilibrium.ForceFreeStates.make_metric(equil, intr.mpert)
    ffit = GeneralizedPerturbedEquilibrium.ForceFreeStates.make_matrix(equil, intr, metric)
    odet, _, _, _ = GeneralizedPerturbedEquilibrium.ForceFreeStates.eulerlagrange_integration(ctrl, equil, ffit, intr)
    vac = GeneralizedPerturbedEquilibrium.ForceFreeStates.free_run(odet, ctrl, equil, ffit, intr)
    return real(vac.et[1]), intr.numpert_total
end

function timed_run(ex; integrator, nwarm=1, nrep=2)
    # Warmup
    for _ in 1:nwarm
        run_ffs(ex; integrator)
    end
    # Timed runs
    times = Float64[]
    local et1, N
    for _ in 1:nrep
        t0 = time()
        et1, N = run_ffs(ex; integrator)
        push!(times, time() - t0)
    end
    return mean(times), et1, N
end

nthreads = Threads.nthreads()
root = joinpath(@__DIR__, "..")
sol_ex = joinpath(root, "test", "test_data", "regression_solovev_ideal_example")
diiid_ex = joinpath(root, "examples", "DIIID-like_ideal_example")

println("\n=== Thread-scaling benchmark ($(nthreads) thread(s)) ===\n")

for (label, ex) in [("Solovev", sol_ex), ("DIIID-like", diiid_ex)]
    t_fwd, et_fwd, N = timed_run(ex; integrator="forward")
    t_ric, et_ric, _ = timed_run(ex; integrator="riccati")

    err_ric = abs(et_ric - et_fwd) / abs(et_fwd) * 100

    println("$label (N=$N, nthreads=$nthreads)")
    @printf("  forward    et[1]=%.5f  t=%.2fs  speedup=1.00×\n", et_fwd, t_fwd)
    @printf("  riccati    et[1]=%.5f  t=%.2fs  speedup=%.2f×  err=%.4f%%\n",
        et_ric, t_ric, t_fwd/t_ric, err_ric)
    println()
end
