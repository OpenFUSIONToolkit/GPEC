# Thread-scaling benchmark for the Riccati chunked propagator integration.
# Runs the Solovev (N=8) and DIIID-like (N=26) examples with the Riccati integrator
# across 1, 2, 4, 8 threads and compares against the forward path. The chunk list is
# thread-independent, so et[1] must not move with the thread count — only wall-clock.
#
# Usage (from JPEC_main root):
#   for t in 1 2 4 8; do julia -t $t --project=. benchmarks/benchmark_threads.jl; done

using GeneralizedPerturbedEquilibrium, TOML, Printf, Statistics

# One solve of the example deck with `integrator` (Forward or Riccati), sharing the deck's ODE numerics.
function run_ffs(ex; integrator)
    GPE = GeneralizedPerturbedEquilibrium
    inputs, config, analytic = GPE.build_inputs_from_toml(ex)
    alg = integrator(; GPE.section_keywords(inputs["Forward"], "Forward", GPE._struct_fields(Forward))...)
    ffs = solve(PlasmaEquilibrium(config, analytic), alg; GPE.force_balance_keywords(inputs, ex)...)
    return real(ffs.free_boundary.et[1]), ffs.numpert_total
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
    t_fwd, et_fwd, N = timed_run(ex; integrator=Forward)
    t_ric, et_ric, _ = timed_run(ex; integrator=Riccati)

    err_ric = abs(et_ric - et_fwd) / abs(et_fwd) * 100

    println("$label (N=$N, nthreads=$nthreads)")
    @printf("  forward    et[1]=%.5f  t=%.2fs  speedup=1.00×\n", et_fwd, t_fwd)
    @printf("  riccati    et[1]=%.5f  t=%.2fs  speedup=%.2f×  err=%.4f%%\n",
        et_ric, t_ric, t_fwd/t_ric, err_ric)
    println()
end
