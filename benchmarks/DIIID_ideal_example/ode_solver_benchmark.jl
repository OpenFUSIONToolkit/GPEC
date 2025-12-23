"""
ODE Solver Benchmark Script

This script benchmarks different ODE solver methods from OrdinaryDiffEq.jl
using the DIIID-like ideal example. It tests various integration methods across
different tolerance levels to determine optimal performance for problems with
singularities.

Solver Methods Tested:
- Tsit5: Current default (Tsitouras 5/4 Runge-Kutta)
- AutoTsit5: Auto-switching for stiff detection
- ExplicitRKN8: 8th order explicit Runge-Kutta (good for oscillatory problems)
- Vern6, Vern7, Vern8: High-order Verner methods
- DP5: Dormand-Prince 5th order (classic method)

Tolerance Scan: [1e-3, 1e-4, 1e-5, 1e-6, 1e-8] where tol_r = tol_nr

Outputs:
- Three plots: steps vs tolerance, timing vs tolerance, efficiency plot
- JLD2 file with raw benchmark results
"""

using Pkg
Pkg.activate("../..")
using JPEC
using OrdinaryDiffEq
using Plots
using JLD2
using Printf
using Statistics

# Import necessary modules
import JPEC.DCON as DCON
import JPEC.Equilibrium as Equilibrium

"""
    ode_step_with_solver!(odet::DCON.OdeState, ctrl::DCON.DconControl, equil, ffit, intr, solver)

Modified version of ode_step! that accepts a solver method as a parameter.
"""
function ode_step_with_solver!(odet::DCON.OdeState, ctrl::DCON.DconControl, equil, ffit, intr, solver)
    # Callback to be run at every step
    cb = DiscreteCallback((u, t, integrator) -> true, DCON.integrator_callback!)

    # Advance differential equation to next singular surface or edge
    rtol = DCON.compute_tols(ctrl, intr, odet)
    prob = ODEProblem(DCON.sing_der!, odet.u, (odet.psifac, odet.psimax), (ctrl, equil, ffit, intr, odet))
    sol = solve(prob, solver; reltol=rtol, callback=cb)

    # Update u and psifac with the solution at the end of the interval
    odet.u .= sol.u[end]
    odet.psifac = sol.t[end]
end

"""
    ode_run_with_solver(ctrl::DCON.DconControl, equil, ffit, intr, solver)

Modified version of ode_run that uses a custom solver method.
Returns the OdeState and the number of integration steps taken.
"""
function ode_run_with_solver(ctrl::DCON.DconControl, equil, ffit, intr, solver)
    # Initialization
    odet = DCON.OdeState(intr.numpert_total, ctrl.numsteps_init, ctrl.numunorms_init, intr.msing)

    if ctrl.sing_start <= 0
        DCON.ode_axis_init!(odet, ctrl, equil, intr)
    else
        error("sing_start > 0 not implemented yet!")
    end

    # Always integrate once
    ode_step_with_solver!(odet, ctrl, equil, ffit, intr, solver)

    # Continue integrating through singular surfaces
    while odet.ising != ctrl.ksing && odet.next == "cross"
        if ctrl.kin_flag
            error("kin_flag = true not implemented yet!")
        else
            DCON.ode_ideal_cross!(odet, ctrl, equil, ffit, intr)
        end
        ode_step_with_solver!(odet, ctrl, equil, ffit, intr, solver)
    end

    # Post-processing
    if ctrl.psiedge < intr.psilim
        odet.step = DCON.findmax_dW_edge!(odet, ctrl, equil, ffit, intr)
        DCON.trim_storage!(odet)
        intr.psilim = odet.psi_store[end]
        intr.qlim = odet.q_store[end]
        odet.u .= odet.u_store[:, :, :, end]
    else
        odet.step -= 1
        DCON.trim_storage!(odet)
    end

    odet.nzero = DCON.evaluate_stability_criterion!(odet, equil)
    DCON.transform_u!(odet, intr)

    return odet, odet.step
end

"""
    run_benchmark(solver, solver_name, tol_values)

Run benchmark for a single solver across all tolerance values.
"""
function run_benchmark(solver, solver_name::String, tol_values, equil, ffit, intr_template, ctrl_template)
    println("\n" * "="^70)
    println("Benchmarking solver: $solver_name")
    println("="^70)

    results = Dict{Float64, Any}()

    for tol in tol_values
        println("\n  Testing tolerance: $tol")

        # Create fresh copies of ctrl and intr for each run
        ctrl = deepcopy(ctrl_template)
        intr = deepcopy(intr_template)

        # Set tolerances
        ctrl.tol_r = tol
        ctrl.tol_nr = tol
        ctrl.verbose = false  # Suppress output during benchmarking

        try
            # Run benchmark (3 samples to get statistics)
            times = Float64[]
            nsteps = 0

            for i in 1:3
                ctrl_test = deepcopy(ctrl)
                intr_test = deepcopy(intr)

                # Time the execution
                start_time = time()
                odet, steps = ode_run_with_solver(ctrl_test, equil, ffit, intr_test, solver)
                elapsed = time() - start_time

                push!(times, elapsed)
                nsteps = steps  # Should be the same for all runs
            end

            results[tol] = Dict(
                "time_mean" => mean(times),
                "time_std" => std(times),
                "steps" => nsteps,
                "success" => true
            )

            println("    ✓ Steps: $nsteps, Time: $(@sprintf("%.3f", results[tol]["time_mean"])) ± $(@sprintf("%.3f", results[tol]["time_std"])) s")

        catch e
            println("    ✗ Failed: $e")
            results[tol] = Dict(
                "time_mean" => NaN,
                "time_std" => NaN,
                "steps" => NaN,
                "success" => false,
                "error" => string(e)
            )
        end
    end

    return results
end

# Main benchmark execution
function main()
    println("="^70)
    println("ODE Solver Benchmark for DCON Integration")
    println("Example: DIIID-like ideal case")
    println("="^70)

    # Load equilibrium and setup
    println("\nLoading equilibrium and initializing...")
    equil = Equilibrium.setup_equilibrium("equil.toml")

    # Setup DCON
    ctrl = DCON.DconControl("dcon.toml")

    # Initialize vacuum and DCON structures
    wv, grri, xzpts = JPEC.Vacuum.mscvac()
    ffit = DCON.fourfit(ctrl, equil)
    intr = DCON.dcon_init(ctrl, equil, wv, grri, xzpts)

    # Save templates for making copies
    ctrl_template = deepcopy(ctrl)
    intr_template = deepcopy(intr)

    # Define solvers to test
    solvers = [
        (Tsit5(), "Tsit5"),
        (AutoTsit5(Rosenbrock23()), "AutoTsit5"),
        (Vern6(), "Vern6"),
        (Vern7(), "Vern7"),
        (Vern8(), "Vern8"),
        (DP5(), "DP5"),
        (BS5(), "BS5")
    ]

    # Tolerance values to scan
    tol_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-8]

    # Run benchmarks
    all_results = Dict{String, Any}()

    for (solver, solver_name) in solvers
        all_results[solver_name] = run_benchmark(solver, solver_name, tol_values, equil, ffit, intr_template, ctrl_template)
    end

    # Save results to JLD2
    println("\n" * "="^70)
    println("Saving results...")
    jldsave("ode_solver_benchmark_results.jld2"; results=all_results, tol_values=tol_values)
    println("Results saved to: ode_solver_benchmark_results.jld2")

    # Generate plots
    println("\nGenerating plots...")
    generate_plots(all_results, tol_values)

    # Print summary table
    print_summary(all_results, tol_values)

    println("\n" * "="^70)
    println("Benchmark complete!")
    println("="^70)
end

"""
    generate_plots(results, tol_values)

Generate three plots: steps vs tolerance, timing vs tolerance, and efficiency plot.
"""
function generate_plots(results, tol_values)
    # Plot 1: Steps vs Tolerance
    p1 = plot(
        xlabel="Tolerance",
        ylabel="Number of Integration Steps",
        title="ODE Solver Steps vs Tolerance",
        xscale=:log10,
        yscale=:log10,
        legend=:topright,
        size=(800, 600),
        grid=true,
        minorgrid=true
    )

    for (solver_name, solver_results) in results
        steps = [get(solver_results[tol], "steps", NaN) for tol in tol_values]
        valid_idx = .!isnan.(steps)
        if any(valid_idx)
            plot!(p1, tol_values[valid_idx], steps[valid_idx],
                  marker=:circle, label=solver_name, linewidth=2)
        end
    end

    savefig(p1, "ode_benchmark_steps_vs_tolerance.png")
    println("  ✓ Saved: ode_benchmark_steps_vs_tolerance.png")

    # Plot 2: Timing vs Tolerance (with error ribbons)
    p2 = plot(
        xlabel="Tolerance",
        ylabel="Execution Time (s)",
        title="ODE Solver Timing vs Tolerance",
        xscale=:log10,
        yscale=:log10,
        legend=:topright,
        size=(800, 600),
        grid=true,
        minorgrid=true
    )

    for (solver_name, solver_results) in results
        times = [get(solver_results[tol], "time_mean", NaN) for tol in tol_values]
        errors = [get(solver_results[tol], "time_std", NaN) for tol in tol_values]
        valid_idx = .!isnan.(times)
        if any(valid_idx)
            plot!(p2, tol_values[valid_idx], times[valid_idx],
                  ribbon=errors[valid_idx],
                  marker=:circle, label=solver_name, linewidth=2,
                  fillalpha=0.2)
        end
    end

    savefig(p2, "ode_benchmark_timing_vs_tolerance.png")
    println("  ✓ Saved: ode_benchmark_timing_vs_tolerance.png")

    # Plot 3: Efficiency (Time vs Steps), colored by tolerance
    p3 = plot(
        xlabel="Number of Steps",
        ylabel="Execution Time (s)",
        title="ODE Solver Efficiency (colored by tolerance)",
        xscale=:log10,
        yscale=:log10,
        legend=:outertopright,
        size=(900, 600),
        grid=true,
        minorgrid=true
    )

    for (solver_name, solver_results) in results
        for (i, tol) in enumerate(tol_values)
            if haskey(solver_results, tol) && solver_results[tol]["success"]
                steps = solver_results[tol]["steps"]
                time = solver_results[tol]["time_mean"]
                scatter!(p3, [steps], [time],
                        marker=:circle, markersize=8,
                        label=(i==1 ? solver_name : ""),
                        color=i)
            end
        end
    end

    savefig(p3, "ode_benchmark_efficiency.png")
    println("  ✓ Saved: ode_benchmark_efficiency.png")
end

"""
    print_summary(results, tol_values)

Print a summary table of all benchmark results.
"""
function print_summary(results, tol_values)
    println("\n" * "="^70)
    println("BENCHMARK SUMMARY")
    println("="^70)

    for (solver_name, solver_results) in sort(collect(results))
        println("\n$solver_name:")
        println("  " * "-"^66)
        println("  Tolerance  | Steps | Time (s) | Success")
        println("  " * "-"^66)
        for tol in tol_values
            if haskey(solver_results, tol)
                res = solver_results[tol]
                if res["success"]
                    println("  $(@sprintf("%.1e", tol))  | $(@sprintf("%5d", res["steps"])) | $(@sprintf("%8.3f", res["time_mean"])) | ✓")
                else
                    println("  $(@sprintf("%.1e", tol))  |   N/A |      N/A | ✗")
                end
            end
        end
    end
end

# Run the benchmark
main()
