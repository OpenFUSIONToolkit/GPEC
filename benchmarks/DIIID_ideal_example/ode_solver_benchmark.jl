"""
ODE Solver Benchmark Script

This script benchmarks different OrdinaryDiffEq.jl solver methods for the DCON ODE integration.
It tests various solvers with different tolerance levels to determine optimal performance
for problems approaching singularities.

Solvers tested:
- Tsit5: Current default, Runge-Kutta 4/5 method (non-stiff)
- AutoTsit5: Automatic stiffness detection with Tsit5
- Vern6: 6th order Verner method (efficient, high accuracy)
- Vern7: 7th order Verner method (higher accuracy)
- Vern8: 8th order Verner method (even higher accuracy)
- DP5: Dormand-Prince 5th order (classic method)
- BS5: Bogacki-Shampine 5th order (good for moderate accuracy)

Tolerance scan: [1e-3, 1e-4, 1e-5, 1e-6, 1e-8]
Output: Plots of steps vs tolerance and timing vs tolerance
"""

using Pkg
Pkg.activate("../..")
using JPEC
using OrdinaryDiffEq
using BenchmarkTools
using Plots
using Printf
using JLD2

# Import necessary DCON functions and types
import JPEC.DCON: DconControl, OdeState, DconInternal, FourFitVars
import JPEC.DCON: ode_axis_init!, ode_ideal_cross!, integrator_callback!
import JPEC.DCON: compute_tols, sing_der!, DiscreteCallback

"""
Custom ode_step! function that accepts a solver method parameter.
This is a modified version of JPEC.DCON.ode_step! that allows
testing different ODE solvers.
"""
function ode_step_with_solver!(
    odet::OdeState,
    ctrl::DconControl,
    equil::JPEC.Equilibrium.PlasmaEquilibrium,
    ffit::FourFitVars,
    intr::DconInternal,
    solver_method
)
    # Callback to be run at every step
    cb = DiscreteCallback((u, t, integrator) -> true, integrator_callback!)

    # Advance differential equation to next singular surface or edge
    rtol = compute_tols(ctrl, intr, odet)
    prob = ODEProblem(sing_der!, odet.u, (odet.psifac, odet.psimax), (ctrl, equil, ffit, intr, odet))
    sol = solve(prob, solver_method; reltol=rtol, callback=cb)

    # Update u and psifac with the solution at the end of the interval
    odet.u .= sol.u[end]
    odet.psifac = sol.t[end]

    return length(sol.t)  # Return number of steps taken
end

"""
Custom ode_run function that uses a specified solver method.
Returns the OdeState and total number of integration steps.
"""
function ode_run_with_solver(
    ctrl::DconControl,
    equil::JPEC.Equilibrium.PlasmaEquilibrium,
    ffit::FourFitVars,
    intr::DconInternal,
    solver_method
)
    # Initialization
    odet = OdeState(intr.numpert_total, ctrl.numsteps_init, ctrl.numunorms_init, intr.msing)
    if ctrl.sing_start <= 0
        ode_axis_init!(odet, ctrl, equil, intr)
    else
        error("sing_start > 0 not implemented in benchmark")
    end

    total_steps = 0

    # Always integrate once
    steps = ode_step_with_solver!(odet, ctrl, equil, ffit, intr, solver_method)
    total_steps += steps

    # If at rational surfaces, continue integration
    while odet.ising != ctrl.ksing && odet.next == "cross"
        ode_ideal_cross!(odet, ctrl, equil, ffit, intr)
        steps = ode_step_with_solver!(odet, ctrl, equil, ffit, intr, solver_method)
        total_steps += steps
    end

    return odet, total_steps
end

# Define solver methods to test
solver_configs = [
    (name="Tsit5", solver=Tsit5(), description="Current default (RK 4/5)"),
    (name="AutoTsit5", solver=AutoTsit5(Rosenbrock23()), description="Auto stiffness detection"),
    (name="Vern6", solver=Vern6(), description="6th order Verner method"),
    (name="Vern7", solver=Vern7(), description="7th order Verner method"),
    (name="Vern8", solver=Vern8(), description="8th order Verner method"),
    (name="DP5", solver=DP5(), description="Dormand-Prince 5th order"),
    (name="BS5", solver=BS5(), description="Bogacki-Shampine 5th order"),
]

# Tolerance values to scan (tol_r = tol_nr for each run)
tolerance_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-8]

# Load example equilibrium and setup
example_dir = "../../examples/DIIID-like_ideal_example"
println("Loading equilibrium from: $example_dir")
equil = JPEC.Equilibrium.setup_equilibrium("$example_dir/equil.toml")

# Load DCON configuration
dcon_config_path = "$example_dir/dcon.toml"
println("Loading DCON configuration from: $dcon_config_path")
ctrl_template = JPEC.DCON.read_dcon_control(dcon_config_path)

# Storage for results
results = Dict(
    "solver_names" => String[],
    "tolerance_values" => tolerance_values,
    "timings" => Dict{String, Vector{Float64}}(),
    "step_counts" => Dict{String, Vector{Int}}(),
    "timing_stdevs" => Dict{String, Vector{Float64}}(),
)

# Run benchmarks
println("\n" * "="^70)
println("Starting ODE Solver Benchmark")
println("="^70)

for solver_config in solver_configs
    solver_name = solver_config.name
    solver = solver_config.solver

    println("\n--- Testing solver: $solver_name ($(solver_config.description)) ---")
    push!(results["solver_names"], solver_name)

    # Initialize storage for this solver
    results["timings"][solver_name] = Float64[]
    results["step_counts"][solver_name] = Int[]
    results["timing_stdevs"][solver_name] = Float64[]

    for tol in tolerance_values
        println("  Tolerance: $tol")

        # Create modified control with new tolerances
        ctrl = DconControl(
            ctrl_template.bal_flag,
            ctrl_template.mat_flag,
            ctrl_template.ode_flag,
            ctrl_template.vac_flag,
            ctrl_template.mer_flag,
            ctrl_template.set_psilim_via_dmlim,
            ctrl_template.dmlim,
            ctrl_template.psiedge,
            ctrl_template.qlow,
            ctrl_template.qhigh,
            ctrl_template.sing_start,
            ctrl_template.nn_low,
            ctrl_template.nn_high,
            ctrl_template.delta_mlow,
            ctrl_template.delta_mhigh,
            ctrl_template.delta_mband,
            ctrl_template.mthvac,
            ctrl_template.thmax0,
            ctrl_template.kin_flag,
            ctrl_template.con_flag,
            ctrl_template.kinfac1,
            ctrl_template.kinfac2,
            ctrl_template.kingridtype,
            ctrl_template.passing_flag,
            ctrl_template.ktanh_flag,
            ctrl_template.ktc,
            ctrl_template.ktw,
            ctrl_template.ion_flag,
            ctrl_template.electron_flag,
            tol,  # tol_nr
            tol,  # tol_r (set equal to tol_nr)
            ctrl_template.crossover,
            ctrl_template.singfac_min,
            ctrl_template.ucrit,
            ctrl_template.numsteps_init,
            ctrl_template.numunorms_init,
            false,  # verbose = false for benchmarking
            ctrl_template.wall_settings,
        )

        # Wrapper function for benchmarking
        function run_benchmark()
            intr = JPEC.DCON.dcon_init(ctrl, equil)
            ffit = JPEC.DCON.four_fourier_fit(ctrl, equil, intr)
            odet, steps = ode_run_with_solver(ctrl, equil, ffit, intr, solver)
            return steps
        end

        try
            # Dry run to compile
            println("    Compiling...")
            steps_dry = run_benchmark()

            # Benchmark with multiple samples
            println("    Benchmarking...")
            bench = @benchmark $run_benchmark() samples=3 seconds=120

            # Extract statistics
            median_time = median(bench).time / 1e9  # Convert to seconds
            std_time = std(bench).time / 1e9

            # Get step count from final run
            steps = run_benchmark()

            push!(results["timings"][solver_name], median_time)
            push!(results["step_counts"][solver_name], steps)
            push!(results["timing_stdevs"][solver_name], std_time)

            println("    Time: $(@sprintf("%.3f", median_time)) ± $(@sprintf("%.3f", std_time)) s, Steps: $steps")

        catch e
            println("    ERROR: Failed with solver $solver_name at tolerance $tol")
            println("    Error message: $e")
            @show stacktrace(catch_backtrace())
            # Store NaN for failed runs
            push!(results["timings"][solver_name], NaN)
            push!(results["step_counts"][solver_name], 0)
            push!(results["timing_stdevs"][solver_name], NaN)
        end
    end
end

# Save results
println("\nSaving results to ode_solver_benchmark_results.jld2")
@save "ode_solver_benchmark_results.jld2" results

# Generate plots
println("Generating plots...")

# Plot 1: Steps vs Tolerance
plt_steps = plot(
    xlabel="Tolerance (tol_r = tol_nr)",
    ylabel="Number of Integration Steps",
    title="ODE Solver Performance: Step Count vs Tolerance",
    xscale=:log10,
    yscale=:log10,
    legend=:topright,
    size=(800, 600),
    dpi=150,
    grid=true,
    minorgrid=true
)

for solver_name in results["solver_names"]
    valid_indices = findall(x -> x > 0, results["step_counts"][solver_name])
    if !isempty(valid_indices)
        plot!(plt_steps,
              tolerance_values[valid_indices],
              results["step_counts"][solver_name][valid_indices],
              label=solver_name,
              marker=:circle,
              markersize=6,
              linewidth=2)
    end
end

savefig(plt_steps, "ode_solver_steps_vs_tolerance.png")
println("  Saved: ode_solver_steps_vs_tolerance.png")

# Plot 2: Timing vs Tolerance
plt_time = plot(
    xlabel="Tolerance (tol_r = tol_nr)",
    ylabel="Median Integration Time (seconds)",
    title="ODE Solver Performance: Timing vs Tolerance",
    xscale=:log10,
    yscale=:log10,
    legend=:topright,
    size=(800, 600),
    dpi=150,
    grid=true,
    minorgrid=true
)

for solver_name in results["solver_names"]
    times = results["timings"][solver_name]
    stdevs = results["timing_stdevs"][solver_name]
    valid_indices = findall(x -> !isnan(x), times)

    if !isempty(valid_indices)
        plot!(plt_time,
              tolerance_values[valid_indices],
              times[valid_indices],
              label=solver_name,
              marker=:circle,
              markersize=6,
              linewidth=2,
              ribbon=stdevs[valid_indices],
              fillalpha=0.2)
    end
end

savefig(plt_time, "ode_solver_timing_vs_tolerance.png")
println("  Saved: ode_solver_timing_vs_tolerance.png")

# Plot 3: Combined efficiency plot (steps vs time for each tolerance)
plt_efficiency = plot(
    xlabel="Number of Integration Steps",
    ylabel="Median Integration Time (seconds)",
    title="ODE Solver Efficiency: Time vs Steps (colored by tolerance)",
    xscale=:log10,
    yscale=:log10,
    legend=:bottomright,
    size=(800, 600),
    dpi=150,
    grid=true
)

# Use different markers for each tolerance level
markers = [:circle, :square, :diamond, :utriangle, :star5]
colors = palette(:tab10)

for (idx, tol) in enumerate(tolerance_values)
    x_vals = Float64[]
    y_vals = Float64[]

    for solver_name in results["solver_names"]
        if !isnan(results["timings"][solver_name][idx]) && results["step_counts"][solver_name][idx] > 0
            push!(x_vals, Float64(results["step_counts"][solver_name][idx]))
            push!(y_vals, results["timings"][solver_name][idx])
        end
    end

    if !isempty(x_vals)
        scatter!(plt_efficiency,
                 x_vals,
                 y_vals,
                 label="tol=$tol",
                 marker=markers[idx],
                 markersize=8,
                 color=colors[idx])
    end
end

savefig(plt_efficiency, "ode_solver_efficiency.png")
println("  Saved: ode_solver_efficiency.png")

println("\n" * "="^70)
println("Benchmark Complete!")
println("="^70)
println("\nResults summary:")
for solver_name in results["solver_names"]
    println("\n$solver_name:")
    for (idx, tol) in enumerate(tolerance_values)
        time = results["timings"][solver_name][idx]
        steps = results["step_counts"][solver_name][idx]
        if !isnan(time) && steps > 0
            println("  tol=$tol: $(@sprintf("%.3f", time)) s, $steps steps")
        else
            println("  tol=$tol: FAILED")
        end
    end
end

println("\nPlots saved in: $(pwd())")
println("\nTo view results:")
println("  - ode_solver_steps_vs_tolerance.png")
println("  - ode_solver_timing_vs_tolerance.png")
println("  - ode_solver_efficiency.png")
println("  - ode_solver_benchmark_results.jld2")
