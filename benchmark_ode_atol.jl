"""
Benchmark script to compare ODE solver performance with and without absolute tolerance.

This script runs the DCON ODE solver on test cases and compares:
1. Total number of integration steps
2. Number of steps in deep core (psi < 0.1)
3. Lowest eigenvalue of total energy matrix (wt)
4. Integration time

Usage:
    julia --project=. benchmark_ode_atol.jl
"""

using JPEC
using Printf
using Statistics

# Test case directories
test_cases = [
    joinpath(@__DIR__, "test", "test_data", "regression_solovev_ideal_example"),
    joinpath(@__DIR__, "test", "test_data", "regression_solovev_ideal_example_multi_n")
]

"""
Count steps in different psi regions
"""
function analyze_steps(odet)
    total_steps = odet.step

    # Count steps in different regions
    deep_core_steps = sum(odet.psi_store[1:total_steps] .< 0.1)
    core_steps = sum(0.1 .<= odet.psi_store[1:total_steps] .< 0.5)
    edge_steps = sum(odet.psi_store[1:total_steps] .>= 0.5)

    return (
        total = total_steps,
        deep_core = deep_core_steps,
        core = core_steps,
        edge = edge_steps,
        pct_deep_core = 100 * deep_core_steps / total_steps
    )
end

"""
Run DCON and collect statistics
"""
function run_and_analyze(test_dir, use_atol::Bool)
    println("\n" * "="^70)
    println("Running: $(basename(test_dir))")
    println("Absolute tolerance: $(use_atol ? "ENABLED" : "DISABLED")")
    println("="^70)

    # Temporarily modify the code to disable atol if needed
    # For now, we'll just run with the current implementation

    t_start = time()

    # Run DCON
    try
        result = JPEC.DCON.Main(test_dir)

        t_elapsed = time() - t_start

        # Analyze steps
        odet = result.odet
        steps = analyze_steps(odet)

        # Get lowest eigenvalue (if available)
        min_eigenval = minimum(odet.crit_store[1:odet.step])

        println("\nResults:")
        println("  Total integration steps: $(steps.total)")
        println("  Steps in deep core (ψ < 0.1): $(steps.deep_core) ($(round(steps.pct_deep_core, digits=1))%)")
        println("  Steps in core (0.1 ≤ ψ < 0.5): $(steps.core)")
        println("  Steps in edge (ψ ≥ 0.5): $(steps.edge)")
        println("  Lowest eigenvalue: $(round(min_eigenval, sigdigits=6))")
        println("  Integration time: $(round(t_elapsed, digits=2)) seconds")

        return (
            success = true,
            steps = steps,
            min_eigenval = min_eigenval,
            time = t_elapsed
        )
    catch e
        println("ERROR: $e")
        return (success = false, error = e)
    end
end

"""
Main benchmark routine
"""
function main()
    println("="^70)
    println("ODE Absolute Tolerance Benchmark")
    println("="^70)
    println("\nThis benchmark compares ODE solver performance with absolute tolerance enabled.")
    println("The current implementation uses absolute tolerance by default.")
    println("\nTo compare with baseline (no atol), you would need to:")
    println("1. Comment out the atol_scale computation in ode_run")
    println("2. Set atol=0.0 in compute_tols")
    println("3. Re-run this benchmark")

    results = Dict()

    for test_case in test_cases
        if isdir(test_case)
            result = run_and_analyze(test_case, true)
            results[basename(test_case)] = result
        else
            println("\nWarning: Test case not found: $test_case")
        end
    end

    println("\n" * "="^70)
    println("Benchmark Summary")
    println("="^70)

    for (name, result) in results
        if result.success
            println("\n$name:")
            println("  Total steps: $(result.steps.total)")
            println("  Deep core %: $(round(result.steps.pct_deep_core, digits=1))%")
            println("  Min eigenvalue: $(round(result.min_eigenval, sigdigits=6))")
            println("  Time: $(round(result.time, digits=2))s")
        else
            println("\n$name: FAILED")
        end
    end

    println("\n" * "="^70)
    println("\nTo properly benchmark the improvement:")
    println("1. Create a baseline by disabling atol (set atol_scale=0.0 in OdeState init)")
    println("2. Run this script and save output as baseline.txt")
    println("3. Re-enable atol (current implementation)")
    println("4. Run this script again and compare")
    println("5. Look for reduction in deep core steps and preserved eigenvalues")
end

# Run the benchmark
main()
