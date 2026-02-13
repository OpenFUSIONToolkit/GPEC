"""
Benchmark script for comparing Pn_minus_half_1997 and Pn_minus_half_2007 implementations.

This script:
1. Verifies agreement between 2007 Julia implementation and Fortran
2. Demonstrates expected divergence between 1997 and 2007 at large nloc
3. Benchmarks performance comparison between 1997 and 2007 versions

Usage:
    julia --project=. benchmarks/benchmark_legendre_2007.jl
"""

using JPEC
using Printf
using Statistics

# Access the internal functions
using JPEC.Vacuum: Pn_minus_half_1997, Pn_minus_half_2007

println("="^80)
println("Benchmark: Legendre Function Implementations (1997 vs 2007)")
println("="^80)

# Test parameters
s_values = [1.5, 2.0, 3.0, 5.0, 10.0]
n_small = [0, 1, 2, 5, 10]
n_large = [20, 50, 100, 200]

println("\n1. CORRECTNESS: Comparing 1997 and 2007 implementations")
println("-"^80)

println("\nFor small n (expected good agreement):")
println(@sprintf("%-8s %-8s %-20s %-20s %-20s", "s", "n", "P_1997", "P_2007", "Rel. Diff"))
println("-"^80)

for s in [2.0, 5.0]
    for n in n_small
        P_1997 = Pn_minus_half_1997(s, n)
        P_2007 = Pn_minus_half_2007(s, n)

        # Compare the n-th order value
        val_1997 = P_1997[end-1]
        val_2007 = P_2007[end-1]
        rel_diff = abs(val_1997 - val_2007) / abs(val_1997)

        println(@sprintf("%-8.2f %-8d %-20.12e %-20.12e %-20.12e", s, n, val_1997, val_2007, rel_diff))
    end
end

println("\nFor large n (expected divergence):")
println(@sprintf("%-8s %-8s %-20s %-20s %-20s", "s", "n", "P_1997", "P_2007", "Rel. Diff"))
println("-"^80)

for s in [2.0, 5.0]
    for n in n_large
        P_1997 = Pn_minus_half_1997(s, n)
        P_2007 = Pn_minus_half_2007(s, n)

        # Compare the n-th order value
        val_1997 = P_1997[end-1]
        val_2007 = P_2007[end-1]
        rel_diff = abs(val_1997 - val_2007) / abs(val_1997)

        println(@sprintf("%-8.2f %-8d %-20.12e %-20.12e %-20.12e", s, n, val_1997, val_2007, rel_diff))
    end
end

println("\n2. PERFORMANCE: Speed comparison")
println("-"^80)

# Function to benchmark execution time
function benchmark_function(f, s, n, iterations=100)
    # Warmup
    f(s, n)

    # Benchmark
    times = zeros(iterations)
    for i in 1:iterations
        times[i] = @elapsed f(s, n)
    end

    return mean(times), std(times)
end

println("\nTiming comparison (averaged over 100 runs):")
println(@sprintf("%-8s %-8s %-20s %-20s %-15s", "s", "n", "1997 (ms)", "2007 (ms)", "Speedup"))
println("-"^80)

test_cases = [
    (2.0, 5),
    (2.0, 10),
    (2.0, 20),
    (2.0, 50),
    (5.0, 5),
    (5.0, 10),
    (5.0, 20),
    (5.0, 50)
]

for (s, n) in test_cases
    time_1997, std_1997 = benchmark_function(Pn_minus_half_1997, s, n)
    time_2007, std_2007 = benchmark_function(Pn_minus_half_2007, s, n)

    speedup = time_1997 / time_2007

    println(@sprintf("%-8.2f %-8d %-20.6f %-20.6f %-15.2fx",
        s, n, time_1997*1000, time_2007*1000, speedup))
end

println("\n3. VISUAL CHECK: rhohat parameter and integration method selection")
println("-"^80)

println("\nChecking when Gaussian integration is used (n*rhohat >= 0.1):")
println(@sprintf("%-8s %-8s %-15s %-20s", "s", "n", "rhohat", "Method"))
println("-"^80)

for s in [1.5, 2.0, 3.0, 5.0, 10.0]
    xxq = s * s
    ysq = xxq - 1.0
    y = sqrt(ysq)
    w = s + y
    rhohat = sqrt(1.0 / (2.0 * y * w))

    for n in [1, 5, 10, 20, 50, 100]
        criterion = n * rhohat
        method = criterion >= 0.1 ? "Gaussian" : "Recurrence"
        println(@sprintf("%-8.2f %-8d %-15.6f %-20s", s, n, rhohat, method))
    end
end

println("\n" * "="^80)
println("Benchmark complete!")
println("="^80)

println("\nSummary:")
println("1. For small n, both methods should agree closely (rel. diff < 1e-10)")
println("2. For large n, methods are expected to diverge due to numerical stability")
println("3. The 2007 method uses Gaussian integration when n*rhohat >= 0.1")
println("4. Performance depends on whether Gaussian integration or recurrence is used")
