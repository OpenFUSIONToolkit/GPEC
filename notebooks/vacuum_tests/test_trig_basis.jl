"""
Test script to verify vectorized trigonometric basis initialization
produces identical results to the original nested loop version.
Written by Claude.
"""

using Printf
using LinearAlgebra

# ============================================================================
# Original version (nested loops)
# ============================================================================

function init_trig_basis_old(mtheta::Int, n::Int, qa::Float64, delta::Vector{Float64},
    mlow::Int, mpert::Int)
    cos_nqdelta = zeros(mtheta)
    sin_nqdelta = zeros(mtheta)
    sin_mstheta = zeros(mtheta, mpert)
    cos_mstheta = zeros(mtheta, mpert)
    sin_mstheta_arg = zeros(mtheta, mpert)
    cos_mstheta_arg = zeros(mtheta, mpert)

    for is in 1:mtheta
        theta = (is-1) * 2π / mtheta
        nqdelta = n * qa * delta[is]
        cos_nqdelta[is] = cos(nqdelta)
        sin_nqdelta[is] = sin(nqdelta)
        for l1 in 1:mpert
            mi = mlow - 1 + l1
            mitheta = mi * theta
            mitheta_arg = mi * theta + nqdelta
            sin_mstheta[is, l1] = sin(mitheta)
            cos_mstheta[is, l1] = cos(mitheta)
            sin_mstheta_arg[is, l1] = sin(mitheta_arg)
            cos_mstheta_arg[is, l1] = cos(mitheta_arg)
        end
    end

    return (cos_nqdelta, sin_nqdelta, sin_mstheta, cos_mstheta,
        sin_mstheta_arg, cos_mstheta_arg)
end

# ============================================================================
# New version (vectorized)
# ============================================================================

function init_trig_basis_new(mtheta::Int, n::Int, qa::Float64, delta::Vector{Float64},
    mlow::Int, mpert::Int)
    # Theta grid
    theta_grid = range(0; stop=2π, length=mtheta + 1)[1:(end-1)]

    # Compute n*q*δ phase term for each poloidal angle
    nqdelta = n .* qa .* delta

    # Basis functions for the phase factor
    cos_nqdelta = cos.(nqdelta)
    sin_nqdelta = sin.(nqdelta)

    # Mode numbers: m = mlow, mlow+1, ..., mlow+mpert-1
    mode_numbers = (mlow-1) .+ (1:mpert)'  # Row vector for broadcasting

    # Outer product: theta_grid (column) × mode_numbers (row) → (mtheta × mpert) matrix
    mitheta = theta_grid * mode_numbers  # Broadcasting: m*θ for all combinations

    # Compute basis functions without phase (pure harmonics)
    sin_mstheta = sin.(mitheta)
    cos_mstheta = cos.(mitheta)

    # Add phase factor: m*θ + n*q*δ (broadcast nqdelta column-wise across modes)
    mitheta_arg = mitheta .+ nqdelta
    sin_mstheta_arg = sin.(mitheta_arg)
    cos_mstheta_arg = cos.(mitheta_arg)

    return (cos_nqdelta, sin_nqdelta, sin_mstheta, cos_mstheta,
        sin_mstheta_arg, cos_mstheta_arg)
end

# ============================================================================
# Test function
# ============================================================================

function test_trig_basis()
    println("\n" * "="^70)
    println("Testing Trigonometric Basis Initialization")
    println("="^70)

    # Test parameters (typical tokamak values)
    mtheta = 512
    mpert = 50
    mlow = 1
    n = 1
    qa = 2.5
    delta = 0.3 .* sin.(range(0, 2π; length=mtheta+1)[1:(end-1)])  # Typical triangularity profile

    println("\nTest configuration:")
    println("  mtheta = $mtheta (poloidal grid points)")
    println("  mpert  = $mpert (Fourier modes)")
    println("  mlow   = $mlow (lowest mode number)")
    println("  n      = $n (toroidal mode number)")
    println("  qa     = $qa (edge safety factor)")
    println("  delta  = triangularity profile (sinusoidal)")

    # Run old version
    print("\nOld (nested loops): ")
    @time (cos_nqd_old, sin_nqd_old, sin_ms_old, cos_ms_old,
        sin_msa_old, cos_msa_old) = init_trig_basis_old(mtheta, n, qa, delta, mlow, mpert)

    # Run new version
    print("New (vectorized):   ")
    @time (cos_nqd_new, sin_nqd_new, sin_ms_new, cos_ms_new,
        sin_msa_new, cos_msa_new) = init_trig_basis_new(mtheta, n, qa, delta, mlow, mpert)

    # Compare results
    println("\n" * "-"^70)
    println("RESULTS")
    println("-"^70)

    arrays = [
        ("cos(n*q*δ)", cos_nqd_old, cos_nqd_new),
        ("sin(n*q*δ)", sin_nqd_old, sin_nqd_new),
        ("sin(m*θ)", sin_ms_old, sin_ms_new),
        ("cos(m*θ)", cos_ms_old, cos_ms_new),
        ("sin(m*θ + n*q*δ)", sin_msa_old, sin_msa_new),
        ("cos(m*θ + n*q*δ)", cos_msa_old, cos_msa_new)
    ]

    all_pass = true
    for (name, old, new) in arrays
        diff = old .- new
        max_diff = maximum(abs.(diff))
        rel_diff = max_diff / (maximum(abs.(old)) + 1e-16)

        pass = max_diff < 1e-14
        status = pass ? "✓ PASS" : "✗ FAIL"
        all_pass = all_pass && pass

        @printf("%-20s: max_diff = %.2e, rel_diff = %.2e  %s\n",
            name, max_diff, rel_diff, status)
    end

    println("-"^70)

    if all_pass
        println("\n🎉 All arrays match within machine precision!")
    else
        println("\n⚠️  Some arrays differ - review output above")
    end

    return all_pass
end

# ============================================================================
# Benchmark comparison
# ============================================================================

function benchmark_trig_basis()
    println("\n" * "="^70)
    println("Performance Benchmark")
    println("="^70)

    test_sizes = [(128, 20), (256, 30), (512, 50), (1024, 100)]

    for (mtheta, mpert) in test_sizes
        println("\nmtheta = $mtheta, mpert = $mpert:")

        # Setup
        mlow = 1
        n = 1
        qa = 2.5
        delta = 0.3 .* sin.(range(0, 2π; length=mtheta+1)[1:(end-1)])

        # Warm-up
        init_trig_basis_old(mtheta, n, qa, delta, mlow, mpert)
        init_trig_basis_new(mtheta, n, qa, delta, mlow, mpert)

        # Benchmark
        print("  Old (nested loops): ")
        t_old = @elapsed for _ in 1:100
            init_trig_basis_old(mtheta, n, qa, delta, mlow, mpert)
        end
        @printf("%.3f ms/call\n", t_old * 10)

        print("  New (vectorized):   ")
        t_new = @elapsed for _ in 1:100
            init_trig_basis_new(mtheta, n, qa, delta, mlow, mpert)
        end
        @printf("%.3f ms/call\n", t_new * 10)

        speedup = t_old / t_new
        @printf("  Speedup: %.1fx\n", speedup)
    end
end

# ============================================================================
# Main
# ============================================================================

function run_tests()
    println("\n")
    println("╔" * "="^68 * "╗")
    println("║" * " "^12 * "TRIGONOMETRIC BASIS INITIALIZATION TEST" * " "^17 * "║")
    println("╚" * "="^68 * "╝")

    pass = test_trig_basis()
    benchmark_trig_basis()

    println("\n" * "="^70)
    println("TEST SUMMARY")
    println("="^70)
    println(pass ? "✓ All tests passed!" : "✗ Some tests failed")
    println("="^70 * "\n")

    return pass
end

# Run tests
if abspath(PROGRAM_FILE) == @__FILE__
    run_tests()
end
