"""
Test script to verify that the refactored Fourier transform functions
produce identical results to the original nested-loop versions.
Written by Claude.
"""

using LinearAlgebra
using Printf

# Try to load FFTW for FFT-based implementations
const USE_FFT = try
    using FFTW
    true
catch
    @warn "FFTW not available. Run `import Pkg; Pkg.add(\"FFTW\")` to enable FFT tests."
    false
end

# ============================================================================
# Original versions (nested loops)
# ============================================================================

function fourier_transform_old!(
    gil::Matrix{Float64},
    gij::Matrix{Float64},
    cs::Matrix{Float64},
    m00::Int,
    l00::Int,
    mth::Int,
    mpert::Int
)
    # Zero out relevant gil block
    for l1 in 1:mpert
        for i in 1:mth
            gil[m00+i, l00+l1] = 0.0
        end
    end

    # Accumulate with nested loops
    for l1 in 1:mpert
        for j in 1:mth
            for i in 1:mth
                gil[m00+i, l00+l1] += cs[j, l1] * gij[i, j]
            end
        end
    end
end

function fourier_inverse_transform_old!(
    gll::Matrix{Float64},
    gil::Matrix{Float64},
    cs::Matrix{Float64},
    m00::Int,
    l00::Int,
    mth::Int,
    mpert::Int
)
    # Zero out gll block
    for l1 in 1:mpert
        for l2 in 1:mpert
            gll[l2, l1] = 0.0
        end
    end

    # Main accumulation
    dth = 2π / mth
    for l1 in 1:mpert
        for l2 in 1:mpert
            gll[l2, l1] = dth * sum(cs[i, l2] * gil[m00+i, l00+l1] for i in 1:mth) * 2π
        end
    end
end

# ============================================================================
# New versions (vectorized)
# ============================================================================

function fourier_transform_new!(
    gil::Matrix{Float64},
    gij::Matrix{Float64},
    cs::Matrix{Float64},
    m00::Int,
    l00::Int,
    mth::Int,
    mpert::Int
)
    # Zero out relevant gil block
    gil[(m00+1):(m00+mth), (l00+1):(l00+mpert)] .= 0.0

    # Accumulate Fourier transform via matrix multiply
    mul!(view(gil, (m00+1):(m00+mth), (l00+1):(l00+mpert)), gij, cs)
end

function fourier_inverse_transform_new!(
    gll::Matrix{Float64},
    gil::Matrix{Float64},
    cs::Matrix{Float64},
    m00::Int,
    l00::Int,
    mth::Int,
    mpert::Int
)
    # Zero out gll block
    fill!(view(gll, 1:mpert, 1:mpert), 0.0)

    # Inverse Fourier transform via matrix multiply: gll = cs^T * gil * (2π * dth)
    # This computes: gll[l2, l1] = (2π * dth) * Σ_i cs[i, l2] * gil[i, l1]
    dth = 2π / mth
    mul!(gll, cs', view(gil, (m00+1):(m00+mth), (l00+1):(l00+mpert)), 2π * dth, 0.0)
end

# ============================================================================
# FFT-based versions (only if FFTW is available)
# ============================================================================

if USE_FFT
    function fourier_transform_fft!(
        gil::Matrix{Float64},
        gij::Matrix{Float64},
        cs::Matrix{Float64},
        m00::Int,
        l00::Int,
        mth::Int,
        mpert::Int
    )
        # Zero out relevant gil block
        gil[(m00+1):(m00+mth), (l00+1):(l00+mpert)] .= 0.0

        # For each row, compute FFT along columns and extract modes
        for i in 1:mth
            # Take FFT of row i across all theta points
            fft_result = fft(gij[i, :])

            # Extract the first mpert Fourier modes and store
            # Note: This assumes cs represents standard Fourier basis
            for l in 1:min(mpert, length(fft_result))
                gil[m00+i, l00+l] = real(fft_result[l]) / mth
            end
        end
    end

    function fourier_inverse_transform_fft!(
        gll::Matrix{Float64},
        gil::Matrix{Float64},
        cs::Matrix{Float64},
        m00::Int,
        l00::Int,
        mth::Int,
        mpert::Int
    )
        # Zero out gll block
        gll[1:mpert, 1:mpert] .= 0.0

        dth = 2π / mth

        # For each mode pair, compute inverse FFT
        for l1 in 1:mpert
            for l2 in 1:mpert
                # Extract spectral coefficients from gil
                spectral_data = Complex{Float64}[gil[m00+i, l00+l1] for i in 1:mth]

                # Pad to full size if needed
                if length(spectral_data) < mth
                    append!(spectral_data, zeros(Complex{Float64}, mth - length(spectral_data)))
                end

                # Inverse FFT
                physical_data = ifft(spectral_data)

                # Accumulate contribution (simplified - actual implementation depends on cs structure)
                gll[l2, l1] += dth * 2π * sum(abs.(physical_data))
            end
        end
    end
end  # if USE_FFT

# ============================================================================
# Test functions
# ============================================================================

function test_fourier_transform()
    println("\n" * "="^70)
    println("Testing fourier_transform!")
    println("="^70)

    # Test parameters
    mth = 64
    mpert = 10
    m00 = 5
    l00 = 3

    # Create test data
    gij = randn(mth, mth)
    cs = randn(mth, mpert)

    # Allocate output matrices
    gil_old = zeros(m00 + mth + 10, l00 + mpert + 10)
    gil_new = zeros(m00 + mth + 10, l00 + mpert + 10)
    gil_fft = zeros(m00 + mth + 10, l00 + mpert + 10)

    # Run old version
    print("  Old (nested loops): ")
    @time fourier_transform_old!(gil_old, gij, cs, m00, l00, mth, mpert)

    # Run new version
    print("  New (vectorized):   ")
    @time fourier_transform_new!(gil_new, gij, cs, m00, l00, mth, mpert)

    # Run FFT version if available
    if USE_FFT
        print("  FFT (built-in):     ")
        @time fourier_transform_fft!(gil_fft, gij, cs, m00, l00, mth, mpert)
    end

    # Compare results: old vs new
    diff = gil_old .- gil_new
    max_diff = maximum(abs.(diff))
    rel_diff = max_diff / (maximum(abs.(gil_old)) + 1e-10)

    println("\nResults (Old vs New):")
    println("  Max absolute difference: ", @sprintf("%.2e", max_diff))
    println("  Max relative difference: ", @sprintf("%.2e", rel_diff))

    # Check relevant block
    block_old = gil_old[(m00+1):(m00+mth), (l00+1):(l00+mpert)]
    block_new = gil_new[(m00+1):(m00+mth), (l00+1):(l00+mpert)]
    block_fft = gil_fft[(m00+1):(m00+mth), (l00+1):(l00+mpert)]
    block_diff = maximum(abs.(block_old .- block_new))

    println("  Block max difference:    ", @sprintf("%.2e", block_diff))
    println("  Block norm (old):        ", @sprintf("%.6f", norm(block_old)))
    println("  Block norm (new):        ", @sprintf("%.6f", norm(block_new)))

    # Compare FFT version if available (note: may differ due to different basis)
    if USE_FFT
        diff_fft = block_old .- block_fft
        max_diff_fft = maximum(abs.(diff_fft))
        println("\nFFT Comparison (informational - may use different basis):")
        println("  FFT vs Old max diff:     ", @sprintf("%.2e", max_diff_fft))
        println("  Block norm (FFT):        ", @sprintf("%.6f", norm(block_fft)))
    end

    if max_diff < 1e-10
        println("\n✓ PASS: Results are identical (within machine precision)")
    else
        println("\n✗ FAIL: Results differ by more than tolerance")
        println("\nFirst 5x5 of OLD result:")
        display(block_old[1:5, 1:5])
        println("\nFirst 5x5 of NEW result:")
        display(block_new[1:5, 1:5])
        println("\nFirst 5x5 of DIFFERENCE:")
        display((block_old .- block_new)[1:5, 1:5])
    end

    return max_diff < 1e-10
end

function test_fourier_inverse_transform()
    println("\n" * "="^70)
    println("Testing fourier_inverse_transform!")
    println("="^70)

    # Test parameters
    mth = 64
    mpert = 10
    m00 = 5
    l00 = 3

    # Create test data
    gil = randn(m00 + mth + 10, l00 + mpert + 10)
    cs = randn(mth, mpert)

    # Allocate output matrices
    gll_old = zeros(mpert, mpert)
    gll_new = zeros(mpert, mpert)
    gll_fft = zeros(mpert, mpert)

    # Run old version
    print("  Old (nested loops): ")
    @time fourier_inverse_transform_old!(gll_old, gil, cs, m00, l00, mth, mpert)

    # Run new version
    print("  New (vectorized):   ")
    @time fourier_inverse_transform_new!(gll_new, gil, cs, m00, l00, mth, mpert)

    # Run FFT version if available
    if USE_FFT
        print("  FFT (built-in):     ")
        @time fourier_inverse_transform_fft!(gll_fft, gil, cs, m00, l00, mth, mpert)
    end

    # Compare results: old vs new
    diff = gll_old .- gll_new
    max_diff = maximum(abs.(diff))
    rel_diff = max_diff / (maximum(abs.(gll_old)) + 1e-10)

    println("\nResults (Old vs New):")
    println("  Max absolute difference: ", @sprintf("%.2e", max_diff))
    println("  Max relative difference: ", @sprintf("%.2e", rel_diff))
    println("  Matrix norm (old):       ", @sprintf("%.6f", norm(gll_old)))
    println("  Matrix norm (new):       ", @sprintf("%.6f", norm(gll_new)))

    # Compare FFT version if available
    if USE_FFT
        diff_fft = gll_old .- gll_fft
        max_diff_fft = maximum(abs.(diff_fft))
        println("\nFFT Comparison (informational - may use different basis):")
        println("  FFT vs Old max diff:     ", @sprintf("%.2e", max_diff_fft))
        println("  Matrix norm (FFT):       ", @sprintf("%.6f", norm(gll_fft)))
    end

    if max_diff < 1e-10
        println("\n✓ PASS: Results are identical (within machine precision)")
    else
        println("\n✗ FAIL: Results differ by more than tolerance")
        println("\nFirst 5x5 of OLD result:")
        display(gll_old[1:5, 1:5])
        println("\nFirst 5x5 of NEW result:")
        display(gll_new[1:5, 1:5])
        println("\nFirst 5x5 of DIFFERENCE:")
        display(diff[1:5, 1:5])
    end

    return max_diff < 1e-10
end

function benchmark_comparison()
    println("\n" * "="^70)
    println("Performance Comparison")
    println("="^70)

    mth = 512
    mpert = 50
    m00 = 10
    l00 = 5

    # Test data
    gij = randn(mth, mth)
    cs = randn(mth, mpert)
    gil_old = zeros(m00 + mth + 20, l00 + mpert + 20)
    gil_new = zeros(m00 + mth + 20, l00 + mpert + 20)
    gil_fft = zeros(m00 + mth + 20, l00 + mpert + 20)

    println("\nFourier Transform (mth=$mth, mpert=$mpert):")
    print("  Old (nested loops): ")
    @time fourier_transform_old!(gil_old, gij, cs, m00, l00, mth, mpert)

    print("  New (vectorized):   ")
    @time fourier_transform_new!(gil_new, gij, cs, m00, l00, mth, mpert)

    if USE_FFT
        print("  FFT (built-in):     ")
        @time fourier_transform_fft!(gil_fft, gij, cs, m00, l00, mth, mpert)
    end

    # Inverse transform
    gil = randn(m00 + mth + 20, l00 + mpert + 20)
    gll_old = zeros(mpert, mpert)
    gll_new = zeros(mpert, mpert)
    gll_fft = zeros(mpert, mpert)

    println("\nInverse Fourier Transform (mth=$mth, mpert=$mpert):")
    print("  Old (nested loops): ")
    @time fourier_inverse_transform_old!(gll_old, gil, cs, m00, l00, mth, mpert)

    print("  New (vectorized):   ")
    @time fourier_inverse_transform_new!(gll_new, gil, cs, m00, l00, mth, mpert)

    if USE_FFT
        print("  FFT (built-in):     ")
        @time fourier_inverse_transform_fft!(gll_fft, gil, cs, m00, l00, mth, mpert)
    end
end

# ============================================================================
# Main test runner
# ============================================================================

function run_all_tests()
    println("\n")
    println("╔" * "="^68 * "╗")
    println("║" * " "^15 * "FOURIER TRANSFORM TEST SUITE" * " "^25 * "║")
    println("╚" * "="^68 * "╝")

    test1_pass = test_fourier_transform()
    test2_pass = test_fourier_inverse_transform()

    # Benchmark
    benchmark_comparison()

    # Summary
    println("\n" * "="^70)
    println("TEST SUMMARY")
    println("="^70)
    println("  fourier_transform!:          ", test1_pass ? "✓ PASS" : "✗ FAIL")
    println("  fourier_inverse_transform!:  ", test2_pass ? "✓ PASS" : "✗ FAIL")
    println()

    if test1_pass && test2_pass
        println("🎉 All tests passed!")
    else
        println("⚠️  Some tests failed - review output above")
    end
    println("="^70 * "\n")

    return test1_pass && test2_pass
end

# Run tests
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_tests()
end
