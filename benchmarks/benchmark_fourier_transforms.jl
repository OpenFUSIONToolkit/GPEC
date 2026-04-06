"""
Benchmark script comparing FourierTransform implementations to FFTW.

This script compares:
1. FourierTransform allocating API: ft(data)
2. FourierTransform in-place API: transform!(output, ft, data)
3. Low-level matrix API: fourier_transform!(gil, gij, cs, m00, l00)
4. FFTW: fft() and rfft()

The custom FourierTransform is designed for truncated Fourier series with
arbitrary mode ranges (mlow:mhigh), while FFTW computes full DFT (modes 0:N-1).
"""

using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Utilities.FourierTransforms
using FFTW
using BenchmarkTools
using Printf

println("="^80)
println("Fourier Transform Benchmark Comparison")
println("="^80)

# Helper function to extract specific modes from full FFT
function extract_modes(fft_result, mlow, mhigh, mtheta)
    """Extract modes mlow:mhigh from full FFT result."""
    modes = zeros(ComplexF64, mhigh - mlow + 1)
    for (i, m) in enumerate(mlow:mhigh)
        if m >= 0
            # Positive frequencies
            modes[i] = fft_result[m + 1] / mtheta  # FFT normalization
        else
            # Negative frequencies (wrap around)
            modes[i] = fft_result[mtheta + m + 1] / mtheta
        end
    end
    return modes
end

# Test configurations
test_cases = [
    (name="Small (mtheta=128, mpert=10)",   mtheta=128,  mpert=10,  mlow=-5),
    (name="Medium (mtheta=256, mpert=20)",  mtheta=256,  mpert=20,  mlow=-10),
    (name="Large (mtheta=480, mpert=40)",   mtheta=480,  mpert=40,  mlow=-20),
    (name="Very Large (mtheta=1024, mpert=80)", mtheta=1024, mpert=80, mlow=-40),
]

for test in test_cases
    println("\n" * "="^80)
    println(test.name)
    println("="^80)

    mtheta = test.mtheta
    mpert = test.mpert
    mlow = test.mlow
    mhigh = mlow + mpert - 1

    # Create test data
    theta = range(0, 2π, length=mtheta+1)[1:end-1]
    data = sin.(3 .* theta) .+ 0.5 .* cos.(7 .* theta) .+ 0.2 .* sin.(11 .* theta)

    # Initialize FourierTransform
    ft = FourierTransform(mtheta, mpert, mlow)

    # Pre-allocate buffers for in-place operations
    modes_buffer = zeros(ComplexF64, mpert)
    theta_buffer = zeros(ComplexF64, mtheta)

    # Pre-allocate for low-level API
    cslth, snlth = compute_fourier_coefficients(mtheta, mpert, mlow)
    gij = reshape(data, mtheta, 1)  # Matrix form
    gil = zeros(Float64, mtheta, mpert)

    # Pre-compute FFTW plan for fair comparison
    fft_plan = plan_fft(data)

    println("\n--- Forward Transform ---")

    # 1. Our allocating API
    print("FourierTransform (allocating):  ")
    t1 = @benchmark $ft($data)
    display(t1)
    modes_alloc = ft(data)

    # 2. Our in-place API
    print("\nFourierTransform (in-place):    ")
    t2 = @benchmark transform!($modes_buffer, $ft, $data)
    display(t2)
    transform!(modes_buffer, ft, data)

    # 3. FFTW
    print("\nFFTW (full DFT):                ")
    t3 = @benchmark $fft_plan * $data
    display(t3)
    fft_result = fft_plan * data
    fft_modes = extract_modes(fft_result, mlow, mhigh, mtheta)

    # Verify results match (for overlapping modes)
    # Note: Our transform uses a different normalization and basis
    println("\n--- Accuracy Check ---")
    println("FourierTransform allocating vs in-place: ",
            @sprintf("%.2e", maximum(abs.(modes_alloc .- modes_buffer))))

    # Compare magnitudes of modes (since basis might differ)
    println("Mode magnitudes comparison (FourierTransform vs FFTW):")
    println("  FourierTransform peak: ", @sprintf("%.4f", maximum(abs.(modes_alloc))))
    println("  FFTW peak:             ", @sprintf("%.4f", maximum(abs.(fft_modes))))

    println("\n--- Inverse Transform ---")

    # Use modes from our forward transform
    modes_test = modes_alloc

    # 1. Our allocating inverse
    print("inverse(ft, modes) [allocating]: ")
    t4 = @benchmark inverse($ft, $modes_test)
    display(t4)
    theta_alloc = inverse(ft, modes_test)

    # 2. Our in-place inverse
    print("\ninverse_transform! [in-place]:   ")
    t5 = @benchmark inverse_transform!($theta_buffer, $ft, $modes_test)
    display(t5)
    inverse_transform!(theta_buffer, ft, modes_test)

    # 3. IFFT
    # Note: IFFT requires full spectrum, not just truncated modes
    print("\nIFFT (full DFT, not comparable): ")
    full_modes = zeros(ComplexF64, mtheta)
    for (i, m) in enumerate(mlow:mhigh)
        if m >= 0
            full_modes[m + 1] = modes_test[i]
        else
            full_modes[mtheta + m + 1] = modes_test[i]
        end
    end
    t6 = @benchmark ifft($full_modes)
    display(t6)

    # Accuracy check
    println("\n--- Inverse Accuracy Check ---")
    println("inverse() allocating vs in-place: ",
            @sprintf("%.2e", maximum(abs.(theta_alloc .- theta_buffer))))
    println("Round-trip error (real part):     ",
            @sprintf("%.2e", maximum(abs.(real.(theta_alloc) .- data))))

    # Performance summary
    println("\n--- Performance Summary ---")
    println(@sprintf("Forward transform speedup (in-place vs allocating): %.2fx",
            median(t1).time / median(t2).time))
    println(@sprintf("Allocations eliminated: %d → %d",
            t1.allocs, t2.allocs))

    # Compare to FFTW
    println(@sprintf("\nFourier vs FFTW (forward): %.2fx %s",
            abs(median(t2).time / median(t3).time),
            median(t2).time < median(t3).time ? "faster" : "slower"))
    println("Note: FFTW computes full DFT (all N modes), we compute truncated series ($mpert modes)")
end

# Additional test: Matrix transforms (batch processing)
println("\n" * "="^80)
println("Batch Processing Test (Multiple Functions at Once)")
println("="^80)

mtheta = 256
mpert = 20
mlow = -10
nbatch = 10  # Transform 10 functions simultaneously

ft = FourierTransform(mtheta, mpert, mlow)
theta = range(0, 2π, length=mtheta+1)[1:end-1]

# Create batch data
data_matrix = zeros(Float64, mtheta, nbatch)
for i in 1:nbatch
    data_matrix[:, i] = sin.(i .* theta)
end

modes_matrix = zeros(ComplexF64, mpert, nbatch)

println("\nTransforming $nbatch functions of length $mtheta:")

print("Allocating (loop):  ")
@btime for i in 1:$nbatch
    modes = $ft($data_matrix[:, i])
end

print("Allocating (matrix):")
@btime $ft($data_matrix)

print("In-place (loop):    ")
modes_buffer = zeros(ComplexF64, mpert)
@btime for i in 1:$nbatch
    transform!($modes_buffer, $ft, $data_matrix[:, i])
end

print("In-place (matrix):  ")
@btime transform!($modes_matrix, $ft, $data_matrix)

# Low-level matrix API test
println("\n" * "="^80)
println("Low-Level Matrix API Test (for Vacuum module)")
println("="^80)

mtheta = 128
mpert = 10
mlow = 1

# Setup for low-level API
cslth, snlth = compute_fourier_coefficients(mtheta, mpert, mlow)
gij = randn(mtheta, mtheta)  # Green's function matrix
gil = zeros(Float64, mtheta, mpert)

println("\nLow-level fourier_transform! with offset support:")
print("With offsets (m00=0, l00=0):    ")
@btime fourier_transform!($gil, $gij, $cslth, 0, 0)

print("With offsets (m00=64, l00=5):   ")
gil_large = zeros(Float64, 256, 20)
@btime fourier_transform!($gil_large, $gij, $cslth, 64, 5)

println("\n" * "="^80)
println("Benchmark Complete")
println("="^80)

println("\nKey Takeaways:")
println("1. In-place API eliminates allocations with same performance")
println("2. Custom transforms optimized for truncated series (arbitrary mode ranges)")
println("3. FFTW is faster but computes full DFT (all modes, fixed range 0:N-1)")
println("4. Use FourierTransform when you need:")
println("   - Arbitrary mode ranges (e.g., m = -20:20)")
println("   - Truncated series (fewer modes than grid points)")
println("   - Phase-shifted basis functions (n*qa*delta terms)")
println("5. Use FFTW when you need:")
println("   - Full DFT (all modes 0:N-1)")
println("   - Maximum performance for dense spectral data")
