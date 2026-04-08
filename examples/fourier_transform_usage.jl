"""
Example: Using FourierTransforms utility for different grids

This example demonstrates how to use the FourierTransforms utility module
to create pre-computed Fourier transform objects for different grids:

1. PerturbedEquilibrium: Simple harmonic basis (no phase shift)
2. Vacuum: Basis with toroidal phase coupling (n*qa*delta)
"""

using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Utilities.FourierTransforms

# ==============================================================================
# Example 1: PerturbedEquilibrium Case (No Phase Shift)
# ==============================================================================
println("\n" * "="^70)
println("Example 1: PerturbedEquilibrium - Simple Harmonic Basis")
println("="^70)

# Grid parameters for PerturbedEquilibrium
mtheta = 480        # Number of theta grid points
mpert = 41          # Number of Fourier modes
mlow = -20          # Lowest mode number

# Create FourierTransform object (no phase shift, default arguments)
ft_pe = FourierTransform(mtheta, mpert, mlow)

println("Created FourierTransform for PerturbedEquilibrium:")
println("  mtheta = $mtheta (theta grid points)")
println("  mpert = $mpert (number of modes)")
println("  mlow = $mlow (lowest mode: m ∈ [$mlow, $(mlow+mpert-1)])")
println("\nBasis functions: cos(m*θ), sin(m*θ) with m = $mlow:$(mlow+mpert-1)")

# Create test data: a simple function in theta space
theta = range(0, 2π; length=mtheta+1)[1:(end-1)]
f_theta = sin.(5 .* theta) .+ 0.5 .* cos.(8 .* theta)

# Forward transform: theta → modes
println("\n--- Forward Transform ---")
modes = ft_pe(f_theta)
println("Input: f(θ) at $mtheta theta points (real-valued)")
println("Output: $mpert complex Fourier modes")
println("Dominant modes (|amplitude| > 0.01):")
for l in 1:mpert
    m = mlow + l - 1
    if abs(modes[l]) > 0.01
        println("  m=$m: amplitude = $(abs(modes[l])), phase = $(angle(modes[l]))")
    end
end

# Inverse transform: modes → theta
println("\n--- Inverse Transform ---")
f_reconstructed = inverse(ft_pe, modes)
println("Input: $mpert complex Fourier modes")
println("Output: $(length(f_reconstructed)) theta-space values")
println("Reconstruction error: $(maximum(abs.(real.(f_reconstructed) .- f_theta)))")

# ==============================================================================
# Example 2: Vacuum Case (With Toroidal Phase)
# ==============================================================================
println("\n" * "="^70)
println("Example 2: Vacuum - Basis with Toroidal Coupling")
println("="^70)

# Grid parameters for Vacuum
mthvac = 480        # Number of vacuum theta grid points
mpert_vac = 41      # Number of Fourier modes
mlow_vac = -20      # Lowest mode number
n = 1               # Toroidal mode number
qa = 2.5            # Safety factor at boundary
delta = zeros(mthvac)  # Toroidal phase shift (zero for this example)

# Create FourierTransform object with toroidal phase
ft_vac = FourierTransform(mthvac, mpert_vac, mlow_vac; n=n, qa=qa, delta=delta)

println("Created FourierTransform for Vacuum:")
println("  mthvac = $mthvac (vacuum theta grid points)")
println("  mpert = $mpert_vac (number of modes)")
println("  mlow = $mlow_vac (lowest mode: m ∈ [$mlow_vac, $(mlow_vac+mpert_vac-1)])")
println("  n = $n (toroidal mode number)")
println("  qa = $qa (safety factor)")
println("\nBasis functions: cos(m*θ + n*qa*δ), sin(m*θ + n*qa*δ)")

# Example: Transform Green's function data
g_theta = randn(mthvac)  # Simulated Green's function in theta space
println("\n--- Forward Transform (Green's function) ---")
g_modes = ft_vac(g_theta)
println("Transformed $mthvac theta-space values to $mpert_vac complex modes")

# Inverse transform back
g_reconstructed = inverse(ft_vac, g_modes)
println("\n--- Inverse Transform ---")
println("Reconstruction error: $(maximum(abs.(real.(g_reconstructed) .- g_theta)))")

# ==============================================================================
# Example 3: Multiple Functions Simultaneously
# ==============================================================================
println("\n" * "="^70)
println("Example 3: Batch Transform (Multiple Functions)")
println("="^70)

# Create matrix of multiple theta-space functions
n_funcs = 10
data_matrix = randn(mtheta, n_funcs)

println("Transforming $n_funcs functions simultaneously")
println("Input shape: ($(size(data_matrix, 1)), $(size(data_matrix, 2)))")

# Forward transform all at once
modes_matrix = ft_pe(data_matrix)
println("Output shape: ($(size(modes_matrix, 1)), $(size(modes_matrix, 2)))")
println("  → Each column is one transformed function")

# Inverse transform
reconstructed_matrix = inverse(ft_pe, modes_matrix)
max_error = maximum(abs.(real.(reconstructed_matrix) .- data_matrix))
println("\nMax reconstruction error across all functions: $max_error")

# ==============================================================================
# Example 4: Replacing FFTW in existing code
# ==============================================================================
println("\n" * "="^70)
println("Example 4: Migration from FFTW")
println("="^70)

using FFTW

# Old approach: using FFTW
function theta_to_modes_fftw(theta_values::Vector{Float64}, mpert::Int)
    mtheta = length(theta_values)
    fft_result = fft(theta_values)
    modes = zeros(ComplexF64, mpert)
    for i in 1:min(mpert, length(fft_result))
        modes[i] = fft_result[i] / mtheta
    end
    return modes
end

# New approach: using FourierTransform
# (already shown above: modes = ft_pe(theta_values))

# Compare approaches
test_data = randn(mtheta)
modes_fft = theta_to_modes_fftw(test_data, mpert)
modes_custom = ft_pe(test_data)

println("Comparison of FFTW vs FourierTransform:")
println("  Both methods transform $mtheta theta points to $mpert modes")
println("  Results differ because:")
println("    - FFTW uses exp(i*k*θ) basis (standard FFT)")
println("    - FourierTransform uses cos(m*θ) + i*sin(m*θ) basis")
println("    - FourierTransform allows custom mode range (m = $mlow:$(mlow+mpert-1))")
println("    - FourierTransform supports phase shifts for vacuum calculations")

# ==============================================================================
# Summary
# ==============================================================================
println("\n" * "="^70)
println("Summary: When to Use FourierTransform")
println("="^70)
println("""
1. Pre-compute transforms for a specific grid:
   ft = FourierTransform(mtheta, mpert, mlow)

2. For PerturbedEquilibrium (no phase shift):
   ft = FourierTransform(mtheta, mpert, mlow)
   modes = ft(theta_data)

3. For Vacuum (with toroidal coupling):
   ft = FourierTransform(mthvac, mpert, mlow; n=n, qa=qa, delta=delta)
   modes = ft(green_function_data)

4. Avoid FFTW when:
   - You need custom mode ranges (not 0:N-1)
   - You need phase-shifted basis functions
   - You want to pre-compute and reuse coefficients
   - You're matching Vacuum module conventions

5. Benefits:
   - Type-stable, efficient functor pattern
   - Pre-computed basis functions (no repeated trig calculations)
   - Direct complex number support
   - Consistent interface across modules
""")

println("="^70)
