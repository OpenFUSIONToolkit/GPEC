"""
Test script to verify conformal wall normal vector calculation

This script checks whether the normal vector calculation for the conformal wall
is correctly implemented. It creates a simple test case (circle) where we can
analytically verify the normal vectors.
"""

using LinearAlgebra
using Printf

println("="^80)
println("TESTING CONFORMAL WALL NORMAL VECTOR CALCULATION")
println("="^80)

# Create a simple circular plasma boundary for testing
mtheta = 64
r_major = 1.7
r_minor = 0.3

theta = range(0, 2π, length=mtheta+1)[1:end-1]
x_plasma = r_major .+ r_minor .* cos.(theta)
z_plasma = r_minor .* sin.(theta)

println("\nTest case: Circular plasma")
println("  Major radius R0 = $r_major m")
println("  Minor radius a = $r_minor m")
println("  Number of points = $mtheta")

# Wall offset parameter
a = 0.5  # Conformal offset in units of minor radius

println("\nWall offset parameter a = $a")
println("  Physical offset distance = $(a * r_minor) m")

# Implement the conformal wall calculation from VacuumStructs.jl
x_wall = zeros(mtheta)
z_wall = zeros(mtheta)

csmin = min(0.1, 0.1 * minimum(x_plasma))

for i in 1:mtheta
    j = (i == 1) ? mtheta : i - 1
    k = (i == mtheta) ? 1 : i + 1

    # Normal vector calculation (from VacuumStructs.jl line 288-290)
    alph = atan(x_plasma[k] - x_plasma[j], z_plasma[j] - z_plasma[k])
    x_wall[i] = max(csmin, x_plasma[i] + a * r_minor * cos(alph))
    z_wall[i] = z_plasma[i] + a * r_minor * sin(alph)
end

# For a circle, the analytical outward normal at angle θ is (cos(θ), sin(θ))
# So the wall should be at radius (r_major + r_minor*(1+a)) with the same angles

# Analytical solution for circular plasma
x_wall_analytical = r_major .+ r_minor .* (1 + a) .* cos.(theta)
z_wall_analytical = r_minor .* (1 + a) .* sin.(theta)

# Compute differences
dx = x_wall .- x_wall_analytical
dz = z_wall .- z_wall_analytical
distance_error = sqrt.(dx.^2 .+ dz.^2)

println("\n" * "="^80)
println("RESULTS")
println("="^80)

println("\nNumerical vs Analytical comparison:")
println("  Max position error: $(maximum(distance_error)) m")
println("  Mean position error: $(sum(distance_error)/length(distance_error)) m")
println("  RMS position error: $(sqrt(sum(distance_error.^2)/length(distance_error))) m")

# Check if errors are reasonable (should be small for a circle)
if maximum(distance_error) < 1e-6
    println("\n✓ PASS: Conformal wall normal calculation is correct for circular plasma")
else
    println("\n✗ FAIL: Conformal wall normal calculation has significant errors")
    println("\nDetailed error analysis:")

    # Find the worst points
    max_error_idx = argmax(distance_error)
    println("\nWorst error at index $max_error_idx:")
    println("  θ = $(theta[max_error_idx])")
    println("  Numerical: ($(x_wall[max_error_idx]), $(z_wall[max_error_idx]))")
    println("  Analytical: ($(x_wall_analytical[max_error_idx]), $(z_wall_analytical[max_error_idx]))")
    println("  Error: $(distance_error[max_error_idx]) m")

    # Print first few points for debugging
    println("\nFirst 5 points:")
    for i in 1:5
        @printf("  i=%2d: θ=%.4f, x_num=%.6f, x_ana=%.6f, dx=%.2e, z_num=%.6f, z_ana=%.6f, dz=%.2e\n",
                i, theta[i], x_wall[i], x_wall_analytical[i], dx[i], z_wall[i], z_wall_analytical[i], dz[i])
    end
end

# Additional check: verify the normal vector direction
println("\n" * "="^80)
println("NORMAL VECTOR DIRECTION CHECK")
println("="^80)

# For each point, compute the numerical normal and compare to analytical
for i in [1, mtheta÷4, mtheta÷2, 3*mtheta÷4]
    # Numerical normal (from the code)
    j = (i == 1) ? mtheta : i - 1
    k = (i == mtheta) ? 1 : i + 1
    alph = atan(x_plasma[k] - x_plasma[j], z_plasma[j] - z_plasma[k])
    n_num = [cos(alph), sin(alph)]

    # Analytical normal for a circle at angle theta[i]
    n_ana = [cos(theta[i]), sin(theta[i])]

    # Dot product should be ~1 if parallel, ~-1 if antiparallel
    dot_product = dot(n_num, n_ana)

    @printf("\nPoint i=%2d (θ=%.4f):\n", i, theta[i])
    @printf("  Numerical normal: [%.6f, %.6f]\n", n_num[1], n_num[2])
    @printf("  Analytical normal: [%.6f, %.6f]\n", n_ana[1], n_ana[2])
    @printf("  Dot product: %.6f ", dot_product)

    if dot_product > 0.99
        println("(✓ parallel, outward)")
    elseif dot_product < -0.99
        println("(✗ antiparallel, inward!)")
    else
        println("(✗ not aligned!)")
    end
end

println("\n" * "="^80)
println("TEST COMPLETE")
println("="^80)
