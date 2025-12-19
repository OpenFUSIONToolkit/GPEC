"""
Test Equal Arc Distribution

This script tests the distribute_to_equal_arc_grid function to ensure it's working
correctly and consistently. This addresses potential bugs in the equal arc logic
mentioned in issue #107.
"""

using Printf
using LinearAlgebra
using Plots

# Need to add the parent directory to LOAD_PATH to access JPEC
push!(LOAD_PATH, joinpath(@__DIR__, "../.."))
using JPEC

println("="^80)
println("TESTING EQUAL ARC DISTRIBUTION")
println("="^80)

# Test 1: Perfect circle - arc lengths should be exactly equal after distribution
println("\nTest 1: Perfect Circle")
println("-"^80)

n_points = 128
theta = range(0, 2π, length=n_points+1)[1:end-1]
r = 1.5

# Create non-uniformly spaced points on the circle (clustered near theta=0)
theta_nonuniform = @. 2π * ((1:n_points) / n_points)^1.5
x_in = @. r * cos(theta_nonuniform)
z_in = @. r * sin(theta_nonuniform)

println("Input: $(length(x_in)) points (non-uniformly distributed)")

# Apply equal arc distribution
x_out, z_out, ell, thgr, thlag = JPEC.Vacuum.distribute_to_equal_arc_grid(x_in, z_in, n_points)

println("Output: $(length(x_out)) points")

# Check that points remain on the circle
radii_out = @. sqrt(x_out^2 + z_out^2)
radius_error = maximum(abs.(radii_out .- r))
println("  Max radius error: $(radius_error) (should be ~0)")

# Check arc length spacing
arc_spacings = diff(ell)
total_arc = ell[end]
expected_spacing = total_arc / (n_points - 1)
spacing_std = std(arc_spacings)
println("  Total arc length: $(total_arc) (analytical: $(2π*r) = $(2π*r))")
println("  Expected uniform spacing: $(expected_spacing)")
println("  Actual spacing std dev: $(spacing_std) (should be small)")
println("  Max spacing: $(maximum(arc_spacings)), Min spacing: $(minimum(arc_spacings))")

if spacing_std < 1e-6 && radius_error < 1e-6
    println("  ✓ PASS: Equal arc distribution working correctly for circle")
else
    println("  ✗ FAIL: Equal arc distribution has issues")
end

# Test 2: Ellipse - more realistic plasma shape
println("\nTest 2: Ellipse (More Realistic)")
println("-"^80)

r_major = 1.7
r_minor = 0.5
theta_ell = range(0, 2π, length=n_points+1)[1:end-1]
x_ell_in = @. r_major + r_minor * cos(theta_ell)
z_ell_in = @. r_minor * sin(theta_ell)

println("Input: Ellipse with R0=$r_major, a=$r_minor")

x_ell_out, z_ell_out, ell_ell, thgr_ell, thlag_ell = JPEC.Vacuum.distribute_to_equal_arc_grid(
    x_ell_in, z_ell_in, n_points
)

# Calculate arc lengths between consecutive output points
arc_lengths_out = zeros(n_points)
for i in 1:n_points
    j = (i == n_points) ? 1 : i + 1
    arc_lengths_out[i] = sqrt((x_ell_out[j] - x_ell_out[i])^2 + (z_ell_out[j] - z_ell_out[i])^2)
end

println("  Arc length between consecutive points:")
println("    Mean: $(sum(arc_lengths_out)/length(arc_lengths_out))")
println("    Std dev: $(std(arc_lengths_out))")
println("    Max: $(maximum(arc_lengths_out)), Min: $(minimum(arc_lengths_out))")
println("    Ratio (max/min): $(maximum(arc_lengths_out)/minimum(arc_lengths_out)) (should be ~1)")

if std(arc_lengths_out) / (sum(arc_lengths_out)/length(arc_lengths_out)) < 0.01
    println("  ✓ PASS: Arc lengths are uniform (within 1%)")
else
    println("  ✗ WARNING: Arc lengths show variation > 1%")
end

# Test 3: D-shaped plasma (very asymmetric)
println("\nTest 3: D-Shaped Plasma (Asymmetric)")
println("-"^80)

theta_d = range(0, 2π, length=n_points+1)[1:end-1]
# D-shape: elongated and tilted
x_d_in = @. r_major + r_minor * cos(theta_d + 0.3 * sin(theta_d))
z_d_in = @. 1.5 * r_minor * sin(theta_d)

println("Input: D-shaped plasma")

x_d_out, z_d_out, ell_d, thgr_d, thlag_d = JPEC.Vacuum.distribute_to_equal_arc_grid(
    x_d_in, z_d_in, n_points
)

# Calculate arc lengths
arc_lengths_d = zeros(n_points)
for i in 1:n_points
    j = (i == n_points) ? 1 : i + 1
    arc_lengths_d[i] = sqrt((x_d_out[j] - x_d_out[i])^2 + (z_d_out[j] - z_d_out[i])^2)
end

println("  Arc length between consecutive points:")
println("    Mean: $(sum(arc_lengths_d)/length(arc_lengths_d))")
println("    Std dev: $(std(arc_lengths_d))")
println("    Ratio (max/min): $(maximum(arc_lengths_d)/minimum(arc_lengths_d))")

# Visualization
println("\n" * "="^80)
println("CREATING VISUALIZATION")
println("="^80)

p = plot(layout=(3,2), size=(1200, 1200), margin=5Plots.mm)

# Plot 1: Circle input vs output
plot!(p[1], x_in, z_in,
     label="Input (non-uniform θ)",
     marker=:circle,
     markersize=2,
     aspect_ratio=:equal,
     title="Circle: Input vs Output",
     xlabel="R [m]",
     ylabel="Z [m]")
plot!(p[1], x_out, z_out,
     label="Output (equal arc)",
     marker=:square,
     markersize=2)

# Plot 2: Arc spacing for circle
plot!(p[2], 1:n_points-1, arc_spacings,
     label="Arc spacing",
     marker=:circle,
     markersize=3,
     title="Circle: Input Arc Spacing",
     xlabel="Segment index",
     ylabel="Arc length [m]")
hline!(p[2], [expected_spacing], label="Expected uniform", linestyle=:dash)

# Plot 3: Ellipse input vs output
plot!(p[3], x_ell_in, z_ell_in,
     label="Input (uniform θ)",
     marker=:circle,
     markersize=2,
     aspect_ratio=:equal,
     title="Ellipse: Input vs Output",
     xlabel="R [m]",
     ylabel="Z [m]")
plot!(p[3], x_ell_out, z_ell_out,
     label="Output (equal arc)",
     marker=:square,
     markersize=2)

# Plot 4: Arc spacing for ellipse
plot!(p[4], 1:n_points, arc_lengths_out,
     label="Arc spacing",
     marker=:circle,
     markersize=3,
     title="Ellipse: Output Arc Spacing",
     xlabel="Segment index",
     ylabel="Arc length [m]")
hline!(p[4], [sum(arc_lengths_out)/length(arc_lengths_out)], label="Mean", linestyle=:dash)

# Plot 5: D-shape input vs output
plot!(p[5], x_d_in, z_d_in,
     label="Input (uniform θ)",
     marker=:circle,
     markersize=2,
     aspect_ratio=:equal,
     title="D-Shape: Input vs Output",
     xlabel="R [m]",
     ylabel="Z [m]")
plot!(p[5], x_d_out, z_d_out,
     label="Output (equal arc)",
     marker=:square,
     markersize=2)

# Plot 6: Arc spacing for D-shape
plot!(p[6], 1:n_points, arc_lengths_d,
     label="Arc spacing",
     marker=:circle,
     markersize=3,
     title="D-Shape: Output Arc Spacing",
     xlabel="Segment index",
     ylabel="Arc length [m]")
hline!(p[6], [sum(arc_lengths_d)/length(arc_lengths_d)], label="Mean", linestyle=:dash)

filepath = joinpath(@__DIR__, "equal_arc_distribution_test.pdf")
savefig(p, filepath)
@info "Saved visualization to $filepath"

println("\n" * "="^80)
println("TEST COMPLETE")
println("="^80)
println("\nSummary:")
println("  - Circle test: radius error = $(radius_error), spacing std = $(spacing_std)")
println("  - Ellipse test: spacing std / mean = $(std(arc_lengths_out) / (sum(arc_lengths_out)/length(arc_lengths_out)))")
println("  - D-shape test: spacing max/min ratio = $(maximum(arc_lengths_d)/minimum(arc_lengths_d))")
println("\nVisualization saved to: $filepath")
