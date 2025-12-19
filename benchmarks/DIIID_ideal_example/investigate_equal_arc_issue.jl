"""
Investigate Equal Arc Potential Bug

This script investigates a potential bug: if equal_arc_wall is applied AFTER
the conformal wall calculation (which uses normal vectors computed from the
original plasma surface), the wall points are redistributed but the offset
distance may not be uniform anymore.

Hypothesis: The conformal wall is computed by offsetting points along the normal,
then equal_arc redistributes those points. This redistribution could cause the
wall-plasma separation to become non-uniform, leading to geometry errors.
"""

using LinearAlgebra
using Printf
using Plots

# Add JPEC to path
push!(LOAD_PATH, joinpath(@__DIR__, "../.."))
using JPEC

println("="^80)
println("INVESTIGATING EQUAL_ARC INTERACTION WITH CONFORMAL WALL")
println("="^80)

# Create a simple circular plasma for testing
mtheta = 128
r_major = 1.7
r_minor = 0.3
a = 0.5  # Wall offset parameter

theta = range(0, 2π, length=mtheta+1)[1:end-1]
x_plasma = r_major .+ r_minor .* cos.(theta)
z_plasma = r_minor .* sin.(theta)

println("\nTest Setup:")
println("  Plasma: Circle with R0=$r_major, a=$r_minor")
println("  Wall offset parameter: a=$a")
println("  Number of points: $mtheta")

# Manually compute conformal wall (mimicking VacuumStructs.jl logic)
println("\n" * "="^80)
println("STEP 1: Compute Conformal Wall (Before Equal Arc)")
println("="^80)

x_wall_before = zeros(mtheta)
z_wall_before = zeros(mtheta)

csmin = min(0.1, 0.1 * minimum(x_plasma))

for i in 1:mtheta
    j = (i == 1) ? mtheta : i - 1
    k = (i == mtheta) ? 1 : i + 1

    # Normal vector calculation
    alph = atan(x_plasma[k] - x_plasma[j], z_plasma[j] - z_plasma[k])
    x_wall_before[i] = max(csmin, x_plasma[i] + a * r_minor * cos(alph))
    z_wall_before[i] = z_plasma[i] + a * r_minor * sin(alph)
end

# Check wall-plasma separation before equal_arc
separations_before = sqrt.((x_wall_before .- x_plasma).^2 .+ (z_wall_before .- z_plasma).^2)

println("Wall-Plasma Separation BEFORE Equal Arc:")
println("  Mean: $(sum(separations_before)/length(separations_before))")
println("  Std: $(std(separations_before))")
println("  Min: $(minimum(separations_before)), Max: $(maximum(separations_before))")
println("  Range/Mean: $((maximum(separations_before) - minimum(separations_before)) / (sum(separations_before)/length(separations_before)))")

# Now apply equal_arc distribution
println("\n" * "="^80)
println("STEP 2: Apply Equal Arc Distribution")
println("="^80)

x_wall_after, z_wall_after, ell, thgr, thlag = JPEC.Vacuum.distribute_to_equal_arc_grid(
    x_wall_before, z_wall_before, mtheta
)

# Check wall-plasma separation after equal_arc
# NOTE: We need to also redistribute plasma points to match, or compare point-by-point will be wrong
# Let's compare the minimum distance from each wall point to ANY plasma point
separations_after = zeros(mtheta)
for i in 1:mtheta
    min_dist = Inf
    for j in 1:mtheta
        dist = sqrt((x_wall_after[i] - x_plasma[j])^2 + (z_wall_after[i] - z_plasma[j])^2)
        min_dist = min(min_dist, dist)
    end
    separations_after[i] = min_dist
end

println("Wall-Plasma Separation AFTER Equal Arc (min distance to any plasma point):")
println("  Mean: $(sum(separations_after)/length(separations_after))")
println("  Std: $(std(separations_after))")
println("  Min: $(minimum(separations_after)), Max: $(maximum(separations_after))")
println("  Range/Mean: $((maximum(separations_after) - minimum(separations_after)) / (sum(separations_after)/length(separations_after)))")

# Compare point-by-point (assuming theta alignment)
println("\n" * "="^80)
println("STEP 3: Point-by-Point Comparison")
println("="^80)

# For a circle, we can compute analytical separations
expected_separation = a * r_minor

println("Expected separation (analytical): $expected_separation")
println("\nBefore equal_arc:")
println("  Error from expected: mean=$(abs(sum(separations_before)/length(separations_before) - expected_separation)), max=$(maximum(abs.(separations_before .- expected_separation)))")
println("\nAfter equal_arc:")
println("  Error from expected: mean=$(abs(sum(separations_after)/length(separations_after) - expected_separation)), max=$(maximum(abs.(separations_after .- expected_separation)))")

# Key question: Are the wall points still at uniform distance from plasma?
println("\n" * "="^80)
println("ANALYSIS")
println("="^80)

if std(separations_before) / (sum(separations_before)/length(separations_before)) < 0.01
    println("✓ BEFORE equal_arc: Separations are uniform (< 1% variation)")
else
    println("✗ BEFORE equal_arc: Separations show variation > 1%")
end

if std(separations_after) / (sum(separations_after)/length(separations_after)) < 0.01
    println("✓ AFTER equal_arc: Separations are uniform (< 1% variation)")
else
    println("✗ AFTER equal_arc: Separations show variation > 1%")
    println("  This indicates equal_arc may be introducing geometry errors!")
end

# Visualization
println("\n" * "="^80)
println("CREATING VISUALIZATION")
println("="^80)

p = plot(layout=(2,2), size=(1200, 1000))

# Plot 1: Geometry before and after
plot!(p[1], x_plasma, z_plasma,
     label="Plasma",
     linewidth=2,
     aspect_ratio=:equal,
     title="Geometry Comparison",
     xlabel="R [m]",
     ylabel="Z [m]",
     color=:black)
plot!(p[1], x_wall_before, z_wall_before,
     label="Wall (before equal_arc)",
     linewidth=2,
     linestyle=:dash,
     color=:blue)
plot!(p[1], x_wall_after, z_wall_after,
     label="Wall (after equal_arc)",
     linewidth=2,
     linestyle=:dot,
     color=:red)

# Plot 2: Separations
plot!(p[2], 1:mtheta, separations_before,
     label="Before equal_arc",
     marker=:circle,
     markersize=2,
     title="Wall-Plasma Separation",
     xlabel="Point index",
     ylabel="Separation [m]",
     linewidth=2)
plot!(p[2], 1:mtheta, separations_after,
     label="After equal_arc",
     marker=:square,
     markersize=2,
     linewidth=2)
hline!(p[2], [expected_separation], label="Expected", linestyle=:dash, color=:black)

# Plot 3: Position differences
dx = x_wall_after .- x_wall_before
dz = z_wall_after .- z_wall_before
dr = sqrt.(dx.^2 .+ dz.^2)

plot!(p[3], 1:mtheta, dr,
     label="Position change",
     marker=:circle,
     markersize=3,
     title="Wall Point Movement from Equal Arc",
     xlabel="Point index",
     ylabel="Distance moved [m]",
     linewidth=2)

# Plot 4: Separation error
error_before = separations_before .- expected_separation
error_after = separations_after .- expected_separation

plot!(p[4], 1:mtheta, error_before,
     label="Before equal_arc",
     marker=:circle,
     markersize=2,
     title="Separation Error from Expected",
     xlabel="Point index",
     ylabel="Error [m]",
     linewidth=2)
plot!(p[4], 1:mtheta, error_after,
     label="After equal_arc",
     marker=:square,
     markersize=2,
     linewidth=2)
hline!(p[4], [0], linestyle=:dash, color=:black)

filepath = joinpath(@__DIR__, "equal_arc_conformal_wall_investigation.pdf")
savefig(p, filepath)
@info "Saved visualization to $filepath"

println("\n" * "="^80)
println("INVESTIGATION COMPLETE")
println("="^80)

# Print conclusion
println("\nConclusion:")
if maximum(abs.(error_after)) > 2 * maximum(abs.(error_before))
    println("  ⚠ WARNING: Equal arc distribution significantly increases separation errors!")
    println("  This could explain the discrepancies in wall calculations.")
    println("  Recommendation: Consider applying equal_arc to BOTH plasma and wall,")
    println("  or compute conformal offset AFTER equal_arc distribution of plasma.")
else
    println("  Equal arc distribution does not significantly affect wall-plasma separation.")
    println("  The geometry discrepancy likely has a different cause.")
end
