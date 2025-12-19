"""
Wall Geometry Comparison Script

This script creates detailed comparisons of the plasma and wall geometry used internally
within compute_vacuum_response. It examines:
1. Plasma (x,z) coordinates at each theta index
2. Wall (x,z) coordinates at each theta index
3. Geometry before and after equal_arc distribution
4. Relative differences between implementations

This addresses issue #107 item 1.
"""

using JLD2
using Pkg
using Plots
using LinearAlgebra
using Printf

Pkg.activate("$(@__DIR__)/../.."); using JPEC

@load "$(@__DIR__)/../../examples/DIIID-like_ideal_example/benchmark_inputs.jld2" benchmark_inputs

(; vac_inputs, wall_settings) = benchmark_inputs

println("="^80)
println("WALL GEOMETRY COMPARISON")
println("="^80)

# Test both with and without equal_arc enabled
test_configs = [
    (name="No Equal Arc", equal_arc=false),
    (name="With Equal Arc", equal_arc=true)
]

for config in test_configs
    println("\n" * "="^80)
    println("Configuration: $(config.name)")
    println("="^80)

    # Modify wall settings to control equal_arc feature
    wall_settings_test = JPEC.Vacuum.WallShapeSettings(
        shape=wall_settings.shape,
        a=wall_settings.a,
        equal_arc_wall=config.equal_arc
    )

    # Initialize plasma surface
    plasma_surf = JPEC.Vacuum.initialize_plasma_surface(vac_inputs)

    # Initialize wall
    wall = JPEC.Vacuum.initialize_wall(vac_inputs, plasma_surf, wall_settings_test)

    println("\nPlasma Surface:")
    println("  Number of points: $(length(plasma_surf.x))")
    println("  R range: [$(minimum(plasma_surf.x)), $(maximum(plasma_surf.x))]")
    println("  Z range: [$(minimum(plasma_surf.z)), $(maximum(plasma_surf.z))]")

    if !wall.nowall
        println("\nWall Geometry:")
        println("  Number of points: $(length(wall.x))")
        println("  R range: [$(minimum(wall.x)), $(maximum(wall.x))]")
        println("  Z range: [$(minimum(wall.z)), $(maximum(wall.z))]")
        println("  Is closed toroidal: $(wall.is_closed_toroidal)")

        # Calculate wall-plasma separation
        separations = sqrt.((wall.x .- plasma_surf.x).^2 .+ (wall.z .- plasma_surf.z).^2)
        println("  Wall-plasma separation: min=$(minimum(separations)), max=$(maximum(separations)), mean=$(sum(separations)/length(separations))")

        # Create detailed plots
        theta_indices = 1:length(plasma_surf.x)

        # Plot 1: R coordinates comparison
        p1 = plot(theta_indices, plasma_surf.x,
                 label="Plasma R",
                 marker=:circle,
                 markersize=3,
                 xlabel="Theta Index",
                 ylabel="R [m]",
                 title="$(config.name): R Coordinates",
                 legend=:best,
                 linewidth=2)
        plot!(p1, theta_indices, wall.x,
             label="Wall R",
             marker=:square,
             markersize=3,
             linewidth=2)

        # Plot 2: Z coordinates comparison
        p2 = plot(theta_indices, plasma_surf.z,
                 label="Plasma Z",
                 marker=:circle,
                 markersize=3,
                 xlabel="Theta Index",
                 ylabel="Z [m]",
                 title="$(config.name): Z Coordinates",
                 legend=:best,
                 linewidth=2)
        plot!(p2, theta_indices, wall.z,
             label="Wall Z",
             marker=:square,
             markersize=3,
             linewidth=2)

        # Plot 3: Wall-plasma separation
        p3 = plot(theta_indices, separations,
                 label="Wall-Plasma Distance",
                 marker=:circle,
                 markersize=3,
                 xlabel="Theta Index",
                 ylabel="Distance [m]",
                 title="$(config.name): Wall-Plasma Separation",
                 legend=:best,
                 linewidth=2)

        # Plot 4: R-Z plane view
        p4 = plot(plasma_surf.x, plasma_surf.z,
                 label="Plasma",
                 marker=:circle,
                 markersize=3,
                 xlabel="R [m]",
                 ylabel="Z [m]",
                 title="$(config.name): Geometry in R-Z Plane",
                 legend=:best,
                 aspect_ratio=:equal,
                 linewidth=2)
        plot!(p4, wall.x, wall.z,
             label="Wall",
             marker=:square,
             markersize=3,
             linewidth=2)

        # Combine plots
        p_combined = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 1000))

        # Save figure
        filename = config.equal_arc ? "wall_geometry_with_equal_arc.pdf" : "wall_geometry_no_equal_arc.pdf"
        filepath = joinpath(@__DIR__, filename)
        savefig(p_combined, filepath)
        @info "Saved geometry comparison to $filepath"

        # Create a detailed difference plot
        p_diff = plot(layout=(2,1), size=(800, 800))

        plot!(p_diff[1], theta_indices, wall.x .- plasma_surf.x,
             label="ΔR = Wall_R - Plasma_R",
             marker=:circle,
             markersize=3,
             ylabel="ΔR [m]",
             title="$(config.name): Wall-Plasma R Differences",
             linewidth=2,
             legend=:best)

        plot!(p_diff[2], theta_indices, wall.z .- plasma_surf.z,
             label="ΔZ = Wall_Z - Plasma_Z",
             marker=:circle,
             markersize=3,
             xlabel="Theta Index",
             ylabel="ΔZ [m]",
             title="$(config.name): Wall-Plasma Z Differences",
             linewidth=2,
             legend=:best)

        diff_filename = config.equal_arc ? "wall_plasma_differences_with_equal_arc.pdf" : "wall_plasma_differences_no_equal_arc.pdf"
        diff_filepath = joinpath(@__DIR__, diff_filename)
        savefig(p_diff, diff_filepath)
        @info "Saved difference plot to $diff_filepath"

        # Print detailed statistics
        println("\nDetailed Statistics:")
        println("  ΔR: min=$(minimum(wall.x .- plasma_surf.x)), max=$(maximum(wall.x .- plasma_surf.x)), std=$(std(wall.x .- plasma_surf.x))")
        println("  ΔZ: min=$(minimum(wall.z .- plasma_surf.z)), max=$(maximum(wall.z .- plasma_surf.z)), std=$(std(wall.z .- plasma_surf.z))")
    else
        println("\nNo wall geometry (nowall=true)")
    end
end

# Now test the equal_arc function directly to understand its behavior
println("\n" * "="^80)
println("EQUAL ARC GRID DISTRIBUTION TEST")
println("="^80)

# Create a simple test case - a circle
n_points = 64
theta_test = range(0, 2π, length=n_points+1)[1:end-1]
r_test = 1.5
x_circ = r_test .* cos.(theta_test)
z_circ = r_test .* sin.(theta_test)

println("\nTesting equal_arc_grid distribution on a perfect circle...")
println("  Input: $(length(x_circ)) points, radius = $r_test")

x_out, z_out, ell, thgr, thlag = JPEC.Vacuum.distribute_to_equal_arc_grid(x_circ, z_circ, n_points)

println("  Output: $(length(x_out)) points")
println("  Max position error: $(maximum(abs.([x_out .- x_circ; z_out .- z_circ])))")

# Check arc length spacing
arc_spacings = diff(ell)
println("  Arc length spacings: min=$(minimum(arc_spacings)), max=$(maximum(arc_spacings)), std=$(std(arc_spacings))")

# For equal arc, final spacings should be uniform
total_arc = ell[end]
expected_spacing = total_arc / (n_points - 1)
println("  Expected uniform spacing: $expected_spacing")
println("  Actual spacing variation: $(std(arc_spacings)) (should be small)")

println("\n" * "="^80)
println("GEOMETRY COMPARISON COMPLETE")
println("="^80)
