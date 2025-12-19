"""
Wall Offset Distance Scan

This script scans the conformal wall offset parameter 'a' to identify
systematic differences in vacuum response calculations.

This addresses issue #107 item 2.
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
println("WALL OFFSET SCAN")
println("="^80)

# Define scan parameters as suggested in issue #107
a_values = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]

# Storage for results
wv_results = []
wall_separations = []
wall_r_ranges = []
wall_z_ranges = []

println("\nScanning wall offset parameter 'a'...")
println("Note: Fortran uses nowall calculation for very large 'a' values\n")

for a_val in a_values
    println("="^80)
    println("Testing a = $a_val")
    println("="^80)

    # Create wall settings with current offset
    wall_settings_test = JPEC.Vacuum.WallShapeSettings(
        shape="conformal",
        a=a_val,
        equal_arc_wall=wall_settings.equal_arc_wall
    )

    # Initialize geometry
    plasma_surf = JPEC.Vacuum.initialize_plasma_surface(vac_inputs)
    wall = JPEC.Vacuum.initialize_wall(vac_inputs, plasma_surf, wall_settings_test)

    # Compute vacuum response
    wv, grri, xzpts = JPEC.Vacuum.compute_vacuum_response(vac_inputs, wall_settings_test)

    # Store results
    push!(wv_results, wv)

    # Calculate separation statistics
    separations = sqrt.((wall.x .- plasma_surf.x).^2 .+ (wall.z .- plasma_surf.z).^2)
    push!(wall_separations, separations)

    push!(wall_r_ranges, (minimum(wall.x), maximum(wall.x)))
    push!(wall_z_ranges, (minimum(wall.z), maximum(wall.z)))

    println("  Wall-plasma separation: min=$(minimum(separations)), max=$(maximum(separations)), mean=$(sum(separations)/length(separations))")
    println("  Wall R range: [$(minimum(wall.x)), $(maximum(wall.x))]")
    println("  Wall Z range: [$(minimum(wall.z)), $(maximum(wall.z))]")
    println("  Vacuum matrix norm: $(norm(wv))")
    println("  Vacuum matrix trace: $(tr(wv))")
    println()
end

println("="^80)
println("ANALYZING RESULTS")
println("="^80)

# Extract key metrics from vacuum matrices
wv_norms = [norm(wv) for wv in wv_results]
wv_traces = [abs(tr(wv)) for wv in wv_results]
wv_max_eigenvalues = [maximum(abs.(eigvals(wv))) for wv in wv_results]
mean_separations = [sum(sep)/length(sep) for sep in wall_separations]

# Print summary table
println("\nSummary Table:")
println("="^80)
@printf("%-10s %-15s %-15s %-15s %-15s\n", "a", "||wv||", "|tr(wv)|", "max|λ|", "mean sep")
println("-"^80)
for i in 1:length(a_values)
    @printf("%-10.2f %-15.6e %-15.6e %-15.6e %-15.6f\n",
            a_values[i], wv_norms[i], wv_traces[i], wv_max_eigenvalues[i], mean_separations[i])
end
println("="^80)

# Create plots
println("\nCreating plots...")

# Plot 1: Vacuum matrix metrics vs wall offset
p1 = plot(a_values, wv_norms,
         label="||wv|| (Frobenius norm)",
         marker=:circle,
         markersize=5,
         xlabel="Wall offset parameter 'a'",
         ylabel="Matrix Norm",
         title="Vacuum Matrix Norm vs Wall Offset",
         legend=:best,
         linewidth=2,
         xscale=:log10,
         yscale=:log10,
         minorgrid=true,
         framestyle=:box)

savefig(p1, joinpath(@__DIR__, "wall_offset_vacuum_norm.pdf"))
@info "Saved vacuum norm plot"

# Plot 2: Trace and max eigenvalue
p2 = plot(layout=(2,1), size=(800, 800))

plot!(p2[1], a_values, wv_traces,
     label="|tr(wv)|",
     marker=:circle,
     markersize=5,
     ylabel="Trace Magnitude",
     title="Vacuum Matrix Trace vs Wall Offset",
     linewidth=2,
     xscale=:log10,
     yscale=:log10,
     minorgrid=true,
     legend=:best)

plot!(p2[2], a_values, wv_max_eigenvalues,
     label="max|λ|",
     marker=:square,
     markersize=5,
     xlabel="Wall offset parameter 'a'",
     ylabel="Max Eigenvalue Magnitude",
     title="Vacuum Matrix Max Eigenvalue vs Wall Offset",
     linewidth=2,
     xscale=:log10,
     yscale=:log10,
     minorgrid=true,
     legend=:best)

savefig(p2, joinpath(@__DIR__, "wall_offset_vacuum_metrics.pdf"))
@info "Saved vacuum metrics plot"

# Plot 3: Mean wall-plasma separation
p3 = plot(a_values, mean_separations,
         label="Mean wall-plasma distance",
         marker=:circle,
         markersize=5,
         xlabel="Wall offset parameter 'a'",
         ylabel="Mean Separation [m]",
         title="Wall-Plasma Separation vs Offset Parameter",
         legend=:best,
         linewidth=2,
         xscale=:log10,
         minorgrid=true,
         framestyle=:box)

savefig(p3, joinpath(@__DIR__, "wall_offset_separation.pdf"))
@info "Saved separation plot"

# Plot 4: Individual matrix elements for smallest and largest 'a'
mpert = size(wv_results[1], 1)
p4 = plot(layout=(2,1), size=(800, 800))

# Real part
heatmap!(p4[1], real(wv_results[1]),
        title="Re(wv) for a=$(a_values[1])",
        xlabel="l'",
        ylabel="l",
        color=:RdBu,
        aspect_ratio=:equal)

heatmap!(p4[2], real(wv_results[end]),
        title="Re(wv) for a=$(a_values[end])",
        xlabel="l'",
        ylabel="l",
        color=:RdBu,
        aspect_ratio=:equal)

savefig(p4, joinpath(@__DIR__, "wall_offset_matrix_comparison.pdf"))
@info "Saved matrix comparison plot"

# Plot 5: Geometry evolution
p5 = plot(xlabel="R [m]", ylabel="Z [m]",
         title="Wall Geometry for Different Offsets",
         aspect_ratio=:equal,
         legend=:outerright,
         size=(1000, 600))

# Plot plasma once
plasma_surf = JPEC.Vacuum.initialize_plasma_surface(vac_inputs)
plot!(p5, plasma_surf.x, plasma_surf.z,
     label="Plasma",
     linewidth=3,
     color=:black)

# Plot walls for each 'a'
colors = palette(:tab10)
for (i, a_val) in enumerate(a_values)
    wall_settings_test = JPEC.Vacuum.WallShapeSettings(
        shape="conformal",
        a=a_val,
        equal_arc_wall=wall_settings.equal_arc_wall
    )
    wall = JPEC.Vacuum.initialize_wall(vac_inputs, plasma_surf, wall_settings_test)

    plot!(p5, wall.x, wall.z,
         label="a=$a_val",
         linewidth=2,
         color=colors[i],
         alpha=0.7)
end

savefig(p5, joinpath(@__DIR__, "wall_offset_geometry_evolution.pdf"))
@info "Saved geometry evolution plot"

println("\n" * "="^80)
println("WALL OFFSET SCAN COMPLETE")
println("="^80)
println("\nKey findings:")
println("  - Tested a values: $a_values")
println("  - Vacuum norm range: [$(minimum(wv_norms)), $(maximum(wv_norms))]")
println("  - Mean separation range: [$(minimum(mean_separations)), $(maximum(mean_separations))]")
println("\nPlots saved to benchmarks/DIIID_ideal_example/")
