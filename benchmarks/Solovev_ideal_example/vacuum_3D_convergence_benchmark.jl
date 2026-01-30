"""
Convergence benchmark for compute_vacuum_response_3D

This script tests how the wv matrix converges with grid resolution for the Solovev
equilibrium. It initializes the equilibrium from sol.toml and constructs vacuum
inputs following the logic in Free.jl.

Scans performed:
1. 3D convergence: mtheta varying with nzeta/mtheta = 1.5 fixed
2. 2D convergence: mtheta varying (standard 2D vacuum response)
3. 2D vs 3D comparison at matched mtheta values

Outputs:
- Convergence plots
- Runtime scaling data
- Relative error metrics vs. high-resolution reference
"""

using Pkg
Pkg.activate("$(@__DIR__)/../..")

using JPEC
using JPEC.Vacuum
using JPEC.Equilibrium
using JPEC: Spl
using LinearAlgebra
using Printf
using Plots
using TOML

default(; markersize=4, linewidth=2)

# ============================================================================
# Helper functions
# ============================================================================

"""
Compute vacuum inputs from equilibrium, following Free.jl logic.
"""
function compute_vacuum_inputs_from_equil(equil::PlasmaEquilibrium, mthvac::Int, mlow::Int, mpert::Int, n::Int)
    # Get boundary flux value (psilim = 1.0 for normalized coordinates)
    psilim = equil.config.control.psihigh

    # Allocate arrays
    mtheta_eq = equil.config.control.mtheta + 1
    R = zeros(Float64, mtheta_eq)
    Z = zeros(Float64, mtheta_eq)
    ν = zeros(Float64, mtheta_eq)
    r_minor = zeros(Float64, mtheta_eq)
    θ_SFL = zeros(Float64, mtheta_eq)

    # Compute geometric quantities on plasma boundary
    for (i, θ) in enumerate(equil.rzphi.ys)
        f = Spl.bicube_eval!(equil.rzphi, psilim, θ)
        r_minor[i] = sqrt(f[1])
        θ_SFL[i] = 2π * (θ + f[2])  # f[2] = λ(ψ, θ) / 2π
        ν[i] = f[3]
    end

    # Compute R and Z on straight-fieldline θ grid
    R .= equil.ro .+ r_minor .* cos.(θ_SFL)
    Z .= equil.zo .+ r_minor .* sin.(θ_SFL)

    return Vacuum.VacuumInput(;
        r=reverse(R),
        z=reverse(Z),
        ν=reverse(ν),
        mlow=mlow,
        mpert=mpert,
        n=n,
        mtheta=mthvac,
        force_wv_symmetry=true
    )
end

"""
Compute accuracy metrics between two wv matrices.
Returns: (relative_frobenius_norm, max_absolute_error, max_eigenvalue_error)
"""
function compute_accuracy_metrics(wv_test::Matrix, wv_ref::Matrix)
    diff = wv_test - wv_ref
    frobenius_norm = norm(diff)
    relative_frobenius = frobenius_norm / norm(wv_ref)
    max_abs_error = maximum(abs.(diff))

    # Eigenvalue comparison
    ev_test = maximum(real.(eigvals(wv_test)))
    ev_ref = maximum(real.(eigvals(wv_ref)))
    ev_rel_error = abs(ev_test - ev_ref) / abs(ev_ref)

    return relative_frobenius, max_abs_error, ev_rel_error
end

# ============================================================================
# Setup equilibrium and parameters
# ============================================================================

println("="^70)
println("3D VACUUM RESPONSE CONVERGENCE BENCHMARK")
println("="^70)

# Load equilibrium from Solovev example
example_dir = joinpath(@__DIR__, "../../examples/Solovev_ideal_example")
equil = Equilibrium.setup_equilibrium(joinpath(example_dir, "equil.toml"))

# Load DCON settings to get mode numbers
dcon_inputs = TOML.parsefile(joinpath(example_dir, "dcon.toml"))
wall_inputs = dcon_inputs["WALL"]
wall_settings = Vacuum.WallShapeSettings(; (Symbol(k) => v for (k, v) in wall_inputs)...)

# Determine mode numbers (following Main.jl logic)
nlow = dcon_inputs["DCON_CONTROL"]["nn_low"]
nhigh = dcon_inputs["DCON_CONTROL"]["nn_high"]
npert = nhigh - nlow + 1

# Poloidal mode range (simplified version of Main.jl logic)
mlow = trunc(Int, min(nlow * equil.params.qmin, 0)) - 4
mhigh = trunc(Int, nhigh * equil.params.qmax)
mpert = mhigh - mlow + 1

println("\nEquilibrium loaded from: $example_dir")
println("  Major radius R0 = $(equil.ro)")
println("  q on axis       = $(equil.params.qmin)")
println("  q at edge       = $(equil.params.qmax)")
println("\nMode numbers:")
println("  n range: $nlow to $nhigh (npert = $npert)")
println("  m range: $mlow to $mhigh (mpert = $mpert)")
println("  Total modes: $(mpert * npert)")

# ============================================================================
# Convergence scan parameters
# ============================================================================

# Fixed aspect ratio for 3D: nzeta/mtheta = 1.5
aspect_ratio_grid = 1.5

# Mtheta values chosen to give total grid points from ~300 to ~30000
# Total points = mtheta * nzeta = mtheta * (1.5 * mtheta) = 1.5 * mtheta^2
# mtheta=14 -> 294 points, mtheta=140 -> 29400 points
mtheta_values = [14, 20, 28, 40, 56, 80]
nzeta_values = [round(Int, aspect_ratio_grid * m) for m in mtheta_values]
total_points = [m * n for (m, n) in zip(mtheta_values, nzeta_values)]

# Reference resolution (high resolution)
mtheta_ref = 112
nzeta_ref = round(Int, aspect_ratio_grid * mtheta_ref)

println("\n" * "="^70)
println("SCAN PARAMETERS")
println("="^70)
println("  Grid aspect ratio (nzeta/mtheta): $aspect_ratio_grid")
println("  Mtheta values: $mtheta_values")
println("  Nzeta values:  $nzeta_values")
println("  Total points:  $total_points")
println("  Reference: mtheta=$mtheta_ref, nzeta=$nzeta_ref ($(mtheta_ref * nzeta_ref) points)")

# ============================================================================
# Compute reference solutions (3D and 2D)
# ============================================================================

println("\n" * "="^70)
println("COMPUTING REFERENCE SOLUTIONS")
println("="^70)

# 3D reference
println("\n3D Reference (mtheta=$mtheta_ref, nzeta=$nzeta_ref)...")
vac_inputs_2D_ref = compute_vacuum_inputs_from_equil(equil, mtheta_ref, mlow, mpert, nlow)
vac_inputs_3D_ref = Vacuum.VacuumInput3D(vac_inputs_2D_ref, nzeta_ref, nlow, npert)

t_ref_3D = @elapsed begin
    wv_ref_3D, _, _, _ = Vacuum.compute_vacuum_response_3D(vac_inputs_3D_ref, wall_settings)
end
println("  Time: $(@sprintf "%.2f" t_ref_3D) s")
println("  wv matrix size: $(size(wv_ref_3D))")
println("  Max eigenvalue: $(@sprintf "%.6e" maximum(real.(eigvals(wv_ref_3D))))")

# 2D reference (use same mtheta as 3D reference)
println("\n2D Reference (mtheta=$mtheta_ref)...")
t_ref_2D = @elapsed begin
    wv_ref_2D, _, _ = Vacuum.compute_vacuum_response(vac_inputs_2D_ref, wall_settings)
end
println("  Time: $(@sprintf "%.2f" t_ref_2D) s")
println("  wv matrix size: $(size(wv_ref_2D))")
println("  Max eigenvalue: $(@sprintf "%.6e" maximum(real.(eigvals(wv_ref_2D))))")

# ============================================================================
# 3D Convergence scan (fixed aspect ratio)
# ============================================================================

println("\n" * "="^70)
println("3D CONVERGENCE SCAN (nzeta/mtheta = $aspect_ratio_grid)")
println("="^70)

rel_errors_3D = Float64[]
ev_errors_3D = Float64[]
times_3D = Float64[]

for (i, mth) in enumerate(mtheta_values)
    nz = nzeta_values[i]
    print("  mtheta=$mth, nzeta=$nz ($(mth*nz) pts) ... ")

    local vac_inputs_2D = compute_vacuum_inputs_from_equil(equil, mth, mlow, mpert, nlow)
    local vac_inputs_3D = Vacuum.VacuumInput3D(vac_inputs_2D, nz, nlow, npert)

    t = @elapsed begin
        wv, _, _, _ = Vacuum.compute_vacuum_response_3D(vac_inputs_3D, wall_settings)
    end

    rel_err, _, ev_err = compute_accuracy_metrics(wv, wv_ref_3D)

    push!(rel_errors_3D, rel_err)
    push!(ev_errors_3D, ev_err)
    push!(times_3D, t)

    println(@sprintf "rel_err=%.2e, ev_err=%.2e, time=%.2fs" rel_err ev_err t)
end

# ============================================================================
# 2D Convergence scan (same mtheta values)
# ============================================================================

println("\n" * "="^70)
println("2D CONVERGENCE SCAN")
println("="^70)

rel_errors_2D = Float64[]
ev_errors_2D = Float64[]
times_2D = Float64[]

for mth in mtheta_values
    print("  mtheta=$mth ... ")

    local vac_inputs_2D = compute_vacuum_inputs_from_equil(equil, mth, mlow, mpert, nlow)

    t = @elapsed begin
        wv, _, _ = Vacuum.compute_vacuum_response(vac_inputs_2D, wall_settings)
    end

    rel_err, _, ev_err = compute_accuracy_metrics(wv, wv_ref_2D)

    push!(rel_errors_2D, rel_err)
    push!(ev_errors_2D, ev_err)
    push!(times_2D, t)

    println(@sprintf "rel_err=%.2e, ev_err=%.2e, time=%.2fs" rel_err ev_err t)
end

# ============================================================================
# 2D vs 3D comparison (compute difference at each mtheta)
# ============================================================================

println("\n" * "="^70)
println("2D vs 3D COMPARISON (at matched mtheta)")
println("="^70)

rel_errors_2D_vs_3D = Float64[]
ev_errors_2D_vs_3D = Float64[]

for (i, mth) in enumerate(mtheta_values)
    nz = nzeta_values[i]
    print("  mtheta=$mth ... ")

    local vac_inputs_2D = compute_vacuum_inputs_from_equil(equil, mth, mlow, mpert, nlow)
    local vac_inputs_3D = Vacuum.VacuumInput3D(vac_inputs_2D, nz, nlow, npert)

    # Compute both 2D and 3D
    wv_2D, _, _ = Vacuum.compute_vacuum_response(vac_inputs_2D, wall_settings)
    wv_3D, _, _, _ = Vacuum.compute_vacuum_response_3D(vac_inputs_3D, wall_settings)

    # Compare 2D block against corresponding 3D block
    # For n=nlow, the 2D wv corresponds to the first mpert×mpert block of the 3D wv
    wv_3D_block = wv_3D[1:mpert, 1:mpert]

    rel_err = norm(wv_2D - wv_3D_block) / norm(wv_2D)
    ev_2D = maximum(real.(eigvals(wv_2D)))
    ev_3D = maximum(real.(eigvals(wv_3D_block)))
    ev_err = abs(ev_2D - ev_3D) / abs(ev_2D)

    push!(rel_errors_2D_vs_3D, rel_err)
    push!(ev_errors_2D_vs_3D, ev_err)

    println(@sprintf "rel_err=%.2e, ev_err=%.2e" rel_err ev_err)
end

# ============================================================================
# Create plots
# ============================================================================

println("\n" * "="^70)
println("CREATING PLOTS")
println("="^70)

# Plot 1: 3D convergence vs total grid points
p1 = plot(total_points, rel_errors_3D;
    label="Relative Frobenius Error",
    marker=:circle,
    xlabel="Total grid points (mtheta × nzeta)",
    ylabel="Relative Error",
    title="3D Vacuum Response Convergence",
    legend=:topright,
    yscale=:log10,
    xscale=:log10,
    minorgrid=true,
    framestyle=:box)
plot!(p1, total_points, ev_errors_3D;
    label="Max Eigenvalue Error",
    marker=:square)

# Plot 2: 2D convergence vs mtheta
p2 = plot(mtheta_values, rel_errors_2D;
    label="Relative Frobenius Error",
    marker=:circle,
    xlabel="mtheta (poloidal points)",
    ylabel="Relative Error",
    title="2D Vacuum Response Convergence",
    legend=:topright,
    yscale=:log10,
    xscale=:log10,
    minorgrid=true,
    framestyle=:box)
plot!(p2, mtheta_values, ev_errors_2D;
    label="Max Eigenvalue Error",
    marker=:square)

# Plot 3: 2D vs 3D comparison vs mtheta
p3 = plot(mtheta_values, rel_errors_2D_vs_3D;
    label="2D vs 3D Rel. Error",
    marker=:circle,
    xlabel="mtheta (poloidal points)",
    ylabel="Relative Difference (2D vs 3D)",
    title="2D vs 3D Comparison (nzeta/mtheta = $aspect_ratio_grid)",
    legend=:topright,
    yscale=:log10,
    xscale=:log10,
    minorgrid=true,
    framestyle=:box)
plot!(p3, mtheta_values, ev_errors_2D_vs_3D;
    label="Eigenvalue Rel. Diff.",
    marker=:square)

# Plot 4: Runtime comparison
p4 = plot(total_points, times_3D;
    label="3D (vs total points)",
    marker=:circle,
    xlabel="Grid points",
    ylabel="Runtime (s)",
    title="Runtime Scaling",
    legend=:topleft,
    xscale=:log10,
    yscale=:log10,
    minorgrid=true,
    framestyle=:box)
plot!(p4, mtheta_values, times_2D;
    label="2D (vs mtheta)",
    marker=:square)

# Add O(N^2) reference lines
N_ref_3D = total_points
t_ref_scale_3D = times_3D[end] * (N_ref_3D ./ N_ref_3D[end]) .^ 2
plot!(p4, N_ref_3D, t_ref_scale_3D; label="O(N²)", linestyle=:dash, color=:gray)

# Combined plot
p_combined = plot(p1, p2, p3, p4; layout=(2, 2), size=(1000, 800))

# Save plots
savefig(p1, joinpath(@__DIR__, "vacuum_3D_convergence_vs_gridpoints.pdf"))
savefig(p2, joinpath(@__DIR__, "vacuum_2D_convergence_vs_mtheta.pdf"))
savefig(p3, joinpath(@__DIR__, "vacuum_2D_vs_3D_comparison.pdf"))
savefig(p_combined, joinpath(@__DIR__, "vacuum_convergence_combined.pdf"))

println("Plots saved to $(joinpath(@__DIR__, "vacuum_*.pdf"))")

# Display plots
display(p_combined)

# ============================================================================
# Summary tables
# ============================================================================

println("\n" * "="^70)
println("SUMMARY")
println("="^70)

println("\n3D Convergence (nzeta/mtheta = $aspect_ratio_grid):")
println("  mtheta | nzeta | Total Pts |  Rel Error  |  EV Error   |  Time (s)")
println("  " * "-"^65)
for i in eachindex(mtheta_values)
    @printf "  %6d | %5d | %9d |  %.2e  |  %.2e  |  %7.2f\n" mtheta_values[i] nzeta_values[i] total_points[i] rel_errors_3D[i] ev_errors_3D[i] times_3D[i]
end

println("\n2D Convergence:")
println("  mtheta  |  Rel Error  |  EV Error   |  Time (s)")
println("  " * "-"^50)
for i in eachindex(mtheta_values)
    @printf "  %6d  |  %.2e  |  %.2e  |  %7.2f\n" mtheta_values[i] rel_errors_2D[i] ev_errors_2D[i] times_2D[i]
end

println("\n2D vs 3D Comparison:")
println("  mtheta  |  Rel Diff   |  EV Diff")
println("  " * "-"^40)
for i in eachindex(mtheta_values)
    @printf "  %6d  |  %.2e  |  %.2e\n" mtheta_values[i] rel_errors_2D_vs_3D[i] ev_errors_2D_vs_3D[i]
end

println("\n" * "="^70)
println("BENCHMARK COMPLETE")
println("="^70)
