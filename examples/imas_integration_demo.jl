"""
IMAS Integration Demonstration for JPEC
========================================

This script demonstrates that the IMAS integration (Chapter 1) is fully functional
by showing:
1. Reading equilibrium data from IMAS format
2. Comparing IMAS path vs EFIT path (they should match)
3. Writing DCON stability results to IMAS format
4. Visualizing the results

Author: Sophia Puscalau
Date: 2026-03-04
"""

using Pkg
Pkg.activate(".")

println("="^70)
println("JPEC IMAS Integration Demonstration")
println("="^70)
println()

# Import required packages
using JPEC
using EFIT
using EFIT.IMASdd
using Plots
using Printf

println("✓ Packages loaded successfully")
println()

#=============================================================================
PART 1: Demonstrate read_imas (Test 1)
==============================================================================#

println("PART 1: Testing IMAS Equilibrium Reading")
println("-"^70)

# Path to test data
data_dir = joinpath(@__DIR__, "..", "test", "test_data", "regression_equilibrium_example")
geqdsk_path = joinpath(data_dir, "EQDSK_COCOS_02")

println("Using test file: ", basename(geqdsk_path))
println()

# Step 1: Read equilibrium via standard EFIT path (reference)
println("Step 1: Reading equilibrium via EFIT g-file path...")
efit_control = JPEC.Equilibrium.EquilibriumControl(;
    eq_type = "efit",
    eq_filename = geqdsk_path,
    jac_type = "boozer",
    psilow = 0.01,
    psihigh = 0.994,
)
efit_config = JPEC.Equilibrium.EquilibriumConfig(efit_control, JPEC.Equilibrium.EquilibriumOutput())
equil_efit = JPEC.Equilibrium.setup_equilibrium(efit_config)

println("✓ EFIT equilibrium loaded")
println("  - Magnetic axis: R₀ = $(round(equil_efit.ro, digits=4)) m, Z₀ = $(round(equil_efit.zo, digits=4)) m")
println("  - Flux swing: ψ₀ = $(round(equil_efit.psio, digits=6)) Wb")
println()

# Step 2: Convert gEQDSK to IMAS format
println("Step 2: Converting gEQDSK to IMAS data dictionary...")
g = EFIT.readg(geqdsk_path; set_time=0.0)
dd = IMASdd.dd()

dd.global_time = g.time
dd.equilibrium.time = [g.time]
resize!(dd.equilibrium.time_slice, 1)
dd.equilibrium.time_slice[1].time = g.time

EFIT.geqdsk2imas!(g, dd.equilibrium.time_slice[1]; wall=nothing)
println("✓ IMAS data dictionary created")
println()

# Step 3: Read equilibrium via IMAS path
println("Step 3: Reading equilibrium via IMAS path...")
imas_control = JPEC.Equilibrium.EquilibriumControl(;
    eq_type = "imas",
    eq_filename = joinpath(data_dir, "imas_placeholder"),
    jac_type = "boozer",
    psilow = 0.01,
    psihigh = 0.994,
)
imas_config = JPEC.Equilibrium.EquilibriumConfig(imas_control, JPEC.Equilibrium.EquilibriumOutput())
equil_imas = JPEC.Equilibrium.setup_equilibrium(imas_config, dd)

println("✓ IMAS equilibrium loaded")
println("  - Magnetic axis: R₀ = $(round(equil_imas.ro, digits=4)) m, Z₀ = $(round(equil_imas.zo, digits=4)) m")
println("  - Flux swing: ψ₀ = $(round(equil_imas.psio, digits=6)) Wb")
println()

# Step 4: Compare the two equilibria
println("Step 4: Comparing EFIT vs IMAS results...")
println()

ro_diff = abs(equil_efit.ro - equil_imas.ro) / equil_efit.ro * 100
zo_diff = abs(equil_efit.zo - equil_imas.zo)
psio_diff = abs(equil_efit.psio - equil_imas.psio) / abs(equil_efit.psio) * 100

println(@sprintf("  Magnetic axis R:  %.6f vs %.6f (diff: %.4f%%)", equil_efit.ro, equil_imas.ro, ro_diff))
println(@sprintf("  Magnetic axis Z:  %.6f vs %.6f (diff: %.6f m)", equil_efit.zo, equil_imas.zo, zo_diff))
println(@sprintf("  Flux swing ψ₀:    %.6f vs %.6f (diff: %.4f%%)", equil_efit.psio, equil_imas.psio, psio_diff))
println()

q_max_diff = maximum(abs.(equil_efit.profiles.q_spline.y .- equil_imas.profiles.q_spline.y) ./ abs.(equil_efit.profiles.q_spline.y)) * 100
P_max_diff = maximum(abs.(equil_efit.profiles.P_spline.y .- equil_imas.profiles.P_spline.y) ./ (abs.(equil_efit.profiles.P_spline.y) .+ 1e-10)) * 100

println(@sprintf("  q profile:        max diff = %.4f%%", q_max_diff))
println(@sprintf("  P profile:        max diff = %.4f%%", P_max_diff))
println()

# Determine pass/fail
all_tests_pass = (ro_diff < 0.1) && (zo_diff < 1e-3) && (psio_diff < 0.1) && (q_max_diff < 1.0) && (P_max_diff < 1.0)

if all_tests_pass
    println("✅ TEST 1 PASSED: IMAS equilibrium reading works correctly!")
else
    println("⚠️  TEST 1 FAILED: Differences exceed tolerance")
end
println()

# Create comparison plots
println("Creating comparison plots...")

# Plot 1: q profiles
p1 = plot(equil_efit.profiles.q_spline.x, equil_efit.profiles.q_spline.y,
    label="EFIT path", lw=2, xlabel="ψ_N", ylabel="q", title="Safety Factor q(ψ)",
    legend=:topleft, dpi=150)
plot!(p1, equil_imas.profiles.q_spline.x, equil_imas.profiles.q_spline.y,
    label="IMAS path", lw=2, ls=:dash)

# Plot 2: Pressure profiles
p2 = plot(equil_efit.profiles.P_spline.x, equil_efit.profiles.P_spline.y,
    label="EFIT path", lw=2, xlabel="ψ_N", ylabel="P [Pa]", title="Pressure P(ψ)",
    legend=:topright, dpi=150)
plot!(p2, equil_imas.profiles.P_spline.x, equil_imas.profiles.P_spline.y,
    label="IMAS path", lw=2, ls=:dash)

# Plot 3: Difference in q
q_diff = abs.(equil_efit.profiles.q_spline.y .- equil_imas.profiles.q_spline.y)
p3 = plot(equil_efit.profiles.q_spline.x, q_diff,
    label="", lw=2, xlabel="ψ_N", ylabel="|Δq|", title="Absolute Difference in q",
    legend=false, dpi=150, color=:red)

# Plot 4: Relative difference in P
P_rel_diff = abs.(equil_efit.profiles.P_spline.y .- equil_imas.profiles.P_spline.y) ./ (abs.(equil_efit.profiles.P_spline.y) .+ 1e-10) * 100
p4 = plot(equil_efit.profiles.P_spline.x, P_rel_diff,
    label="", lw=2, xlabel="ψ_N", ylabel="Rel. diff [%]", title="Relative Difference in P",
    legend=false, dpi=150, color=:red)

# Combine plots
plot_combined = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900))
savefig(plot_combined, joinpath(@__DIR__, "imas_test1_comparison.png"))
println("✓ Saved: imas_test1_comparison.png")
println()

#=============================================================================
PART 2: Demonstrate write_imas (Test 2)
==============================================================================#

println()
println("PART 2: Testing IMAS Stability Results Writing")
println("-"^70)

# Create a minimal IMAS dd for stability output
println("Step 1: Creating minimal IMAS data dictionary...")
dd_out = IMASdd.dd()
dd_out.equilibrium.time = [0.0]
resize!(dd_out.equilibrium.time_slice, 1)
dd_out.equilibrium.time_slice[1].time = 0.0
println("✓ IMAS dd created")
println()

# Create synthetic DCON results
println("Step 2: Creating synthetic DCON stability results...")
n_modes = 7

ctrl = JPEC.DCON.DconControl(;
    nn_low = 1,
    nn_high = 1,
    vac_flag = true,
    verbose = false,
)

intr = JPEC.DCON.DconInternal(;
    nlow = 1,
    nhigh = 1,
    npert = 1,
    mlow = -3,
    mhigh = 3,
    mpert = n_modes,
    numpert_total = n_modes,
)

vac_data = JPEC.DCON.VacuumData(10, n_modes, n_modes)
# First mode unstable (negative), rest stable (positive)
vac_data.et .= ComplexF64[-0.3, 0.1, 0.2, 0.4, 0.6, 0.8, 1.2]

fake_result = (ctrl=ctrl, equil=nothing, intr=intr, ffit=nothing, odet=nothing, vac_data=vac_data)
println("✓ Synthetic results created (7 modes, n=1)")
println("  Eigenvalues: ", real.(vac_data.et))
println()

# Call write_imas
println("Step 3: Writing results to IMAS dd.mhd_linear...")
JPEC.DCON.write_imas(dd_out, fake_result)
println("✓ write_imas() completed")
println()

# Verify the results
println("Step 4: Verifying IMAS mhd_linear structure...")
mhd = dd_out.mhd_linear

println("  - ideal_flag: ", mhd.ideal_flag, " (expected: 1)")
println("  - homogeneous_time: ", mhd.ids_properties.homogeneous_time, " (expected: 1)")
println("  - code.name: \"", mhd.code.name, "\" (expected: \"JPEC\")")
println("  - Number of time slices: ", length(mhd.time_slice), " (expected: 1)")
println("  - Time: ", mhd.time_slice[1].time, " s (expected: 0.0)")
println("  - Number of toroidal modes: ", length(mhd.time_slice[1].toroidal_mode), " (expected: 7)")
println()

# Check each mode
println("  Toroidal modes:")
for (i, tm) in enumerate(mhd.time_slice[1].toroidal_mode)
    stability = tm.energy_perturbed < 0 ? "UNSTABLE" : "stable"
    println(@sprintf("    Mode %d: n_tor=%d, δW=%.2f (%s)", i, tm.n_tor, tm.energy_perturbed, stability))
end
println()

# Verify correctness
test2_pass = (
    mhd.ideal_flag == 1 &&
    mhd.ids_properties.homogeneous_time == 1 &&
    mhd.code.name == "JPEC" &&
    length(mhd.time_slice) == 1 &&
    mhd.time_slice[1].time == 0.0 &&
    length(mhd.time_slice[1].toroidal_mode) == n_modes &&
    all(tm.n_tor == 1 for tm in mhd.time_slice[1].toroidal_mode) &&
    mhd.time_slice[1].toroidal_mode[1].energy_perturbed < 0 &&
    all(isapprox(mhd.time_slice[1].toroidal_mode[i].energy_perturbed, real(vac_data.et[i]), atol=1e-10) for i in 1:n_modes)
)

if test2_pass
    println("✅ TEST 2 PASSED: IMAS stability writing works correctly!")
else
    println("⚠️  TEST 2 FAILED: Structure verification failed")
end
println()

# Create stability visualization
println("Creating stability visualization...")
mode_numbers = 1:n_modes
energies = [tm.energy_perturbed for tm in mhd.time_slice[1].toroidal_mode]
colors = [e < 0 ? :red : :green for e in energies]

p_stability = bar(mode_numbers, energies,
    color=colors, xlabel="Mode number", ylabel="δW (Perturbed Energy)",
    title="DCON Stability Results (n=1)", legend=false, dpi=150,
    ylims=(minimum(energies)*1.2, maximum(energies)*1.2))
hline!([0.0], color=:black, lw=2, ls=:dash, label="")
annotate!(1, energies[1]*0.8, text("UNSTABLE", :red, 10))
annotate!(4, energies[4]*1.1, text("STABLE", :green, 10))

savefig(p_stability, joinpath(@__DIR__, "imas_test2_stability.png"))
println("✓ Saved: imas_test2_stability.png")
println()

#=============================================================================
SUMMARY
==============================================================================#

println()
println("="^70)
println("DEMONSTRATION SUMMARY")
println("="^70)
println()

if all_tests_pass
    println("✅ Test 1 (read_imas): PASSED")
    println("   - IMAS equilibrium loading matches EFIT path")
    println("   - All key quantities agree within tolerance")
    println("   - Plots saved: imas_test1_comparison.png")
else
    println("⚠️  Test 1 (read_imas): FAILED")
end
println()

if test2_pass
    println("✅ Test 2 (write_imas): PASSED")
    println("   - IMAS mhd_linear structure correctly populated")
    println("   - Stability results accurately stored")
    println("   - Plots saved: imas_test2_stability.png")
else
    println("⚠️  Test 2 (write_imas): FAILED")
end
println()

if all_tests_pass && test2_pass
    println("🎉 ALL TESTS PASSED! IMAS integration is fully functional!")
else
    println("⚠️  Some tests failed. Please review output above.")
end
println()
println("="^70)
println("End of demonstration")
println("="^70)
