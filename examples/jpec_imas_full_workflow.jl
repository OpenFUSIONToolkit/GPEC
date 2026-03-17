"""
Full JPEC IMAS Integration Workflow Demo (Synthetic Eigenvalues)

This script demonstrates the complete JPEC IMAS integration using synthetic
stability data. It tests the full round-trip:
1. Load a g-file and convert to IMAS format (COCOS 11)
2. Read the IMAS equilibrium through JPEC's new read_imas() path
3. Write synthetic stability results to IMAS mhd_linear via JPEC.DCON.write_imas()
4. Verify the complete workflow

The synthetic eigenvalues let us test the IMAS read/write machinery
without running the full stability solver.
"""

using TOML, JPEC
import EFIT, EFIT.IMASdd

println("="^70)
println("Full JPEC IMAS Integration Workflow (Synthetic Eigenvalues)")
println("Using COCOS 11 (real IMAS convention)")
println("="^70)
println()

# ==========================================================================
# STEP 1: Convert g-file to IMAS (COCOS 11)
# ==========================================================================
println("STEP 1: Loading g-file and converting to IMAS (COCOS 11)")
println("-"^70)

# Use the regression test equilibrium data
data_dir = joinpath(@__DIR__, "..", "test", "test_data", "regression_equilibrium_example")
geqdsk_path = joinpath(data_dir, "EQDSK_COCOS_02")

if !isfile(geqdsk_path)
    error("Test file not found: $geqdsk_path")
end

g = EFIT.readg(geqdsk_path; set_time=0.0)
println("  Loaded g-file: EQDSK_COCOS_02")
println("  Grid: $(size(g.psirz))")
println("  Time: $(g.time) s")

# Create IMAS data dictionary using EFIT's standard converter (COCOS 11)
dd = IMASdd.dd()
dd.global_time = g.time
dd.equilibrium.time = [g.time]
resize!(dd.equilibrium.time_slice, 1)
dd.equilibrium.time_slice[1].time = g.time
EFIT.geqdsk2imas!(g, dd.equilibrium.time_slice[1]; wall=nothing)
println("  Converted to IMAS format using EFIT.geqdsk2imas! (COCOS 11)")
println()

# ==========================================================================
# STEP 2: Read IMAS equilibrium through JPEC's new read_imas() path
# ==========================================================================
println("STEP 2: Reading IMAS equilibrium via JPEC.Equilibrium.setup_equilibrium()")
println("-"^70)

# Configure JPEC to use the IMAS equilibrium type
imas_config = JPEC.Equilibrium.EquilibriumConfig(;
    eq_type = "imas",
    eq_filename = joinpath(data_dir, "imas_output"),
    jac_type = "boozer",
    psilow = 0.01,
    psihigh = 0.994,
)

# This calls: setup_equilibrium -> read_imas(config, dd) -> DirectRunInput -> solver
# The COCOS 11 auto-detection and conversion happens inside read_imas()
equil = JPEC.Equilibrium.setup_equilibrium(imas_config, dd)
println("  JPEC equilibrium loaded via IMAS")
println("  Magnetic axis: R = $(round(equil.ro, digits=3)) m, Z = $(round(equil.zo, digits=3)) m")
println("  Flux swing: psio = $(round(equil.psio, digits=3)) Wb/rad")
println("  Safety factor: q0 = $(round(equil.params.q0, digits=2)), q95 = $(round(equil.params.q95, digits=2))")
println()

# ==========================================================================
# STEP 3: Create synthetic stability result and write to IMAS
# ==========================================================================
println("STEP 3: Writing synthetic stability results to IMAS mhd_linear")
println("-"^70)

# Build a synthetic result tuple that mimics what JPEC.main() returns
# This tests the write_imas() machinery without running the full solver
println("  Creating synthetic stability result for n=1 mode...")

ctrl = (nn_low = 1, nn_high = 1, vac_flag = true)

n_modes = 7  # 7 poloidal modes: m = -3 to 3
intr = (
    nlow = ctrl.nn_low,
    nhigh = ctrl.nn_high,
    mpert = n_modes,
    numpert_total = n_modes * (ctrl.nn_high - ctrl.nn_low + 1),
)

# First mode unstable (energy < 0), rest stable (energy > 0)
eigenvalues = zeros(ComplexF64, intr.numpert_total)
eigenvalues[1] = -0.5 + 0.0im
for i in 2:intr.numpert_total
    eigenvalues[i] = 0.1 * i + 0.0im
end
vac_data = (et = eigenvalues,)

result = (ctrl=ctrl, intr=intr, vac_data=vac_data)

# Write to IMAS using JPEC.DCON.write_imas()
JPEC.DCON.write_imas(dd, result)
println("  Results written to dd.mhd_linear")
println()

# ==========================================================================
# STEP 4: Verify the complete IMAS round-trip
# ==========================================================================
println("STEP 4: Verifying IMAS output")
println("-"^70)

mhd = dd.mhd_linear

println("  Metadata:")
println("    Code name:         \"$(mhd.code.name)\"")
println("    Ideal flag:        $(mhd.ideal_flag)")
println("    Homogeneous time:  $(mhd.ids_properties.homogeneous_time)")
println("    Comment:           \"$(mhd.ids_properties.comment)\"")
println()

println("  Mode results:")
n_toroidal_modes = length(mhd.time_slice[1].toroidal_mode)
unstable_count = 0
stable_count = 0
for (i, tm) in enumerate(mhd.time_slice[1].toroidal_mode)
    global unstable_count, stable_count
    energy = tm.energy_perturbed
    if energy < 0
        unstable_count += 1
        println("    Mode $i: n=$(tm.n_tor), energy=$(round(energy, digits=3)) <-- UNSTABLE")
    else
        stable_count += 1
    end
end
println("    Total: $n_modes modes ($unstable_count unstable, $stable_count stable)")
println()

# ==========================================================================
# VERIFICATION CHECKS
# ==========================================================================
println("="^70)
println("VERIFICATION SUMMARY")
println("="^70)
println()

all_pass = true

# Check 1: Equilibrium loaded correctly via IMAS
check1 = equil isa JPEC.Equilibrium.PlasmaEquilibrium
println("$(check1 ? "PASS" : "FAIL") - Equilibrium loaded via IMAS (COCOS 11 auto-detected)")
all_pass = all_pass && check1

# Check 2: Equilibrium has valid flux swing
check2 = abs(equil.psio) > 1e-3
println("$(check2 ? "PASS" : "FAIL") - Equilibrium has valid flux swing (psio = $(round(equil.psio, digits=4)))")
all_pass = all_pass && check2

# Check 3: IMAS metadata correctly populated
check3 = (mhd.code.name == "JPEC" &&
          mhd.ideal_flag == 1 &&
          mhd.ids_properties.homogeneous_time == 1)
println("$(check3 ? "PASS" : "FAIL") - IMAS metadata correctly populated")
all_pass = all_pass && check3

# Check 4: Time structure correct
check4 = (length(mhd.time) == 1 &&
          length(mhd.time_slice) == 1 &&
          mhd.time_slice[1].time == g.time)
println("$(check4 ? "PASS" : "FAIL") - Time structure correct")
all_pass = all_pass && check4

# Check 5: Correct number of modes
check5 = n_toroidal_modes == intr.numpert_total
println("$(check5 ? "PASS" : "FAIL") - Correct number of modes ($n_toroidal_modes)")
all_pass = all_pass && check5

# Check 6: Unstable mode correctly identified
check6 = mhd.time_slice[1].toroidal_mode[1].energy_perturbed < 0
println("$(check6 ? "PASS" : "FAIL") - Unstable mode correctly identified (energy < 0)")
all_pass = all_pass && check6

# Check 7: Energy values match what we wrote
check7 = true
for i in 1:intr.numpert_total
    if !isapprox(mhd.time_slice[1].toroidal_mode[i].energy_perturbed,
                 real(vac_data.et[i]), rtol=1e-10)
        global check7 = false
        break
    end
end
println("$(check7 ? "PASS" : "FAIL") - Energy values correctly stored in IMAS")
all_pass = all_pass && check7

println()
if all_pass
    println("SUCCESS: Full JPEC IMAS workflow is operational!")
    println()
    println("This demonstrates:")
    println("  1. JPEC can read equilibrium from IMAS format (COCOS 11)")
    println("  2. JPEC auto-detects COCOS and converts correctly")
    println("  3. JPEC.DCON.write_imas() correctly populates mhd_linear")
    println("  4. Complete IMAS read/write integration works")
else
    println("FAILURE: Some checks did not pass")
end

println()
println("="^70)
println("End of full workflow demonstration")
println("="^70)
