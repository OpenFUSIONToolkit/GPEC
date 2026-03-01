# Tests for JPEC IMAS integration (read_imas + write_imas)
#
# TODO: Create comprehensive tests for IMAS integration
#
# Test 1: Verify read_imas produces same result as read_efit
# Test 2: Verify write_imas correctly populates dd.mhd_linear
#
# Import required packages:
# - import EFIT
# - import EFIT.IMASdd

# TODO: Test 1 - IMAS equilibrium: read_imas matches read_efit
#
# Steps to implement:
# 1. Read test gEQDSK file with EFIT path (reference)
# 2. Convert same gEQDSK to IMAS dd using EFIT.geqdsk2imas!
# 3. Load via IMAS path with same grid settings
# 4. Compare key quantities (should match within ~1% tolerance):
#    - equil.ro (magnetic axis R)
#    - equil.zo (magnetic axis Z)
#    - equil.psio (normalizing flux)
#    - q profile
#    - p profile
#
# Test data location: joinpath(@__DIR__, "test_data", "regression_equilibrium_example", "EQDSK_COCOS_02")

# TODO: Test 2 - IMAS write: write_imas populates dd.mhd_linear
#
# Steps to implement:
# 1. Create empty dd with one equilibrium time slice (t=0.0)
# 2. Build minimal synthetic DCON result:
#    - DconControl with nn_low=1, nn_high=1, vac_flag=true
#    - DconInternal with n_modes=7 (m=-3 to m=3)
#    - VacuumData with synthetic eigenvalues (first one negative = unstable)
# 3. Call JPEC.DCON.write_imas(dd, result)
# 4. Verify dd.mhd_linear is correctly populated:
#    - ideal_flag == 1
#    - code.name == "JPEC"
#    - One time slice at t=0.0
#    - 7 toroidal_mode entries, all with n_tor==1
#    - First eigenmode has energy_perturbed < 0
#    - Energy values match real(vac_data.et[i])

println("\n" * "="^70)
println("IMAS integration tests need to be implemented!")
println("="^70)
