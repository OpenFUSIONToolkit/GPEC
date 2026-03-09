# ============================================================================
# JPEC FUSE Integration: Single-Slice Demonstration
# ============================================================================
#
# This script demonstrates how to use JPEC as a FUSE module to analyze
# MHD stability for ONE equilibrium snapshot (single time slice).
#
# USE CASE:
#   You have a plasma equilibrium at one moment in time (e.g., t = 5.0 s)
#   and want to know: "Is this equilibrium MHD-stable?"
#
# WORKFLOW:
#   1. Create an IMAS data dictionary (dd) with equilibrium data
#   2. Call JPEC.run_jpec_fuse(dd; ...) to analyze stability
#   3. Read results from dd.mhd_linear
#   4. Interpret: stable (δW > 0) or unstable (δW < 0)?
#
# ============================================================================

using JPEC         # Our MHD stability code
import EFIT        # Equilibrium solver (provides geqdsk2imas!)
import EFIT.IMASdd # IMAS data dictionary

println("="^70)
println("JPEC FUSE Single-Slice Demonstration")
println("="^70)
println()

# ──────────────────────────────────────────────────────────────────────────
# STEP 1: Load an equilibrium from a gEQDSK file and convert to IMAS dd
# ──────────────────────────────────────────────────────────────────────────
#
# In a real FUSE workflow, the equilibrium would come from FUSE's equilibrium
# solver (EFIT, CHEASE, etc.) and be stored directly in a dd.
#
# For this demo, we'll:
#   - Read a test gEQDSK file (standard equilibrium format)
#   - Convert it to an IMAS dd using EFIT.geqdsk2imas!
#
# This simulates what FUSE would do before calling JPEC.
# ──────────────────────────────────────────────────────────────────────────

println("Step 1: Loading equilibrium data...")
println("-" * "="^69)

# Path to test equilibrium file (a gEQDSK from JPEC's test suite)
data_dir = joinpath(@__DIR__, "..", "test", "test_data",
                    "regression_equilibrium_example")
geqdsk_path = joinpath(data_dir, "EQDSK_COCOS_02")

println("  Reading gEQDSK file: ", basename(geqdsk_path))

# Read the gEQDSK file into EFIT.jl's internal format
g = EFIT.readg(geqdsk_path; set_time=5.0)  # Simulate time t = 5.0 s

# Create an empty IMAS data dictionary
dd = IMASdd.dd()

# Set up the time structure (required by IMAS)
dd.equilibrium.time = [5.0]                    # Time array
resize!(dd.equilibrium.time_slice, 1)          # Create 1 time slice
dd.equilibrium.time_slice[1].time = 5.0        # Set its time

# Convert gEQDSK → IMAS dd (fills all equilibrium profiles)
# This writes: ψ(R,Z), q(ψ), p(ψ), F(ψ), boundary shape, etc.
EFIT.geqdsk2imas!(g, dd.equilibrium.time_slice[1]; wall=nothing)

println("  ✓ Equilibrium loaded into dd.equilibrium.time_slice[1]")
println("    - Time: ", dd.equilibrium.time_slice[1].time, " s")
println("    - Equilibrium data ready for JPEC analysis")
println()

# ──────────────────────────────────────────────────────────────────────────
# STEP 2: Run JPEC stability analysis via the FUSE wrapper
# ──────────────────────────────────────────────────────────────────────────
#
# Now we call JPEC.run_jpec_fuse(dd; ...) with our parameters.
#
# KEY PARAMETERS:
#   - time_indices = 1         → Analyze only the 1st time slice
#   - nn_low, nn_high = 1, 3   → Check toroidal modes n=1,2,3
#   - vac_flag = true          → Include vacuum (free-boundary) calculation
#   - verbose = true           → Print progress messages
#
# WHAT HAPPENS INSIDE:
#   1. JPEC reads equilibrium from dd.equilibrium.time_slice[1]
#   2. Runs DCON ideal MHD solver to compute eigenvalues
#   3. Writes stability results to dd.mhd_linear.time_slice[1]
# ──────────────────────────────────────────────────────────────────────────

println("Step 2: Running JPEC stability analysis...")
println("-" * "="^69)

result_dd = JPEC.run_jpec_fuse(dd;
    time_indices = 1,       # Analyze only time slice 1
    nn_low       = 1,       # Start at toroidal mode n=1
    nn_high      = 1,       # End at n=1 (just n=1 for this demo)
    vac_flag     = true,    # Free-boundary (include vacuum energy)
    verbose      = true     # Show progress
)

# The result is the same dd, but now with stability results filled in
println()

# ──────────────────────────────────────────────────────────────────────────
# STEP 3: Extract and interpret stability results
# ──────────────────────────────────────────────────────────────────────────
#
# Results are stored in dd.mhd_linear.time_slice[1].toroidal_mode[:]
#
# For each eigenmode, we have:
#   - n_tor              → Toroidal mode number (1, 2, 3, ...)
#   - energy_perturbed   → δW (perturbed energy)
#
# STABILITY CRITERION:
#   δW > 0  →  Mode is STABLE   (energy increases → perturbation decays)
#   δW < 0  →  Mode is UNSTABLE (energy decreases → perturbation grows!)
#
# The MOST UNSTABLE mode is the one with the lowest (most negative) δW.
# ──────────────────────────────────────────────────────────────────────────

println("Step 3: Interpreting stability results...")
println("-" * "="^69)

mhd = result_dd.mhd_linear

# Find the time slice with t=5.0s
ts_idx = findfirst(ts -> ts.time == 5.0, mhd.time_slice)
ts = mhd.time_slice[ts_idx]

println("  Results for t = ", ts.time, " s:")
println()

# Count number of modes and find most unstable
n_modes = length(ts.toroidal_mode)
println("  Total eigenmodes computed: ", n_modes)

# Collect all δW values to find the most unstable
δW_values = [real(mode.energy_perturbed) for mode in ts.toroidal_mode]
δW_min = minimum(δW_values)
most_unstable_idx = argmin(δW_values)

println()
println("  MOST UNSTABLE MODE:")
println("    n_tor = ", ts.toroidal_mode[most_unstable_idx].n_tor)
println("    δW    = ", round(δW_min, digits=6))
println()

# Stability verdict
if δW_min < 0
    println("  ⚠️  VERDICT: PLASMA IS UNSTABLE!")
    println("      → This equilibrium has an MHD instability")
    println("      → The plasma would disrupt in a real tokamak")
else
    println("  ✓  VERDICT: PLASMA IS STABLE")
    println("     → All modes have δW > 0")
    println("     → This equilibrium is MHD-stable")
end

println()
println("  Breakdown by toroidal mode number:")
println("  " * "-"^66)
println("    n  |  δW (energy)  |  Status")
println("  " * "-"^66)

# Group modes by n_tor and show the most unstable for each n
unique_n = sort(unique([m.n_tor for m in ts.toroidal_mode]))
for n in unique_n
    modes_for_n = filter(m -> m.n_tor == n, ts.toroidal_mode)
    if !isempty(modes_for_n)
        δW_n = minimum([real(m.energy_perturbed) for m in modes_for_n])
        status = δW_n < 0 ? "UNSTABLE ⚠️" : "stable ✓"
        println("    $n  |  ", rpad(round(δW_n, digits=6), 13), " |  ", status)
    end
end

println("  " * "-"^66)
println()

# ──────────────────────────────────────────────────────────────────────────
# SUMMARY: What we demonstrated
# ──────────────────────────────────────────────────────────────────────────

println("="^70)
println("DEMONSTRATION COMPLETE")
println("="^70)
println()
println("What we showed:")
println("  1. ✓ Created an IMAS dd with equilibrium data (simulated FUSE)")
println("  2. ✓ Called JPEC.run_jpec_fuse(dd; ...) to analyze stability")
println("  3. ✓ Read stability results from dd.mhd_linear")
println("  4. ✓ Interpreted δW values to determine stability")
println()
println("In a real FUSE workflow:")
println("  - FUSE equilibrium solver → fills dd.equilibrium")
println("  - JPEC.run_jpec_fuse(dd)  → analyzes stability")
println("  - FUSE reads dd.mhd_linear → makes decisions (adjust params, etc.)")
println("="^70)
