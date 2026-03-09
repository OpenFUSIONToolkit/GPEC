# ============================================================================
# JPEC FUSE Integration: Time-Series Demonstration
# ============================================================================
#
# This script demonstrates how to use JPEC as a FUSE module to analyze
# MHD stability evolution across MULTIPLE time slices (time-series run).
#
# USE CASE:
#   You have a plasma scenario evolving over time (e.g., current ramp-up
#   from t=0s to t=10s) and want to track: "When does the plasma become
#   unstable during this discharge?"
#
# WORKFLOW:
#   1. Create an IMAS dd with equilibrium data at multiple times
#   2. Call JPEC.run_jpec_fuse(dd; time_indices=:all) to analyze all slices
#   3. Read results from dd.mhd_linear for each time
#   4. Plot δW vs time to visualize stability evolution
#
# ============================================================================

using JPEC         # Our MHD stability code
import EFIT        # Equilibrium solver
import EFIT.IMASdd # IMAS data dictionary

println("="^70)
println("JPEC FUSE Time-Series Demonstration")
println("="^70)
println()

# ──────────────────────────────────────────────────────────────────────────
# STEP 1: Create a time-evolving equilibrium scenario
# ──────────────────────────────────────────────────────────────────────────
#
# In a real FUSE run, equilibrium would evolve due to:
#   - Current diffusion
#   - Heating ramp-up
#   - Plasma shape changes
#   - etc.
#
# For this demo, we'll simulate a scenario with 5 time slices:
#   t = 0s, 2s, 4s, 6s, 8s
#
# (We'll use the same equilibrium for simplicity - in reality each would
#  be different as the plasma evolves)
# ──────────────────────────────────────────────────────────────────────────

println("Step 1: Creating time-evolving equilibrium scenario...")
println("-" * "="^69)

# Load a reference equilibrium
data_dir = joinpath(@__DIR__, "..", "test", "test_data",
                    "regression_equilibrium_example")
geqdsk_path = joinpath(data_dir, "EQDSK_COCOS_02")

g = EFIT.readg(geqdsk_path; set_time=0.0)

# Define time points for our scenario
times = [0.0, 2.0, 4.0, 6.0, 8.0]  # seconds
n_slices = length(times)

println("  Scenario: Plasma evolution from t=0s to t=8s")
println("  Number of time slices: ", n_slices)
println("  Times: ", times, " s")
println()

# Create IMAS dd with multiple time slices
dd = IMASdd.dd()

dd.equilibrium.time = times
resize!(dd.equilibrium.time_slice, n_slices)

# Fill each time slice with equilibrium data
# (In a real FUSE run, each slice would have different equilibrium data
#  as the plasma evolves. Here we use the same for demonstration.)
for i in 1:n_slices
    dd.equilibrium.time_slice[i].time = times[i]
    EFIT.geqdsk2imas!(g, dd.equilibrium.time_slice[i]; wall=nothing)
end

println("  ✓ Created dd with $n_slices equilibrium time slices")
println()

# ──────────────────────────────────────────────────────────────────────────
# STEP 2: Run JPEC stability analysis across ALL time slices
# ──────────────────────────────────────────────────────────────────────────
#
# KEY DIFFERENCE from single-slice:
#   time_indices = :all   →  Analyze ALL time slices in the dd
#
# WHAT HAPPENS:
#   - JPEC loops over all 5 time slices
#   - For each one: loads equilibrium, runs DCON, writes results
#   - Progress is shown for each slice
# ──────────────────────────────────────────────────────────────────────────

println("Step 2: Running JPEC time-series stability analysis...")
println("-" * "="^69)
println("  This will analyze all $n_slices time slices sequentially")
println()

result_dd = JPEC.run_jpec_fuse(dd;
    time_indices = :all,    # ← KEY: analyze ALL time slices
    nn_low       = 1,       # Check n=1 mode (most important for kink instability)
    nn_high      = 1,
    vac_flag     = true,    # Free-boundary
    verbose      = true     # Show progress for each slice
)

println()

# ──────────────────────────────────────────────────────────────────────────
# STEP 3: Extract stability evolution data
# ──────────────────────────────────────────────────────────────────────────
#
# We'll collect the most unstable δW for each time slice and track how
# stability changes as the plasma evolves.
# ──────────────────────────────────────────────────────────────────────────

println("Step 3: Extracting stability evolution...")
println("-" * "="^69)

mhd = result_dd.mhd_linear

# Arrays to store results
time_array = Float64[]
δW_array   = Float64[]
n_unstable_array = Int[]

println()
println("  Time-series stability summary:")
println("  " * "-"^66)
println("    t (s)  |  Most unstable δW  |  # unstable modes  |  Verdict")
println("  " * "-"^66)

for t in times
    # Find the time slice for this time
    ts_idx = findfirst(ts -> ts.time == t, mhd.time_slice)

    if ts_idx !== nothing
        ts = mhd.time_slice[ts_idx]

        # Find most unstable mode (minimum δW)
        δW_values = [real(mode.energy_perturbed) for mode in ts.toroidal_mode]
        δW_min = minimum(δW_values)
        n_unstable = count(δW -> δW < 0, δW_values)

        # Store for plotting
        push!(time_array, t)
        push!(δW_array, δW_min)
        push!(n_unstable_array, n_unstable)

        # Verdict
        verdict = δW_min < 0 ? "UNSTABLE ⚠️" : "stable ✓"

        println("    ", rpad(t, 6), " |  ", rpad(round(δW_min, digits=4), 17),
                " |  ", rpad(n_unstable, 17), " |  ", verdict)
    else
        println("    ", rpad(t, 6), " |  NO DATA")
    end
end

println("  " * "-"^66)
println()

# ──────────────────────────────────────────────────────────────────────────
# STEP 4: Analyze stability evolution trends
# ──────────────────────────────────────────────────────────────────────────

println("Step 4: Stability evolution analysis...")
println("-" * "="^69)

# Check if stability changes over time
if all(δW -> δW > 0, δW_array)
    println("  ✓ PLASMA REMAINS STABLE throughout the discharge")
    println("    → All time slices have δW > 0")
    println("    → Safe operating scenario")
elseif all(δW -> δW < 0, δW_array)
    println("  ⚠️  PLASMA IS ALWAYS UNSTABLE")
    println("     → All time slices have δW < 0")
    println("     → This scenario would disrupt immediately")
else
    # Mixed stability - find transition time
    first_unstable = findfirst(δW -> δW < 0, δW_array)
    if first_unstable !== nothing
        t_unstable = time_array[first_unstable]
        println("  ⚠️  PLASMA BECOMES UNSTABLE at t ≈ ", t_unstable, " s")
        println("     → Stable for early times")
        println("     → Crosses into instability during evolution")
        println("     → FUSE would need to adjust scenario before this time")
    else
        println("  ✓ PLASMA BECOMES STABLE over time")
        println("    → Started unstable, evolved to stable")
    end
end

println()

# Show stability margin trend
δW_change = δW_array[end] - δW_array[1]
if δW_change > 0
    println("  Stability margin trend: IMPROVING ↑")
    println("    δW increased by ", round(δW_change, digits=4), " over the discharge")
elseif δW_change < 0
    println("  Stability margin trend: DEGRADING ↓")
    println("    δW decreased by ", round(abs(δW_change), digits=4), " over the discharge")
else
    println("  Stability margin trend: CONSTANT →")
    println("    δW unchanged (equilibrium is not evolving in this demo)")
end

println()

# ──────────────────────────────────────────────────────────────────────────
# STEP 5: Simple text-based "plot" of δW vs time
# ──────────────────────────────────────────────────────────────────────────

println("Step 5: Stability evolution visualization (text plot)...")
println("-" * "="^69)
println()
println("  δW vs Time:")
println()

# Find range for scaling
δW_max = maximum(δW_array)
δW_min_val = minimum(δW_array)
δW_range = δW_max - δW_min_val

if δW_range > 0
    plot_width = 50

    for (i, t) in enumerate(time_array)
        δW = δW_array[i]

        # Scale to plot width
        normalized = (δW - δW_min_val) / δW_range
        bar_length = round(Int, normalized * plot_width)

        # Create bar
        bar = "█" ^ bar_length
        marker = δW < 0 ? " ⚠️ UNSTABLE" : " ✓"

        println("  t=", rpad(t, 4), "s  ", rpad(bar, plot_width), "  δW=",
                rpad(round(δW, digits=4), 8), marker)
    end

    println()
    println("  " * "─"^66)
    println("  Stability threshold: δW = 0  (stable if δW > 0)")
    println("  " * "─"^66)
else
    println("  (All δW values are the same - no evolution in this demo)")
end

println()

# ──────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────

println("="^70)
println("TIME-SERIES DEMONSTRATION COMPLETE")
println("="^70)
println()
println("What we showed:")
println("  1. ✓ Created IMAS dd with $n_slices equilibrium time slices")
println("  2. ✓ Ran JPEC.run_jpec_fuse(dd; time_indices=:all)")
println("  3. ✓ Extracted stability results for each time")
println("  4. ✓ Analyzed stability evolution trends")
println("  5. ✓ Visualized δW vs time")
println()
println("Real FUSE use case:")
println("  - FUSE evolves equilibrium over time (transport, heating, etc.)")
println("  - JPEC tracks stability margin throughout the discharge")
println("  - If stability margin drops below threshold → FUSE adjusts")
println("    (e.g., reduce heating, change plasma shape, etc.)")
println("  - Goal: Find stable operating scenarios for reactor design")
println("="^70)
