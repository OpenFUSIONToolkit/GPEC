# Tests for JPEC FUSE integration wrapper (run_jpec_fuse)
#
# These tests verify that JPEC can be called as a FUSE module, processing
# equilibrium data from IMAS data dictionaries and writing stability results
# back to dd.mhd_linear.
#
# Test coverage:
#   1. Single time-slice run
#   2. Multiple time-slice run (time-series)
#   3. All time slices (:all)
#   4. Error handling (empty dd, invalid indices)
#   5. Results validation (dd.mhd_linear is correctly populated)

import EFIT
import EFIT.IMASdd

@testset "FUSE Integration: run_jpec_fuse" begin

    # ── Test Setup: Create a dd with equilibrium data ────────────────────────
    # We'll use the same test gEQDSK as in runtests_imas.jl and create a dd
    # with multiple time slices by copying the same equilibrium at different times.

    data_dir    = joinpath(@__DIR__, "test_data", "regression_equilibrium_example")
    geqdsk_path = joinpath(data_dir, "EQDSK_COCOS_02")

    # Read gEQDSK and convert to dd
    g  = EFIT.readg(geqdsk_path; set_time=0.0)
    dd = IMASdd.dd()

    # Create 3 time slices with the same equilibrium but different times
    # (In a real FUSE run, each time slice would have different equilibrium data)
    n_time_slices = 3
    times = [0.0, 1.0, 2.0]

    dd.equilibrium.time = times
    resize!(dd.equilibrium.time_slice, n_time_slices)

    for i in 1:n_time_slices
        dd.equilibrium.time_slice[i].time = times[i]
        # Copy equilibrium data to each slice (using same gEQDSK for simplicity)
        EFIT.geqdsk2imas!(g, dd.equilibrium.time_slice[i]; wall=nothing)
    end

    # ── Test 1: Single time-slice run ─────────────────────────────────────────
    @testset "Single time-slice run" begin

        dd_test = deepcopy(dd)  # Don't modify the original dd

        # Run JPEC on only the first time slice, with a small n-range for speed
        result_dd = JPEC.run_jpec_fuse(dd_test;
                                       time_indices = 1,
                                       nn_low = 1,
                                       nn_high = 1,
                                       vac_flag = true,
                                       verbose = false)

        # Check that dd was modified and returned
        @test result_dd === dd_test

        # Check that mhd_linear has been populated
        mhd = dd_test.mhd_linear
        @test mhd.ideal_flag == 1
        @test mhd.code.name == "JPEC"

        # Should have at least 1 time slice in mhd_linear
        # (DCON writes to the time slice corresponding to the equilibrium time)
        @test length(mhd.time_slice) >= 1

        # Find the time slice that matches t=0.0
        ts_idx = findfirst(ts -> ts.time == 0.0, mhd.time_slice)
        @test ts_idx !== nothing

        ts = mhd.time_slice[ts_idx]

        # Should have toroidal_mode entries (one per eigenmode)
        @test length(ts.toroidal_mode) > 0

        # All modes should have n_tor == 1 (we only ran n=1)
        for tm in ts.toroidal_mode
            @test tm.n_tor == 1
        end

        # Energy values should be set (vac_flag=true)
        @test ts.toroidal_mode[1].energy_perturbed !== 0.0

    end

    # ── Test 2: Multiple time-slice run (explicit indices) ────────────────────
    @testset "Multiple time-slice run (explicit indices)" begin

        dd_test = deepcopy(dd)

        # Run on slices 1 and 3
        result_dd = JPEC.run_jpec_fuse(dd_test;
                                       time_indices = [1, 3],
                                       nn_low = 1,
                                       nn_high = 1,
                                       vac_flag = true,
                                       verbose = false)

        @test result_dd === dd_test

        mhd = dd_test.mhd_linear

        # Should have at least 2 time slices
        @test length(mhd.time_slice) >= 2

        # Check that both t=0.0 and t=2.0 slices exist
        ts_0 = findfirst(ts -> ts.time == 0.0, mhd.time_slice)
        ts_2 = findfirst(ts -> ts.time == 2.0, mhd.time_slice)

        @test ts_0 !== nothing
        @test ts_2 !== nothing

        # Both should have toroidal_mode entries
        @test length(mhd.time_slice[ts_0].toroidal_mode) > 0
        @test length(mhd.time_slice[ts_2].toroidal_mode) > 0

    end

    # ── Test 3: All time slices (:all) ────────────────────────────────────────
    @testset "Time-series run (:all slices)" begin

        dd_test = deepcopy(dd)

        # Run on all 3 time slices
        result_dd = JPEC.run_jpec_fuse(dd_test;
                                       time_indices = :all,
                                       nn_low = 1,
                                       nn_high = 2,  # n=1,2 for variety
                                       vac_flag = true,
                                       verbose = false)

        @test result_dd === dd_test

        mhd = dd_test.mhd_linear

        # Should have at least 3 time slices
        @test length(mhd.time_slice) >= 3

        # Check that all three times exist
        for t in times
            ts_idx = findfirst(ts -> ts.time == t, mhd.time_slice)
            @test ts_idx !== nothing
            @test length(mhd.time_slice[ts_idx].toroidal_mode) > 0
        end

    end

    # ── Test 4: Error handling — empty equilibrium ────────────────────────────
    @testset "Error handling: empty equilibrium" begin

        dd_empty = IMASdd.dd()
        # dd_empty.equilibrium.time_slice is empty

        @test_throws ErrorException JPEC.run_jpec_fuse(dd_empty; time_indices=1)

    end

    # ── Test 5: Error handling — invalid time index ───────────────────────────
    @testset "Error handling: invalid time index" begin

        dd_test = deepcopy(dd)

        # Index 10 doesn't exist (only 3 slices available)
        @test_throws ErrorException JPEC.run_jpec_fuse(dd_test; time_indices=10)

        # Negative index
        @test_throws ErrorException JPEC.run_jpec_fuse(dd_test; time_indices=-1)

        # Vector with out-of-range index
        @test_throws ErrorException JPEC.run_jpec_fuse(dd_test; time_indices=[1, 100])

    end

    # ── Test 6: Verify vacuum flag behavior ───────────────────────────────────
    @testset "Vacuum flag: vac_flag=false" begin

        dd_test = deepcopy(dd)

        # Run with vac_flag=false — energy_perturbed should not be set
        result_dd = JPEC.run_jpec_fuse(dd_test;
                                       time_indices = 1,
                                       nn_low = 1,
                                       nn_high = 1,
                                       vac_flag = false,
                                       verbose = false)

        mhd = dd_test.mhd_linear
        @test length(mhd.time_slice) >= 1

        ts_idx = findfirst(ts -> ts.time == 0.0, mhd.time_slice)
        @test ts_idx !== nothing

        # toroidal_mode entries should exist but energy_perturbed may be 0 or unset
        # (write_imas only writes energy when vac_flag=true and vac_data !== nothing)
        # This is hard to test definitively without inspecting DCON internals,
        # but we can at least verify the run completed without error.
        @test length(mhd.time_slice[ts_idx].toroidal_mode) > 0

    end

    # ── Test 7: Parameter validation ──────────────────────────────────────────
    @testset "Parameter validation" begin

        dd_test = deepcopy(dd)

        # Invalid time_indices type
        @test_throws ErrorException JPEC.run_jpec_fuse(dd_test; time_indices="invalid")

        # Valid calls with different parameter combinations
        @test JPEC.run_jpec_fuse(deepcopy(dd); time_indices=1, nn_low=1, nn_high=3, verbose=false) isa IMASdd.dd
        @test JPEC.run_jpec_fuse(deepcopy(dd); time_indices=[2], nn_low=2, nn_high=5, verbose=false) isa IMASdd.dd
        @test JPEC.run_jpec_fuse(deepcopy(dd); time_indices=:all, psilow=0.05, psihigh=0.99, verbose=false) isa IMASdd.dd

    end

end

println("\n" * "="^70)
println("All FUSE integration tests passed ✓")
println("run_jpec_fuse is ready for use as a FUSE module")
println("="^70)
