
@testset "JPEC.Equilibrium Unit Tests" begin

    # --- Directory Configuration ---
    # Define the data directory for easy maintenance
    data_dir = joinpath(@__DIR__, "test_data", "regression_equilibrium_example")

    # --- 1. Load EFIT Data (G-EQDSK format) ---
    @testset "Load EFIT Data" begin
        efit_config = JPEC.Equilibrium.EquilibriumConfig(;
            eq_filename=joinpath(data_dir, "EQDSK_COCOS_02"),
            eq_type="efit",
            jac_type="boozer",
            grid_type="ldp",
            psilow=0.01,
            psihigh=0.994
        )
        global plasma_eq_efit = JPEC.Equilibrium.setup_equilibrium(efit_config)

        @test plasma_eq_efit isa JPEC.Equilibrium.PlasmaEquilibrium
        @test 6.5 < plasma_eq_efit.ro < 7.5 # Physical sanity check
    end

    # --- 2. Load CHEASE Binary Data ---
    @testset "Load CHEASE Binary" begin
        binary_config = JPEC.Equilibrium.EquilibriumConfig(;
            eq_filename=joinpath(data_dir, "INP1_binary"),
            eq_type="chease_binary",
            jac_type="boozer",
            grid_type="ldp",
            psilow=0.01,
            psihigh=0.994,
            r0exp=6.8,
            b0exp=7.4
        )
        global plasma_eq_binary = JPEC.Equilibrium.setup_equilibrium(binary_config)

        @test plasma_eq_binary isa JPEC.Equilibrium.PlasmaEquilibrium
    end

    # --- 3. Load CHEASE ASCII Data ---
    @testset "Load CHEASE ASCII" begin
        ascii_config = JPEC.Equilibrium.EquilibriumConfig(;
            eq_filename=joinpath(data_dir, "INP1_ascii"),
            eq_type="chease_ascii",
            jac_type="boozer",
            grid_type="ldp",
            psilow=0.01,
            psihigh=0.994,
            r0exp=6.8,
            b0exp=7.4
        )
        global plasma_eq_ascii = JPEC.Equilibrium.setup_equilibrium(ascii_config)

        @test plasma_eq_ascii isa JPEC.Equilibrium.PlasmaEquilibrium
    end

    # ----------------------------------------------------------------------
    # Further Validation Tests using the loaded data
    # ----------------------------------------------------------------------

    @testset "CHEASE Consistency (ASCII vs Binary)" begin
        # Tolerance set to 1e-12 as these come from the same physical source
        tol = 1e-12

        @testset "Magnetic Axis" begin
            @test isapprox(plasma_eq_ascii.ro, plasma_eq_binary.ro, atol=tol)
            @test isapprox(plasma_eq_ascii.zo, plasma_eq_binary.zo, atol=tol)
        end
    end

    @testset "Data Source Comparison (EFIT vs CHEASE)" begin
        # Verify consistency between different code outputs (EFIT vs CHEASE)
        # Higher tolerance (0.05m) allowed for different physics kernels
        @test isapprox(plasma_eq_efit.ro, plasma_eq_ascii.ro, atol=0.05)
        @test isapprox(plasma_eq_efit.zo, plasma_eq_ascii.zo, atol=1e-3)
    end

    @testset "Edge inverse splines (limited plasma)" begin
        # For a limited plasma, no edge inverse splines are built
        @test isnothing(plasma_eq_efit.profiles.q_spline_iota_inverse)
        @test isnothing(plasma_eq_efit.profiles.dVdpsi_spline_inv)
        @test plasma_eq_efit.params.is_diverted == false
    end

    @testset "Edge inverse splines (diverted plasma)" begin
        # Load the DIIID-like diverted equilibrium
        diiid_dir = joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")
        diiid_config = JPEC.Equilibrium.EquilibriumConfig(;
            eq_type="efit",
            eq_filename=joinpath(diiid_dir, "TKMKR_D3Dlike_default_Hmode.geqdsk"),
            jac_type="hamada",
            grid_type="ldp",
            psilow=1e-4,
            psihigh=0.993,
            mpsi=128,
            mtheta=256
        )
        pe_diverted = JPEC.Equilibrium.setup_equilibrium(diiid_config)

        psihigh = pe_diverted.config.psihigh
        ics = pe_diverted.profiles.q_spline_iota_inverse

        # X-point detection
        @test pe_diverted.params.is_diverted == true
        @test pe_diverted.params.z_xpoint < pe_diverted.zo   # lower null is below the axis

        # Edge splines are built
        @test !isnothing(ics)
        @test !isnothing(pe_diverted.profiles.dVdpsi_spline_inv)

        # qa = Inf for diverted
        @test isinf(pe_diverted.params.qa)

        # Anchor: iota(1.0) = 0 (separatrix)
        @test ics.inner.y[end] ≈ 0.0

        # Coverage: edge spline starts in the near-edge region (between 85% and 100% of psihigh)
        @test 0.85 * psihigh < ics.inner.cache.x[1] < psihigh

        # Continuity at psihigh: q and dV/dψ match to machine precision
        @test ics(psihigh) ≈ pe_diverted.profiles.q_spline_direct(psihigh)
        @test pe_diverted.profiles.dVdpsi_spline_inv(psihigh) ≈ pe_diverted.profiles.dVdpsi_spline(psihigh)
    end

end
