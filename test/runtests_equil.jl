
@testset "JPEC.Equilibrium Unit Tests" begin

    # --- Directory Configuration ---
    # Define the data directory for easy maintenance
    data_dir = joinpath(@__DIR__, "test_data", "CHEASE_test_data")

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
        # Both formats encode the same physical data; differences arise only from
        # floating-point text serialization in the ASCII format vs exact binary storage.
        # We use rtol=1e-6 throughout as a conservative lower bound on ASCII precision.
        rtol = 1e-6

        @testset "Magnetic axis" begin
            @test isapprox(plasma_eq_ascii.ro, plasma_eq_binary.ro; rtol)
            @test isapprox(plasma_eq_ascii.zo, plasma_eq_binary.zo; rtol)
            @test isapprox(plasma_eq_ascii.psio, plasma_eq_binary.psio; rtol)
        end

        @testset "Global parameters" begin
            @test isapprox(plasma_eq_ascii.params.q0, plasma_eq_binary.params.q0; rtol)
            @test isapprox(plasma_eq_ascii.params.qmin, plasma_eq_binary.params.qmin; rtol)
            @test isapprox(plasma_eq_ascii.params.qmax, plasma_eq_binary.params.qmax; rtol)
            @test isapprox(plasma_eq_ascii.params.qa, plasma_eq_binary.params.qa; rtol)
            @test isapprox(plasma_eq_ascii.params.bt0, plasma_eq_binary.params.bt0; rtol)
            @test isapprox(plasma_eq_ascii.params.crnt, plasma_eq_binary.params.crnt; rtol)
        end

        @testset "1D profiles" begin
            @test isapprox(plasma_eq_ascii.profiles.q_spline.y, plasma_eq_binary.profiles.q_spline.y; rtol)
            @test isapprox(plasma_eq_ascii.profiles.F_spline.y, plasma_eq_binary.profiles.F_spline.y; rtol)
            @test isapprox(plasma_eq_ascii.profiles.P_spline.y, plasma_eq_binary.profiles.P_spline.y; rtol)
            @test isapprox(plasma_eq_ascii.profiles.dVdpsi_spline.y, plasma_eq_binary.profiles.dVdpsi_spline.y; rtol)
        end

        @testset "Flux surface geometry" begin
            ascii_rfac2 = plasma_eq_ascii.rzphi_rsquared.nodal_derivs.partials[1, :, :]
            binary_rfac2 = plasma_eq_binary.rzphi_rsquared.nodal_derivs.partials[1, :, :]
            ascii_offset = plasma_eq_ascii.rzphi_offset.nodal_derivs.partials[1, :, :]
            binary_offset = plasma_eq_binary.rzphi_offset.nodal_derivs.partials[1, :, :]
            @test isapprox(ascii_rfac2, binary_rfac2; rtol)
            @test isapprox(ascii_offset, binary_offset; rtol)
        end
    end

    @testset "Data Source Comparison (EFIT vs CHEASE)" begin
        # Verify consistency between different code outputs (EFIT vs CHEASE)
        # Higher tolerance (0.05m) allowed for different physics kernels
        @test isapprox(plasma_eq_efit.ro, plasma_eq_ascii.ro, atol=0.05)
        @test isapprox(plasma_eq_efit.zo, plasma_eq_ascii.zo, atol=1e-3)
    end

end
