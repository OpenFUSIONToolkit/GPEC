
@testset "JPEC.Equilibrium Unit Tests" begin

    # --- Directory Configuration ---
    # Define the data directory for easy maintenance
    data_dir = joinpath(@__DIR__, "test_data", "regression_equilibrium_example")

    # --- 1. Load EFIT Data (G-EQDSK format) ---
    @testset "Load EFIT Data" begin
        efit_control = JPEC.Equilibrium.EquilibriumControl(;
            eq_filename=joinpath(data_dir, "EQDSK_COCOS_02"),
            eq_type="efit",
            jac_type="boozer",
            grid_type="ldp",
            psilow=0.01,
            psihigh=0.994
        )
        efit_config = JPEC.Equilibrium.EquilibriumConfig(efit_control, JPEC.Equilibrium.EquilibriumOutput())
        global plasma_eq_efit = JPEC.Equilibrium.setup_equilibrium(efit_config)

        @test plasma_eq_efit isa JPEC.Equilibrium.PlasmaEquilibrium
        @test 6.5 < plasma_eq_efit.ro < 7.5 # Physical sanity check
    end

    # # --- 2. Load CHEASE Binary Data ---
    # @testset "Load CHEASE Binary" begin
    #     binary_control = JPEC.Equilibrium.EquilibriumControl(;
    #         eq_filename=joinpath(data_dir, "INP1_binary"),
    #         eq_type="chease_binary",
    #         jac_type="boozer",
    #         grid_type="ldp",
    #         psilow=0.01,
    #         psihigh=0.994,
    #         r0exp=6.8,
    #         b0exp=7.4
    #     )
    #     chease_config_chease_binary = JPEC.Equilibrium.EquilibriumConfig(binary_control, JPEC.Equilibrium.EquilibriumOutput())
    #     global plasma_eq_binary = JPEC.Equilibrium.setup_equilibrium(chease_config_chease_binary)

    #     @test plasma_eq_binary isa JPEC.Equilibrium.PlasmaEquilibrium
    # end

    # # --- 3. Load CHEASE ASCII Data ---
    # @testset "Load CHEASE ASCII" begin
    #     ascii_control = JPEC.Equilibrium.EquilibriumControl(;
    #         eq_filename=joinpath(data_dir, "INP1_ascii"),
    #         eq_type="chease_ascii",
    #         jac_type="boozer",
    #         grid_type="ldp",
    #         psilow=0.01,
    #         psihigh=0.994,
    #         r0exp=6.8,
    #         b0exp=7.4
    #     )
    #     chease_config_chease_ascii = JPEC.Equilibrium.EquilibriumConfig(ascii_control, JPEC.Equilibrium.EquilibriumOutput())
    #     global plasma_eq_ascii = JPEC.Equilibrium.setup_equilibrium(chease_config_chease_ascii)

    #     @test plasma_eq_ascii isa JPEC.Equilibrium.PlasmaEquilibrium
    # end

    # # ----------------------------------------------------------------------
    # # Further Validation Tests using the loaded data
    # # ----------------------------------------------------------------------

    # @testset "CHEASE Consistency (ASCII vs Binary)" begin
    #     # Tolerance set to 1e-12 as these come from the same physical source
    #     tol = 1e-12

    #     @testset "Magnetic Axis" begin
    #         @test isapprox(plasma_eq_ascii.ro, plasma_eq_binary.ro, atol=tol)
    #         @test isapprox(plasma_eq_ascii.zo, plasma_eq_binary.zo, atol=tol)
    #     end

    #     # TODO: Rewrite using native FastInterpolations evaluation of rzphi interpolants
    #     # @testset "Spline Evaluation Consistency" begin
    #     #     ... (test removed - rzphi is now multiple CubicInterpolantND instances)
    #     # end
    # end

    # @testset "Data Source Comparison (EFIT vs CHEASE)" begin
    #     # Verify consistency between different code outputs (EFIT vs CHEASE)
    #     # Higher tolerance (0.05m) allowed for different physics kernels
    #     @test isapprox(plasma_eq_efit.ro, plasma_eq_ascii.ro, atol=0.05)
    #     @test isapprox(plasma_eq_efit.zo, plasma_eq_ascii.zo, atol=1e-3)
    # end

end
