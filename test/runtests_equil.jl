
@testset "GeneralizedPerturbedEquilibrium.Equilibrium Unit Tests" begin

    # --- Directory Configuration ---
    # Define the data directory for easy maintenance
    data_dir = joinpath(@__DIR__, "test_data", "regression_equilibrium_example")

    # --- 1. Load EFIT Data (G-EQDSK format) ---
    @testset "Load EFIT Data" begin
        efit_config = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(;
            eq_filename=joinpath(data_dir, "EQDSK_COCOS_02"),
            eq_type="efit",
            jac_type="boozer",
            grid_type="ldp",
            psilow=0.01,
            psihigh=0.994
        )
        global plasma_eq_efit = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(efit_config)

        @test plasma_eq_efit isa GeneralizedPerturbedEquilibrium.Equilibrium.PlasmaEquilibrium
        @test 6.5 < plasma_eq_efit.ro < 7.5 # Physical sanity check
    end

    # --- 2. Load CHEASE Binary Data ---
    @testset "Load CHEASE Binary" begin
        binary_config = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(;
            eq_filename=joinpath(data_dir, "INP1_binary"),
            eq_type="chease_binary",
            jac_type="boozer",
            grid_type="ldp",
            psilow=0.01,
            psihigh=0.994,
            r0exp=6.8,
            b0exp=7.4
        )
        global plasma_eq_binary = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(binary_config)

        @test plasma_eq_binary isa GeneralizedPerturbedEquilibrium.Equilibrium.PlasmaEquilibrium
    end

    # --- 3. Load CHEASE ASCII Data ---
    @testset "Load CHEASE ASCII" begin
        ascii_config = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(;
            eq_filename=joinpath(data_dir, "INP1_ascii"),
            eq_type="chease_ascii",
            jac_type="boozer",
            grid_type="ldp",
            psilow=0.01,
            psihigh=0.994,
            r0exp=6.8,
            b0exp=7.4
        )
        global plasma_eq_ascii = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(ascii_config)

        @test plasma_eq_ascii isa GeneralizedPerturbedEquilibrium.Equilibrium.PlasmaEquilibrium
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

    @testset "Solovev Equilibrium" begin
        # --- Helper constructors ---
        # Minimal valid inputs
        function make_inputs(; mr=4, mz=4, ma=4, e=1.7, a=0.3, r0=1.7, q0=1.0,
            p0fac=1.2, b0fac=1.0, f0fac=1.0)
            equil_inputs = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig()  # or mock/minimal constructor
            sol_inputs = GeneralizedPerturbedEquilibrium.Equilibrium.SolovevConfig(mr, mz, ma, e, a, r0, q0, p0fac, b0fac, f0fac)
            return equil_inputs, sol_inputs
        end

        @testset "sol_run basic functionality" begin
            equil_inputs, sol_inputs = make_inputs()
            dri = GeneralizedPerturbedEquilibrium.Equilibrium.sol_run(equil_inputs, sol_inputs)

            @test isa(dri, GeneralizedPerturbedEquilibrium.Equilibrium.DirectRunInput)
            @test hasproperty(dri, :sq_in)
            @test hasproperty(dri, :psi_in)
            @test hasproperty(dri, :rmin)
            @test hasproperty(dri, :zmax)
        end

        @testset "sol_run clamps p0fac to ≥ 1" begin
            equil_inputs, sol_inputs = make_inputs(p0fac=0.5)
            dri = GeneralizedPerturbedEquilibrium.Equilibrium.sol_run(equil_inputs, sol_inputs)
            @test all(dri.sq_in.y[:, 2] .>= 0)  # no negative pressures
        end

        @testset "sol_run scalar relationships" begin
            equil_inputs, sol_inputs = make_inputs()
            mr, mz, ma = sol_inputs.mr, sol_inputs.mz, sol_inputs.ma
            e, a, r0, q0 = sol_inputs.e, sol_inputs.a, sol_inputs.r0, sol_inputs.q0
            p0fac, b0fac, f0fac = sol_inputs.p0fac, sol_inputs.b0fac, sol_inputs.f0fac

            dri = GeneralizedPerturbedEquilibrium.Equilibrium.sol_run(equil_inputs, sol_inputs)

            # Derived quantities consistency
            f0_expected = r0 * b0fac
            psio_expected = e * f0_expected * a * a / (2 * q0 * r0)

            @test isapprox(dri.psio, psio_expected; rtol=1e-10)

            # Range symmetry
            @test isapprox(dri.zmax, -dri.zmin)
            @test dri.rmax > dri.rmin

            # Proper grid size consistency (coordinates stored in DirectRunInput, not in interpolant)
            @test length(dri.psi_in_xs) == mr + 1
            @test length(dri.psi_in_ys) == mz + 1
        end

        @testset "sol_run spline integrity" begin
            equil_inputs, sol_inputs = make_inputs(mr=6, mz=5, ma=3)
            dri = GeneralizedPerturbedEquilibrium.Equilibrium.sol_run(equil_inputs, sol_inputs)
            sq = dri.sq_in
            psi = dri.psi_in

            # Spline types (using native FastInterpolations)
            @test isa(sq, FastInterpolations.CubicSeriesInterpolant)
            @test isa(psi, FastInterpolations.CubicInterpolantND)

            # Domain monotonicity (coordinates are accessed via different paths)
            @test issorted(sq.cache.x)
            @test issorted(dri.psi_in_xs)  # R coordinates
            @test issorted(dri.psi_in_ys)  # Z coordinates
            # Alternatively via the grids tuple: psi.grids[1] and psi.grids[2]
            @test psi.grids[1] == dri.psi_in_xs
            @test psi.grids[2] == dri.psi_in_ys

            # Check that psi values are finite (stored in nodal_derivs.partials)
            @test all(isfinite, psi.nodal_derivs.partials)
        end

        @testset "sol_run 2D psi field properties" begin
            equil_inputs, sol_inputs = make_inputs(mr=3, mz=3)
            dri = GeneralizedPerturbedEquilibrium.Equilibrium.sol_run(equil_inputs, sol_inputs)
            psi = dri.psi_in

            # Check psi symmetry in z (Solovev equilibrium should be up-down symmetric)
            # nodal_derivs.partials[1, :, :] contains function values on the (R,Z) grid
            # partials[deriv_idx, r_idx, z_idx] where deriv_idx=1 is the function value
            partials = psi.nodal_derivs.partials
            vals_top = partials[1, :, end]      # z = zmax
            vals_bottom = partials[1, :, 1]     # z = zmin
            @test all(abs.(vals_top .- vals_bottom) .< 1e-12)  # nearly symmetric about z=0
        end

        @testset "sol_run parameter sensitivity" begin
            equil_inputs, sol_inputs = make_inputs()
            dri1 = GeneralizedPerturbedEquilibrium.Equilibrium.sol_run(equil_inputs, sol_inputs)
            dri2 = GeneralizedPerturbedEquilibrium.Equilibrium.sol_run(
                equil_inputs,
                GeneralizedPerturbedEquilibrium.Equilibrium.SolovevConfig(sol_inputs.mr, sol_inputs.mz, sol_inputs.ma,
                    sol_inputs.e * 1.1, sol_inputs.a, sol_inputs.r0,
                    sol_inputs.q0, sol_inputs.p0fac, sol_inputs.b0fac,
                    sol_inputs.f0fac)
            )
            @test dri1.psio != dri2.psio  # psio should depend on elongation e
        end

        @testset "sol_run extreme inputs" begin
            # minimal grid (CubicInterpolant requires at least 4 points for extrap BC)
            # mr=3, mz=3 creates 4-point grids (mr+1 points)
            equil_inputs, sol_inputs = make_inputs(mr=3, mz=3, ma=3)
            dri = GeneralizedPerturbedEquilibrium.Equilibrium.sol_run(equil_inputs, sol_inputs)
            @test length(dri.psi_in_xs) == 4
            @test length(dri.psi_in_ys) == 4

            # very high aspect ratio
            equil_inputs, sol_inputs = make_inputs(e=0.8, a=0.1, r0=10.0)
            dri = GeneralizedPerturbedEquilibrium.Equilibrium.sol_run(equil_inputs, sol_inputs)
            @test isfinite(dri.psio)
        end
    end
end
