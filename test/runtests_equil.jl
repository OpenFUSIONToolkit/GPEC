
@testset "Equilibrium Unit Tests" begin

    data_dir = joinpath(@__DIR__, "test_data", "regression_equilibrium_example")

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

    @testset "Load EFIT Data (arclength)" begin
        arclength_config = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(;
            eq_filename=joinpath(data_dir, "EQDSK_COCOS_02"),
            eq_type="efit_arclength",
            jac_type="boozer",
            grid_type="ldp",
            psilow=0.01,
            psihigh=0.994
        )
        global plasma_eq_arclength = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(arclength_config)

        @test plasma_eq_arclength isa GeneralizedPerturbedEquilibrium.Equilibrium.PlasmaEquilibrium
        @test 6.5 < plasma_eq_arclength.ro < 7.5

        # Magnetic axis should agree with the standard efit solver within 1 mm
        @test isapprox(plasma_eq_arclength.ro, plasma_eq_efit.ro, atol=1e-3)
        @test isapprox(plasma_eq_arclength.zo, plasma_eq_efit.zo, atol=1e-3)

        # B field must be finite and positive
        B_nodes = plasma_eq_arclength.eqfun_B.nodal_derivs.partials[1, :, :]
        @test all(isfinite, B_nodes)
        @test all(>(0), B_nodes)
    end

    @testset "Load EFIT Data (by_inversion)" begin
        inversion_config = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(;
            eq_filename=joinpath(data_dir, "EQDSK_COCOS_02"),
            eq_type="efit_by_inversion",
            jac_type="boozer",
            grid_type="ldp",
            psilow=0.01,
            psihigh=0.994
        )
        global plasma_eq_inversion = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(inversion_config)

        @test plasma_eq_inversion isa GeneralizedPerturbedEquilibrium.Equilibrium.PlasmaEquilibrium
        @test 6.5 < plasma_eq_inversion.ro < 7.5

        # Magnetic axis should agree with the standard efit solver within 1 mm
        @test isapprox(plasma_eq_inversion.ro, plasma_eq_efit.ro, atol=1e-3)
        @test isapprox(plasma_eq_inversion.zo, plasma_eq_efit.zo, atol=1e-3)

        # B field must be finite and positive
        B_nodes = plasma_eq_inversion.eqfun_B.nodal_derivs.partials[1, :, :]
        @test all(isfinite, B_nodes)
        @test all(>(0), B_nodes)
    end

    @testset "EFIT Method Consistency" begin
        # All three methods solve the same equilibrium — q-profiles should broadly agree.
        # Tolerance is 5% to allow for method-specific discretisation differences.
        q_efit      = plasma_eq_efit.profiles.q_spline.y
        q_arclength = plasma_eq_arclength.profiles.q_spline.y
        q_inversion = plasma_eq_inversion.profiles.q_spline.y

        @test all(isfinite, q_arclength)
        @test all(>(0), q_arclength)
        @test all(isfinite, q_inversion)
        @test all(>(0), q_inversion)

        # Edge q values should agree to within 10%
        @test isapprox(q_arclength[end], q_efit[end], rtol=0.10)
        @test isapprox(q_inversion[end], q_efit[end], rtol=0.10)
    end

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

    # These tests are specifically designed to catch the theta radians/turns unit bug:
    # if rz_in_ys is passed in radians instead of turns, deta is corrupted (subtracting
    # values up to 2π from values in [0,0.5]), causing physically impossible B fields
    # and wildly non-constant r² on what should be closed flux surfaces.
    @testset "CHEASE Physical Validation" begin
        b0exp = 7.4  # CHEASE normalization field [T]

        B_nodes_binary = plasma_eq_binary.eqfun_B.nodal_derivs.partials[1, :, :]
        B_nodes_ascii  = plasma_eq_ascii.eqfun_B.nodal_derivs.partials[1, :, :]

        # B field must be finite and positive everywhere
        @test all(isfinite, B_nodes_binary)
        @test all(isfinite, B_nodes_ascii)
        @test all(>(0), B_nodes_binary)
        @test all(>(0), B_nodes_ascii)

        # B field must be within a factor of 3 of the normalization field
        @test all(B_nodes_binary .> b0exp / 3)
        @test all(B_nodes_binary .< b0exp * 3)
        @test all(B_nodes_ascii .> b0exp / 3)
        @test all(B_nodes_ascii .< b0exp * 3)

        # q must be finite, positive, and in a physically reasonable range
        q_binary = plasma_eq_binary.profiles.q_spline.y
        q_ascii  = plasma_eq_ascii.profiles.q_spline.y
        @test all(isfinite, q_binary)
        @test all(isfinite, q_ascii)
        @test all(>(0), q_binary)
        @test all(>(0), q_ascii)
        @test all(q_binary .< 20)
        @test all(q_ascii .< 20)
    end

    @testset "LAR Equilibrium" begin
        # Build a zeroth-order (no Shafranov shift) LAR equilibrium. With zeroth=true,
        # surfaces are exactly circular: (R - R0)² + Z² = r² for each flux surface.
        lar_config = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(;
            eq_type="lar",
            jac_type="boozer",
            grid_type="ldp",
            psilow=0.01,
            psihigh=0.99
        )
        lar_input = GeneralizedPerturbedEquilibrium.Equilibrium.LargeAspectRatioConfig(;
            lar_r0=10.0, lar_a=1.0, q0=1.5, beta0=1e-4,
            mtau=64, ma=64, zeroth=true
        )
        plasma_eq_lar = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(lar_config, lar_input)

        @test plasma_eq_lar isa GeneralizedPerturbedEquilibrium.Equilibrium.PlasmaEquilibrium

        # B field must be finite, positive, and near F/R ≈ 1 T (LAR default b0fac=1, R0=10)
        B_nodes = plasma_eq_lar.eqfun_B.nodal_derivs.partials[1, :, :]
        @test all(isfinite, B_nodes)
        @test all(>(0), B_nodes)
        @test all(B_nodes .> 0.3)   # at least 0.3 T everywhere
        @test all(B_nodes .< 5.0)   # no more than 5 T (far from axis, F/R grows modestly)

        # For zeroth-order LAR, flux surfaces are exactly circular, so rzphi_rsquared
        # must be nearly constant over theta on each surface. The radians/turns bug
        # causes O(1) fractional errors here since deta is corrupted by subtracting
        # values up to 2π from values in [0,0.5].
        r2_nodes = plasma_eq_lar.rzphi_rsquared.nodal_derivs.partials[1, :, :]
        for ipsi in axes(r2_nodes, 1)
            r2_row = r2_nodes[ipsi, :]
            r2_mean = sum(r2_row) / length(r2_row)
            r2_mean > 0 || continue  # skip axis region
            relative_variation = (maximum(r2_row) - minimum(r2_row)) / r2_mean
            @test relative_variation < 1e-3
        end

        # q-profile must be positive and monotone
        q_lar = plasma_eq_lar.profiles.q_spline.y
        @test all(>(0), q_lar)
        @test issorted(q_lar)
    end

    @testset "Solovev Equilibrium" begin
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
