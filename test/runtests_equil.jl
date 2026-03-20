
@testset "Equilibrium Unit Tests" begin

    # --- Directory Configuration ---
    # Define the data directory for easy maintenance
    data_dir = joinpath(@__DIR__, "test_data", "CHEASE_test_data")

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

    @testset "Edge inverse splines (limited plasma)" begin
        # For a limited plasma, no edge inverse splines are built
        @test isnothing(plasma_eq_efit.profiles.q_spline_iota_inverse)
        @test isnothing(plasma_eq_efit.profiles.dVdpsi_spline_inv)
        @test plasma_eq_efit.params.is_diverted == false
    end

    @testset "Edge inverse splines (diverted plasma)" begin
        # Load the DIIID-like diverted equilibrium
        diiid_dir = joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")
        diiid_config = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(;
            eq_type="efit",
            eq_filename=joinpath(diiid_dir, "TkMkr_D3Dlike_Hmode.geqdsk"),
            jac_type="hamada",
            grid_type="ldp",
            psilow=1e-4,
            psihigh=0.993,
            mpsi=128,
            mtheta=256
        )
        pe_diverted = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(diiid_config)

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
