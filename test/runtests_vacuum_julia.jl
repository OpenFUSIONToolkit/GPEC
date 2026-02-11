using Test
using JPEC
using JPEC.Vacuum
using LinearAlgebra

@testset "Vacuum.jl Unit Tests" begin

    # ----------------------------------------------------------------------
    @testset "VacuumStructs.jl" begin
        @testset "Constructors" begin
            # Test default constructor for VacuumInput
            vac_in = VacuumInput()
            @test vac_in.mlow == 0
            @test vac_in.n == 0
            @test vac_in.kernelsign == 1.0
            @test vac_in.mtheta == 1
            @test vac_in.force_wv_symmetry == true

            # Test default constructor for WallShapeSettings
            wall_set = WallShapeSettings()
            @test wall_set.shape == "nowall"
            @test wall_set.a == 0.3
        end

        @testset "initialize_plasma_surface" begin
            inputs = VacuumInput(
                r=[1.0, 1.1, 1.2, 1.1],
                z=[0.0, 0.1, 0.0, -0.1],
                ν=zeros(4),
                mtheta=4,
                mpert=1,
                mlow=1,
                n=1
            )
            plasma_surf = JPEC.Vacuum.initialize_plasma_surface(inputs)
            @test length(plasma_surf.x) == 4
            @test length(plasma_surf.z) == 4
            @test length(plasma_surf.delta) == 4
            @test length(plasma_surf.dx_dtheta) == 4
            @test length(plasma_surf.dz_dtheta) == 4
            @test !any(isnan, plasma_surf.dx_dtheta)
            @test !any(isnan, plasma_surf.dz_dtheta)
        end

        @testset "initialize_wall" begin
            inputs = VacuumInput(mtheta=16)
            plasma_surf = JPEC.Vacuum.initialize_plasma_surface(
                VacuumInput(
                    r=1.7 .+ 0.3 .* cos.(range(0, 2pi, length=16)),
                    z=0.3 .* sin.(range(0, 2pi, length=16)),
                    ν=zeros(16),
                    mtheta=16
                )
            )

            # Test "nowall"
            wall_settings = WallShapeSettings(shape="nowall")
            wall_geo = JPEC.Vacuum.initialize_wall(inputs, plasma_surf, wall_settings)
            @test wall_geo.nowall == true

            # Test "conformal" wall
            wall_settings = WallShapeSettings(shape="conformal", a=0.2)
            wall_geo = JPEC.Vacuum.initialize_wall(inputs, plasma_surf, wall_settings)
            @test wall_geo.nowall == false
            @test length(wall_geo.x) == 16
            @test !any(isnan, wall_geo.x)

            # Test that wall with R <= 0 throws an error
            # Create a plasma surface very close to R=0
            plasma_surf_near_zero = JPEC.Vacuum.initialize_plasma_surface(
                VacuumInput(
                    r=0.05 .+ 0.03 .* cos.(range(0, 2pi, length=16)),
                    z=0.03 .* sin.(range(0, 2pi, length=16)),
                    ν=zeros(16),
                    mtheta=16
                )
            )
            # Use a "dee" wall shape with parameters that will produce R < 0
            # Setting cw (offset) to a large negative value will shift the wall left past R=0
            wall_settings_negative = WallShapeSettings(shape="dee", cw=-1.5, a=0.1)
            @test_throws ErrorException JPEC.Vacuum.initialize_wall(inputs, plasma_surf_near_zero, wall_settings_negative)

            # Test that conformal wall R-coordinates are clamped by centerstack_min
            # With a very large 'a' parameter, a conformal wall would naturally go to R < 0,
            # but it should be clamped to centerstack_min = min(0.1, 0.1 * minimum(x_plasma))
            wall_settings_large_a = WallShapeSettings(shape="conformal", a=10.0, equal_arc_wall=false)
            wall_geo_clamped = JPEC.Vacuum.initialize_wall(inputs, plasma_surf_near_zero, wall_settings_large_a)
            expected_min = min(0.1, 0.1 * minimum(plasma_surf_near_zero.x))
            @test all(wall_geo_clamped.x .>= expected_min)
            @test any(wall_geo_clamped.x .<= expected_min + 1e-10)  # At least one point should be at the minimum
        end

        @testset "distribute_to_equal_arc_grid" begin
            # A simple circle (note the function assumes periodic shapes with open loop endpoints)
            theta = range(0, step=2pi/10, length=10)
            xin_circ = cos.(theta)
            zin_circ = sin.(theta)
            xout_circ, zout_circ, _, _, _ = JPEC.Vacuum.distribute_to_equal_arc_grid(xin_circ, zin_circ, 10)
            # The points should still be on the unit circle
            @test all(r -> isapprox(r, 1.0, atol=1e-9), sqrt.(xout_circ .^ 2 + zout_circ .^ 2))
        end
    end

    # ----------------------------------------------------------------------
    @testset "VacuumInternals.jl - Math" begin
        @testset "Elliptic Integrals" begin
            # Test domain
            @test_throws DomainError JPEC.Vacuum.elliptic_integral_k(-0.1)
            @test_throws DomainError JPEC.Vacuum.elliptic_integral_e(1.1)

            # Test some values (these are just for regression, not from a known source)
            @test JPEC.Vacuum.elliptic_integral_k(0.5) isa Float64
            @test JPEC.Vacuum.elliptic_integral_e(0.5) isa Float64
            @test !isnan(JPEC.Vacuum.elliptic_integral_k(0.5))
            @test !isnan(JPEC.Vacuum.elliptic_integral_e(0.5))
        end

        @testset "Legendre Functions (Pn_minus_half)" begin
            s = 1.5
            n = 3
            P_1997 = JPEC.Vacuum.Pn_minus_half_1997(s, n)
            @test length(P_1997) == n + 2
            @test !any(isnan, P_1997)

            # Test Pn_minus_half_2007 implementation
            P_2007 = JPEC.Vacuum.Pn_minus_half_2007(s, n)
            @test length(P_2007) == n + 2
            @test !any(isnan, P_2007)

            # For small n and reasonable s, 1997 and 2007 should agree closely
            @test isapprox(P_1997[1], P_2007[1], rtol=1e-7)  # P^0
            @test isapprox(P_1997[2], P_2007[2], rtol=1e-7)  # P^1
        end

        @testset "Bulirsch Elliptic Integrals" begin
            # Test the new Bulirsch algorithm
            m1 = 0.5
            K, E, conv, iters = JPEC.Vacuum.elliptic_integrals_bulirsch(m1)
            @test K isa Float64
            @test E isa Float64
            @test !isnan(K)
            @test !isnan(E)
            @test conv < 1e-10  # Should converge well

            # Test domain error
            @test_throws DomainError JPEC.Vacuum.elliptic_integrals_bulirsch(-0.1)
            @test_throws DomainError JPEC.Vacuum.elliptic_integrals_bulirsch(1.5)
        end

        @testset "green function" begin
            # Test with simple, non-singular case
            G_n, coupling_n, coupling_0 = JPEC.Vacuum.green(2.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1)
            @test G_n isa Float64
            @test coupling_n isa Float64
            @test coupling_0 isa Float64
            @test !isnan(G_n)
            @test !isnan(coupling_n)
        end
    end

    # ----------------------------------------------------------------------
    @testset "VacuumInternals.jl - Interpolation/Grid" begin
        @testset "lagrange1d" begin
            ax = [1.0, 2.0, 3.0, 4.0]
            af = [2.0, 4.0, 6.0, 8.0] # f(x) = 2x
            m = 4
            nl = 3

            # Test value
            f, df = JPEC.Vacuum.lagrange1d(ax, af, m, nl, 2.5, 0)
            @test isapprox(f, 5.0, atol=1e-12)

            # Test value and derivative
            f, df = JPEC.Vacuum.lagrange1d(ax, af, m, nl, 2.5, 1)
            @test isapprox(f, 5.0, atol=1e-12)
            @test isapprox(df, 2.0, atol=1e-12)
        end

        @testset "interp_to_new_grid" begin
            # Test upsampling a sine wave
            mtheta_in = 17
            theta_in = collect(range(0, 1, length=mtheta_in))
            vecin = sin.(2π .* theta_in)

            mtheta_out = 33
            vecout = JPEC.Vacuum.interp_to_new_grid(vecin, mtheta_out)

            theta_out = (0:(mtheta_out-1)) ./ mtheta_out
            expected_out = sin.(2π .* theta_out)

            @test isapprox(vecout, expected_out, atol=1e-2)

            # Test no-op
            vecout_noop = JPEC.Vacuum.interp_to_new_grid(vecin, mtheta_in)
            @test vecout_noop == vecin
        end

        @testset "periodic_cubic_deriv" begin
            theta = range(0, 2pi, length=101)[1:(end-1)]
            vals = sin.(theta)
            derivs = JPEC.Vacuum.periodic_cubic_deriv(theta, vals)
            @test isapprox(derivs, cos.(theta), atol=1e-3)
        end

        @testset "_create_pickup_grid" begin
            R_grid = [1, 2]
            Z_grid = [3, 4, 5]
            R_points, Z_points = JPEC.Vacuum._create_pickup_grid(R_grid, Z_grid)
            @test R_points == [1, 1, 1, 2, 2, 2]
            @test Z_points == [3, 4, 5, 3, 4, 5]
        end
    end

    # ----------------------------------------------------------------------
    @testset "VacuumInternals.jl - Fourier" begin
        @testset "fourier_transform!" begin
            mtheta, mpert = 4, 2
            gil = zeros(mtheta, mpert)
            gij = [1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0; 9.0 10.0 11.0 12.0; 13.0 14.0 15.0 16.0]
            cs = zeros(mtheta, mpert)
            cs[:, 1] .= 1.0 # Simple basis

            JPEC.Vacuum.fourier_transform!(gil, gij, cs, 0, 0)

            # gil should be the sum of columns of gij
            @test gil[:, 1] == [10.0, 26.0, 42.0, 58.0]
            @test gil[:, 2] == [0.0, 0.0, 0.0, 0.0]
        end

        @testset "fourier_inverse_transform!" begin
            mtheta, mpert = 4, 2
            dth = 2π / mtheta
            gil = zeros(mtheta, mpert)
            gil[:, 1] = [1.0, 2.0, 3.0, 4.0]
            cs = zeros(mtheta, mpert)
            cs[:, 1] .= 1.0
            gll = zeros(mpert, mpert)

            JPEC.Vacuum.fourier_inverse_transform!(gll, gil, cs, 0, 0)

            # gll[1,1] should be sum(cs[:,1] .* gil[:,1]) * 2pi*dth
            expected = sum([1.0, 2.0, 3.0, 4.0]) * dth * 2pi
            @test isapprox(gll[1, 1], expected)
        end
    end

    # ----------------------------------------------------------------------
    @testset "Vacuum.jl - High Level" begin
        @testset "compute_vacuum_response_nowall" begin
            # A simple integration test for compute_vacuum_response
            mtheta = 128
            mtheta_eq = 17
            r_eq = 1.7 .+ 0.3 .* cos.(range(0, 2pi, length=mtheta_eq))
            z_eq = 0.3 .* sin.(range(0, 2pi, length=mtheta_eq))

            inputs = VacuumInput(
                r=collect(r_eq),
                z=collect(z_eq),
                ν=zeros(mtheta_eq),
                mlow=1,
                mpert=2,
                n=1,
                mtheta=mtheta
            )
            wall_settings = WallShapeSettings(shape="nowall")

            wv, grri, grre, xzpts = compute_vacuum_response(inputs, wall_settings)

            @test size(wv) == (2, 2);
            @test !any(isnan, wv);
            @test size(grri) == (2 * 128, 2 * 2);
            @test !any(isnan, grri);
            @test size(grre) == (2 * 128, 2 * 2);
            @test !any(isnan, grre);
            @test size(xzpts) == (128, 4);
            @test !any(isnan, xzpts[:, 1:2]);
        end

        @testset "compute_vacuum_response_conformal" begin
            # A simple integration test for compute_vacuum_response with a conformal wall
            mtheta = 128
            mtheta_eq = 17
            r_eq = 1.7 .+ 0.3 .* cos.(range(0, 2pi, length=mtheta_eq))
            z_eq = 0.3 .* sin.(range(0, 2pi, length=mtheta_eq))

            inputs = VacuumInput(
                r=collect(r_eq),
                z=collect(z_eq),
                ν=zeros(mtheta_eq),
                mlow=1,
                mpert=2,
                n=1,
                mtheta=mtheta
            )
            # Use a conformal wall
            wall_settings = WallShapeSettings(shape="conformal", a=0.5)

            wv, grri, grre, xzpts = compute_vacuum_response(inputs, wall_settings)

            @test size(wv) == (2, 2)
            @test !any(isnan, wv)
            @test size(grri) == (2 * 128, 2 * 2)
            @test !any(isnan, grri)
            @test size(xzpts) == (128, 4)
            # For a conformal wall, wall coordinates should exist and not be NaN
            @test !any(isnan, xzpts)

            # Check that wall coordinates are different from plasma coordinates
            @test !isapprox(xzpts[:, 1], xzpts[:, 3])
            @test !isapprox(xzpts[:, 2], xzpts[:, 4])
        end
    end

end
