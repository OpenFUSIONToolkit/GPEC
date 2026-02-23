@testset "Vacuum.jl Unit Tests" begin

    @testset "Vacuum.jl (2D)" begin

        # -------------------------------------------------------------------------
        @testset "VacuumInput" begin
            @testset "default constructor" begin
                vac = VacuumInput()
                @test vac.r == Float64[]
                @test vac.z == Float64[]
                @test vac.ν == Float64[]
                @test vac.mlow == 0
                @test vac.mpert == 0
                @test vac.nlow == 0
                @test vac.npert == 0
                @test vac.mtheta == 1
                @test vac.nzeta == 1
                @test vac.force_wv_symmetry == true
            end

            @testset "keyword constructor" begin
                vac = VacuumInput(mtheta=32, mpert=3, mlow=1, nlow=2, npert=2, nzeta=1)
                @test vac.mtheta == 32
                @test vac.mpert == 3
                @test vac.mlow == 1
                @test vac.nlow == 2
                @test vac.npert == 2
                @test vac.nzeta == 1
            end
        end

        # -------------------------------------------------------------------------
        @testset "WallShapeSettings" begin
            @testset "default constructor" begin
                w = WallShapeSettings()
                @test w.shape == "nowall"
                @test w.a == 0.3
                @test w.equal_arc_wall == true
            end

            @testset "keyword constructor" begin
                w = WallShapeSettings(shape="conformal", a=0.2, equal_arc_wall=false)
                @test w.shape == "conformal"
                @test w.a == 0.2
                @test w.equal_arc_wall == false
            end
        end

        # -------------------------------------------------------------------------
        @testset "PlasmaGeometry" begin
            @testset "from VacuumInput" begin
                inputs = VacuumInput(
                    r=[1.0, 1.1, 1.2, 1.1, 1.0],
                    z=[0.0, 0.1, 0.0, -0.1, 0.0],
                    ν=zeros(5),
                    mtheta=5,
                    mpert=1,
                    mlow=1,
                    nlow=1,
                    npert=1,
                    nzeta=1
                )
                surf = JPEC.Vacuum.PlasmaGeometry(inputs)
                @test length(surf.x) == 5
                @test length(surf.z) == 5
                @test length(surf.ν) == 5
                # Interpolation onto [0, 2π) grid: first point should be near inputs.r[1], inputs.z[1]
                @test isapprox(surf.x[1], 1.0, atol=0.02)
                @test isapprox(surf.z[1], 0.0, atol=0.02)
            end

            @testset "edge: mtheta larger than input length" begin
                # Periodic spline requires at least 4 points
                inputs = VacuumInput(
                    r=[1.0, 1.1, 1.2, 1.1, 1.0],
                    z=[0.0, 0.1, 0.0, -0.1, 0.0],
                    ν=zeros(5),
                    mtheta=8,
                    mpert=1,
                    mlow=0,
                    nlow=0,
                    npert=1,
                    nzeta=1
                )
                surf = JPEC.Vacuum.PlasmaGeometry(inputs)
                @test length(surf.x) == 8
                @test all(isfinite, surf.x)
                @test all(isfinite, surf.z)
            end
        end

        # -------------------------------------------------------------------------
        @testset "WallGeometry" begin
            _circle_inputs(mtheta) = VacuumInput(
                r=1.7 .+ 0.3 .* cos.(range(0, 2π, length=mtheta)),
                z=0.3 .* sin.(range(0, 2π, length=mtheta)),
                ν=zeros(mtheta),
                mtheta=mtheta,
                nzeta=1
            )

            @testset "nowall" begin
                inputs = _circle_inputs(16)
                plasma_surf = JPEC.Vacuum.PlasmaGeometry(inputs)
                wall_settings = WallShapeSettings(shape="nowall")
                wall = JPEC.Vacuum.WallGeometry(inputs, plasma_surf, wall_settings)
                @test wall.nowall == true
                @test length(wall.x) == 16
                @test length(wall.z) == 16
            end

            @testset "conformal" begin
                inputs = _circle_inputs(16)
                plasma_surf = JPEC.Vacuum.PlasmaGeometry(inputs)
                wall_settings = WallShapeSettings(shape="conformal", a=0.2)
                wall = JPEC.Vacuum.WallGeometry(inputs, plasma_surf, wall_settings)
                @test wall.nowall == false
                @test length(wall.x) == 16
                @test length(wall.z) == 16
                @test all(isfinite, wall.x)
                @test all(isfinite, wall.z)
                # Conformal wall is offset outward from plasma
                @test !isapprox(wall.x, plasma_surf.x)
                @test !isapprox(wall.z, plasma_surf.z)
            end

            @testset "elliptical" begin
                inputs = _circle_inputs(16)
                plasma_surf = JPEC.Vacuum.PlasmaGeometry(inputs)
                wall_settings = WallShapeSettings(shape="elliptical", a=0.5)
                wall = JPEC.Vacuum.WallGeometry(inputs, plasma_surf, wall_settings)
                @test wall.nowall == false
                @test length(wall.x) == 16
                @test all(isfinite, wall.x)
                @test all(isfinite, wall.z)
            end

            @testset "dee" begin
                inputs = _circle_inputs(16)
                plasma_surf = JPEC.Vacuum.PlasmaGeometry(inputs)
                wall_settings = WallShapeSettings(shape="dee", a=0.1, cw=0.0)
                wall = JPEC.Vacuum.WallGeometry(inputs, plasma_surf, wall_settings)
                @test wall.nowall == false
                @test length(wall.x) == 16
                @test all(wall.x .> 0)
            end

            @testset "edge: R <= 0 throws" begin
                inputs = _circle_inputs(16)
                plasma_surf_near_zero = JPEC.Vacuum.PlasmaGeometry(
                    VacuumInput(
                        r=0.05 .+ 0.03 .* cos.(range(0, 2π, length=16)),
                        z=0.03 .* sin.(range(0, 2π, length=16)),
                        ν=zeros(16),
                        mtheta=16,
                        nzeta=1
                    )
                )

                # Use a "dee" wall shape with parameters that will produce R < 0
                # Setting cw (offset) to a large negative value will shift the wall left past R=0
                wall_settings = WallShapeSettings(shape="dee", cw=-1.5, a=0.1)
                @test_throws ErrorException JPEC.Vacuum.WallGeometry(inputs, plasma_surf_near_zero, wall_settings)
            end

            # Test that conformal wall R-coordinates are clamped by centerstack_min
            # With a very large 'a' parameter, a conformal wall would naturally go to R < 0,
            # but it should be clamped to centerstack_min = min(0.1, 0.1 * minimum(x_plasma))
            @testset "edge: conformal centerstack clamp" begin
                inputs = _circle_inputs(16)
                plasma_surf_near_zero = JPEC.Vacuum.PlasmaGeometry(
                    VacuumInput(
                        r=0.05 .+ 0.03 .* cos.(range(0, 2π, length=16)),
                        z=0.03 .* sin.(range(0, 2π, length=16)),
                        ν=zeros(16),
                        mtheta=16,
                        nzeta=1
                    )
                )
                wall_settings = WallShapeSettings(shape="conformal", a=10.0, equal_arc_wall=false)
                wall = JPEC.Vacuum.WallGeometry(inputs, plasma_surf_near_zero, wall_settings)
                expected_min = min(0.1, 0.1 * minimum(plasma_surf_near_zero.x))
                @test all(wall.x .>= expected_min)
                @test any(wall.x .<= expected_min + 1e-10)
            end
        end

        # -------------------------------------------------------------------------
        @testset "distribute_to_equal_arc_grid" begin
            @testset "unit circle" begin
                theta = range(0, step=2π/10, length=10)
                xin = cos.(theta)
                zin = sin.(theta)
                xout, zout = JPEC.Vacuum.distribute_to_equal_arc_grid(xin, zin)
                @test length(xout) == length(xin)
                @test length(zout) == length(zin)
                r = sqrt.(xout .^ 2 .+ zout .^ 2)
                @test all(r -> isapprox(r, 1.0, atol=1e-9), r)
            end

            @testset "edge: ellipse" begin
                theta = range(0, 2π, length=32)
                xin = 2.0 .* cos.(theta)
                zin = 1.0 .* sin.(theta)
                xout, zout = JPEC.Vacuum.distribute_to_equal_arc_grid(xin, zin)
                @test length(xout) == 32
                @test all(isfinite, xout)
                @test all(isfinite, zout)
                # Redistribution preserves curve; points should be in similar region
                @test maximum(abs.(xout)) <= 2.0 + 0.1
                @test maximum(abs.(zout)) <= 1.0 + 0.1
            end
        end

        # -------------------------------------------------------------------------
        @testset "elliptic_integral_k" begin
            @testset "domain errors" begin
                @test_throws DomainError JPEC.Vacuum.elliptic_integral_k(-0.1)
                @test_throws DomainError JPEC.Vacuum.elliptic_integral_k(1.1)
            end

            @testset "known value" begin
                # K(1-m1): for m1=0.5 returns K(0.5) ≈ 1.85407
                K_half = JPEC.Vacuum.elliptic_integral_k(0.5)
                @test isapprox(K_half, 1.8540746773013719, rtol=1e-8)
                @test isfinite(K_half)
            end
        end

        # -------------------------------------------------------------------------
        @testset "elliptic_integral_e" begin
            @testset "domain errors" begin
                @test_throws DomainError JPEC.Vacuum.elliptic_integral_e(-0.1)
                @test_throws DomainError JPEC.Vacuum.elliptic_integral_e(1.1)
            end

            @testset "known value" begin
                # E(1-m1): for m1=0.5 returns E(0.5) ≈ 1.35064
                E_half = JPEC.Vacuum.elliptic_integral_e(0.5)
                @test isapprox(E_half, 1.3506438810476755, rtol=1e-8)
                @test isfinite(E_half)
            end
        end

        # -------------------------------------------------------------------------
        @testset "Pn_minus_half_1997" begin
            @testset "length and finite" begin
                # Returns P^0 through P^{n+1}, so length n+2
                P = JPEC.Vacuum.Pn_minus_half_1997(1.5, 3)
                @test length(P) == 5
                @test !any(isnan, P)
                @test all(isfinite, P)
            end

            @testset "agreement with Pn_minus_half_2007" begin
                s, n = 1.5, 3
                P_1997 = JPEC.Vacuum.Pn_minus_half_1997(s, n)
                P_2007 = JPEC.Vacuum.Pn_minus_half_2007(s, n)
                @test length(P_1997) == length(P_2007)
                @test isapprox(P_1997[1], P_2007[1], rtol=1e-7)
                @test isapprox(P_1997[2], P_2007[2], rtol=1e-7)
            end
        end

        # -------------------------------------------------------------------------
        @testset "Pn_minus_half_2007" begin
            @testset "length and finite" begin
                # Returns P^0 through P^{n+1}, so length n+2
                P = JPEC.Vacuum.Pn_minus_half_2007(2.0, 2)
                @test length(P) == 4
                @test !any(isnan, P)
            end
        end

        # -------------------------------------------------------------------------
        @testset "elliptic_integrals_bulirsch" begin
            @testset "convergence and output" begin
                K, E, conv, iters = JPEC.Vacuum.elliptic_integrals_bulirsch(0.5)
                @test K isa Float64
                @test E isa Float64
                @test isfinite(K)
                @test isfinite(E)
                @test conv < 1e-10
                @test iters >= 1
            end

            @testset "domain errors" begin
                @test_throws DomainError JPEC.Vacuum.elliptic_integrals_bulirsch(-0.1)
                @test_throws DomainError JPEC.Vacuum.elliptic_integrals_bulirsch(1.5)
            end
        end

        # -------------------------------------------------------------------------
        @testset "green" begin
            @testset "basic output structure" begin
                G_n, coupling_n, coupling_0 = JPEC.Vacuum.green(2.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1)
                @test G_n isa Float64
                @test coupling_n isa Float64
                @test coupling_0 isa Float64
                @test isfinite(G_n)
                @test isfinite(coupling_n)
                @test isfinite(coupling_0)
            end

            @testset "uselegacygreenfunction" begin
                G_leg, cpl_leg, c0_leg = JPEC.Vacuum.green(2.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1; uselegacygreenfunction=true)
                G_new, cpl_new, c0_new = JPEC.Vacuum.green(2.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1; uselegacygreenfunction=false)
                @test isfinite(G_leg) && isfinite(G_new)
                # Both implementations should give similar order of magnitude for this non-singular case
                @test isapprox(G_leg, G_new, rtol=1e-5)
            end

            @testset "n=0" begin
                G_n, coupling_n, coupling_0 = JPEC.Vacuum.green(1.5, 0.0, 1.0, 0.0, 0.0, 1.0, 0)
                @test isfinite(G_n)
                @test isfinite(coupling_0)
            end
        end

        # -------------------------------------------------------------------------
        @testset "compute_vacuum_response" begin
            _make_inputs(; mtheta=128, mtheta_eq=17, mpert=2, nlow=1, npert=1) = VacuumInput(
                r=collect(1.7 .+ 0.3 .* cos.(range(0, 2π, length=mtheta_eq))),
                z=collect(0.3 .* sin.(range(0, 2π, length=mtheta_eq))),
                ν=zeros(mtheta_eq),
                mlow=1,
                mpert=mpert,
                nlow=nlow,
                npert=npert,
                nzeta=1,
                mtheta=mtheta
            )

            @testset "nowall" begin
                inputs = _make_inputs()
                wall_settings = WallShapeSettings(shape="nowall")
                wv, grri, grre, plasma_pts, wall_pts = compute_vacuum_response(inputs, wall_settings)

                numpoints = inputs.mtheta * inputs.nzeta
                num_modes = inputs.mpert * inputs.npert

                @test size(wv) == (num_modes, num_modes)
                @test eltype(wv) == ComplexF64
                @test all(isfinite, wv)
                @test size(grri) == (2 * numpoints, 2 * num_modes)
                @test size(grre) == (2 * numpoints, 2 * num_modes)
                @test all(isfinite, grri)
                @test all(isfinite, grre)
                @test size(plasma_pts) == (numpoints, 3)
                @test all(isfinite, plasma_pts)
                @test size(wall_pts) == (numpoints, 3)

                # With force_wv_symmetry (default), wv should be Hermitian
                @test isapprox(wv, wv', rtol=1e-12)
            end

            @testset "conformal wall" begin
                inputs = _make_inputs()
                wall_settings = WallShapeSettings(shape="conformal", a=0.5)
                wv, grri, grre, plasma_pts, wall_pts = compute_vacuum_response(inputs, wall_settings)

                numpoints = inputs.mtheta * inputs.nzeta
                num_modes = inputs.mpert * inputs.npert
                @test size(wv) == (num_modes, num_modes)
                @test size(grri) == (2 * numpoints, 2 * num_modes)
                @test all(isfinite, plasma_pts)
                @test all(isfinite, wall_pts)
                # plasma_pts layout: col1=R, col2=0, col3=Z
                @test !isapprox(plasma_pts[:, 1], wall_pts[:, 1])
                @test !isapprox(plasma_pts[:, 3], wall_pts[:, 3])
                @test isapprox(wv, wv', rtol=1e-12)
            end

            @testset "edge: single poloidal mode mpert=1" begin
                inputs = _make_inputs(mpert=1, npert=1)
                wall_settings = WallShapeSettings(shape="nowall")
                wv, grri, grre, plasma_pts, wall_pts = compute_vacuum_response(inputs, wall_settings)
                @test size(wv) == (1, 1)
                @test all(isfinite, wv)
                @test size(grri, 2) == 2  # 2 * num_modes with num_modes=1
            end

            @testset "edge: small mtheta" begin
                # Keep mtheta_eq=17 so boundary has enough points for periodic spline
                inputs = _make_inputs(mtheta=16, mtheta_eq=17)
                wall_settings = WallShapeSettings(shape="nowall")
                wv, grri, grre, plasma_pts, wall_pts = compute_vacuum_response(inputs, wall_settings)
                @test size(wv) == (2, 2)
                @test size(grri) == (32, 4)  # 2*16, 2*2
                @test size(plasma_pts) == (16, 3)
            end
        end
    end
end
