@testset "Vacuum.jl Unit Tests" begin
    using LinearAlgebra

    @testset "Vacuum.jl (2D)" begin

        # -------------------------------------------------------------------------
        @testset "VacuumInput" begin
            @testset "default constructor" begin
                vac = VacuumInput()
                @test vac.x == Float64[]
                @test vac.y == Float64[]
                @test vac.z == Float64[]
                @test vac.ν == Float64[]
                @test vac.mtheta_in == 0
                @test vac.nzeta_in == 1
                @test vac.mlow == 1
                @test vac.mpert == 1
                @test vac.nlow == 1
                @test vac.npert == 1
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
                    mtheta_in=5,
                    nzeta_in=1,
                    x=[1.0, 1.1, 1.2, 1.1, 1.0],
                    z=[0.0, 0.1, 0.0, -0.1, 0.0],
                    ν=zeros(5),
                    mtheta=5,
                    mpert=1,
                    mlow=1,
                    nlow=1,
                    npert=1,
                    nzeta=1
                )
                surf = GeneralizedPerturbedEquilibrium.Vacuum.PlasmaGeometry(inputs)
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
                    x=[1.0, 1.1, 1.2, 1.1, 1.0],
                    z=[0.0, 0.1, 0.0, -0.1, 0.0],
                    mtheta_in=5,
                    nzeta_in=1,
                    ν=zeros(5),
                    mtheta=8,
                    mpert=1,
                    mlow=0,
                    nlow=0,
                    npert=1,
                    nzeta=1
                )
                surf = GeneralizedPerturbedEquilibrium.Vacuum.PlasmaGeometry(inputs)
                @test length(surf.x) == 8
                @test all(isfinite, surf.x)
                @test all(isfinite, surf.z)
            end
        end

        # -------------------------------------------------------------------------
        @testset "WallGeometry" begin
            _circle_inputs(mtheta) = VacuumInput(
                mtheta_in=mtheta,
                nzeta_in=1,
                x=1.7 .+ 0.3 .* cos.(range(0, 2π, length=mtheta)),
                z=0.3 .* sin.(range(0, 2π, length=mtheta)),
                ν=zeros(mtheta),
                mtheta=mtheta,
                nzeta=1
            )

            @testset "nowall" begin
                inputs = _circle_inputs(16)
                plasma_surf = GeneralizedPerturbedEquilibrium.Vacuum.PlasmaGeometry(inputs)
                wall_settings = WallShapeSettings(shape="nowall")
                wall = GeneralizedPerturbedEquilibrium.Vacuum.WallGeometry(inputs, plasma_surf, wall_settings)
                @test wall.nowall == true
                @test length(wall.x) == 16
                @test length(wall.z) == 16
            end

            @testset "conformal" begin
                inputs = _circle_inputs(16)
                plasma_surf = GeneralizedPerturbedEquilibrium.Vacuum.PlasmaGeometry(inputs)
                wall_settings = WallShapeSettings(shape="conformal", a=0.2)
                wall = GeneralizedPerturbedEquilibrium.Vacuum.WallGeometry(inputs, plasma_surf, wall_settings)
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
                plasma_surf = GeneralizedPerturbedEquilibrium.Vacuum.PlasmaGeometry(inputs)
                wall_settings = WallShapeSettings(shape="elliptical", a=0.5)
                wall = GeneralizedPerturbedEquilibrium.Vacuum.WallGeometry(inputs, plasma_surf, wall_settings)
                @test wall.nowall == false
                @test length(wall.x) == 16
                @test all(isfinite, wall.x)
                @test all(isfinite, wall.z)
            end

            @testset "dee" begin
                inputs = _circle_inputs(16)
                plasma_surf = GeneralizedPerturbedEquilibrium.Vacuum.PlasmaGeometry(inputs)
                wall_settings = WallShapeSettings(shape="dee", a=0.1, cw=0.0)
                wall = GeneralizedPerturbedEquilibrium.Vacuum.WallGeometry(inputs, plasma_surf, wall_settings)
                @test wall.nowall == false
                @test length(wall.x) == 16
                @test all(wall.x .> 0)
            end

            @testset "edge: R <= 0 throws" begin
                inputs = _circle_inputs(16)
                plasma_surf_near_zero = GeneralizedPerturbedEquilibrium.Vacuum.PlasmaGeometry(
                    VacuumInput(
                        mtheta_in=16,
                        nzeta_in=1,
                        x=0.05 .+ 0.03 .* cos.(range(0, 2π, length=16)),
                        z=0.03 .* sin.(range(0, 2π, length=16)),
                        ν=zeros(16),
                        mtheta=16,
                        nzeta=1
                    )
                )

                # Use a "dee" wall shape with parameters that will produce R < 0
                # Setting cw (offset) to a large negative value will shift the wall left past R=0
                wall_settings = WallShapeSettings(shape="dee", cw=-1.5, a=0.1)
                @test_throws ErrorException GeneralizedPerturbedEquilibrium.Vacuum.WallGeometry(inputs, plasma_surf_near_zero, wall_settings)
            end

            # Test that conformal wall R-coordinates are clamped by centerstack_min
            # With a very large 'a' parameter, a conformal wall would naturally go to R < 0,
            # but it should be clamped to centerstack_min = min(0.1, 0.1 * minimum(x_plasma))
            @testset "edge: conformal centerstack clamp" begin
                inputs = _circle_inputs(16)
                plasma_surf_near_zero = GeneralizedPerturbedEquilibrium.Vacuum.PlasmaGeometry(
                    VacuumInput(
                        mtheta_in=16,
                        nzeta_in=1,
                        x=0.05 .+ 0.03 .* cos.(range(0, 2π, length=16)),
                        z=0.03 .* sin.(range(0, 2π, length=16)),
                        ν=zeros(16),
                        mtheta=16,
                        nzeta=1
                    )
                )
                wall_settings = WallShapeSettings(shape="conformal", a=10.0, equal_arc_wall=false)
                wall = GeneralizedPerturbedEquilibrium.Vacuum.WallGeometry(inputs, plasma_surf_near_zero, wall_settings)
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
                xout, zout = GeneralizedPerturbedEquilibrium.Vacuum.distribute_to_equal_arc_grid(xin, zin)
                @test length(xout) == length(xin)
                @test length(zout) == length(zin)
                r = sqrt.(xout .^ 2 .+ zout .^ 2)
                @test all(r -> isapprox(r, 1.0, atol=1e-9), r)
            end

            @testset "edge: ellipse" begin
                theta = range(0, 2π, length=32)
                xin = 2.0 .* cos.(theta)
                zin = 1.0 .* sin.(theta)
                xout, zout = GeneralizedPerturbedEquilibrium.Vacuum.distribute_to_equal_arc_grid(xin, zin)
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
                @test_throws DomainError GeneralizedPerturbedEquilibrium.Vacuum.elliptic_integral_k(-0.1)
                @test_throws DomainError GeneralizedPerturbedEquilibrium.Vacuum.elliptic_integral_k(1.1)
            end

            @testset "known value" begin
                # K(1-m1): for m1=0.5 returns K(0.5) ≈ 1.85407
                K_half = GeneralizedPerturbedEquilibrium.Vacuum.elliptic_integral_k(0.5)
                @test isapprox(K_half, 1.8540746773013719, rtol=1e-8)
                @test isfinite(K_half)
            end
        end

        # -------------------------------------------------------------------------
        @testset "elliptic_integral_e" begin
            @testset "domain errors" begin
                @test_throws DomainError GeneralizedPerturbedEquilibrium.Vacuum.elliptic_integral_e(-0.1)
                @test_throws DomainError GeneralizedPerturbedEquilibrium.Vacuum.elliptic_integral_e(1.1)
            end

            @testset "known value" begin
                # E(1-m1): for m1=0.5 returns E(0.5) ≈ 1.35064
                E_half = GeneralizedPerturbedEquilibrium.Vacuum.elliptic_integral_e(0.5)
                @test isapprox(E_half, 1.3506438810476755, rtol=1e-8)
                @test isfinite(E_half)
            end
        end

        # -------------------------------------------------------------------------
        @testset "Pn_minus_half_1997" begin
            @testset "length and finite" begin
                # Returns P^0 through P^{n+1}, so length n+2
                P = GeneralizedPerturbedEquilibrium.Vacuum.Pn_minus_half_1997(1.5, 3)
                @test length(P) == 5
                @test !any(isnan, P)
                @test all(isfinite, P)
            end

            @testset "agreement with Pn_minus_half_2007" begin
                s, n = 1.5, 3
                P_1997 = GeneralizedPerturbedEquilibrium.Vacuum.Pn_minus_half_1997(s, n)
                P_2007 = GeneralizedPerturbedEquilibrium.Vacuum.Pn_minus_half_2007(s, n)
                @test length(P_1997) == length(P_2007)
                @test isapprox(P_1997[1], P_2007[1], rtol=1e-7)
                @test isapprox(P_1997[2], P_2007[2], rtol=1e-7)
            end
        end

        # -------------------------------------------------------------------------
        @testset "Pn_minus_half_2007" begin
            @testset "length and finite" begin
                # Returns P^0 through P^{n+1}, so length n+2
                P = GeneralizedPerturbedEquilibrium.Vacuum.Pn_minus_half_2007(2.0, 2)
                @test length(P) == 4
                @test !any(isnan, P)
            end
        end

        @testset "Pn_minus_half_2007 regression (develop reference values)" begin
            # Reference values computed on develop branch (pre-optimization).
            # Tests the Gaussian quadrature branch (n*rhohat >= 0.1) across a
            # wide range of s and n values. Tolerance rtol=1e-15 allows for
            # minor FMA (muladd) rounding differences while catching real bugs.
            ref = [
                #  s         n    P^n_{-1/2}                P^{n+1}_{-1/2}
                (1.0001, 1, -1.76771171238955592e-03, 1.40617383201562789e-05),
                (1.001, 1, -5.58842369229195501e-03, 1.40548867015278633e-04),
                (1.01, 1, -1.76226399819901618e-02, 1.39867152712671921e-03),
                (1.1, 1, -5.42197386532770609e-02, 1.33378223659796936e-02),
                (2.0, 1, -1.36668749688715452e-01, 9.03013828502453597e-02),
                (5.0, 1, -1.72809674189718487e-01, 1.66308973488766609e-01),
                (1.001, 2, 1.40548867015278633e-04, -6.54586569482422801e-06),
                (1.01, 2, 1.39867152712671921e-03, -2.05551959799760283e-04),
                (1.1, 2, 1.33378223659796936e-02, -6.06985214017424779e-03),
                (2.0, 2, 9.03013828502453597e-02, -1.09579534774664561e-01),
                (1.001, 3, -6.54586569482422801e-06, 4.48148923318089893e-07),
                (1.01, 5, -1.26853325233827571e-05, 4.51118736952202167e-06),
                (1.1, 5, -3.58868753269622276e-03, 3.94937015546779815e-03),
                (2.0, 5, -4.57188154910584288e-01, 1.33433979548505799e+00),
                (1.01, 10, 3.43344970603451185e-07, -2.42729568546590492e-07),
                (1.1, 10, 2.75532184506539837e-02, -6.02683663742238918e-02),
                (2.0, 10, 4.58572088633330225e+02, -2.65592666160963245e+03),
                (5.0, 10, 1.42588782875344259e+04, -1.17035465963630311e+05),
                (1.01, 20, 3.55246831705678395e-07, -5.01443246471398338e-07),
                (1.1, 20, 2.29121454711561682e+03, -1.00059018411255292e+04),
                (2.0, 20, 6.43835513511152710e+11, -7.44072872498351758e+12),
                (1.1, 30, 4.09548864574038162e+10, -2.68188110057737183e+11),
                (2.0, 30, 1.93743177482701155e+23, -3.35704314650305262e+24),
                (1.1, 50, 1.69509814277651482e+29, -1.84969372821802429e+30),
                (2.0, 50, 2.26828572321442199e+50, -6.54892185301014281e+51)
            ]
            for (s, n, Pn_ref, Pnp1_ref) in ref
                P = GeneralizedPerturbedEquilibrium.Vacuum.Pn_minus_half_2007(s, n)
                @test isapprox(P[end-1], Pn_ref, rtol=1e-15)
                @test isapprox(P[end], Pnp1_ref, rtol=1e-15)
            end
        end

        # -------------------------------------------------------------------------
        @testset "elliptic_integrals_bulirsch" begin
            @testset "convergence and output" begin
                K, E, conv, iters = GeneralizedPerturbedEquilibrium.Vacuum.elliptic_integrals_bulirsch(0.5)
                @test K isa Float64
                @test E isa Float64
                @test isfinite(K)
                @test isfinite(E)
                @test conv < 1e-10
                @test iters >= 1
            end

            @testset "domain errors" begin
                @test_throws DomainError GeneralizedPerturbedEquilibrium.Vacuum.elliptic_integrals_bulirsch(-0.1)
                @test_throws DomainError GeneralizedPerturbedEquilibrium.Vacuum.elliptic_integrals_bulirsch(1.5)
            end
        end

        # -------------------------------------------------------------------------
        @testset "green" begin
            @testset "basic output structure" begin
                G_n, coupling_n, coupling_0 = GeneralizedPerturbedEquilibrium.Vacuum.green(2.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1)
                @test G_n isa Float64
                @test coupling_n isa Float64
                @test coupling_0 isa Float64
                @test isfinite(G_n)
                @test isfinite(coupling_n)
                @test isfinite(coupling_0)
            end

            @testset "uselegacygreenfunction" begin
                G_leg, cpl_leg, c0_leg = GeneralizedPerturbedEquilibrium.Vacuum.green(2.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1; uselegacygreenfunction=true)
                G_new, cpl_new, c0_new = GeneralizedPerturbedEquilibrium.Vacuum.green(2.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1; uselegacygreenfunction=false)
                @test isfinite(G_leg) && isfinite(G_new)
                # Both implementations should give similar order of magnitude for this non-singular case
                @test isapprox(G_leg, G_new, rtol=1e-5)
            end

            @testset "n=0" begin
                G_n, coupling_n, coupling_0 = GeneralizedPerturbedEquilibrium.Vacuum.green(1.5, 0.0, 1.0, 0.0, 0.0, 1.0, 0)
                @test isfinite(G_n)
                @test isfinite(coupling_0)
            end
        end

        # -------------------------------------------------------------------------
        @testset "compute_vacuum_response" begin
            _make_inputs(; mtheta=128, mtheta_eq=17, mpert=2, nlow=1, npert=1) = VacuumInput(
                mtheta_in=mtheta_eq,
                nzeta_in=1,
                x=collect(1.7 .+ 0.3 .* cos.(range(0, 2π, length=mtheta_eq))),
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

        # -------------------------------------------------------------------------
        @testset "fused vs two-step Galerkin (2D, nowall)" begin
            # Small case where both Galerkin paths are cheap: compare K_c, G_c
            # assembled via the full M×M kernel + projection against the fused
            # projected kernels from the unified `kernel!` API.
            inputs = VacuumInput(
                mtheta_in=17,
                nzeta_in=1,
                x=collect(1.7 .+ 0.3 .* cos.(range(0, 2π, length=17))),
                z=collect(0.3 .* sin.(range(0, 2π, length=17))),
                ν=zeros(17),
                mlow=1,
                mpert=2,
                nlow=1,
                npert=1,
                nzeta=1,
                mtheta=32
            )
            wall_settings = WallShapeSettings(shape="nowall")

            plasma_surf = GeneralizedPerturbedEquilibrium.Vacuum.PlasmaGeometry(inputs)
            kparams = GeneralizedPerturbedEquilibrium.Vacuum.KernelParams2D(inputs.nlow)

            # Fourier basis on the surface grid
            exp_mn_basis = GeneralizedPerturbedEquilibrium.Utilities.FourierTransforms.compute_fourier_coefficients(
                inputs.mtheta,
                inputs.mpert,
                inputs.mlow,
                inputs.nzeta,
                inputs.npert,
                inputs.nlow;
                n_2D=inputs.nlow,
                ν=plasma_surf.ν
            )
            M, P = size(exp_mn_basis)
            Gram = fill(ComplexF64(M), P)

            # --- Two-step Galerkin: materialize full kernels then project ---
            grad_green_full = zeros(Float64, 2M, 2M)
            green_full = zeros(Float64, M, M)
            GeneralizedPerturbedEquilibrium.Vacuum.kernel!(
                grad_green_full,
                green_full,
                plasma_surf,
                plasma_surf,
                kparams
            )

            # Exterior projected kernels from full matrices: K_c = Z^H K Z, G_c = Z^H G Z
            K_c_two = zeros(ComplexF64, P, P)
            G_c_two = zeros(ComplexF64, P, P)
            tmp = zeros(ComplexF64, M, P)

            grad_pp = @view grad_green_full[1:M, 1:M]
            mul!(tmp, grad_pp, exp_mn_basis)
            mul!(K_c_two, exp_mn_basis', tmp)
            mul!(tmp, green_full, exp_mn_basis)
            mul!(G_c_two, exp_mn_basis', tmp)

            # --- Fused Galerkin via unified kernel! ---
            K_c_fused = zeros(ComplexF64, P, P)
            G_c_fused = zeros(ComplexF64, P, P)
            GeneralizedPerturbedEquilibrium.Vacuum.projected_kernel!(
                K_c_fused,
                G_c_fused,
                plasma_surf,
                plasma_surf,
                kparams,
                exp_mn_basis,
                Gram
            )

            @test isapprox(K_c_fused, K_c_two; rtol=1e-10, atol=1e-12)
            @test isapprox(G_c_fused, G_c_two; rtol=1e-10, atol=1e-12)
        end

        # -------------------------------------------------------------------------
        @testset "wall Galerkin vs collocation (2D, conformal)" begin
            mtheta_eq = 17
            mtheta = 128
            mpert = 3
            boundary_x = collect(1.7 .+ 0.3 .* cos.(range(0, 2π, length=mtheta_eq)))
            boundary_z = collect(0.3 .* sin.(range(0, 2π, length=mtheta_eq)))

            inputs_colloc = VacuumInput(
                mtheta_in=mtheta_eq, nzeta_in=1,
                x=boundary_x, z=boundary_z, ν=zeros(mtheta_eq),
                mlow=1, mpert=mpert, nlow=1, npert=1,
                nzeta=1, mtheta=mtheta,
                use_galerkin=false
            )
            inputs_galerkin = VacuumInput(
                mtheta_in=mtheta_eq, nzeta_in=1,
                x=boundary_x, z=boundary_z, ν=zeros(mtheta_eq),
                mlow=1, mpert=mpert, nlow=1, npert=1,
                nzeta=1, mtheta=mtheta,
                use_galerkin=true
            )

            wall_settings = WallShapeSettings(shape="conformal", a=0.5)

            wv_c, grri_c, grre_c, _, _ = compute_vacuum_response(inputs_colloc, wall_settings)
            wv_g, grri_g, grre_g, _, _ = compute_vacuum_response(inputs_galerkin, wall_settings)

            M = mtheta
            P = mpert

            @test isapprox(wv_g, wv_c; rtol=1e-8)
            @test isapprox(grre_g[1:M, 1:(2*P)], grre_c[1:M, 1:(2*P)]; rtol=1e-8)
            @test isapprox(grri_g[1:M, 1:(2*P)], grri_c[1:M, 1:(2*P)]; rtol=1e-8)
            @test isapprox(grre_g[(M+1):(2*M), 1:(2*P)], grre_c[(M+1):(2*M), 1:(2*P)]; rtol=1e-8)
            @test isapprox(grri_g[(M+1):(2*M), 1:(2*P)], grri_c[(M+1):(2*M), 1:(2*P)]; rtol=1e-8)
        end
    end

    # -------------------------------------------------------------------------
    # 3D vacuum: nzeta > 1, full (m,n) coupling, PlasmaGeometry3D, WallGeometry3D
    # Kernel requires mtheta, nzeta >= PATCH_DIM (23 for default KernelParams3D(11, 20, 5))
    @testset "Vacuum.jl (3D)" begin
        _make_3d_inputs(; mtheta=32, mtheta_eq=17, mpert=2, nlow=0, npert=2, nzeta=32) = VacuumInput(
            mtheta_in=mtheta_eq,
            nzeta_in=1,
            x=collect(1.7 .+ 0.3 .* cos.(range(0, 2π, length=mtheta_eq))),
            z=collect(0.3 .* sin.(range(0, 2π, length=mtheta_eq))),
            ν=zeros(mtheta_eq),
            mlow=1,
            mpert=mpert,
            nlow=nlow,
            npert=npert,
            nzeta=nzeta,
            mtheta=mtheta
        )

        # Helper: simple nonaxisymmetric (3D) plasma boundary built from SFL-style (θ, ζ) coordinates.
        _make_3d_nonaxis_inputs(; mtheta=24, nzeta=24, mtheta_in=12, nzeta_in=12, mpert=2, nlow=0, npert=2) = begin
            θ_in = range(0, 2π, length=mtheta_in)
            ζ_in = range(0, 2π, length=nzeta_in)

            X = zeros(mtheta_in, nzeta_in)
            Y = zeros(mtheta_in, nzeta_in)
            Z = zeros(mtheta_in, nzeta_in)

            R0 = 1.7
            a = 0.3
            ε = 0.05

            for (i, θ) in enumerate(θ_in), (j, ζ) in enumerate(ζ_in)
                # Base circular cross‑section with a small toroidal corrugation
                R = R0 + a * cos(θ) + ε * cos(2ζ) * cos(θ)
                Z_ij = 0.3 * sin(θ) + ε * sin(2ζ) * sin(θ)
                X[i, j] = R * cos(ζ)
                Y[i, j] = R * sin(ζ)
                Z[i, j] = Z_ij
            end

            VacuumInput(
                x=vec(X),
                y=vec(Y),
                z=vec(Z),
                mtheta_in=mtheta_in,
                nzeta_in=nzeta_in,
                mlow=1,
                mpert=mpert,
                nlow=nlow,
                npert=npert,
                mtheta=mtheta,
                nzeta=nzeta
            )
        end

        @testset "VacuumInput nzeta > 1" begin
            vac = VacuumInput(mtheta=32, nzeta=24, mpert=2, npert=2)
            @test vac.nzeta == 24
            @test vac.mtheta == 32
        end

        @testset "KernelParams3D" begin
            params = GeneralizedPerturbedEquilibrium.Vacuum.KernelParams3D(11, 20, 5)
            @test params.PATCH_RAD == 11
            @test params.RAD_DIM == 20
            @test params.INTERP_ORDER == 5
        end

        @testset "PlasmaGeometry3D" begin
            inputs = _make_3d_inputs(mtheta=32, nzeta=32, mtheta_eq=17)
            surf = GeneralizedPerturbedEquilibrium.Vacuum.PlasmaGeometry3D(inputs)
            num_points = inputs.mtheta * inputs.nzeta
            @test surf.mtheta == 32
            @test surf.nzeta == 32
            @test size(surf.r) == (num_points, 3)
            @test size(surf.dr_dθ) == (num_points, 3)
            @test size(surf.dr_dζ) == (num_points, 3)
            @test size(surf.normal) == (num_points, 3)
            @test surf.normal_orient in (1, -1)
            @test all(isfinite, surf.r)
            @test all(isfinite, surf.normal)
            # Toroidal extrusion: first and (1+mtheta)th points differ only in (X,Y); Z same
            @test isapprox(surf.r[1, 3], surf.r[1+32, 3])
            @test !isapprox(surf.r[1, 1], surf.r[1+32, 1]) || !isapprox(surf.r[1, 2], surf.r[1+32, 2])
        end

        @testset "PlasmaGeometry3D nonaxisymmetric input" begin
            inputs = _make_3d_nonaxis_inputs(mtheta=24, nzeta=24, mtheta_in=12, nzeta_in=12)
            surf = GeneralizedPerturbedEquilibrium.Vacuum.PlasmaGeometry3D(inputs)
            num_points = inputs.mtheta * inputs.nzeta

            @test surf.mtheta == 24
            @test surf.nzeta == 24
            @test size(surf.r) == (num_points, 3)
            @test size(surf.normal) == (num_points, 3)
            @test surf.normal_orient in (1, -1)
            @test all(isfinite, surf.r)
            @test all(isfinite, surf.normal)
            # Nonaxisymmetric boundary should have genuine 3D structure (non‑trivial Y variation)
            @test maximum(abs, surf.r[:, 2]) > 0
        end

        @testset "WallGeometry3D nowall" begin
            inputs = _make_3d_inputs(mtheta=32, nzeta=32, mtheta_eq=17)
            wall = GeneralizedPerturbedEquilibrium.Vacuum.WallGeometry3D(inputs, WallShapeSettings(shape="nowall"))
            @test wall.nowall == true
            @test wall.mtheta == 32
            @test wall.nzeta == 32
            @test size(wall.r) == (32 * 32, 3)
        end

        @testset "WallGeometry3D conformal" begin
            inputs = _make_3d_inputs(mtheta=32, nzeta=32, mtheta_eq=17)
            wall = GeneralizedPerturbedEquilibrium.Vacuum.WallGeometry3D(inputs, WallShapeSettings(shape="conformal", a=0.2))
            @test wall.nowall == false
            @test wall.mtheta == 32
            @test wall.nzeta == 32
            num_points = 32 * 32
            @test size(wall.r) == (num_points, 3)
            @test size(wall.normal) == (num_points, 3)
            @test all(isfinite, wall.r)
            @test all(isfinite, wall.normal)
        end

        @testset "compute_vacuum_response 3D nowall" begin
            inputs = _make_3d_inputs(mtheta=32, nzeta=32, mtheta_eq=17)
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
            # 3D plasma_pts are (X,Y,Z) Cartesian
            @test isapprox(plasma_pts[1, 1]^2 + plasma_pts[1, 2]^2, (1.7 + 0.3)^2, rtol=0.1)
            @test isapprox(plasma_pts[1, 3], 0.0, atol=0.1)
            @test isapprox(wv, wv', rtol=1e-12)
        end

        @testset "compute_vacuum_response 3D nonaxisymmetric boundary" begin
            inputs = _make_3d_nonaxis_inputs(mtheta=24, nzeta=24, mtheta_in=12, nzeta_in=12, mpert=2, nlow=0, npert=2)
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
            @test isapprox(wv, wv', rtol=1e-12)
        end

        @testset "compute_vacuum_response 3D conformal wall" begin
            inputs = _make_3d_inputs(mtheta=32, nzeta=32, mtheta_eq=17)
            wall_settings = WallShapeSettings(shape="conformal", a=0.3)
            wv, grri, grre, plasma_pts, wall_pts = compute_vacuum_response(inputs, wall_settings)

            numpoints = inputs.mtheta * inputs.nzeta
            num_modes = inputs.mpert * inputs.npert
            @test size(wv) == (num_modes, num_modes)
            @test size(grri) == (2 * numpoints, 2 * num_modes)
            @test all(isfinite, plasma_pts)
            @test all(isfinite, wall_pts)
            # Wall and plasma should differ (conformal wall offset from plasma)
            @test !isapprox(plasma_pts, wall_pts)
            @test isapprox(wv, wv', rtol=1e-12)
        end

        @testset "wall Galerkin vs collocation (3D, conformal)" begin
            mtheta_eq = 17
            mtheta = 32
            nzeta = 32
            mpert = 2
            npert = 2
            boundary_x = collect(1.7 .+ 0.3 .* cos.(range(0, 2π, length=mtheta_eq)))
            boundary_z = collect(0.3 .* sin.(range(0, 2π, length=mtheta_eq)))

            inputs_colloc = VacuumInput(
                mtheta_in=mtheta_eq, nzeta_in=1,
                x=boundary_x, z=boundary_z, ν=zeros(mtheta_eq),
                mlow=1, mpert=mpert, nlow=0, npert=npert,
                nzeta=nzeta, mtheta=mtheta,
                use_galerkin=false
            )
            inputs_galerkin = VacuumInput(
                mtheta_in=mtheta_eq, nzeta_in=1,
                x=boundary_x, z=boundary_z, ν=zeros(mtheta_eq),
                mlow=1, mpert=mpert, nlow=0, npert=npert,
                nzeta=nzeta, mtheta=mtheta,
                use_galerkin=true
            )

            wall_settings = WallShapeSettings(shape="conformal", a=0.3)

            wv_c, grri_c, grre_c, _, _ = compute_vacuum_response(inputs_colloc, wall_settings)
            wv_g, grri_g, grre_g, _, _ = compute_vacuum_response(inputs_galerkin, wall_settings)

            M = mtheta * nzeta
            P = mpert * npert

            @test isapprox(wv_g, wv_c; rtol=1e-8)
            @test isapprox(grre_g[1:M, 1:(2*P)], grre_c[1:M, 1:(2*P)]; rtol=1e-8)
            @test isapprox(grri_g[1:M, 1:(2*P)], grri_c[1:M, 1:(2*P)]; rtol=1e-8)
            @test isapprox(grre_g[(M+1):(2*M), 1:(2*P)], grre_c[(M+1):(2*M), 1:(2*P)]; rtol=1e-8)
            @test isapprox(grri_g[(M+1):(2*M), 1:(2*P)], grri_c[(M+1):(2*M), 1:(2*P)]; rtol=1e-8)
        end

        @testset "Kernel3D laplace_single_layer" begin
            x_obs = [1.0, 0.0, 0.0]
            x_src = [2.0, 0.0, 0.0]
            G = GeneralizedPerturbedEquilibrium.Vacuum.laplace_single_layer(x_obs, x_src)
            # Kernel returns 1/|r_obs - r_src| (4π factor applied elsewhere in BIE)
            dist = sqrt((2.0 - 1.0)^2 + 0 + 0)
            @test isapprox(G, 1.0 / dist)
            @test isfinite(G)
        end

        @testset "Kernel3D laplace_double_layer" begin
            x_obs = [1.0, 0.0, 0.0]
            x_src = [2.0, 0.0, 0.0]
            n_src = [1.0, 0.0, 0.0]  # normal pointing away from source
            K = GeneralizedPerturbedEquilibrium.Vacuum.laplace_double_layer(x_obs, x_src, n_src)
            @test K isa Float64
            @test isfinite(K)
        end

        @testset "Kernel3D get_singular_quadrature" begin
            quad = GeneralizedPerturbedEquilibrium.Vacuum.get_singular_quadrature(3, 8, 5)
            @test quad.PATCH_RAD == 3
            @test quad.RAD_DIM == 8
            @test quad.INTERP_ORDER == 5
            @test quad.PATCH_DIM == 2 * 3 + 1
            # Cached: second call returns same object
            quad2 = GeneralizedPerturbedEquilibrium.Vacuum.get_singular_quadrature(3, 8, 5)
            @test quad === quad2
        end

        @testset "Kernel3D extract_patch!" begin
            # 4×4 grid (npol=4, ntor=4), PATCH_DIM=3, center at (2,2); data is (16, 3)
            data = zeros(16, 3)
            data[:, 1] .= 1.0:16.0
            data[:, 2] .= 1.0
            data[:, 3] .= 0.0
            patch_out = zeros(3, 3, 3)
            GeneralizedPerturbedEquilibrium.Vacuum.extract_patch!(patch_out, data, 2, 2, 4, 4, 3)
            # Center of patch should be data at (2,2) = index 2+4*(2-1)=6
            @test isapprox(patch_out[2, 2, 1], 6.0)
            @test all(isfinite, patch_out)
            @test size(patch_out) == (3, 3, 3)
        end
    end
end
