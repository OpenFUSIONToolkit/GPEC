@testset "Vacuum.jl Unit Tests" begin

    @testset "Vacuum.jl (2D)" begin

        @testset "VacuumInput" begin
            @testset "default constructor" begin
                vac = VacuumInput()
                @test vac.x == Float64[]
                @test vac.y == Float64[]
                @test vac.z == Float64[]
                @test vac.ν == Float64[]
                @test vac.mtheta_in == 0
                @test vac.nzeta_in == 1
                @test vac.m_modes == [1]
                @test vac.n_modes == [1]
                @test vac.mtheta == 1
                @test vac.nzeta == 1
            end

            @testset "keyword constructor" begin
                vac = VacuumInput(mtheta=32, m_modes=[1, 2, 3], n_modes=[2, 3], nzeta=1)
                @test vac.mtheta == 32
                @test length(vac.m_modes) == 3
                @test vac.m_modes[1] == 1
                @test vac.n_modes[1] == 2
                @test length(vac.n_modes) == 2
                @test vac.nzeta == 1
            end
        end

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

        @testset "PlasmaGeometry" begin
            @testset "from VacuumInput" begin
                inputs = VacuumInput(
                    mtheta_in=5,
                    nzeta_in=1,
                    x=[1.0, 1.1, 1.2, 1.1, 1.0],
                    z=[0.0, 0.1, 0.0, -0.1, 0.0],
                    ν=zeros(5),
                    mtheta=5,
                    m_modes=[1],
                    n_modes=[1],
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
                    m_modes=[0],
                    n_modes=[0],
                    nzeta=1
                )
                surf = GeneralizedPerturbedEquilibrium.Vacuum.PlasmaGeometry(inputs)
                @test length(surf.x) == 8
                @test all(isfinite, surf.x)
                @test all(isfinite, surf.z)
            end
        end

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

        @testset "compute_vacuum_response" begin
            _make_inputs(; mtheta=128, mtheta_eq=17, m_modes=1:2, n_modes=[1]) = VacuumInput(
                mtheta_in=mtheta_eq,
                nzeta_in=1,
                x=collect(1.7 .+ 0.3 .* cos.(range(0, 2π, length=mtheta_eq))),
                z=collect(0.3 .* sin.(range(0, 2π, length=mtheta_eq))),
                ν=zeros(mtheta_eq),
                m_modes=collect(Int, m_modes),
                n_modes=collect(Int, n_modes),
                nzeta=1,
                mtheta=mtheta
            )

            @testset "nowall" begin
                inputs = _make_inputs()
                wall_settings = WallShapeSettings(shape="nowall")
                (; wv, I_v, plasma_pts, wall_pts) = compute_vacuum_response(inputs, wall_settings)

                numpoints = inputs.mtheta * inputs.nzeta
                num_modes = length(inputs.m_modes) * length(inputs.n_modes)

                @test size(wv) == (num_modes, num_modes)
                @test eltype(wv) == ComplexF64
                @test all(isfinite, wv)
                @test size(I_v) == (num_modes, num_modes)
                @test all(iszero, I_v)  # compute_Iv=false by default
                @test size(plasma_pts) == (numpoints, 3)
                @test all(isfinite, plasma_pts)
                @test size(wall_pts) == (numpoints, 3)

                # wv is always Hermitian-symmetrized
                @test isapprox(wv, wv', rtol=1e-12)
            end

            @testset "nowall compute_Iv=true" begin
                inputs = _make_inputs()
                wall_settings = WallShapeSettings(shape="nowall")
                (; wv, I_v, plasma_pts, wall_pts) = compute_vacuum_response(inputs, wall_settings; compute_Iv=true)

                num_modes = length(inputs.m_modes) * length(inputs.n_modes)
                @test size(wv) == (num_modes, num_modes)
                @test all(isfinite, wv)
                @test size(I_v) == (num_modes, num_modes)
                @test all(isfinite, I_v)
                @test !all(iszero, I_v)
                @test isapprox(wv, wv', rtol=1e-12)
            end

            @testset "conformal wall" begin
                inputs = _make_inputs()
                wall_settings = WallShapeSettings(shape="conformal", a=0.5)
                (; wv, I_v, plasma_pts, wall_pts) = compute_vacuum_response(inputs, wall_settings)

                num_modes = length(inputs.m_modes) * length(inputs.n_modes)
                @test size(wv) == (num_modes, num_modes)
                @test all(iszero, I_v)
                @test all(isfinite, plasma_pts)
                @test all(isfinite, wall_pts)
                # plasma_pts layout: col1=R, col2=0, col3=Z
                @test !isapprox(plasma_pts[:, 1], wall_pts[:, 1])
                @test !isapprox(plasma_pts[:, 3], wall_pts[:, 3])
                @test isapprox(wv, wv', rtol=1e-12)
            end

            @testset "edge: single poloidal mode mpert=1" begin
                inputs = _make_inputs(m_modes=[1], n_modes=[1])
                wall_settings = WallShapeSettings(shape="nowall")
                (; wv, I_v, plasma_pts, wall_pts) = compute_vacuum_response(inputs, wall_settings)
                @test size(wv) == (1, 1)
                @test all(isfinite, wv)
                @test size(I_v) == (1, 1)
            end

            @testset "edge: small mtheta" begin
                # Keep mtheta_eq=17 so boundary has enough points for periodic spline
                inputs = _make_inputs(mtheta=16, mtheta_eq=17)
                wall_settings = WallShapeSettings(shape="nowall")
                (; wv, I_v, plasma_pts, wall_pts) = compute_vacuum_response(inputs, wall_settings)
                @test size(wv) == (2, 2)
                @test size(I_v) == (2, 2)
                @test size(plasma_pts) == (16, 3)
            end

            @testset "in-place compute_vacuum_response! matches wrapper" begin
                # The allocating wrapper is a thin caller of the in-place routine; verify the
                # in-place entry populates caller-owned storage identically.
                for wall_settings in (WallShapeSettings(shape="nowall"), WallShapeSettings(shape="conformal", a=0.5))
                    inputs = _make_inputs()
                    ref = compute_vacuum_response(inputs, wall_settings; compute_Iv=true)

                    vac = VacuumResponse(inputs)
                    compute_vacuum_response!(vac, inputs, wall_settings; compute_Iv=true)

                    @test vac.wv ≈ ref.wv
                    @test vac.I_v ≈ ref.I_v
                    @test vac.plasma_pts ≈ ref.plasma_pts
                    @test vac.wall_pts ≈ ref.wall_pts
                end
            end

            @testset "in-place compute_vacuum_response! clears a reused buffer" begin
                # A buffer left over from an earlier run must not leak into the next result:
                # I_v in particular is only written when compute_Iv=true.
                inputs = _make_inputs()

                vac = VacuumResponse(inputs)
                compute_vacuum_response!(vac, inputs, WallShapeSettings(; shape="conformal", a=0.5); compute_Iv=true)
                @test !all(iszero, vac.I_v)

                compute_vacuum_response!(vac, inputs, WallShapeSettings(; shape="nowall"))
                fresh = compute_vacuum_response(inputs, WallShapeSettings(; shape="nowall"))

                @test all(iszero, vac.I_v)
                @test vac.wv ≈ fresh.wv
            end
        end

        @testset "extract_plasma_surface_at_psi" begin
            # Self-contained analytic Solovev equilibrium (same recipe as runtests_equil.jl).
            eq_config = Equilibrium.EquilibriumConfig(; eq_type="sol", eq_filename="unused", jac_type="pest", grid_type="ldp", psilow=1e-4, psihigh=0.99999, mpsi=64, mtheta=128)
            sol_config = Equilibrium.SolovevConfig(64, 64, 64, 1.6, 0.33, 1.0, 1.9, 1.0, 1.0, 1.0)
            pe = Equilibrium.equilibrium_solver(Equilibrium.sol_run(eq_config, sol_config))

            mtheta = length(pe.rzphi_ys)
            r, z, ν = extract_plasma_surface_at_psi(pe, 0.5)
            @test length(r) == length(z) == length(ν) == mtheta
            @test all(isfinite, r) && all(isfinite, z) && all(isfinite, ν)
            @test all(r .> 0)
            @test minimum(r) < pe.ro < maximum(r)        # surface brackets the magnetic axis in R

            extent(rr, zz) = maximum(hypot.(rr .- pe.ro, zz .- pe.zo))
            r_in, z_in, _ = extract_plasma_surface_at_psi(pe, 0.2)
            r_out, z_out, _ = extract_plasma_surface_at_psi(pe, 0.8)
            @test extent(r_out, z_out) > extent(r_in, z_in)   # minor radius grows outward in ψ
        end

        @testset "calc_surface_inductance" begin
            # Same Solovev recipe as above; exercises the PerturbedEquilibrium helper end-to-end
            eq_config = Equilibrium.EquilibriumConfig(; eq_type="sol", eq_filename="unused", jac_type="pest", grid_type="ldp", psilow=1e-4, psihigh=0.99999, mpsi=64, mtheta=128)
            sol_config = Equilibrium.SolovevConfig(64, 64, 64, 1.6, 0.33, 1.0, 1.9, 1.0, 1.0, 1.0)
            pe = Equilibrium.equilibrium_solver(Equilibrium.sol_run(eq_config, sol_config))

            m_modes = -2:5
            L = PerturbedEquilibrium.calc_surface_inductance(pe, 0.9, 64, m_modes, 1)
            @test size(L) == (length(m_modes), length(m_modes))
            @test all(isfinite, L)
            @test isapprox(L, L', rtol=1e-8)   # Hermitian inductance
        end
    end

    # 3D vacuum: nzeta > 1, full (m,n) coupling, PlasmaGeometry3D, WallGeometry3D
    # Kernel requires mtheta, nzeta >= PATCH_DIM (23 for default PATCH_RAD=11)
    @testset "Vacuum.jl (3D)" begin
        _make_3d_inputs(; mtheta=32, mtheta_eq=17, m_modes=1:2, n_modes=0:1, nzeta=32) = VacuumInput(
            mtheta_in=mtheta_eq,
            nzeta_in=1,
            x=collect(1.7 .+ 0.3 .* cos.(range(0, 2π, length=mtheta_eq))),
            z=collect(0.3 .* sin.(range(0, 2π, length=mtheta_eq))),
            ν=zeros(mtheta_eq),
            m_modes=collect(Int, m_modes),
            n_modes=collect(Int, n_modes),
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
                m_modes=collect(1:mpert),
                n_modes=collect(nlow:(nlow+npert-1)),
                mtheta=mtheta,
                nzeta=nzeta
            )
        end

        @testset "VacuumInput nzeta > 1" begin
            vac = VacuumInput(mtheta=32, nzeta=24, m_modes=[1, 2], n_modes=[1, 2])
            @test vac.nzeta == 24
            @test vac.mtheta == 32
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
            wall = GeneralizedPerturbedEquilibrium.Vacuum.WallGeometry3D(inputs, GeneralizedPerturbedEquilibrium.Vacuum.PlasmaGeometry3D(inputs), WallShapeSettings(shape="nowall"))
            @test wall.nowall == true
            @test wall.mtheta == 32
            @test wall.nzeta == 32
            @test size(wall.r) == (32 * 32, 3)
        end

        @testset "WallGeometry3D conformal" begin
            inputs = _make_3d_inputs(mtheta=32, nzeta=32, mtheta_eq=17)
            wall = GeneralizedPerturbedEquilibrium.Vacuum.WallGeometry3D(
                inputs,
                GeneralizedPerturbedEquilibrium.Vacuum.PlasmaGeometry3D(inputs),
                WallShapeSettings(shape="conformal", a=0.2)
            )
            @test wall.nowall == false
            @test wall.mtheta == 32
            @test wall.nzeta == 32
            num_points = 32 * 32
            @test size(wall.r) == (num_points, 3)
            @test size(wall.normal) == (num_points, 3)
            @test all(isfinite, wall.r)
            @test all(isfinite, wall.normal)
        end

        # Corrugated torus on a genuinely periodic (endpoint-excluded) grid. The shared
        # _make_3d_nonaxis_inputs helper samples range(0, 2π, length=n), which repeats the seam
        # point and leaves the surface non-smooth there — harmless for a nowall response, but it
        # makes the offset surface fold, so the wall tests build their own boundary.
        _make_3d_periodic_inputs(; mtheta=24, nzeta=24, mtheta_in=16, nzeta_in=16) = begin
            θ_in = range(; start=0, length=mtheta_in, step=2π/mtheta_in)
            ζ_in = range(; start=0, length=nzeta_in, step=2π/nzeta_in)
            X = zeros(mtheta_in, nzeta_in)
            Y = similar(X)
            Z = similar(X)
            for (i, θ) in enumerate(θ_in), (j, ζ) in enumerate(ζ_in)
                R = 1.7 + 0.3 * cos(θ) + 0.05 * cos(2ζ) * cos(θ)
                X[i, j] = R * cos(ζ)
                Y[i, j] = R * sin(ζ)
                Z[i, j] = 0.3 * sin(θ) + 0.05 * sin(2ζ) * sin(θ)
            end
            VacuumInput(x=vec(X), y=vec(Y), z=vec(Z), mtheta_in=mtheta_in, nzeta_in=nzeta_in,
                m_modes=[1, 2], n_modes=[0, 1], mtheta=mtheta, nzeta=nzeta)
        end

        @testset "WallGeometry3D conformal, non-axisymmetric input" begin
            inputs = _make_3d_periodic_inputs(mtheta=24, nzeta=24)
            settings = WallShapeSettings(shape="conformal", a=0.2, equal_arc_wall=false)
            plasma = GeneralizedPerturbedEquilibrium.Vacuum.PlasmaGeometry3D(inputs)
            wall = GeneralizedPerturbedEquilibrium.Vacuum.WallGeometry3D(inputs, plasma, settings)

            num_points = 24 * 24
            @test wall.nowall == false
            @test size(wall.r) == (num_points, 3)
            @test all(isfinite, wall.r) && all(isfinite, wall.normal)

            # Uniform offset along the plasma normal: every point moves the same distance, outward.
            offsets = [norm(wall.r[i, :] - plasma.r[i, :]) for i in 1:num_points]
            @test maximum(offsets) - minimum(offsets) < 1e-12
            @test all(hypot.(wall.r[:, 1], wall.r[:, 2]) .> 0)
            R_wall = [hypot(wall.r[i, 1], wall.r[i, 2]) for i in 1:num_points]
            R_plasma = [hypot(plasma.r[i, 1], plasma.r[i, 2]) for i in 1:num_points]
            @test maximum(R_wall) > maximum(R_plasma)

            # The offset point is the closest wall point to its own plasma point, which is the
            # index alignment the near-field patch assumes.
            for i in (1, 300, num_points)
                @test offsets[i] ≈ minimum(norm(wall.r[j, :] - plasma.r[i, :]) for j in 1:num_points)
            end

            # Wall normals face out of the vacuum region, opposite the inward plasma normals.
            aligns = [dot(wall.normal[i, :], plasma.normal[i, :]) for i in 1:num_points]
            @test all(aligns .< 0)
        end

        @testset "WallGeometry3D non-axisymmetric error paths" begin
            inputs = _make_3d_periodic_inputs(mtheta=24, nzeta=24)
            plasma = GeneralizedPerturbedEquilibrium.Vacuum.PlasmaGeometry3D(inputs)
            for shape in ("elliptical", "dee", "mod_dee", "some_wall_file.dat")
                @test_throws ErrorException GeneralizedPerturbedEquilibrium.Vacuum.WallGeometry3D(inputs, plasma, WallShapeSettings(shape=shape))
            end
            # equal_arc_wall has no meaning without a 2D contour and is ignored with a warning
            @test_logs (:warn, r"equal_arc_wall is ignored") match_mode=:any GeneralizedPerturbedEquilibrium.Vacuum.WallGeometry3D(
                inputs,
                plasma,
                WallShapeSettings(shape="conformal", a=0.2, equal_arc_wall=true)
            )
            # a full-torus boundary is required; expand_field_periods must run first
            per_period = VacuumInput(x=inputs.x, y=inputs.y, z=inputs.z, mtheta_in=16, nzeta_in=16,
                m_modes=inputs.m_modes, n_modes=inputs.n_modes, mtheta=24, nzeta=24, nfp=3)
            @test_throws ErrorException GeneralizedPerturbedEquilibrium.Vacuum.WallGeometry3D(per_period, plasma, WallShapeSettings(shape="conformal", a=0.2))
            # an offset that folds the surface is rejected rather than silently returned
            folded = _make_3d_nonaxis_inputs(mtheta=24, nzeta=24, mtheta_in=12, nzeta_in=12)
            @test_throws ErrorException GeneralizedPerturbedEquilibrium.Vacuum.WallGeometry3D(
                folded,
                GeneralizedPerturbedEquilibrium.Vacuum.PlasmaGeometry3D(folded),
                WallShapeSettings(shape="conformal", a=0.2, equal_arc_wall=false)
            )
        end

        @testset "WallGeometry3D shape selection" begin
            # Every shape reaches the right builder through the same chain WallGeometry uses in 2D.
            # elongated so the elliptical branch has a well-defined focal distance
            θ_eq = range(0, 2π, length=33)[1:32]
            axi = VacuumInput(mtheta_in=32, nzeta_in=1, x=collect(1.7 .+ 0.3 .* cos.(θ_eq)),
                z=collect(0.45 .* sin.(θ_eq)), ν=zeros(32), m_modes=[1, 2], n_modes=[0, 1],
                mtheta=32, nzeta=32)
            for (shape, settings) in (("elliptical", WallShapeSettings(shape="elliptical", a=0.5)),
                ("dee", WallShapeSettings(shape="dee", a=0.3)),
                ("mod_dee", WallShapeSettings(shape="mod_dee", a=0.5, cw=1.7)))
                wall = GeneralizedPerturbedEquilibrium.Vacuum.WallGeometry3D(axi, GeneralizedPerturbedEquilibrium.Vacuum.PlasmaGeometry3D(axi), settings)
                @test size(wall.r) == (32 * 32, 3)
                @test all(isfinite, wall.normal)
                # revolved shapes are axisymmetric: R and Z repeat from one toroidal plane to the next
                @test hypot(wall.r[1, 1], wall.r[1, 2]) ≈ hypot(wall.r[1+32, 1], wall.r[1+32, 2])
                @test wall.r[1, 3] ≈ wall.r[1+32, 3]
            end
            # Gap warning uses the measured same-index separation, not a conformal-only offset that
            # is 0 for these shapes. a = 1 m sits well outside one coarser cell, so no sub-cell warning.
            @test_logs min_level=Base.CoreLogging.Warn GeneralizedPerturbedEquilibrium.Vacuum.WallGeometry3D(
                axi, GeneralizedPerturbedEquilibrium.Vacuum.PlasmaGeometry3D(axi), WallShapeSettings(shape="elliptical", a=1.0)
            )
            # an unreadable wall file is still reported by the 2D reader
            @test_throws ErrorException GeneralizedPerturbedEquilibrium.Vacuum.WallGeometry3D(
                axi,
                GeneralizedPerturbedEquilibrium.Vacuum.PlasmaGeometry3D(axi),
                WallShapeSettings(shape="no_such_wall.dat")
            )
        end

        @testset "compute_vacuum_response 3D nowall" begin
            inputs = _make_3d_inputs(mtheta=32, nzeta=32, mtheta_eq=17)
            wall_settings = WallShapeSettings(shape="nowall")
            (; wv, I_v, plasma_pts, wall_pts) = compute_vacuum_response(inputs, wall_settings)

            numpoints = inputs.mtheta * inputs.nzeta
            num_modes = length(inputs.m_modes) * length(inputs.n_modes)
            @test size(wv) == (num_modes, num_modes)
            @test eltype(wv) == ComplexF64
            @test all(isfinite, wv)
            @test size(I_v) == (num_modes, num_modes)
            @test all(iszero, I_v)
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
            (; wv, I_v, plasma_pts, wall_pts) = compute_vacuum_response(inputs, wall_settings)

            numpoints = inputs.mtheta * inputs.nzeta
            num_modes = length(inputs.m_modes) * length(inputs.n_modes)

            @test size(wv) == (num_modes, num_modes)
            @test eltype(wv) == ComplexF64
            @test all(isfinite, wv)
            @test size(I_v) == (num_modes, num_modes)
            @test all(iszero, I_v)
            @test size(plasma_pts) == (numpoints, 3)
            @test all(isfinite, plasma_pts)
            @test size(wall_pts) == (numpoints, 3)
            @test isapprox(wv, wv', rtol=1e-12)
        end

        @testset "compute_vacuum_response 3D conformal wall" begin
            inputs = _make_3d_inputs(mtheta=32, nzeta=32, mtheta_eq=17)
            wall_settings = WallShapeSettings(shape="conformal", a=0.3)
            (; wv, I_v, plasma_pts, wall_pts) = compute_vacuum_response(inputs, wall_settings)

            num_modes = length(inputs.m_modes) * length(inputs.n_modes)
            @test size(wv) == (num_modes, num_modes)
            @test size(I_v) == (num_modes, num_modes)
            @test all(isfinite, plasma_pts)
            @test all(isfinite, wall_pts)
            # Wall and plasma should differ (conformal wall offset from plasma)
            @test !isapprox(plasma_pts, wall_pts)
            @test isapprox(wv, wv', rtol=1e-12)
        end

        @testset "compute_vacuum_response 3D compute_Iv=true" begin
            num_modes(inp) = length(inp.m_modes) * length(inp.n_modes)
            for wall_settings in (WallShapeSettings(shape="nowall"), WallShapeSettings(shape="conformal", a=0.3))
                inputs = _make_3d_inputs(mtheta=32, nzeta=32, mtheta_eq=17)
                (; I_v) = compute_vacuum_response(inputs, wall_settings; compute_Iv=true)
                @test size(I_v) == (num_modes(inputs), num_modes(inputs))
                @test all(isfinite, I_v)
                @test !all(iszero, I_v)
                @test isapprox(I_v, I_v', rtol=1e-8)
            end

            # A reused buffer must not keep a stale I_v from an earlier compute_Iv=true solve
            inputs = _make_3d_inputs(mtheta=32, nzeta=32, mtheta_eq=17)
            vac = GeneralizedPerturbedEquilibrium.Vacuum.VacuumResponse(inputs)
            compute_vacuum_response!(vac, inputs, WallShapeSettings(shape="nowall"); compute_Iv=true)
            @test !all(iszero, vac.I_v)
            compute_vacuum_response!(vac, inputs, WallShapeSettings(shape="nowall"))
            @test all(iszero, vac.I_v)
        end

        # The 3D interior operator is the 2D one shifted by the same scalar, D_int = D_ext - 2I, so an
        # axisymmetric boundary driven through both paths must give the same Iᵛ. Tolerances are loose
        # because Iᵛ is a difference of two solves, which amplifies the 3D toroidal discretization error
        # (~8e-3 on wv here) by an order of magnitude; a wrong shift or sign gives O(1) instead.
        @testset "compute_vacuum_response 3D I_v matches the 2D path" begin
            mtheta = 48
            θ = range(; start=0, length=mtheta, step=2π/mtheta)
            # Up-down asymmetric so Iᵛ is genuinely complex and the θ_VAC → -θ_VAC conjugation is observable
            R = 1.7 .+ 0.3 .* cos.(θ)
            Z = 0.3 .* sin.(θ) .+ 0.08 .* sin.(2θ) .+ 0.08 .* cos.(θ)
            # Arrays are reversed for VACUUM's CW θ, as the equilibrium-based constructor does
            make(nzeta) = VacuumInput(x=collect(reverse(R)), z=collect(reverse(Z)), ν=zeros(mtheta),
                mtheta_in=mtheta, nzeta_in=1, m_modes=[-2, -1, 0, 1, 2], n_modes=[1], mtheta=mtheta, nzeta=nzeta)
            nowall = WallShapeSettings(shape="nowall")

            r2d = compute_vacuum_response(make(1), nowall; compute_Iv=true)
            r3d = compute_vacuum_response(make(mtheta), nowall; compute_Iv=true)

            @test norm(r3d.wv - r2d.wv) / norm(r2d.wv) < 2e-2
            @test norm(r3d.I_v - r2d.I_v) / norm(r2d.I_v) < 0.2
            # The imaginary parts must agree in sign, not be opposed — this is what pins the conjugation
            @test norm(imag.(r3d.I_v) - imag.(r2d.I_v)) < norm(imag.(r3d.I_v) + imag.(r2d.I_v))
        end

        # Field-periodic (layer-2) reduction: an nfp-periodic boundary makes the boundary-integral
        # operators block-circulant, so the reduced per-residue-class solve must reproduce the full
        # torus result (the n_stride=1 bridge case spans several residue classes mod nfp).
        @testset "compute_vacuum_response 3D field-periodic reduction" begin
            # Build one field period of a 3D boundary on the full-torus ζ spacing; expand_field_periods
            # tiles it by rigid rotation, so reduced and full paths share an identical full-torus surface.
            _make_periodic_period(; mtheta, nzeta_p, nfp, R0=1.7, a=0.3, b=0.3, ε=0.04) = begin
                θ = range(; start=0, length=mtheta, step=2π/mtheta)
                X = zeros(mtheta, nzeta_p)
                Y = zeros(mtheta, nzeta_p)
                Z = zeros(mtheta, nzeta_p)
                for j in 1:nzeta_p
                    ζ = (j - 1) * 2π / (nzeta_p * nfp)   # period 0 of the full-torus grid
                    for (i, θi) in enumerate(θ)
                        R = R0 + a * cos(θi) + ε * cos(θi + nfp * ζ)
                        X[i, j] = R * cos(ζ)
                        Y[i, j] = R * sin(ζ)
                        Z[i, j] = b * sin(θi) + ε * sin(θi + nfp * ζ)
                    end
                end
                return vec(X), vec(Y), vec(Z)
            end

            mtheta, nzeta_p, nfp = 24, 8, 3        # full torus nzeta = 24 ≥ PATCH_DIM (23)
            m_modes = collect(-1:1)
            n_modes = [1, 2, 3, 4]                 # residues {1, 2, 0, 1} mod nfp exercise grouping
            Xp, Yp, Zp = _make_periodic_period(; mtheta=mtheta, nzeta_p=nzeta_p, nfp=nfp)

            inputs_red = VacuumInput(
                x=Xp, y=Yp, z=Zp,
                mtheta_in=mtheta, nzeta_in=nzeta_p,
                m_modes=m_modes, n_modes=n_modes,
                mtheta=mtheta, nzeta=nzeta_p,
                nfp=nfp
            )
            wall_settings = WallShapeSettings(shape="nowall")

            # Reduced (block-circulant) path
            vac_red = compute_vacuum_response(inputs_red, wall_settings)
            wv_red, plasma_pts_red = vac_red.wv, vac_red.plasma_pts

            # Full-torus reference: pre-expand so nfp=1 forces the dense path
            inputs_full = GeneralizedPerturbedEquilibrium.Vacuum.expand_field_periods(inputs_red)
            @test inputs_full.nfp == 1
            @test inputs_full.nzeta == nzeta_p * nfp
            wv_full = compute_vacuum_response(inputs_full, wall_settings).wv

            num_modes = length(m_modes) * length(n_modes)
            @test size(wv_red) == (num_modes, num_modes)
            @test eltype(wv_red) == ComplexF64
            @test all(isfinite, wv_red)
            # plasma points are the full torus
            @test size(plasma_pts_red, 1) == mtheta * nzeta_p * nfp

            # Reduced == dense full-torus result (block-circulant solve is an exact reorganization)
            @test isapprox(wv_red, wv_full; rtol=1e-6, atol=1e-7)

            # Modes in different residue classes mod nfp do not couple in the reduced result
            classes = mod.(n_modes, nfp)
            mpert = length(m_modes)
            for in1 in eachindex(n_modes), in2 in eachindex(n_modes)
                if classes[in1] != classes[in2]
                    rows = ((in1-1)*mpert+1):(in1*mpert)
                    cols = ((in2-1)*mpert+1):(in2*mpert)
                    @test all(iszero, wv_red[rows, cols])
                end
            end

            # Hermitian part is enforced after assembly on both paths
            @test isapprox(wv_red, wv_red', rtol=1e-12)

            # The interior solve is a scalar shift of the exterior one, so it decomposes by the same
            # residue class and Iᵛ must reduce exactly like wv does.
            Iv_red = compute_vacuum_response(inputs_red, wall_settings; compute_Iv=true).I_v
            Iv_full = compute_vacuum_response(inputs_full, wall_settings; compute_Iv=true).I_v
            @test all(isfinite, Iv_red)
            @test !all(iszero, Iv_red)
            @test isapprox(Iv_red, Iv_full; rtol=1e-6, atol=1e-7)
            for in1 in eachindex(n_modes), in2 in eachindex(n_modes)
                if classes[in1] != classes[in2]
                    @test all(iszero, Iv_red[((in1-1)*mpert+1):(in1*mpert), ((in2-1)*mpert+1):(in2*mpert)])
                end
            end

            # A wall adds a second source block to the operator, so repeat the check with one present:
            # the field-period fold has to land the wall columns in the right block for both to agree.
            walled = WallShapeSettings(shape="conformal", a=0.2, equal_arc_wall=false)
            vac_red_wall = compute_vacuum_response(inputs_red, walled; compute_Iv=true)
            vac_full_wall = compute_vacuum_response(inputs_full, walled; compute_Iv=true)
            @test isapprox(vac_red_wall.wv, vac_full_wall.wv; rtol=1e-6, atol=1e-7)
            @test isapprox(vac_red_wall.I_v, vac_full_wall.I_v; rtol=1e-6, atol=1e-7)
        end

        @testset "compute_vacuum_response 3D stellarator symmetry" begin
            # Rotating ellipse: R(-θ,-ζ) = R(θ,ζ) and Z(-θ,-ζ) = -Z(θ,ζ), so the surface is
            # stellarator symmetric. Adding `odd` breaks that symmetry without changing anything else.
            _stell_boundary(; mtheta, nzeta_p, nfp, R0=1.7, a=0.3, b=0.09, odd=0.0) = begin
                X = Float64[]
                Y = Float64[]
                Z = Float64[]
                for j in 1:nzeta_p
                    ζ = (j - 1) * 2π / (nzeta_p * nfp)
                    for i in 1:mtheta
                        θi = (i - 1) * 2π / mtheta
                        R = R0 + a * cos(θi) + b * cos(θi - nfp * ζ) + odd * sin(θi - nfp * ζ)
                        push!(X, R * cos(ζ))
                        push!(Y, R * sin(ζ))
                        push!(Z, -a * sin(θi) + b * sin(θi - nfp * ζ) + odd * cos(2θi - nfp * ζ))
                    end
                end
                return X, Y, Z
            end
            _stell_inputs(; mtheta, nzeta_p, nfp, n_modes, odd=0.0) = begin
                X, Y, Z = _stell_boundary(; mtheta=mtheta, nzeta_p=nzeta_p, nfp=nfp, odd=odd)
                return VacuumInput(
                    x=X, y=Y, z=Z,
                    mtheta_in=mtheta, nzeta_in=nzeta_p,
                    m_modes=collect(-1:1), n_modes=n_modes,
                    mtheta=mtheta, nzeta=nzeta_p,
                    nfp=nfp
                )
            end

            mtheta, nzeta_p = 24, 8
            nowall = WallShapeSettings(shape="nowall")
            walled = WallShapeSettings(shape="conformal", a=0.2, equal_arc_wall=false)

            # The whole item rests on the operator inheriting the involution, so assert it directly.
            inputs = _stell_inputs(; mtheta=mtheta, nzeta_p=nzeta_p, nfp=3, n_modes=[1])
            full = GeneralizedPerturbedEquilibrium.Vacuum.expand_field_periods(inputs)
            plasma = GeneralizedPerturbedEquilibrium.Vacuum.PlasmaGeometry3D(full)
            wall = GeneralizedPerturbedEquilibrium.Vacuum.WallGeometry3D(full, plasma, walled)
            npts = plasma.mtheta * plasma.nzeta
            σ_full = [mod1(2 - mod1(p, plasma.mtheta), plasma.mtheta) +
                      plasma.mtheta * (mod1(2 - ((p - 1) ÷ plasma.mtheta + 1), plasma.nzeta) - 1) for p in 1:npts]
            D = zeros(npts, npts)
            S = zeros(npts, npts)
            GeneralizedPerturbedEquilibrium.Vacuum.compute_3D_kernel_matrices!([D], [S], plasma, plasma, 11, 20, 5, [1.0])
            @test isapprox(D[σ_full, σ_full], D; rtol=1e-9, atol=1e-9 * maximum(abs, D))
            @test isapprox(S[σ_full, σ_full], S; rtol=1e-9, atol=1e-9 * maximum(abs, S))

            # Detection: symmetric surfaces are recognised, an odd-parity perturbation is not
            @test GeneralizedPerturbedEquilibrium.Vacuum.stellarator_involution(plasma, wall, 3) !== nothing
            asym = _stell_inputs(; mtheta=mtheta, nzeta_p=nzeta_p, nfp=3, n_modes=[1], odd=0.07)
            asym_full = GeneralizedPerturbedEquilibrium.Vacuum.expand_field_periods(asym)
            asym_plasma = GeneralizedPerturbedEquilibrium.Vacuum.PlasmaGeometry3D(asym_full)
            asym_wall = GeneralizedPerturbedEquilibrium.Vacuum.WallGeometry3D(asym_full, asym_plasma, walled)
            @test GeneralizedPerturbedEquilibrium.Vacuum.stellarator_involution(asym_plasma, asym_wall, 3) === nothing

            # The symmetry-adapted solve must reproduce the untransformed one. nfp = 1 and k = 0 split
            # into two real half-size blocks; k ≠ 0 becomes real at full size; nfp = 4, k = 2 is the
            # self-conjugate class that needs the signed involution rather than the half-twist.
            for (nfp, n_modes, wall_settings) in [
                (1, [1], nowall), (1, [1], walled),
                (3, [0], walled), (3, [1], nowall), (3, [1], walled),
                (3, [1, 2, 3], walled), (4, [2], walled), (2, [1], walled)
            ]
                inp = _stell_inputs(; mtheta=mtheta, nzeta_p=nzeta_p, nfp=nfp, n_modes=n_modes)
                sym = compute_vacuum_response(inp, wall_settings; compute_Iv=true, use_symmetry=true)
                ref = compute_vacuum_response(inp, wall_settings; compute_Iv=true, use_symmetry=false)
                @test isapprox(sym.wv, ref.wv; rtol=1e-9, atol=1e-9 * maximum(abs, ref.wv))
                @test isapprox(sym.I_v, ref.I_v; rtol=1e-9, atol=1e-9 * maximum(abs, ref.I_v))
            end

            # An asymmetric boundary must fall through to exactly the untransformed solve
            sym = compute_vacuum_response(asym, walled; compute_Iv=true, use_symmetry=true)
            ref = compute_vacuum_response(asym, walled; compute_Iv=true, use_symmetry=false)
            @test sym.wv == ref.wv
            @test sym.I_v == ref.I_v
        end

        @testset "compute_vacuum_response 3D conjugate class pairing" begin
            _cg = GeneralizedPerturbedEquilibrium.Vacuum._conjugate_groups

            # k pairs with mod(nfp - k, nfp) when that class is present and k is not self-conjugate
            @test _cg([0, 1, 2], 3, true) == [[1], [2, 3]]
            @test _cg([0, 1, 2], 3, false) == [[1], [2], [3]]
            @test _cg([1, 2, 3], 5, true) == [[1], [2, 3]]   # class 4 absent, so 1 stays alone
            @test _cg([0, 1], 2, true) == [[1], [2]]         # both self-conjugate when nfp = 2
            @test _cg([2], 4, true) == [[1]]                 # k = nfp/2 is self-conjugate
            @test _cg([0], 1, true) == [[1]]

            # Rotating ellipse, stellarator symmetric so both operator paths are reachable; `odd`
            # breaks the symmetry and forces the untransformed path while pairing still applies.
            _pair_boundary(; mtheta, nzeta_p, nfp, odd=0.0) = begin
                R0, a, b = 1.7, 0.3, 0.09
                X = Float64[]
                Y = Float64[]
                Z = Float64[]
                for j in 1:nzeta_p
                    ζ = (j - 1) * 2π / (nzeta_p * nfp)
                    for i in 1:mtheta
                        θi = (i - 1) * 2π / mtheta
                        R = R0 + a * cos(θi) + b * cos(θi - nfp * ζ) + odd * sin(θi - nfp * ζ)
                        push!(X, R * cos(ζ))
                        push!(Y, R * sin(ζ))
                        push!(Z, -a * sin(θi) + b * sin(θi - nfp * ζ) + odd * cos(2θi - nfp * ζ))
                    end
                end
                return X, Y, Z
            end
            _pair_inputs(; mtheta, nzeta_p, nfp, n_modes, odd=0.0) = begin
                X, Y, Z = _pair_boundary(; mtheta=mtheta, nzeta_p=nzeta_p, nfp=nfp, odd=odd)
                return VacuumInput(
                    x=X, y=Y, z=Z,
                    mtheta_in=mtheta, nzeta_in=nzeta_p,
                    m_modes=collect(-1:1), n_modes=n_modes,
                    mtheta=mtheta, nzeta=nzeta_p,
                    nfp=nfp
                )
            end

            mtheta, nzeta_p = 24, 8
            nowall = WallShapeSettings(shape="nowall")
            walled = WallShapeSettings(shape="conformal", a=0.2, equal_arc_wall=false)

            # D̂₋ₖ = conj(D̂ₖ), so serving the conjugate class from the representative's factorization
            # must reproduce an independent solve. Agreement is to roundoff, not bitwise: the two
            # paths build the class phases from different arguments.
            for (nfp, n_modes, wall_settings) in [
                (3, [1, 2, 3, 4], walled), (5, collect(1:4), nowall), (5, [2, 3], walled)
            ]
                for use_symmetry in (false, true)
                    inp = _pair_inputs(; mtheta=mtheta, nzeta_p=nzeta_p, nfp=nfp, n_modes=n_modes)
                    pair = compute_vacuum_response(inp, wall_settings; compute_Iv=true, use_symmetry=use_symmetry, use_conjugate_pairing=true)
                    ref = compute_vacuum_response(inp, wall_settings; compute_Iv=true, use_symmetry=use_symmetry, use_conjugate_pairing=false)
                    @test isapprox(pair.wv, ref.wv; rtol=1e-9, atol=1e-9 * maximum(abs, ref.wv))
                    @test isapprox(pair.I_v, ref.I_v; rtol=1e-9, atol=1e-9 * maximum(abs, ref.I_v))
                end
            end

            # Pairing is independent of stellarator symmetry: an asymmetric boundary still pairs
            asym = _pair_inputs(; mtheta=mtheta, nzeta_p=nzeta_p, nfp=3, n_modes=[1, 2, 3, 4], odd=0.07)
            pair = compute_vacuum_response(asym, walled; compute_Iv=true, use_conjugate_pairing=true)
            ref = compute_vacuum_response(asym, walled; compute_Iv=true, use_conjugate_pairing=false)
            @test isapprox(pair.wv, ref.wv; rtol=1e-9, atol=1e-9 * maximum(abs, ref.wv))
            @test isapprox(pair.I_v, ref.I_v; rtol=1e-9, atol=1e-9 * maximum(abs, ref.I_v))

            # No class can pair when nfp ≤ 2 or when the modes form a single family, so those runs
            # must take exactly the unpaired code path
            for (nfp, n_modes) in [(1, [1]), (2, [1, 2]), (5, [1, 6])]
                inp = _pair_inputs(; mtheta=mtheta, nzeta_p=nzeta_p, nfp=nfp, n_modes=n_modes)
                pair = compute_vacuum_response(inp, walled; compute_Iv=true, use_conjugate_pairing=true)
                ref = compute_vacuum_response(inp, walled; compute_Iv=true, use_conjugate_pairing=false)
                @test pair.wv == ref.wv
                @test pair.I_v == ref.I_v
            end
        end

        @testset "Kernel3D laplace_kernel" begin
            G, K = GeneralizedPerturbedEquilibrium.Vacuum.laplace_kernel(1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0)
            # Kernel returns 1/|r_obs - r_src| (4π factor applied elsewhere in BIE)
            dist = sqrt((2.0 - 1.0)^2 + 0 + 0)
            @test isapprox(G, 1.0 / dist)
            @test isfinite(G)
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
