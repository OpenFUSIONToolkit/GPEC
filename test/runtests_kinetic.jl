
@testset "KineticForces Unit Tests" begin

    KF = GeneralizedPerturbedEquilibrium.KineticForces

    # =========================================================================
    # powspace grid generation
    # =========================================================================
    @testset "powspace" begin
        @testset "basic properties" begin
            for pow in 1:9
                pts, wts = KF.powspace(0.0, 1.0, pow, 100, "both")
                @test length(pts) == 100
                @test length(wts) == 100
                @test pts[1] ≈ 0.0 atol=1e-14
                @test pts[end] ≈ 1.0 atol=1e-14
                # Monotonicity with tolerance for floating-point rounding.
                # Tolerance scales with polynomial degree because higher-pow
                # antiderivatives evaluate ~2*pow+1 terms near |x|=1 and
                # accumulate FMA/non-FMA ordering noise across Julia minor
                # versions (1.11 → 1.12 reorders fused-multiply-add chains).
                @test all(diff(pts) .> -100 * eps(1.0))
            end
        end

        @testset "endpoint modes" begin
            pts_lower, _ = KF.powspace(0.0, 1.0, 2, 50, "lower")
            pts_upper, _ = KF.powspace(0.0, 1.0, 2, 50, "upper")
            pts_both, _  = KF.powspace(0.0, 1.0, 2, 50, "both")

            # All modes should span the full range
            for pts in [pts_lower, pts_upper, pts_both]
                @test pts[1] ≈ 0.0 atol=1e-14
                @test pts[end] ≈ 1.0 atol=1e-14
            end

            # "lower" concentrates near xmin, "upper" near xmax
            mid = 0.5
            lower_below_mid = count(x -> x < mid, pts_lower)
            upper_below_mid = count(x -> x < mid, pts_upper)
            @test lower_below_mid > upper_below_mid
        end

        @testset "scaling" begin
            pts1, wts1 = KF.powspace(0.0, 1.0, 3, 50, "both")
            pts2, wts2 = KF.powspace(2.0, 5.0, 3, 50, "both")
            @test pts2[1] ≈ 2.0 atol=1e-14
            @test pts2[end] ≈ 5.0 atol=1e-14
        end

        @testset "error cases" begin
            @test_throws ErrorException KF.powspace(1.0, 0.0, 2, 50, "both")
            @test_throws ErrorException KF.powspace(0.0, 1.0, 2, 50, "invalid")
        end
    end

    # =========================================================================
    # _powspace_antideriv
    # =========================================================================
    @testset "_powspace_antideriv" begin
        x = collect(range(-1.0, 1.0, length=101))

        @testset "antisymmetry (all pow)" begin
            for pow in 1:9
                ad = KF._powspace_antideriv(x, pow)
                # |(1-x²)|^pow is symmetric (even), so its antiderivative is odd: f(-x) = -f(x)
                @test ad[1] ≈ -ad[end] atol=1e-12
            end
        end

        @testset "monotonicity near origin" begin
            # The integrand |(1-x²)|^pow is positive near x=0
            # so the antiderivative should be increasing there
            x_near_zero = collect(range(-0.5, 0.5, length=21))
            for pow in 1:9
                ad = KF._powspace_antideriv(x_near_zero, pow)
                if isodd(pow)
                    # Odd pow: antiderivative has negative leading term, so decreasing near 0
                    @test ad[1] > ad[11] || ad[11] < ad[end]
                end
            end
        end

        @testset "unsupported pow" begin
            @test_throws ErrorException KF._powspace_antideriv(x, 10)
        end
    end

    # =========================================================================
    # Energy integrand analytical limits
    # =========================================================================
    @testset "energy_integrand_scalar" begin
        @testset "CGL limit" begin
            # In CGL mode, fx = cx^2.5 * exp(-cx) / (i*n) — no resonance denominator
            p = KF.EnergyParams(
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1,
                "zero", "cgl", 1.0, 0.0, false
            )
            x = 1.0
            val = KF.energy_integrand_scalar(x, p)
            # Expected: x^2.5 * exp(-x) / (i*1) = exp(-1) / i = -i*exp(-1)
            @test real(val) ≈ 0.0 atol=1e-14
            @test imag(val) ≈ -exp(-1.0) atol=1e-12
        end

        @testset "collisionless Maxwellian at large x" begin
            # At large x, the integrand should decay exponentially
            p = KF.EnergyParams(
                1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1,
                "zero", "maxwellian", 1.0, 0.0, false
            )
            val_large = KF.energy_integrand_scalar(50.0, p)
            # At x=50, exp(-50) ≈ 1.9e-22, so integrand should be negligible
            @test abs(val_large) < 1e-15
        end

        @testset "heat flux mode" begin
            # With qt=true, integrand is multiplied by (x - 2.5)
            p_no_qt = KF.EnergyParams(
                1.0, 0.0, 1.0, 0.5, 0.5, 0.0, 1.0, 1,
                "zero", "maxwellian", 1.0, 0.0, false
            )
            p_qt = KF.EnergyParams(
                1.0, 0.0, 1.0, 0.5, 0.5, 0.0, 1.0, 1,
                "zero", "maxwellian", 1.0, 0.0, true
            )
            x = 3.0
            val_no = KF.energy_integrand_scalar(x, p_no_qt)
            val_qt = KF.energy_integrand_scalar(x, p_qt)
            # qt multiplies by (x - 2.5) = 0.5
            @test val_qt ≈ (x - 2.5) * val_no atol=1e-14
        end
    end

    # =========================================================================
    # Resonance root solver
    # =========================================================================
    @testset "find_resonance_energies" begin
        # Roots of Ω(x) = leff·wb·√x + n·(we + wd·x); quadratic n·wd·s² + leff·wb·s + n·we = 0.
        @testset "no real root (negative discriminant)" begin
            roots = KF.find_resonance_energies(1.0, 0.3, 1, 1.0, 0.5)
            @test isempty(roots)
        end

        @testset "linear case (wd = 0)" begin
            # b·s + c = 0 with b = leff·wb = 2, c = n·we = -3 ⟹ s = 1.5, x = 2.25.
            roots = KF.find_resonance_energies(1.0, 2.0, 1, -3.0, 0.0)
            @test length(roots) == 1
            @test roots[1] ≈ 2.25 rtol=1e-12
            # Same but positive we ⟹ s = -1.5 < 0 ⟹ no positive root.
            @test isempty(KF.find_resonance_energies(1.0, 2.0, 1, 3.0, 0.0))
        end

        @testset "two positive roots" begin
            # a = 0.5, b = -3, c = 1 ⟹ s = (3 ± √7)/1, both positive.
            roots = KF.find_resonance_energies(1.0, -3.0, 1, 1.0, 0.5)
            @test length(roots) == 2
            for x in roots
                s = sqrt(x)
                # Ω(x) must vanish at each root.
                @test 1.0 * (-3.0) * s + 1 * (1.0 + 0.5 * x) ≈ 0.0 atol=1e-10
            end
        end
    end

    # =========================================================================
    # Energy integration (u-substitution + Sokhotski-Plemelj pole extraction)
    # =========================================================================
    @testset "integrate_energy" begin
        @testset "CGL integration" begin
            # CGL mode: ∫₀^∞ x^2.5·exp(-x)/(i·n) dx = Γ(3.5)/(i·1) = -i·(15/8)√π.
            result = KF.integrate_energy(
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 1, 0.5, 0.5, "fcgl";
                nutype="zero", f0type="cgl", nufac=1.0, ximag=0.0, qt=false,
                atol=1e-12, rtol=1e-10
            )
            gamma_3_5 = 15/8 * sqrt(π)
            @test real(result) ≈ 0.0 atol=1e-6
            @test imag(result) ≈ -gamma_3_5 atol=1e-6
        end

        @testset "collisionless returns finite" begin
            result = KF.integrate_energy(
                1.0, 1.0, 1.0, 0.5, 0.3, 0.0, 0, 1.0, 1, 0.5, 0.5, "fgar";
                nutype="zero", f0type="maxwellian"
            )
            @test isfinite(real(result))
            @test isfinite(imag(result))
        end

        @testset "collisional matches direct x-space quadrature" begin
            # For ν > 0 the physical integrand N(x)·exp(-x)/denom is finite on the
            # real axis, so a high-accuracy direct integral over [0,∞) is an
            # independent reference for the u-substitution + pole-extraction result.
            wn, wt, we, wd, wb, nuk, leff, n = 0.5, 0.8, -2.0, 0.5, 1.0, 0.3, 1.0, 1
            p = KF.EnergyParams(wn, wt, we, wd, wb, nuk, leff, n,
                                "harmonic", "maxwellian", 1.0, 0.0, false)
            x_res = KF.find_resonance_energies(leff, wb, n, we, wd)
            @test length(x_res) == 1   # this case has exactly one resonance
            reference, _ = KF.quadgk(x -> KF.energy_integrand_scalar(x, p),
                                     0.0, x_res[1], Inf; rtol=1e-12, atol=1e-14)
            result = KF.integrate_energy(
                wn, wt, we, wd, wb, nuk, 0, leff, n, 0.5, 0.5, "fgar";
                nutype="harmonic", f0type="maxwellian", atol=1e-12, rtol=1e-10
            )
            @test result ≈ reference rtol=1e-6
        end

        @testset "collisionless is the ν → 0⁺ limit" begin
            # The collisionless result must be continuous with vanishing collisionality.
            args = (0.5, 0.8, -2.0, 0.5, 1.0)
            tail = (0, 1.0, 1, 0.5, 0.5, "fgar")
            collisionless = KF.integrate_energy(args..., 0.0, tail...;
                nutype="zero", f0type="maxwellian", atol=1e-12, rtol=1e-10)
            small_nu = KF.integrate_energy(args..., 1e-6, tail...;
                nutype="krook", f0type="maxwellian", atol=1e-12, rtol=1e-10)
            @test collisionless ≈ small_nu rtol=1e-3
        end

        @testset "xr > 700 Maxwellian-underflow guard" begin
            # Linear case (wd = 0): s = -c/b = -(n·we)/(leff·wb) = 30 ⟹ x_res = 900 > 700.
            # The pole sits where exp(-x_res) underflows; subtraction trips 0·log(-1) = NaN
            # unless the guard at EnergyIntegration.jl:207 fires first.
            @test KF.find_resonance_energies(1.0, 1.0, 1, -30.0, 0.0)[1] ≈ 900.0
            result = KF.integrate_energy(
                0.5, 0.0, -30.0, 0.0, 1.0, 0.3, 0, 1.0, 1, 0.5, 0.5, "fgar";
                nutype="harmonic", f0type="maxwellian"
            )
            @test isfinite(real(result))
            @test isfinite(imag(result))
        end

        @testset "pole_offset overflow guard" begin
            # Tiny wb gives small omega_prime = leff·wb/(2√x_res) = 5e-16 at x_res=1, while
            # nuk = 1e300 forces nu_res / omega_prime = 2e315 → Inf. The guard at
            # EnergyIntegration.jl:220 must skip this pole rather than build a non-finite
            # u_pole. Parameters keep abs(wb) > 1e-30 so find_resonance_energies still returns x_res.
            @test KF.find_resonance_energies(1.0, 1e-15, 1, -1e-15, 0.0)[1] ≈ 1.0
            result = KF.integrate_energy(
                0.5, 0.0, -1e-15, 0.0, 1e-15, 1e300, 0, 1.0, 1, 0.5, 0.5, "fgar";
                nutype="krook", f0type="maxwellian"
            )
            @test isfinite(real(result))
            @test isfinite(imag(result))
        end
    end

    # =========================================================================
    # evaluate_energy_integrand (exported diagnostic vector form)
    # =========================================================================
    @testset "evaluate_energy_integrand" begin
        x_grid = collect(range(0.1, 5.0, length=32))
        vals = KF.evaluate_energy_integrand(x_grid;
            wn=0.5, wt=0.8, we=-2.0, wd=0.5, wb=1.0, nuk=0.3, leff=1.0, n=1,
            nutype="harmonic", f0type="maxwellian")
        @test vals isa Vector{ComplexF64}
        @test length(vals) == length(x_grid)
        @test all(isfinite, vals)
        # Element-wise match against the scalar form documented in the docstring.
        p = KF.EnergyParams(0.5, 0.8, -2.0, 0.5, 1.0, 0.3, 1.0, 1,
                            "harmonic", "maxwellian", 1.0, 0.0, false)
        @test vals ≈ [KF.energy_integrand_scalar(x, p) for x in x_grid]
    end

    # =========================================================================
    # Struct construction and defaults
    # =========================================================================
    @testset "KineticForcesControl defaults" begin
        ctrl = KF.KineticForcesControl()
        @test ctrl.fgar_flag == true
        @test ctrl.tgar_flag == false
        @test ctrl.nn == 1
        @test ctrl.nl == 1
        @test ctrl.zi == 1
        @test ctrl.mi == 2
        @test ctrl.nutype == "harmonic"
        @test ctrl.f0type == "maxwellian"
        @test ctrl.psilims == [0.0, 1.0]
    end

    @testset "KineticForcesInternal defaults" begin
        intr = KF.KineticForcesInternal()
        @test intr.ro == 0.0
        @test intr.bo == 0.0
        @test intr.mpert == 0
        @test intr.chi1 == 0.0
    end

    @testset "METHOD_REGISTRY" begin
        # The registry is the single source of truth for NTV methods. Every entry's
        # flag must be a real KineticForcesControl field (the TOML kwargs splat relies
        # on it) and carry a recognized dispatch kind. No fixed count is asserted —
        # adding a method should not require editing a magic number here.
        for entry in KF.METHOD_REGISTRY
            @test entry.flag in fieldnames(KF.KineticForcesControl)
            @test endswith(string(entry.flag), "_flag")
            @test entry.kind in (:gar, :fcgl, :rlar, :clar)
            @test KF.method_kind(entry.name) == entry.kind
        end
    end

    @testset "KineticForcesState" begin
        state = KF.KineticForcesState()
        @test !state.completed
        @test isempty(state.method_results)
        @test isempty(state.kinetic_matrices)
    end

    @testset "MethodResult defaults" begin
        mr = KF.MethodResult(method="fgar", nn=1)
        @test mr.total_torque == 0.0 + 0.0im
        @test mr.total_energy == 0.0 + 0.0im
        @test isempty(mr.records)
    end
end
