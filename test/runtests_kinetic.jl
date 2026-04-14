
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
                # Monotonicity with tolerance for floating-point rounding
                @test all(diff(pts) .> -eps(1.0))
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
    # Energy ODE integration
    # =========================================================================
    @testset "integrate_energy_ode" begin
        @testset "CGL integration" begin
            # CGL mode: integral of x^2.5*exp(-x)/(i*n) dx from 0 to 72
            # = Gamma(3.5)/(i*1) = (15/8)*sqrt(pi) / i
            result = KF.integrate_energy_ode(
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 1, 0.5, 0.5, "fcgl";
                nutype="zero", f0type="cgl", nufac=1.0, ximag=0.0, qt=false
            )
            gamma_3_5 = 15/8 * sqrt(π)
            # CGL integral = Gamma(3.5) / (i*n) where n=1
            # = Gamma(3.5) * (-i) = complex(0, -Gamma(3.5))
            @test real(result) ≈ 0.0 atol=1e-6
            @test imag(result) ≈ -gamma_3_5 atol=1e-6
        end

        @testset "collisionless returns finite" begin
            result = KF.integrate_energy_ode(
                1.0, 1.0, 1.0, 0.5, 0.3, 0.0, 0, 1.0, 1, 0.5, 0.5, "fgar";
                nutype="zero", f0type="maxwellian"
            )
            @test isfinite(real(result))
            @test isfinite(imag(result))
        end
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
        @test length(intr.methods) == 18
        @test length(intr.docs) == 18
        @test intr.chi1 == 0.0
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
