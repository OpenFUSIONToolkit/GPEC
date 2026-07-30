
@testset "KineticForces Unit Tests" begin

    KF = GeneralizedPerturbedEquilibrium.KineticForces

    # =========================================================================
    # Periodic parallel-velocity spline
    # =========================================================================
    @testset "periodic parallel-velocity spline" begin
        xs = collect(range(0.0, 1.0; length=13))
        B_exact(θ) = 2.0 + 0.25cos(2π * θ) + 0.1sin(4π * θ)
        B_vals = B_exact.(xs)
        B_vpar = KF._fit_vpar_B_spline(xs, B_vals)

        # The physical θ=0/1 seam is C². An endpoint-fit cubic has independent
        # one-sided derivatives here, even when it is wrapped before evaluation.
        @test B_vpar(0.0) ≈ B_vpar(1.0) atol=10eps()
        @test B_vpar(0.0; deriv=DerivOp(1)) ≈
              B_vpar(1.0; deriv=DerivOp(1)) atol=100eps()
        @test B_vpar(0.0; deriv=DerivOp(2)) ≈
              B_vpar(1.0; deriv=DerivOp(2)) atol=1000eps()

        # This level lies above the analytic field maximum. A wrapped endpoint
        # fit creates a false near-seam maximum and two spurious bounce roots.
        artifact_level = 2.3038
        dense_Bmax = maximum(B_exact, range(0.0, 1.0; length=100_001))
        @test dense_Bmax < artifact_level
        @test isempty(KF._find_bounce_roots(B_vpar, 1 / artifact_level, 1.0))

        # Retain PR #310's complete descending root semantics for a physical
        # level that crosses the field twice.
        roots = KF._find_bounce_roots(B_vpar, 1 / 2.2, 1.0)
        @test length(roots) == 2
        @test issorted(roots; rev=true)
        @test all(abs(KF._vpar_from_extrap(B_vpar, 1 / 2.2, 1.0, θ)) < 1e-12
                  for θ in roots)
    end

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

        @testset "collisionless tail-pole resonance (regression: was NaN)" begin
            # Resonance deep in the Maxwellian tail (x_res ≈ 56): n·wd·s² + leff·wb·s + n·we = 0
            # with wb=1e3, wd=5e2, we=-3.5483e4 ⟹ s=√56, x_res≈56. Under the old u-space pole
            # handling u_res = 1-exp(-56) rounds to 1.0, tripping 0·(-Inf)=NaN. The real-x-space
            # branch keeps x_res a well-conditioned O(1) number, and the resonance contribution
            # (∝ exp(-56)) is negligible so the result equals the ν→0⁺ limit (issue #281).
            args = (0.0, 2.0e3, -3.5483e4, 5.0e2, 1.0e3)
            tail = (0, 1.0, 1, 0.5, 0.5, "fgar")
            @test KF.find_resonance_energies(1.0, 1.0e3, 1, -3.5483e4, 5.0e2)[1] ≈ 56.0 rtol = 1e-2
            collisionless = KF.integrate_energy(args..., 0.0, tail...;
                nutype="zero", f0type="maxwellian", atol=1e-12, rtol=1e-10)
            small_nu = KF.integrate_energy(args..., 1e-6, tail...;
                nutype="krook", f0type="maxwellian", atol=1e-12, rtol=1e-10)
            @test isfinite(real(collisionless))
            @test isfinite(imag(collisionless))
            @test collisionless ≈ small_nu rtol = 1e-3
        end

        @testset "Ω′ < 0 collisionless causal branch" begin
            # The unified add-back R·[log(xmax-x_pole) - log(-x_pole)] gets the causal
            # ∓iπ·sign(Ω′) branch from the SIGNED ZERO of pole_offset = ν/Ω′ at ν=0.
            # Pick n·wd < 0 so Ω′ = leff·wb/(2√x) + n·wd flips sign in the tail; the
            # collisionless result must still equal the ν→0⁺ (small-ν krook) limit, which
            # carries the sign naturally through its off-axis pole.
            args = (0.0, 2.0e3, 3.5e4, -5.0e2, 1.0e3)   # we>0, wd<0 ⟹ tail root with Ω′<0
            tail = (0, 1.0, 1, 0.5, 0.5, "fgar")
            collisionless = KF.integrate_energy(args..., 0.0, tail...;
                nutype="zero", f0type="maxwellian", atol=1e-12, rtol=1e-10)
            small_nu = KF.integrate_energy(args..., 1e-6, tail...;
                nutype="krook", f0type="maxwellian", atol=1e-12, rtol=1e-10)
            @test isfinite(collisionless)
            @test collisionless ≈ small_nu rtol = 1e-3
        end

        @testset "tail pole beyond X_ENERGY_MAX dropped" begin
            # Linear case (wd = 0): s = -c/b = -(n·we)/(leff·wb) = 30 ⟹ x_res = 900,
            # well past X_ENERGY_MAX. The pole sits where exp(-x_res) underflows; the
            # `xr >= X_ENERGY_MAX` check in _integrate_energy_resonant drops it cleanly.
            @test KF.find_resonance_energies(1.0, 1.0, 1, -30.0, 0.0)[1] ≈ 900.0
            result = KF.integrate_energy(
                0.5, 0.0, -30.0, 0.0, 1.0, 0.3, 0, 1.0, 1, 0.5, 0.5, "fgar";
                nutype="harmonic", f0type="maxwellian"
            )
            @test isfinite(real(result))
            @test isfinite(imag(result))
        end

        @testset "pole at and within 1e-13 of X_ENERGY_MAX (boundary robustness)" begin
            # Direct analog of the original u→1 bug, relocated to x→X_ENERGY_MAX: a
            # resonance pole sitting exactly at, just below, and just above the upper
            # integration limit. Linear case (wd=0) with leff=wb=n=1 gives
            # x_res = (n·we/(leff·wb))² = we², so we = -√x_target places the pole.
            xmax = KF.X_ENERGY_MAX
            run_xres(x_target) = KF.integrate_energy(
                0.5, 0.3, -sqrt(x_target), 0.0, 1.0, 0.3, 0, 1.0, 1, 0.5, 0.5, "fgar";
                nutype="zero", f0type="maxwellian", atol=1e-12, rtol=1e-10)
            below = run_xres(xmax - 1e-13)   # pole kept, subtracted analytically
            at = run_xres(xmax)              # find_resonance rounds to ≤ xmax; >= guard drops at endpoint
            above = run_xres(xmax + 1e-13)   # pole dropped (outside domain)
            @test isfinite(below) && isfinite(at) && isfinite(above)
            # Continuous across the boundary: the only term that changes is the pole's
            # contribution ∝ exp(-xmax) ~ exp(-100) ~ 4e-44, negligible on the O(1) bulk
            # integral. No NaN from log(xmax - x_pole) at the exact endpoint.
            @test below ≈ at rtol=1e-9
            @test at ≈ above rtol=1e-9
            # Collisionless result equals the ν→0⁺ (small-ν krook) limit.
            small_nu = KF.integrate_energy(
                0.5, 0.3, -sqrt(xmax - 1e-13), 0.0, 1.0, 1e-6, 0, 1.0, 1, 0.5, 0.5, "fgar";
                nutype="krook", f0type="maxwellian", atol=1e-12, rtol=1e-10)
            @test below ≈ small_nu rtol=1e-3
        end

        @testset "pole_offset overflow guard" begin
            # Tiny wb gives small omega_prime = leff·wb/(2√x_res) = 5e-16 at x_res=1, while
            # nuk = 1e300 forces pole_offset = ν/omega_prime = 2e315 → Inf. The
            # `isfinite(pole_offset) || continue` guard in _integrate_energy_resonant must skip
            # this pole. Parameters keep abs(wb) > 1e-30 so find_resonance_energies still returns x_res.
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
        # ψ quadrature is rtol-primary: atol_psi is amplitude-sensitive (NTV ∝ δB²)
        @test ctrl.atol_psi == 0.0
        @test ctrl.rtol_psi == 1e-2
        @test ctrl.maxevals_psi == 2000
    end

    @testset "KineticForcesInternal defaults" begin
        intr = KF.KineticForcesInternal()
        @test intr.ro == 0.0
        @test intr.bo == 0.0
        @test intr.mpert == 0
        @test intr.chi1 == 0.0
        @test isempty(intr.sing_psis)
    end

    @testset "psi_panel_points" begin
        sing_psis = [0.2, 0.5, 0.8]
        # Interior surfaces become panel boundaries, in order
        @test KF.psi_panel_points(sing_psis, 0.1, 0.9) == [0.1, 0.2, 0.5, 0.8, 0.9]
        # Surfaces outside the integration bounds are dropped
        @test KF.psi_panel_points(sing_psis, 0.3, 0.7) == [0.3, 0.5, 0.7]
        # Surfaces at (or within 1e-8 of) a bound would create a degenerate panel — dropped
        @test KF.psi_panel_points(sing_psis, 0.2, 0.8) == [0.2, 0.5, 0.8]
        @test KF.psi_panel_points([0.2 + 5e-9], 0.2, 0.9) == [0.2, 0.9]
        # No surfaces: plain two-point interval
        @test KF.psi_panel_points(Float64[], 0.0, 1.0) == [0.0, 1.0]
        # Unsorted union input (rationals ∪ kinetic resonances) is sorted
        @test KF.psi_panel_points([0.8, 0.2, 0.5], 0.1, 0.9) == [0.1, 0.2, 0.5, 0.8, 0.9]
        # Near-coincident points (kinetic resonance on a rational) collapse to one
        @test KF.psi_panel_points([0.5, 0.5 + 5e-9, 0.2], 0.1, 0.9) == [0.1, 0.2, 0.5, 0.9]
    end

    @testset "find_sign_change_roots" begin
        xs = collect(range(0.0, 1.0; length=101))
        # Two known zeros of a smooth profile
        spl = KF.cubic_interp(xs, @. sin(2π * xs) + 0.5)
        roots = KF.find_sign_change_roots(spl, xs)
        @test length(roots) == 2
        expected = [(π - asin(-0.5)) / (2π), (2π + asin(-0.5)) / (2π)]
        @test isapprox(roots, expected; atol=1e-6)
        # Monotone positive profile: no crossings
        @test isempty(KF.find_sign_change_roots(KF.cubic_interp(xs, 1.0 .+ xs), xs))
        # Exact zero at a node is not a strict sign change — no crash, no root from that pair
        vals = collect(1.0 .- 2 .* xs)
        vals[51] = 0.0  # xs[51] = 0.5 is the true zero
        spl0 = KF.cubic_interp(xs, vals)
        @test length(KF.find_sign_change_roots(spl0, xs)) <= 1
    end

    @testset "kinetic resonance node scan" begin
        # Synthetic frequency closures with analytically known Ω_ℓ(x=1) = 0 locations (evaluated
        # at xeval=1 to keep the closed form): constant ω_b, ω_d and linear ω_E(ψ) = a − b·ψ give
        # ψ_ℓ = (a + ω_d + ℓ·ω_b/n)/b. Constants chosen so no zero falls exactly on a grid node.
        grid = collect(range(0.0, 1.0; length=101))
        wb0, wd0, a, b = 0.1, 0.05, 0.5037, 1.0
        wbhat_f = _ -> wb0
        wdhat_f = _ -> wd0
        welec_f = psi -> a - b * psi
        nodes = sort(KF._resonance_nodes_from_frequencies(wbhat_f, welec_f, wdhat_f, grid; n=1, nl=2, xeval=1.0))
        @test length(nodes) == 5
        @test isapprox(nodes, [0.3537, 0.4537, 0.5537, 0.6537, 0.7537]; atol=1e-10)
        # nl = 0 reduces to the ω_d-shifted ExB resonance alone
        nodes0 = KF._resonance_nodes_from_frequencies(wbhat_f, welec_f, wdhat_f, grid; n=1, nl=0, xeval=1.0)
        @test isapprox(nodes0, [0.5537]; atol=1e-10)
        # No crossings when ω_E never approaches the resonance condition
        @test isempty(KF._resonance_nodes_from_frequencies(wbhat_f, _ -> 10.0, wdhat_f, grid; n=1, nl=2, xeval=1.0))
    end

    @testset "check_psi_quadrature_convergence" begin
        ctrl = KF.KineticForcesControl()  # atol_psi=0, rtol_psi=1e-2
        total = 1.0 + 0.0im
        # Converged: error below rtol*|T|, no warning
        @test_logs KF.check_psi_quadrature_convergence(total, 1e-3, ctrl, "fgar")
        # Hit maxevals: error above tolerance
        @test_logs (:warn, r"maxevals_psi") KF.check_psi_quadrature_convergence(total, 0.5, ctrl, "fgar")
        # Nonzero atol_psi dominating a small torque: the silent-garbage scenario
        ctrl.atol_psi = 1e-2
        @test_logs (:warn, r"atol_psi") KF.check_psi_quadrature_convergence(1e-3 + 0.0im, 1e-3, ctrl, "fgar")
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
        @test mr.total_energy == 0.0
        @test isempty(mr.records)
    end

    # =========================================================================
    # Kinetic profile I/O: HDF5 schema, ASCII back-compat, writer round-trip
    # =========================================================================
    @testset "kinetic profile I/O" begin
        E = GeneralizedPerturbedEquilibrium.Equilibrium
        exdir = joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")
        gpeckf = joinpath(exdir, "TkMkr_D3Dlike_Hmode_kinetic.gpeckf")
        h5 = joinpath(exdir, "TkMkr_D3Dlike_Hmode_kinetic.h5")

        @testset "extension dispatch + raw read" begin
            da = E.read_kinetic_file(gpeckf)
            db = E.read_kinetic_file(h5)
            # legacy ASCII carries no optional fields; the example .h5 carries chi
            @test da.chi_e === nothing && da.chi_phi === nothing
            @test db.chi_e !== nothing && db.chi_phi !== nothing
            # core columns identical (the .h5 was generated from the .gpeckf)
            @test da.psi == db.psi
            @test da.n_e == db.n_e
            @test da.T_i == db.T_i
            @test da.omega_E == db.omega_E
        end

        @testset "NTV splines identical: ASCII vs HDF5" begin
            ka = E.load_kinetic_profiles(gpeckf)
            kb = E.load_kinetic_profiles(h5)
            for p in 0.0:0.1:1.0
                @test ka.ne_spline(p) == kb.ne_spline(p)
                @test ka.Ti_spline(p) == kb.Ti_spline(p)
                @test ka.omegaE_spline(p) == kb.omegaE_spline(p)
                @test ka.zeff_spline(p) == kb.zeff_spline(p)
            end
        end

        @testset "writer round-trip" begin
            db = E.read_kinetic_file(h5)
            tmp = tempname() * ".h5"
            E.write_kinetic_h5(tmp, db; provenance="roundtrip")
            dc = E.read_kinetic_file(tmp)
            @test dc.chi_e ≈ db.chi_e
            @test dc.chi_phi ≈ db.chi_phi
            @test dc.n_e ≈ db.n_e
            @test dc.provenance == "roundtrip"
            rm(tmp; force=true)
        end

        @testset "missing required field errors" begin
            # HDF5 with only psi + n_e: NTV load must error on the missing T_i
            tmp = tempname() * ".h5"
            d = E.KineticProfileData(; psi=[0.0, 0.5, 1.0], n_e=[1e19, 9e18, 8e18])
            E.write_kinetic_h5(tmp, d)
            @test_throws ErrorException E.load_kinetic_profiles(tmp)
            rm(tmp; force=true)
        end
    end
end
