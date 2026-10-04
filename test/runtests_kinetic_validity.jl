@testset "Kinetic near-axis validity boundary" begin

    GPE = GeneralizedPerturbedEquilibrium
    KF = GPE.KineticForces
    EQ = GPE.Equilibrium
    PE = GPE.PerturbedEquilibrium

    # Duck-typed stand-in for PlasmaEquilibrium: the boundary scan reads only these fields.
    mock_equil(q, r, R; ro=1.7, b0=2.0) =
        (; profiles=(; q_spline=q), geometry=(; avg_r_spline=r, avg_R_spline=R), ro=ro, params=(; b0=b0))

    # Kinetic profiles carrying a prescribed Ti(ψ) in Joules; only xs and Ti_spline are read.
    function mock_profiles(xs, Ti_eV)
        ones_ = ones(length(xs))
        Ti = Ti_eV .* 1.602176634e-19
        return EQ.KineticProfileSplines(xs, 1e19 .* ones_, 1e19 .* ones_, Ti, Ti,
            zero(ones_), 17.0 .* ones_, ones_, ones_, ones_)
    end

    @testset "kinetic_axis_validity_envelope" begin
        psi_c = 0.04
        @test KF.kinetic_axis_validity_envelope(0.0, psi_c) == 0.0
        @test KF.kinetic_axis_validity_envelope(psi_c, psi_c) == 0.0
        @test KF.kinetic_axis_validity_envelope(2 * psi_c, psi_c) == 1.0
        @test KF.kinetic_axis_validity_envelope(1.0, psi_c) == 1.0
        @test KF.kinetic_axis_validity_envelope(1.5 * psi_c, psi_c) ≈ 0.5 atol = 1e-12

        # Disabled boundary leaves everything unsuppressed.
        @test KF.kinetic_axis_validity_envelope(0.0, 0.0) == 1.0
        @test KF.kinetic_axis_validity_envelope(0.5, -1.0) == 1.0

        # Monotone across the transition.
        band = range(psi_c, 2 * psi_c; length=101)
        vals = [KF.kinetic_axis_validity_envelope(x, psi_c) for x in band]
        @test all(diff(vals) .>= -1e-15)

        # C² at the band ends: the second difference straddling each end vanishes linearly in h.
        # A C¹-but-not-C² join would leave it constant, and a slope kink would blow it up as 1/h.
        d2(x, h) = (KF.kinetic_axis_validity_envelope(x + h, psi_c) - 2 * KF.kinetic_axis_validity_envelope(x, psi_c) +
                    KF.kinetic_axis_validity_envelope(x - h, psi_c)) / h^2
        for edge in (psi_c, 2 * psi_c)
            h = psi_c * 1e-2
            @test abs(d2(edge, h) / d2(edge, h / 2)) ≈ 2 atol = 0.2
        end
    end

    @testset "clear_rational_windows" begin
        R = EQ.RATIONAL_RES_RADIUS

        @test KF.clear_rational_windows(0.04, Float64[]) == 0.04
        @test KF.clear_rational_windows(0.0, [0.05]) == 0.0

        # A rational well outside the band is left alone.
        @test KF.clear_rational_windows(0.04, [0.5]) == 0.04

        # A rational inside [ψ_c - R, 2ψ_c] pushes the boundary just past it.
        @test KF.clear_rational_windows(0.10, [0.1221]) ≈ 0.1221 + R

        # Cascade guard: rationals spaced so each move widens the band into the next one. The move
        # is capped at VALIDITY_CLEAR_MAX_FACTOR × ψ_c, and the orbit-width boundary is kept.
        cascade = [0.15, 0.25, 0.45]
        local psi_capped
        @test_logs (:warn,) match_mode = :any begin
            psi_capped = KF.clear_rational_windows(0.10, cascade)
        end
        @test psi_capped == 0.10
        @test KF.VALIDITY_CLEAR_MAX_FACTOR > 1
    end

    @testset "kinetic_axis_validity_psi is contiguous with the axis" begin
        xs = collect(0.005:0.005:1.0)
        # r = a√ψ, R constant: ε → 0 on axis, so ρ_θ = qρ/ε diverges there and the criterion fires.
        a, R0 = 0.6, 1.7
        r_of(psi) = a * sqrt(psi)
        R_of(_) = R0
        Ti = fill(2000.0, length(xs))          # 2 keV, flat

        # Smooth q: the invalid region is a single band touching the axis.
        q_smooth(psi) = 1.0 + 2.0 * psi^2
        psi_c = KF.kinetic_axis_validity_psi(mock_profiles(xs, Ti), mock_equil(q_smooth, r_of, R_of))
        @test psi_c > 0
        @test psi_c < 0.5

        # Same profile with a narrow q spike far from the axis. The spike re-triggers the criterion,
        # but the model has not lost validity there, so the boundary must not jump out to it.
        # The accompanying @warn carries maxlog=1, so whether it fires here depends on what ran
        # earlier in the suite; the boundary itself is the contract worth asserting.
        q_spiked(psi) = q_smooth(psi) + 400.0 * exp(-((psi - 0.9) / 0.01)^2)
        psi_c_spiked = KF.kinetic_axis_validity_psi(mock_profiles(xs, Ti), mock_equil(q_spiked, r_of, R_of))
        @test psi_c_spiked ≈ psi_c
        @test psi_c_spiked < 0.5
    end

    @testset "kinetic_regularization_kwargs" begin
        default_reg = PE.PerturbedEquilibriumControl().reg_spot
        ideal = (; mats=(; kinetic=nothing))
        kinetic = (; mats=(; kinetic=(;)))

        # Ideal runs keep whatever the deck asked for, explicit or defaulted.
        @test PE.kinetic_regularization_kwargs(ideal, pairs((; reg_spot=0.05))).reg_spot == 0.05
        @test PE.kinetic_regularization_kwargs(ideal, pairs((;))).reg_spot == default_reg

        # Kinetic runs force 0 — including when the deck omits the key entirely, which must pick up
        # the struct default rather than leaving reg_spot unset.
        @test PE.kinetic_regularization_kwargs(kinetic, pairs((; reg_spot=0.05))).reg_spot == 0.0
        @test PE.kinetic_regularization_kwargs(kinetic, pairs((;))).reg_spot == 0.0
        @test default_reg != 0.0   # otherwise the override above proves nothing

        # Unrelated kwargs pass through untouched.
        out = PE.kinetic_regularization_kwargs(kinetic, pairs((; reg_spot=0.05, filter_modes=true)))
        @test out.filter_modes == true
    end
end
