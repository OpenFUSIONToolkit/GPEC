@testset "SLAYER Riccati Δ" begin
    using GeneralizedPerturbedEquilibrium.InnerLayer
    using StaticArrays

    # Reach into the SLAYER submodule to test the BC selector helper
    # without exporting it (it's an internal of the Riccati port).
    _SLAYER_MOD = GeneralizedPerturbedEquilibrium.InnerLayer.SLAYER

    # A reference deuterium case in the *large-D_norm* regime
    function _ref_params_large_D()
        return slayer_parameters(
            n_e=5.0e19, t_e=1000.0, t_i=1000.0,
            omega=0.0, omega_e=1.0e4, omega_i=5.0e3,
            qval=2.0, sval_r=1.0, bt=2.0,
            rs=0.5, R0=1.7, mu_i=2.0, zeff=1.0,
            chi_perp=1.0, chi_tor=1.0,
            m=2, n=1)
    end

    # A directly-built parameter set in the *small-D_norm* regime
    function _ref_params_small_D()
        return SLAYERParameters(;
            tau=1.0, lu=1.0e7, c_beta=0.05, D_norm=0.05,
            P_perp=20.0, P_tor=10.0,
            Q_e=-1.0, Q_i=0.5, iota_e=2.0/3.0,
            tauk=1.0e-4, tau_r=10.0, delta_n=400.0,
            rs=0.5, R0=1.7, bt=2.0, sval_r=1.0,
            eta=2.5e-8, d_beta=2.0e-4)
    end

    @testset "Interface compliance" begin
        p = _ref_params_large_D()
        Δ = solve_inner(SLAYERModel(), p, 0.5 + 0.2im)
        @test Δ isa InnerLayerResponse
        @test Δ.interchange == zero(ComplexF64)    # pressureless SLAYER has no interchange channel
        @test isfinite(real(Δ.tearing))
        @test isfinite(imag(Δ.tearing))
    end

    @testset "Boundary-condition branch selection" begin
        p_large = _ref_params_large_D()
        p_small = _ref_params_small_D()

        # Sanity-check the regime ordering used by _riccati_f_initial:
        # Branch 1 (large_D) iff D_norm² > iota_e·P_perp/P_tor^(2/3).
        threshold(p) = p.iota_e * p.P_perp / p.P_tor^(2/3)
        @test p_large.D_norm^2 > threshold(p_large)
        @test p_small.D_norm^2 < threshold(p_small)

        _, _, branch_large = _SLAYER_MOD._riccati_f_initial(p_large, 0.5 + 0.0im)
        _, _, branch_small = _SLAYER_MOD._riccati_f_initial(p_small, 0.5 + 0.0im)
        @test branch_large === :large_D
        @test branch_small === :small_D

        # Both branches should yield finite Δ values
        Δl = solve_inner(SLAYERModel(), p_large, 0.5 + 0.1im)
        Δs = solve_inner(SLAYERModel(), p_small, 0.5 + 0.1im)
        @test isfinite(Δl.tearing) && isfinite(Δs.tearing)

        # p_floor (=6 by default) is honored even when the branch
        # formula would produce a smaller value.
        p_start_default, _, _ = _SLAYER_MOD._riccati_f_initial(p_small, 0.5 + 0.0im)
        @test p_start_default >= 6.0
        # …and bumping the floor up bumps p_start up.
        p_start_high, _, _ = _SLAYER_MOD._riccati_f_initial(p_small, 0.5 + 0.0im;
                                                             p_floor=12.0)
        @test p_start_high >= 12.0
    end

    @testset "Smoothness across Q sweep" begin
        p = _ref_params_large_D()
        m = SLAYERModel()
        γ = 0.2
        ωs = collect(range(-2.0; stop=2.0, length=21))
        Δs = [solve_inner(m, p, ω + γ*im).tearing for ω in ωs]
        @test all(isfinite.(real.(Δs)))
        @test all(isfinite.(imag.(Δs)))

        # Adjacent Δ values must be close to each other (smoothness).
        # The largest step on this 0.2-spaced sweep stays well under 1.
        diffs = abs.(diff(Δs))
        @test maximum(diffs) < 1.0

        # Δ is genuinely Q-dependent (sanity check that we are not
        # silently returning a constant)
        @test maximum(diffs) > 1e-6
    end

    @testset "Tolerance self-consistency" begin
        p = _ref_params_large_D()
        m = SLAYERModel()
        Q = 0.5 + 0.2im
        # The default reltol=1e-10 matches the Fortran SLAYER LSODE
        # setting. Tightening to 1e-13 typically agrees to ~4 digits;
        # the long inward integration span amplifies local tolerances
        # by roughly 5 orders of magnitude, so 1e-3 relative is the
        # realistic self-consistency threshold here.
        Δ_default = solve_inner(m, p, Q).tearing
        Δ_tight   = solve_inner(m, p, Q; reltol=1e-13, abstol=1e-13).tearing
        @test abs(Δ_default - Δ_tight) < 1e-3 * abs(Δ_tight)
    end

    @testset "p_min reduction stability" begin
        # Pulling p_min closer to 0 (from the default 1e-6 down to 1e-7)
        # changes Δ only marginally — the solution has well-developed
        # asymptotic structure deep in the inner layer.
        p = _ref_params_large_D()
        m = SLAYERModel()
        Q = 0.5 + 0.2im
        Δ_default = solve_inner(m, p, Q; pmin=1e-6).tearing
        Δ_deeper  = solve_inner(m, p, Q; pmin=1e-7).tearing
        @test abs(Δ_default - Δ_deeper) < 0.05 * abs(Δ_default)
    end
end
