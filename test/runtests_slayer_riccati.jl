@testset "SLAYER Riccati Δ" begin
    using GeneralizedPerturbedEquilibrium.InnerLayer
    using StaticArrays
    using Statistics: median

    # Reach into the SLAYER submodule to test the BC selector helper
    # without exporting it (it's an internal of the Riccati port).
    _SLAYER_MOD = GeneralizedPerturbedEquilibrium.InnerLayer.SLAYER

    # A reference deuterium case in the *large-D_norm* regime.
    # T_e = T_i = 3 keV (vs 1 keV) lifts D_norm² above the iota_e·P_perp/P_tor^(2/3) threshold:
    # D_norm² ∝ T_e² but threshold ∝ T_e^0.5, so D_norm² / threshold ∝ T_e^(3/2). At 3 keV the
    # ratio is ~2.4 (vs ~0.5 at 1 keV), placing the fixture solidly on the large_D side of the
    # branch boundary. All other inputs unchanged.
    function _ref_params_large_D()
        return slayer_parameters(;
            n_e=5.0e19, t_e=3000.0, t_i=3000.0,
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
            Q_e=-1.0, Q_i=0.5, iota_e=2.0 / 3.0,
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
        threshold(p) = p.iota_e * p.P_perp / p.P_tor^(2 / 3)
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

        # Full diamagnetic-band sweep. Δ(ω) is continuous and bounded across
        # the whole band, but in the large-D_norm regime it rotates rapidly
        # through the complex plane near the upper edge (ω ≳ 0.7 here), so a
        # 0.2-spaced grid under-resolves that turn — adjacent samples there are
        # legitimately far apart. We therefore test scale-invariant invariants
        # (no NaN, bounded |Δ|, no isolated discontinuity) rather than an
        # absolute step size, which would only measure sampling resolution and
        # drift with the resistivity model.
        ωs_full = collect(range(-1.5; stop=1.5, length=16))
        Δ_full = [solve_inner(m, p, ω + γ * im).tearing for ω in ωs_full]
        @test all(isfinite.(real.(Δ_full)))
        @test all(isfinite.(imag.(Δ_full)))
        # Bounded magnitude over the full band — no blow-up / pole crossing.
        @test all(0.1 .< abs.(Δ_full) .< 10.0)

        # Discontinuity check on the smooth central window. A solver glitch
        # would show up as one step far larger than the local median; a smooth
        # curve keeps that ratio small regardless of overall scale.
        ωs = collect(range(-1.5; stop=0.5, length=11))
        Δs = [solve_inner(m, p, ω + γ * im).tearing for ω in ωs]
        diffs = abs.(diff(Δs))
        @test maximum(diffs) < 6.0 * median(diffs)

        # Δ is genuinely Q-dependent (not silently constant).
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
        Δ_tight = solve_inner(m, p, Q; reltol=1e-13, abstol=1e-13).tearing
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
        Δ_deeper = solve_inner(m, p, Q; pmin=1e-7).tearing
        @test abs(Δ_default - Δ_deeper) < 0.05 * abs(Δ_default)
    end

    @testset "Resistive layer thickness (del_s)" begin
        p = _ref_params_large_D()

        # riccati_del_s returns the dimensionless δ_s/d_β; must be finite.
        dels_db = riccati_del_s(p)
        @test dels_db isa ComplexF64
        @test isfinite(dels_db)

        lw = slayer_layer_thickness(p)
        @test lw isa LayerWidths
        @test lw.ising == p.ising && lw.m == p.m && lw.n == p.n

        # Meters scaling and the magnitude field are self-consistent.
        @test lw.dels_db == dels_db
        @test lw.delta_s ≈ dels_db * p.d_beta
        @test lw.delta_s_m ≈ abs(dels_db * p.d_beta)
        @test lw.delta_s_m > 0

        # Drift-scale reference is carried through unchanged.
        @test lw.d_beta == p.d_beta

        # The Riccati width should sit within a few orders of magnitude of
        # the β-weighted ion scale it is built from (dels_db is O(1)).
        @test 1e-3 * p.d_beta < lw.delta_s_m < 1e3 * p.d_beta

        # Degenerate integration span (q_start ≤ q_min) returns a NaN
        # sentinel rather than throwing.
        @test isnan(riccati_del_s(p; q_start=1e-6, q_min=1e-5))
    end
end
