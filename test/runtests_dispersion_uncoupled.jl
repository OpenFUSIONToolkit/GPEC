@testset "Dispersion uncoupled root-find" begin
    using GeneralizedPerturbedEquilibrium.InnerLayer
    using GeneralizedPerturbedEquilibrium.InnerLayer: InnerLayerModel, solve_inner
    using GeneralizedPerturbedEquilibrium.Dispersion
    using StaticArrays

    # ---------------------------------------------------------------
    # Synthetic linear inner-layer model with an exactly-known root.
    #   Δ_inner(Q) = a + b·Q
    #   r(Q) = dp_diag - scale·(a + b·Q) - dc
    #   ⇒ Q_root = (dp_diag - dc - a·scale) / (b·scale)
    # ---------------------------------------------------------------
    struct LinearTestModel <: InnerLayerModel
        a::ComplexF64
        b::ComplexF64
    end
    GeneralizedPerturbedEquilibrium.InnerLayer.solve_inner(
        m::LinearTestModel, params, Q::Number) =
        SVector{2,ComplexF64}(m.a + m.b * ComplexF64(Q), zero(ComplexF64))

    function _slayer_ref()
        return slayer_parameters(
            n_e=5.0e19, t_e=1000.0, t_i=1000.0,
            omega=0.0, omega_e=1.0e4, omega_i=5.0e3,
            qval=2.0, sval_r=1.0, bt=2.0,
            rs=0.5, R0=1.7, mu_i=2.0, zeff=1.0,
            chi_perp=1.0, chi_tor=1.0, m=2, n=1)
    end

    @testset "SurfaceCoupling constructor scale defaults" begin
        # SLAYER: scale = lu^(1/3)
        p_sl = _slayer_ref()
        sc_sl = surface_coupling(SLAYERModel(), p_sl, -1.0 + 0.0im)
        @test sc_sl.scale ≈ p_sl.lu^(1/3)
        @test sc_sl.dc == 0.0
        @test sc_sl.dp_diag == ComplexF64(-1.0)

        # GGJ: scale = 1.0
        p_ggj = glasser_wang_2020_eq55()
        sc_ggj = surface_coupling(GGJModel(solver=:shooting), p_ggj,
                                  -1.0 + 0.0im)
        @test sc_ggj.scale == 1.0

        # Generic fallback honors explicit scale kwarg
        sc_lin = surface_coupling(LinearTestModel(0.0im, 1.0+0im), nothing,
                                  3.0 + 0.0im; dc=0.5, scale=2.0)
        @test sc_lin.scale == 2.0
        @test sc_lin.dc == 0.5
    end

    @testset "Test 4: Newton finds analytic root (linear synthetic model)" begin
        # Solve r(Q) = dp_diag - (a + b·Q)·scale - dc = 0
        a, b   = 1.0 + 2.0im, -0.5 + 1.0im
        scale  = 3.0
        dc     = 0.25
        Q_true = -0.7 + 0.3im
        dp_diag = (a + b * Q_true) * scale + dc       # ⇒ Q_true is the root

        sc = surface_coupling(LinearTestModel(a, b), nothing, dp_diag;
                              dc=dc, scale=scale)
        @test sc(Q_true) ≈ 0 atol = 1e-12

        # Newton from a perturbed start converges quadratically (no ODE noise
        # for a linear model — the residual is exact).
        for Q0 in (Q_true + 0.5, Q_true - 0.3im, Q_true + 1.0 - 0.5im)
            res = solve_uncoupled(sc, Q0; tol=1e-12, on_failure=:silent)
            @test res.converged
            @test abs(res.Q - Q_true) < 1e-10
            @test abs(res.residual) < 1e-10
            @test res.iterations < 15        # quadratic convergence
        end
    end

    @testset "Test 4b: SLAYER self-consistent root (build a known root)" begin
        # Pick a Q_true, evaluate Δ there, set dp_diag = scale·Δ ⇒ Q_true is
        # the dispersion root by construction.
        p = _slayer_ref()
        m = SLAYERModel()
        Q_true = 0.3 + 0.4im
        Δ_true = solve_inner(m, p, Q_true)[1]
        dp_diag = p.lu^(1/3) * Δ_true

        sc = surface_coupling(m, p, dp_diag)
        # Residual at Q_true is exactly zero (computed from the same ODE)
        @test abs(sc(Q_true)) < 1e-14

        # Newton from a perturbed start recovers Q_true to ODE-noise precision
        res = solve_uncoupled(sc, Q_true + 0.1 - 0.1im; on_failure=:silent)
        @test res.converged
        @test abs(res.Q - Q_true) < 1e-3       # ODE noise floor ~1e-3·|Δ|
    end

    @testset "Test 8: GGJ ↔ SLAYER interchangeability" begin
        # Both inner-layer models must flow through the same SurfaceCoupling
        # API. Numerical agreement between models is *not* asserted —
        # different physics, different parameter spaces. Only the API
        # contract (constructor type-dispatch + callable residual) is
        # exercised here. SLAYER additionally drives solve_uncoupled to
        # confirm the Newton path works through the abstract interface.
        p_sl  = _slayer_ref()
        sc_sl = surface_coupling(SLAYERModel(), p_sl, -100.0 + 0.0im)
        @test sc_sl isa SurfaceCoupling{SLAYERModel{:fitzpatrick},SLAYERParameters}
        @test sc_sl(0.0 + 0.5im) isa ComplexF64

        p_ggj  = glasser_wang_2020_eq55()
        sc_ggj = surface_coupling(GGJModel(solver=:shooting), p_ggj,
                                   -1.0 + 0.0im)
        @test sc_ggj isa SurfaceCoupling{GGJModel{:shooting},GGJParameters}
        @test sc_ggj(1e-3 + 0.0im) isa ComplexF64

        # SLAYER drives solve_uncoupled successfully through the abstract
        # interface (both models share the same dispatch path).
        res_sl = solve_uncoupled(sc_sl, 0.3 + 0.4im; on_failure=:silent)
        @test res_sl isa NewtonResult
    end

    @testset "Vector dispatch (multi-surface)" begin
        a, b, scale, dc = 1.0+0im, 1.0+0im, 1.0, 0.0
        Q_trues = [0.5+0.1im, -0.3-0.2im, 1.2+0.4im]
        scs = [surface_coupling(LinearTestModel(a, b), nothing,
                                (a + b*Q)*scale + dc; dc=dc, scale=scale)
               for Q in Q_trues]

        # Scalar Q0 broadcast to all surfaces
        results = solve_uncoupled(scs, 0.0 + 0.0im; tol=1e-12,
                                  on_failure=:silent)
        @test length(results) == length(scs)
        for (r, Qt) in zip(results, Q_trues)
            @test r.converged
            @test abs(r.Q - Qt) < 1e-10
        end

        # Per-surface Q0 vector
        results = solve_uncoupled(scs, Q_trues .+ 0.05; tol=1e-12,
                                  on_failure=:silent)
        for (r, Qt) in zip(results, Q_trues)
            @test r.converged
            @test abs(r.Q - Qt) < 1e-10
        end

        # Length mismatch is rejected
        @test_throws ArgumentError solve_uncoupled(scs, [0.0+0im, 0.0+0im];
                                                    on_failure=:silent)
    end

    @testset "on_failure modes" begin
        # Construct a residual whose root is far from Q0 with maxiter=2 so
        # Newton has no chance to converge — exercises the failure handlers.
        sc = surface_coupling(LinearTestModel(1.0+0im, 0.0+0im), nothing,
                              1e6 + 0.0im; dc=0.0, scale=1.0)
        # Δ_inner is constant a=1.0, df=0 ⇒ derivative-zero error path
        @test_throws Exception solve_uncoupled(sc, 0.0+0im; on_failure=:silent)

        # Linear model with non-zero slope but maxiter=1, Q0 far from root
        sc2 = surface_coupling(LinearTestModel(0.0+0im, 1.0+0im), nothing,
                               100.0 + 0.0im; dc=0.0, scale=1.0)
        # tight tol with only 1 iteration ⇒ won't converge in one Newton step
        # from this distance; use :silent so warning doesn't clutter logs
        @test_throws ErrorException solve_uncoupled(sc2, 0.0+0im;
                                                     tol=1e-15, maxiter=1,
                                                     on_failure=:error)

        r = solve_uncoupled(sc2, 0.0+0im; tol=1e-15, maxiter=1,
                            on_failure=:silent)
        @test r isa NewtonResult        # silent path returns the un-converged result
    end
end
