@testset "Dispersion residual (SurfaceCoupling)" begin
    using GeneralizedPerturbedEquilibrium.InnerLayer
    using GeneralizedPerturbedEquilibrium.InnerLayer: InnerLayerModel, solve_inner
    using GeneralizedPerturbedEquilibrium.Dispersion
    using StaticArrays

    # ---------------------------------------------------------------
    # Synthetic linear inner-layer model used to verify the residual
    # arithmetic without ODE noise:
    #   Δ_inner(Q) = a + b·Q
    #   r(Q) = dp_diag - scale·(a + b·Q) - dc
    # ---------------------------------------------------------------
    struct LinearTestModel <: InnerLayerModel
        a::ComplexF64
        b::ComplexF64
    end
    GeneralizedPerturbedEquilibrium.InnerLayer.solve_inner(
        m::LinearTestModel, params, Q::Number) =
        InnerLayerResponse(m.a + m.b * ComplexF64(Q), zero(ComplexF64))

    function _slayer_ref()
        return slayer_parameters(
            n_e=5.0e19, t_e=1000.0, t_i=1000.0,
            omega=0.0, omega_e=1.0e4, omega_i=5.0e3,
            qval=2.0, sval_r=1.0, bt=2.0,
            rs=0.5, R0=1.7, mu_i=2.0, zeff=1.0,
            chi_perp=1.0, chi_tor=1.0, m=2, n=1)
    end

    @testset "Constructor scale defaults" begin
        # SLAYER: scale = lu^(1/3) so the dimensionless Δ from riccati_f
        # is mapped to outer ψ-units (Fortran SLAYER growthrates routine)
        p_sl = _slayer_ref()
        sc_sl = surface_coupling(SLAYERModel(), p_sl, -1.0 + 0.0im)
        @test sc_sl.scale ≈ p_sl.lu^(1/3)
        @test sc_sl.dc == 0.0
        @test sc_sl.dp_diag == ComplexF64(-1.0)

        # GGJ: scale = 1 because rescale_delta is applied inside solve_inner
        p_ggj = glasser_wang_2020_eq55()
        sc_ggj = surface_coupling(GGJModel(solver=:shooting), p_ggj,
            -1.0 + 0.0im)
        @test sc_ggj.scale == 1.0

        # Generic fallback honors explicit scale + dc kwargs
        sc_lin = surface_coupling(LinearTestModel(0.0im, 1.0+0im), nothing,
            3.0 + 0.0im; dc=0.5, scale=2.0)
        @test sc_lin.scale == 2.0
        @test sc_lin.dc == 0.5
    end

    @testset "Residual arithmetic on synthetic linear model" begin
        # r(Q) = dp_diag - scale·(a + b·Q) - dc
        a, b = 1.0 + 2.0im, -0.5 + 1.0im
        scale = 3.0
        dc = 0.25
        Q_root = -0.7 + 0.3im
        dp_diag = (a + b * Q_root) * scale + dc       # construct a known root

        sc = surface_coupling(LinearTestModel(a, b), nothing, dp_diag;
            dc=dc, scale=scale)
        @test sc(Q_root) ≈ 0 atol = 1e-12

        # Off-root residual matches the closed form
        for Q in (0.0+0im, 1.5-0.5im, -0.2+1.2im)
            expected = dp_diag - scale * (a + b * Q) - dc
            @test sc(Q) ≈ expected
        end
    end

    @testset "SLAYER residual: self-consistent zero at known Q" begin
        # Build dp_diag = scale · Δ(Q_pin) so the residual is exactly zero
        # at Q_pin (residual evaluated through the same ODE that produced Δ).
        p = _slayer_ref()
        m = SLAYERModel()
        Q_pin = 0.3 + 0.4im
        Δ_pin = solve_inner(m, p, Q_pin).tearing
        dp_diag = p.lu^(1/3) * Δ_pin

        sc = surface_coupling(m, p, dp_diag)
        @test abs(sc(Q_pin)) < 1e-13       # self-consistent

        # Perturbing Q gives a non-trivial residual
        @test abs(sc(Q_pin + 0.05)) > 1e-3
        @test sc(Q_pin + 0.05) isa ComplexF64
    end

    @testset "Interface compliance: GGJ ↔ SLAYER through abstract dispatch" begin
        # Both inner-layer models flow through the same SurfaceCoupling
        # API. Numerical agreement is *not* asserted (different physics) —
        # only that both pipelines construct and evaluate.
        p_sl = _slayer_ref()
        sc_sl = surface_coupling(SLAYERModel(), p_sl, -100.0 + 0.0im)
        @test sc_sl isa SurfaceCoupling{SLAYERModel{:fitzpatrick},SLAYERParameters}
        @test sc_sl(0.0 + 0.5im) isa ComplexF64

        p_ggj = glasser_wang_2020_eq55()
        sc_ggj = surface_coupling(GGJModel(solver=:shooting), p_ggj,
            -1.0 + 0.0im)
        @test sc_ggj isa SurfaceCoupling{GGJModel{:shooting},GGJParameters}
        @test sc_ggj(1e-3 + 0.0im) isa ComplexF64
    end

    @testset "Residual is callable on grids (broadcast)" begin
        # Brute-force / AMR scans (PR 5/6) will broadcast `sc` over a 2D
        # complex-Q grid; verify that broadcasting works element-wise.
        a, b = 0.0+0im, 1.0+0im
        sc = surface_coupling(LinearTestModel(a, b), nothing, 2.0+0im;
            dc=0.0, scale=1.0)
        Q_grid = [(qr + qi*im) for qr in -1.0:0.5:1.0, qi in -1.0:0.5:1.0]
        Δ_grid = sc.(Q_grid)
        @test size(Δ_grid) == size(Q_grid)
        @test all(d -> d isa ComplexF64, Δ_grid)
        # Closed-form check at one interior grid point
        @test Δ_grid[3, 3] ≈ sc(Q_grid[3, 3])
    end
end
