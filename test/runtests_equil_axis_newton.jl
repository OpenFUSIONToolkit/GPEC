@testset "Direct equilibrium: magnetic-axis search robustness" begin
    using GeneralizedPerturbedEquilibrium.Equilibrium
    using GeneralizedPerturbedEquilibrium.Equilibrium: EquilibriumConfig, SolovevConfig, DirectRunInput, sol_run, read_efit, direct_position!
    using FastInterpolations

    fallback = r"restarting Newton from a first-derivative bisection"
    # Relative Newton step tolerance of the axis search, reused as the bound on the residual |∇ψ|.
    newton_tol = 1e-12

    # Up-down symmetric Solovev ψ whose axis sits off the grid's R-midpoint and off every node, plus a
    # conical cusp -A·ρ·exp(-(ρ/w)⁴) at the axis: the cells there carry a curvature spike, and the
    # cusp strength A is set so that a Newton step overshoots the axis by `ncell` grid cells.
    function cusp_solovev(; ncell, nr=96)
        sol = SolovevConfig(; mr=nr, mz=nr)
        cfg = EquilibriumConfig(; eq_type="sol")
        (; e, a, r0, q0, b0fac) = sol
        psio = e * r0 * b0fac * a^2 / (2 * q0 * r0)
        psifac = psio / (a * r0)^2
        rs = collect(range(r0 - 1.4a, r0 + 1.6a; length=nr + 1))
        zh = collect(range(0.0, 1.5e * a; length=nr ÷ 2 + 1))
        zs = vcat(-reverse(zh[2:end]), zh)
        h = rs[2] - rs[1]
        A = ncell * 2 * psifac * r0^2 * h
        cusp(ρ) = A * ρ * exp(-(ρ / (0.5a))^4)
        psi = [psio - psifac * ((R * Z)^2 / e^2 + (R^2 - r0^2)^2 / 4) - cusp(hypot(R - r0, Z)) for R in rs, Z in zs]
        psi_in = cubic_interp((rs, zs), psi; extrap=ExtendExtrap())
        rp = DirectRunInput(cfg, sol_run(cfg, sol).sq_in, psi_in, rs, zs, rs[1], rs[end], zs[1], zs[end], psio, 1, 1, nothing)
        return rp, r0, h
    end

    # The returned axis must be an O-point of the ψ spline with a vanishing gradient.
    function test_axis_is_o_point(rp, ro, zo)
        d(i, j) = rp.psi_in((ro, zo); deriv=DerivOp(i, j))
        @test d(2, 0) * d(0, 2) - d(1, 1)^2 > 0
        @test hypot(d(1, 0), d(0, 1)) < newton_tol * rp.psio / (rp.rmax - rp.rmin)
    end

    @testset "curvature spike at the axis takes the fallback" begin
        rp, r0, h = cusp_solovev(; ncell=1.75)
        ro, zo, _, _ = @test_logs (:warn, fallback) match_mode = :any direct_position!(rp)
        test_axis_is_o_point(rp, ro, zo)
        @test abs(zo) <= eps(ro)
        @test abs(ro - r0) < h
    end

    @testset "smooth ψ takes the plain Newton path" begin
        rp, r0, h = cusp_solovev(; ncell=0.0)
        ro, zo, _, _ = @test_logs min_level = Base.CoreLogging.Warn direct_position!(rp)
        test_axis_is_o_point(rp, ro, zo)
        @test abs(zo) <= eps(ro)
        @test abs(ro - r0) < h
    end

    @testset "well-behaved geqdsk takes the plain Newton path" begin
        cfg = EquilibriumConfig(; eq_filename=joinpath(@__DIR__, "test_data", "CHEASE_test_data", "EQDSK_COCOS_02"), eq_type="efit")
        rp = read_efit(cfg)
        ro, zo, _, _ = @test_logs min_level = Base.CoreLogging.Warn direct_position!(rp)
        test_axis_is_o_point(rp, ro, zo)
        @test 6.5 < ro < 7.5
    end
end
