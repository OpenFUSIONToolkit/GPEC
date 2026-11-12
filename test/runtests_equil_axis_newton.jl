@testset "Direct equilibrium: magnetic-axis search robustness" begin
    using GeneralizedPerturbedEquilibrium.Equilibrium
    using GeneralizedPerturbedEquilibrium.Equilibrium: EquilibriumConfig, SolovevConfig, DirectRunInput, sol_run, direct_position!
    using FastInterpolations
    EQ = GeneralizedPerturbedEquilibrium.Equilibrium

    fallback = r"restarted from a first-derivative bisection seed"
    # Relative Newton step tolerance of the axis search, reused as the bound on the residual |∇ψ|.
    newton_tol = 1e-12

    # A `DirectRunInput` carrying the cubic spline of `psi_fn(R, Z)` on the grid `rs × zs`, with `psio` as the declared axis flux.
    function direct_input(psi_fn, rs, zs, psio)
        nr = length(rs) - 1
        sol = SolovevConfig(; mr=nr, mz=nr)
        cfg = EquilibriumConfig(; eq_type="sol")
        psi_in = cubic_interp((rs, zs), [psi_fn(R, Z) for R in rs, Z in zs]; extrap=ExtendExtrap())
        return DirectRunInput(cfg, sol_run(cfg, sol).sq_in, psi_in, rs, zs, rs[1], rs[end], zs[1], zs[end], psio, 1, 1, nothing)
    end

    # Up-down symmetric Solovev ψ whose axis sits off the grid's R-midpoint and off every node, plus a
    # conical cusp -A·ρ·exp(-(ρ/w)⁴) at the axis: the cells there carry a curvature spike, and the
    # cusp strength A is set so that a Newton step overshoots the axis by `ncell` grid cells.
    # `r_in`, `r_out` set the grid's extent about r0 in units of a; with r_in > r_out the grid's R-midpoint lies inboard
    # of the axis, so the midplane march takes a step and brackets the axis in R. `z_shift` moves the grid, and with it
    # the march's midplane, in Z in units of a.
    function cusp_solovev(; ncell, nr=96, r_in=1.4, r_out=1.6, z_shift=0.0)
        (; elongation, a, r0, q0, b0fac) = SolovevConfig()
        e = elongation
        psio = e * r0 * b0fac * a^2 / (2 * q0 * r0)
        psifac = psio / (a * r0)^2
        rs = collect(range(r0 - r_in * a, r0 + r_out * a; length=nr + 1))
        zh = collect(range(0.0, 1.5e * a; length=nr ÷ 2 + 1))
        zs = z_shift * a .+ vcat(-reverse(zh[2:end]), zh)
        h = rs[2] - rs[1]
        A = ncell * 2 * psifac * r0^2 * h
        cusp(ρ) = A * ρ * exp(-(ρ / (0.5a))^4)
        psi(R, Z) = psio - psifac * ((R * Z)^2 / e^2 + (R^2 - r0^2)^2 / 4) - cusp(hypot(R - r0, Z))
        return direct_input(psi, rs, zs, psio), r0, h
    end

    # Quadratic ψ = ψ₀ − c (x² + 2b·x·y + y²/e²) about (r0, 0), exact on the cubic spline; `sgn = -1` turns the
    # axis into a minimum of ψ. With b·e near 1 the Hessian is strongly tilted (ψ_RZ ≠ 0), where the Gauss–Seidel
    # bisection sweep contracts by (b e)² per pass. `psio_declared` is the ψ₀ the input claims, against the maximum of 1.
    function quadratic_psi(; b, sgn=1.0, nr=64, psio_declared=1.0)
        (; elongation, a, r0) = SolovevConfig()
        e = elongation
        c = 1.0 / a^2
        rs = collect(range(r0 - 1.5a, r0 + 1.5a; length=nr + 1))
        zs = collect(range(-1.5e * a, 1.5e * a; length=nr + 1))
        psi(R, Z) = 1.0 - sgn * c * ((R - r0)^2 + 2b * (R - r0) * Z + Z^2 / e^2)
        return direct_input(psi, rs, zs, psio_declared), r0, rs[2] - rs[1], e
    end

    # The returned axis must be an O-point of the ψ spline with a vanishing gradient.
    function test_axis_is_o_point(rp, ro, zo)
        d(i, j) = rp.psi_in((ro, zo); deriv=DerivOp(i, j))
        @test d(2, 0) * d(0, 2) - d(1, 1)^2 > 0
        @test hypot(d(1, 0), d(0, 1)) < newton_tol * rp.psio / (rp.rmax - rp.rmin)
    end

    # Newton on the cusp is chaotic: whether it converges, cycles or leaves the grid at a given cusp strength depends
    # on the machine's rounding, so which runs take the restart is not deterministic; that some do over this many is.
    # Whatever it does, the search must return the O-point at the cusp or fail loudly, and the smooth ψ (ncell = 0)
    # must take the plain Newton path on every grid.
    @testset "curvature spike at the axis: a verified axis or a loud failure" begin
        grids = ((;), (; r_in=1.6, r_out=1.4), (; r_in=1.7, r_out=1.3, z_shift=0.7))
        # On the shifted grid the march's midplane lies off the axis's, where the Solovev B_z changes sign at
        # R² = r0² - 2Z²/e², inboard of the axis: the march stops short and the axis lies outside its last step.
        (; elongation) = SolovevConfig()
        rp, r0, _ = cusp_solovev(; ncell=0.0, grids[3]...)
        r_mid, z_mid, dr = (rp.rmin + rp.rmax) / 2, (rp.zmin + rp.zmax) / 2, (rp.rmax - rp.rmin) / 20
        @test r_mid < sqrt(r0^2 - 2 * z_mid^2 / elongation^2) < r_mid + dr < r0

        n_fallback = 0
        for grid in grids, ncell in 0.0:0.25:3.0
            rp, r0, h = cusp_solovev(; ncell, grid...)
            logger = Test.TestLogger(; min_level=Base.CoreLogging.Warn)
            outcome = try
                Base.CoreLogging.with_logger(() -> direct_position!(rp), logger)
            catch err
                err
            end
            if outcome isa Exception
                @test ncell > 0
                @test occursin("Failed to find magnetic axis", sprint(showerror, outcome))
            else
                ro, zo = outcome[1], outcome[2]
                test_axis_is_o_point(rp, ro, zo)
                @test hypot(ro - r0, zo) < h
                fell_back = any(occursin(fallback, log.message) for log in logger.logs)
                n_fallback += fell_back
                if ncell == 0
                    @test !fell_back
                    get(grid, :z_shift, 0.0) == 0.0 && @test abs(zo) <= eps(ro)
                end
            end
        end
        @test n_fallback > 0
    end

    @testset "Newton converging off the ψ grid is an error" begin
        # A smooth ψ on a grid that lies wholly outboard of the axis: Newton follows the spline's extrapolation inboard.
        rp, r0, _ = cusp_solovev(; ncell=0.0, r_in=-0.2, r_out=1.6)
        @test rp.rmin > r0
        @test_throws r"outside the ψ grid" direct_position!(rp)
    end

    @testset "axis checks accept the main maximum of ψ and reject a minimum or a secondary maximum" begin
        rp, r0, _, _ = quadratic_psi(; b=0.0)
        bf = EQ.DirectBField()
        @test EQ._axis_failure(rp, bf, deriv1(rp.sq_in), r0, 0.0; bound_residual=true) === nothing
        rp_min, _, _, _ = quadratic_psi(; b=0.0, sgn=-1.0)
        msg = EQ._axis_failure(rp_min, bf, deriv1(rp_min.sq_in), r0, 0.0; bound_residual=false)
        @test msg isa String && occursin("not the maximum of ψ", msg)
        rp_low, _, _, _ = quadratic_psi(; b=0.0, psio_declared=10.0)
        msg = EQ._axis_failure(rp_low, bf, deriv1(rp_low.sq_in), r0, 0.0; bound_residual=false)
        @test msg isa String && occursin("not the main axis", msg)
    end

    @testset "restart seed and capped Newton recover a tilted axis" begin
        rp, r0, h, e = quadratic_psi(; b=0.9 / 1.6)
        bf, sqd = EQ.DirectBField(), deriv1(rp.sq_in)
        w = (rp.rmax - rp.rmin) / 20
        # Start ten cells away in both directions; a few passes at (b e)² = 0.81 leave it several cells off, so a
        # sweep allowed one pass must report that it did not settle.
        @test occursin("did not settle", EQ._axis_bisection_seed(rp, bf, sqd, r0 + 10h, 10h, w; passes=1)[3])
        rb, zb, failure = EQ._axis_bisection_seed(rp, bf, sqd, r0 + 10h, 10h, w)
        @test failure === nothing
        @test hypot(rb - r0, zb) < h
        ro, zo, failure = EQ._axis_newton(rp, bf, sqd, rb, zb, w)
        @test failure === nothing
        @test abs(ro - r0) < 1e-10 && abs(zo) < 1e-10
    end
end
