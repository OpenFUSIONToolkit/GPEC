# runtests_islands_operators.jl
#
# Islands operator-stack skeleton — verification ladder A1/A2 (docs/src/islands/design/05).
#   A1  MMS: per-operator and assembled-system convergence at design order.
#   A2  JVP (forward-mode AD) vs. finite-difference residual directional derivative.
#   +   allocation regression for the `apply!` / `residual!` hot paths (docs/04 §9).
#
# These are STRUCTURAL (pre-physics) checks. The manufactured coefficients they
# use are arbitrary order-unity test values — not physics — so nothing here is
# [VERIFY]-gated. Physics benchmarks (ladder B+) stay skipped until human-cleared.

const Isl = GeneralizedPerturbedEquilibrium.Islands
const V = Isl.Verify
const PSg = Isl.PhaseSpace
const Op = Isl.Operators

# nx = ny refined together; ξ bandlimited (nxi small is exact); nE small.
_grid(n) = PSg.IslandGrid(; nx=n, nxi=8, ny=n, nE=3, halfwidth_x=6.0, clustering_x=1.0,
    y_max=4.0, y_c=1.0, clustering_y=0.8, order=4)
_order(e_coarse, e_fine, n_coarse, n_fine) = log(e_coarse / e_fine) / log(n_fine / n_coarse)

@testset "Islands operator stack (A1/A2)" begin

    @testset "A1 — per-operator MMS convergence at design order" begin
        # x/y differential operators: fourth-order finite differences.
        for term in (:streaming, :exb, :collisions, :perp)
            e17 = V.mms_operator_error(_grid(17), term)
            e33 = V.mms_operator_error(_grid(33), term)
            @test _order(e17, e33, 17, 33) > 3.3
            @test e33 < e17
        end
        # ξ-only operator (magnetic drift): Fourier spectral → machine precision
        # on the bandlimited manufactured ξ-profile.
        @test V.mms_operator_error(_grid(17), :drift) < 1e-10
        # state-independent source is reproduced exactly (no discretization).
        @test V.mms_operator_error(_grid(17), :drive) < 1e-12
    end

    @testset "A1 — assembled kinetic-residual MMS convergence" begin
        e17 = V.mms_assembled_error(_grid(17))
        e33 = V.mms_assembled_error(_grid(33))
        @test _order(e17, e33, 17, 33) > 3.3
        @test e33 < 5e-3
    end

    @testset "A1 — velocity-moment quadrature convergence (refine y)" begin
        mky(ny) = PSg.IslandGrid(; nx=9, nxi=8, ny=ny, nE=3, halfwidth_x=6.0, clustering_x=1.0,
            y_max=4.0, y_c=1.0, clustering_y=0.8, order=4)
        d1 = V.moment_selfconvergence(mky(17), mky(33))
        d2 = V.moment_selfconvergence(mky(33), mky(65))
        @test _order(d1, d2, 33, 65) > 3.3
        # grids must share (nx, nξ)
        @test_throws ArgumentError V.moment_selfconvergence(mky(17), _grid(17))
    end

    @testset "A2 — AD JVP matches finite-difference directional derivative" begin
        # residual is nonlinear (E×B bracket couples g and Φ), so this exercises
        # the AD plumbing and its correctness together.
        @test V.jvp_fd_maxerror(_grid(9)) < 1e-6
        @test V.jvp_fd_maxerror(_grid(9); seed=7) < 1e-6
    end

    @testset "allocation regression — apply!/residual! are allocation-free" begin
        g = _grid(9)
        for term in (:streaming, :drift, :exb, :collisions, :perp, :drive)
            @test V.term_allocations(g, term) == 0
        end
        @test V.residual_allocations(g) == 0
    end

    @testset "structural — state, stack, flatten/unflatten" begin
        g = _grid(9)
        U, = V.manufactured_state(g)
        @test eltype(U) == Float64
        @test size(U.g) == PSg.nnodes(g)
        @test size(U.Φ) == (9, 8)

        # flatten/unflatten round-trips
        n = Op.statelength(g)
        @test n == prod(PSg.nnodes(g)) + 9 * 8
        v = zeros(n)
        Op.flatten!(v, U)
        U2 = Op.IslandState(g)
        Op.unflatten!(U2, v)
        @test U2.g == U.g && U2.Φ == U.Φ

        # residual! runs and produces a finite, correctly-shaped result
        stack = V.build_stack(g)
        R = Op.IslandState(g)
        cache = Op.IslandCache(g)
        Op.residual!(R, U, stack, g, cache)
        @test all(isfinite, R.g) && all(isfinite, R.Φ)

        # the magnetic-drift variant toggle is carried on the term
        @test Op.MagneticDrift(zeros(1, 1, 1, 1, 1); variant=:improved).variant == :improved
    end
end
