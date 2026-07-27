"""
    Verify

The Islands verification harness (`04 §4`, `05 §A`): manufactured-solution (MMS)
convergence checks and AD-vs-finite-difference JVP checks for the operator
stack. Callable from the test suite (`test/runtests_islands_*.jl`) and from
scripts.

**These are structural (pre-physics) checks — ladder A1/A2.** They exercise the
*discretization order* and the *AD plumbing* using arbitrary, order-unity
manufactured coefficients. That is deliberate and policy-clean: nothing here is
a physics coefficient, so nothing here carries a `[VERIFY]` tag. Physics
benchmarks (ladder B+) live in `benchmarks/islands/` and stay `[VERIFY]`-gated
until human-cleared.

**The MMS radial `x` here is an ABSTRACT discretization-test coordinate, not the
physical `x = (ψ−ψ_s)/ψ_s`.** A manufactured-solution convergence test needs the
box several times the manufactured feature width so the solution is both
well-resolved and boundary-decayed and the discrete system is well-conditioned; the
`halfwidth_x = 6` boxes below are that math-test domain, *not* a physical grid `6×`
the plasma. (Shrinking them to a physical `|x| < 1` steepens the feature and makes
the assembled solve-level MMS ill-conditioned/unsolvable — verified.) All *physics*
grids use physical local domains (`|x| ≲ 0.3`); see design 10 and the
`resolved_island_grid` physical-domain caution.

Design orders checked:

  - `ξ` derivatives — Fourier spectral (a bandlimited manufactured `ξ`-profile is
    differentiated to machine precision).
  - `x`, `y` derivatives — high-order finite differences at the grid `order`
    (default 4): halving the mesh cuts the error by `2^order`.
  - assembled kinetic residual — the min of the above (algebraic, set by `x`/`y`).
"""
module Verify

using LinearAlgebra
import ForwardDiff
import ..PhaseSpace: IslandGrid, MappedFDGrid, GaussGrid, nnodes
import ..Operators: IslandState, IslandCache, IslandStack, residual!, velocity_moment!,
    apply!, statelength, flatten!, unflatten!, g_flat_index,
    ParallelStreaming, MagneticDrift, ExBDrift, Collisions, GradientDrive,
    PerpTransport, RadiationSink, Quasineutrality, FarFieldConditions
import ..Solvers: flat_residual, newton_krylov

export manufactured_state,
    test_coefficients, build_stack,
    mms_operator_error, mms_assembled_error, estimate_order,
    jvp_fd_maxerror, moment_selfconvergence,
    term_allocations, residual_allocations,
    yc_block_sigma_min, solve_mms, zero_drive_setup
export ladder_status, write_state_dashboard
export check_anchor_sync

# ---------------------------------------------------------------------------
# Manufactured solution: smooth, separable, ξ-periodic and bandlimited.
# Each factor's analytic first/second derivatives are known in closed form,
# so the continuous value of every operator is exact at the nodes.
# ---------------------------------------------------------------------------
_Gx(x) = exp(-x^2 / 2)
_Gx′(x) = -x * _Gx(x)
_Gx″(x) = (x^2 - 1) * _Gx(x)
_Gξ(ξ) = sin(ξ) + 0.5 * cos(2ξ)
_Gξ′(ξ) = cos(ξ) - sin(2ξ)
_Gy(y) = exp(-(y - 1)^2 / 2)
_Gy′(y) = -(y - 1) * _Gy(y)
_Gy″(y) = ((y - 1)^2 - 1) * _Gy(y)
_GE(E) = exp(-0.3 * E)
_Gσ(σ) = 1 + 0.25 * σ

_Px(x) = exp(-x^2 / 4)
_Px′(x) = -(x / 2) * _Px(x)
_Pξ(ξ) = cos(ξ)
_Pξ′(ξ) = -sin(ξ)

# Fill a 5D array over (x, ξ, y, E, σ) from a scalar function of the node coords.
function _fill5(grid::IslandGrid, f)
    nx, nξ, ny, nE, nσ = nnodes(grid)
    A = Array{Float64}(undef, nx, nξ, ny, nE, nσ)
    @inbounds for iσ in 1:nσ, iE in 1:nE, iy in 1:ny, iξ in 1:nξ, ix in 1:nx
        A[ix, iξ, iy, iE, iσ] = f(grid.x.nodes[ix], grid.ξ.nodes[iξ], grid.y.nodes[iy],
            grid.E.nodes[iE], grid.σ[iσ])
    end
    return A
end

function _fill2(grid::IslandGrid, f)
    nx, nξ, = nnodes(grid)
    A = Array{Float64}(undef, nx, nξ)
    @inbounds for iξ in 1:nξ, ix in 1:nx
        A[ix, iξ] = f(grid.x.nodes[ix], grid.ξ.nodes[iξ])
    end
    return A
end

"""
    manufactured_state(grid)

Return `(U, deriv)` where `U::IslandState` holds the manufactured `g*`, `Φ*` on
`grid` and `deriv` is a NamedTuple of the analytic partial-derivative arrays
(`dgdx`, `dgdξ`, `dgdy`, `d2gdy2`, `d2gdx2`, `dΦdx`, `dΦdξ`) used to form exact
continuous operator values.
"""
function manufactured_state(grid::IslandGrid)
    g = _fill5(grid, (x, ξ, y, E, σ) -> _Gx(x) * _Gξ(ξ) * _Gy(y) * _GE(E) * _Gσ(σ))
    Φ = _fill2(grid, (x, ξ) -> _Px(x) * _Pξ(ξ))
    U = IslandState(g, Φ)
    deriv = (
        dgdx=_fill5(grid, (x, ξ, y, E, σ) -> _Gx′(x) * _Gξ(ξ) * _Gy(y) * _GE(E) * _Gσ(σ)),
        dgdξ=_fill5(grid, (x, ξ, y, E, σ) -> _Gx(x) * _Gξ′(ξ) * _Gy(y) * _GE(E) * _Gσ(σ)),
        dgdy=_fill5(grid, (x, ξ, y, E, σ) -> _Gx(x) * _Gξ(ξ) * _Gy′(y) * _GE(E) * _Gσ(σ)),
        d2gdy2=_fill5(grid, (x, ξ, y, E, σ) -> _Gx(x) * _Gξ(ξ) * _Gy″(y) * _GE(E) * _Gσ(σ)),
        d2gdx2=_fill5(grid, (x, ξ, y, E, σ) -> _Gx″(x) * _Gξ(ξ) * _Gy(y) * _GE(E) * _Gσ(σ)),
        dΦdx=_fill2(grid, (x, ξ) -> _Px′(x) * _Pξ(ξ)),
        dΦdξ=_fill2(grid, (x, ξ) -> _Px(x) * _Pξ′(ξ))
    )
    return U, deriv
end

# ---------------------------------------------------------------------------
# Test coefficients: arbitrary, smooth, order-unity. NOT physics — they exercise
# the discretization. `a_y > 0` keeps the pitch operator a sensible diffusion.
# ---------------------------------------------------------------------------
"""
    test_coefficients(grid)

Build a NamedTuple of arbitrary smooth manufactured coefficient arrays/scalars
for every term (`a_xi`, `a_x`, `c_D`, `a_y`, `b_y`, `drive`, `c_E`, `χ`, `α`).
These are MMS test values, not physics coefficients.
"""
function test_coefficients(grid::IslandGrid)
    return (
        a_xi=_fill5(grid, (x, ξ, y, E, σ) -> 1.0 + 0.2 * cos(ξ) + 0.1 * x),
        a_x=_fill5(grid, (x, ξ, y, E, σ) -> 0.8 + 0.1 * sin(2ξ) + 0.05 * y),
        c_D=_fill5(grid, (x, ξ, y, E, σ) -> 0.5 * σ * (1 + 0.1 * E)),
        a_y=_fill5(grid, (x, ξ, y, E, σ) -> 0.3 + 0.05 * x^2),
        b_y=_fill5(grid, (x, ξ, y, E, σ) -> 0.1 * (y - 1)),
        drive=_fill5(grid, (x, ξ, y, E, σ) -> 0.2 * exp(-x^2) * cos(ξ)),
        c_E=0.7,
        χ=0.15,
        α=1.3
    )
end

# ---------------------------------------------------------------------------
# Continuous (exact) operator values from the analytic manufactured derivatives.
# ---------------------------------------------------------------------------
function _continuous(term::Symbol, deriv, c)
    if term === :streaming
        return c.a_xi .* deriv.dgdξ .+ c.a_x .* deriv.dgdx
    elseif term === :drift
        return c.c_D .* deriv.dgdξ
    elseif term === :collisions
        return c.a_y .* deriv.d2gdy2 .+ c.b_y .* deriv.dgdy
    elseif term === :perp
        return c.χ .* deriv.d2gdx2
    elseif term === :drive
        return c.drive
    elseif term === :exb
        # broadcast the 2D potential gradients over (y, E, σ)
        dΦdx = reshape(deriv.dΦdx, size(deriv.dΦdx, 1), size(deriv.dΦdx, 2), 1, 1, 1)
        dΦdξ = reshape(deriv.dΦdξ, size(deriv.dΦdξ, 1), size(deriv.dΦdξ, 2), 1, 1, 1)
        return c.c_E .* (dΦdξ .* deriv.dgdx .- dΦdx .* deriv.dgdξ)
    else
        throw(ArgumentError("unknown term $term"))
    end
end

function _term_object(term::Symbol, c)
    if term === :streaming
        return ParallelStreaming(c.a_xi, c.a_x)
    elseif term === :drift
        return MagneticDrift(c.c_D; variant=:original)
    elseif term === :collisions
        return Collisions(c.a_y, c.b_y; model=:pitch_angle)
    elseif term === :perp
        return PerpTransport(c.χ)
    elseif term === :drive
        return GradientDrive(c.drive)
    elseif term === :exb
        return ExBDrift(c.c_E)
    else
        throw(ArgumentError("unknown term $term"))
    end
end

# ---------------------------------------------------------------------------
# MMS error drivers
# ---------------------------------------------------------------------------
"""
    mms_operator_error(grid, term)

Max-norm error between the discrete `apply!` of a single kinetic `term`
(`:streaming`, `:drift`, `:exb`, `:collisions`, `:perp`, `:drive`) on the
manufactured state and its exact continuous value on `grid`.
"""
function mms_operator_error(grid::IslandGrid, term::Symbol)
    U, deriv = manufactured_state(grid)
    c = test_coefficients(grid)
    R = IslandState(grid)
    cache = IslandCache(grid)
    apply!(R, _term_object(term, c), U, grid, cache)
    exact = _continuous(term, deriv, c)
    return maximum(abs, R.g .- exact)
end

"""
    mms_assembled_error(grid; terms=(:streaming,:drift,:exb,:collisions,:perp,:drive))

Max-norm error between the assembled discrete kinetic residual (sum of `terms`)
on the manufactured state and the summed continuous values.
"""
function mms_assembled_error(grid::IslandGrid;
    terms=(:streaming, :drift, :exb, :collisions, :perp, :drive))
    U, deriv = manufactured_state(grid)
    c = test_coefficients(grid)
    R = IslandState(grid)
    cache = IslandCache(grid)
    exact = zeros(size(R.g))
    for t in terms
        apply!(R, _term_object(t, c), U, grid, cache)
        exact .+= _continuous(t, deriv, c)
    end
    return maximum(abs, R.g .- exact)
end

"""
    estimate_order(errors, refine)

Observed convergence order from a sequence of `errors` at mesh-refinement
factors `refine` (ratio of successive node counts): `log(eₖ/eₖ₊₁)/log(rₖ)`.
Returns the vector of per-step orders.
"""
function estimate_order(errors::AbstractVector, refine::AbstractVector)
    return [log(errors[k] / errors[k+1]) / log(refine[k]) for k in 1:(length(errors)-1)]
end

# ---------------------------------------------------------------------------
# Assembled stack for the JVP check (includes the E×B nonlinearity + field).
# ---------------------------------------------------------------------------
"""
    build_stack(grid; coeffs=test_coefficients(grid))

Assemble an `IslandStack` with the full Level-0-shaped term set (streaming,
drift, E×B, collisions, drive) plus the quasineutrality field term, using the
supplied (manufactured) coefficients. Used by the JVP check.
"""
function build_stack(grid::IslandGrid; coeffs=test_coefficients(grid))
    kinetic = (
        ParallelStreaming(coeffs.a_xi, coeffs.a_x),
        MagneticDrift(coeffs.c_D; variant=:original),
        ExBDrift(coeffs.c_E),
        Collisions(coeffs.a_y, coeffs.b_y; model=:pitch_angle),
        GradientDrive(coeffs.drive)
    )
    return IslandStack(kinetic, Quasineutrality(coeffs.α))
end

"""
    jvp_fd_maxerror(grid; h=1e-6, seed=1)

Compare the forward-mode-AD Jacobian–vector product of the assembled residual
against a central finite-difference directional derivative, at the manufactured
state in a deterministic direction. Returns the max-norm difference (ladder A2).
The residual is nonlinear (E×B bracket + g/Φ coupling), so this is a genuine
check of both the AD plumbing and its correctness.
"""
function jvp_fd_maxerror(grid::IslandGrid; h::Float64=1e-6, seed::Int=1)
    stack = build_stack(grid)
    cache_f = IslandCache{Float64}(grid)
    U, = manufactured_state(grid)

    N = statelength(grid)
    u0 = Vector{Float64}(undef, N)
    flatten!(u0, U)
    # deterministic direction (no RNG dependence)
    v = Float64[0.5 + 0.5 * sin(seed * k) for k in 1:N]
    v ./= norm(v)

    # residual as a flat-vector function, generic over eltype
    function resid_vec!(out, uvec)
        T = eltype(uvec)
        Ut = IslandState(reshape(view(uvec, 1:length(U.g)), size(U.g)),
            reshape(view(uvec, (length(U.g)+1):length(uvec)), size(U.Φ)))
        Rt = IslandState(reshape(view(out, 1:length(U.g)), size(U.g)),
            reshape(view(out, (length(U.g)+1):length(out)), size(U.Φ)))
        cache = IslandCache{T}(grid)
        residual!(Rt, Ut, stack, grid, cache)
        return out
    end

    # AD JVP:  d/dε residual(u0 + ε v) |_{ε=0}
    jvp_ad = ForwardDiff.derivative(0.0) do ε
        out = Vector{typeof(ε)}(undef, N)
        resid_vec!(out, u0 .+ ε .* v)
    end

    # central-difference JVP
    rp = similar(u0)
    rm = similar(u0)
    resid_vec!(rp, u0 .+ h .* v)
    resid_vec!(rm, u0 .- h .* v)
    jvp_fd = (rp .- rm) ./ (2h)

    return maximum(abs, jvp_ad .- jvp_fd)
end

# ---------------------------------------------------------------------------
# Velocity-moment quadrature self-convergence (Simpson in y, at grid order).
# ---------------------------------------------------------------------------
"""
    moment_selfconvergence(grid_coarse, grid_fine)

Max-norm difference between the velocity moment of the manufactured `g*` on two
`y`-resolutions. The two grids must share `(nx, nξ)` so the moments live on the
same `(x, ξ)` grid and compare pointwise; refine `ny` (and/or `nE`) between them
to probe the `y`-quadrature order.
"""
function moment_selfconvergence(grid_coarse::IslandGrid, grid_fine::IslandGrid)
    nnodes(grid_coarse)[1:2] == nnodes(grid_fine)[1:2] ||
        throw(ArgumentError("moment_selfconvergence needs grids sharing (nx, nξ)"))
    Uc, = manufactured_state(grid_coarse)
    Uf, = manufactured_state(grid_fine)
    Mc = zeros(nnodes(grid_coarse)[1], nnodes(grid_coarse)[2])
    Mf = zeros(nnodes(grid_fine)[1], nnodes(grid_fine)[2])
    velocity_moment!(Mc, Uc.g, grid_coarse)
    velocity_moment!(Mf, Uf.g, grid_fine)
    return maximum(abs, Mc .- Mf)
end

# ---------------------------------------------------------------------------
# Allocation probes (hot-path allocation regression, `04 §9`).
# ---------------------------------------------------------------------------
"""
    term_allocations(grid, term)

Bytes allocated by a single warmed `apply!` of kinetic `term` on `grid`. Should
be zero for the allocation-free hot path.
"""
function term_allocations(grid::IslandGrid, term::Symbol)
    c = test_coefficients(grid)
    U, = manufactured_state(grid)
    R = IslandState(grid)
    cache = IslandCache(grid)
    t = _term_object(term, c)
    apply!(R, t, U, grid, cache)          # warm up compilation
    return @allocated apply!(R, t, U, grid, cache)
end

"""
    residual_allocations(grid; coeffs=test_coefficients(grid))

Bytes allocated by a single warmed full-`residual!` assembly on `grid`. Should
be zero for the allocation-free hot path.
"""
function residual_allocations(grid::IslandGrid; coeffs=test_coefficients(grid))
    stack = build_stack(grid; coeffs=coeffs)
    U, = manufactured_state(grid)
    R = IslandState(grid)
    cache = IslandCache(grid)
    residual!(R, U, stack, grid, cache)   # warm up compilation
    return @allocated residual!(R, U, stack, grid, cache)
end

# ---------------------------------------------------------------------------
# Solve-level MMS configurations (ladder A1-solve, A5) — the manufactured
# configurations live here per 04 §4 so tests and benchmark scripts share them.
# ---------------------------------------------------------------------------
"""
    zero_drive_setup(grid)

The ladder-A5 configuration: the manufactured advective stack with **all drives
off** (no `GradientDrive`, no source), so `g ≡ 0, Φ ≡ 0` is the exact solution
and the residual there is machine zero (`01 §6`). Returns `(f, N)` — the flat
residual function and state length — for the null test and the trivial-Newton
check. The `RadiationSink(−1)` entry is a **unit relaxation shift** (a
manufactured test coefficient, not physics) making the zero state locally unique.
"""
function zero_drive_setup(grid::IslandGrid)
    c = test_coefficients(grid)
    shift = fill(-1.0, size(c.drive))
    kin = (ParallelStreaming(c.a_xi, c.a_x), MagneticDrift(c.c_D; variant=:original),
        ExBDrift(c.c_E), Collisions(c.a_y, c.b_y; model=:pitch_angle), RadiationSink(shift))
    stack = IslandStack(kin, Quasineutrality(c.α))
    return (f=flat_residual(stack, grid), N=statelength(grid))
end

"""
    solve_mms(nx; nxi=8, ny=9, nE=2, rtol=1e-10, memory=300)

The assembled **solve-level** MMS (the ladder-A1 extension to a full converged
Newton–Krylov solve): the advective manufactured stack (streaming + drift +
E×B + unit relaxation shift + quasineutrality) with far-field matching BCs
taken from the manufactured state, forced by the *analytic* continuous source
so the discrete solution differs from `g*` by the discretization error. The
first-order-in-`x` stack needs only the `x` far-field conditions (the pitch
operator's degenerate zero-flux structure is exercised separately, ladder A4).
Returns `(err, converged, iterations, gmres_iters)` where `err` is the max-norm
solution error against the manufactured state — refining `nx` must show the
design order.
"""
function solve_mms(nx::Int; nxi::Int=8, ny::Int=9, nE::Int=2, rtol::Real=1e-10, memory::Int=300)
    # halfwidth_x = 6 is the ABSTRACT MMS discretization-test box (see the module
    # docstring), NOT a physical grid 6× the plasma — the manufactured feature must sit
    # several widths inside the box for a well-conditioned order test.
    grid = IslandGrid(; nx=nx, nxi=nxi, ny=ny, nE=nE, halfwidth_x=6.0, clustering_x=1.0,
        y_max=4.0, y_c=1.0, clustering_y=0.8, order=4)
    c = test_coefficients(grid)
    shift = fill(-1.0, size(c.drive))
    kin = (ParallelStreaming(c.a_xi, c.a_x), MagneticDrift(c.c_D; variant=:original),
        ExBDrift(c.c_E), RadiationSink(shift))
    stack = IslandStack(kin, Quasineutrality(c.α))
    Ustar, deriv = manufactured_state(grid)
    bc = FarFieldConditions(Ustar.g[1, :, :, :, :], Ustar.g[nx, :, :, :, :], Ustar.Φ[1, :], Ustar.Φ[nx, :])
    f0! = flat_residual(stack, grid; bc=bc)
    N = statelength(grid)
    # continuous source: the analytic values of every active term at (g*, Φ*)
    Sg = c.a_xi .* deriv.dgdξ .+ c.a_x .* deriv.dgdx .+ c.c_D .* deriv.dgdξ .+ Ustar.g
    dPx = reshape(deriv.dΦdx, nx, nxi, 1, 1, 1)
    dPξ = reshape(deriv.dΦdξ, nx, nxi, 1, 1, 1)
    Sg .+= c.c_E .* (dPξ .* deriv.dgdx .- dPx .* deriv.dgdξ)
    Sg[1, :, :, :, :] .= 0.0
    Sg[nx, :, :, :, :] .= 0.0                      # BC rows carry no source
    MΦ = zeros(nx, nxi)
    velocity_moment!(MΦ, Ustar.g, grid)
    SΦ = MΦ .- c.α .* Ustar.Φ                       # field rows: discretely consistent source
    SΦ[1, :] .= 0.0
    SΦ[nx, :] .= 0.0
    S = zeros(N)
    flatten!(S, IslandState(Sg, SΦ))
    f!(out, u) = (f0!(out, u); out .-= S; out)
    sol = newton_krylov(f!, zeros(N); rtol=rtol, atol=1e-13, max_iter=30, memory=memory)
    ustar = zeros(N)
    flatten!(ustar, Ustar)
    return (err=maximum(abs, sol.u .- ustar), converged=sol.converged,
        iterations=sol.iterations, gmres_iters=sol.gmres_iters)
end

# ---------------------------------------------------------------------------
# y_c matching-block conditioning monitor (ladder A8, 04 §3).
# ---------------------------------------------------------------------------
"""
    yc_block_sigma_min(J, grid; window=3)

Ladder-A8 conditioning monitor: the smallest singular value of the pitch-pencil
sub-block of the (tiny-grid debug) Jacobian `J` nearest the trapped–passing
boundary `y_c`, minimized over all `(x, ξ, E, σ)` pencils. The prior art's
`y_c` matching system was intrinsically near-singular (rcond ≈ 1e-16, L23 §4.2)
and produced machine-dependent **noise, not crashes** — so this must be
*tested for*, not observed. Returns `(sigma_min, pencil)` where `pencil` is the
`(ix, iξ, iE, iσ)` of the minimizing block.
"""
function yc_block_sigma_min(J::AbstractMatrix, grid::IslandGrid; window::Int=3)
    nx, nξ, ny, nE, nσ = nnodes(grid)
    window <= ny || throw(ArgumentError("window ($window) exceeds ny ($ny)"))
    # contiguous y-index window centered on the node nearest y_c
    ic = argmin(abs.(grid.y.nodes .- grid.y_c))
    lo = clamp(ic - window ÷ 2, 1, ny - window + 1)
    iys = lo:(lo+window-1)
    σmin = Inf
    pencil = (0, 0, 0, 0)
    idx = Vector{Int}(undef, window)
    for iσ in 1:nσ, iE in 1:nE, iξ in 1:nξ, ix in 1:nx
        for (k, iy) in enumerate(iys)
            idx[k] = g_flat_index(grid, ix, iξ, iy, iE, iσ)
        end
        s = svdvals(J[idx, idx])[end]
        if s < σmin
            σmin = s
            pencil = (ix, iξ, iE, iσ)
        end
    end
    return (sigma_min=σmin, pencil=pencil)
end

# ---------------------------------------------------------------------------
# State dashboard generator (docs/07 §1.3): the auto-generated status table.
# ---------------------------------------------------------------------------
# The docs/05 ladder as a declarative status spec. `status` is the current
# reality: the A-ladder (structural) is green via the islands test suite; the
# B/C physics ladder is gated on the QUESTIONS clearances (never a physics
# result until human-cleared). This is the artifact the dashboard renders; when
# benchmark runs archive convergence artifacts, they refine the B/C rows.
const _LADDER = (
    (id="A1", tier="structural", target="MMS convergence (ξ spectral; x, y order-4)", status="green", gate="test suite"),
    (id="A2", tier="structural", target="AD-vs-FD JVP agreement", status="green", gate="test suite"),
    (id="A3", tier="structural", target="Δ_cos even / Δ_sin odd under ξ-reflection", status="green", gate="test suite"),
    (id="A4", tier="structural", target="mimetic pitch operator: exact discrete conservation + entropy sign", status="green", gate="test suite"),
    (id="A5", tier="structural", target="zero-drive null: g≡0 ⇒ residual machine-zero", status="green", gate="test suite"),
    (id="A7", tier="T1", target="coefficient-free closure identity ⟨∂²h/∂x²⟩_Ω = 0", status="green", gate="test suite"),
    (id="A8", tier="structural", target="y_c matching-block conditioning monitor", status="green", gate="test suite"),
    (id="M2c", tier="structural", target="L0 assembly builds + solves structurally (placeholders)", status="green", gate="test suite"),
    (id="B2", tier="T3", target="large-w scalings Δ_bs+Δ_cur ∝ 1/w, Δ_pol ∝ 1/w³", status="gated", gate="QUESTIONS Q5"),
    (id="B4", tier="T3", target="Δ_pol ∝ ω_E² + sign-reversal existence", status="gated", gate="QUESTIONS Q3, Q5"),
    (id="B5a", tier="T3", target="threshold existence w_c ~ O(ρ_θi), :original", status="gated", gate="QUESTIONS Q5"),
    (id="B5b/E1", tier="T2", target=":original→:improved w_c toggle differential (~×6)", status="gated", gate="QUESTIONS Q5"),
    (id="B5c", tier="T3", target="ν_★ trend dw_c/dν_★ > 0; w_c ∝ ρ̂_θi", status="gated", gate="QUESTIONS Q5"),
    (id="B7", tier="T2", target="DK vs RDK cross-check mode", status="gated", gate="QUESTIONS Q5"),
    (id="C4", tier="T3", target="finite-β/shaping triangularity trend + ε-crossover", status="planned", gate="Level 2")
)

_status_badge(s) = s == "green" ? "✅ green" : s == "gated" ? "🔒 gated" : s == "planned" ? "⏳ planned" : s

"""
    ladder_status()

Return the docs/05 verification-ladder status as a vector of NamedTuples
`(id, tier, target, status, gate)` — the declarative spec the State Dashboard
renders (docs/07 §1.3). The A-ladder rows are `green` (structural, via the
islands test suite); the B/C physics rows are `gated`/`planned` on the QUESTIONS
clearances (no physics result is claimed until human-cleared).
"""
ladder_status() = collect(_LADDER)

"""
    write_state_dashboard(io=stdout; commit="unknown", date="unknown")

Emit the Islands **State Dashboard** (docs/07 §1.3) as Markdown to `io`: the
docs/05 ladder as a status table (ID, tier, target, status, gate) stamped with
`commit` and `date`. This is the generator for
`docs/src/islands/state/STATE.md`; that file is **auto-generated — never
hand-edit it** (docs/07 §1.3), it regenerates from this function (and, as they
land, from archived benchmark artifacts). Deterministic given its arguments.
"""
function write_state_dashboard(io::IO=stdout; commit::AbstractString="unknown", date::AbstractString="unknown")
    println(io, "# Islands — State Dashboard")
    println(io)
    println(io, "!!! warning \"Auto-generated — do not hand-edit\"")
    println(io, "    Generated by `Islands.Verify.write_state_dashboard` (docs/07 §1.3).")
    println(io, "    Regenerate rather than edit; it refreshes from the ladder spec and,")
    println(io, "    as they land, archived benchmark artifacts.")
    println(io)
    println(io, "Snapshot: commit `", commit, "` — ", date, ".")
    println(io)
    println(io, "The docs/05 verification ladder. The **A-ladder** (structural, pre-physics)")
    println(io, "is green via `test/runtests_islands_*.jl`. The **B/C physics ladder** is")
    println(io, "gated on the `QUESTIONS.md` clearances — no physics result is claimed until")
    println(io, "human sign-off (the `[VERIFY]` policy, module CLAUDE.md).")
    println(io)
    println(io, "| ID | Tier | Target | Status | Gate |")
    println(io, "|---|---|---|---|---|")
    for r in _LADDER
        println(io, "| ", r.id, " | ", r.tier, " | ", r.target, " | ", _status_badge(r.status), " | ", r.gate, " |")
    end
    println(io)
    ngreen = count(r -> r.status == "green", _LADDER)
    ngated = count(r -> r.status == "gated", _LADDER)
    nplan = count(r -> r.status == "planned", _LADDER)
    println(io, "Summary: **", ngreen, " green** (structural), ", ngated, " gated (physics, awaiting clearance), ", nplan, " planned.")
    println(io)
    println(io, "Tiers (Decision D9, docs/05 \"Target tiers\"): T1 exact math · T2 internal")
    println(io, "differentials · T3 scalings/trends/existence · T4 absolute (audit-gated).")
    return io
end

# ---------------------------------------------------------------------------
# Anchor-sync check (docs/07 §1.1, §4.2): the bidirectional operator ↔ docs
# consistency the CI enforces.
# ---------------------------------------------------------------------------
# Resolve a dotted symbol (e.g. "Operators.MagneticDrift") from `root` (Islands),
# walking submodules. Returns true iff every component is defined.
function _resolve_symbol(qualified::AbstractString, root::Module)
    m = root
    parts = split(qualified, '.')
    for (i, p) in enumerate(parts)
        sym = Symbol(p)
        isdefined(m, sym) || return false
        i == length(parts) && return true
        v = getfield(m, sym)
        v isa Module || return false
        m = v
    end
    return true
end

# Extract the backtick-quoted symbols from every `Implemented by:` block of a
# doc — the block runs from the marker to the next blank line, so a wrapped
# multi-line list is captured whole.
function _implemented_by_symbols(docfile::AbstractString)
    text = read(docfile, String)
    syms = String[]
    for block in eachmatch(r"Implemented by:(.*?)(?:\n\n|\z)"s, text)
        for m in eachmatch(r"`([^`]+)`", block.captures[1])
            push!(syms, String(m.captures[1]))
        end
    end
    return syms
end

# The AbstractTerm operator names declared in the Operators source. The `<:
# AbstractTerm` must sit *outside* the type-parameter braces, so a struct that
# merely carries an `F<:AbstractTerm` type parameter (e.g. `IslandStack`) is not
# a false match.
function _operator_names(operators_file::AbstractString)
    text = read(operators_file, String)
    return [String(m.captures[1]) for m in eachmatch(r"struct\s+(\w+)(?:\{[^}]*\})?\s*<:\s*AbstractTerm\b", text)]
end

"""
    check_anchor_sync(root=parentmodule(@__MODULE__); operators_file, docfiles)

The docs/07 §1.1 bidirectional anchor-sync check, as a pure function CI can gate
on. Returns `(undocumented, dangling)`:

  - `undocumented` — `AbstractTerm` operators declared in `operators_file` whose
    name is **not** claimed by any `Implemented by:` marker in `docfiles` (an
    operator with no as-implemented documentation);
  - `dangling` — backtick-quoted symbols on an `Implemented by:` line that do
    **not** resolve to a defined name under `root` (a doc pointing at a
    nonexistent/renamed symbol).

Both empty ⇒ the operator stack and the as-implemented docs are in sync. `root`
defaults to the `Islands` module; `operators_file`/`docfiles` default to the
in-repo paths relative to this source file.
"""
function check_anchor_sync(root::Module=parentmodule(@__MODULE__);
    operators_file::AbstractString=normpath(joinpath(@__DIR__, "..", "operators", "Operators.jl")),
    docfiles=[normpath(joinpath(@__DIR__, "..", "..", "..", "docs", "src", "islands", "numerics.md"))])
    implemented = String[]
    for f in docfiles
        isfile(f) && append!(implemented, _implemented_by_symbols(f))
    end
    implemented_leaves = Set(last(split(s, '.')) for s in implemented)
    ops = _operator_names(operators_file)
    undocumented = [op for op in ops if !(op in implemented_leaves)]
    dangling = [s for s in implemented if !_resolve_symbol(s, root)]
    return (undocumented=undocumented, dangling=dangling)
end

end # module Verify
