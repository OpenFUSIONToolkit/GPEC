"""
    PhaseSpace

Phase-space grids and discretization operators for the Islands drift-kinetic
solver: the `(x, ξ, λ→y, E, σ)` coordinates of design doc `03 §1` with the
layer-clustered mappings of `04 §1`.

This module is **pure numerics** — grid node placement, spectral/finite-difference
differentiation matrices, and quadrature weights. It contains no physics
coefficients: nothing here carries a `[VERIFY]`/`[CHECKED]` tag, and the
milestone-M1 discipline (build the discretization, not the physics numbers)
lives one layer up in `Operators`.

Design-order summary (verified by the MMS ladder A1, `Verify`):

  - `ξ` (helical angle): Fourier pseudo-spectral on the periodic domain `[0, L)`
    — exponential convergence for smooth periodic data (`04 §1`).
  - `x` (radial) and `y` (pitch): high-order finite differences on a stretched,
    layer-clustered grid — algebraic convergence at the requested `order`
    (`04 §1`; the layer packing targets `x = 0` and `y = y_c = 1`).
  - `E` (energy): Gauss quadrature on the `F₀`-weighted semi-infinite domain
    (`04 §1`, Maxwellian weight at Level 0 via Gauss–Laguerre).
  - `σ = ±1`: the two `sgn(v_∥)` sheets.
"""
module PhaseSpace

using LinearAlgebra
import FastGaussQuadrature

export FourierGrid, MappedFDGrid, GaussGrid, IslandGrid
export nnodes, differentiate_fourier, fd_weights
export island_clustering_x, central_x_spacing, is_island_resolved, resolved_island_grid
export banded_x_nodes, drift_island_grid

# ---------------------------------------------------------------------------
# Finite-difference weights (Fornberg 1988, Math. Comp. 51, 699) — weights for
# derivatives 0..m of a function sampled at arbitrary `nodes`, evaluated at x0.
# Returns `w[j+1, d+1]` = weight of node j for the d-th derivative.
# ---------------------------------------------------------------------------
"""
    fd_weights(m, nodes, x0)

Finite-difference weights for derivatives `0:m` of a function sampled at
`nodes`, evaluated at `x0`, via Fornberg's recurrence. `w[j, d+1]` multiplies
`f(nodes[j])` in the approximation of the `d`-th derivative. Exact for
polynomials up to degree `length(nodes) - 1`.
"""
function fd_weights(m::Int, nodes::AbstractVector{<:Real}, x0::Real)
    n = length(nodes)
    @assert m < n "need at least m+1 nodes for the m-th derivative"
    w = zeros(Float64, n, m + 1)
    c1 = 1.0
    c4 = nodes[1] - x0
    w[1, 1] = 1.0
    for i in 2:n
        mn = min(i, m + 1)
        c2 = 1.0
        c5 = c4
        c4 = nodes[i] - x0
        for j in 1:(i-1)
            c3 = nodes[i] - nodes[j]
            c2 *= c3
            if j == i - 1
                for k in mn:-1:2
                    w[i, k] = c1 * ((k - 1) * w[i-1, k-1] - c5 * w[i-1, k]) / c2
                end
                w[i, 1] = -c1 * c5 * w[i-1, 1] / c2
            end
            for k in mn:-1:2
                w[j, k] = (c4 * w[j, k] - (k - 1) * w[j, k-1]) / c3
            end
            w[j, 1] = c4 * w[j, 1] / c3
        end
        c1 = c2
    end
    return w
end

# ---------------------------------------------------------------------------
# ξ: Fourier pseudo-spectral, periodic on [0, L).
# ---------------------------------------------------------------------------
"""
    FourierGrid(n; L=2π)

Uniform periodic grid of `n` nodes on `[0, L)` with the Fourier spectral
first-derivative matrix `D1` (`04 §1`, `ξ` coordinate). `n` must be even.

## Fields

  - `n`     — number of nodes.
  - `L`     — period length.
  - `nodes` — node positions `j·L/n`, `j = 0:n-1`.
  - `D1`    — `n×n` spectral first-derivative matrix.
"""
struct FourierGrid
    n::Int
    L::Float64
    nodes::Vector{Float64}
    D1::Matrix{Float64}
end

function FourierGrid(n::Int; L::Real=2π)
    iseven(n) || throw(ArgumentError("FourierGrid needs an even node count (got $n)"))
    h = 2π / n                          # computational grid spacing on [0, 2π)
    nodes = collect((0:(n-1)) .* (L / n))
    D1 = zeros(Float64, n, n)
    @inbounds for i in 1:n, j in 1:n
        if i != j
            k = i - j
            D1[i, j] = 0.5 * (-1.0)^k / tan(k * h / 2)
        end
    end
    D1 .*= (2π / L)                     # rescale d/dξ from [0,2π) to [0,L)
    return FourierGrid(n, Float64(L), nodes, D1)
end

"""
    differentiate_fourier(g, grid)

Spectral first derivative of a periodic sample vector `g` on `grid` (allocating
convenience wrapper around `grid.D1 * g`; hot paths use the matrix directly).
"""
differentiate_fourier(g::AbstractVector, grid::FourierGrid) = grid.D1 * g

# ---------------------------------------------------------------------------
# x, y: high-order finite differences on a layer-clustered stretched grid.
#
# A uniform computational coordinate s ∈ [-1, 1] is mapped to the physical
# coordinate by a smooth, monotonic, layer-clustering map; physical derivative
# matrices follow by the chain rule so the FD order is preserved (`04 §1`).
# ---------------------------------------------------------------------------
"""
    MappedFDGrid(n; halfwidth, clustering=0.0, center=0.0, domain=:symmetric, order=4)

High-order finite-difference grid of `n` nodes with layer clustering.

The uniform computational coordinate `s ∈ [-1, 1]` is mapped to the physical
coordinate by `sinh` stretching (`clustering = β`): points cluster toward the
map center as `β` grows, and the map degenerates to uniform as `β → 0`. This is
the design-doc packing that targets the internal layers — `x = 0` (rational
surface) and `y = y_c = 1` (trapped–passing boundary), `04 §1–2`.

`domain = :symmetric` gives `[center - halfwidth, center + halfwidth]` with
clustering at `center`; `domain = :half` gives `[0, halfwidth]` with clustering
at `center` (used for `y ∈ [0, y_max]` packed at `y_c`).

Physical first/second-derivative matrices `D1`, `D2` are built with Fornberg
weights on windows sized per derivative (`order + d` points for the `d`-th
derivative, shifted one-sided near the boundaries) so both reach `order`-th
order accuracy *uniformly*, including at the boundary rows; the chain rule uses
the analytic map Jacobian `x'(s)` and `x''(s)`.

`wq` holds composite-Simpson quadrature weights on the same nodes (built on the
uniform computational grid and pushed through the map Jacobian), so
`∫ f dx ≈ Σ wq[j] f(nodes[j])` at fourth order — matching the FD order for the
velocity-moment integrals (`03 §2`, moments). Requires an odd `n`.

## Fields

  - `n`, `order` — node count and nominal FD order.
  - `nodes`      — physical node positions.
  - `D1`, `D2`   — `n×n` physical first/second-derivative matrices.
  - `wq`         — composite-Simpson quadrature weights on `nodes`.
"""
struct MappedFDGrid
    n::Int
    order::Int
    nodes::Vector{Float64}
    D1::Matrix{Float64}
    D2::Matrix{Float64}
    wq::Vector{Float64}
end

function MappedFDGrid(n::Int; halfwidth::Real, clustering::Real=0.0, center::Real=0.0,
    domain::Symbol=:symmetric, order::Int=4)
    isodd(n) || throw(ArgumentError("MappedFDGrid needs odd n for composite Simpson (got $n)"))
    # widest window is for D2 (order+2 points); need enough nodes for it.
    n > order + 2 || throw(ArgumentError("need n > order+2 ($n ≤ $(order + 2))"))

    s = collect(range(-1.0, 1.0; length=n))            # uniform computational grid
    hw = Float64(halfwidth)
    β = Float64(clustering)

    # Map s → physical, with analytic first/second derivatives of the map.
    if domain === :symmetric
        # x(s) = center + hw·sinh(β s)/sinh(β): monotone, clusters at s=0 ↔ x=center.
        if abs(β) < 1e-12
            xs = center .+ hw .* s
            dxds = fill(hw, n)
            d2xds2 = zeros(n)
        else
            sb = sinh(β)
            xs = center .+ hw .* sinh.(β .* s) ./ sb
            dxds = hw .* β .* cosh.(β .* s) ./ sb
            d2xds2 = hw .* β^2 .* sinh.(β .* s) ./ sb
        end
    elseif domain === :half
        # Map [-1,1] → [0, hw] with clustering at s* ↔ x=center via sinh about s*.
        # u(s) = sinh(β(s - s*)); x = hw·(u - u(-1))/(u(1) - u(-1)).
        sstar = clamp(2 * (center / hw) - 1, -1.0, 1.0)
        if abs(β) < 1e-12
            xs = hw .* (s .+ 1) ./ 2
            dxds = fill(hw / 2, n)
            d2xds2 = zeros(n)
        else
            u(z) = sinh(β * (z - sstar))
            um1 = u(-1.0)
            span = u(1.0) - um1
            xs = hw .* (u.(s) .- um1) ./ span
            dxds = hw .* β .* cosh.(β .* (s .- sstar)) ./ span
            d2xds2 = hw .* β^2 .* sinh.(β .* (s .- sstar)) ./ span
        end
    else
        throw(ArgumentError("unknown domain $domain (use :symmetric or :half)"))
    end

    D1s = _fd_matrix(s, 1, order)
    D2s = _fd_matrix(s, 2, order)

    # Chain rule: d/dx = (1/x')·d/ds ; d²/dx² = (1/x'²)·d²/ds² − (x''/x'³)·d/ds.
    invp = 1.0 ./ dxds
    D1 = Diagonal(invp) * D1s
    D2 = Diagonal(invp .^ 2) * D2s - Diagonal(d2xds2 .* invp .^ 3) * D1s

    # Composite Simpson on uniform s (ds = 2/(n-1)), times the map Jacobian dx/ds.
    ds = 2.0 / (n - 1)
    wq = similar(dxds)
    @inbounds for j in 1:n
        c = (j == 1 || j == n) ? 1.0 : (iseven(j) ? 4.0 : 2.0)
        wq[j] = c * ds / 3 * dxds[j]
    end

    return MappedFDGrid(n, order, xs, Matrix(D1), Matrix(D2), wq)
end

"""
    MappedFDGrid(nodes::AbstractVector; order=4)

Build a [`MappedFDGrid`](@ref) directly from an explicit, ascending physical node
vector — the general (non-analytic-map) grid used by the drift-island band grid
([`drift_island_grid`](@ref)), whose nodes are a uniform high-resolution central
band plus geometric tails (`04 §1`) rather than a single-`sinh` cluster.

`D1`, `D2` are the Fornberg finite-difference matrices for the given nodes (the
same `_fd_matrix` machinery as the analytic-map path; `order`-th order on
smoothly-varying spacing, degrading locally where the spacing jumps). `wq` is the
composite-**trapezoidal** rule on `nodes` — the radial-`x` quadrature weights are
not used for physics outputs (velocity moments use the `y`/`E` weights; the `Δ`
radial integral uses cubic-spline quadrature in `Moments.delta_moments`), so a
robust low-order `wq` here is deliberate, not an accuracy regression.
"""
function MappedFDGrid(nodes::AbstractVector{<:Real}; order::Int=4)
    n = length(nodes)
    n > order + 2 || throw(ArgumentError("MappedFDGrid(nodes): need n > order+2 ($n ≤ $(order + 2))"))
    issorted(nodes) || throw(ArgumentError("MappedFDGrid(nodes): nodes must be ascending"))
    xs = collect(Float64, nodes)
    all(diff(xs) .> 0) || throw(ArgumentError("MappedFDGrid(nodes): nodes must be strictly increasing"))
    D1 = _fd_matrix(xs, 1, order)
    D2 = _fd_matrix(xs, 2, order)
    wq = zeros(n)
    @inbounds for j in 1:n
        hl = j > 1 ? xs[j] - xs[j - 1] : 0.0
        hr = j < n ? xs[j + 1] - xs[j] : 0.0
        wq[j] = 0.5 * (hl + hr)
    end
    return MappedFDGrid(n, order, xs, Matrix(D1), Matrix(D2), wq)
end

# Dense derivative matrix for the `deriv`-th derivative on an arbitrary 1D node
# set at the requested `order`, using windows of `order + deriv` points (the
# minimum for `order`-th accuracy of the `deriv`-th derivative), rounded up to
# an odd width so the interior window is centered; near the ends the window is
# shifted one-sided. Weights are Fornberg's, exact for polynomials up to the
# window degree.
function _fd_matrix(nodes::AbstractVector{<:Real}, deriv::Int, order::Int)
    n = length(nodes)
    stencil = order + deriv
    isodd(stencil) || (stencil += 1)   # centered interior window needs odd width
    stencil = min(stencil, n)
    D = zeros(Float64, n, n)
    half = stencil ÷ 2
    @inbounds for i in 1:n
        lo = clamp(i - half, 1, n - stencil + 1)
        hi = lo + stencil - 1
        w = fd_weights(deriv, @view(nodes[lo:hi]), nodes[i])
        for (col, j) in enumerate(lo:hi)
            D[i, j] = w[col, deriv+1]
        end
    end
    return D
end

# ---------------------------------------------------------------------------
# E: Gauss quadrature on the F₀-weighted semi-infinite energy domain.
# ---------------------------------------------------------------------------
"""
    GaussGrid(n; kind=:laguerre)

Gauss quadrature nodes/weights for velocity-space (energy) integrals over the
`F₀`-weighted semi-infinite domain (`04 §1`). `kind = :laguerre` gives the
Level-0 Maxwellian weight `∫₀^∞ f(E) e^{-E} dE ≈ Σ wᵢ f(Eᵢ)` (Gauss–Laguerre);
the slowing-down weight (Level 2) only changes `kind`, not the machinery.

## Fields

  - `n`       — number of quadrature nodes.
  - `nodes`   — abscissae `Eᵢ`.
  - `weights` — quadrature weights `wᵢ` (the `F₀` weight is folded in).
"""
struct GaussGrid
    n::Int
    nodes::Vector{Float64}
    weights::Vector{Float64}
end

function GaussGrid(n::Int; kind::Symbol=:laguerre)
    if kind === :laguerre
        nodes, weights = FastGaussQuadrature.gausslaguerre(n)
    elseif kind === :legendre
        nodes, weights = FastGaussQuadrature.gausslegendre(n)
    else
        throw(ArgumentError("unknown quadrature kind $kind"))
    end
    return GaussGrid(n, collect(nodes), collect(weights))
end

# ---------------------------------------------------------------------------
# Bundle: the full Level-0 orbit-averaged 4D phase space (x, ξ, y, E) plus σ.
# ---------------------------------------------------------------------------
"""
    IslandGrid(; nx, nxi, ny, nE, halfwidth_x, clustering_x, y_max, y_c,
               clustering_y, xi_period=2π, energy_kind=:laguerre, order=4)

The assembled Level-0 phase-space grid: radial `x`, helical `ξ`, pitch `y`,
energy `E`, and the two `σ = ±1` sheets (`03 §1`). Grids are packed at the
internal layers (`x = 0`, `y = y_c`) per `04 §1`.

## Fields

  - `x`   — radial `MappedFDGrid` (clustered at `x = 0`).
  - `ξ`   — helical `FourierGrid`.
  - `y`   — pitch `MappedFDGrid` on `[0, y_max]` (clustered at `y_c`).
  - `E`   — energy `GaussGrid`.
  - `σ`   — `[+1.0, -1.0]`.
  - `y_c` — trapped–passing boundary location in `y` (the layer the pitch grid
    packs toward; consumed by the `y_c`-block conditioning monitor, ladder A8).
"""
struct IslandGrid
    x::MappedFDGrid
    ξ::FourierGrid
    y::MappedFDGrid
    E::GaussGrid
    σ::Vector{Float64}
    y_c::Float64
end

function IslandGrid(; nx::Int, nxi::Int, ny::Int, nE::Int,
    halfwidth_x::Real, clustering_x::Real=0.0,
    y_max::Real, y_c::Real=1.0, clustering_y::Real=0.0,
    xi_period::Real=2π, energy_kind::Symbol=:laguerre, order::Int=4)
    x = MappedFDGrid(nx; halfwidth=halfwidth_x, clustering=clustering_x, center=0.0,
        domain=:symmetric, order=order)
    ξ = FourierGrid(nxi; L=xi_period)
    y = MappedFDGrid(ny; halfwidth=y_max, clustering=clustering_y, center=y_c,
        domain=:half, order=order)
    E = GaussGrid(nE; kind=energy_kind)
    return IslandGrid(x, ξ, y, E, [1.0, -1.0], Float64(y_c))
end

"""
    nnodes(grid::IslandGrid)

Tuple `(nx, nξ, ny, nE, nσ)` of the phase-space grid dimensions.
"""
nnodes(g::IslandGrid) = (g.x.n, g.ξ.n, g.y.n, g.E.n, length(g.σ))

# ---------------------------------------------------------------------------
# Island-resolution protocol (04 §2): the radial grid must resolve the island
# half-width — the Δ moments live on the separatrix (x ~ w), not the x = 0 node,
# so a coarse grid gives Lx-/clustering-sensitive garbage. These helpers size the
# radial clustering so Δx(0) ≤ w/K (K nodes across the island half-width).
# ---------------------------------------------------------------------------
"""
    island_clustering_x(w, Lx, nx; K=8, beta_cap=12.0)

The `sinh`-map clustering `β` (the `clustering_x` of [`MappedFDGrid`](@ref)) that
resolves an island of **half-width** `w` on `nx` symmetric nodes over
`[-Lx, Lx]`: it packs the center so the central radial spacing satisfies
`Δx(0) ≤ w/K` — `K` nodes across the island half-width (`04 §2`; the Δ-moment
response peaks at the separatrix `x ~ w`, so `Δx ≪ w` is the adequacy condition).

The exact two-node central spacing of the symmetric map is
`Δx(0) = Lx·sinh(βΔs)/sinh(β)` with `Δs = 2/(nx-1)`; since it decreases
monotonically in `β`, the required `β` is found by bisection. Returns `0.0`
(uniform, no clustering) when the plain spacing `Lx·Δs` already meets `w/K`.
Throws if resolving `w/K` would need `β > beta_cap` (the grid would be so
center-packed the far field is starved — increase `nx` or `Lx`, or lower `K`).
"""
function island_clustering_x(w::Real, Lx::Real, nx::Integer; K::Real=8, beta_cap::Real=12.0)
    (w > 0 && Lx > 0) || throw(ArgumentError("island_clustering_x: w and Lx must be positive"))
    nx >= 3 || throw(ArgumentError("island_clustering_x: nx must be ≥ 3"))
    Δs = 2 / (nx - 1)
    target = (w / (K * Lx)) * (1 - 1e-6)         # required Δx(0)/Lx (hair under, so ≤ holds strictly)
    target >= Δs && return 0.0                   # uniform spacing already resolves it
    h(β) = sinh(β * Δs) / sinh(β)                # Δx(0)/Lx as a function of β (↓)
    lo, hi = 0.0, 1.0
    while h(hi) > target
        hi *= 2
        hi > 1e4 && break                        # bracket search safety (should never trigger)
    end
    for _ in 1:200
        mid = (lo + hi) / 2
        h(mid) > target ? (lo = mid) : (hi = mid)
    end
    β = (lo + hi) / 2
    β > beta_cap && throw(
        ArgumentError(
            "island_clustering_x: resolving w/K = $(w / K) on nx = $nx nodes over Lx = $Lx needs β = $(round(β, digits=2)) > beta_cap = $beta_cap (far field starved) — increase nx or Lx, or lower K"
        )
    )
    return β
end

"""
    central_x_spacing(grid::IslandGrid)

The smallest radial node spacing of `grid` — the central spacing `Δx(0)` for a
center-clustered grid. The island-resolution diagnostic paired with
[`island_clustering_x`](@ref).
"""
central_x_spacing(grid::IslandGrid) = minimum(diff(grid.x.nodes))

"""
    is_island_resolved(grid, w; K=8, min_Lx_over_w=5.0)

Check whether `grid` adequately resolves an island of half-width `w` (`04 §2`,
docs/05 reporting): the central radial spacing must satisfy `Δx(0) ≤ w/K` **and**
the radial domain must reach `≥ min_Lx_over_w · w` (the far field must be truly
far, `L_x/w ≳ 5`). Returns a NamedTuple
`(resolved, central_spacing, nodes_per_halfwidth, Lx_over_w)` for the
two-resolution convergence protocol (no Δ-output benchmark passes on one grid).
"""
function is_island_resolved(grid::IslandGrid, w::Real; K::Real=8, min_Lx_over_w::Real=5.0)
    w > 0 || throw(ArgumentError("is_island_resolved: w must be positive"))
    Δ0 = central_x_spacing(grid)
    Lx = grid.x.nodes[end]                       # right edge = half-width (symmetric, center 0)
    ratio = Lx / w
    return (resolved=(Δ0 <= w / K && ratio >= min_Lx_over_w), central_spacing=Δ0,
        nodes_per_halfwidth=w / Δ0, Lx_over_w=ratio)
end

"""
    resolved_island_grid(; w, nx, K=8, Lx_over_w=6.0, nxi, ny, nE, y_max,
                         y_c=1.0, clustering_y=0.0, xi_period=2π,
                         energy_kind=:laguerre, order=4, beta_cap=12.0)

Build an [`IslandGrid`](@ref) that resolves an island of **half-width** `w`
(`04 §2`): the radial domain is `[-Lx, Lx]` with `Lx = Lx_over_w · w` (far field
`L_x/w ≳ 5`) and `clustering_x` is set by [`island_clustering_x`](@ref) so
`Δx(0) ≤ w/K`. All other coordinates pass through to the `IslandGrid`
constructor. Pair with [`is_island_resolved`](@ref) and run every Δ-output at
`≥ 2` resolutions (vary `nx`/`K`) to confirm convergence.

**Physical-domain caution** (`x = (ψ−ψ_s)/ψ_s`, so the magnetic axis is at
`x = −1`): this is a **local** layer model, so `Lx = Lx_over_w · w` must stay a
local matching radius, `|x| ≲ 0.2–0.3` — i.e. `Lx_over_w · w ≪ 1`. A domain with
`Lx > 1` extends past the plasma (nonexistent flux); "converging" a physics run by
enlarging `Lx` toward the plasma scale is unphysical (design 10). For a physics
scan, fix `Lx` at the physical matching radius, independent of `w` — do not scale
the far field to plasma size.
"""
function resolved_island_grid(; w::Real, nx::Integer, K::Real=8, Lx_over_w::Real=6.0,
    nxi::Integer, ny::Integer, nE::Integer, y_max::Real, y_c::Real=1.0,
    clustering_y::Real=0.0, xi_period::Real=2π, energy_kind::Symbol=:laguerre,
    order::Integer=4, beta_cap::Real=12.0)
    Lx = Lx_over_w * w
    βx = island_clustering_x(w, Lx, nx; K=K, beta_cap=beta_cap)
    return IslandGrid(; nx=nx, nxi=nxi, ny=ny, nE=nE, halfwidth_x=Lx, clustering_x=βx,
        y_max=y_max, y_c=y_c, clustering_y=clustering_y, xi_period=xi_period,
        energy_kind=energy_kind, order=order)
end

# ---------------------------------------------------------------------------
# Drift-island band grid (04 §1): the perturbed response localises on the
# drift-island separatrices, which sit at x shifted from the magnetic island
# (x=0) by ±x_D^island(y,v̂,σ) and are SPREAD across velocity space. A single
# sinh cluster at x=0 resolves only |x|≲w and leaves the shifted drift islands in
# the coarse far field, so the Δ-moment is under-resolved. The band grid instead
# lays a UNIFORM high-resolution central region (spacing ≤ w/K) over the whole
# shift envelope [-R, R] and coarsens geometrically to ±Lx outside — matching the
# "uniform high-res central region covering the drift-shifted island" of the prior
# art (L23 Ch. 3). The physical shift envelope R is a physics quantity computed by
# `Configure.drift_island_shift_envelope`; here R enters as a plain number
# (PhaseSpace stays physics-free).
# ---------------------------------------------------------------------------
"""
    banded_x_nodes(; R, h, Lx, max_ratio=1.3)

Symmetric radial node vector for the drift-island band grid (`04 §1`): a uniform
central band of spacing `h` covering at least `[-R, R]`, then a **graded** tail on
each side out to exactly `±Lx`. `R` is the shift envelope half-width and `h = w/K`
the island-resolution spacing; require `Lx > R` (the far field must lie beyond the
drift-island envelope, not merely beyond the magnetic island).

**Smoothness is essential**: the `(x, ξ)` plane-block preconditioner
(`Solvers.PlaneJacobi`) anti-preconditions on a grid with an abrupt spacing jump.
The tail therefore grows **geometrically with a single ratio `r ≤ max_ratio`**,
solved (bisection) so the geometric sum lands *exactly* on `Lx` — every adjacent
interval ratio is then `r` (including the band→tail join), with no truncated
"sliver" interval. The node count `n_tail` is the smallest that admits an
`r ∈ (1, max_ratio]`; if the tail is shorter than one band step it is a single
uniform step. Returns an ascending, strictly-increasing, symmetric vector
including `0`, the band nodes, and exactly `±Lx`.
"""
function banded_x_nodes(; R::Real, h::Real, Lx::Real, max_ratio::Real=1.3)
    (R > 0 && h > 0) || throw(ArgumentError("banded_x_nodes: R and h must be positive"))
    max_ratio > 1 || throw(ArgumentError("banded_x_nodes: max_ratio must be > 1"))
    Lx > R || throw(ArgumentError("banded_x_nodes: need Lx ($Lx) > R ($R) — far field must lie beyond the drift-island envelope"))
    nband = ceil(Int, R / h)                         # band half-node-count (covers ≥ R)
    Rb = nband * h                                    # actual band edge (≥ R)
    Rb < Lx || throw(ArgumentError("banded_x_nodes: band edge $Rb reaches Lx $Lx — increase Lx or lower R/K"))
    S = (Lx - Rb) / h                                 # tail span in units of h
    right = collect(1:nband) .* h                     # (0, Rb] uniform band nodes
    # geometric tail: n intervals with ratio r, Σ_{k=1}^{n} h rᵏ = Lx-Rb ⇒ sum(r,n)=S.
    # For r>1, sum(r,n) > n, so r ≤ max_ratio is feasible only when n < S and
    # sum(max_ratio,n) ≥ S. Take the smallest n meeting the coarsest-tail bound; if that
    # n ≥ S (tail too short to grade), fall back to a uniform tail (spacing ≤ h — smooth).
    gesum(r, n) = isapprox(r, 1; atol=1e-12) ? float(n) : r * (r^n - 1) / (r - 1)
    n_geo = 1
    while gesum(max_ratio, n_geo) < S
        n_geo += 1
    end
    if n_geo < S - 1e-9                               # graded geometric tail (r > 1 feasible)
        lo, hi = 1.0 + 1e-12, Float64(max_ratio)
        for _ in 1:200
            mid = (lo + hi) / 2
            gesum(mid, n_geo) < S ? (lo = mid) : (hi = mid)
        end
        r = (lo + hi) / 2
        x = Rb
        step = h
        for k in 1:n_geo
            step *= r
            x += step
            push!(right, k == n_geo ? Lx : x)         # force the last node exactly onto Lx
        end
    else                                              # short tail: uniform steps ≤ h (smooth)
        n_u = max(1, ceil(Int, S))
        du = (Lx - Rb) / n_u
        for k in 1:n_u
            push!(right, k == n_u ? Lx : Rb + k * du)
        end
    end
    return vcat(-reverse(right), 0.0, right)
end

"""
    drift_island_grid(; R, w, K=8, Lx_over_w=12.0, max_ratio=1.3, nxi, ny, nE, y_max,
                      y_c=1.0, clustering_y=0.0, xi_period=2π, energy_kind=:laguerre,
                      order=4)

Build an [`IslandGrid`](@ref) whose radial axis is the drift-island **band grid**
([`banded_x_nodes`](@ref)): a uniform central region of spacing `w/K` covering the
shift envelope `[-R, R]`, coarsening (adjacent ratio `≤ max_ratio`) to
`±Lx = ±Lx_over_w·w` outside (`04 §1`). `R` is the drift-island shift envelope
half-width (from `Configure.drift_island_shift_envelope`); pass `R` such that
`Lx > R`. The `ξ`, `y`, `E` axes are built exactly as [`IslandGrid`](@ref). The
resulting `nx` is determined by the band/tail layout (not an input); read it from
the grid.

Pair with [`is_island_resolved`](@ref) for the central-spacing check and run every
Δ-output at `≥ 2` resolutions (vary `K`) to confirm convergence.
"""
function drift_island_grid(; R::Real, w::Real, K::Real=8, Lx_over_w::Real=12.0,
    max_ratio::Real=1.3, nxi::Integer, ny::Integer, nE::Integer, y_max::Real,
    y_c::Real=1.0, clustering_y::Real=0.0, xi_period::Real=2π,
    energy_kind::Symbol=:laguerre, order::Integer=4)
    Lx = Lx_over_w * w
    nodes = banded_x_nodes(; R=R, h=w / K, Lx=Lx, max_ratio=max_ratio)
    x = MappedFDGrid(nodes; order=order)
    ξ = FourierGrid(nxi; L=xi_period)
    y = MappedFDGrid(ny; halfwidth=y_max, clustering=clustering_y, center=y_c, domain=:half, order=order)
    E = GaussGrid(nE; kind=energy_kind)
    return IslandGrid(x, ξ, y, E, [1.0, -1.0], Float64(y_c))
end

end # module PhaseSpace
