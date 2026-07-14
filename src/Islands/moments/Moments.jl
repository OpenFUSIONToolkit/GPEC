"""
    Moments

Output-moment assembly (design `01 §4`, `03 §1`): the parallel current
`J̄_∥(x, ξ)` from species-summed velocity moments, its `cos ξ`/`sin ξ` Ampère
projections `Δ_cos`/`Δ_sin`, and the island flux-surface-average diagnostics
(`Ω` label, `⟨·⟩_Ω`, bootstrap/polarization channel split).

**Gating:** the projection and quadrature machinery here is pure numerics. The
physics enters through (i) the per-species velocity-space weights `W_j` (the
`v̂_∥`-structure — **cleared** `W = v̂_∥ = σ√E√(1−y b_min)`,
`Configure.parallel_flow_weight`, sign-off 2026-07-13, with the physical `∫d³v`
measure `velocity-moment-measure.md`), and (ii) the `Δ` moment prefactors
(`±μ₀R/2ψ̃`). The `ψ̃` amplitude is **cleared** — see
[`island_flux_amplitude`](@ref) (human sign-off 2026-07-11; derivation
`docs/src/islands/derivations/psi-tilde-amplitude.md`, docs/01 §1) — but the
`μ₀R` normalization and the sin-moment normalization pin (`[DERIVED]`,
docs/01 §4) are not, so `delta_moments`' prefactors remain **required
caller-supplied arguments** with no defaults; nothing here assigns them.

The island label convention is the module-CLAUDE.md pin: `Ω = 2x²/w² − cos ξ`,
O-point `Ω = −1`, separatrix `Ω = +1`, `w` = **half**-width.
"""
module Moments

using LinearAlgebra
import QuadGK
import ..PhaseSpace: IslandGrid, nnodes, fd_weights
import ..Operators: weighted_moment!
import ..SpeciesLists: Species

export parallel_current!, delta_moments, omega_label, omega_average, channel_split
export island_flux_amplitude, grid_interpolant, channel_decomposition

"""
    island_flux_amplitude(; w_psi, dq_dpsi, q_s)

The prescribed single-helicity island flux amplitude (`01 §1`, ordering O3):

```math
\\tilde\\psi = \\frac{w_\\psi^2}{4}\\,\\frac{q_s'}{q_s},
\\qquad q_s' = dq/d\\psi|_s ,
```

with `w_psi` the island **half**-width in `ψ`-space and `dq_dpsi`, `q_s` the
safety-factor derivative and value at the rational surface. This is a **cleared
physics relation** — human sign-off 2026-07-11, independent re-derivation
(Decision D7): `docs/src/islands/derivations/psi-tilde-amplitude.md`. It
resolves the former `q_s'/q_s` vs `q_s/q_s'` `[VERIFY]` (I19 as printed has a
typo; the physical form is `q_s'/q_s`, consistent with I19's own Ω convention
and Diss19/D21/L23). Feeds the `Δ`-moment prefactor `∓μ₀R/(2 ψ̃)`
([`delta_moments`](@ref); the `μ₀R` and sin normalization remain gated).
"""
function island_flux_amplitude(; w_psi::Real, dq_dpsi::Real, q_s::Real)
    q_s != 0 || throw(ArgumentError("q_s must be nonzero"))
    return (w_psi^2 / 4) * (dq_dpsi / q_s)
end

"""
    parallel_current!(Jpar, gs, species, weights, grid; wy=grid.y.wq, wE=grid.E.weights)

Assemble `J̄_∥(x, ξ) = Σ_j Z_j ∫ W_j g_j` into `Jpar[ix, iξ]` (`01 §4`): one
`weighted_moment!` per species, charge-scaled and accumulated. `gs`, `species`
and `weights` are aligned vectors; each `W_j` is the **cleared** `(ny, nE, nσ)`
parallel-flow velocity weight `W = v̂_∥ = σ√E√(1−y b_min)`
(`Configure.parallel_flow_weight`, sign-off 2026-07-13, `velocity-moment-measure.md`).
`wy`/`wE` are the velocity-moment quadrature weights; pass the physical `∫d³v`
weights (`Configure.physical_velocity_weights`: the `√E/2` speed and
`1/√(1−y b_min)` pitch Jacobians) so `J̄_∥` carries the physical measure — the
`√(1−y b_min)` of `W` then cancels the pitch Jacobian and `J̄_∥` is regular
(`01 §4`). They default to the grid's flat quadrature for manufactured tests.
"""
function parallel_current!(Jpar, gs::AbstractVector, species::AbstractVector{<:Species}, weights::AbstractVector, grid::IslandGrid;
    wy::AbstractVector=grid.y.wq, wE::AbstractVector=grid.E.weights)
    length(gs) == length(species) == length(weights) ||
        throw(ArgumentError("gs, species, weights must be aligned (got $(length(gs)), $(length(species)), $(length(weights)))"))
    fill!(Jpar, zero(eltype(Jpar)))
    for (g, sp, W) in zip(gs, species, weights)
        weighted_moment!(Jpar, g, W, grid; scale=sp.Z, accumulate=true, wy=wy, wE=wE)
    end
    return Jpar
end

"""
    delta_moments(Jpar, grid; prefactor_cos, prefactor_sin)

The two Ampère projections of `J̄_∥` through the island (`01 §4`):

    Δ_cos = prefactor_cos · ∫dx ∮dξ J̄_∥ cos ξ
    Δ_sin = prefactor_sin · ∫dx ∮dξ J̄_∥ sin ξ

`ξ`-integration is the uniform-grid trapezoid (spectrally exact on the periodic
grid); `x`-integration uses the grid's Simpson weights. The prefactors
(`∓μ₀R/2ψ̃`, `01 §4`) are **gated** — `ψ̃` carries an open `[VERIFY]` and the
sin normalization is `[DERIVED]`-unpinned (QUESTIONS Q4) — so both are required
arguments. Returns `(Δcos, Δsin)`.
"""
function delta_moments(Jpar, grid::IslandGrid; prefactor_cos, prefactor_sin)
    nx, nξ = size(Jpar)
    (nx, nξ) == (grid.x.n, grid.ξ.n) || throw(ArgumentError("Jpar must be (nx, nξ) = $((grid.x.n, grid.ξ.n))"))
    wξ = grid.ξ.L / grid.ξ.n
    pc = zero(eltype(Jpar))
    ps = zero(eltype(Jpar))
    @inbounds for iξ in 1:nξ
        c = cos(grid.ξ.nodes[iξ])
        s = sin(grid.ξ.nodes[iξ])
        for ix in 1:nx
            w = grid.x.wq[ix] * wξ * Jpar[ix, iξ]
            pc += w * c
            ps += w * s
        end
    end
    return (Δcos=prefactor_cos * pc, Δsin=prefactor_sin * ps)
end

"""
    omega_label(x, ξ, w)

The island flux-surface label `Ω = 2x²/w² − cos ξ` (pinned convention, module
CLAUDE.md; O-point `Ω = −1`, separatrix `Ω = +1`, `w` = half-width). A
diagnostic label only — never a solve coordinate (Decision D1).
"""
omega_label(x, ξ, w) = 2 * x^2 / w^2 - cos(ξ)

# x on the Ω surface at helical angle ξ (branch = ±1 selects the x-sign).
_x_on_surface(Ω, ξ, w, branch) = branch * (w / sqrt(2)) * sqrt(max(Ω + cos(ξ), 0.0))

"""
    omega_average(f, Ω, w; branch=+1, rtol=1e-10)

Island flux-surface average `⟨f⟩_Ω = ∮ f (Ω + cos ξ)^{−1/2} dξ / ∮ (Ω + cos ξ)^{−1/2} dξ`
(`01 §4` decomposition weight) of a callable `f(x, ξ)`:

  - `Ω > 1` (outside the separatrix): the `±x` branches are distinct surfaces;
    `branch` selects one, integrating over the full `ξ ∈ [0, 2π)`.
  - `−1 < Ω < 1` (inside): the surface closes through both branches between the
    turning points `cos ξ_b = −Ω`; the endpoint weight singularity is integrable
    and handled by the adaptive quadrature.
"""
function omega_average(f, Ω, w; branch::Int=1, rtol::Real=1e-10)
    if Ω > 1
        num, _ = QuadGK.quadgk(ξ -> f(_x_on_surface(Ω, ξ, w, branch), ξ) / sqrt(Ω + cos(ξ)), 0.0, 2π; rtol=rtol)
        den, _ = QuadGK.quadgk(ξ -> 1.0 / sqrt(Ω + cos(ξ)), 0.0, 2π; rtol=rtol)
        return num / den
    elseif -1 < Ω < 1
        ξb = acos(-Ω)
        integrand(ξ) = (f(_x_on_surface(Ω, ξ, w, +1), ξ) + f(_x_on_surface(Ω, ξ, w, -1), ξ)) / sqrt(Ω + cos(ξ))
        num, _ = QuadGK.quadgk(integrand, -ξb, ξb; rtol=rtol)
        den, _ = QuadGK.quadgk(ξ -> 2.0 / sqrt(Ω + cos(ξ)), -ξb, ξb; rtol=rtol)
        return num / den
    else
        throw(DomainError(Ω, "omega_average needs Ω ∈ (−1, 1) ∪ (1, ∞) (O-point/separatrix excluded)"))
    end
end

"""
    channel_split(Jfun, Ω, w; branch=+1)

The `01 §4` bootstrap/polarization bookkeeping at one `Ω` surface: returns
`(bs, pol)` where `bs = ⟨J⟩_Ω` (the flux-surface-constant "bootstrap+curvature"
part) and `pol(x, ξ) = Jfun(x, ξ) − bs` (the piece that `Ω`-averages to zero).
Diagnostic only — the solve never separates channels; the split is approximate
bookkeeping (L23 Eq. 2.5.3 caveat).
"""
function channel_split(Jfun, Ω, w; branch::Int=1)
    bs = omega_average(Jfun, Ω, w; branch=branch)
    return (bs=bs, pol=(x, ξ) -> Jfun(x, ξ) - bs)
end

# ---------------------------------------------------------------------------
# Grid → callable interpolant (for the Ω-surface decomposition diagnostics)
# ---------------------------------------------------------------------------
# Local Lagrange (0th-derivative Fornberg) weights on a window of `npts` nodes
# bracketing `xq`, clamped to the ends of a monotonic non-periodic node vector.
@inline function _lagrange_x(xnodes, xq, npts)
    n = length(xnodes)
    j = searchsortedfirst(xnodes, xq)
    lo = clamp(j - npts ÷ 2, 1, n - npts + 1)
    idx = lo:(lo + npts - 1)
    w = fd_weights(0, view(xnodes, idx), xq)[:, 1]
    return idx, w
end

# Local Lagrange weights on a uniform periodic ξ-grid (nodes `j·h`, `j = 0:nξ-1`,
# period `L`): build the stencil in unwrapped coordinates (monotonic) so Fornberg
# is well-posed, then wrap the sample indices modulo `nξ`.
@inline function _lagrange_ξ(L, nξ, ξq, npts)
    h = L / nξ
    ξw = mod(ξq, L)
    jc = floor(Int, ξw / h)                       # 0-based node just below ξw
    js = (jc - npts ÷ 2):(jc - npts ÷ 2 + npts - 1)
    localnodes = [j * h for j in js]              # unwrapped, monotonic
    w = fd_weights(0, localnodes, ξw)[:, 1]
    idx = [mod(j, nξ) + 1 for j in js]            # wrapped, 1-based
    return idx, w
end

"""
    grid_interpolant(F, grid; npts_x=grid.x.order+1, npts_ξ=min(6, grid.ξ.n))

Return a callable `(x, ξ) -> value` interpolating the grid field `F[ix, iξ]`
(shape `(nx, nξ)`) off the solve grid — the lift from the discrete `J̄_∥` to the
callable the Ω-surface averages ([`omega_average`](@ref), [`channel_split`](@ref))
consume. Separable **local Lagrange** interpolation: an `npts_x`-node Fornberg
stencil (grid FD order) in the clustered `x`, and an `npts_ξ`-node periodic
stencil in the uniform `ξ`. Query `x` is clamped to the grid span. Diagnostic
post-processing — not an allocation-free hot path.
"""
function grid_interpolant(F::AbstractMatrix, grid::IslandGrid; npts_x::Int=grid.x.order + 1, npts_ξ::Int=min(6, grid.ξ.n))
    size(F) == (grid.x.n, grid.ξ.n) || throw(ArgumentError("F must be (nx, nξ) = $((grid.x.n, grid.ξ.n))"))
    xnodes = grid.x.nodes
    L, nξ = grid.ξ.L, grid.ξ.n
    px = clamp(npts_x, 2, grid.x.n)
    pξ = clamp(npts_ξ, 2, nξ)
    xlo, xhi = xnodes[1], xnodes[end]
    return function (xq, ξq)
        ix, wx = _lagrange_x(xnodes, clamp(xq, xlo, xhi), px)
        iξ, wξ = _lagrange_ξ(L, nξ, ξq, pξ)
        s = zero(eltype(F))
        @inbounds for (b, wξb) in zip(iξ, wξ), (a, wxa) in zip(ix, wx)
            s += wxa * wξb * F[a, b]
        end
        return s
    end
end

# Flux-surface-constant value ⟨J⟩_Ω at the node's Ω, with the correct open/closed
# branch. The O-point/separatrix limits (where ⟨·⟩_Ω degenerates or its endpoint
# weight is non-integrable at the resolution of the node) fall back to the node
# value J_node — the flux-surface average of a point surface is the point itself.
@inline function _flux_surface_value(Jfun, J_node, Ω, x, w; rtol, sep_tol)
    (Ω <= -1 + sep_tol || abs(Ω - 1) < sep_tol) && return J_node
    try
        return -1 < Ω < 1 ? omega_average(Jfun, Ω, w; rtol=rtol) :
               omega_average(Jfun, Ω, w; branch=(x >= 0 ? 1 : -1), rtol=rtol)
    catch err
        err isa Union{ErrorException,DomainError} || rethrow(err)
        return J_node                                 # near-separatrix quadrature miss
    end
end

"""
    channel_decomposition(Jpar, grid, w; prefactor_cos, omega_profile=default,
                          rtol=1e-8, sep_tol=1e-3)

Split the growth moment `Δ_neo` of `J̄_∥` into the island-decomposition channels
(`01 §4`; L23 Eq. 2.5.3, an **approximate diagnostic** bookkeeping — the solve
never separates channels). Lifts `Jpar` to a callable with
[`grid_interpolant`](@ref), reconstructs the **flux-surface-constant part**
`J_bs(x, ξ) = ⟨J̄_∥⟩_{Ω(x, ξ)}` at every node (the bootstrap+curvature current,
constant on each `Ω = 2x²/w² − cos ξ` surface), and returns

  - `bootstrap_curvature` — `Δ_bs+Δ_cur`, the `cos ξ` projection of `J_bs`
    (prefactor `prefactor_cos = −μ₀R/2ψ̃`);
  - `polarization` — `Δ_pol = Δ_neo − (Δ_bs+Δ_cur)`, the piece that
    flux-surface-averages to zero;
  - `omega_average_profile` — `(Ω, value)` sampling `⟨J̄_∥⟩_Ω` (outboard branch
    outside the separatrix) at `omega_profile`, the bootstrap-channel profile.

`w = ` island **half**-width (`Ω`-label scale). Diagnostic post-processing:
per-node `Ω`-surface quadrature, not an allocation-free path.
"""
function channel_decomposition(Jpar::AbstractMatrix, grid::IslandGrid, w::Real; prefactor_cos::Real,
    omega_profile::AbstractVector{<:Real}=[-0.8, -0.5, 0.0, 0.5, 0.8, 1.2, 2.0, 5.0, 10.0],
    rtol::Real=1e-8, sep_tol::Real=1e-3)
    nx, nξ = size(Jpar)
    (nx, nξ) == (grid.x.n, grid.ξ.n) || throw(ArgumentError("Jpar must be (nx, nξ) = $((grid.x.n, grid.ξ.n))"))
    Jfun = grid_interpolant(Jpar, grid)
    Jbs = similar(Jpar)
    @inbounds for iξ in 1:nξ
        ξ = grid.ξ.nodes[iξ]
        for ix in 1:nx
            x = grid.x.nodes[ix]
            Ω = omega_label(x, ξ, w)
            Jbs[ix, iξ] = _flux_surface_value(Jfun, Jpar[ix, iξ], Ω, x, w; rtol=rtol, sep_tol=sep_tol)
        end
    end
    zpref = zero(float(prefactor_cos))
    Δneo = delta_moments(Jpar, grid; prefactor_cos=prefactor_cos, prefactor_sin=zpref).Δcos
    Δbs = delta_moments(Jbs, grid; prefactor_cos=prefactor_cos, prefactor_sin=zpref).Δcos
    prof = [Ω <= -1 + sep_tol || abs(Ω - 1) < sep_tol ? NaN :
            (-1 < Ω < 1 ? omega_average(Jfun, Ω, w; rtol=rtol) : omega_average(Jfun, Ω, w; branch=1, rtol=rtol))
            for Ω in omega_profile]
    return (bootstrap_curvature=Δbs, polarization=Δneo - Δbs,
        omega_average_profile=(Ω=collect(float.(omega_profile)), value=prof))
end

end # module Moments
