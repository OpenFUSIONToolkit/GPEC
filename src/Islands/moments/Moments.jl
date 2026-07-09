"""
    Moments

Output-moment assembly (design `01 §4`, `03 §1`): the parallel current
`J̄_∥(x, ξ)` from species-summed velocity moments, its `cos ξ`/`sin ξ` Ampère
projections `Δ_cos`/`Δ_sin`, and the island flux-surface-average diagnostics
(`Ω` label, `⟨·⟩_Ω`, bootstrap/polarization channel split).

**Gating:** the projection and quadrature machinery here is pure numerics. The
physics enters through (i) the per-species velocity-space weights `W_j` (the
`v̂_∥`-structure — `[VERIFY]`-gated, QUESTIONS Q3), and (ii) the `Δ` moment
prefactors (`±μ₀R/2ψ̃` with `ψ̃` under an open `[VERIFY]`, and the sin-moment
normalization still `[DERIVED]`-unpinned — QUESTIONS Q4). Both are **required
caller-supplied arguments** with no defaults; nothing here assigns them.

The island label convention is the module-CLAUDE.md pin: `Ω = 2x²/w² − cos ξ`,
O-point `Ω = −1`, separatrix `Ω = +1`, `w` = **half**-width.
"""
module Moments

using LinearAlgebra
import QuadGK
import ..PhaseSpace: IslandGrid, nnodes
import ..Operators: weighted_moment!
import ..SpeciesLists: Species

export parallel_current!, delta_moments, omega_label, omega_average, channel_split

"""
    parallel_current!(Jpar, gs, species, weights, grid)

Assemble `J̄_∥(x, ξ) = Σ_j Z_j ∫ W_j g_j` into `Jpar[ix, iξ]` (`01 §4`): one
`weighted_moment!` per species, charge-scaled and accumulated. `gs`, `species`
and `weights` are aligned vectors; each `W_j` is the supplied `(ny, nE, nσ)`
parallel-flow velocity weight (gated physics, QUESTIONS Q3).
"""
function parallel_current!(Jpar, gs::AbstractVector, species::AbstractVector{<:Species}, weights::AbstractVector, grid::IslandGrid)
    length(gs) == length(species) == length(weights) ||
        throw(ArgumentError("gs, species, weights must be aligned (got $(length(gs)), $(length(species)), $(length(weights)))"))
    fill!(Jpar, zero(eltype(Jpar)))
    for (g, sp, W) in zip(gs, species, weights)
        weighted_moment!(Jpar, g, W, grid; scale=sp.Z, accumulate=true)
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

end # module Moments
