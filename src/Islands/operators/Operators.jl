"""
    Operators

The Islands operator stack (`03 §2`): the state vector, the residual assembly,
and the `AbstractTerm` family whose sum is the drift-kinetic residual.

**Milestone-M1 status: allocation-free, AD-compatible *structural* stubs.** Each
term implements its discretized differential/algebraic *structure* — which
derivatives act, in which coordinate, with which sign pattern — but takes every
physics coefficient as *supplied data* (`term.a_xi`, `term.c_D`, …). No literal
physics coefficient, sign, or normalization is written here: those carry
`[VERIFY]`/`[CHECKED]` tags (module `CLAUDE.md`) and are populated only after
human clearance. The verification harness (`Verify`) exercises the discretization
with arbitrary manufactured test coefficients, which is legitimate — it tests
the numerics, not the physics.

Operator-stack rules enforced here (`03 §2`, `CLAUDE.md`):

  - terms are independent — no term inspects which others are active;
  - `apply!` is generic over `eltype(U)` (ForwardDiff duals flow through) and
    allocation-free on the hot path (regression-tested);
  - orderings that *remove* structure are phase-space configurations, not terms.

Term ↔ physics map (`03 §2`, `01 §2`): `ParallelStreaming` (island-induced
streaming, `∂ξ` + `∂x`), `MagneticDrift` (precession, `∂ξ`, `:original`/`:improved`
toggle), `ExBDrift` (the `E×B` Poisson bracket in `(x, ξ)`), `Collisions`
(pitch-angle diffusion in `y`), `GradientDrive` (`(v·∇F₀)` source),
`PerpTransport`/`RadiationSink` (Level-4 stubs), and `Quasineutrality` (the
Level-0 field residual, `01 §3`).
"""
module Operators

using LinearAlgebra
import ..PhaseSpace: IslandGrid, nnodes

export IslandState, IslandCache, IslandStack, AbstractTerm
export ParallelStreaming, MagneticDrift, ExBDrift, Collisions, GradientDrive,
    PerpTransport, RadiationSink, Quasineutrality
export PitchAngleDiffusion, conservative_pitch_operator
export CollisionalDrag, NeoclassicalDiffusion, CollisionalCross
export FarFieldConditions, apply_farfield!
export apply!, residual!, velocity_moment!, weighted_moment!, statelength,
    flatten!, unflatten!, g_flat_index, Φ_flat_index

# ---------------------------------------------------------------------------
# State, cache
# ---------------------------------------------------------------------------
"""
    IslandState(g, Φ)

The Level-0 unknowns (`03 §2`): the orbit-averaged distribution
`g[ix, iξ, iy, iE, iσ]` per grid point and the electrostatic potential
`Φ[ix, iξ]`. Parametric in the element type so ForwardDiff duals flow through.

## Fields

  - `g` — 5D array over `(x, ξ, y, E, σ)`.
  - `Φ` — 2D array over `(x, ξ)`.
"""
struct IslandState{T,A5<:AbstractArray{T,5},A2<:AbstractArray{T,2}}
    g::A5
    Φ::A2
end

function IslandState{T}(grid::IslandGrid) where {T}
    nx, nξ, ny, nE, nσ = nnodes(grid)
    return IslandState(zeros(T, nx, nξ, ny, nE, nσ), zeros(T, nx, nξ))
end
IslandState(grid::IslandGrid) = IslandState{Float64}(grid)

Base.eltype(::IslandState{T}) where {T} = T

function Base.similar(U::IslandState{T}) where {T}
    return IslandState(similar(U.g), similar(U.Φ))
end

function fill_state!(U::IslandState, v)
    fill!(U.g, v)
    fill!(U.Φ, v)
    return U
end

"""
    IslandCache{T}(grid)

Per-solve scratch buffers, typed to the state element type `T` so the hot path
allocates nothing (and so AD duals get dual-typed scratch). Currently holds the
two potential-gradient fields consumed by the `E×B` bracket.
"""
struct IslandCache{T}
    dΦdx::Matrix{T}
    dΦdξ::Matrix{T}
end

function IslandCache{T}(grid::IslandGrid) where {T}
    nx, nξ, = nnodes(grid)
    return IslandCache{T}(zeros(T, nx, nξ), zeros(T, nx, nξ))
end
IslandCache(grid::IslandGrid) = IslandCache{Float64}(grid)

# ---------------------------------------------------------------------------
# Term family
# ---------------------------------------------------------------------------
"""
    AbstractTerm

Supertype of every operator-stack term (`03 §2`). Each concrete term implements
`apply!(R, term, U, grid, cache)` to accumulate its contribution to the residual.
Terms are independent (no term inspects which others are active), generic over
`eltype(U)`, and allocation-free on the hot path.
"""
abstract type AbstractTerm end

"""
    ParallelStreaming(a_xi, a_x)

Island-induced parallel streaming (`01 §2`, the `∂ξ`/`∂x` channel): adds
`a_xi ∂g/∂ξ + a_x ∂g/∂x` to the residual. `a_xi`, `a_x` are supplied coefficient
arrays shaped like `g` (physics values `[VERIFY]`-gated; never literals here).
"""
struct ParallelStreaming{A} <: AbstractTerm
    a_xi::A
    a_x::A
end

"""
    MagneticDrift(c_D; variant=:original)

Orbit-averaged magnetic (precession) drift (`01 §2.1`): adds `c_D ∂g/∂ξ`. The
`variant` selects the `:original` (finite `L̂_B⁻¹`, I19) vs `:improved`
(`L̂_B⁻¹ → 0` proxy, D21) drift-frequency structure — the 8.73→1.46 ρ_bi toggle
(`docs/05 E1`). Structure only: `c_D` (the `ω̂_D` coefficient over `(y, E, σ)`)
is supplied data.
"""
struct MagneticDrift{A} <: AbstractTerm
    c_D::A
    variant::Symbol
end
MagneticDrift(c_D; variant::Symbol=:original) = MagneticDrift(c_D, variant)

"""
    ExBDrift(c_E)

`E×B` advection as the Poisson bracket of `Φ` and `g` in `(x, ξ)`
(`01 §2`): adds `c_E (∂Φ/∂ξ · ∂g/∂x − ∂Φ/∂x · ∂g/∂ξ)`. This is the one Level-0
kinetic term nonlinear in the state (couples `g` and `Φ`). `c_E` is either a
supplied **scalar** (manufactured tests) or a **velocity-dependent array** shaped
like `g` — the cleared coupling `c_E = ½⟨1/v̂_∥⟩_θ` depends on `(y, E, σ)`
(passing σ-odd, trapped ≡ 0; `01 §2`, derivation `exb-coupling.md`).
"""
struct ExBDrift{S} <: AbstractTerm
    c_E::S
end

# c_E may be a scalar (tests) or a (x,ξ,y,E,σ) array (the cleared velocity-
# dependent coupling); dispatch keeps the hot path allocation-free either way.
@inline _cE_at(cE::Number, ix, iξ, iy, iE, iσ) = cE
@inline _cE_at(cE::AbstractArray, ix, iξ, iy, iE, iσ) = @inbounds cE[ix, iξ, iy, iE, iσ]

"""
    Collisions(a_y, b_y; model=:pitch_angle)

Pitch-angle (Lorentz) collision operator (`01 §2.3`) as a second-order operator
in `y`: adds `a_y ∂²g/∂y² + b_y ∂g/∂y`. `model = :pitch_angle` is the Level-0
form; `:fokker_planck` (Level 1) reuses the same slot. `a_y`, `b_y` are supplied
coefficient arrays (the `ν̂`-weighted diffusion structure, `[VERIFY]`-gated).
"""
struct Collisions{A} <: AbstractTerm
    a_y::A
    b_y::A
    model::Symbol
end
Collisions(a_y, b_y; model::Symbol=:pitch_angle) = Collisions(a_y, b_y, model)

"""
    PitchAngleDiffusion(K, c)

Pitch-angle collision operator in **discretely conservative (mimetic) form**
(`01 §2.3` structure; ladder A4): adds `c ⋅ (K g)` along `y`, where `K` is the
divergence-form matrix built by [`conservative_pitch_operator`](@ref) and `c` is
a supplied coefficient array over `(x, ξ, E, σ)` — **`y`-independent by
construction**, so the exact discrete particle conservation and entropy-sign
properties of `K` are preserved (the physical `ν̂(v̂)` energy dependence lives in
`c`'s `E`-dependence and is `[VERIFY]`-gated). The momentum-restoring
field-particle piece of the Level-0 operator is gated physics (QUESTIONS Q3)
and is not part of this structure.
"""
struct PitchAngleDiffusion{M<:AbstractMatrix,A} <: AbstractTerm
    K::M
    c::A
end

"""
    conservative_pitch_operator(ygrid, P, wmeas)

Build the mimetic divergence-form pitch operator on `ygrid`
(`C[g] = (1/w) d/dy(P w dg/dy)` discretized as `K = −Wq⁻¹ Gᵀ diag(P .* wq) G`
with `G = ygrid.D1` and `Wq = wmeas .* ygrid.wq`), returning `(K, Wq)`.

Because `G` differentiates constants exactly and the quadrature weights are
positive, `K` satisfies **exactly** (to machine precision, ladder A4):

  - particle conservation: `Wqᵀ (K g) = 0` for any `g` (zero-flux built in);
  - entropy sign: `gᵀ diag(Wq) K g = −(Gg)ᵀ diag(P .* wq)(Gg) ≤ 0` for `P ≥ 0`.

`P` (the pitch-space diffusivity profile, physically the `λ√(1−λB)`-structure)
and `wmeas` (the velocity-space measure) are **supplied** profiles — their
Level-0 physics forms are `[VERIFY]`-gated (QUESTIONS Q3); any positive test
profiles exercise the conservation structure.
"""
function conservative_pitch_operator(ygrid, P::AbstractVector, wmeas::AbstractVector)
    length(P) == ygrid.n || throw(ArgumentError("P must have length $(ygrid.n)"))
    length(wmeas) == ygrid.n || throw(ArgumentError("wmeas must have length $(ygrid.n)"))
    all(>=(0), P) || throw(ArgumentError("diffusivity profile P must be nonnegative"))
    all(>(0), wmeas) || throw(ArgumentError("measure weights wmeas must be positive"))
    G = ygrid.D1
    Wq = wmeas .* ygrid.wq
    K = -Diagonal(1.0 ./ Wq) * (G' * Diagonal(P .* ygrid.wq) * G)
    return Matrix(K), Wq
end

"""
    CollisionalDrag(a_x)

The orbit-averaged collision `∂_p` drag (term A, `01 §2.3`;
`orbit-averaged-collision.md`): adds `a_x ∂g/∂x`, with `a_x` a supplied
coefficient array shaped like `g` (`−ν̂_ii/m`, passing-only `Θ_y`). Structurally
an `∂_x` advection, distinct from the island `ParallelStreaming` channel.
"""
struct CollisionalDrag{A} <: AbstractTerm
    a_x::A
end

"""
    NeoclassicalDiffusion(c)

The neoclassical `∂²_p` diffusion (term B, `01 §2.3`;
`orbit-averaged-collision.md`) — the only Level-0 term giving cross-`pφ` (radial)
transport: adds `c ∂²g/∂x²`, with `c` a supplied `(x,ξ,y,E,σ)` coefficient array
(`−(ν̂_ii σu ρ̂_θ/2m(1+ε)) y⟨1/√(1−yb)⟩_θ`). Distinct from the Level-4
`PerpTransport` stub (scalar `χ`); here `c` is the cleared collision coefficient.
"""
struct NeoclassicalDiffusion{A} <: AbstractTerm
    c::A
end

"""
    CollisionalCross(c)

The collision `∂²_{yp}` cross term (term C, `01 §2.3`;
`orbit-averaged-collision.md`): adds `c ∂²g/∂x∂y`, with `c` a supplied
`(x,ξ,y,E,σ)` coefficient array (`−2ν̂_ii y/m`, passing-only `Θ_y`). The mixed
`x`,`y` second derivative couples the radial and pitch grids.
"""
struct CollisionalCross{A} <: AbstractTerm
    c::A
end

"""
    GradientDrive(drive)

The `(v_E + v_D + v_ψ̃)·∇F₀` gradient drive (`03 §2`, `01 §2`): a state-independent
source, adds `drive` to the residual. `drive` is a supplied array shaped like
`g`; at Level 0 it is built from the background Maxwellian gradients (`[VERIFY]`).
"""
struct GradientDrive{A} <: AbstractTerm
    drive::A
end

"""
    PerpTransport(χ)

Perpendicular transport (Level-4 closure stub, `03 §2`): adds `χ ∂²g/∂x²`.
Present as structure only; the closure value `χ` is a supplied knob.
"""
struct PerpTransport{S} <: AbstractTerm
    χ::S
end

"""
    RadiationSink(κ)

Radiative energy sink (Level-4 closure stub, `03 §2`): adds `-κ g`. Structure
only; `κ` is a supplied coefficient array.
"""
struct RadiationSink{A} <: AbstractTerm
    κ::A
end

"""
    Quasineutrality(α)
    Quasineutrality(α, source)

The Level-0 field residual (`01 §3`): `R_Φ = M[g] − α Φ + source`, where `M[g]`
is the velocity moment `∫dy ∫dE Σ_σ g` (Gauss in `E`, Simpson in `y`), `α` is
the adiabatic-shielding coefficient `(τ+1)/τ`, and `source` is the
`L̂_{n0}⁻¹(x − ĥ(Ω))` flattened-electron drive of the cleared closure (`01 §3`,
`quasineutrality-closure.md`). Both come from the **human-cleared** closure
(sign-off 2026-07-11) — built by `Configure.configure_level0` from
`Coefficients.quasineutrality_coefficient` and the `ĥ` profile. `source`
defaults to `nothing` (a pure `M[g] − α Φ` residual) for the manufactured-test
and pre-drive configurations. This closes `Φ(x, ξ)` inside the global Newton
system rather than the sources' fragile nested Picard loop (`01 §3`, `03 §3`).

The density moment `M[g] = δn̄_i` uses the **physical `∫d³v` measure** when `wy`/`wE`
are supplied (`Configure.physical_velocity_weights`: the `√E/2` speed and
`1/√(1−y b_min)` pitch Jacobians, `velocity-moment-measure.md`); they default to
`nothing` (the flat quadrature) for manufactured tests.

## Fields

  - `α`      — adiabatic-shielding coefficient multiplying `Φ` (`(τ+1)/τ`).
  - `source` — the `(nx, nξ)` `L̂_{n0}⁻¹(x − ĥ)` field drive, or `nothing`.
  - `wy`, `wE` — physical velocity-moment weights for `M[g]`, or `nothing` (flat).
"""
struct Quasineutrality{S,A,W} <: AbstractTerm
    α::S
    source::A
    wy::W
    wE::W
end
Quasineutrality(α) = Quasineutrality(α, nothing, nothing, nothing)
Quasineutrality(α, source) = Quasineutrality(α, source, nothing, nothing)

# ---------------------------------------------------------------------------
# Discrete kernels (allocation-free, generic over eltype)
#
# Directional-derivative kernels accumulate `coef · ∂g/∂dir` into `Rg`. The
# dense matrices `D` are Float64; `D·g` with `g::Dual` promotes correctly, so
# the whole stack is AD-transparent.
# ---------------------------------------------------------------------------
# adds coef .* (∂g/∂ξ) along dim 2
@inline function _add_dxi!(Rg, coef, g, D)
    nx, nξ, ny, nE, nσ = size(g)
    @inbounds for iσ in 1:nσ, iE in 1:nE, iy in 1:ny, ix in 1:nx
        for a in 1:nξ
            acc = zero(eltype(Rg))
            for b in 1:nξ
                acc += D[a, b] * g[ix, b, iy, iE, iσ]
            end
            Rg[ix, a, iy, iE, iσ] += coef[ix, a, iy, iE, iσ] * acc
        end
    end
    return Rg
end

# adds coef .* (∂g/∂x) along dim 1
@inline function _add_dx!(Rg, coef, g, D)
    nx, nξ, ny, nE, nσ = size(g)
    @inbounds for iσ in 1:nσ, iE in 1:nE, iy in 1:ny, iξ in 1:nξ
        for a in 1:nx
            acc = zero(eltype(Rg))
            for b in 1:nx
                acc += D[a, b] * g[b, iξ, iy, iE, iσ]
            end
            Rg[a, iξ, iy, iE, iσ] += coef[a, iξ, iy, iE, iσ] * acc
        end
    end
    return Rg
end

# adds coef .* (Dⁿ g along dim 3, y).  D is D1 or D2.
@inline function _add_dy!(Rg, coef, g, D)
    nx, nξ, ny, nE, nσ = size(g)
    @inbounds for iσ in 1:nσ, iE in 1:nE, iξ in 1:nξ, ix in 1:nx
        for a in 1:ny
            acc = zero(eltype(Rg))
            for b in 1:ny
                acc += D[a, b] * g[ix, iξ, b, iE, iσ]
            end
            Rg[ix, iξ, a, iE, iσ] += coef[ix, iξ, a, iE, iσ] * acc
        end
    end
    return Rg
end

# scalar-coefficient ∂²/∂x² (PerpTransport): adds χ .* D2x g along dim 1
@inline function _add_dx2_scalar!(Rg, χ, g, D)
    nx, nξ, ny, nE, nσ = size(g)
    @inbounds for iσ in 1:nσ, iE in 1:nE, iy in 1:ny, iξ in 1:nξ
        for a in 1:nx
            acc = zero(eltype(Rg))
            for b in 1:nx
                acc += D[a, b] * g[b, iξ, iy, iE, iσ]
            end
            Rg[a, iξ, iy, iE, iσ] += χ * acc
        end
    end
    return Rg
end

# array-coefficient ∂²/∂x² (NeoclassicalDiffusion): adds coef .* D2x g along dim 1
@inline function _add_dx2!(Rg, coef, g, D)
    nx, nξ, ny, nE, nσ = size(g)
    @inbounds for iσ in 1:nσ, iE in 1:nE, iy in 1:ny, iξ in 1:nξ
        for a in 1:nx
            acc = zero(eltype(Rg))
            for b in 1:nx
                acc += D[a, b] * g[b, iξ, iy, iE, iσ]
            end
            Rg[a, iξ, iy, iE, iσ] += coef[a, iξ, iy, iE, iσ] * acc
        end
    end
    return Rg
end

# mixed ∂²/∂x∂y (CollisionalCross): adds coef .* ∂x(∂y g); apply Dy (dim 3) then Dx (dim 1)
@inline function _add_dxy!(Rg, coef, g, Dx, Dy)
    nx, nξ, ny, nE, nσ = size(g)
    @inbounds for iσ in 1:nσ, iE in 1:nE, iξ in 1:nξ, iy in 1:ny, ix in 1:nx
        acc = zero(eltype(Rg))
        for bx in 1:nx
            dy = zero(eltype(Rg))
            for by in 1:ny
                dy += Dy[iy, by] * g[bx, iξ, by, iE, iσ]
            end
            acc += Dx[ix, bx] * dy
        end
        Rg[ix, iξ, iy, iE, iσ] += coef[ix, iξ, iy, iE, iσ] * acc
    end
    return Rg
end

# ∂Φ/∂x and ∂Φ/∂ξ into cache buffers (allocation-free)
@inline function _potential_gradients!(cache::IslandCache, Φ, Dx, Dξ)
    nx, nξ = size(Φ)
    dΦdx, dΦdξ = cache.dΦdx, cache.dΦdξ
    @inbounds for iξ in 1:nξ, ix in 1:nx
        ax = zero(eltype(Φ))
        for b in 1:nx
            ax += Dx[ix, b] * Φ[b, iξ]
        end
        dΦdx[ix, iξ] = ax
    end
    @inbounds for ix in 1:nx, iξ in 1:nξ
        aξ = zero(eltype(Φ))
        for b in 1:nξ
            aξ += Dξ[iξ, b] * Φ[ix, b]
        end
        dΦdξ[ix, iξ] = aξ
    end
    return cache
end

# ---------------------------------------------------------------------------
# apply!(R, term, U, grid, cache) — accumulate the term's contribution.
# ---------------------------------------------------------------------------
"""
    apply!(R, term, U, grid, cache)

Accumulate `term`'s contribution to the residual `R` at state `U` on `grid`,
using `cache` for scratch. Each `AbstractTerm` defines a method; the assembly in
[`residual!`](@ref) walks the stack. Allocation-free and generic over `eltype(U)`
so ForwardDiff duals flow through (verification ladder A2).
"""
function apply!(R::IslandState, t::ParallelStreaming, U::IslandState, grid::IslandGrid, ::IslandCache)
    _add_dxi!(R.g, t.a_xi, U.g, grid.ξ.D1)
    _add_dx!(R.g, t.a_x, U.g, grid.x.D1)
    return R
end

function apply!(R::IslandState, t::MagneticDrift, U::IslandState, grid::IslandGrid, ::IslandCache)
    _add_dxi!(R.g, t.c_D, U.g, grid.ξ.D1)
    return R
end

function apply!(R::IslandState, t::Collisions, U::IslandState, grid::IslandGrid, ::IslandCache)
    _add_dy!(R.g, t.a_y, U.g, grid.y.D2)
    _add_dy!(R.g, t.b_y, U.g, grid.y.D1)
    return R
end

function apply!(R::IslandState, t::PitchAngleDiffusion, U::IslandState, grid::IslandGrid, ::IslandCache)
    K = t.K
    c = t.c
    g = U.g
    nx, nξ, ny, nE, nσ = size(g)
    @inbounds for iσ in 1:nσ, iE in 1:nE, iξ in 1:nξ, ix in 1:nx
        cv = c[ix, iξ, iE, iσ]
        for a in 1:ny
            acc = zero(eltype(R.g))
            for b in 1:ny
                acc += K[a, b] * g[ix, iξ, b, iE, iσ]
            end
            R.g[ix, iξ, a, iE, iσ] += cv * acc
        end
    end
    return R
end

function apply!(R::IslandState, t::GradientDrive, U::IslandState, ::IslandGrid, ::IslandCache)
    @inbounds @. R.g += t.drive
    return R
end

function apply!(R::IslandState, t::PerpTransport, U::IslandState, grid::IslandGrid, ::IslandCache)
    _add_dx2_scalar!(R.g, t.χ, U.g, grid.x.D2)
    return R
end

function apply!(R::IslandState, t::CollisionalDrag, U::IslandState, grid::IslandGrid, ::IslandCache)
    _add_dx!(R.g, t.a_x, U.g, grid.x.D1)
    return R
end

function apply!(R::IslandState, t::NeoclassicalDiffusion, U::IslandState, grid::IslandGrid, ::IslandCache)
    _add_dx2!(R.g, t.c, U.g, grid.x.D2)
    return R
end

function apply!(R::IslandState, t::CollisionalCross, U::IslandState, grid::IslandGrid, ::IslandCache)
    _add_dxy!(R.g, t.c, U.g, grid.x.D1, grid.y.D1)
    return R
end

function apply!(R::IslandState, t::RadiationSink, U::IslandState, ::IslandGrid, ::IslandCache)
    @inbounds @. R.g += -t.κ * U.g
    return R
end

function apply!(R::IslandState, t::ExBDrift, U::IslandState, grid::IslandGrid, cache::IslandCache)
    _potential_gradients!(cache, U.Φ, grid.x.D1, grid.ξ.D1)
    dΦdx, dΦdξ = cache.dΦdx, cache.dΦdξ
    g = U.g
    Dx, Dξ = grid.x.D1, grid.ξ.D1
    cE = t.c_E
    nx, nξ, ny, nE, nσ = size(g)
    @inbounds for iσ in 1:nσ, iE in 1:nE, iy in 1:ny, iξ in 1:nξ, ix in 1:nx
        dgdx = zero(eltype(R.g))
        for b in 1:nx
            dgdx += Dx[ix, b] * g[b, iξ, iy, iE, iσ]
        end
        dgdξ = zero(eltype(R.g))
        for b in 1:nξ
            dgdξ += Dξ[iξ, b] * g[ix, b, iy, iE, iσ]
        end
        cEv = _cE_at(cE, ix, iξ, iy, iE, iσ)
        R.g[ix, iξ, iy, iE, iσ] += cEv * (dΦdξ[ix, iξ] * dgdx - dΦdx[ix, iξ] * dgdξ)
    end
    return R
end

function apply!(R::IslandState, t::Quasineutrality, U::IslandState, grid::IslandGrid, ::IslandCache)
    if t.wy === nothing
        velocity_moment!(R.Φ, U.g, grid; accumulate=true)
    else
        velocity_moment!(R.Φ, U.g, grid; accumulate=true, wy=t.wy, wE=t.wE)
    end
    @inbounds @. R.Φ += -t.α * U.Φ
    t.source === nothing || @inbounds(@. R.Φ += t.source)
    return R
end

# ---------------------------------------------------------------------------
# Velocity moment  M[g](x,ξ) = ∫dy ∫dE Σ_σ g   (Simpson in y, Gauss in E).
# ---------------------------------------------------------------------------
"""
    velocity_moment!(M, g, grid; accumulate=false, wy=grid.y.wq, wE=grid.E.weights)

Accumulate the phase-space velocity moment `∫dy ∫dE Σ_σ g` into `M[ix, iξ]`. The
`wy`/`wE` weights default to the grid's flat quadrature (Simpson `y`, Gauss `E`);
pass the **physical** `∫d³v` weights (`Configure.physical_velocity_weights`: the
`√E/2` speed Jacobian and the `1/√(1−y b_min)` pitch Jacobian, `velocity-moment-measure.md`)
to get the physical moment (`01 §4`). `accumulate=true` adds into `M`.
"""
function velocity_moment!(M, g, grid::IslandGrid; accumulate::Bool=false,
    wy::AbstractVector=grid.y.wq, wE::AbstractVector=grid.E.weights)
    accumulate || fill!(M, zero(eltype(M)))
    nx, nξ, ny, nE, nσ = size(g)
    @inbounds for iσ in 1:nσ, iE in 1:nE, iy in 1:ny
        w = wy[iy] * wE[iE]
        for iξ in 1:nξ, ix in 1:nx
            M[ix, iξ] += w * g[ix, iξ, iy, iE, iσ]
        end
    end
    return M
end

"""
    weighted_moment!(M, g, W, grid; scale=1, accumulate=false, wy=grid.y.wq, wE=grid.E.weights)

Accumulate the weighted velocity moment `scale · ∫dy ∫dE Σ_σ W(y, E, σ) g` into
`M[ix, iξ]` (`03 §2`, `01 §4`). `W` is the `(ny, nE, nσ)` velocity weight — for the
parallel flow `W = v̂_∥ = σ√E√(1−yb)` (`Configure.parallel_flow_weight`, cleared
`velocity-moment-measure.md`). `wy`/`wE` default to the grid's flat quadrature;
pass the physical `∫d³v` weights for the physical moment. `scale` carries a
per-species factor (charge `Z_j`). `velocity_moment!` is the `W ≡ 1` case.
"""
function weighted_moment!(M, g, W, grid::IslandGrid; scale=1, accumulate::Bool=false,
    wy::AbstractVector=grid.y.wq, wE::AbstractVector=grid.E.weights)
    accumulate || fill!(M, zero(eltype(M)))
    nx, nξ, ny, nE, nσ = size(g)
    size(W) == (ny, nE, nσ) || throw(ArgumentError("weight W must have size (ny, nE, nσ) = $((ny, nE, nσ))"))
    @inbounds for iσ in 1:nσ, iE in 1:nE, iy in 1:ny
        w = scale * wy[iy] * wE[iE] * W[iy, iE, iσ]
        for iξ in 1:nξ, ix in 1:nx
            M[ix, iξ] += w * g[ix, iξ, iy, iE, iσ]
        end
    end
    return M
end

# ---------------------------------------------------------------------------
# Far-field boundary conditions (01 §3, 04 §1): match g and Φ to supplied
# far-field states at |x| = L_x. NEVER bare Neumann — L23 traced its spurious
# "winged" solution branch to Neumann non-uniqueness; the physical far field is
# the neoclassical (no-island) solution, which is [VERIFY]-gated physics and
# therefore *supplied* here, not computed.
# ---------------------------------------------------------------------------
"""
    FarFieldConditions(g_left, g_right, Φ_left, Φ_right)

Dirichlet-type far-field matching data at the two radial boundaries (`01 §3`):
`g → g_far` and `Φ̂ → Φ̂_far` as `|x| → L_x`. The residual rows at `ix = 1, nx`
are replaced by `(U − far)` so the Newton solve pins the far field. The
neoclassical no-island `g_far` is gated physics (QUESTIONS Q3) — callers supply
it (tests use manufactured far fields).

Also carries the **forbidden-pitch domain condition**: where `forbidden_y[iy]`
(no valid orbit, `y ≥ 1/b_min` — no particles), `g` is pinned to `0` at **all**
`ix` (not just the boundaries), since every physics coefficient vanishes there and
the node would otherwise be unconstrained (`01 §2.3`, `orbit-averaged-collision.md`).

## Fields

  - `g_left`, `g_right` — `(nξ, ny, nE, nσ)` far-field distributions at `ix = 1`, `ix = nx`.
  - `Φ_left`, `Φ_right` — length-`nξ` far-field potentials at `ix = 1`, `ix = nx`.
  - `forbidden_y` — length-`ny` `Bool` mask; `true` pins `g = 0` at that pitch node
    for all `ix` (the forbidden region). Defaults to all-`false` (no pinning) for
    the 4-argument constructor used by manufactured tests.
"""
struct FarFieldConditions{T,A4<:AbstractArray{T,4},V<:AbstractVector{T}}
    g_left::A4
    g_right::A4
    Φ_left::V
    Φ_right::V
    forbidden_y::Vector{Bool}
end

FarFieldConditions(g_left, g_right, Φ_left, Φ_right) =
    FarFieldConditions(g_left, g_right, Φ_left, Φ_right, fill(false, size(g_left, 2)))

"""
    apply_farfield!(R, U, bc, grid)

Replace the residual rows at the radial boundaries with the far-field matching
conditions `U − far` (`01 §3`). Called after the operator stack in
[`residual!`](@ref); allocation-free and AD-transparent.
"""
function apply_farfield!(R::IslandState, U::IslandState, bc::FarFieldConditions, grid::IslandGrid)
    nx, nξ, ny, nE, nσ = size(U.g)
    @inbounds for iσ in 1:nσ, iE in 1:nE, iy in 1:ny, iξ in 1:nξ
        R.g[1, iξ, iy, iE, iσ] = U.g[1, iξ, iy, iE, iσ] - bc.g_left[iξ, iy, iE, iσ]
        R.g[nx, iξ, iy, iE, iσ] = U.g[nx, iξ, iy, iE, iσ] - bc.g_right[iξ, iy, iE, iσ]
    end
    # forbidden-pitch domain condition: pin g = 0 (no particles) at all ix
    @inbounds for iσ in 1:nσ, iE in 1:nE, iy in 1:ny
        bc.forbidden_y[iy] || continue
        for iξ in 1:nξ, ix in 1:nx
            R.g[ix, iξ, iy, iE, iσ] = U.g[ix, iξ, iy, iE, iσ]
        end
    end
    @inbounds for iξ in 1:nξ
        R.Φ[1, iξ] = U.Φ[1, iξ] - bc.Φ_left[iξ]
        R.Φ[nx, iξ] = U.Φ[nx, iξ] - bc.Φ_right[iξ]
    end
    return R
end

# ---------------------------------------------------------------------------
# Stack + residual assembly
# ---------------------------------------------------------------------------
"""
    IslandStack(kinetic, field)

A named configuration (`03 §2`): the tuple of kinetic terms acting on `R.g` and
the `field` term (`Quasineutrality`) closing `R.Φ`. Stored as a tuple for type
stability so `residual!` stays allocation-free.

## Fields

  - `kinetic` — tuple of `AbstractTerm`s contributing to the kinetic residual.
  - `field`   — the field-equation term.
"""
struct IslandStack{K<:Tuple,F<:AbstractTerm}
    kinetic::K
    field::F
end

"""
    residual!(R, U, stack, grid, cache)

Assemble the full residual `R = (R_g, R_Φ)` in place: zero `R`, apply every
kinetic term, then the field term. Allocation-free and AD-transparent.
"""
function residual!(R::IslandState, U::IslandState, stack::IslandStack, grid::IslandGrid, cache::IslandCache)
    fill_state!(R, zero(eltype(R)))
    _apply_kinetic!(R, stack.kinetic, U, grid, cache)
    apply!(R, stack.field, U, grid, cache)
    return R
end

"""
    residual!(R, U, stack, grid, cache, bc)

Residual assembly with far-field boundary conditions: assemble the stack, then
replace the radial-boundary rows with the `FarFieldConditions` matching
residual (`01 §3`).
"""
function residual!(R::IslandState, U::IslandState, stack::IslandStack, grid::IslandGrid, cache::IslandCache, bc::FarFieldConditions)
    residual!(R, U, stack, grid, cache)
    apply_farfield!(R, U, bc, grid)
    return R
end

# recursive tuple walk keeps the loop type-stable (no dynamic dispatch, no alloc)
@inline _apply_kinetic!(R, ::Tuple{}, U, grid, cache) = R
@inline function _apply_kinetic!(R, terms::Tuple, U, grid, cache)
    apply!(R, terms[1], U, grid, cache)
    return _apply_kinetic!(R, Base.tail(terms), U, grid, cache)
end

# ---------------------------------------------------------------------------
# Flatten / unflatten for the Newton–Krylov / JVP interface
# ---------------------------------------------------------------------------
"""
    statelength(grid)

Number of scalar unknowns in the flattened state (`g` plus `Φ`).
"""
function statelength(grid::IslandGrid)
    nx, nξ, ny, nE, nσ = nnodes(grid)
    return nx * nξ * ny * nE * nσ + nx * nξ
end

"""
    flatten!(v, U)

Copy state `U` into the flat vector `v` (`g` block then `Φ` block).
"""
function flatten!(v, U::IslandState)
    ng = length(U.g)
    copyto!(view(v, 1:ng), vec(U.g))
    copyto!(view(v, (ng+1):(ng+length(U.Φ))), vec(U.Φ))
    return v
end

"""
    unflatten!(U, v)

Copy the flat vector `v` back into state `U`.
"""
function unflatten!(U::IslandState, v)
    ng = length(U.g)
    copyto!(vec(U.g), view(v, 1:ng))
    copyto!(vec(U.Φ), view(v, (ng+1):(ng+length(U.Φ))))
    return U
end

"""
    g_flat_index(grid, ix, iξ, iy, iE, iσ)

Flat-state-vector index of `g[ix, iξ, iy, iE, iσ]` under the `flatten!` layout
(`g` block column-major, then `Φ`). Used by the `y_c`-block conditioning
monitor (ladder A8) to address pitch-pencil sub-blocks of the dense Jacobian.
"""
function g_flat_index(grid::IslandGrid, ix::Int, iξ::Int, iy::Int, iE::Int, iσ::Int)
    nx, nξ, ny, nE, = nnodes(grid)
    return ix + nx * ((iξ - 1) + nξ * ((iy - 1) + ny * ((iE - 1) + nE * (iσ - 1))))
end

"""
    Φ_flat_index(grid, ix, iξ)

Flat-state-vector index of `Φ[ix, iξ]` under the `flatten!` layout.
"""
function Φ_flat_index(grid::IslandGrid, ix::Int, iξ::Int)
    nx, nξ, ny, nE, nσ = nnodes(grid)
    return nx * nξ * ny * nE * nσ + ix + nx * (iξ - 1)
end

end # module Operators
