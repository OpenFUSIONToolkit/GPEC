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
export apply!, residual!, velocity_moment!, statelength, flatten!, unflatten!

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
kinetic term nonlinear in the state (couples `g` and `Φ`); `c_E` is a supplied
scalar coupling.
"""
struct ExBDrift{S} <: AbstractTerm
    c_E::S
end

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

The Level-0 field residual (`01 §3`): `R_Φ = M[g] − α Φ`, where `M[g]` is the
velocity moment `∫dy ∫dE Σ_σ g` (Gauss in `E`, Simpson in `y`) and `α` encodes
the flattened-electron closure scaling (structure of `e_iΦ̂/T_i/(2 L̂_{n0})`,
supplied). This closes `Φ(x, ξ)` inside the global Newton system rather than the
sources' fragile nested Picard loop (`01 §3`, `03 §3`).
"""
struct Quasineutrality{S} <: AbstractTerm
    α::S
end

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

function apply!(R::IslandState, t::GradientDrive, U::IslandState, ::IslandGrid, ::IslandCache)
    @inbounds @. R.g += t.drive
    return R
end

function apply!(R::IslandState, t::PerpTransport, U::IslandState, grid::IslandGrid, ::IslandCache)
    _add_dx2_scalar!(R.g, t.χ, U.g, grid.x.D2)
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
        R.g[ix, iξ, iy, iE, iσ] += cE * (dΦdξ[ix, iξ] * dgdx - dΦdx[ix, iξ] * dgdξ)
    end
    return R
end

function apply!(R::IslandState, t::Quasineutrality, U::IslandState, grid::IslandGrid, ::IslandCache)
    velocity_moment!(R.Φ, U.g, grid; accumulate=true)
    @inbounds @. R.Φ += -t.α * U.Φ
    return R
end

# ---------------------------------------------------------------------------
# Velocity moment  M[g](x,ξ) = ∫dy ∫dE Σ_σ g   (Simpson in y, Gauss in E).
# ---------------------------------------------------------------------------
"""
    velocity_moment!(M, g, grid; accumulate=false)

Accumulate the phase-space velocity moment `∫dy ∫dE Σ_σ g` into `M[ix, iξ]`
using the Simpson `y`-weights and Gauss `E`-weights of `grid` (`03 §2`). Set
`accumulate=true` to add into `M` (residual assembly); otherwise `M` is zeroed
first.
"""
function velocity_moment!(M, g, grid::IslandGrid; accumulate::Bool=false)
    accumulate || fill!(M, zero(eltype(M)))
    wy = grid.y.wq
    wE = grid.E.weights
    nx, nξ, ny, nE, nσ = size(g)
    @inbounds for iσ in 1:nσ, iE in 1:nE, iy in 1:ny
        w = wy[iy] * wE[iE]
        for iξ in 1:nξ, ix in 1:nx
            M[ix, iξ] += w * g[ix, iξ, iy, iE, iσ]
        end
    end
    return M
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

end # module Operators
