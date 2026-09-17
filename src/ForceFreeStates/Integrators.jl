"""
    AbstractIntegrator

Supertype of the three force-free-states formalisms selected by the scripting API
`solve(equil, alg; ...)`: [`Forward`](@ref), [`Riccati`](@ref) and [`Galerkin`](@ref).

An integrator object is pure configuration. `solve` translates it into the matching
`ForceFreeStatesControl` keywords, so the struct fields and the TOML `[ForceFreeStates]`
keys always describe the same solve — `ForceFreeStatesControl` stays the single source of
truth and the TOML path is unaffected.
"""
abstract type AbstractIntegrator end

"""
    Forward()

Serial Euler-Lagrange integrator: sweeps the full radial domain and stores the dense ξ
solution. The only formalism that supports kinetic runs and the only one whose solution
feeds the profile-based PerturbedEquilibrium outputs. Maps onto `integrator = "forward"`.
"""
struct Forward <: AbstractIntegrator end

"""
    Riccati(; nchunks=0)

STRIDE-style chunked Riccati integrator: solves the Euler-Lagrange system on independent
radial chunks and couples them through a boundary-value problem, which is what unlocks the
inter-surface Δ′ matrix. Threads come from `julia -t`; the chunk count is the only tunable
and never depends on the thread count. Maps onto `integrator = "riccati"`.

## Fields

  - `nchunks::Int` - Chunk-count target; `0` derives it from the number of singular surfaces.
"""
@kwdef struct Riccati <: AbstractIntegrator
    nchunks::Int = 0
end

"""
    Galerkin(; solver="LU", nx=256, ...)

RDCON outer-region singular Galerkin solver: solves the same Euler-Lagrange system
variationally on a finite-element grid packed around the rational surfaces, producing Δ′
without a radial ODE sweep. Maps onto `integrator = "galerkin"`; every field is the
matching `gal_*` control key without the prefix.

## Fields

  - `solver::String` - Banded linear solver, `"LU"` (zgbtrf/zgbtrs) or `"cholesky"` (zpbtrf/zpbtrs). Matching requires `"LU"`.
  - `nx::Int` - Elements per interval between singular surfaces.
  - `nq::Int` - Gauss-Lobatto quadrature order per element.
  - `pfac::Float64` - Grid packing ratio near singular surfaces.
  - `dx0::Float64` - Resonant-element integration truncation distance, in units of 1/|n q′|.
  - `dx1::Float64` - Resonant-element size, in units of 1/|n q′|.
  - `dx2::Float64` - Extension-element size, in units of 1/|n q′|.
  - `cutoff::Int` - Number of elements carrying the large solution as driving term.
  - `tol::Float64` - Resonant-quadrature tolerance.
  - `gnstep::Int` - Maximum resonant-quadrature evaluations.
  - `dx1dx2_flag::Bool` - Enable the special dx1/dx2 treatment of resonant and extension elements.
  - `sing_order::Int` - Base power-series order for the singular asymptotics.
  - `sing_order_ceiling::Bool` - Auto-raise the order per surface for a high Mercier index.
  - `rpec_flag::Bool` - Append the mpert coil-response columns to the Δ′ solve. Forced on when a [`ResistiveMatch`](@ref) is requested.
  - `edge_onesided::Bool` - Pack the two end intervals one-sided toward their single rational end instead of the Fortran symmetric pack.
"""
@kwdef struct Galerkin <: AbstractIntegrator
    solver::String = "LU"
    nx::Int = 256
    nq::Int = 6
    pfac::Float64 = 0.001
    dx0::Float64 = 5e-4
    dx1::Float64 = 1e-3
    dx2::Float64 = 1e-3
    cutoff::Int = 10
    tol::Float64 = 1e-10
    gnstep::Int = 20000
    dx1dx2_flag::Bool = true
    sing_order::Int = 6
    sing_order_ceiling::Bool = true
    rpec_flag::Bool = false
    edge_onesided::Bool = false
end

"""
    ResistiveMatch(; eta=[], rho=[], rotation=[], gamma=5/3, ideal=false, inner_solver="ray", ...)

Inner-layer matching configuration, passed to `solve` as `match=` and independent of the
integrator that produced the outer solution. Requesting a match closes the basis with a
resistive inner-layer solution instead of the ideal jump condition, so the result carries
`closure = :matched` and a non-zero `bpen`.

Only [`Galerkin`](@ref) implements the match today; a `Riccati` or `Forward` solve with
`match` set errors. The per-surface vectors are ordered core to edge and must have one
entry per matched rational surface.

## Fields

  - `eta::Vector{Float64}` - Per-surface resistivity η.
  - `rho::Vector{Float64}` - Per-surface mass density ρ in kg/m³.
  - `rotation::Vector{Float64}` - Per-surface rotation frequency f in Hz; the forced eigenvalue is γ_s = 2πi·n·f.
  - `gamma::Float64` - Ratio of specific heats Γ in the resistive-layer coefficients.
  - `ideal::Bool` - Build the ideal (perfectly shielded) matched solution: skip the inner layer and use the bare coil columns. `eta`, `rho` and `rotation` are then unread.
  - `inner_solver::String` - Inner-layer Δ backend, `"ray"` (rotated-contour collocation) or `"galerkin"` (Hermite-cubic elements).
  - `inner_xfac::Float64` - Asymptotic-matching radius multiplier of the `"galerkin"` backend.
  - `inner_nx::Int` - Grid cells of the `"galerkin"` backend.
  - `inner_nq::Int` - Quadrature order per cell of the `"galerkin"` backend.
  - `inner_cutoff::Int` - Cells carrying the large solution as driving term in the `"galerkin"` backend.
  - `inner_kmax::Int` - Large-x asymptotic series order of the `"galerkin"` backend.
"""
@kwdef struct ResistiveMatch
    eta::Vector{Float64} = Float64[]
    rho::Vector{Float64} = Float64[]
    rotation::Vector{Float64} = Float64[]
    gamma::Float64 = 5 / 3
    ideal::Bool = false
    inner_solver::String = "ray"
    inner_xfac::Float64 = 10.0
    inner_nx::Int = 1280
    inner_nq::Int = 5
    inner_cutoff::Int = 5
    inner_kmax::Int = 8
end

"""
    _integrator_symbol(alg) -> Symbol

The `ForceFreeStatesControl.integrator` token an [`AbstractIntegrator`](@ref) selects.
"""
_integrator_symbol(::Forward) = :forward
_integrator_symbol(::Riccati) = :riccati
_integrator_symbol(::Galerkin) = :galerkin

"""
    _set_ctrl!(kwargs, key, value, source) -> kwargs

Write one `ForceFreeStatesControl` keyword derived from `source`, rejecting a duplicate the
caller also passed to `solve` — the same knob would otherwise be set in two places.
"""
function _set_ctrl!(kwargs::Dict{Symbol,Any}, key::Symbol, value, source)
    haskey(kwargs, key) &&
        error("`$key` is controlled by the $(nameof(typeof(source))) object; set it there instead of as a `solve` keyword")
    kwargs[key] = value
    return kwargs
end

"""
    _apply_alg!(kwargs, alg) -> kwargs

Translate an [`AbstractIntegrator`](@ref) into `ForceFreeStatesControl` keywords on
`kwargs`. Pure translation: every field maps onto the control key of the same meaning.
"""
function _apply_alg!(kwargs::Dict{Symbol,Any}, alg::AbstractIntegrator)
    return _set_ctrl!(kwargs, :integrator, String(_integrator_symbol(alg)), alg)
end

function _apply_alg!(kwargs::Dict{Symbol,Any}, alg::Riccati)
    _set_ctrl!(kwargs, :integrator, String(_integrator_symbol(alg)), alg)
    return _set_ctrl!(kwargs, :nchunks, alg.nchunks, alg)
end

function _apply_alg!(kwargs::Dict{Symbol,Any}, alg::Galerkin)
    _set_ctrl!(kwargs, :integrator, String(_integrator_symbol(alg)), alg)
    for name in fieldnames(Galerkin)
        _set_ctrl!(kwargs, Symbol(:gal_, name), getfield(alg, name), alg)
    end
    return kwargs
end

"""
    _apply_match!(kwargs, match, alg) -> kwargs

Translate a [`ResistiveMatch`](@ref) into the `gal_*` matching keywords, or error for an
integrator whose resonant matching is not implemented yet. `nothing` leaves `kwargs` alone,
which is the ideal-closure default.
"""
_apply_match!(kwargs::Dict{Symbol,Any}, ::Nothing, ::AbstractIntegrator) = kwargs

function _apply_match!(kwargs::Dict{Symbol,Any}, ::ResistiveMatch, alg::AbstractIntegrator)
    return error("resonant matching for this integrator is not yet implemented (requested with $(nameof(typeof(alg))))")
end

function _apply_match!(kwargs::Dict{Symbol,Any}, match::ResistiveMatch, ::Galerkin)
    kwargs[:gal_match_flag] = true
    # The match consumes the coil-response columns, so it implies the rpec solve.
    kwargs[:gal_rpec_flag] = true
    for name in fieldnames(ResistiveMatch)
        key = name === :ideal ? :gal_ideal_flag : Symbol(:gal_, name)
        _set_ctrl!(kwargs, key, getfield(match, name), match)
    end
    return kwargs
end
