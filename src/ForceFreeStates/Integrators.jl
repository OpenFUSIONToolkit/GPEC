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
  - `rpec_flag::Bool` - Append the mpert coil-response columns to the Δ′ solve. Required for a later `MatchProblem` solve on the result.
  - `edge_onesided::Bool` - Pack the two end intervals one-sided toward their single rational end instead of the Fortran symmetric pack.
  - `cut_solution::Bool` - Also reconstruct the cut solution (`xi_cut`), which a later `MatchProblem` solve needs for the composite inner-region profiles.
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
    cut_solution::Bool = false
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

