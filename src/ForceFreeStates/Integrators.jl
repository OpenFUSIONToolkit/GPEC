"""
    AbstractIntegrator

Numerical algorithms for the force-balance problem. Physical inputs remain on the
problem; each integrator carries its own accuracy, storage, and discretization settings.
"""
abstract type AbstractIntegrator end

"""
    Forward(; kwargs...)

Serial forward integration with Gaussian reduction and dense solution profiles.

## Fields

  - `eulerlagrange_tolerance::Float64` - Relative tolerance for ODE integration.
  - `renorm_threshold::Float64` - Column-norm growth threshold for Gaussian reduction or Riccati renormalization.
  - `numsteps_init::Int` - Initial capacity for stored ODE solution steps.
  - `numunorms_init::Int` - Initial capacity for solution-normalization records.
  - `save_interval::Int` - Save every Nth ODE step; always retain steps near rational surfaces.
  - `nchunks::Int` - Target chunk count; zero derives the count from the singular surfaces.
"""
@kwdef struct Forward <: AbstractIntegrator
    eulerlagrange_tolerance::Float64 = 1e-8
    renorm_threshold::Float64 = 1e3
    numsteps_init::Int = 4000
    numunorms_init::Int = 100
    save_interval::Int = 1
    nchunks::Int = 0
end

"""
    Riccati(; kwargs...)

Chunked Riccati integration and the singular-surface boundary-value solve.

## Fields

  - `eulerlagrange_tolerance::Float64` - Relative tolerance for ODE integration.
  - `renorm_threshold::Float64` - Column-norm growth threshold for Gaussian reduction or Riccati renormalization.
  - `numsteps_init::Int` - Initial capacity for stored ODE solution steps.
  - `numunorms_init::Int` - Initial capacity for solution-normalization records.
  - `save_interval::Int` - Save every Nth ODE step; always retain steps near rational surfaces.
  - `nchunks::Int` - Target chunk count; zero derives the count from the singular surfaces.
  - `extended_precision_bvp::Bool` - Use Complex{Double64} for the Delta-prime BVP and PEST3 combination to reduce cancellation error.
"""
@kwdef struct Riccati <: AbstractIntegrator
    eulerlagrange_tolerance::Float64 = 1e-8
    renorm_threshold::Float64 = 1e3
    numsteps_init::Int = 4000
    numunorms_init::Int = 100
    save_interval::Int = 1
    nchunks::Int = 0
    extended_precision_bvp::Bool = true
end

"""
    Galerkin(; nx=256, ...)

RDCON outer-region singular Galerkin solver: solves the same Euler-Lagrange system
variationally on a finite-element grid packed around the rational surfaces, producing Δ′
without a radial ODE sweep. Maps onto `integrator = "Galerkin"`; every field is the
corresponding discretization setting. Factorization is selected automatically.

## Fields

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
  - `frobenius_order_ceiling::Bool` - Auto-raise the order per surface for a high Mercier index.
  - `rpec_flag::Bool` - Append the mpert coil-response columns to the Δ′ solve. Required for a later `MatchProblem` solve on the result.
  - `edge_onesided::Bool` - Pack the two end intervals one-sided toward their single rational end instead of the Fortran symmetric pack.
  - `basis_output::Bool` - Save the raw Galerkin basis for development diagnostics.
  - `cut_solution::Bool` - Also keep the cut solution, for a later `MatchProblem`'s composite inner profiles.
"""
@kwdef struct Galerkin <: AbstractIntegrator
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
    frobenius_order_ceiling::Bool = true
    rpec_flag::Bool = false
    edge_onesided::Bool = false
    cut_solution::Bool = false
    basis_output::Bool = false
end

_integrator_symbol(::Forward) = :forward
_integrator_symbol(::Riccati) = :riccati
_integrator_symbol(::Galerkin) = :galerkin
