# Models.jl
#
# User-facing inner-layer model configuration for the solve grammar: the alg slot of the
# matching problems, mirroring how Forward/Riccati/Galerkin configure the integrators. The
# structs are pure configuration; `InnerLayer`'s type tags (`GGJModel{S}`, `SLAYERModel{S}`)
# stay the dispatch currency of `solve_inner`, and the GGJ forwarding below translates one
# into the other. Plasma-composition and matching-procedure knobs (mu_i, zeff, resistivity
# model, per-surface η/ρ/rotation) belong to the PROBLEM side — see `layer_parameters`.

"""
    GGJ(; solver=:ray, inner_xfac=10.0, inner_nx=1280, inner_nq=5, inner_cutoff=5, inner_kmax=8)

Glasser-Greene-Johnson resistive inner-layer model (Glasser, Wang & Park 2016), the
finite-β two-parity layer: it supplies both the tearing and interchange matching channels
and reconstructs the layer field profiles, so it can CLOSE a matched outer solution.

## Fields

  - `solver::Symbol` - Δ(Q) backend: `:ray` (rotated-contour collocation, certified) or `:galerkin` (Hermite-cubic elements).
  - `inner_xfac::Float64` - Asymptotic-matching radius multiplier of the `:galerkin` backend.
  - `inner_nx::Int` - Grid cells of the `:galerkin` backend.
  - `inner_nq::Int` - Quadrature order per cell of the `:galerkin` backend.
  - `inner_cutoff::Int` - Cells carrying the large solution as driving term in the `:galerkin` backend.
  - `inner_kmax::Int` - Large-x asymptotic series order of the `:galerkin` backend.
"""
@kwdef struct GGJ <: InnerLayer.InnerLayerModel
    solver::Symbol = :ray
    inner_xfac::Float64 = 10.0
    inner_nx::Int = 1280
    inner_nq::Int = 5
    inner_cutoff::Int = 5
    inner_kmax::Int = 8
end

"""
    SLAYER(; chi_perp=1.0, chi_tor=1.0)

SLAYER slab resistive inner-layer model (Fitzpatrick formulation): a pressureless slab
layer supplying the tearing matching channel only — no interchange channel and no
reconstructable layer field profiles, so it can drive a free-eigenvalue tearing solve but
can never close a matched outer solution (see [`closure_capable`](@ref)).

## Fields

  - `chi_perp::Float64` - Fallback perpendicular heat diffusivity χ⊥ in m²/s, used when the kinetic profiles carry no usable χ_e.
  - `chi_tor::Float64` - Fallback toroidal momentum diffusivity χ_φ in m²/s, used when the kinetic profiles carry no usable χ_φ.
"""
@kwdef struct SLAYER <: InnerLayer.InnerLayerModel
    chi_perp::Float64 = 1.0
    chi_tor::Float64 = 1.0
end

"""
    closure_capable(model) -> Bool

Whether an inner-layer model can CLOSE a matched outer solution: that takes both parity
channels of the matching data and the reconstructed layer field profiles. [`GGJ`](@ref) can;
[`SLAYER`](@ref) cannot (slab: single parity, no interchange channel, no layer profiles) —
a `SLAYER` model is restricted to the free-eigenvalue tearing solve.
"""
closure_capable(::GGJ) = true
closure_capable(::SLAYER) = false

# The GGJ configuration forwards onto the InnerLayer dispatch tags; the backend grid knobs
# only exist on the :galerkin backend, matching the deck's gal_inner_* keys.
function InnerLayer.solve_inner(model::GGJ, params, γ::Number)
    model.solver in (:ray, :galerkin) ||
        error("GGJ solver must be :ray or :galerkin (got :$(model.solver))")
    return InnerLayer.solve_inner(InnerLayer.GGJModel(; solver=model.solver), params, γ)
end

function InnerLayer.solve_inner_profile(model::GGJ, params, γ::Number)
    model.solver === :ray &&
        return InnerLayer.solve_inner_profile(InnerLayer.GGJModel(; solver=:ray), params, γ)
    model.solver === :galerkin &&
        return InnerLayer.solve_inner_profile(InnerLayer.GGJModel(; solver=:galerkin), params, γ;
            xfac=model.inner_xfac, nx=model.inner_nx, nq=model.inner_nq,
            cutoff=model.inner_cutoff, kmax=model.inner_kmax)
    error("GGJ solver must be :ray or :galerkin (got :$(model.solver))")
end
