# ToroidalLayer.jl
#
# Shared library for toroidal resistive inner-layer models whose equations take the GGJ form: three fields
# (Ψ, Ξ, Υ), second order, coefficients polynomial in x, and a far field of two Mercier power-like solutions
# and four parabolic exponentials. It holds the Wasow far-field construction (GW2020), the rotated-ray
# solver and the real-axis Hermite (Galerkin) solver; a model supplies its equations through the interface
# below and forwards its `solve_inner` / `solve_inner_profile` to the backends. GGJ is the reference model.
#
#   GWP2016 — A. H. Glasser, Z. R. Wang & J.-K. Park, Phys. Plasmas 23, 112506 (2016)
#   GW2020  — A. H. Glasser & Z. R. Wang, Phys. Plasmas 27, 012506 (2020)

module ToroidalLayer

using LinearAlgebra
using StaticArrays
using Random
using Printf
using DoubleFloats: Double64
using ...Utilities.HermiteFEM: hermite_basis, pack_grid

import ..InnerLayerModel, ..InnerLayerResponse, ..InnerLayerParameters
import ..solve_inner_profile

"""
    ToroidalLayerModel{S} <: InnerLayerModel

Abstract supertype of the inner-layer models built on this library; `S` is the solver backend symbol
(`:ray`, `:galerkin`, or a model's own). Concrete models forward `solve_inner` to [`ray_response`](@ref) or
[`galerkin_response`](@ref) and `solve_inner_profile` to [`ray_profile`](@ref) or [`galerkin_profile`](@ref).
"""
abstract type ToroidalLayerModel{S} <: InnerLayerModel end

"""
    ToroidalLayerParameters <: InnerLayerParameters

Abstract supertype of the per-surface parameters of a [`ToroidalLayerModel`](@ref). The shared solvers read a
subtype only through this interface, so each model keeps whatever numbers its equations need:

  - [`layer_rows`](@ref) and [`layer_weak_form`](@ref) - the layer equations;
  - [`wasow_basis`](@ref), [`split_block`](@ref) and [`power_exponents`](@ref) - the far-field Wasow construction;
  - [`default_ray_angle`](@ref), [`parabolic_rate`](@ref), [`ohm_rate`](@ref) and [`fast_rates`](@ref) - the
    far-field rates that set the ray contour and mesh;
  - [`inner_Q`](@ref), [`rescale_delta`](@ref) and [`profile_conversions`](@ref) - the normalization between the
    physical growth rate, the outer region and the layer variables.
"""
abstract type ToroidalLayerParameters <: InnerLayerParameters end

"""
    layer_rows([CT,] params, Q, x) -> (Ũ, Ṽ)

The layer equations solved for the second derivatives, `u'' = Ũ·u + Ṽ·u'` for u = (Ψ, Ξ, Υ), as 3×3 matrices
of element type `CT` (default `ComplexF64`) at real or complex `x`.
"""
function layer_rows end
layer_rows(params::ToroidalLayerParameters, Q::Number, x::Number) = layer_rows(ComplexF64, params, Q, x)

"""
    layer_weak_form(params, Q::ComplexF64, x) -> (I, U, V)

3×3 coefficients of `I·u'' − V·u' − U·u = 0` at real or complex `x`, the form the Hermite assembly integrates
by parts (GWP2016 Eqs. 12–15, 32); `I` must be independent of `x`.
"""
function layer_weak_form end

"""
    wasow_basis(params, Q::ComplexF64) -> (T, (J₀, J₁, J₂), splitter)

The far-field system u' = x(A₀ + A₁/x² + A₂/x⁴)u in the state u = (xΨ, Ξ, Υ, (xΨ)'/x, Ξ'/x, Υ'/x) (GW2020
Eqs. 2–5), as a basis `T` whose first two columns span the power-like subspace and last four the
exponential one, the rotated `Jₖ = T⁻¹AₖT` (GW2020 Eqs. 9–10), and the `splitter` passed to
[`split_block`](@ref).
"""
function wasow_basis end

"""
    split_block(splitter, K) -> (B, P)

One order of the GW2020 Eq. (22) splitting equations: `B` the block-diagonal part of `K` and `P` its
off-diagonal blocks, for the exponential block of J₀ the `splitter` from [`wasow_basis`](@ref) encodes.
"""
function split_block end

"""
    power_exponents(params) -> SVector{2,Float64}

Exponents (r₊, r₋) of the two power-like far-field solutions in the Wasow state, x^r for xΨ (GW2020 Eq. 49);
r₊ the large solution, r₋ the small one.
"""
function power_exponents end

"""
    inner_Q(params, γ::Number) -> ComplexF64

The layer's scaled growth rate Q for the physical rate `γ` in 1/s.
"""
function inner_Q end

"""
    rescale_delta(Δ, params) -> SVector{2,ComplexF64}

Matching data (Δ₊, Δ₋) in the layer normalization, as the solvers return them, converted to the outer region's.
"""
function rescale_delta end

"""
    profile_conversions(params) -> (; dψdx, rescale)

For `solve_inner_profile`: poloidal-flux distance per unit layer coordinate, `δψ = dψdx · x`, and the factor
taking the layer-normalized profiles to the outer δψ normalization.
"""
function profile_conversions end

"""
    default_ray_angle(params, Q::ComplexF64) -> Float64

Contour angle θ of the ray x = e^{iθ}s on which the exponential solutions decay monotonically.
"""
function default_ray_angle end

"""
    parabolic_rate(params, Q::ComplexF64, θ) -> Float64

Decay rate per unit s² of the slowest parabolic exponential pair along the ray at angle θ; must be positive.
"""
function parabolic_rate end

"""
    ohm_rate(params, Q::ComplexF64) -> Float64

Rate of the fast exponential pair from the resistive Ohm's-law balance.
"""
function ohm_rate end

"""
    fast_rates(params, Q::ComplexF64) -> Float64

Sum of the fast exponential rates every Hermite cell resolves, alongside the parabolic pair.
"""
function fast_rates end

include("InnerAsymptotics.jl")
include("Galerkin.jl")
include("Ray.jl")
include("Backends.jl")

export ToroidalLayerModel, ToroidalLayerParameters
export layer_rows, layer_weak_form, wasow_basis, split_block, power_exponents, default_ray_angle, parabolic_rate, ohm_rate, fast_rates
export inner_Q, rescale_delta, profile_conversions
export solve_ray, pick_smax, delta_convergence, solution_profile, asymptotic_profile
export ray_response, ray_profile, galerkin_response, galerkin_profile, backend_options

end # module ToroidalLayer
