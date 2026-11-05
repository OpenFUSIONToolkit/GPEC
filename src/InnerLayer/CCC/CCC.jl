# CCC.jl
#
# Drift-resistive inner-layer model of Cho, Choi & Choi, IEEE Trans. Plasma Sci. PS-15, 418 (1987), in the form of
# Kim, Cho & Choi, Phys. Fluids 31, 2659 (1988), Eqs. (1a)–(1c): the GGJ layer with electron diamagnetic drift (Q*)
# and perpendicular transport (M⊥/M). A ToroidalLayer model on the shared rotated-ray solver; Q* = 0 and M⊥/M = 1
# give the GGJ equations.
#
# Its far field keeps GGJ's two Mercier power-like solutions, but the four parabolic exponentials split into two
# pairs with different rates and phases (Kim Eq. 2), so the Wasow basis needs a general Sylvester splitting solve
# and the ray angle is centred between the two families.

module CCC

using LinearAlgebra
using StaticArrays

import ..InnerLayerResponse
import ..solve_inner, ..solve_inner_profile
using ..ToroidalLayer
import ..ToroidalLayer: inner_Q, rescale_delta, profile_conversions, power_exponents
import ..ToroidalLayer:
    layer_rows, layer_weak_form, wasow_basis, split_block, default_ray_angle, parabolic_rate,
    parabolic_resolution_rate, ray_angle_admissible, ohm_rate, fast_rates
using ..GGJ: GGJParameters

"""
    CCCModel{S,O} <: ToroidalLayerModel{S}
    CCCModel(; solver=:ray, options...)

Drift-resistive inner-layer model of Cho, Choi & Choi (1987) in the form of Kim, Cho & Choi (1988), solved on
[`CCCParameters`](@ref). `S` selects the
solver backend; only `:ray` (the rotated-contour solver of [`solve_ray`](@ref)) is available. Backend keywords given
at construction, e.g. `CCCModel(; cfl=0.1)`, apply to every solve; a keyword the backend does not take is an error
here, not at the first solve.

## Fields

  - `options::NamedTuple` - Backend keywords forwarded to every solve; call-site keywords win.
"""
struct CCCModel{S,O<:NamedTuple} <: ToroidalLayerModel{S}
    options::O
end

function CCCModel(; solver::Symbol=:ray, options...)
    solver === :ray || throw(ArgumentError("CCCModel: unknown solver :$solver (only :ray is available)"))
    allowed = backend_options(solver)
    unknown = setdiff(keys(options), allowed)
    isempty(unknown) || throw(ArgumentError("CCCModel(solver=:$solver) does not take $(join(unknown, ", ")); its options are $(join(allowed, ", "))"))
    return CCCModel{solver,typeof(values(options))}(values(options))
end
CCCModel{S}() where {S} = CCCModel{S,@NamedTuple{}}((;))

# The backends implement the option-free model; a model carrying options forwards them.
const _BareCCC{S} = CCCModel{S,@NamedTuple{}}
function solve_inner(m::CCCModel{S}, params, γ::Number; kwargs...) where {S}
    m isa _BareCCC && throw(MethodError(solve_inner, (m, params, γ)))
    return solve_inner(CCCModel{S}(), params, γ; m.options..., kwargs...)
end
function solve_inner_profile(m::CCCModel{S}, params, γ::Number; kwargs...) where {S}
    m isa _BareCCC && throw(MethodError(solve_inner_profile, (m, params, γ)))
    return solve_inner_profile(CCCModel{S}(), params, γ; m.options..., kwargs...)
end

include("CCCParameters.jl")
include("Equations.jl")

"""
    solve_inner(::CCCModel{:ray}, params::CCCParameters, γ::Number; kwargs...) -> InnerLayerResponse

Rotated-ray matching data (Δ₊, Δ₋) as the `interchange` and `tearing` fields of `InnerLayerResponse`
([`ray_response`](@ref)); keywords forward to [`solve_ray`](@ref). `γ = s` is the growth rate in the surface's
E×B frame, in 1/s.
"""
solve_inner(::_BareCCC{:ray}, params::CCCParameters, γ::Number; kwargs...) = ray_response(params, γ; kwargs...)

"""
    solve_inner_profile(::CCCModel{:ray}, params::CCCParameters, γ::Number; kwargs...)

Rotated-ray implementation of the `solve_inner_profile` interface ([`ray_profile`](@ref)).
"""
solve_inner_profile(::_BareCCC{:ray}, params::CCCParameters, γ::Number; kwargs...) = ray_profile(params, γ; kwargs...)

export CCCModel, CCCParameters, parabolic_scales

end # module CCC
