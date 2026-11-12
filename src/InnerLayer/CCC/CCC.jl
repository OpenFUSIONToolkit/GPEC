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
import ..layer_omega_star_e
using ..ToroidalLayer
import ..ToroidalLayer: inner_Q, rescale_delta, profile_conversions, power_exponents
import ..ToroidalLayer:
    layer_rows, layer_weak_form, wasow_basis, split_block, default_ray_angle, parabolic_rate,
    parabolic_resolution_rate, ray_angle_admissible, ohm_rate, fast_rates
import ..ToroidalLayer: layer_parameters, check_layer_inputs, mercier_di, growth_rate_scale, layer_parameter_columns
using ..GGJ: GGJParameters, q0

"""
    CCCModel{S,O,Q,M} <: ToroidalLayerModel{S}
    CCCModel(; solver=:ray, Qstar=nothing, mperp=1.0, options...)

Drift-resistive inner-layer model of Cho, Choi & Choi (1987) in the form of Kim, Cho & Choi (1988), solved on
[`CCCParameters`](@ref). `S` selects the solver backend; only `:ray` (the rotated-contour solver of
[`solve_ray`](@ref)) is available. Backend keywords given at construction, e.g. `CCCModel(; cfl=0.1)`, apply to
every solve; a keyword the backend does not take is an error here, not at the first solve. `Qstar` and `mperp` set
the parameters the tearing and matching stages build ([`layer_parameters`](@ref)); a solve on given
`CCCParameters` uses theirs.

## Fields

  - `options::NamedTuple` - Backend keywords forwarded to every solve; call-site keywords win.
  - `Qstar` - Scaled electron diamagnetic frequency Q* at every surface (a number), at each surface core to edge
    (a vector), or `nothing` to take Q* = −n·ω*ₑ/Q₀ from the equilibrium's kinetic data
    ([`layer_omega_star_e`](@ref)).
  - `mperp` - Perpendicular-transport weight M⊥/M at every surface (a number) or at each surface core to edge (a
    vector); positive.
"""
struct CCCModel{S,O<:NamedTuple,Q<:Union{Nothing,Float64,Vector{Float64}},M<:Union{Float64,Vector{Float64}}} <: ToroidalLayerModel{S}
    options::O
    Qstar::Q
    mperp::M
end

function CCCModel(; solver::Symbol=:ray, Qstar=nothing, mperp=1.0, options...)
    solver === :ray || throw(ArgumentError("CCCModel: unknown solver :$solver (only :ray is available)"))
    allowed = backend_options(solver)
    unknown = setdiff(keys(options), allowed)
    isempty(unknown) || throw(ArgumentError("CCCModel(solver=:$solver) does not take $(join(unknown, ", ")); its options are $(join(allowed, ", "))"))
    per_surface(v) = v isa AbstractVector ? collect(Float64, v) : Float64(v)
    qstar = Qstar === nothing ? nothing : per_surface(Qstar)
    qstar === nothing || all(isfinite, qstar) || throw(ArgumentError("CCCModel: Qstar must be finite"))
    m = per_surface(mperp)
    all(v -> isfinite(v) && v > 0, m) || throw(ArgumentError("CCCModel: mperp must be positive and finite"))
    return CCCModel{solver,typeof(values(options)),typeof(qstar),typeof(m)}(values(options), qstar, m)
end
CCCModel{S}() where {S} = CCCModel(; solver=S)

include("CCCParameters.jl")
include("Equations.jl")

"""
    solve_inner(m::CCCModel{:ray}, params::CCCParameters, γ::Number; kwargs...) -> InnerLayerResponse

Rotated-ray matching data (Δ₊, Δ₋) as the `interchange` and `tearing` fields of `InnerLayerResponse`
([`ray_response`](@ref)); the model's options and `kwargs` forward to [`solve_ray`](@ref). `γ = s` is the growth
rate in the surface's E×B frame, in 1/s.
"""
solve_inner(m::CCCModel{:ray}, params::CCCParameters, γ::Number; kwargs...) = ray_response(params, γ; m.options..., kwargs...)

"""
    solve_inner_profile(m::CCCModel{:ray}, params::CCCParameters, γ::Number; kwargs...)

Rotated-ray implementation of the `solve_inner_profile` interface ([`ray_profile`](@ref)).
"""
solve_inner_profile(m::CCCModel{:ray}, params::CCCParameters, γ::Number; kwargs...) = ray_profile(params, γ; m.options..., kwargs...)

# A per-surface model input at surface `k`: the scalar for every surface, or the vector's entry.
_at_surface(v::Float64, k::Integer) = v
_at_surface(v::Vector{Float64}, k::Integer) = v[k]

function check_layer_inputs(model::CCCModel, nsurfaces::Integer)
    for (name, v) in (("Qstar", model.Qstar), ("mperp", model.mperp))
        v isa Vector && length(v) != nsurfaces &&
            throw(ArgumentError("CCCModel: $name has length $(length(v)), expected $nsurfaces (one per surface, core to edge)"))
    end
    return nothing
end

function layer_parameters(model::CCCModel, ggj::GGJParameters, equil, surface)
    Qstar = if model.Qstar === nothing
        equil.kinetic === nothing &&
            throw(ArgumentError("CCC layer: give Qstar, or load kinetic profiles (kinetic_file in [Equilibrium], or attach_kinetic_profiles!)"))
        -only(surface.n) * only(layer_omega_star_e(equil, (surface,))) / q0(ggj)
    else
        _at_surface(model.Qstar, ggj.ising)
    end
    return CCCParameters(ggj, Qstar, _at_surface(model.mperp, ggj.ising))
end

export CCCModel, CCCParameters, parabolic_scales

end # module CCC
