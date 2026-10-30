# SLAYER.jl
#
# SLAYER (Slab Layer) two-fluid drift-MHD inner-layer model (Fitzpatrick,
# Tearing Mode Dynamics in Tokamak Plasmas, IOP 2023; Park 2022,
# Phys. Plasmas 29, 072506; Burgess et al. 2026). The dispersion path uses
# the Fitzpatrick layer formulation — P_perp / P_tor transport, c_beta
# compressibility, D_norm normalized ion-skin scale, two-fluid drift
# coupling via Q_e, Q_i, iota_e (see `Riccati.jl`).
#
# A separate `del_s` layer-thickness diagnostic (`slayer_layer_thickness` in
# `LayerThickness.jl`) returns the resistive layer width in meters at each
# rational surface.
#
# Type-parameter `S` of `SLAYERModel{S}` selects the Riccati formulation
# used for the dispersion relation; only `:fitzpatrick` is implemented.
#
# `Q = ω + iγ` is passed directly to `solve_inner` rather than stored on
# the parameter struct.

module SLAYER

using LinearAlgebra
using StaticArrays

import ..InnerLayerModel, ..InnerLayerResponse, ..solve_inner, ..InnerLayerParameters
using ...Utilities.PhysicalConstants
using ...Utilities.NeoclassicalResistivity
using ...Utilities.NeoclassicalResistivity: NeoResistivityModel, SpitzerModel,
    SpitzerHarmModel, SauterNeoModel, RedlNeoModel,
    coulomb_log_e, eta_spitzer, tau_ee_spitzer_harm, eta_spitzer_harm,
    trapped_fraction_eps, nu_star_e, eta_neoclassical

"""
    SLAYERModel(; variant=:fitzpatrick, P_perp_model=:D_perp, D_c_type=:none, D_R=nothing, D_geo=nothing, theta_sample=0.0)

SLAYER inner-layer model: the Riccati formulation and the layer closures that turn the plasma
(the equilibrium's kinetic data) into layer parameters. The type parameter `S` of
`SLAYERModel{S}` is the formulation:

  - `:fitzpatrick` -- P_perp/P_tor Fitzpatrick formulation (default;
    Fitzpatrick 2023 two-fluid drift-MHD layer)

Future dispersion variants (e.g. `:standard`) may be added but are not
currently implemented. The `del_s` formulation is exposed separately as
the layer-thickness diagnostic [`slayer_layer_thickness`](@ref), not as
a dispersion `solve_inner` path.

## Fields

  - `P_perp_model` - How the perpendicular Prandtl number is built: `:D_perp` (default; needs χ⊥,i),
    `:chi_perp_e`, `:chi_perp_i`, `:P_phi`, `:c_beta` or `:tau_E`; see [`slayer_parameters`](@ref).
  - `D_c_type` - Critical-Δ offset: `:none` (default), `:lar`, `:fitzpatrick` or `:toroidal`
    (χ_‖-matching formulas of Connor, Ham, Hastie & Liu 2015).
  - `D_R`, `D_geo` - Critical-Δ inputs; `nothing` (default) derives them per surface from the equilibrium:
    `D_R = E + F + H²`, and `D_geo` the Connor et al. 2015 Eq. 59 toroidal geometric factor in the `r_s`
    reference (consumed only by `:toroidal`, which also takes its parallel wavenumber from the equilibrium,
    `toroidal_kpar`). An explicit `D_R = 0.0` disables the offset for every `D_c_type`; `D_geo = 0.0` only for `:toroidal`.
  - `theta_sample` - Poloidal angle (normalized to [0, 1)) of the minor-radius chord of the `:midplane`
    radial label, used only by API callers that pass `rs_method = :midplane`; the default toroidal-flux
    label does not depend on angle, so a nonzero value warns (default 0, the outboard midplane).
"""
struct SLAYERModel{S} <: InnerLayerModel
    P_perp_model::Symbol
    D_c_type::Symbol
    D_R::Union{Nothing,Float64}
    D_geo::Union{Nothing,Float64}
    theta_sample::Float64
end

const _D_C_TYPES = (:none, :lar, :fitzpatrick, :toroidal)

function SLAYERModel(; variant::Symbol=:fitzpatrick, P_perp_model=:D_perp, D_c_type=:none, D_R=nothing, D_geo=nothing,
    theta_sample::Real=0.0)
    P_perp_model, D_c_type = Symbol(P_perp_model), Symbol(D_c_type)
    P_perp_model in ALLOWED_P_PERP_MODELS || throw(ArgumentError("SLAYERModel: P_perp_model = $P_perp_model not in $ALLOWED_P_PERP_MODELS"))
    D_c_type in _D_C_TYPES || throw(ArgumentError("SLAYERModel: D_c_type = $D_c_type not in $_D_C_TYPES"))
    iszero(theta_sample) || @warn "SLAYERModel: theta_sample = $theta_sample has no effect: the layer quantities " *
          "use the toroidal-flux radial label, which does not depend on poloidal angle." maxlog = 1
    return SLAYERModel{variant}(P_perp_model, D_c_type, D_R === nothing ? nothing : Float64(D_R),
        D_geo === nothing ? nothing : Float64(D_geo), Float64(theta_sample))
end

include("LayerParameters.jl")
include("Riccati.jl")
include("LayerThickness.jl")
include("LayerInputs.jl")

export SLAYERModel, SLAYERParameters, slayer_parameters
export r_based_shear
export riccati_del_s, slayer_layer_thickness, slayer_algebraic_widths, LayerWidths
export surface_minor_radius, surface_da_dpsi, radial_label, build_slayer_inputs, surface_resistivity, layer_resistivity,
    toroidal_dgeo, toroidal_kpar
export NeoResistivityModel, SpitzerModel, SpitzerHarmModel, SauterNeoModel, RedlNeoModel

end # module SLAYER
