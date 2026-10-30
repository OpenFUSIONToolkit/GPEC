# InnerLayer.jl
#
# Resistive inner-layer models for matched-asymptotic resistive/extended MHD stability.
# Provides an abstract `InnerLayerModel` interface and its implementations: the GGJ
# (Glasser–Greene–Johnson) and SLAYER submodules.

module InnerLayer

using LinearAlgebra
using StaticArrays
using FastInterpolations: cubic_interp

using ..Utilities

include("InnerLayerInterface.jl")
include("GGJ/GGJ.jl")
include("SLAYER/SLAYER.jl")

import .GGJ: GGJModel, GGJParameters, mercier_di, mercier_dr, inner_Q, rescale_delta
import .GGJ: solve_ray, pick_smax, delta_convergence, solution_profile, asymptotic_profile

import .SLAYER: SLAYERModel, SLAYERParameters, slayer_parameters, r_based_shear
import .SLAYER: riccati_del_s, slayer_layer_thickness, LayerWidths
import .SLAYER: surface_minor_radius, surface_da_dpsi, radial_label, build_slayer_inputs, toroidal_dgeo, toroidal_kpar

include("LayerPlasma.jl")

"""
    model_name(model) -> String

The `inner_model` token the tearing output records for `model`, e.g. `"ggj_ray"` or `"slayer_fitzpatrick"`.
"""
model_name(::GGJModel{S}) where {S} = "ggj_$S"
model_name(::SLAYERModel{S}) where {S} = "slayer_$S"

export InnerLayerModel, InnerLayerParameters, InnerLayerResponse, solve_inner, solve_inner_profile
export layer_eta, layer_rho, layer_omega_E
export GGJ, GGJModel, GGJParameters
export mercier_di, mercier_dr, inner_Q, rescale_delta
export solve_ray, pick_smax, delta_convergence, solution_profile, asymptotic_profile

export SLAYER, SLAYERModel, SLAYERParameters, slayer_parameters, r_based_shear
export riccati_del_s, slayer_layer_thickness, LayerWidths
export surface_minor_radius, surface_da_dpsi, radial_label, build_slayer_inputs, toroidal_dgeo, toroidal_kpar

end # module InnerLayer
