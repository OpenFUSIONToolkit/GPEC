# InnerLayer.jl
#
# Resistive inner-layer models for matched-asymptotic resistive/extended MHD stability.
# Provides an abstract `InnerLayerModel` interface and its implementations: the GGJ
# (Glasser–Greene–Johnson), CCC (Cho–Choi–Choi drift-resistive) and SLAYER submodules. GGJ and CCC are
# built on the ToroidalLayer library of shared Wasow-basis, ray and Galerkin solvers.

module InnerLayer

using LinearAlgebra
using StaticArrays
using FastInterpolations: cubic_interp, DerivOp

using ..Utilities

include("InnerLayerInterface.jl")
include("ToroidalLayer/ToroidalLayer.jl")
include("GGJ/GGJ.jl")
include("SLAYER/SLAYER.jl")

import .ToroidalLayer: ToroidalLayerModel, ToroidalLayerParameters, inner_Q, rescale_delta
import .ToroidalLayer: layer_parameters, check_layer_inputs, growth_rate_scale, layer_parameter_columns
import .ToroidalLayer: solve_ray, pick_smax, delta_convergence, solution_profile, asymptotic_profile
import .GGJ: GGJModel, GGJParameters, mercier_di, mercier_dr

import .SLAYER: SLAYERModel, SLAYERParameters, slayer_parameters, r_based_shear
import .SLAYER: riccati_del_s, slayer_layer_thickness, LayerWidths
import .SLAYER: surface_minor_radius, surface_da_dpsi, radial_label, build_slayer_inputs, toroidal_dgeo, toroidal_kpar

include("LayerPlasma.jl")
# CCC reads the electron diamagnetic frequency from LayerPlasma.
include("CCC/CCC.jl")
import .CCC: CCCModel, CCCParameters

"""
    model_name(model) -> String

The `inner_model` token the tearing output records for `model`, e.g. `"ggj_ray"` or `"slayer_fitzpatrick"`.
"""
model_name(::GGJModel{S}) where {S} = "ggj_$S"
model_name(::CCCModel{S}) where {S} = "ccc_$S"
model_name(::SLAYERModel{S}) where {S} = "slayer_$S"

export InnerLayerModel, InnerLayerParameters, InnerLayerResponse, solve_inner, solve_inner_profile, layer_response
export layer_eta, layer_rho, layer_omega_E, layer_omega_star_e
export ToroidalLayer, ToroidalLayerModel, ToroidalLayerParameters
export GGJ, GGJModel, GGJParameters
export CCC, CCCModel, CCCParameters
export mercier_di, mercier_dr, inner_Q, rescale_delta
export layer_parameters, check_layer_inputs, growth_rate_scale, layer_parameter_columns
export solve_ray, pick_smax, delta_convergence, solution_profile, asymptotic_profile

export SLAYER, SLAYERModel, SLAYERParameters, slayer_parameters, r_based_shear
export riccati_del_s, slayer_layer_thickness, LayerWidths
export surface_minor_radius, surface_da_dpsi, radial_label, build_slayer_inputs, toroidal_dgeo, toroidal_kpar

end # module InnerLayer
