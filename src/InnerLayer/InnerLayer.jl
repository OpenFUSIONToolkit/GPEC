# InnerLayer.jl
#
# Resistive inner-layer models for matched-asymptotic resistive/extended MHD stability.
# Provides an abstract `InnerLayerModel` interface and its implementations: the GGJ
# (Glasser–Greene–Johnson) and SLAYER submodules.

module InnerLayer

using LinearAlgebra
using StaticArrays

using ..Utilities

include("InnerLayerInterface.jl")
include("GGJ/GGJ.jl")
include("SLAYER/SLAYER.jl")

import .GGJ: GGJModel, GGJParameters, build_asymptotics, evaluate_asymptotics
import .GGJ: InnerAsymptoticsCache, mercier_di, mercier_dr, inner_Q, rescale_delta
import .GGJ: glasser_wang_2020_eq55
import .GGJ: solve_ray, RaySolveResult, pick_smax, physical_ua_dua
import .GGJ: delta_convergence, solution_profile, asymptotic_profile, q4_surface_benchmark

import .SLAYER: SLAYERModel, SLAYERParameters, slayer_parameters, r_based_shear
import .SLAYER: riccati_del_s, slayer_layer_thickness, LayerWidths
import .SLAYER: surface_minor_radius, surface_da_dpsi, radial_label, build_slayer_inputs

"""
    model_name(model) -> String

The deck `inner_model` token of `model`, e.g. `"ggj_ray"` or `"slayer_fitzpatrick"`.
"""
model_name(::GGJModel{S}) where {S} = "ggj_$S"
model_name(::SLAYERModel{S}) where {S} = "slayer_$S"

export InnerLayerModel, InnerLayerParameters, InnerLayerResponse, solve_inner, solve_inner_profile
export GGJ, GGJModel, GGJParameters
export build_asymptotics, evaluate_asymptotics, InnerAsymptoticsCache
export mercier_di, mercier_dr, inner_Q, rescale_delta
export glasser_wang_2020_eq55
export solve_ray, RaySolveResult, pick_smax, physical_ua_dua
export delta_convergence, solution_profile, asymptotic_profile, q4_surface_benchmark

export SLAYER, SLAYERModel, SLAYERParameters, slayer_parameters, r_based_shear
export riccati_del_s, slayer_layer_thickness, LayerWidths
export surface_minor_radius, surface_da_dpsi, radial_label, build_slayer_inputs

end # module InnerLayer
