# InnerLayer.jl
#
# Resistive inner-layer models for matched-asymptotic resistive/extended MHD stability.
# Provides an abstract `InnerLayerModel` interface and the GGJ (Glasser–Greene–
# Johnson) submodule. Future submodules (SLAYER, other inner layers) will plug in via the
# same interface.

module InnerLayer

using LinearAlgebra
using StaticArrays

using ..Utilities

include("InnerLayerInterface.jl")
include("GGJ/GGJ.jl")
include("SLAYER/SLAYER.jl")

import .GGJ: GGJModel, GGJParameters, build_asymptotics, evaluate_asymptotics, pick_xmax
import .GGJ: InnerAsymptoticsCache, mercier_di, mercier_dr, inner_Q, rescale_delta
import .GGJ: glasser_wang_2020_eq55
import .GGJ: solve_inner_converged  # experimental, not exported (reachable as a qualified call only)

import .SLAYER: SLAYERModel, SLAYERParameters, slayer_parameters, r_based_shear
import .SLAYER: riccati_del_s, slayer_layer_thickness, LayerWidths
import .SLAYER: surface_minor_radius, surface_da_dpsi, build_slayer_inputs

export InnerLayerModel, InnerLayerParameters, InnerLayerResponse, solve_inner
export GGJ, GGJModel, GGJParameters
export build_asymptotics, evaluate_asymptotics, pick_xmax, InnerAsymptoticsCache
export mercier_di, mercier_dr, inner_Q, rescale_delta
export glasser_wang_2020_eq55

export SLAYER, SLAYERModel, SLAYERParameters, slayer_parameters, r_based_shear
export riccati_del_s, slayer_layer_thickness, LayerWidths
export surface_minor_radius, surface_da_dpsi, build_slayer_inputs

end # module InnerLayer
