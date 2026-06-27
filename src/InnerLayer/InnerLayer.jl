# InnerLayer.jl
#
# Resistive inner-layer models for matched-asymptotic resistive/extended MHD stability.
# Provides an abstract `InnerLayerModel` interface and the GGJ (Glasser–Greene–
# Johnson) submodule. Future submodules (SLAYER, other inner layers) will plug in via the
# same interface.

module InnerLayer

using LinearAlgebra
using StaticArrays

include("InnerLayerInterface.jl")
include("GGJ/GGJ.jl")
# include("SLAYER/Slayer.jl") --- SLAYER code goes here

import .GGJ: GGJModel, GGJParameters, build_asymptotics, evaluate_asymptotics, pick_xmax
import .GGJ: InnerAsymptoticsCache, mercier_di, mercier_dr, inner_Q, rescale_delta
import .GGJ: glasser_wang_2020_eq55, solve_inner_converged  # solve_inner_converged: experimental, not exported (reachable as a qualified call only)
# SLAYER imports go here

export InnerLayerModel, solve_inner
export GGJ, GGJModel, GGJParameters
export build_asymptotics, evaluate_asymptotics, pick_xmax, InnerAsymptoticsCache
export mercier_di, mercier_dr, inner_Q, rescale_delta
export glasser_wang_2020_eq55

# SLAYER exports go here


end # module InnerLayer
