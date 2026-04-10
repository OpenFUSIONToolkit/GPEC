# InnerLayer.jl
#
# Resistive inner-layer models for matched-asymptotic resistive MHD stability.
# Provides an abstract `InnerLayerModel` interface and the GGJ (Glasser–Greene–
# Johnson) submodule. Future submodules (SLAYER, kinetic) will plug in via the
# same interface.

module InnerLayer

using LinearAlgebra
using StaticArrays

include("InnerLayerInterface.jl")
include("GGJ/GGJ.jl")

import .GGJ: GGJModel, GGJParameters, build_asymptotics, evaluate_asymptotics, pick_xmax
import .GGJ: InnerAsymptoticsCache, mercier_di, mercier_dr, inner_Q, rescale_delta
import .GGJ: glasser_wang_2020_eq55

export InnerLayerModel, solve_inner
export GGJ, GGJModel, GGJParameters
export build_asymptotics, evaluate_asymptotics, pick_xmax, InnerAsymptoticsCache
export mercier_di, mercier_dr, inner_Q, rescale_delta
export glasser_wang_2020_eq55

end # module InnerLayer
