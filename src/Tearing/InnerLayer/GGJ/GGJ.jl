# GGJ.jl
#
# Glasser–Greene–Johnson resistive inner-layer model. Provides two
# interchangeable solvers selected via the `solver` type-parameter of
# `GGJModel`:
#
#   - `:shooting`  – stable backward shoot from X_max → 0 (Phase 3)
#   - `:galerkin`  – Hermite-cubic finite element method (Phase 4)
#
# Both solvers share the same `inps` Wasow asymptotic-basis kernel
# (`InnerAsymptotics.jl`) for the large-x boundary condition. They return
# the parity-projected matching data `(Δ_odd, Δ_even)` of Glasser, Wang &
# Park, Phys. Plasmas **23**, 112506 (2016), Eqs. (34)–(35).

module GGJ

using LinearAlgebra
using StaticArrays

import ..InnerLayerModel, ..InnerLayerResponse, ..solve_inner

"""
    GGJModel{S} <: InnerLayerModel

Glasser–Greene–Johnson resistive inner-layer model. The type parameter `S`
selects the solver: `:galerkin` (default) for the Hermite-cubic finite element 
solver and `:shooting` for the backward stable-shoot solver. Both
implementations consume the same `inps` asymptotic-basis kernel and return
the parity-projected matching data.
"""
struct GGJModel{S} <: InnerLayerModel end

GGJModel(; solver::Symbol=:galerkin) = GGJModel{solver}()

include("GGJParameters.jl")
include("InnerAsymptotics.jl")
include("Reference.jl")
include("Shooting.jl")
include("Galerkin.jl")
include("LayerInputs.jl")

export GGJModel, GGJParameters
export mercier_di, mercier_dr, inner_Q, rescale_delta
export build_asymptotics, evaluate_asymptotics, pick_xmax
export InnerAsymptoticsCache
export glasser_wang_2020_eq55
export build_ggj_inputs
export NeoResistivityModel, SpitzerModel, SauterNeoModel, RedlNeoModel

end # module GGJ
