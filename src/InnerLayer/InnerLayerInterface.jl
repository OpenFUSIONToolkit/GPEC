# InnerLayerInterface.jl
#
# Abstract interface for resistive inner-layer models. Concrete models
# (e.g. GGJ, SLAYER, kinetic) live in submodules and specialize `solve_inner`.

"""
    InnerLayerModel

Abstract supertype for resistive inner-layer models. Each concrete model is a
small, parameter-free type tag (often parameterized by a solver-choice symbol)
that selects a `solve_inner` method.

Implementations live in submodules of `InnerLayer`, e.g. `InnerLayer.GGJ`.
"""
abstract type InnerLayerModel end

"""
    solve_inner(model::InnerLayerModel, params, γ::ComplexF64; kwargs...) -> SVector{2,ComplexF64}

Compute the parity-projected matching data `(Δ_odd, Δ_even)` for the given
inner-layer `model`, physical parameters `params`, and complex growth rate
`γ`. Concrete models specialize this function.

The two returned components correspond to the homogeneous odd / even parity
solutions of the half-domain inner-layer problem (parity boundary conditions
imposed at the rational surface, X = 0). They are the Δ_{j,±}(γ) of
Glasser, Wang & Park, Phys. Plasmas **23**, 112506 (2016), Eqs. (34)–(35).
"""
function solve_inner end
