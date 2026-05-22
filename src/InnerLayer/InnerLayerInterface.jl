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

"""
    solve_inner_full(model::InnerLayerModel, params, γ::ComplexF64; kwargs...)

Compute the **full-domain** matching data for an inner-layer `model` on
X ∈ [−Xmax, +Xmax], with asymptotic matching at both ends and no parity
reduction at the rational surface (X = 0). Returns a model-defined matching
object (e.g. `WasowGalerkin.FullDomainMatching`) that generalizes — and for a
parity-symmetric system recombines to — the half-domain `(Δ_odd, Δ_even)` pair
returned by [`solve_inner`](@ref).

This path is required when the inner-layer physics breaks the X → −X parity of
the single-fluid system (e.g. a rotating two-fluid drift-MHD layer), so the
even/odd decomposition no longer applies. Concrete models specialize this
function.
"""
function solve_inner_full end
