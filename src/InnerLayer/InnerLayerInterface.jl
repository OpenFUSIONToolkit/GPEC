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
    solve_inner_profile(model::InnerLayerModel, params, γ::Number; kwargs...)
        -> (; Δ, x, Ψ, Ξ, dψdx, rescale, ...)

Compute the inner-layer matching data **and** the reconstructed layer field
profiles for the given `model` — everything an outer↔inner matching driver
needs from the layer, so drivers never touch model internals. Returns a named
tuple with at least:

  - `Δ`       — the same `(Δ_odd, Δ_even)` matching data as [`solve_inner`](@ref)
  - `x`       — real ascending grid in the model's stretched inner coordinate,
                `x ≥ 0` with the rational surface at `x = 0`
  - `Ψ`, `Ξ`  — `length(x) × 2` profiles, columns (odd, even) parity, in the
                model's inner normalization: `Ψ` the normal-field
                (reconnected-flux) variable, `Ξ` the displacement
  - `dψdx`    — conversion to poloidal-flux distance, `δψ = dψdx · x`
  - `rescale` — amplitude factor converting the inner-normalized profiles to
                the outer δψ-normalized convention (companion of the Δ rescale)

Concrete models may return additional diagnostic fields (e.g. a solve-quality
certificate). Solver-knob keywords are model-specific.
"""
function solve_inner_profile end
