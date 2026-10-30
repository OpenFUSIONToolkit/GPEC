# InnerLayerInterface.jl
#
# Abstract interface for resistive inner-layer models. Concrete models (GGJ, SLAYER) live in
# submodules and specialize `solve_inner`.

"""
    InnerLayerModel

Abstract supertype for resistive inner-layer models. Each concrete model is a
small, parameter-free type tag (often parameterized by a solver-choice symbol)
that selects a `solve_inner` method.

Implementations live in submodules of `InnerLayer`, e.g. `InnerLayer.GGJ`.
"""
abstract type InnerLayerModel end

"""
    InnerLayerParameters

Abstract supertype for the per-surface physical-parameter structs consumed by
the inner-layer models (`SLAYERParameters`, `GGJParameters`). Lets the
dispersion runner and HDF5 output dispatch generically over a vector of
single-model parameters.
"""
abstract type InnerLayerParameters end

"""
    InnerLayerResponse

Parity-projected inner-layer matching data at one rational surface. The two
components correspond to the homogeneous parity solutions of the half-domain
inner-layer problem (parity boundary conditions imposed at X = 0). They are
the `Δ_{j,±}(γ)` of Glasser, Wang & Park, Phys. Plasmas **23**, 112506
(2016), Eqs. (34)–(35).

# Fields

  - `tearing` — odd-parity matching coefficient Δ₋: Ξ odd, Ψ even, the reconnecting channel
    whose Δ is the constant-ψ tearing Δ′ (Pletzer, Bondeson & Dewar, J. Comput. Phys. 115, 530
    (1994), Eqs. 77, 86–87, 89). Every resistive inner-layer model populates it.
  - `interchange` — even-parity matching coefficient Δ₊: Ξ even, Ψ odd, the non-reconnecting
    interchange channel carrying the Glasser, Greene & Johnson (1975) stabilization.
    Pressureless models (SLAYER) set it to zero.

Parity is that of the displacement Ξ, matched to the outer ξ (GWP2016 Eq. 34; Pletzer,
Bondeson & Dewar Eqs. 32–33), with + even and − odd. The fields are named by the physics
channel because the parity of the flux Ψ is the opposite. Positional
pairs elsewhere in `InnerLayer` (e.g. the `Δ` of [`solve_inner_profile`](@ref)) are
ordered (Δ₊, Δ₋) = (interchange, tearing).
"""
struct InnerLayerResponse
    tearing::ComplexF64
    interchange::ComplexF64
end

InnerLayerResponse(; tearing::Number=0, interchange::Number=0) =
    InnerLayerResponse(ComplexF64(tearing), ComplexF64(interchange))

"""
    solve_inner(model::InnerLayerModel, params, s::Number; kwargs...) -> InnerLayerResponse

Compute the parity-projected matching data (the `tearing` and `interchange` fields of
`InnerLayerResponse`) for the given inner-layer `model`, physical parameters `params`, and complex
growth rate `s = γ − iω` in 1/s, in the surface's E×B frame. Every model takes this same `s`.
Concrete models specialize this function.

See `InnerLayerResponse` for the physics-oriented field definitions.
Pressureless models (SLAYER) populate only `tearing` and leave
`interchange` at zero; finite-pressure models (GGJ) populate both.
"""
function solve_inner end

"""
    layer_response(model::InnerLayerModel, params, z::Number; kwargs...) -> InnerLayerResponse

[`solve_inner`](@ref) in the variable `model`'s dispersion scans run in, so the layer is evaluated at exactly the value
scanned: the growth rate `s` itself for GGJ, and Q = (ω + iγ)·τ_k = i·s·τ_k for SLAYER.
"""
layer_response(model::InnerLayerModel, params, z::Number; kwargs...) = solve_inner(model, params, z; kwargs...)

"""
    solve_inner_profile(model::InnerLayerModel, params, γ::Number; kwargs...)
        -> (; Δ, x, Ψ, Ξ, dψdx, rescale, ...)

Compute the inner-layer matching data **and** the reconstructed layer field
profiles for the given `model` — everything an outer↔inner matching driver
needs from the layer, so drivers never touch model internals. Returns a named
tuple with at least:

  - `Δ`       — the same matching data as [`solve_inner`](@ref), as the pair
    (Δ₊, Δ₋) = (interchange, tearing)
  - `x`       — real ascending grid in the model's stretched inner coordinate,
    `x ≥ 0` with the rational surface at `x = 0`
  - `Ψ`, `Ξ`  — `length(x) × 2` profiles, columns in the order of `Δ`, in the
    model's inner normalization: `Ψ` the normal-field
    (reconnected-flux) variable, `Ξ` the displacement
  - `dψdx`    — conversion to poloidal-flux distance, `δψ = dψdx · x`
  - `rescale` — amplitude factor converting the inner-normalized profiles to
    the outer δψ-normalized convention (companion of the Δ rescale)

Concrete models may return additional diagnostic fields (e.g. a solve-quality
certificate). Solver-knob keywords are model-specific.
"""
function solve_inner_profile end
