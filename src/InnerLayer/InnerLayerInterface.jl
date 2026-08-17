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

  - `tearing` — the **odd-parity** matching coefficient (GWP Δ_+, the
    "odd mode"). Corresponds to a flux perturbation W that is EVEN in x and
    a velocity/temperature perturbation that is ODD — i.e., the
    reconnecting mode with a current sheet at the rational surface. This is
    the tearing drive that appears as Δ' in the classical constant-ψ
    tearing equation. Must be populated by every resistive inner-layer model.

  - `interchange` — the **even-parity** matching coefficient (GWP Δ_−, the
    "even mode"). Corresponds to W odd, N and Θ even — i.e., the
    non-reconnecting interchange/ballooning channel. Its dissipative piece
    in toroidal geometry is the Glasser, Greene & Johnson stabilization
    term that opposes tearing growth (Glasser, Greene & Johnson 1975;
    Lütjens-Bondeson-Roy 1993). Pressureless inner-layer models (e.g.
    SLAYER's Fitzpatrick Riccati) set this identically zero.

The naming follows the physics channel rather than a mathematical parity
label because `odd/even` carries different meanings across the literature
depending on whether you label by the parity of W (GWP paper convention)
or the parity of (N, Θ). Using `tearing` and `interchange` avoids ambiguity.
"""
struct InnerLayerResponse
    tearing::ComplexF64
    interchange::ComplexF64
end

InnerLayerResponse(; tearing::Number=0, interchange::Number=0) =
    InnerLayerResponse(ComplexF64(tearing), ComplexF64(interchange))

"""
    solve_inner(model::InnerLayerModel, params, γ::Number; kwargs...) -> InnerLayerResponse

Compute the parity-projected matching data `(Δ_tearing, Δ_interchange)` for
the given inner-layer `model`, physical parameters `params`, and complex
growth rate `γ`. Concrete models specialize this function.

See `InnerLayerResponse` for the physics-oriented field definitions.
Pressureless models (SLAYER) populate only `tearing` and leave
`interchange` at zero; two-fluid / finite-β models (GGJ) populate both.
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
