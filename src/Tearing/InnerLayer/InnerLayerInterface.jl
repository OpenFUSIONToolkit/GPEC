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
    InnerLayerResponse

Parity-projected inner-layer matching data at one rational surface. The two
components correspond to the homogeneous parity solutions of the half-domain
inner-layer problem (parity boundary conditions imposed at X = 0). They are
the `Δ_{j,±}(γ)` of Glasser, Wang & Park, Phys. Plasmas **23**, 112506
(2016), Eqs. (34)–(35).

# Fields

  - `tearing` — the **odd-parity** matching coefficient (GWP Δ_+; Fortran
    `rmatch/deltac.f` "odd mode"). Corresponds to a flux perturbation W
    that is EVEN in x and a velocity/temperature perturbation that is ODD
    — i.e., the reconnecting mode with a current sheet at the rational
    surface. This is the tearing drive that appears as Δ' in the
    classical constant-ψ tearing equation. Must be populated by every
    resistive inner-layer model.

  - `interchange` — the **even-parity** matching coefficient (GWP Δ_−;
    Fortran `rmatch/deltac.f` "even mode"). Corresponds to W odd, N and
    Θ even — i.e., the non-reconnecting interchange/ballooning channel.
    Its dissipative piece in toroidal geometry is the Glasser, Greene &
    Johnson stabilization term that opposes tearing growth (Glasser 1975;
    Lütjens-Bondeson-Roy 1993). Pressureless inner-layer models (e.g.
    SLAYER's Fitzpatrick Riccati) set this identically zero.

The naming follows the physics channel rather than a mathematical
parity label because `odd/even` carries different meanings across the
literature depending on whether you label by the parity of W (GWP paper
convention) or the parity of (N, Θ) (Fortran `rmatch/deltac.f`
convention). Using `tearing` and `interchange` avoids ambiguity.
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
