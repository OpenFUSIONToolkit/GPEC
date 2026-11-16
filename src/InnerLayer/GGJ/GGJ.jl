# GGJ.jl
#
# Glasser–Greene–Johnson resistive inner-layer model, the reference ToroidalLayer model: its equations
# and Wasow basis (Equations.jl) drive the shared ray and Galerkin solvers. Three solvers are selected by
# the `solver` type parameter of `GGJModel`:
#
#   - `:shooting`  – backward shoot from X_max → 0; valid only for |Q| ≪ 1 (tests and comparisons)
#   - `:galerkin`  – Hermite-cubic finite elements with resonant/extension matching cells
#                    (GWP2016 Sec. III); accuracy depends on the surface and arg Q
#   - `:ray`       – the same Hermite assembly on the rotated contour, with the far-field
#                    condition marched in from where the series converges; robust to |Q| ~ 500
#
# They return the matching data Δ± of GWP2016 Eqs. (34)–(35). Parity is that of the displacement Ξ:
# Δ₊ (even) is the interchange channel, Δ₋ (odd) the tearing channel. Positional pairs are always
# ordered (Δ₊, Δ₋), as d₊, d₋ in GWP2016 Eq. (38) and Pletzer, Bondeson & Dewar (1994) Eqs. 83, 85.
#
# Main source papers, cited by equation number throughout:
#
#   GWP2016 — A. H. Glasser, Z. R. Wang & J.-K. Park, "Computation of resistive
#             instabilities by matched asymptotic expansions", Phys. Plasmas 23,
#             112506 (2016). Inner-region equations (Eq. 11), singular-Galerkin method
#             (Sec. III), matching data (Eqs. 34–35), scale factors (Appendix, Eqs. A8–A16).
#
#   GW2020  — A. H. Glasser & Z. R. Wang, "Asymptotic solutions and convergence
#             studies of the resistive inner region equations", Phys. Plasmas 27,
#             012506 (2020). Wasow construction of the large-x asymptotic basis
#             (Eqs. 1–55); recursion in ToroidalLayer/InnerAsymptotics.jl, GGJ matrices in Equations.jl.

module GGJ

using LinearAlgebra
using StaticArrays
using DoubleFloats: Double64

import ..InnerLayerResponse
import ..solve_inner, ..solve_inner_profile
using ..ToroidalLayer
import ..ToroidalLayer: inner_Q, rescale_delta, profile_conversions, power_exponents
import ..ToroidalLayer: layer_rows, layer_mass, wasow_basis, split_block, default_ray_angle, parabolic_rate, ohm_rate, fast_rates

"""
    GGJModel{S,O} <: ToroidalLayerModel{S}
    GGJModel(; solver=:ray, options...)

Glasser–Greene–Johnson resistive inner-layer model. `S` selects the solver
backend: `:ray` (default; rotated contour with an inward-marched far-field condition,
robust at large |Q|), `:galerkin` (real-axis Hermite FEM with resonant/extension matching
cells; accuracy depends on the surface and arg Q, even at small |Q|), or `:shooting` (|Q| ≪ 1 only). Backend keywords given at
construction, e.g. `GGJModel(; solver=:galerkin, nx=1280)`, apply to every solve; a keyword the backend does not
take is an error here, not at the first solve.

## Fields

  - `options::NamedTuple` - Backend keywords forwarded to every solve; call-site keywords win.
"""
struct GGJModel{S,O<:NamedTuple} <: ToroidalLayerModel{S}
    options::O
end

function GGJModel(; solver::Symbol=:ray, options...)
    allowed = _ggj_options(solver)
    unknown = setdiff(keys(options), allowed)
    isempty(unknown) || throw(ArgumentError("GGJModel(solver=:$solver) does not take $(join(unknown, ", ")); its options are $(join(allowed, ", "))"))
    return GGJModel{solver,typeof(values(options))}(values(options))
end
GGJModel{S}() where {S} = GGJModel{S,@NamedTuple{}}((;))

# The option-free model, which the shooting solve implements.
const _BareGGJ{S} = GGJModel{S,@NamedTuple{}}

include("GGJParameters.jl")
include("Equations.jl")
include("Reference.jl")
include("Shooting.jl")

# A shooting model carrying options forwards them to the option-free one.
function solve_inner(m::GGJModel{:shooting}, params::GGJParameters, γ::Number; kwargs...)
    m isa _BareGGJ && throw(MethodError(solve_inner, (m, params, γ)))
    return solve_inner(GGJModel{:shooting}(), params, γ; m.options..., kwargs...)
end

# Each backend's option names: the shared backends' keywords, or the shooting solve's.
function _ggj_options(solver::Symbol)
    solver in (:ray, :galerkin) && return backend_options(solver)
    solver === :shooting && return Base.kwarg_decl(which(solve_inner, (_BareGGJ{:shooting}, GGJParameters, ComplexF64)))
    throw(ArgumentError("GGJModel: unknown solver :$solver (expected :ray, :galerkin or :shooting)"))
end

export GGJModel, GGJParameters
export mercier_di, mercier_dr, inner_Q, rescale_delta

end # module GGJ
