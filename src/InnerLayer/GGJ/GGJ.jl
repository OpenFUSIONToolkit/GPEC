# GGJ.jl
#
# Glasser–Greene–Johnson resistive inner-layer model, with three solvers selected by the
# `solver` type parameter of `GGJModel`:
#
#   - `:shooting`  – backward shoot from X_max → 0; valid only for |Q| ≪ 1 (tests and comparisons)
#   - `:galerkin`  – Hermite-cubic finite elements with resonant/extension matching cells
#                    (GWP2016 Sec. III); accuracy depends on the surface and arg Q
#   - `:ray`       – the same Hermite assembly on the rotated contour (Ray.jl), with the far-field
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
#             (Eqs. 1–55); implemented in InnerAsymptotics.jl.

module GGJ

using LinearAlgebra
using StaticArrays
using Random
using Printf
using DoubleFloats: Double64
using ...Utilities.HermiteFEM: hermite_basis, pack_grid

import ..InnerLayerModel, ..InnerLayerResponse, ..InnerLayerParameters
import ..solve_inner, ..solve_inner_profile

"""
    GGJModel{S,O} <: InnerLayerModel
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
struct GGJModel{S,O<:NamedTuple} <: InnerLayerModel
    options::O
end

function GGJModel(; solver::Symbol=:ray, options...)
    allowed = _ggj_options(solver)
    unknown = setdiff(keys(options), allowed)
    isempty(unknown) || throw(ArgumentError("GGJModel(solver=:$solver) does not take $(join(unknown, ", ")); its options are $(join(allowed, ", "))"))
    return GGJModel{solver,typeof(values(options))}(values(options))
end
GGJModel{S}() where {S} = GGJModel{S,@NamedTuple{}}((;))

# The backends implement the option-free model; a model carrying options forwards them.
const _BareGGJ{S} = GGJModel{S,@NamedTuple{}}
function solve_inner(m::GGJModel{S}, params, γ::Number; kwargs...) where {S}
    m isa _BareGGJ && throw(MethodError(solve_inner, (m, params, γ)))
    return solve_inner(GGJModel{S}(), params, γ; m.options..., kwargs...)
end
function solve_inner_profile(m::GGJModel{S}, params, γ::Number; kwargs...) where {S}
    m isa _BareGGJ && throw(MethodError(solve_inner_profile, (m, params, γ)))
    return solve_inner_profile(GGJModel{S}(), params, γ; m.options..., kwargs...)
end

include("GGJParameters.jl")
include("InnerAsymptotics.jl")
include("Reference.jl")
include("Shooting.jl")
include("Galerkin.jl")
include("Ray.jl")

# Each backend's option names: the keywords of the functions its solves forward them to.
_kwarg_names(f) = unique!(Symbol[k for m in methods(f) for k in Base.kwarg_decl(m) if !endswith(string(k), "...")])
function _ggj_options(solver::Symbol)
    solver === :ray && return union(_kwarg_names(solve_ray), (:real_profiles, :npc, :certify_rtol))
    solver === :galerkin && return _kwarg_names(_galerkin_solve)
    solver === :shooting && return Base.kwarg_decl(which(solve_inner, (_BareGGJ{:shooting}, GGJParameters, ComplexF64)))
    throw(ArgumentError("GGJModel: unknown solver :$solver (expected :ray, :galerkin or :shooting)"))
end

export GGJModel, GGJParameters
export mercier_di, mercier_dr, inner_Q, rescale_delta
export solve_ray, pick_smax, delta_convergence, solution_profile, asymptotic_profile

end # module GGJ
