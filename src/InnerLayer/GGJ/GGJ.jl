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
import ..ToroidalLayer: solve_ray, pick_smax, delta_convergence, solution_profile, asymptotic_profile
import ..ToroidalLayer: layer_rows, layer_weak_form, wasow_basis, split_block, default_ray_angle, parabolic_rate, ohm_rate, fast_rates
# Also reached as GGJ.<name> by callers and tests.
import ..ToroidalLayer: ode_matrix, decaying_pair, RaySolveResult

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
include("Equations.jl")
include("Reference.jl")
include("Shooting.jl")

"""
    solve_inner(::GGJModel{:ray}, params::GGJParameters, γ::Number; kwargs...) -> InnerLayerResponse

Rotated-ray backend ([`ray_response`](@ref)): converts γ via `inner_Q`, forwards keywords to
[`solve_ray`](@ref), and returns the parity-projected matching data (Δ₊, Δ₋) as the
`interchange` and `tearing` fields of [`InnerLayerResponse`](@ref).
Preferred for |Q| ≳ 1 and near the imaginary axis; use `solve_ray` directly
when the full [`RaySolveResult`](@ref) is wanted. `real_profiles`, `npc` and `certify_rtol` are
profile-only model options (see `solve_inner_profile`) and are ignored here.
"""
solve_inner(::_BareGGJ{:ray}, params::GGJParameters, γ::Number; kwargs...) = ray_response(params, γ; kwargs...)

"""
    solve_inner_profile(::GGJModel{:ray}, params::GGJParameters, γ::Number;
                        real_profiles=false, npc=8, certify_rtol=1e-3, kwargs...)
        -> (; Δ, x, Ψ, Ξ, dψdx, rescale, certΔ)

Rotated-ray implementation of the [`solve_inner_profile`](@ref) interface ([`ray_profile`](@ref)). `Δ` and the
layer-center fields come from the solve at θ = arg(Q)/4; x = 0 lies on every contour, so
Ψ(0), Ξ(0) are the real-axis values, and by default only they are returned (`x = [0.0]`,
`certΔ = NaN`). With `real_profiles = true` the fields across the layer (x > 0) come from an
extra θ = 0 solve (`npc` points per cell), a diagnostic; the x = 0 row stays the rotated
solve's, and `certΔ` is the largest relative difference of the two solves' Δ₊, Δ₋, warned
above `certify_rtol`. Other keywords forward to [`solve_ray`](@ref), except θ.
"""
solve_inner_profile(::_BareGGJ{:ray}, params::GGJParameters, γ::Number; kwargs...) = ray_profile(params, γ; kwargs...)

"""
    solve_inner(::GGJModel{:galerkin}, params::GGJParameters, γ::Number;
                kmax::Int=8, nx::Int=512, nq::Int=4, pfac::Float64=1.0,
                cutoff::Int=5, xfac::Float64=1.0, tol_res::Float64=1e-5)
                -> InnerLayerResponse

Solve the GGJ inner-layer matching problem using the Hermite-cubic finite
element (Galerkin) method (GWP2016 Sec. III), in the "resonant + noexp + inps"
configuration ([`galerkin_response`](@ref)).

Returns the parity-projected matching data (GWP2016 Eqs. 34–35), rescaled by
[`rescale_delta`](@ref), as an `InnerLayerResponse`: `interchange` is the even-parity Δ₊
(isol = 1), `tearing` the odd-parity Δ₋ (isol = 2).

Knobs: `kmax` series order; `nx` cell count; `nq` Gauss–Lobatto order; `pfac` grid packing;
`cutoff` extension-cell span (cells nx−cutoff−1 … nx−1); `xfac` scale on the two inner matching radii; `tol_res`
resonant-cell quadrature tolerance.
"""
solve_inner(::_BareGGJ{:galerkin}, params::GGJParameters, γ::Number; kwargs...) = galerkin_response(params, γ; kwargs...)

"""
    solve_inner_profile(::GGJModel{:galerkin}, params::GGJParameters, γ::Number; kwargs...)
        -> (; Δ, x, Ψ, Ξ, dψdx, rescale)

Hermite-FEM implementation of the [`solve_inner_profile`](@ref) interface ([`galerkin_profile`](@ref)):
real-axis solve, so `Δ` and the profiles come from the same solution. Same
numerics/kwargs as `solve_inner(GGJModel(; solver=:galerkin), ...)`.
"""
solve_inner_profile(::_BareGGJ{:galerkin}, params::GGJParameters, γ::Number; kwargs...) = galerkin_profile(params, γ; kwargs...)

# Each backend's option names: the shared backends' keywords, or the shooting solve's.
function _ggj_options(solver::Symbol)
    solver in (:ray, :galerkin) && return backend_options(solver)
    solver === :shooting && return Base.kwarg_decl(which(solve_inner, (_BareGGJ{:shooting}, GGJParameters, ComplexF64)))
    throw(ArgumentError("GGJModel: unknown solver :$solver (expected :ray, :galerkin or :shooting)"))
end

export GGJModel, GGJParameters
export mercier_di, mercier_dr, inner_Q, rescale_delta
export solve_ray, pick_smax, delta_convergence, solution_profile, asymptotic_profile

end # module GGJ
