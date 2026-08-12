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
# the parity-projected matching data `(Δ_odd, Δ_even)` of GWP2016 Eqs. (34)–(35).
#
# Equation references throughout this module use two source papers:
#
#   GWP2016 — A. H. Glasser, Z. R. Wang & J.-K. Park, "Computation of resistive
#             instabilities by matched asymptotic expansions", Phys. Plasmas 23,
#             112506 (2016). Inner-region equations (Eq. 11), matrix form
#             A Ψ'' + B Ψ' + C Ψ = 0 (Eqs. 12–15), singular-Galerkin weak form
#             (Eq. 32), grid packing (Eq. 33), matching data (Eqs. 34–35),
#             dimensionless parameters / scale factors (Appendix, Eqs. A8–A15).
#
#   GW2020  — A. H. Glasser & Z. R. Wang, "Asymptotic solutions and convergence
#             studies of the resistive inner region equations", Phys. Plasmas 27,
#             012506 (2020). Wasow construction of the large-x asymptotic basis
#             (Eqs. 1–55); implemented in InnerAsymptotics.jl.
#
# The inner-region equations are identical in both: GW2020 Eq. (1) ≡ GWP2016 Eq. (11).

module GGJ

using LinearAlgebra
using StaticArrays

import ..InnerLayerModel, ..InnerLayerResponse, ..solve_inner, ..InnerLayerParameters

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

export GGJModel, GGJParameters
export mercier_di, mercier_dr, inner_Q, rescale_delta
export build_asymptotics, evaluate_asymptotics, pick_xmax
export InnerAsymptoticsCache
export glasser_wang_2020_eq55

end # module GGJ
