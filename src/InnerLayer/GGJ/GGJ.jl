# GGJ.jl
#
# Glasser–Greene–Johnson resistive inner-layer model. Provides three
# interchangeable solvers selected via the `solver` type-parameter of
# `GGJModel`:
#
#   - `:shooting`  – stable backward shoot from X_max → 0;
#                    numerically stable only for |Q| ≪ 1, should not be used
#   - `:galerkin`  – Hermite-cubic finite element method;
#                    real-axis method, degrades for |Q| ≳ 1 off the real axis
#   - `:ray`       – rotated-contour spectral-element collocation (Ray.jl);
#                    robust to |Q| ~ 500 on/near the imaginary axis
#
# They return the parity-projected matching data
# `(Δ_odd, Δ_even)` of GWP2016 Eqs. (34)–(35) in the same (deltac) convention.
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
using SparseArrays
using Random
using Printf
using DoubleFloats: Double64

import ..InnerLayerModel, ..solve_inner, ..solve_inner_profile

"""
    GGJModel{S} <: InnerLayerModel

Glasser–Greene–Johnson resistive inner-layer model. `S` selects the solver
backend: `:ray` (default; robust at large |Q| on/near the imaginary axis),
`:galerkin` (real-axis Hermite FEM; degrades for |Q| ≳ 1), or `:shooting`
(|Q| ≪ 1 only). The backends take different numerical-knob keywords.
"""
struct GGJModel{S} <: InnerLayerModel end

GGJModel(; solver::Symbol=:ray) = GGJModel{solver}()

include("GGJParameters.jl")
include("InnerAsymptotics.jl")
include("RayAsymptotics.jl")
include("Reference.jl")
include("Shooting.jl")
include("Galerkin.jl")
include("Ray.jl")

export GGJModel, GGJParameters
export mercier_di, mercier_dr, inner_Q, rescale_delta
export build_asymptotics, evaluate_asymptotics, pick_xmax
export InnerAsymptoticsCache
export glasser_wang_2020_eq55
export solve_ray, RaySolveResult, pick_smax, physical_ua_dua
export delta_convergence, solution_profile, asymptotic_profile, q4_surface_benchmark

end # module GGJ
