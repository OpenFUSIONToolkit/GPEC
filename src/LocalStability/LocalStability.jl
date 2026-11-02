module LocalStability

# Local high-n stability: Mercier D_I, resistive interchange D_R, and the ballooning Δ'
# scans. Depends only on Equilibrium (plus math libraries) — no stability-solver state.
using LinearAlgebra
using FFTW
using OrdinaryDiffEq
using FastInterpolations
using StaticArrays: SVector
import CommonSolve

import ..Equilibrium

include("Ballooning.jl")
include("Problem.jl")

export LocalStabilityProblem, Ballooning, LocalStabilityResult, mercier_D_I
export compute_ballooning_stability!, ballooning_alpha_boundary, ballooning_alpha_boundaries

include("Output.jl")

end
