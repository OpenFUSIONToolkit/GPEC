module LocalStability

# Local high-n stability: Mercier D_I, resistive interchange D_R, and the ballooning Δ'
# scans. Depends only on Equilibrium (plus math libraries) — no stability-solver state.
using LinearAlgebra
using FFTW
using OrdinaryDiffEq
using FastInterpolations
using StaticArrays: SVector

import ..Equilibrium

include("Ballooning.jl")

export compute_local_stability, compute_ballooning_stability!, ballooning_alpha_boundary, ballooning_alpha_boundaries

end
