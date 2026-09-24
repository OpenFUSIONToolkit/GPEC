module ForceFreeStates

# All imports and includes for the ForceFreeStates module
using LinearAlgebra
using LinearAlgebra.LAPACK
using TOML
using FFTW
using OrdinaryDiffEq
using HDF5
using JLD2
using FastInterpolations
using AdaptiveArrayPools
using Roots
using FastGaussQuadrature: gausslobatto
using QuadGK: quadgk, quadgk!

import ..Equilibrium

"""
Explicit Runge-Kutta methods benchmarked for the Euler-Lagrange solves; `ode_solver` must name one.
"""
const EL_ODE_SOLVERS = (Vern6=Vern6(), Vern7=Vern7(), Vern8=Vern8(), Vern9=Vern9(), DP8=DP8())

"""
    el_ode_algorithm(ctrl) -> OrdinaryDiffEq algorithm

The solver named by `ctrl.ode_solver`, from [`EL_ODE_SOLVERS`](@ref).
"""
function el_ode_algorithm(ctrl)
    name = Symbol(ctrl.ode_solver)
    haskey(EL_ODE_SOLVERS, name) ||
        throw(ArgumentError("ode_solver = \"$(ctrl.ode_solver)\" is not one of the benchmarked Euler-Lagrange solvers $(keys(EL_ODE_SOLVERS))"))
    return EL_ODE_SOLVERS[name]
end
import ..Utilities
import ..Vacuum
import ..InnerLayer
using Printf
using DoubleFloats
import StaticArrays: @MMatrix

# Types with cross-subsystem consumers, loaded before the code that dispatches on them
include("Surfaces/Types.jl")
include("CoreTypes.jl")
include("Riccati/Types.jl")
include("Fourfit.jl")
include("Matching/DeltaPrime.jl")

include("EulerLagrange.jl")

# Singular-surface machinery: finding/filtering, Frobenius asymptotics, GGJ coefficients
include("Surfaces/Finding.jl")
include("Surfaces/Asymptotics.jl")
include("Surfaces/Resist.jl")
include("Surfaces/ResistEval.jl")

# Outer<->inner resistive matching
include("Matching/ResonantMatch.jl")

include("FixedKineticMatrices.jl")
include("Kinetic.jl")
include("FixedBoundaryStability.jl")
include("Utils.jl")
include("Free.jl")

# Chunked fundamental-matrix (Riccati/STRIDE) driver
include("Riccati/Propagators.jl")
include("Riccati/Crossings.jl")
include("Riccati/DeltaPrimeBVP.jl")
include("Riccati/Driver.jl")

# RDCON outer-region singular Galerkin Δ′ solver (gal_solve port)
include("Galerkin/GalerkinStructs.jl")
include("Galerkin/GalerkinGrid.jl")
include("Galerkin/GalerkinAssembly.jl")
include("Galerkin/GalerkinSolution.jl")
include("Galerkin/GalerkinMatch.jl")
include("Galerkin/GalerkinSolve.jl")

# Scripting-API integrator selectors: pure configuration translated onto ForceFreeStatesControl.
include("Integrators.jl")

# The published solve product; last, so it can name every type the stages above define.
include("Result.jl")

# These are used for various small tolerances and root finders throughout ForceFreeStates
global eps = 1e-10
global itmax = 50

end
