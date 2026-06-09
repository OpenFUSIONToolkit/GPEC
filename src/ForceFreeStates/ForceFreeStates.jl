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
using QuadGK: quadgk

import ..Equilibrium
import ..Utilities
import ..Vacuum
using Printf
using DoubleFloats
import StaticArrays: @MMatrix

# Include all necessary files
include("ForceFreeStatesStructs.jl")
include("Mercier.jl")
include("Bal.jl")
include("EulerLagrange.jl")
include("Sing.jl")
include("Fourfit.jl")
include("FixedKineticMatrices.jl")
include("Kinetic.jl")
include("FixedBoundaryStability.jl")
include("Utils.jl")
include("PowerNorm.jl")
include("Free.jl")
include("Riccati.jl")

# RDCON outer-region singular Galerkin Δ′ solver (gal_solve port)
include("Galerkin/GalerkinStructs.jl")
include("Galerkin/GalerkinGrid.jl")
include("Galerkin/GalerkinAssembly.jl")
include("Galerkin/GalerkinSolve.jl")

# These are used for various small tolerances and root finders throughout ForceFreeStates
global eps = 1e-10
global itmax = 50

end
