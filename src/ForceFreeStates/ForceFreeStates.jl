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
using FastInterpolations: cubic_interp, deriv1, PeriodicBC, LinearBinary

import ..Equilibrium
import ..Utilities
import ..Vacuum
using Printf
import StaticArrays: @MMatrix

# Include all necessary files
include("ForceFreeStatesStructs.jl")
include("Mercier.jl")
include("Bal.jl")
include("EulerLagrange.jl")
include("Sing.jl")
include("Fourfit.jl")
include("FixedBoundaryStability.jl")
include("Utils.jl")
include("Free.jl")

# These are used for various small tolerances and root finders throughout ForceFreeStates
global eps = 1e-10
global itmax = 50

end
