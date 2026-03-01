module DCON

# All imports and includes for the DCON module
using LinearAlgebra
using LinearAlgebra.LAPACK
using TOML
using FFTW
using OrdinaryDiffEq
using HDF5
using JLD2
using FastInterpolations

import ..Equilibrium
import ..Spl
import ..Util
import ..Vacuum
using Printf
import StaticArrays: @MVector, @MMatrix
# TODO (IMAS Edit #1): Add IMASdd import here
#   Add: import IMASdd

# Include all necessary files
include("DconStructs.jl")
include("Main.jl")
include("Mercier.jl")
include("Bal.jl")
include("EulerLagrange.jl")
include("Sing.jl")
include("Fourfit.jl")
include("FixedBoundaryStability.jl")
include("Utils.jl")
include("Free.jl")
# TODO (IMAS Edit #2): Include WriteImas.jl here
#   Add: include("WriteImas.jl")

# These are used for various small tolerances and root finders throughout DCON
global eps = 1e-10
global itmax = 50

end
