module DCON

# All imports and includes for the DCON module
using LinearAlgebra
using LinearAlgebra.LAPACK
using TOML
using FFTW
using OrdinaryDiffEq
using HDF5
using JLD2

import ..Equilibrium
import ..Spl
import ..Vacuum
using Printf
import StaticArrays: @MVector, @MMatrix

# Include all necessary files
include("DconStructs.jl")
include("Kinetic.jl")
include("Main.jl")
include("Mercier.jl")
include("Ode.jl")
include("Sing.jl")
include("Fourfit.jl")
include("FixedBoundaryStability.jl")
include("Utils.jl")
include("Free.jl")

# These are used for various small tolerances and root finders throughout DCON
global eps = 1e-10
global itmax = 50

end
