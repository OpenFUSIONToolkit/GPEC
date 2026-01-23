module PENTRC 

using LinearAlgebra
using LinearAlgebra.LAPACK
using TOML
using FFTW
using OrdinaryDiffEq
using HDF5
using Printf
using Statistics

import ..Spl
import ..DCON
import ..Equilibrium

include("Torque.jl")
include("Energy.jl")
include("grid_mod.jl")
include("Pitch.jl")
include("Utils.jl")
include("Main.jl")

global mp = 1.672_614e-27      # proton mass (kg)
global me = 9.109_1e-31        # electron mass (kg)
global e  = 1.602_191_7e-19    # elementary charge (C)
global eV = e                  # joules per electron-volt

global π     = 3.141_592_653_589_793
global twopi = 2π
global μ₀    = 4e-7 * π
global rad2deg = 180 / π
global deg2rad = π / 180
global iunit = 1im               # equivalent to Fortran's (0,1)


# # Method labels
# const METHODS = [
#     "fgar","tgar","pgar","rlar","clar","fcgl",
#     "fwmm","twmm","pwmm","ftmm","ttmm","ptmm",
#     "fkmm","tkmm","pkmm","frmm","trmm","prmm",
# ]

# # Grid types
# const GRIDS = [
#     "lsode",
#     "equil",
#     "input",
# ]

# # Human-readable descriptions
# const DOCS = [
#     "Full general-aspect-ratio calculation",
#     "Trapped particle general-aspect-ratio calculation",
#     "Passing particle general-aspect-ratio calculation",
#     "Trapped particle large-aspect-ratio calculation",
#     "Trapped particle cylindrical large-aspect-ratio calculation",
#     "Fluid Chew–Goldberger–Low calculation",
#     "Full energy calculation using MXM Euler–Lagrange matrix",
#     "Trapped energy calculation using MXM Euler–Lagrange matrix",
#     "Passing energy calculation using MXM Euler–Lagrange matrix",
#     "Full torque calculation using MXM Euler–Lagrange matrix",
#     "Trapped torque calculation using MXM Euler–Lagrange matrix",
#     "Passing torque calculation using MXM Euler–Lagrange matrix",
#     "Full MXM Euler–Lagrange energy matrix norm calculation",
#     "Trapped MXM Euler–Lagrange energy matrix norm calculation",
#     "Passing MXM Euler–Lagrange energy matrix norm calculation",
#     "Full MXM Euler–Lagrange torque matrix norm calculation",
#     "Trapped MXM Euler–Lagrange torque matrix norm calculation",
#     "Passing MXM Euler–Lagrange torque matrix norm calculation",
# ]


end