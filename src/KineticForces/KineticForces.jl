module KineticForces

"""
KineticForces - neoclassical toroidal viscosity (NTV) torque and kinetic energy of a perturbed equilibrium.

Computes the torque by bounce-averaged pitch-angle and energy integration, and supplies the calculated kinetic
matrices to ForceFreeStates when the problem's kinetic source is a `KineticForcesControl`.
"""

using LinearAlgebra
using LinearAlgebra.LAPACK
using FFTW
using QuadGK
using HDF5
using Printf
using Statistics

using FastInterpolations
using Roots
import ..ForceFreeStates
import ..Equilibrium
using ..Equilibrium: IonSpecies
import ..Utilities
import ..PerturbedEquilibrium

# Supporting data structures and utilities
include("KineticForcesStructs.jl")
include("Utils.jl")
include("Output.jl")

# Core library functions
include("BounceAveraging.jl")
include("Torque.jl")
include("EnergyIntegration.jl")
include("PitchIntegration.jl")

# High-level computation functions
include("Compute.jl")
include("CalculatedKineticMatrices.jl")

# Global constants
const mp = 1.672_614e-27      # proton mass (kg)
const me = 9.109_1e-31        # electron mass (kg)
const e  = 1.602_191_7e-19    # elementary charge (C)
const eV = e                  # joules per electron-volt

const twopi = 2π
const μ₀    = 4e-7 * π
const rad2deg = 180 / π
const deg2rad = π / 180

# Treat-as-zero threshold for singular denominators and degenerate coefficients.
const SINGULAR_EPS = 1e-30

export kinetic_torque, KineticForcesResult

# Diagnostic exports for interactive energy-integrand inspection
export evaluate_energy_integrand, energy_integrand_scalar, EnergyParams

end  # module KineticForces
