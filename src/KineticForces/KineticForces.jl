module KineticForces

"""
KineticForces - Kinetic torque and energy calculations for perturbed equilibria

Calculates neoclassical toroidal viscosity (NTV) and kinetic energy contributions
to MHD stability through torque and energy deposition calculations.

## Module Structure

### Core Library Functions
- `Torque.jl`: tpsi!() - Core torque calculation at a single flux surface
- `EnergyIntegration.jl`: ODE-based energy-space integration
- `PitchIntegration.jl`: ODE-based pitch-angle integration

### High-level Computation
- `Compute.jl`: Orchestration routines
  - compute_torque_all_methods!()
- `CalculatedKineticMatrices.jl`: compute_calculated_kinetic_matrices()
  ← Callback registered with ForceFreeStates.make_kinetic_matrix when
  kinetic_source="calculated"

### Supporting Functions
- `KineticForcesStructs.jl`: Data structures (KineticForcesControl, KineticForcesInternal)
- `Output.jl`: HDF5 output
"""

using LinearAlgebra
using LinearAlgebra.LAPACK
using FFTW
using OrdinaryDiffEq
using HDF5
using Printf
using Statistics

using FastInterpolations
using Roots
import ..ForceFreeStates
import ..Equilibrium

# Supporting data structures and utilities
include("KineticForcesStructs.jl")
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
# ============================================================================
const mp = 1.672_614e-27      # proton mass (kg)
const me = 9.109_1e-31        # electron mass (kg)
const e  = 1.602_191_7e-19    # elementary charge (C)
const eV = e                  # joules per electron-volt

const twopi = 2π
const μ₀    = 4e-7 * π
const rad2deg = 180 / π
const deg2rad = π / 180
const iunit = 1im               # equivalent to Fortran's (0,1)

end  # module KineticForces
