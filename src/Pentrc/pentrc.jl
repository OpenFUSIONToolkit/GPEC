module PENTRC 

"""
PENTRC - Perturbed Equilibrium Nonambipolar Transport Code

A Julia implementation of the PENTRC code for calculating kinetic effects
on equilibrium stability through torque and energy deposition calculations.

Can operate in two modes:
1. Standalone: Calculate kinetic torque/energy for a given equilibrium
2. Library: Provide kinetic contributions to ForceFreeStates's stability analysis (kinetic_flag=true)

## Module Structure

### [Tier 1] Core Library Functions (Low-level)
These functions are designed to be called from both PENTRC and ForceFreeStates:
- `Torque.jl`: tpsi!() - Core torque calculation (low-level API)
- `Energy.jl`: Kinetic energy calculations
- `Pitch.jl`: Pitch angle dependent calculations

### [Tier 2] High-level Computation Functions
Orchestrates Tier 1 functions for specific use cases:
- `Compute.jl`: Main computational routines
  - compute_torque_all_methods!()
  - compute_matrix_calculation!()
  - compute_kinetic_contribution() ← Called by ForceFreeStates when kinetic_flag=true

### [Tier 3] Standalone Program
Only used for independent PENTRC execution:
- `Main.jl`: Entry point Main(path::String)

### [Tier 4] Supporting Functions
- `PentrcStructs.jl`: Data structures (PentrcControl, PentrcInternal, PentrcOutput)
- `Input.jl`: Configuration reading and parsing
- `Output.jl`: File I/O and formatting
- `Grid.jl`: Grid management and manipulation
- `Utils.jl`: Common utilities

## Public API

### For PENTRC standalone:
```julia
PENTRC.Main("/path/to/config")
```

### For ForceFreeStates kinetic_flag=true:
```julia
kinetic_results = PENTRC.compute_kinetic_contribution(ctrl, equil, ffit)
```

### For direct torque calculation:
```julia
tpsi!(tpsi_var, psi, n, l, zi, mi, wdfac, divxfac, electron, method)
```
"""

using LinearAlgebra
using LinearAlgebra.LAPACK
using TOML
using FFTW
using OrdinaryDiffEq
using HDF5
using Printf
using Statistics

using FastInterpolations
using Roots
import ..ForceFreeStates
import ..Equilibrium

# ============================================================================
# [TIER 4] Supporting data structures and utilities
# ============================================================================
include("PentrcStructs.jl")
include("Input.jl")
include("Output.jl")
include("Grid.jl")
include("Utils.jl")

# ============================================================================
# [TIER 1] Core library functions (low-level, can be called from ForceFreeStates)
# ============================================================================
include("torque.jl")
include("energy.jl")
include("energy_integration.jl")
include("special.jl")
include("Pitch.jl")

# ============================================================================
# [TIER 2] High-level computation functions
# ============================================================================
include("Compute.jl")

# ============================================================================
# [TIER 3] Standalone program entry point
# ============================================================================
include("Main.jl")

# ============================================================================
# Global constants
# ============================================================================
const mp = 1.672_614e-27      # proton mass (kg)
const me = 9.109_1e-31        # electron mass (kg)
const e  = 1.602_191_7e-19    # elementary charge (C)
const eV = e                  # joules per electron-volt

const π     = 3.141_592_653_589_793
const twopi = 2π
const μ₀    = 4e-7 * π
const rad2deg = 180 / π
const deg2rad = π / 180
const iunit = 1im               # equivalent to Fortran's (0,1)

end  # module PENTRC
