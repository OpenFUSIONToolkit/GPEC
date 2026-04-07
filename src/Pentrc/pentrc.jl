module PENTRC 

"""
PENTRC - Perturbed Equilibrium Nonambipolar Transport Code

A Julia implementation of the PENTRC code for calculating kinetic effects
on equilibrium stability through torque and energy deposition calculations.

Can operate in two modes:
1. Standalone: Calculate kinetic torque/energy for a given equilibrium
2. Library: Provide kinetic contributions to DCON's stability analysis (kinetic_flag=true)

## Module Structure

### [Tier 1] Core Library Functions (Low-level)
These functions are designed to be called from both PENTRC and DCON:
- `Torque.jl`: tpsi!() - Core torque calculation (low-level API)
- `Energy.jl`: Kinetic energy calculations
- `Pitch.jl`: Pitch angle dependent calculations

### [Tier 2] High-level Computation Functions
Orchestrates Tier 1 functions for specific use cases:
- `Compute.jl`: Main computational routines
  - compute_torque_all_methods!()
  - compute_matrix_calculation!()
  - compute_kinetic_contribution() ← Called by DCON when kinetic_flag=true

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

### For DCON kinetic_flag=true:
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
using DelimitedFiles

import ..Spl
import ..DCON
import ..Equilibrium

const version = "3.00"

const mp = 1.672_614e-27
const me = 9.109_1e-31
const e  = 1.602_191_7e-19
const eV = e

const π = 3.141_592_653_589_793
const twopi = 2π
const μ₀ = 4e-7 * π
const mu0 = μ₀
const rad2deg = 180 / π
const deg2rad = π / 180
const iunit = 1im
const xj = 1im

const npsi_out = 30
const nlambda_out = 5
const nell_out = 3
const grids = ["lsode", "equil", "input"]
const methods = [
    "fgar", "tgar", "pgar", "rlar", "clar", "fcgl",
    "fwmm", "twmm", "pwmm", "ftmm", "ttmm", "ptmm",
    "fkmm", "tkmm", "pkmm", "frmm", "trmm", "prmm",
]
const docs = [
    "Full general-aspect-ratio calculation",
    "Trapped particle general-aspect-ratio calculation",
    "Passing particle general-aspect-ratio calculation",
    "Trapped particle large-aspect-ratio calculation",
    "Trapped particle cylindrical large-aspect-ratio calculation",
    "Fluid Chew-Goldberger-Low calculation",
    "Full energy calculation using MXM Euler-Lagrange matrix",
    "Trapped energy calculation using MXM Euler-Lagrange matrix",
    "Passing energy calculation using MXM Euler-Lagrange matrix",
    "Full torque calculation using MXM Euler-Lagrange matrix",
    "Trapped torque calculation using MXM Euler-Lagrange matrix",
    "Passing torque calculation using MXM Euler-Lagrange matrix",
    "Full MXM Euler-Lagrange energy matrix norm calculation",
    "Trapped MXM Euler-Lagrange energy matrix norm calculation",
    "Passing MXM Euler-Lagrange energy matrix norm calculation",
    "Full MXM Euler-Lagrange torque matrix norm calculation",
    "Trapped MXM Euler-Lagrange torque matrix norm calculation",
    "Passing MXM Euler-Lagrange torque matrix norm calculation",
]
const nmethods = length(methods)

global atol_psi = 1e-3
global rtol_psi = 1e-6
global tdebug = false
global verbose = false
global qt = false
global nlmda = 128
global ntheta = 128
global trace_enabled = false
global trace_method = "none"
global trace_psi = -1.0
global trace_tol = 1e-8
global trace_ell = typemin(Int)

global sq = nothing
global geom = nothing
global eqfun = nothing
global rzphi = nothing
global smats = nothing
global tmats = nothing
global xmats = nothing
global ymats = nothing
global zmats = nothing
global kin = nothing
global xs_m = Any[nothing, nothing, nothing]
global dbob_m = nothing
global divx_m = nothing
global fnml = nothing

global chi1 = 1.0
global psio = 0.0
global ro = 0.0
global zo = 0.0
global bo = 0.0
global nn = 0
global jac_type = "hamada"
global power_bp = 0
global power_b = 0
global power_r = 0
global shotnum = 0.0
global shottime = 0.0
global machine = "unknown"

global mfac = Int[]
global mpert = 0
global mthsurf = 0
global psifac = Float64[]
global theta = Float64[]

function default_mode_numbers(nm::Int)
    start = -fld(nm, 2)
    return collect(start:start + nm - 1)
end

function set_mode_numbers!(mode_numbers::AbstractVector{<:Integer})
    global mfac = Int.(mode_numbers)
    global mpert = length(mfac)
    return mfac
end

function active_mode_numbers(nm::Int; mode_numbers=nothing)
    if mode_numbers !== nothing
        return set_mode_numbers!(mode_numbers)
    end
    if !isempty(mfac) && length(mfac) == nm
        return mfac
    end
    return set_mode_numbers!(default_mode_numbers(nm))
end

function set_trace!(; enabled::Bool=false, method::AbstractString="none",
                    psi::Real=-1.0, tol::Real=1e-8, ell::Integer=typemin(Int))
    global trace_enabled = enabled
    global trace_method = lowercase(String(method))
    global trace_psi = Float64(psi)
    global trace_tol = Float64(tol)
    global trace_ell = Int(ell)
    return nothing
end

function clear_trace!()
    return set_trace!()
end

diagnose_all() = nothing

# ============================================================================
# [TIER 4] Supporting data structures and utilities
# ============================================================================
include("PentrcStructs.jl")
include("Utils.jl")
include("Input.jl")
include("Output.jl")
include("Grid.jl")
include("special.jl")

# ============================================================================
# [TIER 1] Core library functions (low-level, can be called from DCON)
# ============================================================================
include("torque.jl")
include("energy.jl")
include("Pitch.jl")

# ============================================================================
# [TIER 2] High-level computation functions
# ============================================================================
include("Compute.jl")

# ============================================================================
# [TIER 3] Standalone program entry point
# ============================================================================
include("Main.jl")

end  # module PENTRC
