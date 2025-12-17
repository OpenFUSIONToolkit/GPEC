"""
    struct VacuumInput

Holds plasma boundary and mode data as provided from DCON namelist and computed quantities.

- `r`: Plasma boundary R-coordinate as a function of poloidal angle.
- `z`: Plasma boundary Z-coordinate as a function of poloidal angle.
- `delta`: -dphi/qa; 0 for coordinate systems using machine angle (e.g., PEST basis).
- `mlow`: Lower poloidal mode number for spectral representation.
- `mhigh`: Upper poloidal mode number for spectral representation.
- `n`: The toroidal mode number. Paper: n.
- `qa`: Safety factor at the plasma boundary.
- `mtheta_in`: Number of poloidal angles in the input boundary arrays.
- `kernelsign`: Sign for kernel; +1 or -1, only ≠ 1 for mutual inductance calculations.
- `force_wv_symmetry`: Boolean flag to enforce symmetry in the vacuum response matrix. Set in dcon.toml
"""
@kwdef mutable struct VacuumInput
    r::Vector{Float64} = Float64[]
    z::Vector{Float64} = Float64[]
    delta::Vector{Float64} = Float64[]
    mlow::Int = 0
    mhigh::Int = 0
    mpert::Int = 0
    n::Int = 0
    qa::Float64 = 0.0
    mtheta_eq::Int = 1
    mtheta::Int = 1
    kernelsign::Float64 = 1.0
    force_wv_symmetry::Bool = true
end

"""
    struct PlasmaGeometry

Holds plasma geometry data on the mth grid for vacuum calculations. Arrays are of length `mth`,
where `mth` is the number of poloidal grid points and θ ∈ [0, 1).

# Fields

- `x`: Plasma surface R.
- `z`: Plasma surface Z.
- `delta`: Toroidal angle offset divided by qa (i.e. -ν/qa where ϕ = 2πζ + ν(ψ, θ)) at plasma surface.
- `dx_dtheta`: dR/dθ at plasma surface.
- `dz_dtheta`: dZ/dθ at plasma surface.
- `cnqd`: cos(n * qa * delta) at plasma surface.
- `snqd`: sin(n * qa * delta) at plasma surface.
- `sinlt`: sin(l * θ) basis functions for poloidal modes at plasma surface.
- `coslt`: cos(l * θ) basis functions for poloidal modes at plasma surface.
- `snlth`: sin(l * θ + n * qa * delta) basis functions for poloidal modes at plasma surface.
- `cslth`: cos(l * θ + n * qa * delta) basis functions for poloidal modes at plasma surface.
"""
@kwdef struct PlasmaGeometry
    x::Vector{Float64}
    z::Vector{Float64}
    delta::Vector{Float64}
    dx_dtheta::Vector{Float64}
    dz_dtheta::Vector{Float64}
    cnqd::Vector{Float64}
    snqd::Vector{Float64}
    sinlt::Matrix{Float64}
    coslt::Matrix{Float64}
    snlth::Matrix{Float64}
    cslth::Matrix{Float64}
end

"""
    struct WallGeometry

Holds wall geometry data for vacuum calculations. Arrays are of length `mth`,
where `mth` is the number of poloidal grid points and θ ∈ [0, 1).

# Fields
- `nowall`: Boolean flag indicating if there is no wall.
- `is_closed_toroidal`: Boolean flag indicating if the wall is a closed toroidal surface.
- `x`: Wall R coordinates
- `z`: Wall Z coordinates
- `dx_dtheta`: dR/dθ at wall.
- `dz_dtheta`: dZ/dθ at wall.
"""
@kwdef struct WallGeometry
    nowall::Bool = true
    is_closed_toroidal::Bool = true
    x::Vector{Float64} = Float64[]
    z::Vector{Float64} = Float64[]
    dx_dtheta::Vector{Float64} = Float64[]
    dz_dtheta::Vector{Float64} = Float64[]
end

"""
    struct WallShapeSettings

Input settings for vacuum wall and geometry.

- `shape` : String selecting wall shape. Options are:
    - "nowall": No wall.
    - "dee": Dee-shaped wall.
    - "moddee": Modified Dee-shaped wall.
    - "custom": Custom wall shape provided in wall_geo.dat. TODO: Describe file format here.
- `a` : The distance of the shell from the plasma in units of major radius (conformal), or minor radius parameter (others).
- `aw` : Half-thickness of the shell.
- `bw` : Elongation of the shell.
- `cw` : Offset of the center of the shell from the major radius.
- `dw` : Triangularity of shell.
- `tw` : Sharpness of the corners of the shell. Try 0.05 as a good initial value.
- `equal_arc_wall`: 1 turns on equal arcs distribution of the nodes on the shell. Best results unless
  the wall is very close to the plasma.
"""
@kwdef mutable struct WallShapeSettings 

    # Core shape selection
    shape::String = "nowall"
    
    # Standard geometric parameters for Dee/Mod-Dee
    a::Float64 = 0.3
    aw::Float64 = 0.05
    bw::Float64 = 1.5
    cw::Float64 = 0.0
    dw::Float64 = 0.5
    tw::Float64 = 0.05
    
    # Algorithmic options
    equal_arc_wall::bool = true
end