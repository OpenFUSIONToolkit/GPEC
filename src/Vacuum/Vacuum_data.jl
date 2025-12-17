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
    cn0::Float64 = 1.0
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

- `shape` key word replacement for fortran `ishape`: Integer. Options for the wall shape.
    * `< 0`: Spherical topology.
    * `< 10`: Closed toroidal topology.
    * 2: Elliptical shell confocal to the plasma's radius and height. The radius of the shell is `a`.
    * 4: Modified dee-shaped wall independent of plasma geometry with triangularity `dw`, squareness(?), and 2nd harmonic of `zwal` in `aw` and `tw`. Centered at `cw`, radius `a`, and elongation `bw`.
    * 5: Dee-shaped wall scaled to the radius and geometric center of the plasma. Offset of `cw`. Other variables as option 4.
    * 6: Conforming shell at distance `a * p_rad`. Best for a close fitting shell.
    * 7: Enclosing bean-shaped wall.
    * 8: Wall of DIII-D.
    * `< 20`: Solid conductors not linking plasma.
    * 11: Dee-shaped conductor.
    * 12: Solid bean-shaped conductor on right.
    * 13: Solid bean-shaped conductor on left.
    * `< 30`: Toroidal conductor with a toroidally symmetric gap, geometry correlated to plasma.
    * 21: Shell scaled to plasma. Gap on inner side.
    * 24: Shell scaled to plasma. Gap on outer side.
    * `< 40`: Toroidal conductor with a toroidally symmetric gap, geometry independent of plasma.
    * 31: Shell independent of plasma. Gap on inner side.
    * 34: Shell independent of plasma. Gap on outer side.
- `aw` (a_w): Half-thickness of the shell.
- `bw` (b_w): Elongation of the shell.
- `cw` (c_w): Offset of the center of the shell from the major radius, X_{maj}.
- `dw` (delta_w): Triangularity of shell.
- `tw` (tau_w): Sharpness of the corners of the shell. Try 0.05 as a good initial value.
- `nsing`: Not referenced.
- `epsq`: Not referenced.

- `leqarcw`: 1 turns on equal arcs distribution of the nodes on the shell. Best results unless
  the wall is very close to the plasma. See `ishape=6` option.
- `a` (a): Usually the distance of the shell from the plasma in units of the plasma radius p_{rad} at the outer side. If a geq 10, the wall is assumed to be at infty.
"""
@kwdef mutable struct WallShapeSettings 

    # Core shape selection
    shape::String = "nowall"
    
    # Standard geometric parameters for Dee/Mod-Dee
    aw::Float64 = 0.05
    bw::Float64 = 1.5
    cw::Float64 = 0.0
    dw::Float64 = 0.5
    tw::Float64 = 0.05
    
    # Scale and center
    a::Float64 = 1.2    # Distance or scaling factor
    xma::Float64 = 1.0
    zma::Float64 = 0.0
    
    # Algorithmic options
    leqarcw::Int = 1    # Re-parameterization flag
    
    # Numerical controls (Keep only if used in Green's function integration)
    nsing::Int = 500
    epsq::Float64 = 1e-05
end