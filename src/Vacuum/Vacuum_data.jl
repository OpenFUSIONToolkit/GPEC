# Vacuum_data.jl
# This file defines the data structures for the Vacuum module in Julia.
#
# Field comments below are excerpted and paraphrased from the official documentation
# and paper tables. For each field, the "paper equivalent" and/or precise meaning is noted.

##############################
# Vacuum DCON Input Struct   #
##############################

"""
    struct VacuumInputType

Holds plasma boundary and mode data as provided from DCON or equivalent upstream code.

- `r`: Plasma boundary R-coordinate as a function of poloidal angle.
- `z`: Plasma boundary Z-coordinate as a function of poloidal angle.
- `delta`: -dphi/qa; 0 for coordinate systems using machine angle (e.g., PEST basis).
- `mlow`: Lower poloidal mode number for spectral representation.
- `mhigh`: Upper poloidal mode number for spectral representation.
- `n`: The toroidal mode number. Paper: n.
- `qa`: Safety factor at the plasma boundary.
- `mtheta_in`: Number of poloidal angles in the input boundary arrays.
- `farwall_flag`: Boolean flag indicating if the conducting wall is at infinity.
- `kernelsign`: Sign for kernel; +1 or -1, only ≠ 1 for mutual inductance calculations.
"""
@kwdef mutable struct VacuumInputType
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
    farwall_flag::Bool = false
    kernelsign::Float64 = 1.0
    force_wv_symmetry::Bool = true
end

"""
    struct PlasmaGeometry

FILL THIS IN LATER

# Fields

- `xinf`: Plasma surface R (theta grid, length `mth1`).
- `zinf`: Plasma surface Z (theta grid, length `mth1`).
- `delta`: Surface offset or Shafranov shift (length `mth1`).
- `xplap`: dR/dtheta at plasma surface (computed from `xinf`).
- `zplap`: dZ/dtheta at plasma surface (computed from `zinf`).
- FILL THE REST IN LATER
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

FILL THIS IN LATER

# Fields
- `xwal`: Wall R coordinates (length `mth1` or `mth2`).
- `zwal`: Wall Z coordinates (length `mth1` or `mth2`).
- `xwalp`: dR/dtheta at wall (computed).
- `zwalp`: dZ/dtheta at wall (computed).
"""
@kwdef struct WallGeometry
    is_closed_toroidal::Bool = true
    x::Vector{Float64} = Float64[]
    z::Vector{Float64} = Float64[]
    dx_dtheta::Vector{Float64} = Float64[]
    dz_dtheta::Vector{Float64} = Float64[]
end

#############################################
# Vacuum Settings (Input Namelist) Structs  #
#############################################

"""
    struct WallShape

Parameters for vacuum wall and geometry.

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
- `noutv`: Number of grid points for the eddy current plots.
- `idgt`: Not referenced now. Used to be approx. number of digits accuracy in the Gaussian elimination used in the calculation. A value of idgt=6 is usually sufficient.
- `idot`: Not referenced.
- `idsk`: Not referenced.
- `delg`: Non-integer. Size of arrows for the eddy current plots. Integer part is length of shaft and decimal part is size of the head.
- `delfac`: Controls grid size to calculate derivatives in `spark` type calculations.
- `cn0`: Constant added to the cal K matrix to make it nonsingular for n=0 modes.

- `leqarcw`: 1 turns on equal arcs distribution of the nodes on the shell. Best results unless
  the wall is very close to the plasma. See `ishape=6` option.
- `ipshp`: 0 gets the plasma boundary and safety factor, qedge, etc. from input files. 1 ignores input data files, sets qedge = qain. Shape of plasma is dee-shaped centered at `xpl`, radius `apl`, elongation `bpl`, and triangularity `dpl`. The straight-line coordinate variable delta(theta) is set to zero.
- `isph` : 0 all vacuum R values are positive, 1 is not.
- `inside` : 
- `xpl`: Plasma center R coordinate.
- `apl`: Plasma minor radius.
- `bpl`: Plasma elongation.
- `dpl`: Plasma triangularity.
- `qain`: Input value for qedge when `ipshp = 1`.
- `r`: Not referenced.
- `a` (a): Usually the distance of the shell from the plasma in units of the plasma radius p_{rad} at the outer side. If a geq 10, the wall is assumed to be at infty.
- `b` (beta): Subtending half-angle of the shell in degrees.
- `abulg` (a_b): The size of the bulge along the major radius, normalized to the mean plasma radius.
- `bbulg` (beta_b): Subtending half-angle of the extent of the bulge.
- `tbulg` (tau_b): Inverse roundedness of the bulge corners.
- `xma` : shifting major radius point.

"""
@kwdef mutable struct WallShapeSettings
    shape::String = "conformal"
    aw::Float64 = 0.05
    bw::Float64 = 1.5
    cw::Float64 = 0.0
    dw::Float64 = 0.5
    tw::Float64 = 0.05
    nsing::Int = 500
    epsq::Float64 = 1e-05
    noutv::Int = 37
    idgt::Int = 6
    idot::Int = 0
    idsk::Int = 0
    delg::Float64 = 15.01
    delfac::Float64 = 0.001
    cn0::Int = 1

    leqarcw::Int = 1
    ipshp::Int = 0
    isph::Int = 0
    inside::Int = 0
    xpl::Float64 = 100.0
    apl::Float64 = 1.0
    a::Float64 = 20.0
    b::Float64 = 170.0
    bpl::Float64 = 1.0
    dpl::Float64 = 0.0
    r::Float64 = 1.0
    abulg::Float64 = 0.932
    bbulg::Float64 = 17.0
    tbulg::Float64 = 0.02
    qain::Float64 = 2.5
    xma::Float64 = 1.0
    zma::Float64 = 0.0
end
