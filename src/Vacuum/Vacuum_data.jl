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
"""
struct VacuumInputType
    r::Vector{Float64}
    z::Vector{Float64}
    delta::Vector{Float64}
    mlow::Int
    mhigh::Int
    n::Int
    qa::Float64
    mtheta_in::Int
end

#################################
# Vacuum Globals Struct         #
#################################

"""
    struct VacuumGlobalsType

Holds all derived, static, and globally shared parameters for a vacuum calculation.
This struct contains only values that are computed once at initialization and are
immutable/thread-safe for the duration of a run. Anything set directly by user input
(namelist/TOML) lives in `VacuumSettingsType`.

# Fields

## Grid Parameters
- `mth`: Number of poloidal grid points (theta grid), derived from input/settings.
- `mth1`: `mth + 1` (theta grid plus periodic point).
- `mth2`: `mth + 2` (theta grid plus periodic and one extra point).
- `nfm`: Number of Fourier modes (often equals `mpert` from settings).
- `mtot`: Total number of modes (may be equal to `nfm`).

## Mode Indices (Lagrange/Fourier Basis)
- `lmin`: Lower indices for basis modes (length `jmax1`).
- `lmax`: Upper indices for basis modes (length `jmax1`).

## Plasma Geometry (Derived)
- `xinf`: Plasma surface R (theta grid, length `mth1`).
- `zinf`: Plasma surface Z (theta grid, length `mth1`).
- `delta`: Surface offset or Shafranov shift (length `mth1`).
- `xplap`: dR/dtheta at plasma surface (computed from `xinf`).
- `zplap`: dZ/dtheta at plasma surface (computed from `zinf`).

## Physical Derived Parameters
- `qa1`: Safety factor at plasma boundary (copied from DCON or computed).
- `ga1`: Geometric factor, e.g., R*Bphi or similar (from geometry).
- `fa1`: Poloidal flux normalization (from geometry).

## Wall Geometry (if present, derived from input/settings)
- `xwal`: Wall R coordinates (length `mth1` or `mth2`).
- `zwal`: Wall Z coordinates (length `mth1` or `mth2`).
- `xwalp`: dR/dtheta at wall (computed).
- `zwalp`: dZ/dtheta at wall (computed).

## Miscellaneous Derived/Initialization Values
- `dth`: Poloidal angle step (`2π/mth`).
- `wall`: Wall present/enabled (from settings, but may be recomputed).
- `farwal`: "Far wall" logic, set by geometry checks.
- `kernelsign`: Sign for kernel; +1 or -1 (set by main routine, default +1).
"""
@kwdef mutable struct VacuumGlobalsType
    n::Int      = 1
    mth::Int    = 512
    mth1::Int   = 513
    mth2::Int   = 514
    nfm::Int    = 0
    mtot::Int   = 0

    lmin::Vector{Int} = Int[]
    lmax::Vector{Int} = Int[]

    xinf::Vector{Float64} = Float64[]
    zinf::Vector{Float64} = Float64[]
    delta::Vector{Float64} = Float64[]
    xplap::Vector{Float64} = Float64[]
    zplap::Vector{Float64} = Float64[]

    qa1::Float64 = 0.0
    ga1::Float64 = 0.0
    fa1::Float64 = 0.0

    xwal::Vector{Float64} = Float64[]
    zwal::Vector{Float64} = Float64[]
    xwalp::Vector{Float64} = Float64[]
    zwalp::Vector{Float64} = Float64[]

    dth::Float64    = 0.0
    wall::Bool      = true
    farwal::Bool    = true
    kernelsign::Bool = true
end

#############################################
# Vacuum Settings (Input Namelist) Structs  #
#############################################

"""
    struct Modes

Numerical and input control parameters for grid and harmonics.
(Paper Table: "modes")

- `mth` : Even integer. Number of grid points used for the calculation. The values of 
  the needed quantities on these points are interpolated from those gotten from 
  the plasma information in `mp0`, `mp1`, `vacin`, `invacuum`, or equivalent.
- `xiin`: Array. Input Fourier modes of xi_l(edge). See `ieig` in Sec. diags.
- `lsymz`: .true. Symmetrizes the vacuum matrix.
- `leqarcw`: 1 turns on equal arcs distribution of the nodes on the shell. Best results unless
  the wall is very close to the plasma. See `ishape=6` option.
"""
@kwdef mutable struct Modes
    mth::Int = 480
    xiin::NTuple{9, Int} = (0, 0, 0, 0, 0, 0, 0, 1, 0)
    lsymz::Bool = true
    leqarcw::Int = 1
end

"""
    struct Vacdat

Parameters for vacuum wall and geometry. (Paper Table: "vacdat")

- `ishape`: Integer. Options for the wall shape.
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
"""
@kwdef mutable struct Vacdat
    ishape::Int = 6
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
end

"""
    struct Shape

Plasma and wall geometric parameters. (Paper Table: "shape")

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
@kwdef mutable struct Shape
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

"""
    struct Diagns

Diagnostics and output control parameters. (Paper Table: "diags")

- `lkdis`: Logical. Turns on the eddy current calculations. Calls `subroutine kdis`.
- `ieig`: Integer. Options for getting the surface eigenfunctions xi(l).
    * 1: From pest-1. Writes the omega^2 and xi(l).
    * 4: Reads from file `outdist`.
    * 5: Gets xi(l) from the input `xiin` in namelist `modes`.
    * 8: Re[xi(k)] and Im[xi(k)] from input file `vacin` for `gato`'s input.
- `iloop`: Integer. Turns on Mirnov coils calculation.
    * 1: Coil locations given by dee-shaped geometry set by parameters below.
    * 2: PBX's Mirnov coil positions.
- `lpsub`: Integer. Uses subroutines for coil positions. Otherwise uses namelist inputs `(xloop, zloop)`.
- `nloop`: Number of coils around the plasma.
- `nloopr`: Number of radial loops.
- `nphil`: Number of phi positions for the loop calculations.
- `nphse`: Number of phi positions for the eddy current plots.
- `xofsl`: Offset of the loop positions.
- `ntloop`: Number of loop positions distributed along the shell.
- `aloop`: Distance of the loop "dee" from plasma.
- `bloop`: Elongation of the loop "dee".
- `dloop`: Triangularity of the loop "dee".
- `rloop`: Not referenced.
- `deloop`: Delta fraction of `plrad` to calculate magnetic field from the derivative of chi.
- `mx`, `mz`: Contour grid for chi.
- `nph`: Number of such contours in phi.
- `nxlpin`, `nzlpin`: Number of X and Z grid points for field evaluation.
- `epslp`: Grid tolerance for field evaluation.
- `xlpmin`, `xlpmax`: X grid bounds for field evaluation.
- `zlpmin`, `zlpmax`: Z grid bounds for field evaluation.
- `linterior`: Field logic (0 = exterior, 1 = interior).
"""
@kwdef mutable struct Diagns
    lkdis::Bool = false
    ieig::Int = 0
    iloop::Int = 0
    lpsub::Int = 1
    nloop::Int = 128
    nloopr::Int = 0
    nphil::Int = 3
    nphse::Int = 1
    xofsl::Float64 = 0.0
    ntloop::Int = 32
    aloop::Float64 = 0.01
    bloop::Float64 = 1.6
    dloop::Float64 = 0.5
    rloop::Float64 = 1.0
    deloop::Float64 = 0.001
    mx::Int = 21
    mz::Int = 21
    nph::Int = 0
    nxlpin::Int = 6
    nzlpin::Int = 11
    epslp::Float64 = 0.02
    xlpmin::Float64 = 0.7
    xlpmax::Float64 = 2.7
    zlpmin::Float64 = -1.5
    zlpmax::Float64 = 1.5
    linterior::Int = 2
end

"""
    struct Sprk

Spark and Feedback type variables. (Paper Table: "sprk")

Note: Under Development.
"""
@kwdef mutable struct Sprk
    nminus::Int = 0
    nplus::Int = 0
    mphi::Int = 16
    lwrt11::Int = 0
    civ::Float64 = 0.0
    sp2sgn1::Int = 1
    sp2sgn2::Int = 1
    sp2sgn3::Int = 1
    sp2sgn4::Int = 1
    sp2sgn5::Int = 1
    sp3sgn1::Int = -1
    sp3sgn2::Int = -1
    sp3sgn3::Int = -1
    sp3sgn4::Int = 1
    sp3sgn5::Int = 1
    lff::Int = 0
    ff::Float64 = 1.6
    fv::Vector{Float64} = [1.6 for i in 1:34]
end

"""
    struct VacuumSettingsType

Holds all user-configurable namelist (TOML) options for the vacuum solver, grouped by
Fortran input namelist group. This struct is the canonical place for all user-input fields.

- `modes`: Numerical discretization and symmetry options.
- `vacdat`: Wall and physics parameters.
- `shape`: Plasma and wall geometry parameters.
- `diagns`: Diagnostics and output control.
- `sprk`: Miscellaneous spark/advanced features.
- `old_version`: Using old version of vacuum if it's true.
"""
@kwdef mutable struct VacuumSettingsType
    modes::Modes = Modes()
    vacdat::Vacdat = Vacdat()
    shape::Shape = Shape()
    diagns::Diagns = Diagns()
    sprk::Sprk = Sprk()
    old_version::Bool = false
end