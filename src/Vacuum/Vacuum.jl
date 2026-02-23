module Vacuum

using TOML, SpecialFunctions, LinearAlgebra, Printf
using FastInterpolations: cubic_interp, deriv1, PeriodicBC, NaturalBC
using FastGaussQuadrature: gausslegendre
using StaticArrays: SVector
using SparseArrays

# Import parent modules
import ..Equilibrium
using ..Utilities.FourierTransforms: compute_fourier_coefficients, fourier_transform!, fourier_inverse_transform!

include("DataTypes.jl")
include("Kernel2D.jl")
include("Kernel3D.jl")
include("MathUtils.jl")
include("VacuumFromEquilibrium.jl")

export VacuumInput, WallShapeSettings
export compute_vacuum_response, compute_vacuum_field
export extract_plasma_surface_at_psi

"""
    compute_vacuum_response(inputs::Union{VacuumInput, VacuumInput3D}, wall_settings::WallShapeSettings;
        green_only=false, PATCH_RAD=11, RAD_DIM=20, INTERP_ORDER=5)

Compute the vacuum response matrix and both Green's functions using provided vacuum inputs.

Single entry point for 2D (axisymmetric, single toroidal mode) and 3D (multiple toroidal modes)
vacuum calculations. Dispatches on input type: use `VacuumInput` for 2D and `VacuumInput3D` for 3D.

This is the pure Julia implementation that replaces the Fortran `mscvac` function.
It computes both interior (grri) and exterior (grre) Green's functions for GPEC response calculations.

# Arguments

  - `inputs`: `VacuumInput` (2D) or `VacuumInput3D` (3D) with mode numbers, grid resolution, and boundary info.
  - `wall_settings::WallShapeSettings`: Wall geometry configuration.
  - `green_only`: If true, skip building the response matrix `wv` and return zeros for `wv` and `xzpts`.
  - `PATCH_RAD`, `RAD_DIM`, `INTERP_ORDER`: 3D kernel quadrature parameters (ignored for 2D).

# Returns

  - `wv`: Complex vacuum response matrix (mpert×mpert for 2D, num_modes×num_modes for 3D).
  - `grri`: Interior Green's function matrix.
  - `grre`: Exterior Green's function matrix.
  - `xzpts`: Coordinate array (mtheta×4 for 2D, mtheta*nzeta×4 for 3D) [R_plasma, Z_plasma, R_wall, Z_wall].

# Notes

  - For n=0 modes with closed walls (2D), a regularization factor is added to prevent singularities.
"""
function compute_vacuum_response(
    inputs::Union{VacuumInput,VacuumInput3D},
    wall_settings::WallShapeSettings;
    green_only::Bool=false,
    PATCH_RAD::Int=11,
    RAD_DIM::Int=20,
    INTERP_ORDER::Int=5
)
    # Initialize surface geometries
    plasma_surf = inputs isa VacuumInput3D ? PlasmaGeometry3D(inputs) : PlasmaGeometry(inputs)
    wall = inputs isa VacuumInput3D ? WallGeometry3D(inputs, plasma_surf, wall_settings) : WallGeometry(inputs, plasma_surf, wall_settings)

    # Compute Fourier basis coefficients and create kernel parameters
    if inputs isa VacuumInput3D
        cos_mn_basis, sin_mn_basis = compute_fourier_coefficients(
            inputs.mtheta, inputs.mpert, inputs.mlow, inputs.nzeta, inputs.npert, inputs.nlow; ν=plasma_surf.ν)
        kparams = KernelParams3D(PATCH_RAD, RAD_DIM, INTERP_ORDER)
    else
        cos_mn_basis, sin_mn_basis = compute_fourier_coefficients(
            inputs.mtheta, inputs.mpert, inputs.mlow; n=inputs.n, ν=plasma_surf.ν)
        kparams = KernelParams2D(inputs.n)
    end
    num_points_plasma, num_modes = size(cos_mn_basis)

    # If no wall, only plasma points; if wall, plasma + wall points
    num_points = wall.nowall ? num_points_plasma : 2 * num_points_plasma

    # Allocate matrices, accounting for whether wall is present or not
    grri = zeros(num_points, 2 * num_modes) # Interior (kernelsign=-1)
    grre = zeros(num_points, 2 * num_modes) # Exterior (kernelsign=+1)
    grad_green = zeros(num_points, num_points)
    green_temp = zeros(num_points_plasma, num_points_plasma)

    # Plasma–Plasma block
    kernel!(grad_green, green_temp, plasma_surf, plasma_surf, kparams)

    # Fourier transform obs=plasma, src=plasma block
    fourier_transform!(grre, green_temp, cos_mn_basis)
    fourier_transform!(grre, green_temp, sin_mn_basis; col_offset=num_modes)

    if !wall.nowall
        # Plasma–Wall block
        kernel!(grad_green, green_temp, plasma_surf, wall, kparams)
        # Wall–Wall block
        kernel!(grad_green, green_temp, wall, wall, kparams)
        # Wall–Plasma block
        kernel!(grad_green, green_temp, wall, plasma_surf, kparams)
        # Fourier transform obs=wall, src=plasma block
        fourier_transform!(grre, green_temp, cos_mn_basis; row_offset=num_points_plasma)
        fourier_transform!(grre, green_temp, sin_mn_basis; row_offset=num_points_plasma, col_offset=num_modes)
    end

    # Compute both Green's functions: exterior (kernelsign=+1) then interior (kernelsign=-1)
    grri .= grre # start from same as exterior

    # Solve exterior first, then overwrite grad_green with interior kernel to avoid extra allocations.
    grre .= grad_green \ grre

    # Interior flips the sign of the normal, but not the diagonal terms, so we multiply by -1 and add 2I to the diagonal
    grad_green .*= -1
    for i in 1:num_points
        grad_green[i, i] += 2.0
    end
    grri .= grad_green \ grri

    wv = zeros(ComplexF64, num_modes, num_modes)
    plasma_pts = zeros(num_points_plasma, 3)
    wall_pts = zeros(num_points_plasma, 3)
    if !green_only
        # Perform inverse Fourier transforms to get response matrix components [Chance Phys. Plasmas 2007 052506 eq. 115-118]
        arr, aii, ari, air = ntuple(_ -> zeros(num_modes, num_modes), 4)
        fourier_inverse_transform!(arr, grre, cos_mn_basis)
        fourier_inverse_transform!(aii, grre, sin_mn_basis; col_offset=num_modes)
        fourier_inverse_transform!(ari, grre, sin_mn_basis)
        fourier_inverse_transform!(air, grre, cos_mn_basis; col_offset=num_modes)

        # Final form of vacuum response matrix [Chance Phys. Plasmas 2007 052506 eq. 114]
        wv .= complex.(arr .+ aii, air .- ari)
        inputs.force_wv_symmetry && hermitianpart!(wv)

        # Fill xzpts array - this is temporary until we have a better way to handle the different geometries
        plasma_pts = plasma_surf isa PlasmaGeometry3D ? plasma_surf.r : [plasma_surf.x zeros(num_points_plasma, 1) plasma_surf.z]
        wall_pts = wall isa WallGeometry3D ? wall.r : [wall.x zeros(num_points_plasma, 1) wall.z]
    end

    # Append zeros for the wall points onto grri and grre arrays for expected output shape
    if wall.nowall
        grri = vcat(grri, zeros(num_points_plasma, 2 * num_modes))
        grre = vcat(grre, zeros(num_points_plasma, 2 * num_modes))
    end

    return wv, grri, grre, plasma_pts, wall_pts
end

"""
    compute_vacuum_field(inputs::VacuumInput, plasma_surf::PlasmaGeometry, wall::WallGeometry,
           Bn::Vector{<:Number}, R_grid::AbstractVector, Z_grid::AbstractVector)

Calculate the perturbed magnetic field in the vacuum region resulting from a normal
magnetic field perturbation (`Bn`) at the plasma surface. Replaces `mscfld` from Fortran.

This function orchestrates the vacuum field calculation by:

 1. Calling `vaccal!` to compute the vacuum response kernel (`grri`)
 2. Defining a grid of points (`R_grid`, `Z_grid`) where the field is to be calculated
 3. Calling `_pickup_field` to compute the magnetic field components on that grid using the kernel
    and the source perturbation `Bn`

# Arguments

  - `inputs::VacuumInput`: Struct containing vacuum calculation parameters (n, mpert, mtheta, etc.)
  - `plasma_surf::PlasmaGeometry`: Struct with plasma surface geometry and basis functions
  - `wall::WallGeometry`: Struct with wall geometry
  - `Bn::Vector{<:Number}`: Complex vector of Fourier harmonics of the normal magnetic field
    perturbation at the plasma surface, `B_n = B_n_real + i*B_n_imag`. Length must be `mpert`.
  - `R_grid::AbstractVector`: Vector of R coordinates for the output field grid
  - `Z_grid::AbstractVector`: Vector of Z coordinates for the output field grid

# Returns

  - `B_R::Matrix{ComplexF64}`: The R-component of the magnetic field on the grid
  - `B_Z::Matrix{ComplexF64}`: The Z-component of the magnetic field on the grid
  - `B_phi::Matrix{ComplexF64}`: The toroidal component of the magnetic field on the grid
  - `grid_info::Matrix{Int}`: Information about the grid points (1=inside plasma, 0=outside)
"""
function compute_vacuum_field(inputs::VacuumInput, plasma_surf::PlasmaGeometry, wall::WallGeometry,
    Bn::Vector{<:Number}, R_grid::AbstractVector, Z_grid::AbstractVector)

    # 1. Call vaccal! to get the inverted Green's function matrix
    # The Fortran version calls the whole chain (ent33 -> vaccal),
    # here we assume vaccal! provides what we need.
    wv, grri = vaccal!(inputs, plasma_surf, wall)

    # Separate real and imaginary parts of the source perturbation
    Bn_real = real.(Bn)
    Bn_imag = imag.(Bn)

    # 2. Define grid and parameters for pickup routine
    nx = length(R_grid)
    nz = length(Z_grid)

    # 3. Call the field pickup routine
    B_R, B_Z, B_phi, grid_info = _pickup_field(
        inputs, plasma_surf, grri, Bn_real, Bn_imag, R_grid, Z_grid
    )

    return B_R, B_Z, B_phi, grid_info
end
end
