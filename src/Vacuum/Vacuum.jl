module Vacuum

using TOML, SpecialFunctions, LinearAlgebra, Printf
using FastInterpolations: cubic_interp, deriv1, PeriodicBC, NaturalBC
using StaticArrays
using FastGaussQuadrature
using SparseArrays

# Import parent modules
import ..Equilibrium

# Import FourierTransforms utility for coefficient calculation and transforms
using ..Utilities.FourierTransforms: compute_fourier_coefficients, fourier_transform!, fourier_inverse_transform!

# Include core data structures and functions first
include("DataTypes.jl")
include("Kernel2D.jl")
include("Kernel3D.jl")
include("MathUtils.jl")

# Include VacuumFromEquilibrium after DataTypes so VacuumInput is defined
include("VacuumFromEquilibrium.jl")

export VacuumInput, compute_vacuum_response, compute_vacuum_response_3D
export compute_vacuum_field
export kernel!
export WallShapeSettings
export extract_plasma_surface_at_psi, create_vacuum_input_at_psi

"""
    compute_vacuum_response(inputs::VacuumInput, wall_settings::WallShapeSettings)

Compute the vacuum response matrix and both Green's functions using provided vacuum inputs.

This is the pure Julia implementation that replaces the Fortran `mscvac` function.
It computes both interior (grri) and exterior (grre) Green's functions for GPEC response calculations.

# Arguments

  - `inputs::VacuumInput`: Struct containing vacuum calculation parameters including mode numbers,
    grid resolution, toroidal mode number, and plasma boundary information.
  - `wall_settings::WallShapeSettings`: Struct specifying the wall geometry configuration.

# Returns

  - `wv`: Complex vacuum response matrix (mpert × mpert) relating plasma perturbations to vacuum response
  - `grri`: Interior Green's function matrix (2*mtheta × 2*mpert) with kernelsign=-1
  - `grre`: Exterior Green's function matrix (2*mtheta × 2*mpert) with kernelsign=+1
  - `xzpts`: Coordinate array (mtheta × 4) containing [R_plasma, Z_plasma, R_wall, Z_wall]

# Notes

  - This function computes both Green's functions to enable proper surface inductance calculations
  - Uses FourierTransforms utility internally for consistent coefficient calculation
  - The vacuum response includes plasma-plasma and plasma-wall coupling effects
  - For n=0 modes with closed walls, a regularization factor is added to prevent singularities
"""
function compute_vacuum_response(inputs::VacuumInput, wall_settings::WallShapeSettings; green_only=false)

    # Initialization and allocations
    (; mtheta, mpert, mlow, n, force_wv_symmetry) = inputs
    plasma_surf = PlasmaGeometry(inputs)
    wall = WallGeometry(inputs, plasma_surf, wall_settings)

    # Compute Fourier basis coefficients
    cos_mn_basis, sin_mn_basis = compute_fourier_coefficients(mtheta, mpert, mlow; n=n, ν=plasma_surf.ν)

    # If no wall, only plasma points; if wall, plasma + wall points
    num_points = wall.nowall ? mtheta : 2 * mtheta

    # Allocate arrays for both Green's functions
    grri = zeros(num_points, 2 * mpert)  # Interior (kernelsign=-1)
    grre = zeros(num_points, 2 * mpert)  # Exterior (kernelsign=+1)
    grad_green = zeros(num_points, num_points)
    green_temp = zeros(mtheta, mtheta)

    # Fourier transforms offsets into grri/grre: first mtheta rows are plasma as observer, second are wall
    # First mpert columns are real, second mpert are imaginary
    WALL_ROW_OFFSET = mtheta
    IMAG_COL_OFFSET = mpert

    # Plasma–Plasma block
    kernel!(grad_green, green_temp, plasma_surf, plasma_surf, n)

    # Fourier transform obs=plasma, src=plasma block
    fourier_transform!(grre, green_temp, cos_mn_basis)
    fourier_transform!(grre, green_temp, sin_mn_basis; col_offset=IMAG_COL_OFFSET)

    if !wall.nowall
        # Plasma–Wall block
        kernel!(grad_green, green_temp, plasma_surf, wall, n)
        # Wall–Wall block
        kernel!(grad_green, green_temp, wall, wall, n)
        # Wall–Plasma block
        kernel!(grad_green, green_temp, wall, plasma_surf, n)

        # Fourier transform obs=wall, src=plasma block
        fourier_transform!(grre, green_temp, cos_mn_basis; row_offset=WALL_ROW_OFFSET)
        fourier_transform!(grre, green_temp, sin_mn_basis; row_offset=WALL_ROW_OFFSET, col_offset=IMAG_COL_OFFSET)
    end

    # Add cn0 to make grdgre nonsingular for n=0 modes
    cn0 = 1.0 # expose to user if anyone ever actually tries to use this
    (n == 0 && !wall.nowall) && begin
        @warn "Adding $cn0 to diagonal of grdgre to regularize n=0 mode; this may affect accuracy of results."
        mth12 = wall.nowall ? mtheta : 2 * mtheta
        for i in 1:mth12, j in 1:mth12
            grad_green[i, j] += cn0
        end
    end

    # Compute both Green's functions: exterior (kernelsign=+1) then interior (kernelsign=-1).
    # Solve exterior first, then overwrite grad_green with interior kernel to avoid extra allocations.
    grre .= grad_green \ grre

    # TODO: Update this comment. I think the minus sign comes from a change of sign of the unit normal vector
    # and we add 2I since we included the terms from eq. 69 in the kernel! function. We could just
    # remove the diagonal terms from kernel! and do the mutliplication by -1 here before adding to both
    # Interior kernel is -grad_green + 2I on diagonal [Chance Phys. Plasmas 1997 2161 eq. 69].
    grad_green .*= -1
    for i in 1:num_points
        grad_green[i, i] += 2.0
    end
    grri .= grad_green \ grri

    wv = zeros(ComplexF64, mpert, mpert)
    xzpts = zeros(inputs.mtheta, 4)
    if !green_only
        # Perform inverse Fourier transforms to get response matrix components [Chance Phys. Plasmas 2007 052506 eq. 115-118]
        arr, aii, ari, air = ntuple(_ -> zeros(mpert, mpert), 4)
        fourier_inverse_transform!(arr, grre, cos_mn_basis)
        fourier_inverse_transform!(aii, grre, sin_mn_basis; col_offset=IMAG_COL_OFFSET)
        fourier_inverse_transform!(ari, grre, sin_mn_basis)
        fourier_inverse_transform!(air, grre, cos_mn_basis; col_offset=IMAG_COL_OFFSET)

        # Final form of vacuum response matrix [Chance Phys. Plasmas 2007 052506 eq. 114]
        wv .= complex.(arr .+ aii, air .- ari)

        # Force symmetry of response matrix if desired
        force_wv_symmetry && hermitianpart!(wv)

        # Create xzpts array
        @views xzpts[:, 1] .= plasma_surf.x
        @views xzpts[:, 2] .= plasma_surf.z
        @views xzpts[:, 3] .= wall.x
        @views xzpts[:, 4] .= wall.z
    end

    return wv, grri, grre, xzpts
end

function compute_vacuum_response_3D(inputs::VacuumInput3D, wall_settings::WallShapeSettings; PATCH_RAD::Int=11, RAD_DIM::Int=20, INTERP_ORDER::Int=5)

    # Initialize surfaces
    # TODO: Currently only supports axisymmetric surfaces
    plasma_surf = PlasmaGeometry3D(inputs)
    wall = WallGeometry3D(inputs, plasma_surf, wall_settings)

    (; mtheta, mpert, mlow, force_wv_symmetry, nzeta, npert, nlow) = inputs
    num_modes = npert * mpert
    # If no wall, only plasma points; if wall, plasma + wall points
    num_points = wall.nowall ? nzeta * mtheta : 2 * nzeta * mtheta

    # Compute Fourier basis coefficients
    cos_mn_basis, sin_mn_basis = compute_fourier_coefficients(mtheta, mpert, mlow, nzeta, npert, nlow; ν=plasma_surf.ν)

    # Allocate matrices, accounting for whether wall is present or not
    grri = zeros(num_points, 2 * num_modes)  # Interior (kernelsign=-1)
    grre = zeros(num_points, 2 * num_modes)  # Exterior (kernelsign=+1)
    grad_green = zeros(num_points, num_points)
    green_temp = zeros(nzeta * mtheta, nzeta * mtheta)

    # Fourier transforms offsets into grri/grre: first mtheta rows are plasma as observer, second are wall
    # First mpert columns are real, second mpert are imaginary
    WALL_ROW_OFFSET = nzeta * mtheta
    IMAG_COL_OFFSET = num_modes

    # Plasma–Plasma block
    compute_3D_kernel_matrix!(grad_green, green_temp, plasma_surf, plasma_surf, PATCH_RAD, RAD_DIM, INTERP_ORDER)

    # Fourier transform obs=plasma, src=plasma block into grre
    fourier_transform!(grre, green_temp, cos_mn_basis)
    fourier_transform!(grre, green_temp, sin_mn_basis; col_offset=IMAG_COL_OFFSET)

    if !wall.nowall
        # Plasma–Wall block
        compute_3D_kernel_matrix!(grad_green, green_temp, plasma_surf, wall, PATCH_RAD, RAD_DIM, INTERP_ORDER)
        # Wall–Wall block
        compute_3D_kernel_matrix!(grad_green, green_temp, wall, wall, PATCH_RAD, RAD_DIM, INTERP_ORDER)
        # Wall–Plasma block
        compute_3D_kernel_matrix!(grad_green, green_temp, wall, plasma_surf, PATCH_RAD, RAD_DIM, INTERP_ORDER)

        # Fourier transform obs=wall, src=plasma block into grre
        fourier_transform!(grre, green_temp, cos_mn_basis; row_offset=WALL_ROW_OFFSET)
        fourier_transform!(grre, green_temp, sin_mn_basis; row_offset=WALL_ROW_OFFSET, col_offset=IMAG_COL_OFFSET)
    end

    # Add the term that comes from the volume integral of Green's identity
    for i in 1:num_points
        grad_green[i, i] += 1.0
    end
    # Compute both Green's functions: exterior (kernelsign=+1) then interior (kernelsign=-1).
    # Solve exterior first, then overwrite grad_green with interior kernel to avoid extra allocations.
    grre .= grad_green \ grre

    # Interior flips the sign of the normal, but not the diagonal terms, so we multiply by -1 and add 2I to the diagonal
    grad_green .*= -1
    for i in 1:num_points
        grad_green[i, i] += 2.0
    end
    grri .= grad_green \ grri

    # Perform inverse Fourier transforms to get response matrix components (eq. 115-118 of Chance 2007)
    arr, aii, ari, air = ntuple(_ -> zeros(num_modes, num_modes), 4)
    fourier_inverse_transform!(arr, grre, cos_mn_basis)
    fourier_inverse_transform!(aii, grre, sin_mn_basis; col_offset=IMAG_COL_OFFSET)
    fourier_inverse_transform!(ari, grre, sin_mn_basis)
    fourier_inverse_transform!(air, grre, cos_mn_basis; col_offset=IMAG_COL_OFFSET)

    # Final form of vacuum response matrix (eq. 114 of Chance 2007)
    wv = complex.(arr .+ aii, air .- ari)

    # Force symmetry of response matrix if desired
    force_wv_symmetry && hermitianpart!(wv)

    return wv, grre, plasma_surf.r, wall.r
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
