module Vacuum

using TOML, SpecialFunctions, LinearAlgebra, Printf
using FastInterpolations: cubic_interp, deriv1, PeriodicBC, NaturalBC
using FastGaussQuadrature: gausslegendre
using StaticArrays: SVector
using AdaptiveArrayPools

# Import parent modules
import ..Equilibrium
using ..Utilities.FourierTransforms: compute_fourier_coefficients, fourier_transform!, fourier_inverse_transform!

include("DataTypes.jl")
include("PnQuadCache.jl")
include("Kernel2D.jl")
include("MathUtils.jl")
include("VacuumFromEquilibrium.jl")

export VacuumInput, compute_vacuum_response
export compute_vacuum_field
export kernel!
export WallShapeSettings
export extract_plasma_surface_at_psi

"""
    apply_kernelsign!(grad_greenfunction_mat, kernelsign, mtheta)

Apply kernelsign transformation to Green's function matrix.
For kernelsign < 0 (interior potential), multiply by -1 and add 2 to diagonal.
"""
function apply_kernelsign!(grad_greenfunction_mat::AbstractMatrix{Float64}, kernelsign::Float64, mtheta::Int)
    if kernelsign < 0
        grad_greenfunction_mat .*= kernelsign
        # Account for factor of 2 in diagonal terms [Chance Phys. Plasmas 1997 2161 eq. 90]
        for i in 1:(2*mtheta)
            grad_greenfunction_mat[i, i] += 2.0
        end
    end
    return nothing
end

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
@with_pool pool function compute_vacuum_response(inputs::VacuumInput, wall_settings::WallShapeSettings; green_only=false)

    # Initialization and allocations
    (; mtheta, mpert, mlow, n, force_wv_symmetry) = inputs
    plasma_surf = PlasmaGeometry(inputs)
    wall = WallGeometry(inputs, plasma_surf, wall_settings)

    # Compute Fourier basis coefficients using FourierTransforms utility
    # We only need the coefficient arrays for the existing fourier_transform! functions
    cos_mn_basis, sin_mn_basis = compute_fourier_coefficients(mtheta, mpert, mlow; n=n, ν=plasma_surf.ν)

    # Allocate arrays for both Green's functions
    grri = zeros(2 * mtheta, 2 * mpert)  # Interior (kernelsign=-1)
    grre = zeros(2 * mtheta, 2 * mpert)  # Exterior (kernelsign=+1)
    grad_green = zeros!(pool, 2 * mtheta, 2 * mtheta)
    green_temp = zeros!(pool, mtheta, mtheta)

    # Fourier transforms offsets into grri/grre: first mtheta rows are plasma as observer, second are wall
    # First mpert columns are real (cosine), second mpert are imaginary (sine)
    PLASMA_ROW_OFFSET = 0
    WALL_ROW_OFFSET = mtheta
    COS_COL_OFFSET = 0
    SIN_COL_OFFSET = mpert

    # Plasma–Plasma block
    kernel!(grad_green, green_temp, plasma_surf, plasma_surf, n)

    # Fourier transform obs=plasma, src=plasma block
    fourier_transform!(grri, green_temp, cos_mn_basis, PLASMA_ROW_OFFSET, COS_COL_OFFSET)
    fourier_transform!(grri, green_temp, sin_mn_basis, PLASMA_ROW_OFFSET, SIN_COL_OFFSET)
    fourier_transform!(grre, green_temp, cos_mn_basis, PLASMA_ROW_OFFSET, COS_COL_OFFSET)
    fourier_transform!(grre, green_temp, sin_mn_basis, PLASMA_ROW_OFFSET, SIN_COL_OFFSET)

    if !wall.nowall
        # Plasma–Wall block
        kernel!(grad_green, green_temp, plasma_surf, wall, n)
        # Wall–Wall block
        kernel!(grad_green, green_temp, wall, wall, n)
        # Wall–Plasma block
        kernel!(grad_green, green_temp, wall, plasma_surf, n)

        # Fourier transform obs=wall, src=plasma block
        fourier_transform!(grri, green_temp, cos_mn_basis, WALL_ROW_OFFSET, COS_COL_OFFSET)
        fourier_transform!(grri, green_temp, sin_mn_basis, WALL_ROW_OFFSET, SIN_COL_OFFSET)
        fourier_transform!(grre, green_temp, cos_mn_basis, WALL_ROW_OFFSET, COS_COL_OFFSET)
        fourier_transform!(grre, green_temp, sin_mn_basis, WALL_ROW_OFFSET, SIN_COL_OFFSET)
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

    # Compute both Green's functions with different kernel signs
    # grri: interior potential (kernelsign=-1)
    # grre: exterior potential (kernelsign=+1)

    # Make copies for each kernelsign
    grad_green_interior = similar!(pool, grad_green)
    grad_green_exterior = similar!(pool, grad_green)
    grad_green_interior .= grad_green
    grad_green_exterior .= grad_green

    # Apply kernelsign transformations
    apply_kernelsign!(grad_green_interior, -1.0, mtheta)  # Interior
    apply_kernelsign!(grad_green_exterior, +1.0, mtheta)  # Exterior (no-op)

    # Invert the vacuum response system for both cases
    # If plasma only, lower blocks will be empty
    if wall.nowall
        F_int = lu!(@view grad_green_interior[1:mtheta, 1:mtheta])
        ldiv!(F_int, @view grri[1:mtheta, :])
        F_ext = lu!(@view grad_green_exterior[1:mtheta, 1:mtheta])
        ldiv!(F_ext, @view grre[1:mtheta, :])
    else
        F_int = lu!(grad_green_interior)
        ldiv!(F_int, grri)
        F_ext = lu!(grad_green_exterior)
        ldiv!(F_ext, grre)
    end

    # There's some logic that computes xpass/zpass and chiwc/chiws here, might eventually be needed?

    wv = zeros(ComplexF64, mpert, mpert)
    xzpts = zeros(inputs.mtheta, 4)
    if !green_only
        # Perform inverse Fourier transforms to get response matrix components [Chance Phys. Plasmas 2007 052506 eq. 115-118]
        arr, aii, ari, air = ntuple(_ -> zeros(mpert, mpert), 4)
        fourier_inverse_transform!(arr, grre, cos_mn_basis, PLASMA_ROW_OFFSET, COS_COL_OFFSET)
        fourier_inverse_transform!(aii, grre, sin_mn_basis, PLASMA_ROW_OFFSET, SIN_COL_OFFSET)
        fourier_inverse_transform!(ari, grre, sin_mn_basis, PLASMA_ROW_OFFSET, COS_COL_OFFSET)
        fourier_inverse_transform!(air, grre, cos_mn_basis, PLASMA_ROW_OFFSET, SIN_COL_OFFSET)

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
