module Vacuum

using TOML, SpecialFunctions, LinearAlgebra, Printf
using FastInterpolations: cubic_interp, deriv1, PeriodicBC, NaturalBC
using FastGaussQuadrature: gausslegendre
using StaticArrays: SVector
using SparseArrays
using AdaptiveArrayPools

# Import parent modules
import ..Equilibrium
using ..Utilities.FourierTransforms: compute_fourier_coefficients

include("Utilities.jl")
include("DataTypes.jl")
include("PnQuadCache.jl")
include("Kernel2D.jl")
include("Kernel3D.jl")
include("Field.jl")

export VacuumInput, WallShapeSettings
export compute_vacuum_response, compute_vacuum_response!, compute_vacuum_field
export extract_plasma_surface_at_psi

"""
    compute_vacuum_response(inputs::VacuumInput, wall_settings::WallShapeSettings)

Compute the vacuum response matrix and both Green's functions using provided vacuum inputs.

Single entry point for vacuum calculations.

  - For **3D** (`inputs.nzeta > 1`), computes the full coupled response across all (m,n) modes defined
    by `inputs.(mlow, mpert, nlow, npert)`.

  - For **2D geometry** (`inputs.nzeta == 1`), supports either:

      + **single-n** (`inputs.npert == 1`): computes (m,n) response for `n = inputs.nlow`
      + **multi-n** (`inputs.npert > 1`): loops over `n = inputs.nlow:(inputs.nlow+inputs.npert-1)` and returns
        **blocks** of the full response matrices with one block per toroidal mode number.

This is the pure Julia implementation that replaces the Fortran `mscvac` function.
It computes both interior (grri) and exterior (grre) Green's functions for GPEC response calculations.

# Arguments

  - `inputs`: `VacuumInput` struct with mode numbers, grid resolution, and boundary info.
  - `wall_settings::WallShapeSettings`: Wall geometry configuration.

# Returns

  - `wv`: Complex vacuum response matrix.

      + 2D single-n: `mpert × mpert`
      + 2D multi-n: `(mpert*npert) × (mpert*npert)` (block diagonal)
      + 3D: `num_modes × num_modes` (full coupled)

  - `grri`: Interior Green's function matrix.

  - `grre`: Exterior Green's function matrix.

  - `xzpts`: Coordinate array (mtheta×4 for 2D, mtheta*nzeta×4 for 3D) [R_plasma, Z_plasma, R_wall, Z_wall].
"""
@with_pool pool function _compute_vacuum_response_single!(
    wv::AbstractMatrix{ComplexF64},
    grri_in::AbstractMatrix{Float64},
    grre_in::AbstractMatrix{Float64},
    plasma_pts::AbstractMatrix{Float64},
    wall_pts::AbstractMatrix{Float64},
    inputs::VacuumInput,
    wall_settings::WallShapeSettings;
    n_override::Union{Nothing,Int}=nothing
)

    (; mtheta, mpert, mlow, nzeta, npert, nlow) = inputs

    # Initialize surface geometries
    plasma_surf = nzeta > 1 ? PlasmaGeometry3D(inputs) : PlasmaGeometry(inputs)
    wall = nzeta > 1 ? WallGeometry3D(inputs, wall_settings) : WallGeometry(inputs, plasma_surf, wall_settings)

    # Compute Fourier basis coefficients
    ν = hasproperty(plasma_surf, :ν) ? plasma_surf.ν : nothing
    exp_mn_basis = compute_fourier_coefficients(mtheta, mpert, mlow, nzeta, npert, nlow; n_2D=n_override, ν=ν)
    num_points_surf, num_modes = size(exp_mn_basis)

    # Create kernel parameters structs used to dispatch to the correct kernel
    # Hardcode these values for now - can expose to the user in the future
    PATCH_RAD = 11
    RAD_DIM = 20
    INTERP_ORDER = 5
    kparams = nzeta > 1 ? KernelParams3D(PATCH_RAD, RAD_DIM, INTERP_ORDER) : KernelParams2D(n_override)

    # Active rows for computation (plasma only if no wall, plasma+wall if wall present)
    num_points_total = wall.nowall ? num_points_surf : 2 * num_points_surf

    # Views into output Green's function matrices for the active rows/columns
    grre = @view grre_in[1:num_points_total, :]
    grri = @view grri_in[1:num_points_total, :]

    # Complex buffer for projecting to mode space (G*Z) and back; grre/grri stay real for backwards compatibility
    M = num_points_surf
    P = num_modes
    temp = zeros!(pool, ComplexF64, M, P)

    # ================================================================
    # Galerkin: solve system in P×P mode space. Uses complex basis
    # Z = C + iS so projected matrices are P×P complex.
    #
    # Fused (fuse_projection=true): kernel assembly + Fourier projection
    # in one pass. The full M×M kernel matrices are never materialized —
    # instead the P×P projected matrices grad_green_fourier and G_c are
    # accumulated row by row as kernel values are computed.
    # Memory:  O(MP + P²)  instead of  O(M²)
    #
    # FLOPs:  O(M²P + P³)
    # ================================================================
    # Gram matrix required by projected_kernel! for the diagonal residue and for interior solve
    Gram = zeros!(pool, ComplexF64, P, P)
    mul!(Gram, exp_mn_basis', exp_mn_basis)

    # Projected kernel matrices [P × P complex]
    K_ext = zeros!(pool, ComplexF64, 2P, 2P)
    G_ext = zeros!(pool, ComplexF64, 2P, P)
    K_int = similar!(pool, K_ext)
    G_int = similar!(pool, G_ext)

    # Fused projected kernel: compute Z^H K Z and Z^H G Z for all operator blocks
    # Plasma-plasma block
    kernel!(K_ext, G_ext, plasma_surf, plasma_surf, kparams, exp_mn_basis, Gram)
    # Wall-plasma, plasma-wall, wall-wall blocks
    if !wall.nowall
        # Wall-plasma, plasma-wall, wall-wall blocks
        kernel!(K_ext, G_ext, plasma_surf, wall, kparams, exp_mn_basis, Gram)
        kernel!(K_ext, G_ext, wall, plasma_surf, kparams, exp_mn_basis, Gram)
        kernel!(K_ext, G_ext, wall, wall, kparams, exp_mn_basis, Gram)
    end

    # Interior kernel in real space: K_int = 2I - K_ext → Fourier transformed: K_int = 2·Gram - K_ext
    K_int .= -K_ext
    K_int[1:P, 1:P] .+= 2 .* Gram
    if !wall.nowall
        K_int[(P+1):(2*P), (P+1):(2*P)] .+= 2 .* Gram
    end
    G_int .= G_ext

    # Solve projected BIEs for exterior and interior kernels
    if wall.nowall
        F_ext = lu!(K_ext[1:P, 1:P])
        ldiv!(F_ext, @view(G_ext[1:P, :]))
        F_int = lu!(K_int[1:P, 1:P])
        ldiv!(F_int, @view(G_int[1:P, :]))
    else
        F_ext = lu!(K_ext)
        ldiv!(F_ext, G_ext)
        F_int = lu!(K_int)
        ldiv!(F_int, G_int)
    end

    # Construct the vacuum response matrix: wv = (4π²/M) · Gram · G
    mul!(wv, Gram, view(G_ext, 1:P, :))
    wv .*= (4π^2 / M)

    # Backward-compatible reconstruction: grre/grri in M×2P real layout
    # Need to convert mode space to physical space and unpack the real and imaginary parts
    # TODO: propagate complex M * P grri/grre matrices to perturbed equilibrium code
    # perhaps make it a complex P * P matrix? Then don't need any of this section
    mul!(temp, exp_mn_basis, view(G_ext, 1:P, :))
    @view(grre[1:M, 1:P]) .= real.(temp)
    @view(grre[1:M, (P+1):(2*P)]) .= imag.(temp)
    mul!(temp, exp_mn_basis, view(G_int, 1:P, :))
    @view(grri[1:M, 1:P]) .= real.(temp)
    @view(grri[1:M, (P+1):(2*P)]) .= imag.(temp)
    if !wall.nowall
        mul!(temp, exp_mn_basis, view(G_ext, (P+1):(2*P), :))
        @view(grre[(M+1):(2*M), 1:P]) .= real.(temp)
        @view(grre[(M+1):(2*M), (P+1):(2*P)]) .= imag.(temp)
        mul!(temp, exp_mn_basis, view(G_int, (P+1):(2*P), :))
        @view(grri[(M+1):(2*M), 1:P]) .= real.(temp)
        @view(grri[(M+1):(2*M), (P+1):(2*P)]) .= imag.(temp)
    end

    # Enforce symmetry in the vacuum response matrix if desired
    inputs.force_wv_symmetry && hermitianpart!(wv)

    if nzeta > 1 # 3D
        plasma_pts .= plasma_surf.r
        wall_pts .= wall.r
    else # 2D
        @views begin
            plasma_pts[:, 1] .= plasma_surf.x
            plasma_pts[:, 2] .= 0.0
            plasma_pts[:, 3] .= plasma_surf.z
            wall_pts[:, 1] .= wall.x
            wall_pts[:, 2] .= 0.0
            wall_pts[:, 3] .= wall.z
        end
    end
end

"""
    compute_vacuum_response(inputs::VacuumInput, wall_settings::WallShapeSettings)

Allocate and return the vacuum response matrix and Green's functions for the given
vacuum inputs.

This is a thin allocating wrapper around the in‑place [`compute_vacuum_response!`]
implementation. For performance‑critical paths that already own preallocated storage
(e.g. `ForceFreeStates.VacuumData`), prefer the in‑place method to avoid extra
heap allocations.
"""
@with_pool pool function compute_vacuum_response(inputs::VacuumInput, wall_settings::WallShapeSettings)

    # Allocate storage for the vacuum response matrix and Green's functions
    numpoints = inputs.mtheta * inputs.nzeta
    num_modes = inputs.mpert * inputs.npert
    vac = (
        wv=zeros!(pool, ComplexF64, num_modes, num_modes),
        grri=zeros!(pool, 2 * numpoints, 2 * num_modes),
        grre=zeros!(pool, 2 * numpoints, 2 * num_modes),
        plasma_pts=zeros!(pool, numpoints, 3),
        wall_pts=zeros!(pool, numpoints, 3)
    )

    compute_vacuum_response!(vac, inputs, wall_settings)

    return vac.wv, vac.grri, vac.grre, vac.plasma_pts, vac.wall_pts
end

"""
    compute_vacuum_response!(vac_data, inputs::VacuumInput, wall_settings::WallShapeSettings)

In-place variant that computes the vacuum response and directly populates the arrays
stored in `vac_data`.

The `vac_data` argument is expected to provide the following writable fields with
compatible sizes:

  - `wv::AbstractMatrix{ComplexF64}`             – vacuum response matrix
  - `grri::AbstractMatrix{Float64}`              – interior Green's functions
  - `grre::AbstractMatrix{Float64}`              – exterior Green's functions
  - `plasma_pts::AbstractMatrix{Float64}`        – plasma surface coordinates
  - `wall_pts::AbstractMatrix{Float64}`          – wall surface coordinates

This is designed to work with `ForceFreeStates.VacuumData` but does not depend on
its concrete type (duck-typed on field names only).
"""
function compute_vacuum_response!(vac_data, inputs::VacuumInput, wall_settings::WallShapeSettings)

    mpert = inputs.mpert
    npert = inputs.npert
    numpert_total = mpert * npert

    # 3D vacuum: full coupled response across (m,n) from a single kernel call
    if inputs.nzeta > 1
        _compute_vacuum_response_single!(
            vac_data.wv,
            vac_data.grri,
            vac_data.grre,
            vac_data.plasma_pts,
            vac_data.wall_pts,
            inputs,
            wall_settings
        )
    else
        # 2D vacuum: fill diagonal blocks of the response matrix
        vac_data.wv .= 0

        # Each n is independent in 2D geometry → fill diagonal blocks inside preallocated arrays
        ns = inputs.nlow:(inputs.nlow+inputs.npert-1)
        for (idx_n, n) in enumerate(ns)
            block_idx = ((idx_n-1)*mpert+1):(idx_n*mpert)
            cols = vcat(block_idx, numpert_total .+ block_idx)

            wv_block = @view vac_data.wv[block_idx, block_idx]
            grri_block = @view vac_data.grri[:, cols]
            grre_block = @view vac_data.grre[:, cols]

            _compute_vacuum_response_single!(
                wv_block,
                grri_block,
                grre_block,
                vac_data.plasma_pts,
                vac_data.wall_pts,
                inputs,
                wall_settings;
                n_override=n
            )
        end
    end
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
