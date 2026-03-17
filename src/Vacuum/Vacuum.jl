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
    _compute_vacuum_response_single!(
        wv, grri_in, grre_in, plasma_pts, wall_pts,
        inputs::VacuumInput, wall_settings::WallShapeSettings;
        n_override=nothing
    )

Compute a single vacuum solve (one coupled 3D solve, or one `n`-slice in 2D) by building and solving
the boundary integral equation in mode space with an optional conducting wall present and writing out the results:

  - `wv`: complex vacuum response matrix in straight-fieldline mode space
  - `grri_in`: interior Green's function sampled on the plasma surface in straight-fieldline mode space (real layout for backward compatibility)
  - `grre_in`: exterior Green's function sampled on the plasma surface in straight-fieldline mode space (real layout for backward compatibility)
  - `plasma_pts` / `wall_pts`: output point clouds for downstream plotting/diagnostics

## Fused kernel assembly + projection

This routine uses a **newer kernel evaluation path** that never forms dense point-space kernel matrices.
Instead, it fuses kernel evaluation and Fourier/Galerkin projection into a single pass.

The key idea is:

  - Assemble and solve the boundary integral equation directly in `P×P` mode space.

  - Avoid materializing `M×M` (2D) or `N×N` (3D) kernel matrices.

  - Uses complex basis `Z = C + iS` so projected operators are `P×P` complex.

  - The projected operators are accumulated row-by-row while kernel values are computed.

  - Memory drops from `O(M^2)` (or `O(N^2)`) down to `O(MP + P^2)` (or `O(NP + P^2)`).

  - FLOPs remain dominated by the same scaling as the two-step approach (kernel evaluation + projection),
    plus an additional `O(P^3)` for the LU factorization/solve in mode space.

  - **Projected matrices**

      + Exterior projected kernel blocks are assembled into `K_ext` and `G_ext`.
      + Interior operators are formed from the exterior ones using the discrete Green-identity diagonal term:
        the implementation uses `K_int = 2*Gram - K_ext` for same-type source/observer blocks. This effectively
        computes the kernel with an negative normal direction without recalculating the kernel.

  - **Solves**

      + If `nowall`, solve the plasma-only `P×P` system.
      + If a wall is present, solve the coupled `2P×2P` block system.

  - **Back-compat outputs**

      + Although the solve is performed in mode space, `grri_in` and `grre_in` are reconstructed into the
        legacy real `M×(2P)` layout for downstream code paths that still expect that shape.

## Arguments

  - **`wv::AbstractMatrix{ComplexF64}`**: output vacuum response matrix (modified in-place)
  - **`grri_in::AbstractMatrix{Float64}`**: output interior Green's function (modified in-place; real/legacy layout)
  - **`grre_in::AbstractMatrix{Float64}`**: output exterior Green's function (modified in-place; real/legacy layout)
  - **`plasma_pts::AbstractMatrix{Float64}`**: plasma surface coordinates (modified in-place)
  - **`wall_pts::AbstractMatrix{Float64}`**: wall surface coordinates (modified in-place)
  - **`inputs::VacuumInput`**: mode ranges, grid resolution, and geometry settings
  - **`wall_settings::WallShapeSettings`**: wall geometry configuration
  - **`n_override::Union{Nothing,Int}`**: optional toroidal mode number override (only used for 2D)

## 2D vs 3D behavior

  - **3D (`inputs.nzeta > 1`)**: computes the full coupled response across all `(m, n)` modes specified by
    `inputs.(mlow, mpert, nlow, npert)` in a single call using the 3D kernel method in Kernel3D.jl.
  - **2D (`inputs.nzeta == 1`)**:
      + If `inputs.npert == 1`, computes the response for `n = inputs.nlow` using the 2D kernel method in Kernel2D.jl.
      + If `inputs.npert > 1`, the public driver loops over `n` and calls this function once per `n`,
        writing block columns into the full output matrices using the 2D kernel method in Kernel2D.jl.
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
    num_points, num_modes = size(exp_mn_basis)

    # Create kernel parameters structs used to dispatch to the correct kernel
    # Hardcode these values for now - can expose to the user in the future
    PATCH_RAD = 11
    RAD_DIM = 20
    INTERP_ORDER = 5
    kparams = nzeta > 1 ? KernelParams3D(PATCH_RAD, RAD_DIM, INTERP_ORDER) : KernelParams2D(n_override)

    # Scales kernel matrix sizes by a factor of 2 if a wall is present (don't allocate unless needed)
    wall_fac = wall.nowall ? 1 : 2

    # Gram matrix required by projected_kernel! for the diagonal residue and for interior solve
    Gram = zeros!(pool, ComplexF64, num_modes, num_modes)
    mul!(Gram, exp_mn_basis', exp_mn_basis)

    # Projected kernel matrices [P × P complex]
    K_ext = zeros!(pool, ComplexF64, wall_fac * num_modes, wall_fac * num_modes)
    G_ext = zeros!(pool, ComplexF64, wall_fac * num_modes, num_modes)
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
    K_int[1:num_modes, 1:num_modes] .+= 2 .* Gram
    if !wall.nowall
        K_int[(num_modes+1):(2*num_modes), (num_modes+1):(2*num_modes)] .+= 2 .* Gram
    end
    G_int .= G_ext

    # Solve projected BIEs for exterior and interior kernels
    F_ext = lu!(K_ext)
    ldiv!(F_ext, G_ext)
    F_int = lu!(K_int)
    ldiv!(F_int, G_int)

    # Construct the vacuum response matrix: wv = (4π²/M) · Gram · G
    mul!(wv, Gram, view(G_ext, 1:num_modes, :))
    wv .*= (4π^2 / num_points)

    # Enforce Hermitian symmetry if desired
    inputs.force_wv_symmetry && hermitianpart!(wv)

    # Backward-compatible reconstruction: grre/grri in M×2P real layout
    # Need to convert mode space to physical space and unpack the real and imaginary parts
    # TODO: propagate complex M * P grri/grre matrices to perturbed equilibrium code
    # perhaps make it a complex P * P matrix? Then don't need any of this section
    # Views into output Green's function matrices for the active rows/columns
    grre = @view grre_in[1:(wall_fac*num_points), :]
    grri = @view grri_in[1:(wall_fac*num_points), :]
    temp = zeros!(pool, ComplexF64, num_points, num_modes)

    mul!(temp, exp_mn_basis, view(G_ext, 1:num_modes, :))
    @view(grre[1:num_points, 1:num_modes]) .= real.(temp)
    @view(grre[1:num_points, (num_modes+1):(2*num_modes)]) .= imag.(temp)
    mul!(temp, exp_mn_basis, view(G_int, 1:num_modes, :))
    @view(grri[1:num_points, 1:num_modes]) .= real.(temp)
    @view(grri[1:num_points, (num_modes+1):(2*num_modes)]) .= imag.(temp)
    if !wall.nowall
        mul!(temp, exp_mn_basis, view(G_ext, (num_modes+1):(2*num_modes), :))
        @view(grre[(num_points+1):(2*num_points), 1:num_modes]) .= real.(temp)
        @view(grre[(num_points+1):(2*num_points), (num_modes+1):(2*num_modes)]) .= imag.(temp)
        mul!(temp, exp_mn_basis, view(G_int, (num_modes+1):(2*num_modes), :))
        @view(grri[(num_points+1):(2*num_points), 1:num_modes]) .= real.(temp)
        @view(grri[(num_points+1):(2*num_points), (num_modes+1):(2*num_modes)]) .= imag.(temp)
    end

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
