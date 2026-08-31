module Vacuum

using TOML, SpecialFunctions, LinearAlgebra, Printf
using FastInterpolations
using FastGaussQuadrature: gausslegendre
using StaticArrays: SVector
using SparseArrays
using AdaptiveArrayPools

# Import parent modules
import ..Equilibrium
using ..Utilities.FourierTransforms: FourierTransform, compute_fourier_coefficients

include("Utilities.jl")
include("DataTypes.jl")
include("PnQuadCache.jl")
include("Kernel2D.jl")
include("Kernel3D.jl")
include("Field.jl")

export VacuumInput, VacuumResponse, WallShapeSettings
export compute_vacuum_response, compute_vacuum_response!, compute_vacuum_field
export extract_plasma_surface_at_psi
export PlasmaGeometry

# Relative anti-Hermitian residual above which we warn that the vacuum grid should be refined.
const _HERMITICITY_WARN_TOL = 1e-4

"""
    _warn_and_symmetrize!(mat, name)

Replace `mat` by its Hermitian part in place, warning first if the anti-Hermitian residual exceeds
`_HERMITICITY_WARN_TOL`.

The matrices passed here are Hermitian in exact arithmetic — δW_v = ξ†Wᵛξ is a real energy, and Iᵛ
is the inverse of the Hermitian surface inductance up to a real scalar — so any anti-Hermitian part
is a discretization artifact that vanishes as the vacuum grid is refined.
"""
function _warn_and_symmetrize!(mat::AbstractMatrix, name::String)
    herm_norm = norm(mat + mat')
    if herm_norm > 0
        # Relative anti-Hermitian residual ‖½(M−M†)‖/‖½(M+M†)‖
        rel_residual = norm(mat - mat') / herm_norm
        if rel_residual > _HERMITICITY_WARN_TOL
            @warn "$name is non-Hermitian above tolerance $(rel_residual) > $(_HERMITICITY_WARN_TOL) before " *
                  "symmetrization. Increase vacuum grid resolution to reduce it."
        end
    end
    hermitianpart!(mat)
end

"""
    _fill_Iv_block!(I_v_block, basis, grre, grri, num_points)

Surface-current matrix Iᵛ from the exterior and interior potentials, Park 2007 eq. 21b:
`μ₀I^v = χ^(vi) - χ^(vo)`. Overwrites the plasma-observer rows of `grri`.

The difference is taken as `grre - grri` and the result conjugated because VACUUM builds the
operators in its CW-θ frame while GPEC uses CCW-θ, flipping the outward-normal sign.
"""
function _fill_Iv_block!(I_v_block::AbstractMatrix, basis::AbstractMatrix, grre::AbstractMatrix, grri::AbstractMatrix, num_points::Int)
    g_diff = @view grri[1:num_points, :]
    g_diff .= @view(grre[1:num_points, :]) .- g_diff
    mul!(I_v_block, basis, g_diff)
    conj!(I_v_block) # Flip θ_VAC → -θ_VAC to get I^v in GPEC's CCW-θ frame.
    I_v_block ./= num_points
    return I_v_block
end

"""
    _compute_vacuum_response_2d!(vac_data::VacuumResponse, inputs::VacuumInput, wall_settings::WallShapeSettings; compute_Iv=false)

2D (axisymmetric) vacuum response calculation.

Each toroidal mode `n` decouples in 2D geometry, so the routine loops over `inputs.n_modes`,
building the double-/single-layer operators, solving the exterior system for `wv`, and
optionally the interior system to build `I_v` when `compute_Iv=true`.
Green's functions are internal scratch only.
"""
@with_pool pool function _compute_vacuum_response_2d!(vac_data::VacuumResponse, inputs::VacuumInput, wall_settings::WallShapeSettings; compute_Iv::Bool=false)

    mpert = length(inputs.m_modes)
    mlow = inputs.m_modes[1]
    num_points_surf = inputs.mtheta

    fill!(vac_data.wv, 0)
    fill!(vac_data.I_v, 0)

    # Form the plasma and wall geometries
    plasma_surf = PlasmaGeometry(inputs)
    wall = WallGeometry(inputs, plasma_surf, wall_settings)

    # Loop over all decoupled toroidal modes
    for (idx_n, n) in enumerate(inputs.n_modes)
        ft = FourierTransform(inputs.mtheta, mpert, mlow; n=n, ν=plasma_surf.ν)

        # Diagonal block of wv (and I_v when requested)
        block_idx = ((idx_n-1)*mpert+1):(idx_n*mpert)
        wv_block = @view vac_data.wv[block_idx, block_idx]

        # Active rows for computation (plasma only if no wall, plasma+wall if wall present)
        num_points_total = wall.nowall ? num_points_surf : 2 * num_points_surf

        # Local work matrices
        grad_green = zeros!(pool, num_points_total, num_points_total)
        green_temp = zeros!(pool, num_points_surf, num_points_surf)
        grre = zeros!(pool, ComplexF64, num_points_total, mpert)

        # Plasma–Plasma block
        compute_2D_kernel_matrices!(grad_green, green_temp, plasma_surf, plasma_surf, n)

        # Project plasma observer onto source basis exp(i*(mθ - nν))
        mul!(view(grre, 1:num_points_surf, :), green_temp, ft.basis')

        if !wall.nowall
            # Plasma–Wall block
            compute_2D_kernel_matrices!(grad_green, green_temp, plasma_surf, wall, n)
            # Wall–Wall block
            compute_2D_kernel_matrices!(grad_green, green_temp, wall, wall, n)
            # Wall–Plasma block
            compute_2D_kernel_matrices!(grad_green, green_temp, wall, plasma_surf, n)
            # Project wall observer onto source basis exp(i*(mθ - nν))
            mul!(view(grre, (num_points_surf+1):num_points_total, :), green_temp, ft.basis')
        end

        if compute_Iv
            # Copy RHS before exterior solve overwrites grre; keep a kernel copy for interior
            grri = similar!(pool, grre)
            grri .= grre
            grad_green_interior = similar!(pool, grad_green)
            grad_green_interior .= grad_green

            # Exterior operator D_ext = 2I + 𝒦 (Chance 1997 eq. 89); the solve gives
            # grre = -(2π)²χ^(vo), the vacuum-outside potential. Overwrites grad_green to save memory.
            ldiv!(lu!(grad_green), grre)

            # Interior operator D_int = D_ext - 2I: the double-layer jump between the two one-sided
            # boundary limits is 2I here, giving the vacuum-inside potential grri = χ^(vi).
            for i in 1:num_points_total
                grad_green_interior[i, i] -= 2.0
            end
            ldiv!(lu!(grad_green_interior), grri)

            _fill_Iv_block!(@view(vac_data.I_v[block_idx, block_idx]), ft.basis, grre, grri, num_points_surf)
        else
            # Only need exterior system for wv
            ldiv!(lu!(grad_green), grre)
        end

        # Project exterior kernel onto observer basis exp(-i*(mθ - nν)) and scale to get the response matrix
        mul!(wv_block, ft.basis, @view(grre[1:num_points_surf, :]))
        wv_block .*= 4π^2 / num_points_surf
    end

    # Remove any non-Hermitian residual from Hermitian matrices due to discretization
    _warn_and_symmetrize!(vac_data.wv, "Wᵛ")
    compute_Iv && _warn_and_symmetrize!(vac_data.I_v, "Iᵛ")

    # Populate coordinate arrays
    @views begin
        vac_data.plasma_pts[:, 1] .= plasma_surf.x
        vac_data.plasma_pts[:, 2] .= 0.0
        vac_data.plasma_pts[:, 3] .= plasma_surf.z
        vac_data.wall_pts[:, 1] .= wall.x
        vac_data.wall_pts[:, 2] .= 0.0
        vac_data.wall_pts[:, 3] .= wall.z
    end
end

"""
    _compute_vacuum_response_3d!(vac_data::VacuumResponse, inputs::VacuumInput, wall_settings::WallShapeSettings; compute_Iv=false)

3D (`inputs.nzeta > 1`) vacuum response via block-circulant field-period reduction.

The `nfp`-periodic boundary makes the single-/double-layer operators `S`, `D` block-circulant in the
field-period index, so the problem block-diagonalizes by toroidal residue class `k = mod(n, nfp)`:
modes with different `k` do not couple, and within a class

    D̂ₖ = Σ_d D_d ω^{k d},   Ŝₖ = Σ_d S_d ω^{k d},   ω = exp(-2πi/nfp),

with `D_d`, `S_d` the blocks coupling observers in field period 0 to sources in period `d`. Each class
needs one solve `wv[class k] = (4π²/M)·E_localᴴ·(D̂ₖ \\ Ŝₖ)|_plasma·E_local` (`E` the complex Fourier
basis, `M = mtheta·nzeta`), so the routine loops over classes exactly as the 2D routine loops over
decoupled `n`. The phase sum is folded into the kernel write, so only the reduced `[nb·M × nb·M]`
operator is ever stored; for `nfp == 1` every phase is unity and the operators stay real.

With `compute_Iv=true` each class also solves the interior operator `D_int = D_ext - 2I`, as in 2D.
That shift is block-diagonal in the field-period index, so it enters only the `d = 0` block and the
reduction stays exact.
"""
@with_pool pool function _compute_vacuum_response_3d!(vac_data::VacuumResponse, inputs::VacuumInput, wall_settings::WallShapeSettings; compute_Iv::Bool=false)

    (; mtheta, nzeta, nfp, m_modes, n_modes) = inputs
    fill!(vac_data.wv, 0)
    fill!(vac_data.I_v, 0)

    # Full-torus geometry for source surface; observers are restricted to one field period
    full = expand_field_periods(inputs)
    plasma_surf = PlasmaGeometry3D(full)
    wall = WallGeometry3D(full, plasma_surf, wall_settings)

    num_points_per_fp = mtheta * nzeta  # points per field period
    mpert = length(m_modes)
    nb = wall.nowall ? 1 : 2            # surface blocks: plasma, or [plasma; wall]
    n_obs = nb * num_points_per_fp      # observer rows: plasma (and wall) points of one field period

    # Complex Fourier basis exp(-i(mθ-nζ)) on a single field period
    exp_mn_basis = compute_fourier_coefficients(mtheta, m_modes, nzeta * nfp, n_modes; nfp=nfp)

    # Kernel parameters, hardcoded for now
    PATCH_RAD = 11
    RAD_DIM = 20
    INTERP_ORDER = 5

    # Work matrices, real for nfp == 1 where every field-period phase is unity
    T = nfp == 1 ? Float64 : ComplexF64
    grad_green = zeros!(pool, T, n_obs, nb * num_points_per_fp) # reduced double-layer D̂ₖ (with +I on the diagonals)
    green_temp = zeros!(pool, T, n_obs, num_points_per_fp)      # reduced single-layer Ŝₖ (plasma sources only)
    grre = zeros!(pool, ComplexF64, n_obs, mpert * length(n_modes))
    grri = compute_Iv ? similar!(pool, grre) : zeros!(pool, ComplexF64, 0, 0)
    grad_green_interior = compute_Iv ? similar!(pool, grad_green) : zeros!(pool, T, 0, 0)

    # Loop over all decoupled toroidal residue classes
    for k in unique(mod.(n_modes, nfp))
        cols = [(idx_m + (idx_n-1)*mpert) for (idx_n, n) in enumerate(n_modes) if mod(n, nfp) == k for idx_m in 1:mpert]
        # A contiguous class (always so for nfp == 1) keeps the basis and output views strided, and so on the BLAS path
        mode_cols = length(cols) == cols[end] - cols[1] + 1 ? (cols[1]:cols[end]) : cols
        # Diagonal block of wv (and I_v when requested)
        wv_block = @view vac_data.wv[mode_cols, mode_cols]
        E = @view exp_mn_basis[mode_cols, :]
        phases = T[cis(-2π * (k * d) / nfp) for d in 0:(nfp-1)]
        grre_k = @view grre[:, 1:length(mode_cols)]

        # Plasma–Plasma block
        compute_3D_kernel_matrices!(grad_green, view(green_temp, 1:num_points_per_fp, :), plasma_surf, plasma_surf, PATCH_RAD, RAD_DIM, INTERP_ORDER, phases)

        if !wall.nowall
            # Plasma–Wall block
            compute_3D_kernel_matrices!(grad_green, view(green_temp, 1:num_points_per_fp, :), plasma_surf, wall, PATCH_RAD, RAD_DIM, INTERP_ORDER, phases)
            # Wall–Plasma block
            compute_3D_kernel_matrices!(grad_green, view(green_temp, (num_points_per_fp+1):n_obs, :), wall, plasma_surf, PATCH_RAD, RAD_DIM, INTERP_ORDER, phases)
            # Wall–Wall block
            compute_3D_kernel_matrices!(grad_green, view(green_temp, (num_points_per_fp+1):n_obs, :), wall, wall, PATCH_RAD, RAD_DIM, INTERP_ORDER, phases)
        end

        # Project observers onto source basis exp(i(mθ-nζ)); D⁻¹ acts on rows, so this commutes with the solve
        mul!(grre_k, green_temp, E')

        if compute_Iv
            # Copy RHS before exterior solve overwrites grre; keep a kernel copy for interior
            grri_k = @view grri[:, 1:length(mode_cols)]
            grri_k .= grre_k
            grad_green_interior .= grad_green

            # Exterior operator D_ext = 2I + 𝒦 (Chance 1997 eq. 89); the solve gives
            # grre = -(2π)²χ^(vo), the vacuum-outside potential. Overwrites grad_green to save memory.
            ldiv!(lu!(grad_green), grre_k)

            # Interior operator D_int = D_ext - 2I: the double-layer jump between the two one-sided
            # boundary limits is 2I here, giving the vacuum-inside potential grri = χ^(vi).
            for i in 1:n_obs
                grad_green_interior[i, i] -= 2.0
            end
            ldiv!(lu!(grad_green_interior), grri_k)

            _fill_Iv_block!(@view(vac_data.I_v[mode_cols, mode_cols]), E, grre_k, grri_k, num_points_per_fp)
        else
            # Only need exterior system for wv
            ldiv!(lu!(grad_green), grre_k)
        end

        # Project exterior kernel onto observer basis exp(-i(mθ-nζ)) and scale to get the response matrix
        mul!(wv_block, E, @view(grre_k[1:num_points_per_fp, :]))
        wv_block .*= 4π^2 / num_points_per_fp
    end

    # Remove any non-Hermitian residual from Hermitian matrices due to discretization
    _warn_and_symmetrize!(vac_data.wv, "Wᵛ")
    compute_Iv && _warn_and_symmetrize!(vac_data.I_v, "Iᵛ")

    # Populate coordinate arrays
    vac_data.plasma_pts .= plasma_surf.r
    vac_data.wall_pts .= wall.r
end

"""
    compute_vacuum_response(inputs::VacuumInput, wall_settings::WallShapeSettings; compute_Iv=false) -> VacuumResponse

Compute the vacuum response for the given inputs. Allocating wrapper around
[`compute_vacuum_response!`](@ref); pass a preallocated [`VacuumResponse`](@ref) to that method
instead when reusing storage across calls. Pass `compute_Iv=true` to additionally populate the
surface-current matrix `I_v`.
"""
function compute_vacuum_response(inputs::VacuumInput, wall_settings::WallShapeSettings; compute_Iv::Bool=false)
    vac = VacuumResponse(inputs)
    compute_vacuum_response!(vac, inputs, wall_settings; compute_Iv)
    return vac
end

"""
    compute_vacuum_response!(vac_data::VacuumResponse, inputs::VacuumInput, wall_settings::WallShapeSettings; compute_Iv=false)

In-place variant that populates the arrays of an existing [`VacuumResponse`](@ref). Dispatches on
dimensionality only: 2D (`inputs.nzeta == 1`) routes to [`_compute_vacuum_response_2d!`], 3D to
[`_compute_vacuum_response_3d!`].
"""
function compute_vacuum_response!(vac_data::VacuumResponse, inputs::VacuumInput, wall_settings::WallShapeSettings; compute_Iv::Bool=false)
    if inputs.nzeta == 1
        _compute_vacuum_response_2d!(vac_data, inputs, wall_settings; compute_Iv)
    else
        _compute_vacuum_response_3d!(vac_data, inputs, wall_settings; compute_Iv)
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
