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
const _WV_HERMITICITY_WARN_TOL = 1e-4

"""
    _symmetrize_vacuum_energy!(wv)

Enforce Hermiticity of the vacuum energy matrix Wᵛ in place.

Wᵛ is the generator of the vacuum magnetic energy, δW_v = ξ† Wᵛ ξ. Because that energy is a real
number for every perturbation ξ, the exact operator is Hermitian. The finite-resolution
boundary-integral quadrature (finite `mtheta`/`nzeta`) breaks exact Hermiticity, leaving a small
anti-Hermitian residual that is a pure discretization artifact and vanishes as the grid is refined.
We replace Wᵛ by its Hermitian part to restore this physical property, warning when the residual
is large enough that the vacuum grid should be refined.
"""
function _symmetrize_vacuum_energy!(wv::AbstractMatrix)

    herm_norm = norm(wv + wv')
    if herm_norm > 0
        # Relative anti-Hermitian residual ‖½(W−W†)‖/‖½(W+W†)‖
        rel_residual = norm(wv - wv') / herm_norm
        if rel_residual > _WV_HERMITICITY_WARN_TOL
            @warn "Vacuum energy matrix Wᵛ is non-Hermitian above tolerance $(rel_residual) > $(_WV_HERMITICITY_WARN_TOL) before " *
                  "symmetrization. Increase vacuum grid resolution to reduce it."
        end
    end
    hermitianpart!(wv)
end

"""
    _compute_vacuum_response_2d!(vac_data::VacuumResponse, inputs::VacuumInput, wall_settings::WallShapeSettings)

2D (axisymmetric) vacuum response calculation.

Each toroidal mode `n` decouples in 2D geometry, so the routine loops over `inputs.n_modes`,
building the double-/single-layer operators, solving the exterior and interior systems, and
filling the corresponding diagonal block of the response matrix and the matching column block
of the Green's functions.
"""
@with_pool pool function _compute_vacuum_response_2d!(vac_data::VacuumResponse, inputs::VacuumInput, wall_settings::WallShapeSettings)

    mpert = length(inputs.m_modes)
    num_points_surf = inputs.mtheta

    vac_data.wv .= 0

    # Form the plasma and wall geometries
    plasma_surf = PlasmaGeometry(inputs)
    wall = WallGeometry(inputs, plasma_surf, wall_settings)

    # Loop over all decoupled toroidal modes
    for (idx_n, n) in enumerate(inputs.n_modes)
        ft = FourierTransform(inputs.mtheta, mpert, inputs.m_modes[1]; n=n, ν=plasma_surf.ν)

        # Diagonal block of wv and matching column block of the Green's functions
        block_idx = ((idx_n-1)*mpert+1):(idx_n*mpert)
        wv_block = @view vac_data.wv[block_idx, block_idx]
        grri_block = @view vac_data.grri[:, block_idx]
        grre_block = @view vac_data.grre[:, block_idx]

        # Active rows for computation (plasma only if no wall, plasma+wall if wall present)
        num_points_total = wall.nowall ? num_points_surf : 2 * num_points_surf

        # Local work matrices
        grad_green = zeros!(pool, num_points_total, num_points_total)
        green_temp = zeros!(pool, num_points_surf, num_points_surf)

        # Views into output Green's function matrices for the active rows/columns
        grre = @view grre_block[1:num_points_total, :]
        grri = @view grri_block[1:num_points_total, :]

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

        # Compute both Green's functions: exterior (kernelsign=+1) then interior (kernelsign=-1)
        grri .= grre # start from same as exterior
        grad_green_interior = similar!(pool, grad_green)
        grad_green_interior .= grad_green

        # Solve exterior first, overwriting grad_green to save memory since we already have the interior kernel
        ldiv!(lu!(grad_green), grre)

        # Interior flips the sign of the normal, but not the diagonal terms, so we multiply by -1 and add 2I to the diagonal
        grad_green_interior .*= -1
        for i in 1:num_points_total
            grad_green_interior[i, i] += 2.0
        end
        ldiv!(lu!(grad_green_interior), grri)

        # Project exterior kernel onto observer basis exp(-i*(mθ - nν)) and scale to get the response matrix
        mul!(wv_block, ft.basis, @view(grre[1:num_points_surf, :]))
        wv_block .*= 4π^2 / num_points_surf
    end


    # δW_v = ξ† Wᵛ ξ is real, so Wᵛ must be Hermitian; remove any residual from discretization
    _symmetrize_vacuum_energy!(vac_data.wv)

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
    _compute_vacuum_response_3d!(vac_data::VacuumResponse, inputs::VacuumInput, wall_settings::WallShapeSettings)

3D (`inputs.nzeta > 1`) vacuum response via block-circulant field-period reduction. For `nfp == 1`
the block-circulant assembly and residue-class loop are skipped in favour of a more efficient
direct real solve on the built kernel matrices (equivalent to 2D).

The `nfp`-periodic boundary makes the single-/double-layer operators `S`, `D` block-circulant in
the field-period index. Writing the response as `wv = (4π²/N)·Eᴴ·D⁻¹S·E` (`E` the complex Fourier
basis, `N = mtheta·nzeta_full`), the structure block-diagonalizes the problem by toroidal residue
class `k = mod(n, nfp)`: modes with different `k` do not couple, and within a class

    D̂ₖ = Σ_d D_d ω^{k d},   Ŝₖ = Σ_d S_d ω^{k d},   ω = exp(-2πi/nfp),

with `D_d`, `S_d` the blocks coupling observers in field period 0 to sources in period `d`. Each
class needs one solve `wv[class k] = (4π²/M)·E_localᴴ·(D̂ₖ \\ Ŝₖ)|_plasma·E_local`. Only the first
block-row of the operators is built, so the kernel cost drops by `nfp` and the dense `O(N³)`
factorization is replaced by per-class `O(M³)` solves (`M = N/nfp`).

Only `wv` is produced currently; `grri`/`grre` are returned zeroed and are not yet supported in the 3D path.
Extension point: per residue class, apply the per-period basis to `D̂ₖ⁻¹Ŝₖ` for the exterior columns and the
interior variant `-D + 2I` for the interior columns, then scatter back into the `[2N × 2·num_modes]` arrays.
"""
@with_pool pool function _compute_vacuum_response_3d!(vac_data::VacuumResponse, inputs::VacuumInput, wall_settings::WallShapeSettings)

    (; mtheta, nzeta, nfp, m_modes, n_modes) = inputs
    fill!(vac_data.wv, 0)

    # Full-torus geometry for source surface; observers are restricted to one field period
    full = expand_field_periods(inputs)
    plasma_surf = PlasmaGeometry3D(full)
    wall = WallGeometry3D(full, wall_settings)

    num_points_per_fp = mtheta * nzeta # points per field period
    num_points = num_points_per_fp * nfp # full-torus point count
    mpert = length(m_modes)
    nb = wall.nowall ? 1 : 2            # surface blocks: plasma, or [plasma; wall]
    n_obs = nb * num_points_per_fp       # number of observer points per field period

    # Complex Fourier basis exp(-i(mθ-nζ)) on a single field period
    exp_mn_basis = compute_fourier_coefficients(mtheta, m_modes, nzeta * nfp, n_modes; nfp=nfp)

    # Local work matrices
    grad_green = zeros!(pool, n_obs, nb * num_points)   # double-layer D (with +I on the period-0 diagonals)
    green_temp = zeros!(pool, n_obs, num_points)       # single-layer S (plasma sources only)

    # Kernel parameters, hardcoded for now
    PATCH_RAD = 11
    RAD_DIM = 20
    INTERP_ORDER = 5

    # Plasma–plasma block
    compute_3D_kernel_matrices!(grad_green, view(green_temp, 1:num_points_per_fp, :), plasma_surf, plasma_surf, PATCH_RAD, RAD_DIM, INTERP_ORDER)

    if !wall.nowall
        # Plasma–Wall block
        compute_3D_kernel_matrices!(grad_green, view(green_temp, 1:num_points_per_fp, :), plasma_surf, wall, PATCH_RAD, RAD_DIM, INTERP_ORDER)
        # Wall–Plasma block
        compute_3D_kernel_matrices!(grad_green, view(green_temp, (num_points_per_fp+1):(2*num_points_per_fp), :), wall, plasma_surf, PATCH_RAD, RAD_DIM, INTERP_ORDER)
        # Wall–Wall block
        compute_3D_kernel_matrices!(grad_green, view(green_temp, (num_points_per_fp+1):(2*num_points_per_fp), :), wall, wall, PATCH_RAD, RAD_DIM, INTERP_ORDER)
    end

    if nfp == 1
        # nfp == 1 case: direct real solve on the built kernel matrices (equivalent to 2D)
        ldiv!(lu!(grad_green), green_temp)
        G_plasma = @view green_temp[1:num_points_per_fp, :]
        vac_data.wv .= (4π^2 / num_points_per_fp) .* (exp_mn_basis * (G_plasma * exp_mn_basis'))
    else
        # nfp > 1 case: block-circulant solve on the built kernel matrices
        # Solve one reduced (nb·M)×(nb·M) system per toroidal residue class k = mod(n, nfp)
        D̂ = zeros!(pool, ComplexF64, n_obs, n_obs)
        Ŝ = zeros!(pool, ComplexF64, n_obs, num_points_per_fp)
        for k in unique(mod.(n_modes, nfp))
            # Reduced operators D̂ₖ = Σ_d D_d ω^{k d}, Ŝₖ = Σ_d S_d ω^{k d} (fused, allocation-free).
            # D̂ keeps the nb-block source structure (plasma cols 1:M, wall cols M+1:2M); Ŝ is plasma-
            # source-only over all nb·M observer rows.
            fill!(D̂, 0)
            fill!(Ŝ, 0)
            for d in 0:(nfp-1)
                phase = cis(-2π * (k * d) / nfp)
                for s in 0:(nb-1)
                    src_cols = (s*num_points+d*num_points_per_fp+1):(s*num_points+(d+1)*num_points_per_fp)
                    dst_cols = (s*num_points_per_fp+1):((s+1)*num_points_per_fp)
                    @views @. D̂[:, dst_cols] += phase * grad_green[:, src_cols]
                end
                scols = (d*num_points_per_fp+1):((d+1)*num_points_per_fp)
                @views @. Ŝ += phase * green_temp[:, scols]
            end

            # Exterior response operator for this sector: solve D̂ₖ G = Ŝₖ in place (Ŝ ← G = D̂ₖ⁻¹Ŝₖ)
            ldiv!(lu!(D̂), Ŝ)
            mode_cols = [(idx_m + (idx_n-1)*mpert) for (idx_n, n) in enumerate(n_modes) if mod(n, nfp) == k for idx_m in 1:mpert]
            E = @view exp_mn_basis[mode_cols, :]
            G_plasma = @view Ŝ[1:num_points_per_fp, :]   # plasma-observer rows of the combined exterior operator
            vac_data.wv[mode_cols, mode_cols] .= (4π^2 / num_points_per_fp) .* (E * (G_plasma * E'))
        end
    end

    _symmetrize_vacuum_energy!(vac_data.wv)

    # Zero out the Green's function matrices (not tested in 3D yet)
    fill!(vac_data.grri, 0)
    fill!(vac_data.grre, 0)

    # Populate coordinate arrays
    vac_data.plasma_pts .= plasma_surf.r
    vac_data.wall_pts .= wall.r
end

"""
    compute_vacuum_response(inputs::VacuumInput, wall_settings::WallShapeSettings) -> VacuumResponse

Compute the vacuum response for the given inputs. Allocating wrapper around
[`compute_vacuum_response!`](@ref); pass a preallocated [`VacuumResponse`](@ref) to that method
instead when reusing storage across calls.
"""
function compute_vacuum_response(inputs::VacuumInput, wall_settings::WallShapeSettings)
    vac = VacuumResponse(inputs)
    compute_vacuum_response!(vac, inputs, wall_settings)
    return vac
end

"""
    compute_vacuum_response!(vac_data::VacuumResponse, inputs::VacuumInput, wall_settings::WallShapeSettings)

In-place variant that populates the arrays of an existing [`VacuumResponse`](@ref). Dispatches on
dimensionality only: 2D (`inputs.nzeta == 1`) routes to [`_compute_vacuum_response_2d!`], 3D to
[`_compute_vacuum_response_3d!`].
"""
function compute_vacuum_response!(vac_data::VacuumResponse, inputs::VacuumInput, wall_settings::WallShapeSettings)
    if inputs.nzeta == 1
        _compute_vacuum_response_2d!(vac_data, inputs, wall_settings)
    else
        _compute_vacuum_response_3d!(vac_data, inputs, wall_settings)
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
