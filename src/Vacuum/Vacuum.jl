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
include("Symmetry3D.jl")
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
            # grre = -(2π)²χ^(vo), the vacuum-outside potential. Overwrites the block to save memory.
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
    _compute_vacuum_response_3d!(vac_data::VacuumResponse, inputs::VacuumInput, wall_settings::WallShapeSettings; compute_Iv=false, use_symmetry=true)

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

When both surfaces are stellarator symmetric the class operator is additionally transformed by the
[`StellaratorBasis`](@ref) for that class, which makes it real and — for a self-conjugate class —
splits it into two blocks of roughly half the size. That halves the operator memory and the kernel
work, and cuts the factorization ~2.6-2.8×. Pass `use_symmetry=false` to force the untransformed
solve; a boundary that fails the symmetry test falls through to it automatically.

With `compute_Iv=true` each class also solves the interior operator `D_int = D_ext - 2I`, as in 2D.
That shift is block-diagonal in the field-period index and invariant under the basis change, so both
reductions stay exact.
"""
@with_pool pool function _compute_vacuum_response_3d!(
    vac_data::VacuumResponse,
    inputs::VacuumInput,
    wall_settings::WallShapeSettings;
    compute_Iv::Bool=false,
    use_symmetry::Bool=true
)

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

    # Stellarator symmetry halves both the operator and the kernel work when the surfaces allow it
    σ = use_symmetry ? stellarator_involution(plasma_surf, wall, nfp) : nothing
    classes = unique(mod.(n_modes, nfp))
    bases = [σ === nothing ? nothing : StellaratorBasis(σ, mtheta, k, nfp) for k in classes]
    block_sizes = [b === nothing ? [num_points_per_fp] : b.block_size for b in bases]

    # One flat buffer per operator, carved into this class's blocks each pass. Sized for the widest
    # class so the pool does not grow with the number of classes.
    T = σ === nothing ? (nfp == 1 ? Float64 : ComplexF64) : Float64
    grad_len = maximum(sum((nb * sz)^2 for sz in szs) for szs in block_sizes)
    green_len = maximum(sum(nb * sz * sz for sz in szs) for szs in block_sizes)
    grad_buffer = zeros!(pool, T, grad_len)
    green_buffer = zeros!(pool, T, green_len)
    interior_buffer = compute_Iv ? zeros!(pool, T, grad_len) : zeros!(pool, T, 0)
    grre = zeros!(pool, ComplexF64, n_obs, mpert * length(n_modes))
    grri = compute_Iv ? similar!(pool, grre) : zeros!(pool, ComplexF64, 0, 0)
    basis_buffer = zeros!(pool, ComplexF64, mpert * length(n_modes), σ === nothing ? 0 : num_points_per_fp)

    # Carve a flat buffer into this class's blocks; a reshaped contiguous view stays strided, so the
    # factorizations and matrix products below stay on the BLAS path.
    function carve(buffer, szs, ncols)
        offsets = cumsum([0; [nb * sz * ncols(sz) for sz in szs]])
        return [reshape(view(buffer, (offsets[i]+1):offsets[i+1]), nb * szs[i], ncols(szs[i])) for i in eachindex(szs)]
    end

    # Loop over all decoupled toroidal residue classes
    for (idx_k, k) in enumerate(classes)
        cols = [(idx_m + (idx_n-1)*mpert) for (idx_n, n) in enumerate(n_modes) if mod(n, nfp) == k for idx_m in 1:mpert]
        # A contiguous class (always so for nfp == 1) keeps the basis and output views strided, and so on the BLAS path
        mode_cols = length(cols) == cols[end] - cols[1] + 1 ? (cols[1]:cols[end]) : cols
        # Diagonal block of wv (and I_v when requested)
        wv_block = @view vac_data.wv[mode_cols, mode_cols]
        E = @view exp_mn_basis[mode_cols, :]
        sym = bases[idx_k]
        szs = block_sizes[idx_k]

        # Phases are real whenever ω^k = ±1, which keeps the whole class on the real BLAS path
        phases = if (σ === nothing ? nfp == 1 : mod(2k, nfp) == 0)
            sgn = mod(2k, nfp) == 0 && isodd(2k ÷ nfp) ? -1.0 : 1.0
            Float64[sgn^d for d in 0:(nfp-1)]
        else
            ComplexF64[cis(-2π * (k * d) / nfp) for d in 0:(nfp-1)]
        end

        grad_blocks = carve(grad_buffer, szs, sz -> nb * sz)
        green_blocks = carve(green_buffer, szs, sz -> sz)

        # Plasma–Plasma block
        compute_3D_kernel_matrices!(grad_blocks, green_blocks, plasma_surf, plasma_surf, PATCH_RAD, RAD_DIM, INTERP_ORDER, phases, sym)

        if !wall.nowall
            # Plasma–Wall block
            compute_3D_kernel_matrices!(grad_blocks, green_blocks, plasma_surf, wall, PATCH_RAD, RAD_DIM, INTERP_ORDER, phases, sym)
            # Wall–Plasma block
            compute_3D_kernel_matrices!(grad_blocks, green_blocks, wall, plasma_surf, PATCH_RAD, RAD_DIM, INTERP_ORDER, phases, sym)
            # Wall–Wall block
            compute_3D_kernel_matrices!(grad_blocks, green_blocks, wall, wall, PATCH_RAD, RAD_DIM, INTERP_ORDER, phases, sym)
        end

        # Mode basis in the symmetry-adapted basis; Ẽ = E·U carries the transform through the solve
        mode_basis = if sym === nothing
            [E]
        else
            mb = [view(basis_buffer, 1:length(mode_cols), (sum(szs[1:(i-1)])+1):sum(szs[1:i])) for i in eachindex(szs)]
            transform_mode_basis!(mb, E, sym)
            mb
        end

        interior_blocks = compute_Iv ? carve(interior_buffer, szs, sz -> nb * sz) : grad_blocks

        row_offset = 0
        for (b, sz) in enumerate(szs)
            nrow = nb * sz
            rows = (row_offset+1):(row_offset+nrow)
            Ẽ = mode_basis[b]
            grre_k = @view grre[rows, 1:length(mode_cols)]

            # The mode basis acts on columns and D⁻¹ on rows, so (D⁻¹S)Eᴴ == D⁻¹(SEᴴ): projecting the
            # RHS before the solve is exact and carries this class's modes instead of one column per point.
            mul!(grre_k, green_blocks[b], Ẽ')

            if compute_Iv
                # Copy RHS before exterior solve overwrites grre; keep a kernel copy for interior
                grri_k = @view grri[rows, 1:length(mode_cols)]
                grri_k .= grre_k
                interior_blocks[b] .= grad_blocks[b]

                # Exterior operator D_ext = 2I + 𝒦 (Chance 1997 eq. 89); the solve gives
                # grre = -(2π)²χ^(vo), the vacuum-outside potential. Overwrites the block to save memory.
                ldiv!(lu!(grad_blocks[b]), grre_k)

                # Interior operator D_int = D_ext - 2I: the double-layer jump between the two one-sided
                # boundary limits is 2I here, giving the vacuum-inside potential grri = χ^(vi).
                for i in 1:nrow
                    interior_blocks[b][i, i] -= 2.0
                end
                ldiv!(lu!(interior_blocks[b]), grri_k)

                # μ₀Iᵛ = χ^(vi) - χ^(vo) (Park 2007 eq. 21b), accumulated over the parity blocks
                g_diff = @view grri_k[1:sz, :]
                g_diff .= @view(grre_k[1:sz, :]) .- g_diff
                mul!(@view(vac_data.I_v[mode_cols, mode_cols]), Ẽ, g_diff, 1, 1)
            else
                # Only need exterior system for wv
                ldiv!(lu!(grad_blocks[b]), grre_k)
            end

            # Project exterior kernel onto observer basis exp(-i(mθ-nζ)), summed over the blocks
            mul!(wv_block, Ẽ, @view(grre_k[1:sz, :]), 1, 1)
            row_offset += nrow
        end
        wv_block .*= 4π^2 / num_points_per_fp

        if compute_Iv
            # Flip θ_VAC → -θ_VAC to get I^v in GPEC's CCW-θ frame, and normalize
            Iv_block = @view vac_data.I_v[mode_cols, mode_cols]
            conj!(Iv_block)
            Iv_block ./= num_points_per_fp
        end
    end

    # Remove any non-Hermitian residual from Hermitian matrices due to discretization
    _warn_and_symmetrize!(vac_data.wv, "Wᵛ")
    compute_Iv && _warn_and_symmetrize!(vac_data.I_v, "Iᵛ")

    # Populate coordinate arrays
    vac_data.plasma_pts .= plasma_surf.r
    vac_data.wall_pts .= wall.r
end

"""
    compute_vacuum_response(inputs::VacuumInput, wall_settings::WallShapeSettings; compute_Iv=false, use_symmetry=true) -> VacuumResponse

Compute the vacuum response for the given inputs. Allocating wrapper around
[`compute_vacuum_response!`](@ref); pass a preallocated [`VacuumResponse`](@ref) to that method
instead when reusing storage across calls. Pass `compute_Iv=true` to additionally populate the
surface-current matrix `I_v`, and `use_symmetry=false` to skip the 3D stellarator-symmetry solve.
"""
function compute_vacuum_response(inputs::VacuumInput, wall_settings::WallShapeSettings; compute_Iv::Bool=false, use_symmetry::Bool=true)
    vac = VacuumResponse(inputs)
    compute_vacuum_response!(vac, inputs, wall_settings; compute_Iv, use_symmetry)
    return vac
end

"""
    compute_vacuum_response!(vac_data::VacuumResponse, inputs::VacuumInput, wall_settings::WallShapeSettings; compute_Iv=false, use_symmetry=true)

In-place variant that populates the arrays of an existing [`VacuumResponse`](@ref). Dispatches on
dimensionality only: 2D (`inputs.nzeta == 1`) routes to [`_compute_vacuum_response_2d!`], 3D to
[`_compute_vacuum_response_3d!`]. `use_symmetry` applies to the 3D path only.
"""
function compute_vacuum_response!(vac_data::VacuumResponse, inputs::VacuumInput, wall_settings::WallShapeSettings; compute_Iv::Bool=false, use_symmetry::Bool=true)
    if inputs.nzeta == 1
        _compute_vacuum_response_2d!(vac_data, inputs, wall_settings; compute_Iv)
    else
        _compute_vacuum_response_3d!(vac_data, inputs, wall_settings; compute_Iv, use_symmetry)
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
