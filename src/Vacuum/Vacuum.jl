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

            # Surface-current matrix, Park 2007 eq. 21b: μ₀I^v = χ^(vi) - χ^(vo) = grri - grre
            # They are flipped because VACUUM builds the operators in its CW-θ frame while GPEC
            # uses CCW-θ, flipping the outward-normal sign.
            g_diff = @view grri[1:num_points_surf, :]
            g_diff .= @view(grre[1:num_points_surf, :]) .- g_diff
            I_v_block = @view vac_data.I_v[block_idx, block_idx]
            mul!(I_v_block, ft.basis, g_diff)
            conj!(I_v_block) # Flip θ_VAC → -θ_VAC to get I^v in GPEC's CCW-θ frame.
            I_v_block ./= num_points_surf
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
    _unsplit!(dest, split)

Recombine a real `[Re Im]` column pair into the complex `dest`.
"""
function _unsplit!(dest::AbstractMatrix{<:Complex}, split::AbstractMatrix{Float64})
    mc = size(split, 2) ÷ 2
    @views dest .= complex.(split[:, 1:mc], split[:, (mc+1):(2mc)])
    return dest
end

"""
    _solve_projected_rhs!(grre, grri, Ẽ, op, scratch)

Form `S Ẽᴴ` and solve `D \\ ·` into `grre` (exterior) and optionally `grri` (interior).
`op = (; green, lu_ext, lu_int)` with `lu_int === nothing` when `I_v` is not needed.
`scratch = (; basis_real, rhs_real, rhs_real_int)` holds the `[Re Im]` workspace for a real operator.
A real `green` uses that split because BLAS has no mixed real/complex product.
"""
function _solve_projected_rhs!(
    grre::AbstractMatrix{<:Complex},
    grri::AbstractMatrix{<:Complex},
    Ẽ::AbstractMatrix{<:Complex},
    op::NamedTuple,
    scratch::NamedTuple
)
    (; green, lu_ext, lu_int) = op
    ncols, sz = size(Ẽ)
    nrow = size(grre, 1)
    compute_Iv = lu_int !== nothing

    if eltype(green) === Float64
        (; basis_real, rhs_real, rhs_real_int) = scratch
        Er = @view basis_real[1:sz, 1:(2ncols)]
        @views Er[:, 1:ncols] .= real.(Ẽ')
        @views Er[:, (ncols+1):(2ncols)] .= imag.(Ẽ')
        grre_real = @view rhs_real[1:nrow, 1:(2ncols)]
        mul!(grre_real, green, Er)
        grri_real = @view rhs_real_int[1:nrow, 1:(2ncols)]
        compute_Iv && (grri_real .= grre_real)
        ldiv!(lu_ext, grre_real)
        _unsplit!(grre, grre_real)
        if compute_Iv
            ldiv!(lu_int, grri_real)
            _unsplit!(grri, grri_real)
        end
    else
        mul!(grre, green, Ẽ')
        compute_Iv && (grri .= grre)
        ldiv!(lu_ext, grre)
        compute_Iv && ldiv!(lu_int, grri)
    end
end

"""
    _compute_vacuum_response_3d!(vac_data, inputs, wall_settings; compute_Iv=false)

3D (`inputs.nzeta > 1`) vacuum response via block-circulant field-period reduction.

The `nfp`-periodic boundary makes the layer operators block-circulant in the field-period index, so
the problem block-diagonalizes by toroidal residue class `k = mod(n, nfp)`:

    D̂ₖ = Σ_d D_d ω^{k d},   Ŝₖ = Σ_d S_d ω^{k d},   ω = exp(-2πi/nfp),

with `D_d`, `S_d` coupling observers in field period 0 to sources in period `d`. Each class is one
solve `wv[k] = (4π²/M)·Ẽᴴ·(D̂ₖ \\ Ŝₖ)|_plasma·Ẽ`, the 3D analogue of the 2D loop over decoupled `n`.
The phase sum is folded into the kernel write, so only the reduced operator is stored.

Real `D_d`, `S_d` give `D̂₋ₖ = conj(D̂ₖ)`, so class `mod(nfp - k, nfp)` reuses this factorization with
a conjugated mode basis. A stellarator-symmetric boundary gets a [`StellSymBasis`](@ref), making the
class operator real and splitting a self-conjugate class into two half-size blocks.
"""
@with_pool pool function _compute_vacuum_response_3d!(vac_data::VacuumResponse, inputs::VacuumInput, wall_settings::WallShapeSettings; compute_Iv::Bool=false)

    (; mtheta, nzeta, nfp, m_modes, n_modes) = inputs
    fill!(vac_data.wv, 0)
    fill!(vac_data.I_v, 0)

    # Full-torus geometry for sources; observers are one field period
    full = expand_field_periods(inputs)
    plasma_surf = PlasmaGeometry3D(full)
    wall = WallGeometry3D(full, plasma_surf, wall_settings)

    num_points_per_fp = mtheta * nzeta
    mpert = length(m_modes)
    nb = wall.nowall ? 1 : 2  # plasma, or [plasma; wall]
    n_obs = nb * num_points_per_fp
    num_modes = mpert * length(n_modes)

    # exp(-i(mθ − nζ)) on one field period
    exp_mn_basis = compute_fourier_coefficients(mtheta, m_modes, nzeta * nfp, n_modes; nfp=nfp)

    # 23×23 patch, 20 radial / 40 angular polar nodes, 5-point Lagrange. Malhotra JCP 397 (2019) 108791 sec 3.2.2
    # grows these as N^(1/4); over our grids that spans M = 19..35 with no measured accuracy gain, so they are fixed.
    PATCH_RAD = 11
    RAD_DIM = 20
    INTERP_ORDER = 5

    # Shared within-period reflection when both surfaces allow it; `nothing` leaves the operator untransformed
    σ_map = stell_sym_map(plasma_surf, wall, nfp)
    
    # Find all mode families that share one vacuum operator D̂ₖ, and group them by conjugate pairs
    classes = unique(mod.(n_modes, nfp))
    groups = get_conjugate_groups(classes, nfp)

    # Real under stellarator symmetry, or when every class is self-conjugate (ω^k = ±1)
    T = (σ_map !== nothing || all(k -> is_self_conjugate(k, nfp), classes)) ? Float64 : ComplexF64

    # grre/grri are O(n_obs · num_modes), small beside the O(n_obs²) operator, so both are allocated unconditionally
    grre = zeros!(pool, ComplexF64, n_obs, num_modes)
    grri = zeros!(pool, ComplexF64, n_obs, num_modes)
    # Holds this class's mode basis, symmetry-transformed and/or conjugated as the class requires
    basis_buffer = zeros!(pool, ComplexF64, num_modes, num_points_per_fp)
    # Dense per-class accumulators for the wv / I_v projections
    wv_acc = zeros!(pool, ComplexF64, num_modes, num_modes)
    Iv_acc = zeros!(pool, ComplexF64, num_modes, num_modes)
    # Split [Re Im] scratch; zero-sized (unallocated) when the operator is complex
    rhs_real = zeros!(pool, Float64, T === Float64 ? n_obs : 0, 2 * num_modes)
    rhs_real_int = zeros!(pool, Float64, T === Float64 ? n_obs : 0, 2 * num_modes)
    basis_real = zeros!(pool, Float64, T === Float64 ? num_points_per_fp : 0, 2 * num_modes)
    real_scratch = (; basis_real, rhs_real, rhs_real_int)

    for group in groups
        # This class's operator is the only large allocation; rewind so peak memory stays at one class
        checkpoint!(pool)
        k = classes[group[1]]
        sym_basis = σ_map === nothing ? nothing : StellSymBasis(σ_map, mtheta, k, nfp)
        sizes = sym_basis === nothing ? [num_points_per_fp] : sym_basis.block_sizes
        offsets = cumsum([0; sizes])
        block_cols = [(offsets[i]+1):offsets[i+1] for i in eachindex(sizes)]

        # Field-period phases ω^{k d}. Real ±1 for a self-conjugate class, which keeps it off the complex path.
        ωkd = ComplexF64[cis(-2π * (k * d) / nfp) for d in 0:(nfp-1)]
        phases = is_self_conjugate(k, nfp) ? round.(real.(ωkd)) : ωkd

        # `sym_basis` makes each reduced block real even when `phases` are complex
        grad_blocks = [zeros!(pool, T, nb * sz, nb * sz) for sz in sizes]
        green_blocks = [zeros!(pool, T, nb * sz, sz) for sz in sizes]

        # Plasma–Plasma block
        compute_3D_kernel_matrices!(grad_blocks, green_blocks, plasma_surf, plasma_surf, PATCH_RAD, RAD_DIM, INTERP_ORDER, phases, sym_basis)

        if !wall.nowall
            # Plasma–Wall block
            compute_3D_kernel_matrices!(grad_blocks, green_blocks, plasma_surf, wall, PATCH_RAD, RAD_DIM, INTERP_ORDER, phases, sym_basis)
            # Wall–Wall block
            compute_3D_kernel_matrices!(grad_blocks, green_blocks, wall, wall, PATCH_RAD, RAD_DIM, INTERP_ORDER, phases, sym_basis)
            # Wall–Plasma block
            compute_3D_kernel_matrices!(grad_blocks, green_blocks, wall, plasma_surf, PATCH_RAD, RAD_DIM, INTERP_ORDER, phases, sym_basis)
        end

        # Factor once per group (conjugate partner reuses it)
        # D_ext = 2I + 𝒦 (Chance 1997 eq. 89); lu! overwrites, so copy D_int = D_ext - 2I first
        lu_int = if compute_Iv
            map(grad_blocks) do g
                interior = similar!(pool, g)
                interior .= g
                for i in axes(interior, 1)
                    interior[i, i] -= 2
                end
                lu!(interior)
            end
        else
            LU{T,Matrix{T},Vector{Int}}[]
        end
        lu_ext = [lu!(g) for g in grad_blocks]

        for (member, idx_class) in enumerate(group)
            # Conjugate class: conj(D̂ₖ)x = conj(Ŝₖ)Eᴴ ⇔ D̂ₖ acts on conj(E); conjugate the result back.
            partner = member > 1
            # Columns of wv for this class; collapse to a range when contiguous for cheaper indexing
            n_idxs = findall(n -> mod(n, nfp) == classes[idx_class], n_modes)
            cols = [idx_m + (idx_n - 1) * mpert for idx_n in n_idxs for idx_m in 1:mpert]
            mode_cols = length(cols) == cols[end] - cols[1] + 1 ? (cols[1]:cols[end]) : cols
            E = @view exp_mn_basis[mode_cols, :]

            # E_out = conj?(E)·U carries the change of basis through the solve. U is unitary, so the
            # right-hand side, the wv projection and I_v are all unchanged by it.
            E_out = [view(basis_buffer, 1:length(mode_cols), r) for r in block_cols]
            transform_mode_basis!(E_out, E, sym_basis; conjugate=partner)

            # Dense scratch: a non-contiguous class would make the vac_data view non-strided and drop these projections off BLAS.
            ncols = length(mode_cols)
            wv_block = @view wv_acc[1:ncols, 1:ncols]
            Iv_block = @view Iv_acc[1:ncols, 1:ncols]
            fill!(wv_block, 0)
            compute_Iv && fill!(Iv_block, 0)

            row_offset = 0
            for (b, sz) in enumerate(sizes)
                nrow = nb * sz
                rows = (row_offset+1):(row_offset+nrow)
                row_offset += nrow
                Ẽ = E_out[b]
                grre_k = @view grre[rows, 1:ncols]
                grri_k = @view grri[rows, 1:ncols]

                # Project S onto Ẽ before the solve: (D⁻¹S)Ẽᴴ = D⁻¹(SẼᴴ), so the RHS is this class's modes, not one column per point
                op = (; green=green_blocks[b], lu_ext=lu_ext[b], lu_int=compute_Iv ? lu_int[b] : nothing)
                _solve_projected_rhs!(grre_k, grri_k, Ẽ, op, real_scratch)

                if compute_Iv
                    # μ₀Iᵛ = χ^(vi) - χ^(vo) (Park 2007 eq. 21b), accumulated over the blocks
                    g_diff = @view grri_k[1:sz, :]
                    g_diff .= @view(grre_k[1:sz, :]) .- g_diff
                    mul!(Iv_block, Ẽ, g_diff, 1, 1)
                end

                # Project the exterior kernel onto the observer basis exp(-i(mθ-nζ)), summed over the blocks
                mul!(wv_block, Ẽ, @view(grre_k[1:sz, :]), 1, 1)
            end

            wv_block .*= 4π^2 / num_points_per_fp
            # Partner solved the conjugated system: conjugate wv back
            partner && conj!(wv_block)
            @views vac_data.wv[mode_cols, mode_cols] .= wv_block

            if compute_Iv
                # The θ_VAC → -θ_VAC flip on I_v cancels that conjugation for a partner, so only the representative conj!s I_v.
                partner || conj!(Iv_block)
                Iv_block ./= num_points_per_fp
                @views vac_data.I_v[mode_cols, mode_cols] .= Iv_block
            end
        end
        rewind!(pool)
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

Allocating version of [`compute_vacuum_response!`](@ref).
"""
function compute_vacuum_response(inputs::VacuumInput, wall_settings::WallShapeSettings; compute_Iv::Bool=false)
    vac = VacuumResponse(inputs)
    compute_vacuum_response!(vac, inputs, wall_settings; compute_Iv)
    return vac
end

"""
    compute_vacuum_response!(vac_data::VacuumResponse, inputs::VacuumInput, wall_settings::WallShapeSettings; compute_Iv=false)

Compute the vacuum response into `vac_data`. 2D (`inputs.nzeta == 1`) and 3D otherwise. Pass
`compute_Iv=true` to also fill the surface-current matrix `I_v`.
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
