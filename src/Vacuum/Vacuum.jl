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

            # μ₀Iᵛ = χ^(vi) - χ^(vo) (Park 2007 eq. 21b). Overwrites the plasma-observer rows of grri.
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
    _conjugate_groups(classes, nfp, enabled) -> Vector{Vector{Int}}

Group each class with its conjugate `mod(nfp - k, nfp)` when `enabled` and that class is present.
Self-conjugate classes (`mod(2k, nfp) == 0`) never pair; with `enabled = false` nothing pairs.
"""
function _conjugate_groups(classes::AbstractVector{<:Integer}, nfp::Integer, enabled::Bool)
    enabled || return [[i] for i in eachindex(classes)]
    groups = Vector{Int}[]
    taken = falses(length(classes))
    for (i, k) in enumerate(classes)
        taken[i] && continue
        taken[i] = true
        j = mod(2k, nfp) == 0 ? nothing : findfirst(==(mod(nfp - k, nfp)), classes)
        if j === nothing || taken[j]
            push!(groups, [i])
        else
            taken[j] = true
            push!(groups, [i, j])
        end
    end
    return groups
end

"""
    _split_project!(dest, S, Ẽ, work) -> AbstractMatrix{Float64}

Real `[Re Im]` form of `S·Ẽ'` for a real block `S`, written into `dest`. BLAS has no mixed
real/complex product or triangular solve, so the two halves go through the solve independently and
[`_unsplit!`](@ref) recombines them.
"""
function _split_project!(dest::AbstractMatrix{Float64}, S::AbstractMatrix{Float64}, Ẽ::AbstractMatrix{<:Complex}, work::AbstractMatrix{Float64})
    mc, sz = size(Ẽ)
    adj = Ẽ'
    Er = @view work[1:sz, 1:(2mc)]
    @views Er[:, 1:mc] .= real.(adj)
    @views Er[:, (mc+1):(2mc)] .= imag.(adj)
    out = @view dest[1:size(S, 1), 1:(2mc)]
    mul!(out, S, Er)
    return out
end

"""
    _unsplit!(dest, split)

Recombine the `[Re Im]` column pair of `split` into the complex `dest`.
"""
function _unsplit!(dest::AbstractMatrix{<:Complex}, split::AbstractMatrix{Float64})
    mc = size(split, 2) ÷ 2
    @views dest .= complex.(split[:, 1:mc], split[:, (mc+1):(2mc)])
    return dest
end

"""
    _class_mode_columns(n_modes, nfp, k, mpert)

Columns of `wv` for residue class `k`; a range when contiguous, which keeps views of it strided.
"""
function _class_mode_columns(n_modes::AbstractVector{<:Integer}, nfp::Integer, k::Integer, mpert::Integer)
    cols = [idx_m + (idx_n - 1) * mpert for (idx_n, n) in enumerate(n_modes) if mod(n, nfp) == k for idx_m in 1:mpert]
    return length(cols) == cols[end] - cols[1] + 1 ? (cols[1]:cols[end]) : cols
end

"""
    _solve_residue_class!(vac_data, mode_cols, mode_basis, op, work, nb, num_points, compute_Iv, partner)

Solve one residue class against the already-factored `op`, accumulating its diagonal blocks of `wv`
and `I_v`. `mode_basis` is this class's Fourier basis per operator block, already carrying the
symmetry transform and, for a conjugate `partner`, the conjugation.
"""
function _solve_residue_class!(
    vac_data::VacuumResponse,
    mode_cols,
    mode_basis::AbstractVector{<:AbstractMatrix},
    op::NamedTuple,
    work::NamedTuple,
    nb::Int,
    num_points::Int,
    compute_Iv::Bool,
    partner::Bool
)
    # Accumulate into dense scratch rather than straight into `vac_data`: a class whose modes are not
    # contiguous gives a non-strided view, which drops the projections below off BLAS entirely.
    ncols = length(mode_cols)
    wv_block = @view work.wv_acc[1:ncols, 1:ncols]
    Iv_block = @view work.Iv_acc[1:ncols, 1:ncols]
    fill!(wv_block, 0)
    compute_Iv && fill!(Iv_block, 0)

    row_offset = 0
    for (b, sz) in enumerate(op.sizes)
        nrow = nb * sz
        rows = (row_offset+1):(row_offset+nrow)
        row_offset += nrow
        Ẽ = mode_basis[b]
        grre_k = @view work.grre[rows, 1:ncols]
        grri_k = @view work.grri[rows, 1:ncols]

        # The mode basis acts on columns and D⁻¹ on rows, so (D⁻¹S)Ẽᴴ == D⁻¹(SẼᴴ): projecting the RHS
        # before the solve is exact and carries this class's modes instead of one column per point.
        # The exterior solve gives grre = -(2π)²χ^(vo), the vacuum-outside potential.
        if eltype(op.green[b]) === Float64
            ext = _split_project!(work.rhs_real, op.green[b], Ẽ, work.basis_real)
            int = @view work.rhs_real_int[1:nrow, 1:size(ext, 2)]
            compute_Iv && (int .= ext)
            ldiv!(op.lu_ext[b], ext)
            _unsplit!(grre_k, ext)
            if compute_Iv
                ldiv!(op.lu_int[b], int)
                _unsplit!(grri_k, int)
            end
        else
            mul!(grre_k, op.green[b], Ẽ')
            compute_Iv && (grri_k .= grre_k)
            ldiv!(op.lu_ext[b], grre_k)
            compute_Iv && ldiv!(op.lu_int[b], grri_k)
        end

        if compute_Iv
            # μ₀Iᵛ = χ^(vi) - χ^(vo) (Park 2007 eq. 21b), accumulated over the blocks
            g_diff = @view grri_k[1:sz, :]
            g_diff .= @view(grre_k[1:sz, :]) .- g_diff
            mul!(Iv_block, Ẽ, g_diff, 1, 1)
        end

        # Project the exterior kernel onto the observer basis exp(-i(mθ-nζ)), summed over the blocks
        mul!(wv_block, Ẽ, @view(grre_k[1:sz, :]), 1, 1)
    end

    wv_block .*= 4π^2 / num_points
    # A partner solved the conjugated system, so its block conjugates back here
    partner && conj!(wv_block)
    @views vac_data.wv[mode_cols, mode_cols] .= wv_block

    if compute_Iv
        # Flip θ_VAC → -θ_VAC to get I^v in GPEC's CCW-θ frame, and normalize. For a partner that
        # flip and the conjugation of its result cancel.
        partner || conj!(Iv_block)
        Iv_block ./= num_points
        @views vac_data.I_v[mode_cols, mode_cols] .= Iv_block
    end
end

"""
    _compute_vacuum_response_3d!(vac_data, inputs, wall_settings; compute_Iv=false, use_symmetry=true, use_conjugate_pairing=true)

3D (`inputs.nzeta > 1`) vacuum response via block-circulant field-period reduction.

The `nfp`-periodic boundary makes the layer operators block-circulant in the field-period index, so
the problem block-diagonalizes by toroidal residue class `k = mod(n, nfp)`:

    D̂ₖ = Σ_d D_d ω^{k d},   Ŝₖ = Σ_d S_d ω^{k d},   ω = exp(-2πi/nfp),

with `D_d`, `S_d` coupling observers in field period 0 to sources in period `d`. Each class needs one
solve `wv[k] = (4π²/M)·Ẽᴴ·(D̂ₖ \\ Ŝₖ)|_plasma·Ẽ`, so this loops over classes exactly as the 2D routine
loops over decoupled `n`. The phase sum is folded into the kernel write, so only the reduced operator
is stored.

Two optional reductions apply on top. `use_symmetry` gives a stellarator-symmetric boundary a
[`StellaratorBasis`](@ref), making the class operator real and splitting a self-conjugate class into
two half-size blocks. `use_conjugate_pairing` exploits real `D_d`, `S_d`, so `D̂₋ₖ = conj(D̂ₖ)` and
class `mod(nfp - k, nfp)` reuses this class's factorization with a conjugated mode basis.
"""
@with_pool pool function _compute_vacuum_response_3d!(
    vac_data::VacuumResponse,
    inputs::VacuumInput,
    wall_settings::WallShapeSettings;
    compute_Iv::Bool=false,
    use_symmetry::Bool=true,
    use_conjugate_pairing::Bool=true
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
    num_modes = mpert * length(n_modes)

    # Complex Fourier basis exp(-i(mθ-nζ)) on a single field period
    exp_mn_basis = compute_fourier_coefficients(mtheta, m_modes, nzeta * nfp, n_modes; nfp=nfp)

    # Singular quadrature: 23x23 patch, 20 radial / 40 angular polar nodes, 5-point Lagrange.
    # Malhotra JCP 397 (2019) 108791 sec 3.2.2 grows these as N^(1/4); over our grids,
    # that spans M = 19..35 with no measured accuracy gain, so they are fixed.
    PATCH_RAD = 11
    RAD_DIM = 20
    INTERP_ORDER = 5

    # One symmetry basis per class when both surfaces allow it; `nothing` falls through untransformed
    mirror = use_symmetry ? stellarator_mirror(plasma_surf, wall, nfp) : nothing
    classes = unique(mod.(n_modes, nfp))
    bases = [mirror === nothing ? nothing : StellaratorBasis(mirror, mtheta, k, nfp) for k in classes]
    block_sizes = [b === nothing ? [num_points_per_fp] : b.block_size for b in bases]

    # Group each class with its conjugate; the representative is the only one assembled and factored
    groups = _conjugate_groups(classes, nfp, use_conjugate_pairing)

    # The operator is real under the symmetry basis, and also when every class is self-conjugate
    # (`ω^k = ±1`, so the field-period phases are real). That halves its storage and cuts the
    # factorization ~3.5x, at the cost of carrying the complex right-hand side as an [Re Im] pair.
    T = (mirror !== nothing || all(k -> mod(2k, nfp) == 0, classes)) ? Float64 : ComplexF64

    # `grre`/`grri` hold the exterior and interior solves; both are O(n_obs · num_modes), small beside
    # the O(n_obs²) operator, so they are allocated unconditionally to keep the solve branch-free.
    grre = zeros!(pool, ComplexF64, n_obs, num_modes)
    grri = zeros!(pool, ComplexF64, n_obs, num_modes)
    # Holds this class's mode basis, symmetry-transformed and/or conjugated as the class requires
    basis_buffer = zeros!(pool, ComplexF64, num_modes, num_points_per_fp)
    # Scratch for a real operator's split right-hand side, each the size of one `grre`
    # Dense per-class accumulators for the wv / I_v projections (see `_solve_residue_class!`)
    wv_acc = zeros!(pool, ComplexF64, num_modes, num_modes)
    Iv_acc = zeros!(pool, ComplexF64, num_modes, num_modes)
    rhs_real = zeros!(pool, Float64, T === Float64 ? n_obs : 0, 2 * num_modes)
    rhs_real_int = zeros!(pool, Float64, T === Float64 ? n_obs : 0, 2 * num_modes)
    basis_real = zeros!(pool, Float64, T === Float64 ? num_points_per_fp : 0, 2 * num_modes)
    work = (; grre, grri, rhs_real, rhs_real_int, basis_real, wv_acc, Iv_acc)

    # Loop over the representatives of the conjugate-paired residue classes
    for group in groups
        # This class's operator is the only large allocation; rewind at the end of the pass so the
        # next class reuses the same memory and peak usage stays at one class.
        checkpoint!(pool)
        idx_rep = group[1]
        k = classes[idx_rep]
        sym = bases[idx_rep]
        szs = block_sizes[idx_rep]
        offsets = cumsum([0; szs])
        block_ranges = [(offsets[i]+1):offsets[i+1] for i in eachindex(szs)]

        # Field-period phases ω^{k d}. Exactly ±1 when 2k ≡ 0 (mod nfp), and taking them real there
        # keeps a self-conjugate class off the complex path.
        ω = ComplexF64[cis(-2π * (k * d) / nfp) for d in 0:(nfp-1)]
        phases = mod(2k, nfp) == 0 ? round.(real.(ω)) : ω

        grad_blocks = [zeros!(pool, T, nb * sz, nb * sz) for sz in szs]
        green_blocks = [zeros!(pool, T, nb * sz, sz) for sz in szs]

        # Plasma–Plasma block
        compute_3D_kernel_matrices!(grad_blocks, green_blocks, plasma_surf, plasma_surf, PATCH_RAD, RAD_DIM, INTERP_ORDER, phases, sym)

        if !wall.nowall
            # Plasma–Wall block
            compute_3D_kernel_matrices!(grad_blocks, green_blocks, plasma_surf, wall, PATCH_RAD, RAD_DIM, INTERP_ORDER, phases, sym)
            # Wall–Wall block
            compute_3D_kernel_matrices!(grad_blocks, green_blocks, wall, wall, PATCH_RAD, RAD_DIM, INTERP_ORDER, phases, sym)
            # Wall–Plasma block
            compute_3D_kernel_matrices!(grad_blocks, green_blocks, wall, plasma_surf, PATCH_RAD, RAD_DIM, INTERP_ORDER, phases, sym)
        end

        # Factor once per group; the conjugate partner reuses these factorizations. The exterior
        # operator is D_ext = 2I + 𝒦 (Chance 1997 eq. 89). Factoring overwrites the blocks, so the
        # interior operator D_int = D_ext - 2I is copied off first (2D comment explains the shift).
        lu_int = if compute_Iv
            interior = [zeros!(pool, T, nb * sz, nb * sz) for sz in szs]
            for (b, sz) in enumerate(szs)
                interior[b] .= grad_blocks[b]
                for i in 1:(nb*sz)
                    interior[b][i, i] -= 2
                end
            end
            [lu!(g) for g in interior]
        else
            LU{T,Matrix{T},Vector{Int}}[]
        end
        lu_ext = [lu!(g) for g in grad_blocks]
        op = (; green=green_blocks, lu_ext, lu_int, sizes=szs)

        for (member, idx_class) in enumerate(group)
            # The conjugate class solves conj(D̂ₖ)x = conj(Ŝₖ)Eᴴ; conjugating that identity turns it
            # into the representative's operator acting on conj(E), with the result conjugated back.
            partner = member > 1
            mode_cols = _class_mode_columns(n_modes, nfp, classes[idx_class], mpert)
            E = @view exp_mn_basis[mode_cols, :]

            # Ẽ = conj?(E)·U carries the change of basis through the solve. U is unitary, so the
            # right-hand side, the wv projection and I_v are all unchanged by it.
            mode_basis = [view(basis_buffer, 1:length(mode_cols), r) for r in block_ranges]
            if sym === nothing
                mode_basis[1] .= partner ? conj.(E) : E
            else
                transform_mode_basis!(mode_basis, E, sym; conjugate=partner)
            end

            _solve_residue_class!(vac_data, mode_cols, mode_basis, op, work, nb, num_points_per_fp, compute_Iv, partner)
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
    compute_vacuum_response(inputs::VacuumInput, wall_settings::WallShapeSettings; compute_Iv=false, use_symmetry=true, use_conjugate_pairing=true) -> VacuumResponse

Compute the vacuum response for the given inputs. Allocating wrapper around
[`compute_vacuum_response!`](@ref); pass a preallocated [`VacuumResponse`](@ref) to that method
instead when reusing storage across calls. Pass `compute_Iv=true` to additionally populate the
surface-current matrix `I_v`, `use_symmetry=false` to skip the 3D stellarator-symmetry solve, and
`use_conjugate_pairing=false` to solve conjugate residue classes independently.
"""
function compute_vacuum_response(inputs::VacuumInput, wall_settings::WallShapeSettings; compute_Iv::Bool=false, use_symmetry::Bool=true, use_conjugate_pairing::Bool=true)
    vac = VacuumResponse(inputs)
    compute_vacuum_response!(vac, inputs, wall_settings; compute_Iv, use_symmetry, use_conjugate_pairing)
    return vac
end

"""
    compute_vacuum_response!(vac_data::VacuumResponse, inputs::VacuumInput, wall_settings::WallShapeSettings; compute_Iv=false, use_symmetry=true, use_conjugate_pairing=true)

In-place variant that populates the arrays of an existing [`VacuumResponse`](@ref). Dispatches on
dimensionality only: 2D (`inputs.nzeta == 1`) routes to [`_compute_vacuum_response_2d!`], 3D to
[`_compute_vacuum_response_3d!`]. `use_symmetry` and `use_conjugate_pairing` apply to the 3D path only.
"""
function compute_vacuum_response!(
    vac_data::VacuumResponse,
    inputs::VacuumInput,
    wall_settings::WallShapeSettings;
    compute_Iv::Bool=false,
    use_symmetry::Bool=true,
    use_conjugate_pairing::Bool=true
)
    if inputs.nzeta == 1
        _compute_vacuum_response_2d!(vac_data, inputs, wall_settings; compute_Iv)
    else
        _compute_vacuum_response_3d!(vac_data, inputs, wall_settings; compute_Iv, use_symmetry, use_conjugate_pairing)
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
