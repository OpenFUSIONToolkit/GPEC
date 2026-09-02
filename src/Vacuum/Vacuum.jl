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
    _conjugate_groups(classes, nfp, enabled) -> Vector{Vector{Int}}

Partition indices into `classes` so that each group is one class plus, when `enabled`, the conjugate
class `mod(nfp - k, nfp)` if it is also present. Self-conjugate classes (`mod(2k, nfp) == 0`) never
pair. With `enabled = false` every class is its own group.
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
    _as_eltype(R, v)

View the `Float64` storage `v` as a vector of element type `R`: the identity for `Float64`, a strided
reinterpret for `ComplexF64`. Both remain `StridedArray`, so a matrix reshaped from either stays on
the BLAS path.
"""
_as_eltype(::Type{Float64}, v::AbstractVector{Float64}) = v
_as_eltype(::Type{ComplexF64}, v::AbstractVector{Float64}) = reinterpret(ComplexF64, v)

"""
    _split_project!(dest, S, Ẽ, basis_work) -> AbstractMatrix{Float64}

Real form of `S·Ẽ'` for a real operator block `S` and a complex mode basis `Ẽ`, returned as the
`[Re Im]` column pair written into `dest`.

BLAS offers no mixed real/complex `gemm` or triangular solve, and the generic fallback for the solve
is ~30x slower than a real one of twice the width. Because `S` and its factorization are real, the
real and imaginary parts propagate through the solve independently; [`_unsplit!`](@ref) recombines
them afterwards.
"""
function _split_project!(dest::AbstractMatrix{Float64}, S::AbstractMatrix{Float64}, Ẽ::AbstractMatrix{<:Complex}, basis_work::AbstractMatrix{Float64})
    mc, sz = size(Ẽ)
    adj = Ẽ'
    Er = @view basis_work[1:sz, 1:(2mc)]
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
    _compute_vacuum_response_3d!(vac_data::VacuumResponse, inputs::VacuumInput, wall_settings::WallShapeSettings; compute_Iv=false, use_symmetry=true, use_conjugate_pairing=true)

3D (`inputs.nzeta > 1`) vacuum response via block-circulant field-period reduction.

The `nfp`-periodic boundary makes the single-/double-layer operators `S`, `D` block-circulant in the
field-period index, so the problem block-diagonalizes by toroidal residue class `k = mod(n, nfp)`:
modes with different `k` do not couple, and within a class

    D̂ₖ = Σ_d D_d ω^{k d},   Ŝₖ = Σ_d S_d ω^{k d},   ω = exp(-2πi/nfp),

with `D_d`, `S_d` the blocks coupling observers in field period 0 to sources in period `d`. Each class
needs one solve `wv[class k] = (4π²/M)·E_localᴴ·(D̂ₖ \\ Ŝₖ)|_plasma·E_local` (`E` the complex Fourier
basis, `M = mtheta·nzeta`), so the routine loops over classes exactly as the 2D routine loops over
decoupled `n`. The phase sum is folded into the kernel write, so only the reduced `[nb·M × nb·M]`
operator is ever stored.

The operator element type is decided per class rather than per call. A self-conjugate class
(`mod(2k, nfp) == 0`, i.e. `k = 0` and, for even `nfp`, `k = nfp/2`) has `ω^k = ±1`, so its phases and
its blocks are real; every other class is complex. Both are carved from the same `Float64` backing
store — the complex ones through a strided reinterpret — so the peak allocation is byte-for-byte that
of a complex buffer when any class is complex, and half of it when every class is real (`nfp <= 2`, or
any stellarator-symmetric run). A real class factorizes ~3.5x faster and, because BLAS has no mixed
real/complex triangular solve, carries its complex right-hand side through the solve as a real
`[Re Im]` pair (see [`_split_project!`](@ref)).

The field-period blocks `D_d`, `S_d` are real for any boundary, so `D̂₋ₖ = conj(D̂ₖ)` and
`Ŝ₋ₖ = conj(Ŝₖ)`: a class and its conjugate `mod(nfp - k, nfp)` share one assembly and one
factorization, the partner differing only by conjugating its Fourier basis and its output block.
Pass `use_conjugate_pairing=false` to solve every class independently. Self-conjugate classes
(`mod(2k, nfp) == 0`, which is every class when `nfp <= 2`) are unaffected.

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

    # Group each class with its conjugate; the representative is the only one assembled and factored
    groups = _conjugate_groups(classes, nfp, use_conjugate_pairing)
    has_pairs = any(g -> length(g) > 1, groups)

    # The operator element type is a per-class property. A self-conjugate class has ω^k = ±1, so its
    # field-period phases and hence its blocks are real; the symmetry-adapted basis makes every class
    # real. A real class halves the operator storage and cuts the factorization ~3.5x, so it is worth
    # keeping off the complex path even though its neighbours in the loop are complex.
    real_class(k) = σ !== nothing || mod(2k, nfp) == 0
    reps = [classes[g[1]] for g in groups]
    any_complex = any(k -> !real_class(k), reps)
    any_real = any(k -> real_class(k), reps)

    # One flat Float64 buffer per operator, carved into this class's blocks each pass and
    # reinterpreted as complex for the classes that need it. Sizing the backing store in Float64
    # rather than in the operator type keeps the peak allocation byte-for-byte identical to a complex
    # buffer when any class is complex, and halves it when every class is real (nfp ≤ 2, or any
    # stellarator-symmetric run), so the mixed-type solve below costs no memory.
    grad_len = maximum(sum((nb * sz)^2 for sz in szs) for szs in block_sizes)
    green_len = maximum(sum(nb * sz * sz for sz in szs) for szs in block_sizes)
    stride_T = any_complex ? 2 : 1
    grad_buffer = zeros!(pool, Float64, stride_T * grad_len)
    green_buffer = zeros!(pool, Float64, stride_T * green_len)
    interior_buffer = compute_Iv ? zeros!(pool, Float64, stride_T * grad_len) : zeros!(pool, Float64, 0)
    grre = zeros!(pool, ComplexF64, n_obs, mpert * length(n_modes))
    grri = compute_Iv ? similar!(pool, grre) : zeros!(pool, ComplexF64, 0, 0)
    basis_buffer = zeros!(pool, ComplexF64, mpert * length(n_modes), (σ === nothing && !has_pairs) ? 0 : num_points_per_fp)

    # Scratch for the real classes' split right-hand side (see `_split_project!`). Each is the size of
    # one `grre`, negligible beside the O(N_p²) operator it lets us keep real.
    rhs_cols = mpert * length(n_modes)
    rhs_real = any_real ? zeros!(pool, Float64, n_obs, 2 * rhs_cols) : zeros!(pool, Float64, 0, 0)
    rhs_real_int = (any_real && compute_Iv) ? similar!(pool, rhs_real) : zeros!(pool, Float64, 0, 0)
    basis_real = any_real ? zeros!(pool, Float64, num_points_per_fp, 2 * rhs_cols) : zeros!(pool, Float64, 0, 0)

    # Carve a flat buffer into this class's blocks. A reshaped contiguous view stays strided, and so
    # does a complex reinterpret of one, so the factorizations and matrix products below stay on the
    # BLAS path either way.
    function carve(buffer, szs, ncols, ::Type{R}) where {R}
        w = R === ComplexF64 ? 2 : 1
        offsets = cumsum([0; [nb * sz * ncols(sz) for sz in szs]])
        return [reshape(_as_eltype(R, view(buffer, (w*offsets[i]+1):(w*offsets[i+1]))), nb * szs[i], ncols(szs[i])) for i in eachindex(szs)]
    end

    # Loop over the representatives of the conjugate-paired residue classes
    for group in groups
        idx_k = group[1]
        k = classes[idx_k]
        sym = bases[idx_k]
        szs = block_sizes[idx_k]

        # Phases are real whenever ω^k = ±1, i.e. for a self-conjugate class, which keeps the whole
        # class on the real BLAS path. Under the symmetry-adapted basis the operator is real for every
        # class, so a complex phase there is still emitted into a real block by `emit_symmetric_row!`.
        self_conj = mod(2k, nfp) == 0
        phases = if self_conj
            sgn = isodd(2k ÷ nfp) ? -1.0 : 1.0
            Float64[sgn^d for d in 0:(nfp-1)]
        else
            ComplexF64[cis(-2π * (k * d) / nfp) for d in 0:(nfp-1)]
        end

        Top = real_class(k) ? Float64 : ComplexF64
        grad_blocks = carve(grad_buffer, szs, sz -> nb * sz, Top)
        green_blocks = carve(green_buffer, szs, sz -> sz, Top)

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

        interior_blocks = compute_Iv ? carve(interior_buffer, szs, sz -> nb * sz, Top) : grad_blocks
        if compute_Iv
            # Interior operator D_int = D_ext - 2I: the double-layer jump between the two one-sided
            # boundary limits is 2I here, giving the vacuum-inside potential. Copy before the
            # factorization below overwrites the exterior block.
            for (b, sz) in enumerate(szs)
                interior_blocks[b] .= grad_blocks[b]
                for i in 1:(nb*sz)
                    interior_blocks[b][i, i] -= 2.0
                end
            end
        end

        # Factor once per group; the conjugate class reuses these factorizations. The exterior
        # operator is D_ext = 2I + 𝒦 (Chance 1997 eq. 89). Overwrites the blocks to save memory.
        lu_ext = [lu!(g) for g in grad_blocks]
        lu_int = compute_Iv ? [lu!(g) for g in interior_blocks] : lu_ext

        for (member, idx_class) in enumerate(group)
            # The conjugate class solves conj(D̂ₖ)x = conj(Ŝₖ)Eᴴ; conjugating that identity turns it
            # into the representative's operator acting on conj(E), with the result conjugated back.
            partner = member > 1
            cols = [(idx_m + (idx_n-1)*mpert) for (idx_n, n) in enumerate(n_modes) if mod(n, nfp) == classes[idx_class] for idx_m in 1:mpert]
            # A contiguous class (always so for nfp == 1) keeps the basis and output views strided, and so on the BLAS path
            mode_cols = length(cols) == cols[end] - cols[1] + 1 ? (cols[1]:cols[end]) : cols
            # Diagonal block of wv (and I_v when requested)
            wv_block = @view vac_data.wv[mode_cols, mode_cols]
            E = @view exp_mn_basis[mode_cols, :]

            # Mode basis in the symmetry-adapted basis; Ẽ = E·U carries the transform through the solve
            mode_basis = if sym !== nothing
                mb = [view(basis_buffer, 1:length(mode_cols), (sum(szs[1:(i-1)])+1):sum(szs[1:i])) for i in eachindex(szs)]
                transform_mode_basis!(mb, E, sym; conjugate=partner)
                mb
            elseif partner
                Ec = @view basis_buffer[1:length(mode_cols), :]
                Ec .= conj.(E)
                [Ec]
            else
                [E]
            end

            row_offset = 0
            for (b, sz) in enumerate(szs)
                nrow = nb * sz
                rows = (row_offset+1):(row_offset+nrow)
                Ẽ = mode_basis[b]
                grre_k = @view grre[rows, 1:length(mode_cols)]

                # The mode basis acts on columns and D⁻¹ on rows, so (D⁻¹S)Eᴴ == D⁻¹(SEᴴ): projecting the
                # RHS before the solve is exact and carries this class's modes instead of one column per point.
                # A real class carries the projection and both solves as a real [Re Im] pair, since BLAS
                # has no mixed real/complex product or triangular solve.
                if Top === Float64
                    ext_split = _split_project!(rhs_real, green_blocks[b], Ẽ, basis_real)

                    if compute_Iv
                        grri_k = @view grri[rows, 1:length(mode_cols)]
                        int_split = @view rhs_real_int[1:nrow, 1:size(ext_split, 2)]
                        int_split .= ext_split

                        # The exterior solve gives grre = -(2π)²χ^(vo), the vacuum-outside potential
                        ldiv!(lu_ext[b], ext_split)
                        ldiv!(lu_int[b], int_split)
                        _unsplit!(grre_k, ext_split)
                        _unsplit!(grri_k, int_split)
                    else
                        ldiv!(lu_ext[b], ext_split)
                        _unsplit!(grre_k, ext_split)
                    end
                else
                    mul!(grre_k, green_blocks[b], Ẽ')

                    if compute_Iv
                        # Copy RHS before exterior solve overwrites grre; keep a kernel copy for interior
                        grri_k = @view grri[rows, 1:length(mode_cols)]
                        grri_k .= grre_k

                        # The exterior solve gives grre = -(2π)²χ^(vo), the vacuum-outside potential
                        ldiv!(lu_ext[b], grre_k)
                        ldiv!(lu_int[b], grri_k)
                    else
                        # Only need exterior system for wv
                        ldiv!(lu_ext[b], grre_k)
                    end
                end

                if compute_Iv
                    # μ₀Iᵛ = χ^(vi) - χ^(vo) (Park 2007 eq. 21b), accumulated over the parity blocks
                    grri_k = @view grri[rows, 1:length(mode_cols)]
                    g_diff = @view grri_k[1:sz, :]
                    g_diff .= @view(grre_k[1:sz, :]) .- g_diff
                    mul!(@view(vac_data.I_v[mode_cols, mode_cols]), Ẽ, g_diff, 1, 1)
                end

                # Project exterior kernel onto observer basis exp(-i(mθ-nζ)), summed over the blocks
                mul!(wv_block, Ẽ, @view(grre_k[1:sz, :]), 1, 1)
                row_offset += nrow
            end
            wv_block .*= 4π^2 / num_points_per_fp
            partner && conj!(wv_block)

            if compute_Iv
                # Flip θ_VAC → -θ_VAC to get I^v in GPEC's CCW-θ frame, and normalize. For the
                # conjugate class that flip and the conjugation of its result cancel.
                Iv_block = @view vac_data.I_v[mode_cols, mode_cols]
                partner || conj!(Iv_block)
                Iv_block ./= num_points_per_fp
            end
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
