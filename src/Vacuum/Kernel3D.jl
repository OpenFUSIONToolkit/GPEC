# Polar singular-correction quadrature for the 3D vacuum kernels
# [Malhotra et al., J. Comput. Phys. 397 (2019) 108791, Sec. 3.2].
#
# Translated to Julia from BIEST (https://github.com/dmalhotra/BIEST):
#   include/biest/singular_correction.hpp  SingularCorrection::InitPrecomp, Setup, SetPatch
#   include/biest/surface_op.txx           SurfaceOp::SetupSingularCorrection, EvalSurfInteg
# Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors
# Original implementation: Dhairya Malhotra
#
# Modifications Copyright (C) 2026 Columbia Fusion Research Center and Seoul National University:
#   - Julia translation; runtime patch, quadrature and interpolation orders, with the patch clamped on coarse grids
#   - stencil corner by rounding, so odd orders are equivariant under the patch's pi-rotation
#   - one near-field routine for every row: polar nodes interpolated and their weights scattered by per-node Lagrange
#     stencils (no stored interpolation matrix), about the observer or, off the source surface, the closest source point
#   - VACUUM kernel convention: 1/r and n.grad'(1/r) over 2pi, unit Green's-identity diagonal
#   - far field by the trapezoidal rule on the collocation grid, without upsampling
#   - patch geometry upsampled 2x by trigonometric interpolation to refine the polar interpolation stencil,
#     correction weights folded back to the collocation nodes by its transpose (BIEST instead upsamples the
#     whole surface and density and shrinks the patch)
#   - plasma-wall rows: sinh-mapped radial nodes [Johnston & Elliott, Int. J. Numer. Meth. Engng 62 (2005) 564]
#   - dense operator row assembly, field-period folding, stellarator-symmetric basis, threading
#
# SPDX-License-Identifier: Apache-2.0
# Full text in LICENSES/Apache-2.0.txt; attribution in NOTICE.

"""
Thread-local scratch for `compute_3D_kernel_matrices!`, one per thread so the parallel observer loop does not race.
`patch` holds the upsampled source patch and `M_grid` the double- and single-layer correction weights on it;
`T` is the operator row's element type.
"""
struct KernelWorkspace{T<:Number}
    patch::Matrix{SurfacePoint}
    M_grid::Array{Float64,3}
    fold_tmp::Matrix{Float64}
    folded_patch::Matrix{T}
    row_single::Vector{T}
    row_double::Vector{T}
    sym_work::Vector{ComplexF64}
end

function KernelWorkspace{T}(UP_DIM::Int, mtheta::Int, nzeta_fp::Int, nrow::Int, nsym::Int) where {T}
    return KernelWorkspace{T}(
        fill(zero(SurfacePoint), UP_DIM, UP_DIM),
        zeros(UP_DIM, UP_DIM, 2),
        zeros(UP_DIM, mtheta),
        zeros(T, nzeta_fp, mtheta),
        zeros(T, nrow),
        zeros(T, nrow),
        zeros(ComplexF64, nsym)
    )
end

"""
    laplace_kernel(ox, oy, oz, sx, sy, sz, nx, ny, nz) -> (single, double)

Fused Laplace kernels: `single = 1/r`, `double = (Δx·n)/r³`. Shares `√(r²)`; returns `(0, 0)` at coincidence (`r² < 1e-30`).
"""
@fastmath @inline function laplace_kernel(
    ox::Float64, oy::Float64, oz::Float64,
    sx::Float64, sy::Float64, sz::Float64,
    nx::Float64, ny::Float64, nz::Float64
)
    dx = ox - sx
    dy = oy - sy
    dz = oz - sz
    r2 = dx * dx + dy * dy + dz * dz
    r2 < 1e-30 && return (0.0, 0.0)
    rinv = inv(sqrt(r2))
    single = rinv
    r3inv = rinv * rinv * rinv
    double = (dx * nx + dy * ny + dz * nz) * r3inv
    return (single, double)
end

"""
Extract a periodically wrapped `PATCH_DIM × PATCH_DIM` patch centered at `(idx_pol_center, idx_tor_center)`.
"""
function extract_patch!(patch::AbstractMatrix, data::AbstractVector, idx_pol_center::Int, idx_tor_center::Int, npol::Int, ntor::Int, PATCH_DIM::Int)
    PATCH_RAD = (PATCH_DIM - 1) ÷ 2
    @inbounds for j in 1:PATCH_DIM, i in 1:PATCH_DIM
        idx_pol = periodic_wrap(idx_pol_center - PATCH_RAD + i - 1, npol)
        idx_tor = periodic_wrap(idx_tor_center - PATCH_RAD + j - 1, ntor)
        patch[i, j] = data[idx_pol+npol*(idx_tor-1)]
    end
end

"""
    SourceQuadrature

The source surface's quadrature data for `compute_3D_kernel_matrices!`, built once per call and shared by every thread
(the per-thread scratch is [`KernelWorkspace`](@ref)): the source on the 2× trigonometrically upsampled grid
(`points_up`, node `2i - 1` is collocation node `i`) and at the collocation nodes (`points`), the upsampling restricted
to the patch about the first node (see [`fold_patch_to_row!`](@ref)), and the polar rules and interpolation orders of
self (same-surface) and cross (plasma–wall) rows.
"""
struct SourceQuadrature{T<:Number,P<:AbstractVector{T},S,NS,NC,R}
    source::S
    points_up::Vector{SurfacePoint}
    points::Vector{SVector{6,Float64}}             # (r, normal) per node, packed for the far-field sum
    period_col::Vector{Tuple{Int,Int}}             # (field period, period-0 column) per node
    phases::P
    Uθ_patch::Matrix{Float64}
    Uζ_patch::Matrix{T}                            # field periods folded onto period 0
    patch_rad::Int
    ANG_DIM::Int
    self_radial::Vector{Tuple{Float64,Float64}}    # Gauss on [0, 1]
    cross_radial_rule::R                           # sinh-mapped, expanded per row by `sinh_nodes`
    self_order::Val{NS}
    cross_order::Val{NC}
    populate_greenfunction::Bool                   # fill the single layer (plasma source only)
end

function SourceQuadrature(source, phases::AbstractVector{<:Number}, patch_rad::Int, RAD_DIM::Int, INTERP_ORDER::Int)
    mtheta, nzeta, nfp = source.mtheta, source.nzeta, length(phases)
    UP_DIM, nzeta_fp = 4 * patch_rad + 1, nzeta ÷ nfp
    @assert INTERP_ORDER <= UP_DIM "Must have INTERP_ORDER <= UP_DIM, got INTERP_ORDER=$INTERP_ORDER, UP_DIM=$UP_DIM"

    Uθ, Uζ = trig_interp_matrix(mtheta, 2 * mtheta), trig_interp_matrix(nzeta, 2 * nzeta)
    up = trig_upsample(hcat(source.r, source.dr_dθ, source.dr_dζ), Uθ, Uζ, mtheta, nzeta)
    points_up = [SurfacePoint(up[k, SVector(1, 2, 3)], up[k, SVector(4, 5, 6)], up[k, SVector(7, 8, 9)]) for k in axes(up, 1)]

    Uθ_patch = Uθ[periodic_wrap.((1:UP_DIM) .- 2 * patch_rad, 2 * mtheta), :]
    Uζ_patch = zeros(eltype(phases), nzeta_fp, UP_DIM)
    for i in 1:UP_DIM, d in eachindex(phases), j in 1:nzeta_fp
        Uζ_patch[j, i] += phases[d] * Uζ[periodic_wrap(i - 2 * patch_rad, 2 * nzeta), j+(d-1)*nzeta_fp]
    end

    num_points = mtheta * nzeta
    self_radial = let (t, w) = gausslegendre(RAD_DIM)
        collect(zip((t .+ 1) ./ 2, w ./ 2))
    end
    return SourceQuadrature(
        source, points_up,
        [SVector{6}(source.r[k, 1], source.r[k, 2], source.r[k, 3], source.normal[k, 1], source.normal[k, 2], source.normal[k, 3]) for k in 1:num_points],
        [fldmod1(k, num_points ÷ nfp) for k in 1:num_points],
        phases, Uθ_patch, Uζ_patch, patch_rad, 2 * RAD_DIM,
        # Plasma–wall rows: 1–4 × RAD_DIM nodes, RAD_DIM per 4 units of the sinh variable
        self_radial, sinh_rule(RAD_DIM, 4, 4.0),
        Val(INTERP_ORDER), Val(INTERP_ORDER),
        source isa PlasmaGeometry3D
    )
end

"""
Add correction weights on the upsampled patch about collocation node `(i_obs, j_obs)` to the operator row:
`Uζ_patch * M_up' * Uθ_patch` (the transposed trigonometric upsampling about the first node, field periods folded into
`Uζ_patch`) maps them onto the collocation nodes, shifted to `(i_obs, j_obs)` by translation invariance. `j_obs` lies
in field period 0; a patch centred in another period passes that period's phase as `phase0`.
"""
function fold_patch_to_row!(row::Vector{T}, M_up::AbstractMatrix{Float64}, quad::SourceQuadrature, ws::KernelWorkspace{T},
    i_obs::Int, j_obs::Int, phase0::T) where {T<:Number}
    (; fold_tmp, folded_patch) = ws
    mul!(fold_tmp, transpose(M_up), quad.Uθ_patch)
    mul!(folded_patch, quad.Uζ_patch, fold_tmp)

    # Columns the shift pushes into the next field period fold back with one more phase step;
    # this assumes geometric phases, phases[d] = wrap_phase^(d-1)
    wrap_phase = quad.phases[min(2, end)]
    nzeta_fp, mtheta = size(folded_patch)
    @inbounds for j in 1:nzeta_fp
        j_fold = j - j_obs + 1
        phase = j_fold < 1 ? phase0 * wrap_phase : phase0
        j_fold < 1 && (j_fold += nzeta_fp)
        for i in 1:mtheta
            row[i+mtheta*(j-1)] += phase * folded_patch[j_fold, periodic_wrap(i - i_obs + 1, mtheta)]
        end
    end
end

"""
    closest_source_point!(patch, quad, r) -> (; i, j, U, V, d, hθ, hζ)

Source point closest to `r`, for a near-field correction about it. `patch` is left holding the upsampled patch about
collocation node `(i, j)`, re-centred as the Gauss–Newton search moves, and `(U, V)` are the point's coordinates on it;
`d` is the distance and `hθ`, `hζ` the local node spacings.
"""
function closest_source_point!(patch::Matrix{SurfacePoint}, quad::SourceQuadrature, r::SVector{3,Float64})
    mtheta, nzeta = quad.source.mtheta, quad.source.nzeta
    UP_DIM, val = size(patch, 1), quad.cross_order
    c = (UP_DIM - 1) / 2  # patch centre
    dθ, dζ = 2π / mtheta, 2π / nzeta

    k = argmin(k -> sum(abs2, r - quad.points[k][SVector(1, 2, 3)]), eachindex(quad.points))
    i, j = mod1(k, mtheta), (k - 1) ÷ mtheta + 1

    U, V = c, c
    for iter in 1:3
        extract_patch!(patch, quad.points_up, 2i - 1, 2j - 1, 2mtheta, 2nzeta, UP_DIM)
        U, V = closest_point(patch, r, U, V, dθ / 2, dζ / 2, val)
        su, sv = round(Int, (U - c) / 2), round(Int, (V - c) / 2)
        (su == sv == 0 || iter == 3) && break
        i, j, U, V = mod1(i + su, mtheta), mod1(j + sv, nzeta), U - 2su, V - 2sv
    end
    q = interpolate_patch(patch, U, V, val)[1]
    d, hθ, hζ = norm(r - q.r), norm(q.rθ) * dθ, norm(q.rζ) * dζ
    d < 1e-3 * min(hθ, hζ) && throw(ArgumentError("plasma and wall touch (gap $(d) m)"))
    return (; i, j, U, V, d, hθ, hζ)
end

"""
    near_correction!(ws, quad, r_obs, i, j, U, V, radial, Val(N))

Near-field correction of the row in `ws` about source point `(U, V)` of the upsampled patch about collocation node
`(i, j)`, which `ws.patch` already holds: polar quadrature on the radial nodes `radial` ([`foreach_polar_node`](@ref)),
folded onto the row, minus the trapezoidal sum it replaces.
"""
function near_correction!(ws::KernelWorkspace{T}, quad::SourceQuadrature, r_obs::SVector{3,Float64}, i::Int, j::Int,
    U::Float64, V::Float64, radial, ::Val{N}) where {T<:Number,N}
    (; patch, M_grid, row_single, row_double) = ws
    (; source, phases, patch_rad, populate_greenfunction) = quad
    UP_DIM, PATCH_DIM = size(patch, 1), 2 * patch_rad + 1
    mtheta, nzeta = source.mtheta, source.nzeta
    dθdζ = 4π^2 / (mtheta * nzeta)

    # Polar quadrature, scattered onto the upsampled patch by each node's Lagrange stencil
    fill!(M_grid, 0)
    M_double, M_single = view(M_grid,:,:,1), view(M_grid,:,:,2)
    foreach_polar_node(patch, U, V, radial, patch_rad, quad.ANG_DIM, Val(N)) do q, cu, cv, wu, wv, w
        near_single, near_double = laplace_kernel(r_obs..., q.r..., (source.normal_orient * cross(q.rθ, q.rζ))...)
        @inbounds for b in 1:N, a in 1:N
            L = w * dθdζ * wu[a] * wv[b]
            M_double[cu+a, cv+b] += near_double * L
            populate_greenfunction && (M_single[cu+a, cv+b] += near_single * L)
        end
    end
    period, j_fp = fldmod1(j, nzeta ÷ length(phases))
    fold_patch_to_row!(row_double, M_double, quad, ws, i, j_fp, phases[period])
    populate_greenfunction && fold_patch_to_row!(row_single, M_single, quad, ws, i, j_fp, phases[period])

    # Remove the trapezoidal sum under the patch: net weight (1 - χ), with χ about (U, V)
    δu, δv = (U - (UP_DIM - 1) / 2) / 2, (V - (UP_DIM - 1) / 2) / 2
    @inbounds for b in (-patch_rad-1):(patch_rad+1), a in (-patch_rad-1):(patch_rad+1)
        χ = patch_pou(hypot(a - δu, b - δv) / patch_rad, PATCH_DIM)
        χ == 0 && continue
        idx_src = periodic_wrap(i + a, mtheta) + mtheta * (periodic_wrap(j + b, nzeta) - 1)
        d_src, idx_col = quad.period_col[idx_src]
        far_single, far_double = laplace_kernel(r_obs..., quad.points[idx_src]...)
        populate_greenfunction && (row_single[idx_col] -= phases[d_src] * (far_single * χ * dθdζ))
        row_double[idx_col] -= phases[d_src] * (far_double * χ * dθdζ)
    end
end

"""
    kernel_row!(ws, quad, observer, idx_obs)

Fill `ws.row_double` (and `ws.row_single` for a plasma source) with observer `idx_obs`'s operator row: trapezoidal rule
over all sources, then the near-field correction about the observer (same surface) or about the closest source point.
"""
function kernel_row!(ws::KernelWorkspace, quad::SourceQuadrature, observer, idx_obs::Int)
    (; patch, row_single, row_double) = ws
    (; source, phases, populate_greenfunction) = quad
    dθdζ = 4π^2 / (source.mtheta * source.nzeta)
    fill!(row_double, 0)
    populate_greenfunction && fill!(row_single, 0)

    i_obs = mod1(idx_obs, observer.mtheta)
    j_obs = (idx_obs - 1) ÷ observer.mtheta + 1
    r_obs = SVector{3}(@view observer.r[idx_obs, :])

    # Far field: trapezoidal rule over all sources (the kernels return zero at r_src = r_obs); fold each source's field
    # period onto period 0
    @inbounds for idx_src in eachindex(quad.points)
        s = quad.points[idx_src]
        far_single, far_double = laplace_kernel(r_obs[1], r_obs[2], r_obs[3], s[1], s[2], s[3], s[4], s[5], s[6])
        d, idx_col = quad.period_col[idx_src]
        populate_greenfunction && (row_single[idx_col] += phases[d] * (far_single * dθdζ))
        row_double[idx_col] += phases[d] * (far_double * dθdζ)
    end

    if observer === source
        # Polar quadrature with singular correction about the observer, at the centre of its upsampled patch
        extract_patch!(patch, quad.points_up, 2 * i_obs - 1, 2 * j_obs - 1, 2 * source.mtheta, 2 * source.nzeta, size(patch, 1))
        c = 2.0 * quad.patch_rad
        near_correction!(ws, quad, r_obs, i_obs, j_obs, c, c, quad.self_radial, quad.self_order)
    else
        # About the closest source point, with radial nodes graded to the gap
        cp = closest_source_point!(patch, quad, r_obs)
        ε = cp.d / (quad.patch_rad * min(cp.hθ, cp.hζ))
        near_correction!(ws, quad, r_obs, cp.i, cp.j, cp.U, cp.V, sinh_nodes(ε, 0.0, 1.0, quad.cross_radial_rule), quad.cross_order)
    end

    # Normalize so the Green's-identity term is a unit shift and the exterior/interior jump is 2I, as in the 2D kernel
    row_double ./= 2π
    populate_greenfunction && (row_single ./= 2π)
end

"""
    compute_3D_kernel_matrices!(grad_blocks, green_blocks, observer, source, PATCH_RAD, RAD_DIM, INTERP_ORDER, phases, sym_basis=nothing)

3D single- and double-layer kernels with Malhotra's polar singular correction (PPCF 62 (2020) 024004).

Far field: trapezoidal rule. Near field ([`near_correction!`](@ref)): polar quadrature blended by a partition of unity,
on source data upsampled 2× by trigonometric interpolation, about the observer on Gauss radial nodes, or for a
plasma–wall row about the closest source point on sinh-mapped ones.

A source in field period `d` is accumulated onto the period-0 column with weight `phases[d+1]`, so
the block-circulant reduction is written in and the full-torus source blocks are never stored.
`phases = [1.0]` is the single-period (real) case.

`grad_blocks` is `∇_{x_src} φ · n_src`; `green_blocks` is `φ`, filled only when `source` is the
plasma (Chance 1997 eqs. 26–27). `sym_basis` evaluates one point per reflection pair and emits the
real reduced operator; omit it for the untransformed matrix.
"""
function compute_3D_kernel_matrices!(
    grad_blocks::AbstractVector{<:AbstractMatrix{<:Number}},
    green_blocks::AbstractVector{<:AbstractMatrix{<:Number}},
    observer::Union{PlasmaGeometry3D,WallGeometry3D},
    source::Union{PlasmaGeometry3D,WallGeometry3D},
    PATCH_RAD::Int,
    RAD_DIM::Int,
    INTERP_ORDER::Int,
    phases::AbstractVector{<:Number},
    sym_basis::Union{Nothing,StellSymBasis}=nothing
)
    @assert (observer.mtheta, observer.nzeta) == (source.mtheta, source.nzeta) "observer and source must share one grid"
    num_points = observer.mtheta * observer.nzeta
    @assert num_points % length(phases) == 0 "full-torus grid ($num_points points) is not divisible by nfp ($(length(phases)))"
    num_points_per_fp = num_points ÷ length(phases) # observer/source points in one field period

    # Surface sub-block of the operator this call fills
    col_index = (source isa PlasmaGeometry3D ? 1 : 2)
    row_index = (observer isa PlasmaGeometry3D ? 1 : 2)

    # Each reflection pair's second row follows from its first, so only the representatives are evaluated
    observers = sym_basis === nothing ? (1:num_points_per_fp) : sym_basis.pair_reps
    sym_basis === nothing || @assert isodd(INTERP_ORDER) "stellarator symmetry needs an odd INTERP_ORDER; an even stencil is not π-rotation equivariant"

    # Clamp the patch on coarse grids, with a warning
    patch_rad = min(PATCH_RAD, (min(source.mtheta, source.nzeta) - 1) ÷ 2)
    if patch_rad < PATCH_RAD
        @warn "PATCH_RAD=$(PATCH_RAD) is greater than half the number of points in the toroidal or poloidal direction, which is not supported. Setting PATCH_RAD to $(patch_rad), be sure to check if outputs are converged. This can be avoided by setting mtheta and nzeta to be greater than $(2 * PATCH_RAD + 1)."
    end
    quad = SourceQuadrature(source, phases, patch_rad, RAD_DIM, INTERP_ORDER)

    # One operator row is accumulated per observer, so the field-period fold and the basis change both happen before anything is stored.
    # `store_kernel_row!` needs two rows of scratch: the reconstructed partner and their combination.
    nsym = sym_basis === nothing ? 0 : 2 * num_points_per_fp
    workspaces = [KernelWorkspace{eltype(phases)}(4 * patch_rad + 1, source.mtheta, source.nzeta ÷ length(phases), num_points_per_fp, nsym)
                  for _ in 1:Threads.maxthreadid()]

    # :static pins each task to its thread, so the threadid()-indexed scratch below stays private
    Threads.@threads :static for i_pair in eachindex(observers)
        ws = workspaces[Threads.threadid()]
        kernel_row!(ws, quad, observer, observers[i_pair])
        store_kernel_row!(grad_blocks, sym_basis, ws.row_double, ws.sym_work, i_pair, row_index, col_index)
        quad.populate_greenfunction && store_kernel_row!(green_blocks, sym_basis, ws.row_single, ws.sym_work, i_pair, row_index, 1)
    end

    # Volume-integral term of Green's identity; invariant under the basis change, so it is the same unit shift on every block
    if typeof(source) == typeof(observer)
        for (b, H) in enumerate(grad_blocks)
            nsize = sym_basis === nothing ? num_points_per_fp : sym_basis.block_sizes[b]
            for i in 1:nsize
                H[(row_index-1)*nsize+i, (col_index-1)*nsize+i] += 1.0
            end
        end
    end
end
