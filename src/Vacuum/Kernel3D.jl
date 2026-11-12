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
#   - one sparse interpolation matrix, applied forward to the geometry and transposed to scatter weights
#   - VACUUM kernel convention: 1/r and n.grad'(1/r) over 2pi, unit Green's-identity diagonal
#   - far field by the trapezoidal rule on the collocation grid, without upsampling
#   - patch geometry upsampled 2x by trigonometric interpolation to refine the polar interpolation stencil,
#     correction weights folded back to the collocation nodes by its transpose (BIEST instead upsamples the
#     whole surface and density and shrinks the patch)
#   - singular correction applied to plasma-wall cross-surface blocks
#   - dense operator row assembly, field-period folding, stellarator-symmetric basis, threading
#
# SPDX-License-Identifier: Apache-2.0
# Full text in LICENSES/Apache-2.0.txt; attribution in NOTICE.

"""
Thread-local scratch for `compute_3D_kernel_matrices!`, one per thread so the parallel observer loop does not race.
`patch` and `polar` hold (r, ∂r/∂θ, ∂r/∂ζ) per node; `T` is the operator row's element type.
"""
struct KernelWorkspace{T<:Number}
    patch::Matrix{SVector{9,Float64}}
    polar::Matrix{SVector{9,Float64}}
    M_polar_single::Matrix{Float64}
    M_polar_double::Matrix{Float64}
    M_grid_flat::Vector{Float64}
    fold_tmp::Matrix{Float64}
    folded_patch::Matrix{T}
    row_single::Vector{T}
    row_double::Vector{T}
    sym_work::Vector{ComplexF64}
end

function KernelWorkspace{T}(UP_DIM::Int, RAD_DIM::Int, ANG_DIM::Int, mtheta::Int, nzeta_fp::Int, nrow::Int, nsym::Int) where {T}
    return KernelWorkspace{T}(
        zeros(SVector{9,Float64}, UP_DIM, UP_DIM),
        zeros(SVector{9,Float64}, RAD_DIM, ANG_DIM),
        zeros(RAD_DIM, ANG_DIM),
        zeros(RAD_DIM, ANG_DIM),
        zeros(UP_DIM^2),
        zeros(UP_DIM, mtheta),
        zeros(T, nzeta_fp, mtheta),
        zeros(T, nrow),
        zeros(T, nrow),
        zeros(ComplexF64, nsym)
    )
end

"""
    SingularQuadratureData

Precomputed polar singular-correction quadrature (Malhotra JCP 397 (2019) 108791, Sec. 3.2).

## Fields

  - `Gpou::Matrix{Float64}`: Partition of unity on the Cartesian patch, `Gpou = -χ` (PATCH_DIM × PATCH_DIM)
  - `Ppou::Matrix{Float64}`: Partition of unity times polar quadrature weights (RAD_DIM × ANG_DIM)
  - `P2G::SparseMatrixCSC{Float64,Int}`: Lagrange interpolation matrix on the 2× upsampled patch: `polar = P2G' * patch`, `grid = P2G * polar`
  - `PATCH_DIM::Int`: Patch width `2 * PATCH_RAD + 1`
  - `UP_DIM::Int`: Upsampled patch width `2 * PATCH_DIM - 1`
  - `PATCH_RAD::Int`: Number of points on each side of the source point treated as singular
  - `ANG_DIM::Int`: Number of angular quadrature points
  - `RAD_DIM::Int`: Number of radial quadrature points
  - `INTERP_ORDER::Int`: Lagrange interpolation order
"""
struct SingularQuadratureData
    Gpou::Matrix{Float64}
    Ppou::Matrix{Float64}
    P2G::SparseMatrixCSC{Float64,Int}
    PATCH_DIM::Int
    UP_DIM::Int
    PATCH_RAD::Int
    ANG_DIM::Int
    RAD_DIM::Int
    INTERP_ORDER::Int
end

"""
    SingularQuadratureData(PATCH_RAD, RAD_DIM, INTERP_ORDER)

Build the polar quadrature, partitions of unity, and Lagrange interpolant for the singular patch.
`ANG_DIM = 2 * RAD_DIM`. `INTERP_ORDER` must be `≤ 4 * PATCH_RAD + 1`, the upsampled patch width.

# Arguments

  - `PATCH_RAD::Int`: Number of points adjacent to source point to treat as singular
  - `RAD_DIM::Int`: Radial quadrature order
  - `INTERP_ORDER::Int`: Lagrange interpolation order

# Returns

  - `SingularQuadratureData`: Precomputed quadrature data
"""
function SingularQuadratureData(PATCH_RAD::Int, RAD_DIM::Int, INTERP_ORDER::Int)

    # Total size of square patch extracted around singular point (odd number: 2*PATCH_RAD+1)
    PATCH_DIM = 2 * PATCH_RAD + 1
    # We upsample the patch by 2x in each direction for better accuracy
    UP_DIM = 2 * PATCH_DIM - 1
    @assert INTERP_ORDER <= UP_DIM "Must have INTERP_ORDER <= UP_DIM, got INTERP_ORDER=$INTERP_ORDER, UP_DIM=$UP_DIM"
    # Number of angular quadrature nodes in polar coordinates (uniformly distributed around circle)
    ANG_DIM = 2 * RAD_DIM

    # Setup radial quadrature
    qx_raw, qw_raw = gausslegendre(RAD_DIM) # points on [-1,1]
    qx = (qx_raw .+ 1) ./ 2  # [-1, 1] → [0, 1]
    qw = qw_raw ./ 2         # Adjust weights for interval change

    # χ(r) = exp(-36 r^p), p from PATCH_DIM: χ blends on the collocation grid, so it keys on the coarse patch, not UP_DIM
    pou_power = PATCH_DIM > 45 ? 10 : (PATCH_DIM > 20 ? 8 : 6)
    pou(r) = r ≥ 1.0 ? 0.0 : exp(-36.0 * r^pou_power)

    # Gpou = -χ on the Cartesian patch
    Gpou = zeros(PATCH_DIM, PATCH_DIM)
    coords = LinRange(-1.0, 1.0, PATCH_DIM)
    for (i, x) in enumerate(coords), (j, y) in enumerate(coords)
        Gpou[i, j] = -pou(sqrt(x^2 + y^2))
    end

    # Ppou = χ(ρ) M²/4 r dr dθ [Malhotra JCP 397 (2019) 108791 eq. 38]
    Ppou = zeros(RAD_DIM, ANG_DIM)
    dθ = 2π / ANG_DIM
    for j in 1:ANG_DIM, i in 1:RAD_DIM
        dr = qw[i] * PATCH_RAD
        rdθ = qx[i] * PATCH_RAD * dθ
        Ppou[i, j] = pou(qx[i]) * dr * rdθ
    end

    # Spacing between Lagrange interpolation nodes in [0,1] for INTERP_ORDER-point stencil
    h = 1.0 / (INTERP_ORDER - 1)

    # Compute 2D tensor-product Lagrange basis function at (x0, x1) in local
    # stencil coordinates for basis node (i0, i1) on uniform grid with spacing h
    @inline function lagrange_interp(x0::Float64, x1::Float64, i0::Int, i1::Int)
        Lx = Ly = 1.0
        ξ0 = x0 / h
        ξ1 = x1 / h
        for j0 in 0:(INTERP_ORDER-1)
            j0 != i0 && (Lx *= (ξ0 - j0) / (i0 - j0))
        end
        for j1 in 0:(INTERP_ORDER-1)
            j1 != i1 && (Ly *= (ξ1 - j1) / (i1 - j1))
        end
        return Lx * Ly
    end

    # grid = P2G * polar, polar = P2G' * grid; each column is the INTERP_ORDER² Lagrange stencil
    Ngrid = UP_DIM * UP_DIM
    Npolar = RAD_DIM * ANG_DIM

    # Preallocate COO storage:
    #   I_coo[k], J_coo[k] = (row, column) index of kth nonzero
    #   V_coo[k]           = interpolation weight
    nnz_per_polar = INTERP_ORDER^2
    I_coo = Vector{Int}(undef, Npolar * nnz_per_polar)
    J_coo = Vector{Int}(undef, Npolar * nnz_per_polar)
    V_coo = Vector{Float64}(undef, Npolar * nnz_per_polar)

    idx = 1
    for ir in 1:RAD_DIM, ia in 1:ANG_DIM
        # Map polar node to unit square: x0, x1 ∈ [0,1] × [0,1]
        x0 = 0.5 + 0.5 * qx[ir] * cos(dθ * (ia - 1))
        x1 = 0.5 + 0.5 * qx[ir] * sin(dθ * (ia - 1))

        # Lower-left corner indices of INTERP_ORDER × INTERP_ORDER stencil centered on (x0,x1).
        # Round, don't truncate, and keep INTERP_ORDER odd: the stencil must be equivariant under the patch's π-rotation or stellarator symmetry is lost.
        y0 = clamp(round(Int, x0 * (UP_DIM - 1)) - (INTERP_ORDER - 1) ÷ 2, 0, UP_DIM - INTERP_ORDER)
        y1 = clamp(round(Int, x1 * (UP_DIM - 1)) - (INTERP_ORDER - 1) ÷ 2, 0, UP_DIM - INTERP_ORDER)

        # Local coordinates within INTERP_ORDER×INTERP_ORDER stencil, normalized to [0,1]
        z0 = (x0 * (UP_DIM - 1) - y0) * h
        z1 = (x1 * (UP_DIM - 1) - y1) * h

        # Polar point index (column in P2G)
        j_polar = ir + RAD_DIM * (ia - 1)

        # Populate stencil contributions for this polar node
        for i0 in 1:INTERP_ORDER, i1 in 1:INTERP_ORDER
            # Grid point index (row in P2G), using column-major layout
            i_grid = (y0 + i0) + UP_DIM * (y1 + i1 - 1)
            I_coo[idx] = i_grid
            J_coo[idx] = j_polar
            V_coo[idx] = lagrange_interp(z0, z1, i0 - 1, i1 - 1)
            idx += 1
        end
    end

    # Assemble sparse interpolation matrix
    P2G = sparse(I_coo, J_coo, V_coo, Ngrid, Npolar)

    return SingularQuadratureData(Gpou, Ppou, P2G, PATCH_DIM, UP_DIM, PATCH_RAD, ANG_DIM, RAD_DIM, INTERP_ORDER)
end

# Cached singular quadrature; rebuilt if PATCH_RAD / RAD_DIM / INTERP_ORDER change
const SINGULAR_QUAD_CACHE = Ref{Union{Nothing,SingularQuadratureData}}(nothing)
const _SINGULAR_QUAD_LOCK = ReentrantLock()

"""
    get_singular_quadrature(PATCH_RAD, RAD_DIM, INTERP_ORDER)

Return the cached [`SingularQuadratureData`](@ref), rebuilding it if the parameters changed. Takes a lock,
so concurrent solves on different grids each get the data for their own parameters.
"""
function get_singular_quadrature(PATCH_RAD::Int, RAD_DIM::Int, INTERP_ORDER::Int)
    @lock _SINGULAR_QUAD_LOCK begin
        cached = SINGULAR_QUAD_CACHE[]
        if !isnothing(cached) &&
           cached.PATCH_RAD == PATCH_RAD &&
           cached.RAD_DIM == RAD_DIM &&
           cached.INTERP_ORDER == INTERP_ORDER
            return cached
        end
        data = SingularQuadratureData(PATCH_RAD, RAD_DIM, INTERP_ORDER)
        SINGULAR_QUAD_CACHE[] = data
        return data
    end
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
Interpolate the patch onto the polar nodes, `polar = P2G' * patch`. Each node carries all nine fields as one `SVector`,
so `P2G` is swept once rather than once per field as `mul!` would.
"""
function interpolate_to_polar!(polar::Matrix{SVector{9,Float64}}, patch::Matrix{SVector{9,Float64}}, P2G::SparseMatrixCSC{Float64,Int})
    rv, nz = rowvals(P2G), nonzeros(P2G)
    @inbounds for c in axes(P2G, 2)
        # Accumulates (r, ∂r/∂θ, ∂r/∂ζ) at polar node c, no allocations
        acc = zero(SVector{9,Float64})
        for p in nzrange(P2G, c)
            # Lagrange weight × all nine fields at that patch node
            acc += nz[p] * patch[rv[p]]
        end
        polar[c] = acc
    end
end

"""
Add the polar correction weights `M_polar` to the operator row: `P2G` scatters them onto the upsampled patch, and
`Uζ_patch * M_up' * Uθ_patch` (the transposed trigonometric upsampling about the first node, field periods folded into
`Uζ_patch`) onto the collocation nodes, shifted to the observer `(i_obs, j_obs)` by translation invariance.
"""
function scatter_polar_to_row!(
    row::Vector{T},
    M_polar::Matrix{Float64},
    P2G::SparseMatrixCSC{Float64,Int},
    Uθ_patch::Matrix{Float64},
    Uζ_patch::Matrix{T},
    ws::KernelWorkspace{T},
    i_obs::Int,
    j_obs::Int,
    wrap_phase::T
) where {T<:Number}

    (; M_grid_flat, fold_tmp, folded_patch) = ws
    UP_DIM = size(Uθ_patch, 1)

    # Scatter the polar corrections back to the upsampled patch
    mul!(M_grid_flat, P2G, vec(M_polar))

    # Fold the upsampled patch back to the collocation nodes
    mul!(fold_tmp, transpose(reshape(M_grid_flat, UP_DIM, UP_DIM)), Uθ_patch)
    mul!(folded_patch, Uζ_patch, fold_tmp)

    nzeta_fp, mtheta = size(folded_patch)
    @inbounds for j in 1:nzeta_fp
        # Columns the shift pushes into the next field period fold back with one more phase step;
        # this assumes geometric phases, phases[d] = wrap_phase^(d-1)
        j_fold = j - j_obs + 1
        phase = j_fold < 1 ? wrap_phase : one(T)
        j_fold < 1 && (j_fold += nzeta_fp)
        for i in 1:mtheta
            row[i+mtheta*(j-1)] += phase * folded_patch[j_fold, periodic_wrap(i - i_obs + 1, mtheta)]
        end
    end
end

"""
    compute_3D_kernel_matrices!(grad_blocks, green_blocks, observer, source, PATCH_RAD, RAD_DIM, INTERP_ORDER, phases, sym_basis=nothing)

3D single- and double-layer kernels with Malhotra's polar singular correction (PPCF 62 (2020) 024004).

Far field: trapezoidal rule. Near field: polar quadrature blended by a partition of unity, on source data
upsampled 2× by trigonometric interpolation.

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
    num_points = observer.mtheta * observer.nzeta
    @assert num_points % length(phases) == 0 "full-torus grid ($num_points points) is not divisible by nfp ($(length(phases)))"
    num_points_per_fp = num_points ÷ length(phases) # observer/source points in one field period
    dθdζ = 4π^2 / num_points

    # Surface sub-block of the operator this call fills
    col_index = (source isa PlasmaGeometry3D ? 1 : 2)
    row_index = (observer isa PlasmaGeometry3D ? 1 : 2)

    # 𝒢ⁿ only needed for plasma as source term (RHS of eqs. 26/27 in Chance 1997)
    populate_greenfunction = source isa PlasmaGeometry3D

    # Each reflection pair's second row follows from its first, so only the representatives are evaluated
    observers = sym_basis === nothing ? (1:num_points_per_fp) : sym_basis.pair_reps
    sym_basis === nothing || @assert isodd(INTERP_ORDER) "stellarator symmetry needs an odd INTERP_ORDER; an even stencil is not π-rotation equivariant"

    # Clamp the patch on coarse grids, with a warning.
    # Bound once into a new name: reassigning PATCH_RAD inside the branch would box it in the threaded closure below.
    patch_rad = min(PATCH_RAD, (min(source.mtheta, source.nzeta) - 1) ÷ 2)
    if patch_rad < PATCH_RAD
        @warn "PATCH_RAD=$(PATCH_RAD) is greater than half the number of points in the toroidal or poloidal direction, which is not supported. Setting PATCH_RAD to $(patch_rad), be sure to check if outputs are converged. This can be avoided by setting mtheta and nzeta to be greater than $(2 * PATCH_RAD + 1)."
    end

    quad_data = get_singular_quadrature(patch_rad, RAD_DIM, INTERP_ORDER)
    (; PATCH_DIM, UP_DIM, ANG_DIM, Ppou, Gpou, P2G) = quad_data

    # Source r, ∂r/∂θ, ∂r/∂ζ on the 2× upsampled grid, where node 2i-1 is collocation node i
    Uθ, Uζ = trig_upsample_matrix(source.mtheta), trig_upsample_matrix(source.nzeta)
    src_up = SVector{9}.(eachrow(trig_upsample(hcat(source.r, source.dr_dθ, source.dr_dζ), Uθ, Uζ, source.mtheta, source.nzeta)))

    # Upsampling restricted to the patch about the first node, with the field periods folded onto period 0 in ζ (see `scatter_polar_to_row!`)
    Trow = eltype(phases)
    nzeta_fp = source.nzeta ÷ length(phases)
    Uθ_patch = Uθ[periodic_wrap.((1:UP_DIM) .- 2 * patch_rad, 2 * source.mtheta), :]
    Uζ_patch = zeros(Trow, nzeta_fp, UP_DIM)
    for i in 1:UP_DIM, d in eachindex(phases), j in 1:nzeta_fp
        Uζ_patch[j, i] += phases[d] * Uζ[periodic_wrap(i - 2 * patch_rad, 2 * source.nzeta), j+(d-1)*nzeta_fp]
    end
    wrap_phase = phases[min(2, end)]

    # One operator row is accumulated per observer, so the field-period fold and the basis change both happen before anything is stored.
    # `store_kernel_row!` needs two rows of scratch: the reconstructed partner and their combination.
    nsym = sym_basis === nothing ? 0 : 2 * num_points_per_fp
    workspaces = [KernelWorkspace{Trow}(UP_DIM, RAD_DIM, ANG_DIM, source.mtheta, nzeta_fp, num_points_per_fp, nsym) for _ in 1:Threads.maxthreadid()]

    # :static pins each task to its thread, so the threadid()-indexed scratch below stays private
    Threads.@threads :static for i_pair in eachindex(observers)
        ws = workspaces[Threads.threadid()]
        (; patch, polar, M_polar_single, M_polar_double, row_single, row_double, sym_work) = ws
        fill!(row_double, 0)
        populate_greenfunction && fill!(row_single, 0)

        # Convert linear index to 2D indices
        idx_obs = observers[i_pair]
        i_obs = mod1(idx_obs, observer.mtheta)
        j_obs = (idx_obs - 1) ÷ observer.mtheta + 1
        r_obs = @view observer.r[idx_obs, :]

        # Far field: trapezoidal rule over all sources (the kernels return zero at r_src = r_obs)
        @inbounds for idx_src in 1:num_points
            # Evaluate kernels at grid points
            r_src = @view source.r[idx_src, :]
            n_src = @view source.normal[idx_src, :]
            far_single, far_double = laplace_kernel(r_obs[1], r_obs[2], r_obs[3], r_src[1], r_src[2], r_src[3], n_src[1], n_src[2], n_src[3])

            # Periodic trapezoidal rule (constant weights); fold this source's field period onto period 0
            d, idx_col = fldmod1(idx_src, num_points_per_fp)
            populate_greenfunction && (row_single[idx_col] += phases[d] * (far_single * dθdζ))
            row_double[idx_col] += phases[d] * (far_double * dθdζ)
        end

        # Near field: polar quadrature with singular correction, on the upsampled patch about the observer
        extract_patch!(patch, src_up, 2 * i_obs - 1, 2 * j_obs - 1, 2 * source.mtheta, 2 * source.nzeta, UP_DIM)

        # Interpolate the geometry onto the polar nodes
        interpolate_to_polar!(polar, patch, P2G)

        # Evaluate kernels and apply quadrature weights: area element × POU, where POU contains rdrdθ already
        @inbounds for ia in 1:ANG_DIM, ir in 1:RAD_DIM
            q = polar[ir, ia] # = (r, ∂r/∂θ, ∂r/∂ζ) at this polar node
            # n = ∂r/∂θ × ∂r/∂ζ from the interpolated tangents, need to reapply orientation
            n = source.normal_orient * cross(q[SVector(4, 5, 6)], q[SVector(7, 8, 9)])
            near_single, near_double = laplace_kernel(r_obs[1], r_obs[2], r_obs[3], q[1], q[2], q[3], n[1], n[2], n[3])

            M_polar_single[ir, ia] = near_single * Ppou[ir, ia] * dθdζ
            M_polar_double[ir, ia] = near_double * Ppou[ir, ia] * dθdζ
        end

        # Scatter the singular correction on the polar nodes back to the collocation nodes
        scatter_polar_to_row!(row_double, M_polar_double, P2G, Uθ_patch, Uζ_patch, ws, i_obs, j_obs, wrap_phase)
        populate_greenfunction && scatter_polar_to_row!(row_single, M_polar_single, P2G, Uθ_patch, Uζ_patch, ws, i_obs, j_obs, wrap_phase)

        # Remove the trapezoidal far field where the polar correction applies: Gpou = -χ, net weight is (1 - χ)
        @inbounds for j in 1:PATCH_DIM, i in 1:PATCH_DIM
            # Map back to global indices
            idx_pol = periodic_wrap(i_obs - patch_rad + i - 1, source.mtheta)
            idx_tor = periodic_wrap(j_obs - patch_rad + j - 1, source.nzeta)
            idx_src = idx_pol + source.mtheta * (idx_tor - 1)
            d, idx_col = fldmod1(idx_src, num_points_per_fp)

            # Far field at this patch node (weighted by Gpou = −χ)
            r_src = @view source.r[idx_src, :]
            n_src = @view source.normal[idx_src, :]
            far_single, far_double = laplace_kernel(r_obs[1], r_obs[2], r_obs[3], r_src[1], r_src[2], r_src[3], n_src[1], n_src[2], n_src[3])

            populate_greenfunction && (row_single[idx_col] += phases[d] * (far_single * Gpou[i, j] * dθdζ))
            row_double[idx_col] += phases[d] * (far_double * Gpou[i, j] * dθdζ)
        end

        # Normalize so the Green's-identity term below is a unit shift and the exterior/interior jump is 2I, as in the 2D kernel
        row_double ./= 2π
        populate_greenfunction && (row_single ./= 2π)

        store_kernel_row!(grad_blocks, sym_basis, row_double, sym_work, i_pair, row_index, col_index)
        populate_greenfunction && store_kernel_row!(green_blocks, sym_basis, row_single, sym_work, i_pair, row_index, 1)
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
