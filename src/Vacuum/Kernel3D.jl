"""
    SingularQuadratureData

Precomputed polar singular-correction quadrature (BIEST / Malhotra). `P2G` maps polar samples to
the Cartesian patch (`grid = P2G * polar`, `polar = P2G' * patch`). `Gpou`/`Ppou` are the Cartesian
and polar partitions of unity; `Gpou = -χ`.
"""
struct SingularQuadratureData
    qx::Vector{Float64}
    qw::Vector{Float64}
    Gpou::Matrix{Float64}
    Ppou::Matrix{Float64}
    P2G::SparseMatrixCSC{Float64,Int}
    PATCH_DIM::Int
    PATCH_RAD::Int
    ANG_DIM::Int
    RAD_DIM::Int
    INTERP_ORDER::Int
end

"""
    SingularQuadratureData(PATCH_RAD, RAD_DIM, INTERP_ORDER)

Build the polar quadrature, partitions of unity, and Lagrange interpolant for the singular patch.
`ANG_DIM = 2 * RAD_DIM`. `INTERP_ORDER` must be `≤ 2 * PATCH_RAD + 1`.
"""
function SingularQuadratureData(PATCH_RAD::Int, RAD_DIM::Int, INTERP_ORDER::Int)

    PATCH_DIM = 2 * PATCH_RAD + 1
    @assert INTERP_ORDER <= PATCH_DIM "Must have INTERP_ORDER <= PATCH_DIM, got INTERP_ORDER=$INTERP_ORDER, PATCH_DIM=$PATCH_DIM"
    ANG_DIM = 2 * RAD_DIM

    qx_raw, qw_raw = gausslegendre(RAD_DIM)
    qx = (qx_raw .+ 1) ./ 2  # [-1, 1] → [0, 1]
    qw = qw_raw ./ 2

    # χ(r) = exp(-36 r^p), p from PATCH_DIM (BIEST)
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

    h = 1.0 / (INTERP_ORDER - 1)

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
    Ngrid = PATCH_DIM * PATCH_DIM
    Npolar = RAD_DIM * ANG_DIM
    nnz_per_polar = INTERP_ORDER^2
    I_coo = Vector{Int}(undef, Npolar * nnz_per_polar)
    J_coo = Vector{Int}(undef, Npolar * nnz_per_polar)
    V_coo = Vector{Float64}(undef, Npolar * nnz_per_polar)

    idx = 1
    for ir in 1:RAD_DIM, ia in 1:ANG_DIM
        x0 = 0.5 + 0.5 * qx[ir] * cos(dθ * (ia - 1))
        x1 = 0.5 + 0.5 * qx[ir] * sin(dθ * (ia - 1))

        # Round, don't truncate: the stencil must be equivariant under the patch's π-rotation or stellarator symmetry is lost.
        y0 = clamp(round(Int, x0 * (PATCH_DIM - 1)) - (INTERP_ORDER - 1) ÷ 2, 0, PATCH_DIM - INTERP_ORDER)
        y1 = clamp(round(Int, x1 * (PATCH_DIM - 1)) - (INTERP_ORDER - 1) ÷ 2, 0, PATCH_DIM - INTERP_ORDER)

        z0 = (x0 * (PATCH_DIM - 1) - y0) * h
        z1 = (x1 * (PATCH_DIM - 1) - y1) * h
        j_polar = ir + RAD_DIM * (ia - 1)

        for i0 in 1:INTERP_ORDER, i1 in 1:INTERP_ORDER
            i_grid = (y0 + i0) + PATCH_DIM * (y1 + i1 - 1)
            I_coo[idx] = i_grid
            J_coo[idx] = j_polar
            V_coo[idx] = lagrange_interp(z0, z1, i0 - 1, i1 - 1)
            idx += 1
        end
    end

    P2G = sparse(I_coo, J_coo, V_coo, Ngrid, Npolar)

    return SingularQuadratureData(qx, qw, Gpou, Ppou, P2G, PATCH_DIM, PATCH_RAD, ANG_DIM, RAD_DIM, INTERP_ORDER)
end

# Cached singular quadrature; rebuilt if PATCH_RAD / RAD_DIM / INTERP_ORDER change
const SINGULAR_QUAD_CACHE = Ref{Union{Nothing,SingularQuadratureData}}(nothing)

"""
    get_singular_quadrature(PATCH_RAD, RAD_DIM, INTERP_ORDER)

Return the cached [`SingularQuadratureData`](@ref), rebuilding it if the parameters changed.
"""
function get_singular_quadrature(PATCH_RAD::Int, RAD_DIM::Int, INTERP_ORDER::Int)
    cached = SINGULAR_QUAD_CACHE[]
    if !isnothing(cached) &&
       cached.PATCH_RAD == PATCH_RAD &&
       cached.RAD_DIM == RAD_DIM &&
       cached.INTERP_ORDER == INTERP_ORDER
        return cached
    end
    SINGULAR_QUAD_CACHE[] = SingularQuadratureData(PATCH_RAD, RAD_DIM, INTERP_ORDER)
    return SINGULAR_QUAD_CACHE[]
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
    r2 = dx*dx + dy*dy + dz*dz
    r2 < 1e-30 && return (0.0, 0.0)
    rinv = inv(sqrt(r2))
    single = rinv
    r3inv = rinv * rinv * rinv
    double = (dx*nx + dy*ny + dz*nz) * r3inv
    return (single, double)
end

"""Extract a periodically wrapped `PATCH_DIM × PATCH_DIM` patch centered at `(idx_pol_center, idx_tor_center)`."""
function extract_patch!(patch::Array{Float64,3}, data::Matrix{Float64}, idx_pol_center::Int, idx_tor_center::Int, npol::Int, ntor::Int, PATCH_DIM::Int)
    PATCH_RAD = (PATCH_DIM - 1) ÷ 2
    @inbounds for j in 1:PATCH_DIM, i in 1:PATCH_DIM
        idx_pol = periodic_wrap(idx_pol_center - PATCH_RAD + i - 1, npol)
        idx_tor = periodic_wrap(idx_tor_center - PATCH_RAD + j - 1, ntor)
        idx_src = idx_pol + npol * (idx_tor - 1)
        patch[i, j, 1] = data[idx_src, 1]
        patch[i, j, 2] = data[idx_src, 2]
        patch[i, j, 3] = data[idx_src, 3]
    end
end

"""Interpolate a Cartesian patch onto polar quadrature nodes: `polar = P2G' * patch`."""
function interpolate_to_polar!(polar_data::Array{Float64,3}, patch::Array{Float64,3}, P2G::SparseMatrixCSC{Float64,Int})
    patch_flat = reshape(patch, :, size(patch, 3))
    mul!(reshape(polar_data, :, size(patch, 3)), P2G', patch_flat)
end

"""
    compute_polar_normal!(n_polar, dr_dθ, dr_dζ, normal_orient)

`n = ∂r/∂θ × ∂r/∂ζ` at the polar nodes. Re-apply `normal_orient`: these normals are rebuilt from
interpolated tangents, so they do not inherit the stored surface orientation.
"""
function compute_polar_normal!(n_polar::Array{Float64,3}, dr_dθ::Array{Float64,3}, dr_dζ::Array{Float64,3}, normal_orient::Int)
    @inbounds for ia in axes(dr_dθ, 2), ir in axes(dr_dθ, 1)
        n_polar[ir, ia, 1] = dr_dθ[ir, ia, 2] * dr_dζ[ir, ia, 3] - dr_dθ[ir, ia, 3] * dr_dζ[ir, ia, 2]
        n_polar[ir, ia, 2] = dr_dθ[ir, ia, 3] * dr_dζ[ir, ia, 1] - dr_dθ[ir, ia, 1] * dr_dζ[ir, ia, 3]
        n_polar[ir, ia, 3] = dr_dθ[ir, ia, 1] * dr_dζ[ir, ia, 2] - dr_dθ[ir, ia, 2] * dr_dζ[ir, ia, 1]
    end
    n_polar .*= normal_orient
end

"""Thread-local scratch for `compute_3D_kernel_matrices!`. One instance per thread so the parallel observer loop does not race."""
struct KernelWorkspace
    r_patch::Array{Float64,3}
    dr_dθ_patch::Array{Float64,3}
    dr_dζ_patch::Array{Float64,3}
    r_polar::Array{Float64,3}
    dr_dθ_polar::Array{Float64,3}
    dr_dζ_polar::Array{Float64,3}
    n_polar::Array{Float64,3}
    M_polar_single::Matrix{Float64}
    M_polar_double::Matrix{Float64}
    M_grid_single_flat::Vector{Float64}
    M_grid_double_flat::Vector{Float64}
end

"""
    KernelWorkspace(PATCH_DIM, RAD_DIM, ANG_DIM)

Preallocated patch, polar, and grid buffers for one observer row.
"""
function KernelWorkspace(PATCH_DIM::Int, RAD_DIM::Int, ANG_DIM::Int)
    return KernelWorkspace(
        zeros(PATCH_DIM, PATCH_DIM, 3),
        zeros(PATCH_DIM, PATCH_DIM, 3),
        zeros(PATCH_DIM, PATCH_DIM, 3),
        zeros(RAD_DIM, ANG_DIM, 3),
        zeros(RAD_DIM, ANG_DIM, 3),
        zeros(RAD_DIM, ANG_DIM, 3),
        zeros(RAD_DIM, ANG_DIM, 3),
        zeros(RAD_DIM, ANG_DIM),
        zeros(RAD_DIM, ANG_DIM),
        zeros(PATCH_DIM^2),
        zeros(PATCH_DIM^2)
    )
end

"""
    compute_3D_kernel_matrices!(grad_blocks, green_blocks, observer, source, PATCH_RAD, RAD_DIM, INTERP_ORDER, phases, sym_basis=nothing)

3D single- and double-layer kernels with Malhotra's polar singular correction (PPCF 2019 024004).

Far field: trapezoidal rule. Near field: polar quadrature blended by a partition of unity.

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
    num_points_per_fp = num_points ÷ length(phases) # observer/source points in one field period
    dθdζ = 4π^2 / num_points

    # Surface sub-block of the operator this call fills
    col_index = (source isa PlasmaGeometry3D ? 1 : 2)
    row_index = (observer isa PlasmaGeometry3D ? 1 : 2)

    # 𝒢ⁿ only needed for plasma as source term (RHS of eqs. 26/27 in Chance 1997)
    populate_greenfunction = source isa PlasmaGeometry3D

    # Each reflection pair's second row follows from its first, so only the representatives are evaluated
    observers = sym_basis === nothing ? (1:num_points_per_fp) : sym_basis.pair_reps

    # This allows the code to run at lower resolution without erroring out, but will warn the user.
    # Bound once into a new name: reassigning PATCH_RAD inside the branch would box it in the threaded closure below.
    patch_rad = min(PATCH_RAD, (min(source.mtheta, source.nzeta) - 1) ÷ 2)
    if patch_rad < PATCH_RAD
        @warn "PATCH_RAD=$(PATCH_RAD) is greater than half the number of points in the toroidal or poloidal direction, which is not supported. Setting PATCH_RAD to $(patch_rad), be sure to check if outputs are converged. This can be avoided by setting mtheta and nzeta to be greater than $(2 * PATCH_RAD + 1)."
    end

    quad_data = get_singular_quadrature(patch_rad, RAD_DIM, INTERP_ORDER)
    (; PATCH_DIM, ANG_DIM, Ppou, Gpou, P2G) = quad_data

    # One operator row is accumulated per observer, so the field-period fold and the basis change both happen before anything is stored
    max_threadid = Threads.maxthreadid()
    workspaces = [KernelWorkspace(PATCH_DIM, RAD_DIM, ANG_DIM) for _ in 1:max_threadid]
    Trow = eltype(phases)
    rows_double = [zeros(Trow, num_points_per_fp) for _ in 1:max_threadid]
    rows_single = [zeros(Trow, num_points_per_fp) for _ in 1:max_threadid]
    # `store_kernel_row!` needs two rows of scratch: the reconstructed partner and their combination
    sym_work = [zeros(ComplexF64, sym_basis === nothing ? 0 : 2 * num_points_per_fp) for _ in 1:max_threadid]

    # :static pins each task to its thread, so the threadid()-indexed scratch below stays private
    Threads.@threads :static for i_pair in eachindex(observers)
        tid = Threads.threadid()
        ws = workspaces[tid]
        (; r_patch, dr_dθ_patch, dr_dζ_patch, r_polar, dr_dθ_polar, dr_dζ_polar,
            n_polar, M_polar_single, M_polar_double, M_grid_single_flat, M_grid_double_flat) = ws
        row_double = rows_double[tid]
        row_single = rows_single[tid]
        fill!(row_double, 0)
        populate_greenfunction && fill!(row_single, 0)

        # Convert linear index to 2D indices
        idx_obs = observers[i_pair]
        i_obs = mod1(idx_obs, observer.mtheta)
        j_obs = (idx_obs - 1) ÷ observer.mtheta + 1
        r_obs = @view observer.r[idx_obs, :]

        # ============================================================
        # FAR FIELD: Trapezoidal rule for nonsingular source points
        # Note: kernels return zero for r_src = r_obs
        # ============================================================
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

        # ============================================================
        # NEAR FIELD: Polar quadrature with singular correction
        # ============================================================
        # Extract patches of source data around the singular point (size = PATCH_DIM x PATCH_DIM x dof)
        extract_patch!(r_patch, source.r, i_obs, j_obs, source.mtheta, source.nzeta, PATCH_DIM)
        extract_patch!(dr_dθ_patch, source.dr_dθ, i_obs, j_obs, source.mtheta, source.nzeta, PATCH_DIM)
        extract_patch!(dr_dζ_patch, source.dr_dζ, i_obs, j_obs, source.mtheta, source.nzeta, PATCH_DIM)

        # Interpolate coordinates and tangent vectors to polar quadrature points
        interpolate_to_polar!(r_polar, r_patch, P2G)
        interpolate_to_polar!(dr_dθ_polar, dr_dθ_patch, P2G)
        interpolate_to_polar!(dr_dζ_polar, dr_dζ_patch, P2G)

        # Compute normal vectors at polar points from interpolated tangent vectors
        compute_polar_normal!(n_polar, dr_dθ_polar, dr_dζ_polar, source.normal_orient)

        # Evaluate kernels and apply quadrature weights: area element × POU, where POU contains rdrdθ already
        @inbounds for ia in 1:ANG_DIM, ir in 1:RAD_DIM
            # Evaluate kernels using recomputed normal (use @view to avoid allocation)
            r_src = @view r_polar[ir, ia, :]
            n_src = @view n_polar[ir, ia, :]
            near_single, near_double = laplace_kernel(r_obs[1], r_obs[2], r_obs[3], r_src[1], r_src[2], r_src[3], n_src[1], n_src[2], n_src[3])

            # Apply quadrature weights: area element × POU, where POU contains rdrdθ already
            M_polar_single[ir, ia] = near_single * Ppou[ir, ia] * dθdζ
            M_polar_double[ir, ia] = near_double * Ppou[ir, ia] * dθdζ
        end

        # Distribute polar singular corrections back to Cartesian grid using sparse matrix
        # grid = P2G * polar (maps Npolar → Ngrid)
        mul!(M_grid_double_flat, P2G, vec(M_polar_double))
        M_grid_double = reshape(M_grid_double_flat, PATCH_DIM, PATCH_DIM)
        if populate_greenfunction
            mul!(M_grid_single_flat, P2G, vec(M_polar_single))
            M_grid_single = reshape(M_grid_single_flat, PATCH_DIM, PATCH_DIM)
        end

        # POU correction: singular correction + (1 + Gpou) * far-field terms
        @inbounds for j in 1:PATCH_DIM, i in 1:PATCH_DIM
            # Map back to global indices
            idx_pol = periodic_wrap(i_obs - patch_rad + i - 1, source.mtheta)
            idx_tor = periodic_wrap(j_obs - patch_rad + j - 1, source.nzeta)
            idx_src = idx_pol + source.mtheta * (idx_tor - 1)
            d, idx_col = fldmod1(idx_src, num_points_per_fp)

            # Remainder of far-field contribution on the singular grid: Gpou = -χ
            r_src = @view source.r[idx_src, :]
            n_src = @view source.normal[idx_src, :]
            far_single, far_double = laplace_kernel(r_obs[1], r_obs[2], r_obs[3], r_src[1], r_src[2], r_src[3], n_src[1], n_src[2], n_src[3])

            # Apply near + far contributions
            populate_greenfunction && (row_single[idx_col] += phases[d] * (M_grid_single[i, j] + far_single * Gpou[i, j] * dθdζ))
            row_double[idx_col] += phases[d] * (M_grid_double[i, j] + far_double * Gpou[i, j] * dθdζ)
        end

        # Normalize so the Green's-identity term below is a unit shift and the exterior/interior jump is 2I, as in the 2D kernel
        row_double ./= 2π
        populate_greenfunction && (row_single ./= 2π)

        store_kernel_row!(grad_blocks, sym_basis, row_double, sym_work[tid], i_pair, row_index, col_index)
        populate_greenfunction && store_kernel_row!(green_blocks, sym_basis, row_single, sym_work[tid], i_pair, row_index, 1)
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
