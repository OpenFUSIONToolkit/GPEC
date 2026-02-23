"""
    periodic_wrap(x, n) -> Int

Inline function for fast periodic wrapping for indices near the valid range [1, n].
Equivalent to `mod1(x, n)` but avoids division for small offsets.
Only valid when `x` is within one period of the valid range (i.e., `1-n < x < 2n`).
"""
@inline periodic_wrap(x::Int, n::Int) = x < 1 ? x + n : (x > n ? x - n : x)

"""
    SingularQuadratureData

Precomputed data for singular correction quadrature following BIEST approach.
Initialized once on first use.

## Fields

    - `qx::Vector{Float64}`: Radial quadrature points in [0,1]
    - `qw::Vector{Float64}`: Radial quadrature weights
    - `Gpou::Matrix{Float64}`: Partition of unity on Cartesian grid (PATCH_DIM × PATCH_DIM)
    - `Ppou::Matrix{Float64}`: Partition of unity on polar grid (RAD_DIM × ANG_DIM)
    - `P2G::SparseMatrixCSC{Float64,Int}`: Sparse interpolation matrix (Ngrid × Npolar) mapping polar quadrature points to Cartesian grid
        - Forward (patch→polar): `polar = P2G' * patch`
        - Backward (polar→grid): `grid = P2G * polar`.
    - `PATCH_DIM::Int`: Patch dimension (odd integer)
    - `PATCH_RAD::Int`: Patch radius (number of points adjacent to source point treated as singular)
    - `ANG_DIM::Int`: Number of angular quadrature points
    - `RAD_DIM::Int`: Number of radial quadrature points
    - `INTERP_ORDER::Int`: Lagrange interpolation order
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
    SingularQuadratureData(PATCH_RAD::Int, RAD_DIM::Int, INTERP_ORDER::Int)

Constructor which initializes quadrature points, weights, partition-of-unity functions, and
interpolation matrices for singular correction based on input parameters. Follows BIEST's approach.

# Arguments

  - `PATCH_RAD::Int`: Number of points adjacent to source point to treat as singular
  - `RAD_DIM::Int`: Radial quadrature order
  - `INTERP_ORDER::Int`: Lagrange interpolation order

# Returns

  - `SingularQuadratureData`: Precomputed quadrature data
"""
function SingularQuadratureData(PATCH_RAD::Int, RAD_DIM::Int, INTERP_ORDER::Int)

    # Total size of square patch extracted around singular point (odd number: 2*PATCH_DIM0+1)
    PATCH_DIM = 2 * PATCH_RAD + 1
    @assert INTERP_ORDER <= PATCH_DIM "Must have INTERP_ORDER <= PATCH_DIM, got INTERP_ORDER=$INTERP_ORDER, PATCH_DIM=$PATCH_DIM"
    # Number of angular quadrature nodes in polar coordinates (uniformly distributed around circle)
    ANG_DIM = 2 * RAD_DIM

    # Setup radial quadrature
    qx_raw, qw_raw = gausslegendre(RAD_DIM) # points on [-1,1]
    qx = (qx_raw .+ 1) ./ 2  # Map [-1, 1] to [0, 1]
    qw = qw_raw ./ 2         # Adjust weights for interval change

    # Partition of unity function, exp(-36 * r^p) where p depends on PATCH_DIM
    pou_power = PATCH_DIM > 45 ? 10 : (PATCH_DIM > 20 ? 8 : 6)
    pou(r) = r ≥ 1.0 ? 0.0 : exp(-36.0 * r^pou_power)

    # Partition of Unity on Cartesian grid
    Gpou = zeros(PATCH_DIM, PATCH_DIM)
    coords = LinRange(-1.0, 1.0, PATCH_DIM)
    for (i, x) in enumerate(coords), (j, y) in enumerate(coords)
        Gpou[i, j] = -pou(sqrt(x^2 + y^2))
    end

    # Partition of Unity on polar grid including transformation Jacobian - Ppou = χ(ρ) M²/4 r dr dt, eq. 38 in Malhotra 2019
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

    # Build sparse interpolation operator P2G ∈ ℝ^{Ngrid × Npolar}
    #   grid_values  = P2G  * polar_values
    #   polar_values = P2G' * grid_values
    # Each column of P2G contains the INTERP_ORDER² Lagrange weights
    # mapping one polar sample to its surrounding Cartesian grid stencil.
    Ngrid = PATCH_DIM * PATCH_DIM
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

        # Lower-left corner indices of INTERP_ORDER × INTERP_ORDER stencil centered on (x0,x1)
        y0 = clamp(trunc(Int, x0 * (PATCH_DIM - 1) - (INTERP_ORDER - 1) ÷ 2), 0, PATCH_DIM - INTERP_ORDER)
        y1 = clamp(trunc(Int, x1 * (PATCH_DIM - 1) - (INTERP_ORDER - 1) ÷ 2), 0, PATCH_DIM - INTERP_ORDER)

        # Local coordinates within INTERP_ORDER×INTERP_ORDER stencil, normalized to [0,1]
        z0 = (x0 * (PATCH_DIM - 1) - y0) * h
        z1 = (x1 * (PATCH_DIM - 1) - y1) * h

        # Polar point index (column in P2G)
        j_polar = ir + RAD_DIM * (ia - 1)

        # Populate stencil contributions for this polar node
        for i0 in 1:INTERP_ORDER, i1 in 1:INTERP_ORDER
            # Grid point index (row in P2G), using column-major layout
            i_grid = (y0 + i0) + PATCH_DIM * (y1 + i1 - 1)
            I_coo[idx] = i_grid
            J_coo[idx] = j_polar
            V_coo[idx] = lagrange_interp(z0, z1, i0 - 1, i1 - 1)
            idx += 1
        end
    end

    # Assemble sparse interpolation matrix
    P2G = sparse(I_coo, J_coo, V_coo, Ngrid, Npolar)

    return SingularQuadratureData(qx, qw, Gpou, Ppou, P2G, PATCH_DIM, PATCH_RAD, ANG_DIM, RAD_DIM, INTERP_ORDER)
end

# Global cache for quadrature data (initialized on first use)
const SINGULAR_QUAD_CACHE = Ref{Union{Nothing,SingularQuadratureData}}(nothing)

"""
    get_singular_quadrature(PATCH_RAD::Int, RAD_DIM::Int, INTERP_ORDER::Int)

Get cached singular quadrature data, initializing if necessary. Returns cached data
if parameters match the cached initialization; reinitializes if parameters differ.
This allows the user to change quadrature parameters between calls, but prevents
redundant reinitialization when parameters are unchanged.
"""
function get_singular_quadrature(PATCH_RAD::Int, RAD_DIM::Int, INTERP_ORDER::Int)

    # Check if cache exists and parameters match
    cached = SINGULAR_QUAD_CACHE[]
    if !isnothing(cached) &&
       cached.PATCH_RAD == PATCH_RAD &&
       cached.RAD_DIM == RAD_DIM &&
       cached.INTERP_ORDER == INTERP_ORDER
        return cached
    end

    # Reinitialize if parameters changed or cache is empty
    SINGULAR_QUAD_CACHE[] = SingularQuadratureData(PATCH_RAD, RAD_DIM, INTERP_ORDER)
    return SINGULAR_QUAD_CACHE[]
end

"""
    laplace_single_layer(x_obs, x_src) -> Float64

Evaluate the Laplace single-layer (FxU) kernel between two 3D points. Returns
0.0 if the observation point coincides with the source point to avoid singularity.

The single-layer kernel φ is the fundamental solution to Laplace's equation:

```
φ(x_obs, x_src) = 1 / |x_obs - x_src|
```

# Arguments

  - `x_obs`: Observation point (3D Cartesian coordinates, any AbstractVector)
  - `x_src`: Source point (3D Cartesian coordinates, any AbstractVector)

# Returns

  - `Float64`: Kernel value φ(x_obs, x_src)
"""
function laplace_single_layer(x_obs::AbstractVector{<:Real}, x_src::AbstractVector{<:Real})
    @inbounds begin
        dx = x_obs[1] - x_src[1]
        dy = x_obs[2] - x_src[2]
        dz = x_obs[3] - x_src[3]
    end
    r2 = dx*dx + dy*dy + dz*dz
    r2 < 1e-30 && return 0.0
    return inv(sqrt(r2))
end
"""
    laplace_double_layer(x_obs, x_src, n_src) -> Float64

Evaluate the Laplace double-layer (DxU) kernel between a point and a surface element. Returns
0.0 if the observation point coincides with the source point to avoid singularity. Allocation-free
scalar arithmetic is used for maximum performance.

The double-layer kernel K is the normal derivative of the fundamental solution:

```
K(x_obs, x_src, n_src) = ∇_{x_src} φ · n_src = (x_obs - x_src) · n_src / |x_obs - x_src|³
```

# Arguments

  - `x_obs`: Observation point (3D Cartesian coordinates, any AbstractVector)
  - `x_src`: Source point on surface (3D Cartesian coordinates, any AbstractVector)
  - `n_src`: Outward UNIT normal at source point (must be normalized!, any AbstractVector)

# Returns

  - `Float64`: Kernel value K(x_obs, x_src, n_src)
"""
function laplace_double_layer(x_obs::AbstractVector{<:Real}, x_src::AbstractVector{<:Real}, n_src::AbstractVector{<:Real})
    @inbounds begin
        dx = x_obs[1] - x_src[1]
        dy = x_obs[2] - x_src[2]
        dz = x_obs[3] - x_src[3]
        nx = n_src[1]
        ny = n_src[2]
        nz = n_src[3]
    end
    r2 = dx*dx + dy*dy + dz*dz
    r2 < 1e-30 && return 0.0
    rinv = inv(sqrt(r2))
    r3inv = rinv * rinv * rinv
    return (dx*nx + dy*ny + dz*nz) * r3inv
end

"""
    extract_patch!(patch, data, Nt, Np, t0, p0, PATCH_DIM)

Extract a PATCH_DIM × PATCH_DIM patch of data centered at (t0, p0) with periodic wrapping.

# Arguments

  - `patch`: Preallocated output array for data around the singular point (PATCH_DIM × PATCH_DIM × dof)
  - `data`: Source data array (can be coordinates, normals, or area elements)
  - `Nt, Np`: Grid dimensions (toroidal, poloidal)
  - `t0, p0`: Center indices (1-based)
  - `PATCH_DIM`: Patch size (must be odd)
"""
function extract_patch!(patch::Array{Float64,3}, data::Matrix{Float64}, idx_pol_center::Int, idx_tor_center::Int, npol::Int, ntor::Int, PATCH_DIM::Int)

    PATCH_RAD = (PATCH_DIM - 1) ÷ 2
    @inbounds for j in 1:PATCH_DIM, i in 1:PATCH_DIM
        # Enforce periodicity
        idx_pol = periodic_wrap(idx_pol_center - PATCH_RAD + i - 1, npol)
        idx_tor = periodic_wrap(idx_tor_center - PATCH_RAD + j - 1, ntor)
        # Copy data to the patch using direct indexing (avoids view allocation)
        idx_src = idx_pol + npol * (idx_tor - 1)
        patch[i, j, 1] = data[idx_src, 1]
        patch[i, j, 2] = data[idx_src, 2]
        patch[i, j, 3] = data[idx_src, 3]
    end
end

"""
    interpolate_to_polar!(polar_data, patch, quad_data)

Interpolate Cartesian patch data to polar quadrature points using sparse matrix multiply.
Overwrites `polar_data` using mul! function arguments, mul!(C, A, B, α, β) -> C where
C = α * A * B + β * C.

# Arguments

  - `polar_data`: Preallocated output array for polar data (RAD_DIM × ANG_DIM × dof)
  - `patch`: Patch data (PATCH_DIM × PATCH_DIM × dof)
  - `P2G`: Sparse interpolation matrix
"""
function interpolate_to_polar!(polar_data::Array{Float64,3}, patch::Array{Float64,3}, P2G::SparseMatrixCSC{Float64,Int})
    # Flatten patch to (Ngrid × dof), apply P2G' to get (Npolar × dof)
    patch_flat = reshape(patch, :, size(patch, 3))
    mul!(reshape(polar_data, :, size(patch, 3)), P2G', patch_flat, 1.0, 0.0)
end

"""
    compute_polar_normal!(n_polar, dr_dθ_polar, dr_dζ_polar)

Compute normal vector (= ∂r/∂θ × ∂r/∂ζ) at polar quadrature points from interpolated tangent vectors.
We already scaled the normals by normal_orient in the geometry construction, so we need to reapply
that here since we are recomputing the normals from the derivatives.

# Arguments

  - `n_polar`: Preallocation unit normal vector at each polar point (RAD_DIM × ANG_DIM × 3)
  - `dr_dθ_polar`: Interpolated ∂r/∂θ at polar points (RAD_DIM × ANG_DIM × 3)
  - `dr_dζ_polar`: Interpolated ∂r/∂ζ at polar points (RAD_DIM × ANG_DIM × 3)
  - `normal_orient`: Multiplier applied to normals to make them orient out of vacuum region (+1 or -1)
"""
function compute_polar_normal!(n_polar::Array{Float64,3}, dr_dθ::Array{Float64,3}, dr_dζ::Array{Float64,3}, normal_orient::Int)
    # Inline cross product to avoid slice allocation
    @inbounds for ia in axes(dr_dθ, 2), ir in axes(dr_dθ, 1)
        n_polar[ir, ia, 1] = dr_dθ[ir, ia, 2] * dr_dζ[ir, ia, 3] - dr_dθ[ir, ia, 3] * dr_dζ[ir, ia, 2]
        n_polar[ir, ia, 2] = dr_dθ[ir, ia, 3] * dr_dζ[ir, ia, 1] - dr_dθ[ir, ia, 1] * dr_dζ[ir, ia, 3]
        n_polar[ir, ia, 3] = dr_dθ[ir, ia, 1] * dr_dζ[ir, ia, 2] - dr_dθ[ir, ia, 2] * dr_dζ[ir, ia, 1]
    end
    n_polar .*= normal_orient
end

"""
    KernelWorkspace

Thread-local workspace for `compute_3D_kernel_matrix!` to enable parallel execution.
Each thread gets its own workspace to avoid data races on temporary arrays.
"""
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

Create a new workspace with pre-allocated arrays for kernel matrix computation.
"""
function KernelWorkspace(PATCH_DIM::Int, RAD_DIM::Int, ANG_DIM::Int)
    return KernelWorkspace(
        zeros(PATCH_DIM, PATCH_DIM, 3),      # r_patch
        zeros(PATCH_DIM, PATCH_DIM, 3),      # dr_dθ_patch
        zeros(PATCH_DIM, PATCH_DIM, 3),      # dr_dζ_patch
        zeros(RAD_DIM, ANG_DIM, 3),          # r_polar
        zeros(RAD_DIM, ANG_DIM, 3),          # dr_dθ_polar
        zeros(RAD_DIM, ANG_DIM, 3),          # dr_dζ_polar
        zeros(RAD_DIM, ANG_DIM, 3),          # n_polar
        zeros(RAD_DIM, ANG_DIM),             # M_polar_single
        zeros(RAD_DIM, ANG_DIM),             # M_polar_double
        zeros(PATCH_DIM^2),                  # M_grid_single_flat
        zeros(PATCH_DIM^2)                   # M_grid_double_flat
    )
end

"""
    compute_3D_kernel_matrix!(grad_greenfunction, greenfunction, observer, source; PATCH_RAD=3, RAD_DIM=15, INTERP_ORDER=6)

Compute boundary integral kernel matrices for 3D geometries with the singular correction
algorithm from Malhotra et al. 2019. Uses multi-threading for parallel computation over
observer points.

  - Far regions: Rectangle rule with uniform weights (1/N)
  - Singular regions: Polar quadrature with partition-of-unity blending

grad_greenfunction is the double-layer kernel matrix, where each entry is
∇_{x_src} φ(x_obs, x_src) · n_src, and greenfunction is the single-layer kernel matrix,
where each entry is φ(x_obs, x_src).

# Arguments

  - `grad_greenfunction`: Double-layer kernel matrix (Nobs × Nsrc) filled in place

  - `greenfunction`: Single-layer kernel matrix (Nobs × Nsrc) filled in place
  - `observer`: Observer geometry (PlasmaGeometry3D)
  - `source`: Source geometry (PlasmaGeometry3D)
  - `PATCH_RAD`: Number of points adjacent to source point to treat as singular

      + Total patch size in # of gridpoints = (2 * PATCH_RAD + 1) x (2 * PATCH_RAD + 1)
  - `RAD_DIM`: Polar radial quadrature order. Angular order = 2 * RAD_DIM
  - `INTERP_ORDER`: Lagrange interpolation order

      + Must be ≤ (2 * PATCH_RAD + 1)

# Threading

This function automatically uses all available threads (`Threads.nthreads()`).
Start Julia with `julia -t auto` or set `JULIA_NUM_THREADS` to enable multi-threading.
"""
function compute_3D_kernel_matrices!(
    grad_greenfunction::Matrix{Float64},
    greenfunction::Matrix{Float64},
    observer::Union{PlasmaGeometry3D,WallGeometry3D},
    source::Union{PlasmaGeometry3D,WallGeometry3D},
    PATCH_RAD::Int,
    RAD_DIM::Int,
    INTERP_ORDER::Int
)
    num_points = observer.mtheta * observer.nzeta
    dθdζ = 4π^2 / (num_points)

    # Get block of grad green function matrix
    col_index = (source isa PlasmaGeometry3D ? 1 : 2)
    row_index = (observer isa PlasmaGeometry3D ? 1 : 2)
    grad_greenfunction_block = view(
        grad_greenfunction,
        ((row_index-1)*num_points+1):(row_index*num_points),
        ((col_index-1)*num_points+1):(col_index*num_points)
    )

    # Zero out green function matrix
    fill!(greenfunction, 0.0)
    # 𝒢ⁿ only needed for plasma as source term (RHS of eqs. 26/27 in Chance 1997)
    populate_greenfunction = source isa PlasmaGeometry3D

    # Initialize quadrature data
    # This allows the code to run at lower resolution without erroring out, but will warn the user.
    if PATCH_RAD > (observer.nzeta - 1) ÷ 2
        warn(
            "PATCH_RAD is greater than the number of points in the toroidal direction, which is not supported. Setting PATCH_RAD to $((observer.nzeta - 1) ÷ 2). Double check that you are converged."
        )
        PATCH_RAD = (observer.nzeta - 1) ÷ 2
    end
    quad_data = get_singular_quadrature(PATCH_RAD, RAD_DIM, INTERP_ORDER)
    (; PATCH_DIM, PATCH_RAD, ANG_DIM, RAD_DIM, Ppou, Gpou, P2G) = quad_data
    @assert observer.mtheta ≥ PATCH_DIM "Must have observer.mtheta ≥ PATCH_DIM, got observer.mtheta=$(observer.mtheta), PATCH_DIM=$PATCH_DIM"
    @assert observer.nzeta ≥ PATCH_DIM "Must have observer.nzeta ≥ PATCH_DIM, got observer.nzeta=$(observer.nzeta), PATCH_DIM=$PATCH_DIM"

    # Allocate thread-local workspaces (one per thread)
    nthreads = Threads.nthreads()
    workspaces = [KernelWorkspace(PATCH_DIM, RAD_DIM, ANG_DIM) for _ in 1:nthreads]

    # Parallel loop through observer points
    Threads.@threads for idx_obs in 1:num_points
        # Get thread-local workspace
        ws = workspaces[Threads.threadid()]
        (; r_patch, dr_dθ_patch, dr_dζ_patch, r_polar, dr_dθ_polar, dr_dζ_polar,
            n_polar, M_polar_single, M_polar_double, M_grid_single_flat, M_grid_double_flat) = ws

        # Convert linear index to 2D indices
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
            K_single = laplace_single_layer(r_obs, r_src)
            K_double = laplace_double_layer(r_obs, r_src, n_src)

            # Apply weights (periodic trapezoidal rule = constant weights)
            if populate_greenfunction
                greenfunction[idx_obs, idx_src] = K_single * dθdζ
            end
            grad_greenfunction_block[idx_obs, idx_src] = K_double * dθdζ
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

        # Evaluate kernels at polar points with POU weighting
        @inbounds for ia in 1:ANG_DIM, ir in 1:RAD_DIM
            # Evaluate kernels using recomputed normal (use @view to avoid allocation)
            r_src = @view r_polar[ir, ia, :]
            n_src = @view n_polar[ir, ia, :]
            K_single = laplace_single_layer(r_obs, r_src)
            K_double = laplace_double_layer(r_obs, r_src, n_src)

            # Apply quadrature weights: area element × POU, where POU contains rdrdθ already
            M_polar_single[ir, ia] = K_single * Ppou[ir, ia] * dθdζ
            M_polar_double[ir, ia] = K_double * Ppou[ir, ia] * dθdζ
        end

        # Distribute polar singular corrections back to Cartesian grid using sparse matrix
        # grid = P2G * polar (maps Npolar → Ngrid)
        mul!(M_grid_single_flat, P2G, vec(M_polar_single))
        mul!(M_grid_double_flat, P2G, vec(M_polar_double))
        M_grid_single = reshape(M_grid_single_flat, PATCH_DIM, PATCH_DIM)
        M_grid_double = reshape(M_grid_double_flat, PATCH_DIM, PATCH_DIM)

        # Compute remaining far-field POU contribution and near-field polar quadrature result
        # We include this region in the far-field trapezoidal rule, so use Gpou = -χ here to get 1-χ
        @inbounds for j in 1:PATCH_DIM, i in 1:PATCH_DIM
            # Map back to global indices
            idx_pol = periodic_wrap(i_obs - PATCH_RAD + i - 1, source.mtheta)
            idx_tor = periodic_wrap(j_obs - PATCH_RAD + j - 1, source.nzeta)
            idx_src = idx_pol + source.mtheta * (idx_tor - 1)

            # Remainder of far-field contribution on the singular grid: Gpou = -χ
            r_src = @view source.r[idx_src, :]
            n_src = @view source.normal[idx_src, :]
            far_single = laplace_single_layer(r_obs, r_src) * Gpou[i, j] * dθdζ
            far_double = laplace_double_layer(r_obs, r_src, n_src) * Gpou[i, j] * dθdζ

            # Apply near + far contributions
            if populate_greenfunction
                greenfunction[idx_obs, idx_src] += M_grid_single[i, j] + far_single
            end
            grad_greenfunction_block[idx_obs, idx_src] += M_grid_double[i, j] + far_double
        end
    end

    # Use the same normalization as in the 2D kernel so we can just add I to the diagonal
    # This makes the grri logic identical to the 2D kernel.
    grad_greenfunction_block ./= 2π
    greenfunction ./= 2π

    # Add the term that comes from the volume integral of Green's identity
    typeof(source) == typeof(observer) && begin
        for i in 1:num_points
            grad_greenfunction_block[i, i] += 1.0
        end
    end
end

"""
    kernel!(grad_greenfunction, greenfunction, observer, source, params::KernelParams3D)

Dispatch wrapper for 3D kernel that forwards to `compute_3D_kernel_matrices!` with params.
"""
function kernel!(
    grad_greenfunction::Matrix{Float64},
    greenfunction::Matrix{Float64},
    observer::Union{PlasmaGeometry3D,WallGeometry3D},
    source::Union{PlasmaGeometry3D,WallGeometry3D},
    params::KernelParams3D
)
    return compute_3D_kernel_matrices!(
        grad_greenfunction,
        greenfunction,
        observer,
        source,
        params.PATCH_RAD,
        params.RAD_DIM,
        params.INTERP_ORDER
    )
end
