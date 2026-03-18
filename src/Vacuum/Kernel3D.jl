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

    # Partition of Unity on polar grid including transformation Jacobian - Ppou = χ(ρ) M²/4 r dr dt, [Malhotra Journal of Comp. Phys. 2019 108791 eq. 38]
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
@fastmath function laplace_single_layer(x_obs::AbstractVector{<:Real}, x_src::AbstractVector{<:Real})
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
Scalar-argument single-layer kernel. Avoids view creation in tight loops.
"""
@fastmath @inline function laplace_single_layer(
    ox::Float64, oy::Float64, oz::Float64,
    sx::Float64, sy::Float64, sz::Float64
)
    dx = ox - sx;
    dy = oy - sy;
    dz = oz - sz
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
@fastmath function laplace_double_layer(x_obs::AbstractVector{<:Real}, x_src::AbstractVector{<:Real}, n_src::AbstractVector{<:Real})
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
Scalar-argument double-layer kernel. Avoids view creation in tight loops.
"""
@fastmath @inline function laplace_double_layer(
    ox::Float64, oy::Float64, oz::Float64,
    sx::Float64, sy::Float64, sz::Float64,
    nx::Float64, ny::Float64, nz::Float64
)
    dx = ox - sx;
    dy = oy - sy;
    dz = oz - sz
    r2 = dx*dx + dy*dy + dz*dz
    r2 < 1e-30 && return 0.0
    rinv = inv(sqrt(r2))
    r3inv = rinv * rinv * rinv
    return (dx*nx + dy*ny + dz*nz) * r3inv
end

"""
    extract_patch!(patch, data, idx_pol_center, idx_tor_center, npol, ntor, PATCH_DIM)

Extract a PATCH_DIM × PATCH_DIM patch of data centered at (idx_pol_center, idx_tor_center) with periodic wrapping.

# Arguments

  - `patch`: Preallocated output array for data around the singular point (PATCH_DIM × PATCH_DIM × dof)
  - `data`: Source data array (can be coordinates, normals, or area elements)
  - `idx_pol_center`: Center poloidal index
  - `idx_tor_center`: Center toroidal index
  - `npol`: Number of poloidal points
  - `ntor`: Number of toroidal points
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

Thread-local workspace for `compute_3D_kernel_matrices!` to enable parallel execution.
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
    compute_3D_kernel_matrices!(K, G, observer, source, PATCH_RAD, RAD_DIM, INTERP_ORDER, Z, Gram)

Compute the **Fourier/Galerkin-projected** 3D vacuum boundary-integral kernel blocks for
Laplace’s equation, using a high-order singular quadrature / partition-of-unity (POU)
scheme on a tensor-product `(θ, ζ)` surface grid.

Like the 2D kernel, this routine implements the **fused projection path** used by the vacuum solve:
it produces the projected operators in mode space **without materializing a dense**
`N×N` point-space kernel (where `N = mtheta * nzeta`).

## Mathematical object being discretized

Let `x(θ, ζ) ∈ ℝ^3` be a surface parametrization (plasma or wall surface) with outward
unit normal `n(θ, ζ)`. The Laplace kernels are:

  - **Single-layer**: `φ(x_obs, x_src) = 1 / |x_obs - x_src|`
  - **Double-layer**: `∂φ/∂n_src = ∇_{x_src} φ ⋅ n_src = (x_obs - x_src) ⋅ n_src / |x_obs - x_src|^3`

This routine computes the *discrete, projected* operators corresponding to these kernels,
using a uniform quadrature weight `dθdζ = 4π^2 / N` for the far field and a specialized
near-field correction for the singular region.

## Arguments and block layout

  - `Kc`: Complex global projected double-layer kernel matrix (2P×2P).
  - `Gc`: Complex global projected single-layer kernel matrix (2P×P).
  - `observer`: `PlasmaGeometry3D` or `WallGeometry3D` object providing geometry data.
  - `source`: `PlasmaGeometry3D` or `WallGeometry3D` object providing geometry data.
  - `PATCH_RAD`: Half-width of the singular patch in grid points. Must satisfy `PATCH_RAD ≤ (min(source.mtheta, source.nzeta) - 1) ÷ 2` to avoid errors.
  - `RAD_DIM`: Radial quadrature order on the polar grid (angular order is `2*RAD_DIM`).
  - `INTERP_ORDER`: Lagrange interpolation order used to build `P2G` (must satisfy `INTERP_ORDER ≤ 2*PATCH_RAD+1`).
  - `Z`: Complex Fourier basis sampled on the surface grid, shaped `N×P` (`P = number of retained modes`). `Z[idx, :]` contains the basis values at the surface node `idx`.
  - `Gram`: Diagonal of the mode-space Gram matrix used to add the analytic “identity” term when `typeof(source) == typeof(observer)` (i.e. the same operator block that receives the Green’s-identity diagonal contribution).

This routine fills exactly one `P×P` block view `Kc_block` (and optionally the corresponding `Gc_block`)
selected by whether observer/source are plasma or wall.

## Numerical treatment of the singularity

The kernel is weakly singular as `x_src → x_obs`. The implementation follows the
approach used in [Malhotra Journal of Comp. Phys. 2019 108791 eq. 38]

  - **Far field** (nonsingular sources):

      + Use a uniform trapezoidal/rectangle rule on the `(θ, ζ)` grid.
      + For each observer point, a square patch of size `PATCH_DIM = 2*PATCH_RAD+1`
        surrounding the singularity is excluded from the far-field sum.

  - **Near field** (singular patch):

      + Extract a Cartesian `PATCH_DIM×PATCH_DIM` patch of the source geometry around the
        observer-aligned source index.
      + Interpolate the patch to a **polar quadrature grid** (`RAD_DIM × ANG_DIM`, with `ANG_DIM=2*RAD_DIM`)
        using a precomputed sparse interpolation operator `P2G` built from tensor-product
        Lagrange stencils (`INTERP_ORDER` controls the stencil width).
      + Evaluate kernels on the polar grid and weight them with a **partition-of-unity**
        quadrature factor `Ppou` that includes the polar Jacobian factor (roughly `r * dr * dθ`)
        and a smooth cutoff function `χ(ρ)` that localizes the singular correction.
      + Map the polar correction back onto the Cartesian patch via `P2G` and blend with the
        far-field trapezoid contribution using `Gpou`, so the combined weight is effectively
        `trap*(1-χ) + singular_correction`.

## Fused projection and threading

This function is written to be parallel over observer points:

  - Each thread owns a `KernelWorkspace` (scratch arrays for patch extraction, polar interpolation,
    and temporary kernel values), plus per-thread accumulation buffers `proj_k` / `proj_g`
    (length `P`) and a boolean `is_patch` mask to skip patch indices in the far-field loop.

  - For a given observer index `idx_obs`, the code accumulates the **projected row**
    `(kernel row idx_obs) · Z` directly into `proj_k` / `proj_g` using `_accum_row!`, and then writes
    these into shared buffers `KZt[:, idx_obs]` and `GZt[:, idx_obs]`. This is race-free because
    each observer writes to a unique column.

  - After the threaded loop completes, the final `P×P` blocks are assembled efficiently with BLAS:

        Kc = Zᴴ * (KZt)'
        Gc = Zᴴ * (GZt)'

    implemented as `mul!(Kc_block, Z', transpose(KZt))` (and similarly for `Gc`).

Normalization by `2π` is applied to match the 2D kernel convention so the downstream “add identity”
logic is consistent between 2D/3D.

## Important parameters

  - `PATCH_RAD`: half-width of the singular patch in grid points. Must satisfy `PATCH_RAD ≤ (min(source.mtheta, source.nzeta) - 1) ÷ 2` to avoid errors.
  - `RAD_DIM`: radial quadrature order on the polar grid (angular order is `2*RAD_DIM`).
  - `INTERP_ORDER`: Lagrange interpolation order used to build `P2G` (must satisfy `INTERP_ORDER ≤ 2*PATCH_RAD+1`).

## Performance notes / numerical optimizations

  - **Cached quadrature data**: `get_singular_quadrature` memoizes `P2G`, `Gpou`, `Ppou`, etc. for a given
    `(PATCH_RAD, RAD_DIM, INTERP_ORDER)` triple to avoid expensive rebuilds.
  - **Allocation control**: all near-field arrays live in thread-local `KernelWorkspace` objects; no per-observer
    heap allocation is intended in the hot path.
  - **Scalar kernel evaluation**: the Laplace kernels have scalar-argument overloads to avoid view/slice creation
    and to enable LLVM to keep values in registers.
"""
function compute_3D_kernel_matrices!(
    K::AbstractMatrix{ComplexF64},
    G::AbstractMatrix{ComplexF64},
    observer::Union{PlasmaGeometry3D,WallGeometry3D},
    source::Union{PlasmaGeometry3D,WallGeometry3D},
    PATCH_RAD::Int,
    RAD_DIM::Int,
    INTERP_ORDER::Int,
    Z::AbstractMatrix{ComplexF64},
    Gram::AbstractVector{ComplexF64}
)
    N, P = size(Z) # N = mtheta * nzeta, P = num_modes
    dθdζ = 4π^2 / N
    Zt = Matrix{ComplexF64}(transpose(Z))  # [P × M] for contiguous column access

    # Take a view of the corresponding block of the K and G matrices
    col_index = (source isa PlasmaGeometry3D ? 1 : 2)
    row_index = (observer isa PlasmaGeometry3D ? 1 : 2)
    K_block = view(K, ((row_index-1)*P+1):(row_index*P), ((col_index-1)*P+1):(col_index*P))
    G_block = view(G, ((row_index-1)*P+1):(row_index*P), :)

    # G only needed for plasma as source term (RHS of eqs. 26/27 in Chance 1997)
    populate_greenfunction = source isa PlasmaGeometry3D

    # Initialize quadrature data
    # This allows the code to run at lower resolution without erroring out, but will warn the user.
    if PATCH_RAD > (min(source.mtheta, source.nzeta) - 1) ÷ 2
        @warn "PATCH_RAD is greater than the number of points in the toroidal or poloidal direction, which is not supported. Setting PATCH_RAD to $((min(source.mtheta, source.nzeta) - 1) ÷ 2). Be sure to check if outputs are converged."
        PATCH_RAD = (min(source.mtheta, source.nzeta) - 1) ÷ 2
    end
    quad_data = get_singular_quadrature(PATCH_RAD, RAD_DIM, INTERP_ORDER)
    (; PATCH_DIM, PATCH_RAD, ANG_DIM, RAD_DIM, Ppou, Gpou, P2G) = quad_data
    @assert observer.mtheta ≥ PATCH_DIM "Must have observer.mtheta ≥ PATCH_DIM, got observer.mtheta=$(observer.mtheta), PATCH_DIM=$PATCH_DIM"
    @assert observer.nzeta ≥ PATCH_DIM "Must have observer.nzeta ≥ PATCH_DIM, got observer.nzeta=$(observer.nzeta), PATCH_DIM=$PATCH_DIM"

    # Buffers for the projection: column idx_obs holds (kernel row idx_obs) · Z
    KZt = zeros(ComplexF64, P, N)
    GZt = zeros(ComplexF64, P, N)

    # Allocate thread-local workspaces (one per thread)
    max_threadid = Threads.maxthreadid()
    workspaces = [KernelWorkspace(PATCH_DIM, RAD_DIM, ANG_DIM) for _ in 1:max_threadid]
    proj_k_all = [zeros(ComplexF64, P) for _ in 1:max_threadid]
    proj_g_all = [zeros(ComplexF64, P) for _ in 1:max_threadid]
    is_patch_all = [falses(N) for _ in 1:max_threadid]

    # Parallel loop through observer points
    Threads.@threads for idx_obs in 1:N
        # Get thread-local workspace
        ws = workspaces[Threads.threadid()]
        (; r_patch, dr_dθ_patch, dr_dζ_patch, r_polar, dr_dθ_polar, dr_dζ_polar,
            n_polar, M_polar_single, M_polar_double, M_grid_single_flat, M_grid_double_flat) = ws
        proj_k = proj_k_all[Threads.threadid()]
        proj_g = proj_g_all[Threads.threadid()]
        is_patch = is_patch_all[Threads.threadid()]

        fill!(proj_k, 0.0)
        fill!(proj_g, 0.0)
        fill!(is_patch, false)

        # Convert linear index to 2D indices
        i_obs = mod1(idx_obs, observer.mtheta)
        j_obs = (idx_obs - 1) ÷ observer.mtheta + 1
        @inbounds ox = observer.r[idx_obs, 1]
        @inbounds oy = observer.r[idx_obs, 2]
        @inbounds oz = observer.r[idx_obs, 3]

        # Mark patch source indices so the far-field loop can skip them
        @inbounds for jj in 1:PATCH_DIM, ii in 1:PATCH_DIM
            idx_pol = periodic_wrap(i_obs - PATCH_RAD + ii - 1, source.mtheta)
            idx_tor = periodic_wrap(j_obs - PATCH_RAD + jj - 1, source.nzeta)
            is_patch[idx_pol+source.mtheta*(idx_tor-1)] = true
        end

        # ============================================================
        # FAR FIELD: Trapezoidal rule for nonsingular source points
        # Note: kernels return zero for r_src = r_obs
        # ============================================================
        @inbounds for idx_src in 1:N
            is_patch[idx_src] && continue

            sx = source.r[idx_src, 1]
            sy = source.r[idx_src, 2]
            sz = source.r[idx_src, 3]
            nx = source.normal[idx_src, 1]
            ny = source.normal[idx_src, 2]
            nz = source.normal[idx_src, 3]

            w_double = laplace_double_layer(ox, oy, oz, sx, sy, sz, nx, ny, nz) * dθdζ
            _accum_row!(proj_k, w_double, Zt, idx_src)

            if populate_greenfunction
                w_single = laplace_single_layer(ox, oy, oz, sx, sy, sz) * dθdζ
                _accum_row!(proj_g, w_single, Zt, idx_src)
            end
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
            # Evaluate kernels and apply quadrature weights: area element × POU, where POU contains rdrdθ already
            rsx = r_polar[ir, ia, 1]
            rsy = r_polar[ir, ia, 2]
            rsz = r_polar[ir, ia, 3]
            nsx = n_polar[ir, ia, 1]
            nsy = n_polar[ir, ia, 2]
            nsz = n_polar[ir, ia, 3]
            M_polar_single[ir, ia] = laplace_single_layer(ox, oy, oz, rsx, rsy, rsz) * Ppou[ir, ia] * dθdζ
            M_polar_double[ir, ia] = laplace_double_layer(ox, oy, oz, rsx, rsy, rsz, nsx, nsy, nsz) * Ppou[ir, ia] * dθdζ
        end

        # Distribute polar singular corrections back to Cartesian grid using sparse matrix
        # grid = P2G * polar (maps Npolar → Ngrid)
        mul!(M_grid_single_flat, P2G, vec(M_polar_single))
        mul!(M_grid_double_flat, P2G, vec(M_polar_double))
        M_grid_single = reshape(M_grid_single_flat, PATCH_DIM, PATCH_DIM)
        M_grid_double = reshape(M_grid_double_flat, PATCH_DIM, PATCH_DIM)

        # POU correction: read back far-field trapezoidal values instead of re-evaluating kernels.
        # trap + M_grid + trap*Gpou = trap*(1+Gpou) + M_grid = trap*(1-χ) + M_grid
        @inbounds for j in 1:PATCH_DIM, i in 1:PATCH_DIM
            # Map back to global indices
            idx_pol = periodic_wrap(i_obs - PATCH_RAD + i - 1, source.mtheta)
            idx_tor = periodic_wrap(j_obs - PATCH_RAD + j - 1, source.nzeta)
            idx_src = idx_pol + source.mtheta * (idx_tor - 1)

            sx = source.r[idx_src, 1]
            sy = source.r[idx_src, 2]
            sz = source.r[idx_src, 3]
            nx = source.normal[idx_src, 1]
            ny = source.normal[idx_src, 2]
            nz = source.normal[idx_src, 3]

            far_double = laplace_double_layer(ox, oy, oz, sx, sy, sz, nx, ny, nz) * (1.0 + Gpou[i, j]) * dθdζ
            _accum_row!(proj_k, M_grid_double[i, j] + far_double, Zt, idx_src)

            # Apply near + far contributions
            if populate_greenfunction
                far_single = laplace_single_layer(ox, oy, oz, sx, sy, sz) * (1.0 + Gpou[i, j]) * dθdζ
                _accum_row!(proj_g, M_grid_single[i, j] + far_single, Zt, idx_src)
            end
        end

        # ── Write projected column to buffer (each idx_obs owns its column) ──
        @inbounds for p in 1:P
            KZt[p, idx_obs] = proj_k[p]
        end
        if populate_greenfunction
            @inbounds for p in 1:P
                GZt[p, idx_obs] = proj_g[p]
            end
        end
    end

    # Use the same normalization as in the 2D kernel so we can just add I to the diagonal
    # This makes the grri logic identical to the 2D kernel.
    mul!(K_block, Z', transpose(KZt))
    K_block ./= 2π
    if populate_greenfunction
        mul!(G_block, Z', transpose(GZt))
        G_block ./= 2π
    end

    # Add the term that comes from the volume integral of Green's identity.
    if typeof(source) == typeof(observer)
        @inbounds for p in 1:P
            K_block[p, p] += Gram[p]
        end
    end
end

"""
    kernel!(Kc, Gc, observer, source, params::KernelParams3D, Z, Gram)

Public 3D kernel entry point. Forwards to:

`compute_3D_kernel_matrices!(Kc, Gc, observer, source, params.PATCH_RAD, params.RAD_DIM, params.INTERP_ORDER, Z, Gram)`.
"""
function kernel!(
    Kc::AbstractMatrix{ComplexF64},
    Gc::AbstractMatrix{ComplexF64},
    observer::Union{PlasmaGeometry3D,WallGeometry3D},
    source::Union{PlasmaGeometry3D,WallGeometry3D},
    params::KernelParams3D,
    Z::AbstractMatrix{ComplexF64},
    Gram::AbstractVector{ComplexF64}
)
    return compute_3D_kernel_matrices!(
        Kc,
        Gc,
        observer,
        source,
        params.PATCH_RAD,
        params.RAD_DIM,
        params.INTERP_ORDER,
        Z,
        Gram
    )
end
