const INV_4PI = 1.0 / (4π)

"""
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
end

# Global cache for quadrature data (initialized on first use)
const SINGULAR_QUAD_CACHE = Ref{Union{Nothing,SingularQuadratureData}}(nothing)

"""
    init_singular_quadrature(PATCH_RAD::Int, RAD_DIM::Int, INTERP_ORDER::Int)

Initialize quadrature points, weights, partition-of-unity functions, and
interpolation matrices for singular correction. Follows BIEST's approach.

Conversion references:

  - Quadrature/Patch setup adapted from biest/singular_correction.hpp

# Arguments

  - `PATCH_RAD::Int`: Number of points adjacent to source point to treat as singular
  - `RAD_DIM::Int`: Radial quadrature order
  - `INTERP_ORDER::Int`: Lagrange interpolation order

# Returns

  - `SingularQuadratureData`: Precomputed quadrature data
"""
function init_singular_quadrature(PATCH_RAD::Int, RAD_DIM::Int, INTERP_ORDER::Int)

    # Total size of square patch extracted around singular point (odd number: 2*PATCH_DIM0+1)
    PATCH_DIM = 2 * PATCH_RAD + 1
    @assert INTERP_ORDER <= PATCH_DIM "Must have INTERP_ORDER <= PATCH_DIM"
    # Number of angular quadrature nodes in polar coordinates (uniformly distributed around circle)
    ANG_DIM = 2 * RAD_DIM

    # Setup radial quadrature
    qx_raw, qw_raw = FastGaussQuadrature.gausslegendre(RAD_DIM) # points on [-1,1]
    qx = (qx_raw .+ 1) ./ 2  # Map [-1, 1] to [0, 1]
    qw = qw_raw ./ 2         # Adjust weights for interval change

    # Partition of unity function, exp(-36 * r^p) where p depends on PATCH_DIM
    pou_power = PATCH_DIM > 45 ? 10 : (PATCH_DIM > 20 ? 8 : 6)
    pou(r) = r ≥ 1.0 ? 0.0 : exp(-36.0 * r^pou_power)

    # Partition of Unity on Cartesian grid
    Gpou = zeros(PATCH_DIM, PATCH_DIM)
    for i in 1:PATCH_DIM, j in 1:PATCH_DIM
        r = sqrt((i - 1 - PATCH_RAD)^2 + (j - 1 - PATCH_RAD)^2) / PATCH_RAD
        Gpou[i, j] = -pou(r)
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

    return SingularQuadratureData(qx, qw, Gpou, Ppou, P2G, PATCH_DIM, PATCH_RAD, ANG_DIM, RAD_DIM)
end

"""
    get_singular_quadrature(PATCH_RAD::Int, RAD_DIM::Int, INTERP_ORDER::Int)

Get cached singular quadrature data, initializing if necessary.

Conversion references:

  - Follows caching pattern used around FieldPeriodBIOp setup in biest/boundary_integ_op.hpp
"""
function get_singular_quadrature(PATCH_RAD::Int, RAD_DIM::Int, INTERP_ORDER::Int)
    if isnothing(SINGULAR_QUAD_CACHE[])
        SINGULAR_QUAD_CACHE[] = init_singular_quadrature(PATCH_RAD, RAD_DIM, INTERP_ORDER)
    end
    return SINGULAR_QUAD_CACHE[]
end

"""
    laplace_single_layer(x_obs::Vector{Float64}, x_src::Vector{Float64}) -> Float64

Evaluate the Laplace single-layer (FxU) kernel between two 3D points. Returns
0.0 if the observation point coincides with the source point to avoid singularity.

The single-layer kernel φ is the fundamental solution to Laplace's equation:

```
φ(x_obs, x_src) = 1 / (4π |x_obs - x_src|)
```

# Arguments

  - `x_obs::Vector{Float64}`: Observation point (3D Cartesian coordinates)
  - `x_src::Vector{Float64}`: Source point (3D Cartesian coordinates)

# Returns

  - `Float64`: Kernel value φ(x_obs, x_src)
"""
function laplace_single_layer(x_obs::Vector{Float64}, x_src::Vector{Float64})
    # Single-layer kernel: 1/(4π r)
    @inbounds begin
        dx = x_obs[1] - x_src[1]
        dy = x_obs[2] - x_src[2]
        dz = x_obs[3] - x_src[3]
    end
    r2 = dx*dx + dy*dy + dz*dz
    r2 < 1e-30 && return 0.0
    return INV_4PI * inv(sqrt(r2))
end

"""
    laplace_double_layer(x_obs::Vector{Float64}, x_src::Vector{Float64}, n_src::Vector{Float64}) -> Float64

Evaluate the Laplace double-layer (DxU) kernel between a point and a surface element. Returns
0.0 if the observation point coincides with the source point to avoid singularity. Allocation-free
scalar arithmetic is used for maximum performance.

The double-layer kernel K is the normal derivative of the fundamental solution:

```
K(x_obs, x_src, n_src) = ∇_{x_src} φ · n̂_src
                       = -1/(4π) * (x_obs - x_src) · n̂_src / |x_obs - x_src|³
```

# Arguments

  - `x_obs::Vector{Float64}`: Observation point (3D Cartesian coordinates)
  - `x_src::Vector{Float64}`: Source point on surface (3D Cartesian coordinates)
  - `n_src::Vector{Float64}`: Outward UNIT normal at source point (must be normalized!)

# Returns

  - `Float64`: Kernel value K(x_obs, x_src, n_src)
"""
function laplace_double_layer(x_obs::Vector{Float64}, x_src::Vector{Float64}, n_src::Vector{Float64})
    # Double-layer kernel: -1/(4π) * (r·n) / r³
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
    return -(dx*nx + dy*ny + dz*nz) * (r3inv * INV_4PI)
end

"""
    extract_patch(data, Nt, Np, t0, p0, PATCH_DIM)

Extract a PATCH_DIM × PATCH_DIM patch of data centered at (t0, p0) with periodic wrapping.

# Arguments

  - `data`: Source data array (can be coordinates, normals, or area elements)
  - `Nt, Np`: Grid dimensions (toroidal, poloidal)
  - `t0, p0`: Center indices (1-based)
  - `PATCH_DIM`: Patch size (must be odd)

# Returns

  - `patch`: Extracted patch of data around the singular point (PATCH_DIM × PATCH_DIM × dof)
"""
function extract_patch(data::Matrix{Float64}, idx_pol_center::Int, idx_tor_center::Int, npol::Int, ntor::Int, PATCH_DIM::Int)

    PATCH_RAD = (PATCH_DIM - 1) ÷ 2
    dof = size(data, 2)  # Number of components (3 for coords, 1 for scalars)
    patch = zeros(PATCH_DIM, PATCH_DIM, dof)
    for i in 1:PATCH_DIM, j in 1:PATCH_DIM
        # Enforce periodicity
        idx_pol = mod1(idx_pol_center - PATCH_RAD + i - 1, npol)
        idx_tor = mod1(idx_tor_center - PATCH_RAD + j - 1, ntor)
        # Copy data to the patch (for each dof)
        @views patch[i, j, :] .= data[idx_pol+npol*(idx_tor-1), :]
    end
    return patch # (PATCH_DIM, PATCH_DIM, dof)
end

"""
    interpolate_to_polar(patch, quad_data)

Interpolate Cartesian patch data to polar quadrature points using sparse matrix multiply.

# Arguments

  - `patch`: Patch data (PATCH_DIM × PATCH_DIM × dof)
  - `quad_data`: Precomputed quadrature data

# Returns

  - `polar_data`: Interpolated data at polar points (RAD_DIM × ANG_DIM × dof)
"""
function interpolate_to_polar(patch::Array{Float64,3}, quad_data::SingularQuadratureData)

    (; P2G, RAD_DIM, ANG_DIM) = quad_data
    dof = size(patch, 3)

    # Flatten patch to (Ngrid × dof), apply P2G' to get (Npolar × dof)
    patch_flat = reshape(patch, :, dof)
    polar_flat = P2G' * patch_flat
    return reshape(polar_flat, RAD_DIM, ANG_DIM, dof)
end

"""
    compute_polar_normal(dr_dθ_polar, dr_dζ_polar)

Compute normal vector (= ∂r/∂θ × ∂r/∂ζ) at polar quadrature points from interpolated tangent vectors.

# Arguments

  - `dr_dθ_polar`: Interpolated ∂r/∂θ at polar points (RAD_DIM × ANG_DIM × 3)
  - `dr_dζ_polar`: Interpolated ∂r/∂ζ at polar points (RAD_DIM × ANG_DIM × 3)

# Returns

  - `n_polar`: Unit normal vector at each polar point (RAD_DIM × ANG_DIM × 3)
"""
function compute_polar_normal(dr_dθ::Array{Float64,3}, dr_dζ::Array{Float64,3})

    n_polar = similar(dr_dθ)
    @views for ia in axes(dr_dθ, 2), ir in axes(dr_dθ, 1)
        n_polar[ir, ia, :] .= cross(dr_dθ[ir, ia, :], dr_dζ[ir, ia, :])
    end
    return n_polar
end

"""
    compute_3D_kernel_matrix!(grad_greenfunction, greenfunction, observer, source; PATCH_RAD=3, RAD_DIM=12, INTERP_ORDER=6)

Compute boundary integral kernel matrices for 3D geometries with the singular correction
algorithm from Malhotra et al. 2019.

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
  - `PATCH_RAD`: Number of points adjacent to source point to treat as singular (default 3)

      + Total patch size in # of gridpoints = (2 * PATCH_RAD + 1) x (2 * PATCH_RAD + 1)
  - `RAD_DIM`: Polar radial quadrature order (default 12). Angular order = 2 * RAD_DIM
  - `INTERP_ORDER`: Lagrange interpolation order (default 6)
"""
function compute_3D_kernel_matrix!(
    grad_greenfunction::Matrix{Float64},
    greenfunction::Matrix{Float64},
    observer::Union{PlasmaGeometry3D,WallGeometry3D},
    source::Union{PlasmaGeometry3D,WallGeometry3D};
    PATCH_RAD::Int=3,
    RAD_DIM::Int=15,
    INTERP_ORDER::Int=6
)

    # Zero out matrices
    fill!(grad_greenfunction, 0.0)
    fill!(greenfunction, 0.0)

    # Initialize quadrature data (cached)
    quad_data = get_singular_quadrature(PATCH_RAD, RAD_DIM, INTERP_ORDER)
    (; PATCH_DIM, PATCH_RAD, ANG_DIM, RAD_DIM, Ppou, Gpou, P2G) = quad_data
    @assert observer.mtheta ≥ PATCH_DIM
    @assert observer.nzeta ≥ PATCH_DIM
    dθdζ = (2π / observer.mtheta) * (2π / observer.nzeta)

    # Loop through observer points
    for j_obs in 1:observer.nzeta, i_obs in 1:observer.mtheta
        idx_obs = i_obs + (j_obs - 1) * observer.mtheta
        r_obs = observer.r[idx_obs, :]

        # ============================================================
        # FAR FIELD: Trapezoidal rule for nonsingular source points
        # Note: kernels return zero for r_src = r_obs
        # ============================================================
        for j_src in 1:source.nzeta, i_src in 1:source.mtheta
            # Evaluate kernels at grid points
            idx_src = i_src + (j_src - 1) * source.mtheta
            K_single = laplace_single_layer(r_obs, source.r[idx_src, :])
            K_double = laplace_double_layer(r_obs, source.r[idx_src, :], source.normal[idx_src, :])

            # Apply weights (periodic trapezoidal rule = constant weights)
            greenfunction[idx_obs, idx_src] = K_single * dθdζ
            grad_greenfunction[idx_obs, idx_src] = K_double * dθdζ
        end

        # ============================================================
        # NEAR FIELD: Polar quadrature with singular correction
        # ============================================================
        # Extract patches of source data around the singular point (size = PATCH_DIM x PATCH_DIM x dof)
        r_patch = extract_patch(source.r, i_obs, j_obs, source.mtheta, source.nzeta, PATCH_DIM)
        dr_dθ_patch = extract_patch(source.dr_dθ, i_obs, j_obs, source.mtheta, source.nzeta, PATCH_DIM)
        dr_dζ_patch = extract_patch(source.dr_dζ, i_obs, j_obs, source.mtheta, source.nzeta, PATCH_DIM)

        # Interpolate coordinates and tangent vectors to polar quadrature points
        r_polar = interpolate_to_polar(r_patch, quad_data)
        dr_dθ_polar = interpolate_to_polar(dr_dθ_patch, quad_data)
        dr_dζ_polar = interpolate_to_polar(dr_dζ_patch, quad_data)

        # Compute normal vectors at polar points from interpolated tangent vectors
        n_polar = compute_polar_normal(dr_dθ_polar, dr_dζ_polar)

        # Evaluate kernels at polar points with POU weighting
        M_polar_single = zeros(RAD_DIM, ANG_DIM)
        M_polar_double = zeros(RAD_DIM, ANG_DIM)
        for ia in 1:ANG_DIM, ir in 1:RAD_DIM
            # Evaluate kernels using recomputed normal
            r_src, n_src = r_polar[ir, ia, :], n_polar[ir, ia, :]
            K_single = laplace_single_layer(r_obs, r_src)
            K_double = laplace_double_layer(r_obs, r_src, n_src)

            # Apply quadrature weights: area element × POU, where POU contains rdrdθ already
            M_polar_single[ir, ia] = K_single * Ppou[ir, ia] * dθdζ
            M_polar_double[ir, ia] = K_double * Ppou[ir, ia] * dθdζ
        end

        # Distribute polar singular corrections back to Cartesian grid using sparse matrix
        # grid = P2G * polar (maps Npolar → Ngrid)
        M_grid_single = reshape(P2G * vec(M_polar_single), PATCH_DIM, PATCH_DIM)
        M_grid_double = reshape(P2G * vec(M_polar_double), PATCH_DIM, PATCH_DIM)

        # Compute remaining far-field POU contribution and near-field polar quadrature result
        # We include this region in the far-field trapezoidal rule, so use Gpou = -χ here to get 1-χ
        for j in 1:PATCH_DIM, i in 1:PATCH_DIM
            # Map back to global indices
            idx_pol = mod1(i_obs - PATCH_RAD + i - 1, source.mtheta)
            idx_tor = mod1(j_obs - PATCH_RAD + j - 1, source.nzeta)
            idx_src = idx_pol + source.mtheta * (idx_tor - 1)

            # Remainder of far-field contribution on the singular grid: Gpou = -χ
            r_src, n_src = source.r[idx_src, :], source.normal[idx_src, :]
            far_single = laplace_single_layer(r_obs, r_src) * Gpou[i, j] * dθdζ
            far_double = laplace_double_layer(r_obs, r_src, n_src) * Gpou[i, j] * dθdζ

            # Apply near + far contributions
            greenfunction[idx_obs, idx_src] += M_grid_single[i, j] + far_single
            grad_greenfunction[idx_obs, idx_src] += M_grid_double[i, j] + far_double
        end
    end

    # TODO: Don't delete this yet - signs might change depending on convention. I think it might be -1 for wall,
    # since we calculate n = dr_dθ × dr_dζ which points inward for a toroidal surface. Should add in normal
    # orient for this later for generalization.
    # Account for normal direction pointing out of vacuum integration region in 𝒦ⁿ ⋅ dS
    # Negative for plasma since dS = ∇ψ J dθdζ and ∇ψ points outward but outward normal is inward
    # @views grad_greenfunction .*= (source isa PlasmaGeometry3D ? -1 : 1)
end
