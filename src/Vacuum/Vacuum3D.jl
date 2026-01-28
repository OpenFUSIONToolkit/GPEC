"""
Precomputed data for singular correction quadrature following BIEST approach.
Initialized once on first use.

## Fields

    - `qx::Vector{Float64}`: Radial quadrature points in [0,1]
    - `qw::Vector{Float64}`: Radial quadrature weights
    - `Gpou::Matrix{Float64}`: Partition of unity on Cartesian grid (PATCH_DIM × PATCH_DIM)
    - `Ppou::Matrix{Float64}`: Partition of unity on polar grid (RAD_DIM × ANG_DIM)
    - `M_G2P::Array{Float64, 4}`: 2D tensor-product Lagrange basis function values for interpolating from the Cartesian patch grid to polar quadrature points (RAD_DIM × ANG_DIM × INTERP_ORDER × INTERP_ORDER)
    - `I_G2P::Array{Int, 3}`: Indices of lower-left corner of INTERP_ORDER × INTERP_ORDER stencil in PATCH_DIM × PATCH_DIM grid
    - `PATCH_DIM::Int`: Patch dimension (odd integer)
    - `PATCH_RAD::Int`: Patch radius (number of points adjacent to source point treated as singular)
    - `INTERP_ORDER::Int`: Interpolation order
    - `ANG_DIM::Int`: Number of angular quadrature points
    - `RAD_DIM::Int`: Number of radial quadrature points
"""
struct SingularQuadratureData
    qx::Vector{Float64}
    qw::Vector{Float64}
    Gpou::Matrix{Float64}
    Ppou::Matrix{Float64}
    M_G2P::Array{Float64,4}
    I_G2P::Array{Int,3}
    PATCH_DIM::Int
    PATCH_RAD::Int
    INTERP_ORDER::Int
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

    # Setup radial quadrature (Gauss-Legendre transformed to [0,1])
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
    dt = 2π / ANG_DIM
    for j in 1:ANG_DIM, i in 1:RAD_DIM
        dr = qw[i] * PATCH_RAD
        rdt = qx[i] * PATCH_RAD * dt;
        Ppou[i, j] = pou(qx[i]) * dr * rdt
    end

    # Spacing between Lagrange interpolation nodes in [0,1] for INTERP_ORDER-point stencil
    h = 1.0 / (INTERP_ORDER - 1)
    # Compute 2D tensor-product Lagrange basis function at (x0, x1) in local stencil coordinates
    # for basis node (i0, i1) on uniform grid with spacing h
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

    # Build Lagrange interpolation matrix from Cartesian grid to polar (G2P) points
    M_G2P = zeros(RAD_DIM, ANG_DIM, INTERP_ORDER, INTERP_ORDER)
    I_G2P = zeros(Int, RAD_DIM, ANG_DIM, 2)  # y0,y1 indices of lower-left corner of stencil in PATCH_DIM × PATCH_DIM grid
    Δθ = 2π / ANG_DIM
    for ir in 1:RAD_DIM, ia in 1:ANG_DIM
        # Map polar node to unit square: x0, x1 ∈ [0,1] × [0,1]
        x0 = 0.5 + 0.5 * qx[ir] * cos(Δθ * (ia - 1))
        x1 = 0.5 + 0.5 * qx[ir] * sin(Δθ * (ia - 1))

        # Lower-left corner indices of INTERP_ORDER × INTERP_ORDER stencil centered on (x0,x1)
        # C++: (Integer)(x0 * (PATCH_DIM - 1) - (INTERP_ORDER - 1) / 2) -- truncate AFTER subtraction
        y0 = clamp(trunc(Int, x0 * (PATCH_DIM - 1) - (INTERP_ORDER - 1) ÷ 2), 0, PATCH_DIM - INTERP_ORDER)
        y1 = clamp(trunc(Int, x1 * (PATCH_DIM - 1) - (INTERP_ORDER - 1) ÷ 2), 0, PATCH_DIM - INTERP_ORDER)

        # Local coordinates within INTERP_ORDER×INTERP_ORDER stencil, normalized to [0,1]
        z0 = (x0 * (PATCH_DIM - 1) - y0) * h
        z1 = (x1 * (PATCH_DIM - 1) - y1) * h

        # Indices of lower-left corner of INTERP_ORDER × INTERP_ORDER stencil in PATCH_DIM × PATCH_DIM grid
        I_G2P[ir, ia, :] .= [y0, y1]

        # 2D tensor-product Lagrange basis function values on the polar grid
        for j0 in 1:INTERP_ORDER, j1 in 1:INTERP_ORDER
            M_G2P[ir, ia, j0, j1] = lagrange_interp(z0, z1, j0 - 1, j1 - 1)
        end
    end

    return SingularQuadratureData(qx, qw, Gpou, Ppou, M_G2P, I_G2P, PATCH_DIM, PATCH_RAD, INTERP_ORDER, ANG_DIM, RAD_DIM)
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

Evaluate the Laplace single-layer (FxU) kernel between two 3D points.

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
function laplace_single_layer(x_obs::Vector{Float64}, x_src::Vector{Float64})::Float64

    # Single-layer kernel: 1/(4π r)
    r = norm(x_obs - x_src)
    r < 1e-30 && return 0.0
    return 1.0 / (4π * r)
end

"""
    laplace_double_layer(x_obs::Vector{Float64}, x_src::Vector{Float64}, n_src::Vector{Float64}) -> Float64

Evaluate the Laplace double-layer (DxU) kernel between a point and a surface element.

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
function laplace_double_layer(x_obs::Vector{Float64}, x_src::Vector{Float64}, n_src::Vector{Float64})::Float64

    # Double-layer kernel: -1/(4π) * (r·n) / r³
    r = norm(x_obs - x_src)
    r < 1e-30 && return 0.0
    return -dot(x_obs - x_src, n_src) / (4π * r^3)
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

  - `patch`: Extracted patch array

Conversion references:

  - Mirrors patch extraction used inside SingularCorrection::SetPatch in biest/singular_correction.hpp
"""
function extract_patch(data::Matrix{Float64}, idx_pol_center::Int, idx_tor_center::Int, npol::Int, ntor::Int, PATCH_DIM::Int)

    PATCH_RAD = (PATCH_DIM - 1) ÷ 2
    dof = size(data, 2)  # Number of components (3 for coords, 1 for scalars)
    patch = zeros(PATCH_DIM, PATCH_DIM, dof)
    for i in 1:PATCH_DIM, j in 1:PATCH_DIM
        # Enforce periodicity
        idx_pol = mod1(idx_pol_center - PATCH_RAD + i - 1, npol)
        idx_tor = mod1(idx_tor_center - PATCH_RAD + j - 1, ntor)
        # Copy data to the patch
        @views patch[i, j, :] .= data[idx_pol+npol*(idx_tor-1), :]
    end
    return patch # (PATCH_DIM, PATCH_DIM, dof)
end

"""
    interpolate_to_polar(patch, quad_data)

Interpolate Cartesian patch data to polar quadrature points using precomputed matrix.

# Arguments

  - `patch`: Patch data (PATCH_DIM × PATCH_DIM × dof)
  - `quad_data`: Precomputed quadrature data

# Returns

  - `polar_data`: Interpolated data at polar points (RAD_DIM × ANG_DIM × dof)

Conversion references:

  - Grid→polar interpolation follows M_G2P application in biest/singular_correction.hpp
"""
function interpolate_to_polar(patch::Array{Float64,3}, quad_data::SingularQuadratureData)

    (; M_G2P, I_G2P, INTERP_ORDER, RAD_DIM, ANG_DIM) = quad_data
    dof = size(patch, 3)
    polar_data = zeros(RAD_DIM, ANG_DIM, dof)
    for ir in 1:RAD_DIM, ia in 1:ANG_DIM
        ycorner = I_G2P[ir, ia, :]
        for i0 in 1:INTERP_ORDER, i1 in 1:INTERP_ORDER
            polar_data[ir, ia, :] .+= M_G2P[ir, ia, i0, i1] .* patch[ycorner[1]+i0, ycorner[2]+i1, :]
        end
    end
    return polar_data
end

"""
    compute_dA_and_normal_polar(dr_dθ_polar, dr_dζ_polar)

Compute area elements at polar quadrature points from interpolated tangent vectors.

The area element is |∂r/∂θ × ∂r/∂ζ| dθ dζ, computed from the cross product of
the interpolated tangent vectors.

# Arguments

  - `dr_dθ_polar`: Interpolated ∂r/∂θ at polar points (RAD_DIM × ANG_DIM × 3)
  - `dr_dζ_polar`: Interpolated ∂r/∂ζ at polar points (RAD_DIM × ANG_DIM × 3)

# Returns

  - `dA_polar`: Area element at each polar point (RAD_DIM × ANG_DIM)
  - `n_polar`: Unit normal vector at each polar point (RAD_DIM × ANG_DIM × 3)
"""
function compute_dA_and_normal_polar(dr_dθ::Array{Float64,3}, dr_dζ::Array{Float64,3})

    RAD_DIM, ANG_DIM, _ = size(dr_dθ)
    dA_polar = zeros(RAD_DIM, ANG_DIM)
    n_polar = zeros(RAD_DIM, ANG_DIM, 3)
    for ia in 1:ANG_DIM, ir in 1:RAD_DIM
        # Extract tangent vectors at this polar point
        t_θ = @view dr_dθ[ir, ia, :]
        t_ζ = @view dr_dζ[ir, ia, :]

        # Cross product: ∂r/∂θ × ∂r/∂ζ
        cross_prod = cross(t_θ, t_ζ)

        # Magnitude gives area element
        dA_polar[ir, ia] = norm(cross_prod)

        # Unit normal (handle zero jacobian gracefully)
        if dA_polar[ir, ia] > 1e-30
            n_polar[ir, ia, :] .= cross_prod ./ dA_polar[ir, ia]
        end
    end

    return dA_polar, n_polar
end

"""
    compute_3D_kernel_matrix!(
        grad_greenfunction, greenfunction,
        observer, source;
        PATCH_DIM=7, RAD_DIM=12
    )

Compute boundary integral kernel matrices for 3D geometries with singular correction.

Uses BIEST-style approach:

  - Far regions: Rectangle rule with uniform weights (1/N)
  - Singular regions: Polar quadrature with partition-of-unity blending

# Arguments

  - `grad_greenfunction`: Double-layer kernel matrix (Nobs × Nsrc)
  - `greenfunction`: Single-layer kernel matrix (Nobs × Nsrc)
  - `observer`: Observer geometry (PlasmaGeometry3D)
  - `source`: Source geometry (PlasmaGeometry3D)
  - `PATCH_RAD`: Number of points adjacent to source point to treat as singular (default 3)
  - `RAD_DIM`: Radial quadrature order (default 12)

Conversion references:

  - Far/near split and Eval flow adapted from FieldPeriodBIOp and BoundaryIntegralOp in biest/boundary_integ_op.hpp
  - Rectangle-rule weights mirror SurfNormalAreaElem/EvalSurfInteg in biest/surface_op.txx
"""
function compute_3D_kernel_matrix!(
    grad_greenfunction::Matrix{Float64},
    greenfunction::Matrix{Float64},
    observer::PlasmaGeometry3D,
    source::PlasmaGeometry3D;
    PATCH_RAD::Int=3,
    RAD_DIM::Int=15,
    INTERP_ORDER::Int=6
)

    # Zero out matrices
    fill!(grad_greenfunction, 0.0)
    fill!(greenfunction, 0.0)

    # Initialize quadrature data (cached)
    quad_data = get_singular_quadrature(PATCH_RAD, RAD_DIM, INTERP_ORDER)
    (; PATCH_DIM, PATCH_RAD, INTERP_ORDER, ANG_DIM, RAD_DIM, qw, Ppou, Gpou, M_G2P, I_G2P) = quad_data
    @assert observer.mtheta ≥ PATCH_DIM
    @assert observer.nzeta ≥ PATCH_DIM

    # Loop through observer points
    for j_obs in 1:observer.nzeta, i_obs in 1:observer.mtheta
        idx_obs = i_obs + (j_obs - 1) * observer.mtheta
        r_obs = observer.r[idx_obs, :]

        # ============================================
        # FAR FIELD: Trapezoidal rule for nonsingular source points
        # Note: kernels return zero for r_src = r_obs
        # ============================================
        for j_src in 1:source.nzeta, i_src in 1:source.mtheta
            # Evaluate kernels at grid points
            idx_src = i_src + (j_src - 1) * source.mtheta
            K_single = laplace_single_layer(r_obs, source.r[idx_src, :])
            K_double = laplace_double_layer(r_obs, source.r[idx_src, :], source.normal[idx_src, :])

            # Apply area element (periodic trapezoidal rule: w = dA = J∇ψdθdζ)
            greenfunction[idx_obs, idx_src] = K_single * (4π^2 / (observer.mtheta * observer.nzeta)) # * source.dA[idx_src]
            grad_greenfunction[idx_obs, idx_src] = K_double * source.dA[idx_src]
        end

        # ============================================
        # NEAR FIELD: Polar quadrature with singular correction
        # ============================================
        # Extract patches of source data around the singular point (size = PATCH_DIM x PATCH_DIM x dof)
        r_patch = extract_patch(source.r, i_obs, j_obs, source.mtheta, source.nzeta, PATCH_DIM)
        dr_dθ_patch = extract_patch(source.dr_dθ, i_obs, j_obs, source.mtheta, source.nzeta, PATCH_DIM)
        dr_dζ_patch = extract_patch(source.dr_dζ, i_obs, j_obs, source.mtheta, source.nzeta, PATCH_DIM)

        # Interpolate coordinates and tangent vectors to polar quadrature points
        r_polar = interpolate_to_polar(r_patch, quad_data)
        dr_dθ_polar = interpolate_to_polar(dr_dθ_patch, quad_data)
        dr_dζ_polar = interpolate_to_polar(dr_dζ_patch, quad_data)

        # Compute area elements and unit normals at polar points from interpolated tangent vectors
        dA_polar, n_polar = compute_dA_and_normal_polar(dr_dθ_polar, dr_dζ_polar)

        # Evaluate kernels at polar points with POU weighting
        M_polar_single = zeros(RAD_DIM, ANG_DIM)
        M_polar_double = zeros(RAD_DIM, ANG_DIM)
        for ia in 1:ANG_DIM, ir in 1:RAD_DIM
            # Evaluate kernels using recomputed normal
            r_src, n_src = r_polar[ir, ia, :], n_polar[ir, ia, :]
            K_single = laplace_single_layer(r_obs, r_src)
            K_double = laplace_double_layer(r_obs, r_src, n_src)

            # Apply quadrature weights: area element × POU, where POU contains rdrdθ already
            wt = dA_polar[ir, ia] * Ppou[ir, ia]
            M_polar_single[ir, ia] = K_single * Ppou[ir, ia] * (4π^2 / (observer.mtheta * observer.nzeta))
            M_polar_double[ir, ia] = K_double * wt
        end

        # Distribute corrections back to Cartesian grid using interpolation matrix
        # Correction at grid point = sum over polar points of (kernel_value * interp_weight)
        M_grid_single = zeros(PATCH_DIM, PATCH_DIM)
        M_grid_double = zeros(PATCH_DIM, PATCH_DIM)
        for ir in 1:RAD_DIM, ia in 1:ANG_DIM
            ycorner = I_G2P[ir, ia, :]
            for i0 in 1:INTERP_ORDER, i1 in 1:INTERP_ORDER
                M_grid_single[ycorner[1]+i0, ycorner[2]+i1] += M_G2P[ir, ia, i0, i1] * M_polar_single[ir, ia]
                M_grid_double[ycorner[1]+i0, ycorner[2]+i1] += M_G2P[ir, ia, i0, i1] * M_polar_double[ir, ia]
            end
        end

        # Add far-field POU contribution (Gpou = -χ on grid) and near-field polar quadrature result
        for j in 1:PATCH_DIM, i in 1:PATCH_DIM
            # Map back to global indices
            idx_pol = mod1(i_obs - PATCH_RAD + i - 1, source.mtheta)
            idx_tor = mod1(j_obs - PATCH_RAD + j - 1, source.nzeta)
            idx_src = idx_pol + source.mtheta * (idx_tor - 1)

            # Remainder of far-field contribution on the singular grid: -χGᵢⱼdA
            r_src, n_src, dA_src = source.r[idx_src, :], source.normal[idx_src, :], source.dA[idx_src]
            far_single = laplace_single_layer(r_obs, r_src) * Gpou[i, j] * (4π^2 / (observer.mtheta * observer.nzeta)) #* dA_src
            far_double = laplace_double_layer(r_obs, r_src, n_src) * dA_src * Gpou[i, j]

            # Apply far + near contributions
            greenfunction[idx_obs, idx_src] += M_grid_single[i, j] + far_single
            grad_greenfunction[idx_obs, idx_src] += M_grid_double[i, j] + far_double
        end
    end

    # Account for normal direction pointing out of vacuum integration region in 𝒦ⁿ ⋅ dS
    # Negative for plasma since dS = ∇ψ J dθdζ and ∇ψ points outward but outward normal is inward
    # @views grad_greenfunction .*= (source isa PlasmaGeometry3D ? -1 : 1)
end
