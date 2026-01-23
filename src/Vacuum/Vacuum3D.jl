"""
Precomputed data for singular correction quadrature following BIEST approach.
Initialized once on first use.

Conversion references:

  - Patch/polar layout and POU weights mirror setup in biest/singular_correction.hpp
"""
struct SingularQuadratureData
    qx::Vector{Float64}       # Radial quadrature points
    qw::Vector{Float64}       # Radial quadrature weights
    Gpou::Vector{Float64}     # Partition of unity on Cartesian grid
    Ppou::Vector{Float64}     # Partition of unity on polar grid
    M_G2P::Vector{Float64}    # Interpolation matrix: grid → polar
    I_G2P::Vector{Int}        # Sparse indices for interpolation
    PATCH_DIM::Int            # Patch dimension
    INTERP_ORDER::Int         # Interpolation order
    Npolar::Int             # Number of polar points
end

# Global cache for quadrature data (initialized on first use)
const SINGULAR_QUAD_CACHE = Ref{Union{Nothing,SingularQuadratureData}}(nothing)

"""
    init_singular_quadrature(PATCH_RAD::Int, RAD_DIM::Int, INTERP_ORDER::Int=6)

Initialize quadrature points, weights, partition-of-unity functions, and
interpolation matrices for singular correction. Follows BIEST's approach.

Conversion references:

  - Quadrature/Patch setup adapted from biest/singular_correction.hpp

# Arguments

  - `PATCH_RAD::Int`: Number of points adjacent to source point to treat as singular
  - `RAD_DIM::Int`: Radial quadrature order
  - `INTERP_ORDER::Int`: Lagrange interpolation order (default 6)

# Returns

  - `SingularQuadratureData`: Precomputed quadrature data
"""
function init_singular_quadrature(PATCH_RAD::Int, RAD_DIM::Int, INTERP_ORDER::Int=6)

    # Total size of square patch extracted around singular point (odd number: 2*PATCH_DIM0+1)
    PATCH_DIM = 2 * PATCH_RAD + 1
    @assert INTERP_ORDER <= PATCH_DIM "Must have INTERP_ORDER <= PATCH_DIM"
    # Number of angular quadrature nodes in polar coordinates (uniformly distributed around circle)
    ANG_DIM = 2 * RAD_DIM
    # Total number of points in the square Patch
    Ngrid = PATCH_DIM * PATCH_DIM
    # Total number of polar quadrature nodes
    Npolar = RAD_DIM * ANG_DIM

    # Setup radial quadrature (Gauss-Legendre on [0,1])
    qx, qw = FastGaussQuadrature.gausslegendre(RAD_DIM)

    # Partition of unity function, exp(-36 * r^p) where p depends on PATCH_DIM
    pou_power = PATCH_DIM > 45 ? 10 : (PATCH_DIM > 20 ? 8 : 6)
    pou_func(r) = r ≥ 1.0 ? 0.0 : exp(-36.0 * r^pou_power)

    # Evaluate POU on Cartesian grid
    Gpou = zeros(Ngrid)
    for i in 1:PATCH_DIM, j in 1:PATCH_DIM
        r = sqrt((i - 1 - PATCH_RAD)^2 + (j - 1 - PATCH_RAD)^2) / PATCH_RAD
        Gpou[j+(i-1)*PATCH_DIM] = -pou_func(r)
    end

    # Evaluate POU on polar grid including transformation Jacobian - Ppou = χ(ρ) M²/4 r dr dt, eq. 38 in Malhotra 2019
    Ppou = zeros(Npolar)
    dt = 2π / ANG_DIM
    for j in 1:ANG_DIM, i in 1:RAD_DIM
        dr = qw[i] * PATCH_RAD
        rdt = qx[i] * PATCH_RAD * dt;
        Ppou[j+(i-1)*ANG_DIM] = pou_func(qx[i]) * dr * rdt
    end

    # Build Lagrange interpolation matrix from Cartesian grid to polar points
    M_G2P = zeros(Npolar * INTERP_ORDER^2)
    I_G2P = zeros(Int, Npolar)

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

    for ia in 1:ANG_DIM
        theta_polar = 2π * (ia - 1) / ANG_DIM
        for ir in 1:RAD_DIM
            # Map polar node to unit square: x0, x1 ∈ [0,1] × [0,1]
            x0 = 0.5 + 0.5 * qx[ir] * cos(theta_polar)
            x1 = 0.5 + 0.5 * qx[ir] * sin(theta_polar)
            println("x0 = $x0, x1 = $x1")

            # Lower-left corner indices of INTERP_ORDER × INTERP_ORDER stencil centered on (x0,x1)
            # Clamped to stay within patch boundaries
            i0_center = trunc(Int, x0 * (PATCH_DIM - 1) - (INTERP_ORDER - 1) / 2)
            y0 = clamp(i0_center, 0, PATCH_DIM - INTERP_ORDER)
            i1_center = trunc(Int, x1 * (PATCH_DIM - 1) - (INTERP_ORDER - 1) / 2)
            y1 = clamp(i1_center, 0, PATCH_DIM - INTERP_ORDER)

            # Local coordinates within INTERP_ORDER×INTERP_ORDER stencil, normalized to [0,1]
            z0 = (x0 * (PATCH_DIM - 1) - y0) * h
            z1 = (x1 * (PATCH_DIM - 1) - y1) * h

            idx = ia + ANG_DIM * (ir - 1)
            I_G2P[idx] = y0 * PATCH_DIM + y1
            for j0 in 0:(INTERP_ORDER-1)
                for j1 in 0:(INTERP_ORDER-1)
                    M_G2P[1+j1+INTERP_ORDER*(j0+INTERP_ORDER*(idx-1))] = lagrange_interp(z0, z1, j0, j1)
                end
            end
        end
    end

    return SingularQuadratureData(qx, qw, Gpou, Ppou, M_G2P, I_G2P, PATCH_DIM, INTERP_ORDER, Npolar)
end

"""
    get_singular_quadrature(PATCH_RAD::Int, RAD_DIM::Int)

Get cached singular quadrature data, initializing if necessary.

Conversion references:

  - Follows caching pattern used around FieldPeriodBIOp setup in biest/boundary_integ_op.hpp
"""
function get_singular_quadrature(PATCH_RAD::Int, RAD_DIM::Int)
    if isnothing(SINGULAR_QUAD_CACHE[])
        SINGULAR_QUAD_CACHE[] = init_singular_quadrature(PATCH_RAD, RAD_DIM)
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

Conversion references:

  - Fundamental solution form matches FxU kernel definition in biest/kernel.hpp
"""
function laplace_single_layer(x_obs::Vector{Float64}, x_src::Vector{Float64})::Float64

    # Single-layer kernel: 1/(4π r)
    r = norm(x_obs - x_src)

    # if r < 1e-10
    #     @warn("laplace_single_layer: Observation and source points are too close (r = $r). Returning 0.0 to avoid singularity. Increase PATCH_RAD to improve accuracy.")
    #     return 0.0
    # end

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

Conversion references:

  - Normal-derivative kernel mirrors DxU in biest/kernel.hpp
"""
function laplace_double_layer(x_obs::Vector{Float64}, x_src::Vector{Float64}, n_src::Vector{Float64})::Float64

    # Double-layer kernel: -1/(4π) * (r·n) / r³
    r = norm(x_obs - x_src)

    # if r < 1e-10
    #     @warn("laplace_double_layer: Observation and source points are too close (r = $r). Returning 0.0 to avoid singularity. Increase PATCH_RAD to improve accuracy.")
    #     return 0.0
    # end

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
function extract_patch(data::Matrix{Float64}, Nt::Int, Np::Int, t0::Int, p0::Int, PATCH_DIM::Int)

    @assert size(data, 1) == Nt * Np
    @assert Nt ≥ PATCH_DIM
    @assert Np ≥ PATCH_DIM

    PATCH_RAD = (PATCH_DIM - 1) ÷ 2
    t_start = t0 - PATCH_RAD
    p_start = p0 - PATCH_RAD
    dof = size(data, 2)  # Number of components (3 for coords, 1 for scalars)
    patch = zeros(PATCH_DIM * PATCH_DIM, dof)

    for i in 1:PATCH_DIM, j in 1:PATCH_DIM
        # Enforce periodicity
        t = mod1(t_start + i - 1, Nt)
        p = mod1(p_start + j - 1, Np)

        # Copy data to the patch
        idx_in = p + Np * (t - 1)
        idx_out = j + PATCH_DIM * (i - 1)
        @views patch[idx_out, :] .= data[idx_in, :]
    end

    return patch # (Ngrid, dof)
end

"""
    interpolate_to_polar(patch, M_G2P, Npolar)

Interpolate Cartesian patch data to polar quadrature points using precomputed matrix.

# Arguments

  - `patch`: Patch data (PATCH_DIM² × dof)
  - `quad_data`: Precomputed quadrature data

# Returns

  - `polar_data`: Interpolated data at polar points (Npolar × dof)

Conversion references:

  - Grid→polar interpolation follows M_G2P application in biest/singular_correction.hpp
"""
function interpolate_to_polar(patch::Matrix{Float64}, quad_data::SingularQuadratureData)

    (; M_G2P, I_G2P, INTERP_ORDER, PATCH_DIM, Npolar) = quad_data
    dof = size(patch, 2)
    polar_data = zeros(Npolar, dof)
    for j in 1:Npolar
        # Base offset for interpolation matrix block
        M_offset = (j - 1) * INTERP_ORDER * INTERP_ORDER
        for k in 1:dof
            sum = 0.0
            for i0 in 0:(INTERP_ORDER-1), i1 in 0:(INTERP_ORDER-1)
                # Index into interpolation weights
                M_idx = M_offset + i0 * INTERP_ORDER + i1 + 1
                # Index into the patch for this stencil point
                patch_idx = I_G2P[j] + i0 * PATCH_DIM + i1 + 1
                sum += M_G2P[M_idx] * patch[patch_idx, k]
            end
            polar_data[j, k] = sum
        end
    end

    return polar_data
end

"""
    compute_singular_correction!(
        greenfunction, grad_greenfunction,
        obs_idx, source_geom, quad_data,
        PATCH_DIM, RAD_DIM
    )

Compute singular correction for observer point using polar quadrature.
Follows BIEST's approach: extract patch, interpolate to polar coordinates,
evaluate kernels with POU weighting, distribute back to grid.

# Arguments

  - `greenfunction`: Single-layer matrix to update
  - `grad_greenfunction`: Double-layer matrix to update
  - `obs_idx`: Linear index of observer point
  - `source_geom`: Source geometry (PlasmaGeometry3D)
  - `quad_data`: Precomputed quadrature data
  - `RAD_DIM`: Radial quadrature order

Conversion references:

  - Patch → polar correction mirrors SingularCorrection operator in biest/singular_correction.hpp
  - Subtract/add near-field correction follows EvalSurfInteg flow in biest/surface_op.txx
"""
function compute_singular_correction!(
    greenfunction::Matrix{Float64},
    grad_greenfunction::Matrix{Float64},
    obs_idx::Int,
    obs_geom::PlasmaGeometry3D,
    source_geom::PlasmaGeometry3D,
    quad_data::SingularQuadratureData,
    t0::Int, p0::Int,
    RAD_DIM::Int
)
    (; nzeta, ntheta, r, n, dA) = source_geom
    (; PATCH_DIM, INTERP_ORDER, Npolar, qw, Ppou, Gpou, M_G2P, I_G2P) = quad_data
    PATCH_RAD = (PATCH_DIM - 1) ÷ 2
    ANG_DIM = 2 * RAD_DIM
    # Observer position
    r_obs = obs_geom.r[obs_idx, :]

    # Extract patches of source geometry (size = Ngrid x dof)
    r_patch = extract_patch(r, nzeta, ntheta, t0, p0, PATCH_DIM)
    n_patch = extract_patch(n, nzeta, ntheta, t0, p0, PATCH_DIM)
    dA_patch = extract_patch(reshape(dA, :, 1), nzeta, ntheta, t0, p0, PATCH_DIM)

    # Interpolate to polar quadrature points (size = Npolar x dof)
    r_polar = interpolate_to_polar(r_patch, quad_data)
    n_polar = interpolate_to_polar(n_patch, quad_data)
    dA_polar = interpolate_to_polar(dA_patch, quad_data)[:, 1]

    # Compute area elements at polar points (cross product of tangent vectors)
    # For now use interpolated dA; could recompute from derivatives if needed

    # Evaluate kernels at polar points with POU weighting
    M_polar_single = zeros(Npolar)
    M_polar_double = zeros(Npolar)
    dtheta = 2π / ANG_DIM
    for ia in 1:ANG_DIM, ir in 1:RAD_DIM
        ipolar = ir + RAD_DIM * (ia - 1)
        r_src, n_src = r_polar[ipolar, :], n_polar[ipolar, :]

        # Evaluate kernels
        K_single = laplace_single_layer(r_obs, r_src)
        K_double = laplace_double_layer(r_obs, r_src, n_src)

        # Apply quadrature weights: radial weight × angular weight × area element × POU
        wt = qw[ir] * dtheta * dA_polar[ipolar] * Ppou[ipolar]
        M_polar_single[ipolar] = K_single * wt
        M_polar_double[ipolar] = K_double * wt
    end

    # Distribute corrections back to Cartesian grid using interpolation matrix
    # Correction at grid point = sum over polar points of (kernel_value * interp_weight)
    M_grid_single = zeros(PATCH_DIM * PATCH_DIM)
    M_grid_double = zeros(PATCH_DIM * PATCH_DIM)
    M_G2P_view = reshape(M_G2P, INTERP_ORDER, INTERP_ORDER, Npolar)
    for j in 1:Npolar
        for i0 in 0:(INTERP_ORDER-1), i1 in 0:(INTERP_ORDER-1)
            patch_idx = I_G2P[j] + i0 * PATCH_DIM + i1 + 1
            M_grid_single[patch_idx] += M_G2P_view[i0+1, i1+1, j] * M_polar_single[j]
            M_grid_double[patch_idx] += M_G2P_view[i0+1, i1+1, j] * M_polar_double[j]
        end
    end

    # Subtract far-field contribution (Gpou = -χ on grid) and add near-field polar quadrature result
    for j in 1:PATCH_DIM, i in 1:PATCH_DIM
        igrid = i + PATCH_DIM * (j - 1)

        # Map back to global indices
        it = mod1(t0 - PATCH_RAD - 1 + i, nzeta)
        ip = mod1(p0 - PATCH_RAD - 1 + j, ntheta)
        idx_src = it + nzeta * (ip - 1)

        # Far-field contribution to subtract (computed with rectangle rule)
        r_src, n_src, dA_src = r[idx_src, :], n[idx_src, :], dA[idx_src]
        far_single = laplace_single_layer(r_obs, r_src) * dA_src * Gpou[igrid]
        far_double = laplace_double_layer(r_obs, r_src, n_src) * dA_src * Gpou[igrid]

        # Apply correction: subtract far, add near
        greenfunction[obs_idx, idx_src] += M_grid_single[igrid] + far_single
        grad_greenfunction[obs_idx, idx_src] += M_grid_double[igrid] + far_double
    end
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
    RAD_DIM::Int=12
)

    # Zero out matrices
    fill!(grad_greenfunction, 0.0)
    fill!(greenfunction, 0.0)

    # Initialize quadrature data (cached)
    quad_data = get_singular_quadrature(PATCH_RAD, RAD_DIM)

    # Helper for periodic distance
    @inline periodic_dist(i, j, n) = min(abs(i - j), n - abs(i - j))

    # Loop through observer points
    for j_obs in 1:observer.nzeta, i_obs in 1:observer.ntheta
        idx_obs = i_obs + (j_obs - 1) * observer.ntheta
        r_obs = observer.r[idx_obs, :]

        # Get indices excluding the singular region (square of radius ±PATCH_RAD around observer point)
        # Include points where either θ and ζ distance is greater than patch
        # (i.e., at least one coordinate distance > PATCH_RAD)
        nonsing_idx = Vector{Int}(undef, 0)
        sizehint!(nonsing_idx, observer.ntheta * observer.nzeta)
        for j in 1:source.nzeta, i in 1:source.ntheta
            if periodic_dist(i, i_obs, source.ntheta) > PATCH_RAD ||
               periodic_dist(j, j_obs, source.nzeta) > PATCH_RAD
                push!(nonsing_idx, i + (j - 1) * source.ntheta)
            end
        end

        # ============================================
        # FAR FIELD: Rectangle rule for nonsingular source points
        # ============================================
        for idx_src in nonsing_idx
            # Evaluate kernels at grid points
            K_single = laplace_single_layer(r_obs, source.r[idx_src, :])
            K_double = laplace_double_layer(r_obs, source.r[idx_src, :], source.n[idx_src, :])

            # Apply area element (periodic trapezoidal rule: w = dA = J∇ψdθdζ)
            greenfunction[idx_obs, idx_src] = K_single * source.dA[idx_src]
            grad_greenfunction[idx_obs, idx_src] = K_double * source.dA[idx_src]
        end

        # ============================================
        # NEAR FIELD: Polar quadrature with singular correction
        # ============================================
        compute_singular_correction!(
            greenfunction, grad_greenfunction,
            idx_obs, observer, source,
            quad_data, i_obs, j_obs,
            RAD_DIM
        )
    end

    # Account for normal direction pointing out of vacuum integration region in 𝒦ⁿ ⋅ dS
    # Negative for plasma since dS = ∇ψ J dθdζ and ∇ψ points outward but outward normal is inward
    @views grad_greenfunction .*= (source isa PlasmaGeometry3D ? -1 : 1)
end
