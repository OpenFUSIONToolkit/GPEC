"""
    GaussLegendreRule{N,T}

Allocation-free Gauss–Legendre nodes/weights on the canonical interval [-1, 1].
Stored as `SVector`s so tight loops can index them efficiently.
"""
struct GaussLegendreRule{N,T}
    x::SVector{N,T}
    w::SVector{N,T}
end

@inline function gausslegendre_rule(::Val{N}) where {N}
    x, w = gausslegendre(N) # canonical [-1, 1]
    return GaussLegendreRule{N,Float64}(
        SVector{N,Float64}(ntuple(i -> Float64(x[i]), N)),
        SVector{N,Float64}(ntuple(i -> Float64(w[i]), N))
    )
end

# Precomputed Gauss-Legendre rule used in the hot path (`kernel!`).
const GL8 = gausslegendre_rule(Val(8))

"""
    precompute_lagrange_stencils(gaussian_points)

Precompute 5-point Lagrange interpolation stencils for Gaussian quadrature points.

Returns a tuple `(left, right)` where each entry is a Vector of SVector{5,Float64}
containing the stencil weights for points on the left/right panel.
"""
function precompute_lagrange_stencils(gaussian_points::AbstractVector{<:Real})
    stencil_points = SVector(-2, -1, 0, 1, 2)
    npts = length(gaussian_points)
    left = Vector{SVector{5,Float64}}(undef, npts)
    right = Vector{SVector{5,Float64}}(undef, npts)

    for ig in 1:npts
        p_left = -1.0 + gaussian_points[ig]
        p_right = 1.0 + gaussian_points[ig]

        left[ig] = ntuple(5) do i
            xi = stencil_points[i]
            prod(j -> j == i ? 1.0 : (p_left - stencil_points[j]) / (xi - stencil_points[j]), 1:5)
        end |> SVector

        right[ig] = ntuple(5) do i
            xi = stencil_points[i]
            prod(j -> j == i ? 1.0 : (p_right - stencil_points[j]) / (xi - stencil_points[j]), 1:5)
        end |> SVector
    end

    return left, right
end

# Precomputed 5-point Lagrange stencils for the 8-point Gaussian nodes.
const GL8_LAGRANGE_STENCILS = precompute_lagrange_stencils(GL8.x)

# Pre-computed Gauss quadrature constants (_PN_TG02, _PN_WANUMR, _PN_AGAUS, _PN_BGAUS)
# and per-n sinh/cosh cache are defined in PnQuadCache.jl.

"""
    compute_2D_kernel_matrices!(K, G, observer, source, n, Z)

Compute the **Fourier/Galerkin-projected** 2D vacuum boundary-integral kernel blocks for
Laplace’s equation in an axisymmetric torus, **without ever forming the dense
`mtheta×mtheta` “point-to-point” kernel matrices.

This is the fused “evaluate kernel + project” path that the vacuum solver uses:

    Kc = Zᴴ * K * Z
    Gc = Zᴴ * G

where:

  - `K` is the **double-layer** kernel (normal derivative of the Green’s function),
  - `G` is the **single-layer** kernel (Green’s function itself; only needed for plasma-as-source),
  - `Z ∈ ℂ^{mtheta×num_modes}` is the complex Fourier basis on the poloidal grid,
    and `Zᴴ` is its conjugate transpose.

Rather than computing a full kernel row `K[j, :]` and then multiplying by `Z`, this routine
projects **on the fly**:

  - For each observer node `j`, it accumulates the projected row-vector
    `proj_k = (K[j,:] * weights) · Z` and (optionally) `proj_g = (G[j,:] * weights) · Z`
    into length-`num_modes` work buffers.

  - It then performs a **rank-1 update** into the appropriate projected block:

        Kc += conj(Z[j, :])' * proj_k
        Gc += conj(Z[j, :])' * proj_g

This reduces peak memory from `O(mtheta^2)` to `O(mtheta*num_modes + num_modes^2)` while keeping the same
mathematical discretization.

## Arguments

  - `Kc`: Complex global projected double-layer kernel matrix.
  - `Gc`: Complex global projected single-layer kernel matrix.
  - `observer`: `PlasmaGeometry` or `WallGeometry` object providing `x(θ)` and `z(θ)`.
  - `source`: `PlasmaGeometry` or `WallGeometry` object providing `x(θ)` and `z(θ)`.
  - `n`: Integer representing the order of the toroidal Fourier component.
  - `Z`: Complex Fourier basis sampled on the `θ` grid.

## Block layout

`Kc` and `Gc` are the complex global projected matrices. `Kc` contains four blocks corresponding to
plasma/wall as observer/source, `Gc` contains two blocks corresponding to plasma/wall as observer.
This function writes **only one block** to each of `Kc` and `Gc` per call:

  - `Kc_block` is a `num_modes×num_modes` view into `Kc` selected by `(observer isa PlasmaGeometry ? 1 : 2, source isa PlasmaGeometry ? 1 : 2)`.
  - `Gc_block` is a `num_modes×(2*num_modes)` view into `Gc` with the same observer block-row; only the columns
    corresponding to the source being plasma are populated (when `source isa PlasmaGeometry`).

## Toroidal Green's functions

  - The scalar `G_n` returned by `green(...)` is `2π * 𝒢ⁿ(θ, θ′)` (Chance 1997),
    the `n`-th toroidal Fourier component of the Laplace Green’s function in axisymmetry.
  - The scalar `gradG_n` returned by `green(...)` corresponds to the toroidal-mode `n`
    contribution to the **double-layer** integrand `𝒥 * (∇′𝒢ⁿ · ∇′ℒ)`
    (Chance 1997), i.e. the normal-derivative factor multiplied by the geometric Jacobian.
  - `gradG_0` is the `n = 0` piece used for analytic diagonal/singularity bookkeeping.

## Numerical treatment of the singularity

The toroidal Green’s function kernel is weakly singular as `θ′ → θ`. The implementation follows
Chance *Phys. Plasmas* **4**, 2161 (1997) and uses a mixed strategy:

  - **Far field (nonsingular region)**: composite Simpson’s `1/3` rule on the uniform `θ` grid,
    skipping the near-singular stencil around `j`.
  - **Near field (singular panels)**: 8-point Gauss–Legendre quadrature on the two panels
    of length `dtheta` immediately to the left/right of `θ_j`, using:
      + periodic cubic splines of `source.x(θ)` and `source.z(θ)` to evaluate geometry at off-grid nodes,
      + precomputed 5-point Lagrange stencils to map each Gauss node back to the five neighboring
        discrete source indices `[j-2, j-1, j, j+1, j+2]` *without allocations*,
      + analytic logarithmic/singular correction terms (`log_correction_array`) for the single-layer
        kernel when the observer/source block is plasma–plasma (Chance 1997, e.g. eqs. 75, 78),
      + an analytic diagonal/residue correction for the double-layer kernel (Chance 1997, Table I / residues).

## Performance and allocation avoidance (hot-path optimizations)

This routine is intentionally written to be allocation-light in tight loops:

  - **Precomputed quadrature tables**: `GL8` and `GL8_LAGRANGE_STENCILS` are global constants.
  - **Hoisted n-dependent constants**: `gamma_prefactor = 2√π * Γ(1/2 - n)` is computed once per call
    and passed into `green(...)` rather than recomputed per quadrature node.
  - **Spline derivative batching**: `∂R/∂θ` and `∂Z/∂θ` are evaluated on the full grid once (for Simpson),
    while off-grid Gauss points evaluate splines directly.
  - **Projection reuse**: the transpose `Zt = transpose(Z)` is materialized so that “row-accumulate”
    operations `_accum_row!` can access `Z` with contiguous column memory.
  - **Rank-1 assembly**: the final projected update uses `conj(Z[j,:]) ⊗ proj_*` via `_rank1_conj!`,
    avoiding constructing intermediate `num_modes×num_modes` temporaries.

## Caveats / limitations

  - **Closed-wall assumption**: the current residue/diagonal handling is written for closed,
    periodic toroidal boundaries. Open-wall residue logic is not implemented.
"""
@with_pool pool function compute_2D_kernel_matrices!(
    Kc::AbstractMatrix{ComplexF64},
    Gc::AbstractMatrix{ComplexF64},
    observer::Union{PlasmaGeometry,WallGeometry},
    source::Union{PlasmaGeometry,WallGeometry},
    n::Int,
    Z::AbstractMatrix{ComplexF64}
)

    mtheta, num_modes = size(Z)
    Zt = Matrix{ComplexF64}(transpose(Z))  # [num_modes × mtheta] for contiguous column access
    dtheta = 2π / mtheta
    theta_grid = range(; start=0, length=mtheta, step=dtheta)

    # Take a view of the corresponding block of the grad_greenfunction
    col_idx = (source isa PlasmaGeometry ? 1 : 2)
    row_idx = (observer isa PlasmaGeometry ? 1 : 2)
    Kc_block = view(Kc, ((row_idx-1)*num_modes+1):(row_idx*num_modes), ((col_idx-1)*num_modes+1):(col_idx*num_modes))
    Gc_block = view(Gc, ((row_idx-1)*num_modes+1):(row_idx*num_modes), :)

    # 𝒢ⁿ only needed for plasma as source term (RHS of eqs. 26/27 in Chance 1997)
    populate_greenfunction = source isa PlasmaGeometry

    # S₁ᵢ logarithmic correction factors [Chance Phys. Plasmas 1997 2161 eq. 78]
    log_correction_0=16.0*dtheta*(log(2*dtheta)-68.0/15.0)/15.0
    log_correction_1=128.0*dtheta*(log(2*dtheta)-8.0/15.0)/45.0
    log_correction_2=4.0*dtheta*(7.0*log(2*dtheta)-11.0/15.0)/45.0
    log_correction_array = SVector(log_correction_2, log_correction_1, log_correction_0, log_correction_1, log_correction_2)

    # Precompute the n-dependent prefactor 2√π·Γ(1/2-n) [Chance Phys. Plasmas 1997 2161 eq. 40]
    # This constant is only computed once for each n
    gamma_prefactor = 2 * sqrt(π) * gamma(0.5 - n)

    # Set up periodic splines used for off-grid Gaussian quadrature points
    spline_x = cubic_interp(theta_grid, source.x; bc=PeriodicBC(; endpoint=:exclusive, period=2π))
    spline_z = cubic_interp(theta_grid, source.z; bc=PeriodicBC(; endpoint=:exclusive, period=2π))
    d1_spline_x = deriv1(spline_x)
    d1_spline_z = deriv1(spline_z)

    # Precompute 5-point Lagrange stencils for the 8-point Gaussian nodes.
    stencils_left, stencils_right = GL8_LAGRANGE_STENCILS
    sing_idx = zeros!(pool, Int, 5)

    # Precompute source derivatives on the theta grid once used in Simpson integration
    # The Gaussian singular-panel points are off-grid, so those still use spline evaluation directly.
    dx_dtheta_grid = acquire!(pool, eltype(source.x), mtheta)
    dz_dtheta_grid = acquire!(pool, eltype(source.z), mtheta)

    # Call in-place API to avoid allocations
    d1_spline_x(dx_dtheta_grid, theta_grid)
    d1_spline_z(dz_dtheta_grid, theta_grid)

    # Pre-allocated Legendre buffer (hoisted out of green() to avoid per-call pool acquisition)
    legendre_buf = acquire!(pool, Float64, n + 2)

    # Per-observer projection vectors: proj = (kernel row) · Z
    proj_k = zeros!(pool, ComplexF64, num_modes)
    proj_g = zeros!(pool, ComplexF64, num_modes)

    # Loop through observer points
    for j in 1:mtheta
        # Get observer coordinates
        x_obs, z_obs, theta_obs = observer.x[j], observer.z[j], theta_grid[j]

        # Zero out projection terms
        fill!(proj_k, 0.0)
        fill!(proj_g, 0.0)
        diag_accum = 0.0

        # ============================================================
        # FAR FIELD: Simpson integration for nonsingular source points
        # ============================================================
        # Nonsingular region endpoints are at j±2, so exclude j-1, j, and j+1.
        @inbounds for k in 1:(mtheta-3)
            isrc = mod1(j + 1 + k, mtheta)
            G_n, gradG_n, gradG_0 = green(x_obs, z_obs, source.x[isrc], source.z[isrc], dx_dtheta_grid[isrc], dz_dtheta_grid[isrc], n, legendre_buf; gamma_prefactor)

            # Composite Simpson's 1/3 rule weights, excluding singular points
            # Note we set to 4 for even/2 for odd since we index from 1 while the formula assumes indexing from 0
            wsimpson = dtheta / 3 * ((k == 1 || k == mtheta - 3) ? 1 : (iseven(k) ? 4 : 2))

            # Sum and project contributions to Green's function matrices
            if populate_greenfunction
                _accum_row!(proj_g, G_n * wsimpson, Zt, isrc)
            end
            _accum_row!(proj_k, gradG_n * wsimpson, Zt, isrc)
            # Subtract regular integral component of δⱼᵢK⁰ [Chance Phys. Plasmas 1997 2161 eq. 83]
            diag_accum -= gradG_0 * wsimpson
        end

        # ============================================================
        # NEAR FIELD: Gaussian quadrature with singular correction
        # ============================================================
        # Indices of the singularity region, [j-2, j-1, j, j+1, j+2] (allocation-free)
        for (offset_idx, offset) in enumerate(-2:2)
            sing_idx[offset_idx] = mod1(j + offset + mtheta, mtheta)
        end

        # Integrate region of length 2 * dtheta on left/right of singularity
        for leftpanel in (true, false)
            gauss_mid = theta_obs + (leftpanel ? -dtheta : dtheta)
            @inbounds for ig in 1:8 # 8-point Gaussian quadrature
                # Compute green function for this Gaussian point
                theta_gauss = gauss_mid + GL8.x[ig] * dtheta
                theta_gauss0 = mod(theta_gauss, 2π)
                x_gauss = spline_x(theta_gauss0)
                dx_dtheta_gauss = d1_spline_x(theta_gauss0)
                z_gauss = spline_z(theta_gauss0)
                dz_dtheta_gauss = d1_spline_z(theta_gauss0)
                G_n, gradG_n, gradG_0 = green(x_obs, z_obs, x_gauss, z_gauss, dx_dtheta_gauss, dz_dtheta_gauss, n, legendre_buf; gamma_prefactor)

                # Get stencil and weight for the Gaussian point
                s = leftpanel ? stencils_left[ig] : stencils_right[ig]
                wgauss = GL8.w[ig] * dtheta

                # First type of singularity: 𝒢ⁿ [Chance Phys. Plasmas 1997 2161 eq. 75]
                if populate_greenfunction
                    if observer isa PlasmaGeometry
                        # Remove singular behavior by adding on leading-order term
                        G_n += log((theta_obs - theta_gauss)^2) / x_obs
                    end
                    @inbounds for stencil_idx in 1:5
                        _accum_row!(proj_g, G_n * s[stencil_idx] * wgauss, Zt, sing_idx[stencil_idx])
                    end
                end

                # Second type of singularity: 𝒦ⁿ [Chance Phys. Plasmas 1997 2161 eq. 83, 86]
                @inbounds for stencil_idx in 1:5
                    _accum_row!(proj_k, gradG_n * s[stencil_idx] * wgauss, Zt, sing_idx[stencil_idx])
                end
                # Subtract off the diverging singular n=0 component
                diag_accum -= gradG_0 * wgauss
            end
        end

        # Subtract off analytic singular integral [Chance Phys. Plasmas 1997 2161 eq. 75] if plasma-plasma block
        if populate_greenfunction && observer isa PlasmaGeometry
            @inbounds for stencil_idx in 1:5
                _accum_row!(proj_g, -log_correction_array[stencil_idx] / x_obs, Zt, sing_idx[stencil_idx])
            end
        end

        # Project the n=0 diagonal accumulation
        _accum_row!(proj_k, diag_accum, Zt, j)

        # ── Rank-1 accumulate: K/G += conj(Z[j,:]) ⋅ proj_k/g ──
        _rank1_conj!(Kc_block, Zt, j, proj_k)
        if populate_greenfunction
            _rank1_conj!(Gc_block, Zt, j, proj_g)
        end
    end

    # Normals need to point outward from vacuum region. In VACUUM clockwise θ convention, normal points
    # out of vacuum for wall but inward for plasma, so we multiply by -1 for plasma sources
    if source isa PlasmaGeometry
        Kc_block .*= -1
    end

    # Add analytic singular integral (second type) to block diagonal [Chance Phys. Plasmas 1997 2161 Table I, eq. 69, 89]
    # The residue term is diagonal in mode space and is scaled by the number of points in Fourier space.
    residue = (observer isa WallGeometry) ? 0.0 : (source isa PlasmaGeometry ? 2.0 : -2.0)
    @inbounds for i in 1:num_modes
        Kc_block[i, i] += residue * mtheta
    end

    # Since we computed 2π𝒢, divide by 2π to get 𝒢
    if populate_greenfunction
        Gc_block ./= 2π
    end
end

"""
    kernel!(Kc, Gc, observer, source, params::KernelParams2D, Z)

Public 2D kernel entry point. This is a thin wrapper that forwards to
`compute_2D_kernel_matrices!(Kc, Gc, observer, source, params.n, Z)`.
"""
function kernel!(
    Kc::AbstractMatrix{ComplexF64},
    Gc::AbstractMatrix{ComplexF64},
    observer::Union{PlasmaGeometry,WallGeometry},
    source::Union{PlasmaGeometry,WallGeometry},
    params::KernelParams2D,
    Z::AbstractMatrix{ComplexF64}
)
    return compute_2D_kernel_matrices!(Kc, Gc, observer, source, params.n, Z)
end

#############################################################
# Legendre function of the first kind eq.(47)~(50) , replacing aleg. (verified)
#############################################################

"""
    This function is different from elliptic integral K(k). Be careful.

Returns : K(1-m1)
"""
function elliptic_integral_k(m1)

    (m1 < 0.0 || m1 > 1.0) && throw(DomainError(m1, "Input `m1` must be in the range (0, 1]."))
    log_m1 = log(m1)

    ak0 = 1.38629436112
    ak1 = 0.09666344259
    ak2 = 0.03590092383
    ak3 = 0.03742563713
    ak4 = 0.01451196212
    bk0 = 0.5
    bk1 = 0.12498593597
    bk2 = 0.06880248576
    bk3 = 0.03328355346
    bk4 = 0.00441787012

    p = @evalpoly(m1, ak0, ak1, ak2, ak3, ak4)
    q = @evalpoly(m1, bk0, bk1, bk2, bk3, bk4)

    ellipk = p - q * log_m1
    return ellipk
end


"""
    This function is different from elliptic integral E(k). Be careful.

Returns : E(1-m1)
"""
function elliptic_integral_e(m1)

    (m1 < 0.0 || m1 > 1.0) && throw(DomainError(m1, "Input `x1` must be in the range (0, 1]."))
    log_x1 = log(m1)

    ae1=0.44325141463
    ae2=0.0626060122
    ae3=0.04757383546
    ae4=0.01736506451
    be1=0.2499836831
    be2=0.09200180037
    be3=0.04069697526
    be4=0.00526449639

    p = @evalpoly(m1, 1.0, ae1, ae2, ae3, ae4)
    q = @evalpoly(m1, 0.0, be1, be2, be3, be4)

    ellipe = p - q * log_x1
    return ellipe

end


# Chance 1997 eq.(49) (original)
function P0_minus_half(s)
    m1 = 2 / (s + 1)
    return 2 / π * sqrt(m1) * elliptic_integral_k(m1)
end

# Chance 1997 eq.(50) (original)
# This is the case where the paper has a typo, the -1/4 exponent is written in the paper as +1/2
function P0_plus_half(s)
    m1 = (s + sqrt(s^2 - 1))^(-2)
    return 2 / π * m1^(-1/4) * elliptic_integral_e(m1) # This is correct
end


# Chance 1997 eq.(48) (original)
function P1_minus_half(s)
    return 0.5 / ((s^2 - 1)^0.5) * (P0_plus_half(s) - s * P0_minus_half(s))
end

"""
    Pn_minus_half_1997(s, n)

Compute the Legendre function of the first kind of order -1/2, P^n_{-1/2}(s),
recursively using Chance 1997 equations (47)-(50).

The implementation follows the original Fortran code. Note: equation (50) in the paper
has a typo where the exponent should be -1/4 instead of +1/2.

# Arguments

  - `s::Real`: Legendre function parameter (s > 1)
  - `n::Int`: Maximum order n (n ≥ 0)

# Returns

  - `P::Vector{Float64}`: Array of values P^0_{-1/2}(s) through P^{n+1}_{-1/2}(s)

# Notes

  - Uses recursive relation from Chance 1997 eq. (47)
  - Base cases computed from eqs. (48)-(50) using elliptic integrals
"""
function Pn_minus_half_1997(s::Real, n::Int)
    P = Vector{Float64}(undef, n + 2)
    return Pn_minus_half_1997!(P, s, n)
end

function Pn_minus_half_1997!(P::AbstractVector{Float64}, s::Real, n::Int)

    #initialize
    P .= 0.0

    # n = 0
    P[1] = P0_minus_half(s)
    P[2] = P1_minus_half(s)
    if n == 0
        return P
    end

    # n ≥ 1
    for i in 1:n
        # Chance 1997 eq.(47)
        P[i+2] = -2 * i * s / sqrt(s^2 - 1) * P[i+1] - (i - 0.5)^2 * P[i]
    end

    return P
end

"""
    elliptic_integrals_bulirsch(m1; error=1e-8, maxit=10)

Compute complete elliptic integrals K(m1) and E(m1) using Bulirsch's algorithm.
This is the Julia equivalent of the Fortran `ek3` subroutine.

# Arguments

  - `m1::Float64`: Complementary parameter (1 - k²), where k is the elliptic modulus
  - `error::Float64`: Convergence tolerance (default 1e-8)
  - `maxit::Int`: Maximum iterations (default 10)

# Returns

  - `K::Float64`: Complete elliptic integral of the first kind K(m1)
  - `E::Float64`: Complete elliptic integral of the second kind E(m1)
  - `convergence::Float64`: Convergence metric
  - `iterations::Int`: Number of iterations performed

# Notes

  - Based on Bulirsch's method as described in Numerical Recipes
  - Precision is approximately error²
  - Reference: JCP 221 (2007) 330-348
"""
function elliptic_integrals_bulirsch(m1::Float64; error::Float64=1e-8, maxit::Int=10)

    # Check valid input
    if m1 <= 0.0 || m1 > 1.0
        throw(DomainError(m1, "Input m1 must be in range (0, 1]"))
    end

    # Initialize for K and E calculation
    pp = 1.0
    aa = 1.0
    bb1 = 1.0      # for K
    bb2 = abs(m1)  # for E

    qcval = sqrt(abs(m1))
    aval0 = aa
    bval1 = bb1
    bval2 = bb2
    pval0 = pp

    eval = qcval
    emval = 1.0

    # Initialize based on pval0 > 0
    if pval0 > 0.0
        pval = sqrt(pval0)
        aval1 = aval0
        aval2 = aval0
        bval1 = bval1 / pval
        bval2 = bval2 / pval
    else
        fval = qcval * qcval
        tval = 1.0 - fval
        gval = 1.0 - pval0
        fval = fval - pval0
        qval1 = tval * (bval1 - aval0 * pval0)
        qval2 = tval * (bval2 - aval0 * pval0)

        pval = sqrt(fval / gval)
        aval1 = (aval0 - bval1) / gval
        aval2 = (aval0 - bval2) / gval
        bval1 = aval1 * pval - qval1 / (gval * gval * pval)
        bval2 = aval2 * pval - qval2 / (gval * gval * pval)
    end

    # Iterate until convergence
    kounter = 0
    sval = 0.0

    while kounter < maxit
        kounter += 1

        hval1 = aval1
        hval2 = aval2
        aval1 = aval1 + bval1 / pval
        aval2 = aval2 + bval2 / pval
        rval = eval / pval
        bval1 = bval1 + hval1 * rval
        bval1 = bval1 + bval1
        bval2 = bval2 + hval2 * rval
        bval2 = bval2 + bval2
        pval = rval + pval

        sval = emval
        emval = qcval + emval

        if abs(sval - qcval) <= sval * error
            break
        end

        qcval = sqrt(eval)
        qcval = qcval + qcval
        eval = qcval * emval
    end

    # Calculate convergence metric
    snorm = (sval != 0.0) ? sval * sval : 1.0
    convergence = (sval - qcval)^2 / snorm
    convergence = max(convergence, 1.0e-100)

    # Calculate final K and E values
    K = π/2 * (bval1 + aval1 * emval) / (emval * (emval + pval))
    E = π/2 * (bval2 + aval2 * emval) / (emval * (emval + pval))

    return K, E, convergence, kounter
end

"""
    Pn_minus_half_2007(s, n)

Compute the Legendre function of the first kind of order -1/2, P^n_{-1/2}(s),
using methods from Chance J. Comp. Phys 221 (2007) 330-348.

This implementation uses:

 1. Bulirsch's algorithm for elliptic integrals (more accurate than polynomial approximations)
 2. Gaussian integration for large mode numbers (n*rhohat >= 0.1) where rhohat = 1/√(2*y*w)
 3. Upward recurrence for small mode numbers

# Arguments

  - `s::Real`: Legendre function parameter (s > 1)
  - `n::Int`: Maximum order n (n ≥ 0)

# Returns

  - `P::Vector{Float64}`: Array of values P^0_{-1/2}(s) through P^{n+1}_{-1/2}(s)

# Notes

  - This version is more accurate than Pn_minus_half_1997 for large n
  - Expected to diverge from 1997 version at large nloc
  - Reference: JCP 221 (2007) 330-348    # Constants
"""
function Pn_minus_half_2007(s::Real, n::Int)
    P = Vector{Float64}(undef, n + 2)
    return Pn_minus_half_2007!(P, s, n)
end

function Pn_minus_half_2007!(P::AbstractVector{Float64}, s::Real, n::Int)

    # Constants
    pii = 2.0 / π

    # Initialize output array
    P .= 0.0

    # Preliminary computations
    xxq = s * s
    ysq = xxq - 1.0
    y = sqrt(ysq)
    w = s + y

    # rhohat parameter for determining integration method
    rhohatsq = 1.0 / (2.0 * y * w)
    rhohat = sqrt(rhohatsq)

    # Compute m1 = 1/w (complementary parameter for elliptic integrals)
    m1 = 1.0 / w
    m1sq = m1 * m1
    m1sqrt = sqrt(m1)      # m1^(1/4)
    m1sqrti = sqrt(w)      # m1^(-1/4)

    # Compute elliptic integrals using Bulirsch algorithm
    K, E, conv, iters = elliptic_integrals_bulirsch(m1sq; error=1e-15, maxit=20)

    # Base cases: P^0 and P^1
    pn = pii * m1sqrt * K
    pnp = pii * m1sqrti * E

    P[1] = pn  # P^0_{-1/2}
    pp = (pnp - s * pn) / (2.0 * y)
    P[2] = pp  # P^1_{-1/2}

    # Use Gaussian integration if n*rhohat >= 0.1
    if n * rhohat >= 0.1

        pn_cache = get_pn_quad_cache(n)

        gint = 0.0
        gintp = 0.0

        @inbounds for ig in 1:32
            # dnom² = s·sinh²(x) + sinh(x)·cosh(x), x = tg0²/(2n)
            # Half the denominator of [Chance JCP 2007 eq. A.18]; factor √2 absorbed in sqtwo prefactor
            sh = pn_cache.sinh[ig]
            ch = pn_cache.cosh[ig]
            dnom = sqrt(muladd(s, sh * sh, sh * ch))

            shp = pn_cache.sinhp[ig]
            chp = pn_cache.coshp[ig]
            dnomp = sqrt(muladd(s, shp * shp, shp * chp))

            wanumr = _PN_WANUMR[ig]
            gint = muladd(wanumr, inv(dnom), gint)
            gintp = muladd(wanumr, inv(dnomp), gintp)
        end

        gint *= _PN_BGAUS
        gintp *= _PN_BGAUS

        # pcoef = √((s-1)/(s+1)) is the only s-dependent factor in the final assembly.
        # The Γ-function prefactors and normalization constants are pre-cached in
        # pn_cache.gauss_norm_n / pn_cache.gauss_norm_np1 (see PnQuadCache.jl for derivation).
        pcoef = sqrt((s - 1.0) / (s + 1.0))
        pcoef_n = pcoef^n  # pcoef^(n+1) = pcoef_n · pcoef; reuse to avoid a second pow call

        P[end-1] = pcoef_n * gint * pn_cache.gauss_norm_n    # P^n_{-1/2}
        P[end] = pcoef_n * pcoef * gintp * pn_cache.gauss_norm_np1  # P^{n+1}_{-1/2}

    else
        # Use upward recurrence for small n*rhohat < 0.1
        if n == 0
            return P
        end

        for i in 1:n
            ak02 = 0.5 - i
            pm = pn
            pn = pp
            pp = -2.0 * i * s * pn / y - ak02 * ak02 * pm
        end

        P[end-1] = pn   # P^n_{-1/2}
        P[end] = pp     # P^{n+1}_{-1/2}
    end

    return P
end

"""
    green(x_obs, z_obs, x_source, z_source, dx_dtheta, dz_dtheta, n; gamma_prefactor, uselegacygreenfunction=false)

Compute the Green's function and related quantities for axisymmetric geometry
according to equations (36)-(42) of Chance 1997. Replaces `green` from Fortran code.

# Arguments

  - `x_obs`: Observation point R-coordinate (Float64)
  - `z_obs`: Observation point Z-coordinate (Float64)
  - `x_source`: Source point R-coordinate (Float64)
  - `z_source`: Source point Z-coordinate (Float64)
  - `dx_dtheta`: Derivative ∂R'/∂θ at source point (Float64)
  - `dz_dtheta`: Derivative ∂Z'/∂θ at source point (Float64)
  - `n`: Toroidal mode number (Int)
  - `gamma_prefactor`: Precomputed value of `2√π · Γ(1/2 - n)` [Chance Phys. Plasmas 1997 eq. 40].
    Constant for a given `n`; callers in tight loops should compute this once and pass it in.
    Defaults to `2 * sqrt(π) * gamma(0.5 - n)` if omitted.
  - `uselegacygreenfunction::Bool`: Flag to use the 1997 version of the Legendre function (default false, uses 2007 version)

# Returns

  - `G_n`: 2π𝒢ⁿ(θ,θ′) — Green's function value
  - `coupling_n`: 𝒥 ∇'𝒢ⁿ∇'ℒ — Coupling term for mode n
  - `coupling_0`: 1/(2π) 𝒥 ∇'𝒢⁰∇'ℒ — Coupling term for mode 0

# Notes

  - Uses Legendre functions P^n_{-1/2}(s) computed via elliptic integrals
  - Implements analytical derivatives from Chance 1997 equations
  - The coupling terms include the Jacobian factor from the coordinate transformation
  - By default uses the 2007 Legendre function implementation (Bulirsch + Gaussian integration)

An overload accepting a pre-allocated `legendre_buf::Vector{Float64}` of length `n+2` is available.
Callers in tight loops should allocate this buffer once and pass it in to avoid per-call pool acquisition.
"""
function green(
    x_obs::Float64,
    z_obs::Float64,
    x_source::Float64,
    z_source::Float64,
    dx_dtheta::Float64,
    dz_dtheta::Float64,
    n::Int,
    legendre::AbstractVector{Float64};
    gamma_prefactor::Float64=2 * sqrt(π) * gamma(0.5 - n),
    uselegacygreenfunction::Bool=false
)
    x_obs2 = x_obs^2
    x_source2 = x_source^2
    x_minus2 = (x_obs - x_source)^2
    x_multiple = x_obs * x_source
    ζ = (z_obs - z_source)
    ζ2 = ζ^2

    ρ2 = x_minus2 + ζ2

    # Distance parameter ℛ [Chance Phys. Plasmas 1997 2161 eq. 41]
    R4 = ρ2 * (ρ2 + 4 * x_multiple)
    R2 = sqrt(R4)
    R = sqrt(R2)
    R5 = R4 * R

    # Argument of Legendre function 𝘴 [Chance Phys. Plasmas 1997 2161 eq. 42]
    s = (x_obs2 + x_source2 + ζ2) / R2

    # Legendre functions: P⁰ = p0, P¹ = p1, Pⁿ = pn, Pⁿ⁺¹ = pnp1
    if uselegacygreenfunction
        Pn_minus_half_1997!(legendre, s, n)
    else
        Pn_minus_half_2007!(legendre, s, n)
    end

    p0 = legendre[1]
    p1 = legendre[2]
    pnp1 = legendre[end]
    pn = legendre[end-1]

    # Green's function 2π𝒢ⁿ = G_n [Chance Phys. Plasmas 1997 2161 eq. 40]
    gg = gamma_prefactor / R
    G_n = gg * pn

    # Gradient factor [Chance Phys. Plasmas 1997 2161 eq. 44]
    # NOTE: Paper has erroneous extra factor of 2π
    grad_gg = gg / R4 / 2π

    # Derivatives of Green's function [Chance Phys. Plasmas 1997 2161 eq. 36-38]
    # ∂Gⁿ/∂X' using chain rule: ∂Gⁿ/∂X' = (∂Gⁿ/∂R)(∂R/∂X') + (∂Gⁿ/∂s)(∂s/∂X')
    xterm1 = (n * (x_obs2 + x_source2 + ζ2) * (x_obs2 - x_source2 + ζ2) - x_source2*(x_source2-x_obs2+ζ2)) * pn
    xterm2 = (2.0 * x_source * x_obs * (x_obs2-x_source2+ζ2)) * pnp1
    dG_dX = grad_gg * (xterm1 + xterm2) / x_source

    # ∂Gⁿ/∂Z' using chain rule
    zterm1 = (2.0 * n + 1.0) * (x_obs2 + x_source2 + ζ2) * pn
    zterm2 = 4.0 * x_multiple * pnp1
    dG_dZ = grad_gg * (zterm1 + zterm2) * ζ

    # Coupling term 𝒥 ∇'𝒢ⁿ∇'ℒ [Chance Phys. Plasmas 1997 2161 eq. 51]
    # Jacobian factor from coordinate transformation
    coupling_n = -x_source * (dz_dtheta * dG_dX - dx_dtheta * dG_dZ)

    # Special case for n=0: coupling_0 = 1/(2π) 𝒥 ∇'𝒢⁰∇'ℒ
    dG_dX0_R5 = ((2.0 * x_obs * (x_obs2-x_source2+ζ2)) * p1 - x_source * (x_source2-x_obs2+ζ2) * p0)
    dG_dZ0_R5 = ζ * ((x_obs2 + x_source2 + ζ2) * p0 + 4.0 * x_multiple * p1)
    coupling_0 = -x_source * (dz_dtheta * dG_dX0_R5 - dx_dtheta * dG_dZ0_R5) / R5
    return G_n, coupling_n, coupling_0
end
