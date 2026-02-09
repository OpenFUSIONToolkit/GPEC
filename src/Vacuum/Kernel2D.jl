# 32-point Gaussian quadrature abscissae (used for Pn_minus_half_2007 function when nρ̂>0.1)
const GAUSSIANWEIGHTS32 = [
    0.007018610009470096600, 0.016274394730905670605,
    0.025392065309262059456, 0.034273862913021433103,
    0.042835898022226680657, 0.050998059262376176196,
    0.058684093478535547145, 0.065822222776361846838,
    0.072345794108848506225, 0.078193895787070306472,
    0.083311924226946755222, 0.087652093004403811143,
    0.091173878695763884713, 0.093844399080804565639,
    0.095638720079274859419, 0.096540088514727800567,
    0.096540088514727800567, 0.095638720079274859419,
    0.093844399080804565639, 0.091173878695763884713,
    0.087652093004403811143, 0.083311924226946755222,
    0.078193895787070306472, 0.072345794108848506225,
    0.065822222776361846838, 0.058684093478535547145,
    0.050998059262376176196, 0.042835898022226680657,
    0.034273862913021433103, 0.025392065309262059456,
    0.016274394730905670605, 0.007018610009470096600
]

const GAUSSIANPOINTS32 = [
    -0.997263861849481563545, -0.985611511545268335400,
    -0.964762255587506430774, -0.934906075937739689171,
    -0.896321155766052123965, -0.849367613732569970134,
    -0.794483795967942406963, -0.732182118740289680387,
    -0.663044266930215200975, -0.587715757240762329041,
    -0.506899908932229390024, -0.421351276130635345364,
    -0.331868602282127649780, -0.239287362252137074545,
    -0.144471961582796493485, -0.048307665687738316235,
    0.048307665687738316235, 0.144471961582796493485,
    0.239287362252137074545, 0.331868602282127649780,
    0.421351276130635345364, 0.506899908932229390024,
    0.587715757240762329041, 0.663044266930215200975,
    0.732182118740289680387, 0.794483795967942406963,
    0.849367613732569970134, 0.896321155766052123965,
    0.934906075937739689171, 0.964762255587506430774,
    0.985611511545268335400, 0.997263861849481563545
]

# Cache for Lagrange stencils keyed by Gaussian order
const LAGRANGE_STENCIL_CACHE = Dict{Int,Tuple{Vector{SVector{5,Float64}},Vector{SVector{5,Float64}}}}()

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

"""
    get_lagrange_stencils(gaussian_points)

Return cached 5-point Lagrange stencils keyed by Gaussian order. Initializes
and caches the stencils on first use for a given order.
"""
function get_lagrange_stencils(gaussian_points::AbstractVector{<:Real})
    order = length(gaussian_points)
    return get!(LAGRANGE_STENCIL_CACHE, order) do
        precompute_lagrange_stencils(gaussian_points)
    end
end

"""
    kernel!(grad_greenfunction, greenfunction, observer, source, n)

Compute kernels of integral equation for Laplace's equation in a torus.

**WARNING: This kernel only supports closed toroidal walls currently.
The residue calculation needs to be updated for open walls.**

# Arguments

  - `grad_greenfunction`: Gradient Green's function matrix (output)
  - `greenfunction`: Green's function matrix (output)
  - `observer`: Observer geometry struct (PlasmaGeometry or WallGeometry)
  - `source`: Source geometry struct (PlasmaGeometry or WallGeometry)
  - `n`: Toroidal mode number

# Returns

Modifies `grad_greenfunction` and `greenfunction` in place.
Note that greenfunction is zeroed each time this function is called,
but grad_greenfunction is not since it fills a different block of the
(2 * mtheta, 2 * mtheta) depending on the source/observer.

# Notes

  - Uses Simpson's rule for integration away from singular points
  - Uses Gaussian quadrature near singular points for improved accuracy
  - Implements analytical singularity removal following Chance 1997
"""
function kernel!(
    grad_greenfunction::Matrix{Float64},
    greenfunction::Matrix{Float64},
    observer::Union{PlasmaGeometry,WallGeometry},
    source::Union{PlasmaGeometry,WallGeometry},
    n::Int;
    GAUSS_ORDER::Int=8
)

    mtheta = length(observer.x)
    dtheta = 2π / mtheta
    theta_grid = range(; start=0, length=mtheta, step=dtheta)

    # Take a view of the corresponding block of the grad_greenfunction
    col_index = (source isa PlasmaGeometry ? 1 : 2)
    row_index = (observer isa PlasmaGeometry ? 1 : 2)
    grad_greenfunction_block = view(
        grad_greenfunction,
        ((row_index-1)*mtheta+1):(row_index*mtheta),
        ((col_index-1)*mtheta+1):(col_index*mtheta)
    )

    # Zero out greenfunction at start of each kernel call
    fill!(greenfunction, 0.0)
    # 𝒢ⁿ only needed for plasma as source term (RHS of eqs. 26/27 in Chance 1997)
    populate_greenfunction = source isa PlasmaGeometry

    # S₁ᵢ in Chance 1997, eq.(78)
    log_correction_0=16.0*dtheta*(log(2*dtheta)-68.0/15.0)/15.0
    log_correction_1=128.0*dtheta*(log(2*dtheta)-8.0/15.0)/45.0
    log_correction_2=4.0*dtheta*(7.0*log(2*dtheta)-11.0/15.0)/45.0
    log_correction = [log_correction_2, log_correction_1, log_correction_0, log_correction_1, log_correction_2]

    # Precompute composite Simpson's 1/3 rule weights, excluding singular points
    # Note we set to 4 for even/2 for odd since we index from 1 while the formula assumes indexing from 0
    nsrc = mtheta - 3
    simpson_weights = dtheta / 3 .* [(k == 1 || k == nsrc) ? 1 : (iseven(k) ? 4 : 2) for k in 1:nsrc]

    # Precompute quantities for Gaussian quadrature
    GAUSSIANPOINTS, GAUSSIANWEIGHTS = FastGaussQuadrature.gausslegendre(GAUSS_ORDER) # on [-1, 1]
    wgauss = GAUSSIANWEIGHTS .* dtheta # scale to [-Δθ, Δθ]
    xgauss_left = -dtheta .+ GAUSSIANPOINTS .* dtheta # [-2Δθ, 0]
    xgauss_right = dtheta .+ GAUSSIANPOINTS .* dtheta # [0, 2Δθ]
    lagrange_left, lagrange_right = get_lagrange_stencils(GAUSSIANPOINTS)

    # Set up periodic splines used for off-grid Gaussian quadrature points
    theta_closed = vcat(collect(theta_grid), theta_grid[end] + dtheta)
    x_closed = vcat(source.x, source.x[1])
    z_closed = vcat(source.z, source.z[1])
    spline_x = cubic_interp(theta_closed, x_closed; bc=PeriodicBC())
    spline_z = cubic_interp(theta_closed, z_closed; bc=PeriodicBC())
    d1_spline_x = deriv1(spline_x)
    d1_spline_z = deriv1(spline_z)

    # Loop through observer points
    for j in 1:mtheta
        # Get observer coordinates
        x_obs, z_obs, theta_obs = observer.x[j], observer.z[j], theta_grid[j]

        # Obtain nonsingular region (endpoints at j+2 and j-2, so exclude j-1, j, and j+1)
        nonsing_idx = mod1.((j+2):(j+mtheta-2), mtheta) # mod1 ensures isrc is in [1, mtheta]

        # Perform Simpson integration for nonsingular source points
        for (isrc, wsimpson) in zip(nonsing_idx, simpson_weights)
            G_n, gradG_n, gradG_0 = green(x_obs, z_obs, source.x[isrc], source.z[isrc], source.dx_dtheta[isrc], source.dz_dtheta[isrc], n)

            # Sum contributions to Green's function matrices using Simpson weight
            if populate_greenfunction
                greenfunction[j, isrc] += G_n * wsimpson
            end
            grad_greenfunction_block[j, isrc] += gradG_n * wsimpson
            # Subtract regular integral component of δⱼᵢK⁰ in eq. 83
            grad_greenfunction_block[j, j] -= gradG_0 * wsimpson
        end

        # Perform Gaussian quadrature for singular points (source = obs point)
        # Get indices of the singularity region, [j-2, j-1, j, j+1, j+2]
        sing_idx = mod1.(j .+ ((mtheta-2):(mtheta+2)), mtheta)
        # Integrate region of length 2 * dtheta on left/right of singularity
        for region in ["left", "right"]
            # Get precomputed quadrature data
            lagrange_stencil = (region == "left") ? lagrange_left : lagrange_right
            theta_gauss = theta_obs .+ (region == "left" ? xgauss_left : xgauss_right)
            for ig in 1:GAUSS_ORDER
                # Compute green function for this Gaussian point
                theta_gauss0 = mod(theta_gauss[ig], 2π)
                x_gauss = spline_x(theta_gauss0)
                dx_dtheta_gauss = d1_spline_x(theta_gauss0)
                z_gauss = spline_z(theta_gauss0)
                dz_dtheta_gauss = d1_spline_z(theta_gauss0)
                G_n, gradG_n, gradG_0 = green(x_obs, z_obs, x_gauss, z_gauss, dx_dtheta_gauss, dz_dtheta_gauss, n)

                # First type of singularity: 𝒢ⁿ (Eq. 75: 2π𝒢ⁿ + log(θ-θ')²/X')
                if populate_greenfunction
                    if observer isa PlasmaGeometry
                        # Remove singular behavior by adding on leading-order term, Chance eq.(75)
                        G_n += log((theta_obs - theta_gauss[ig])^2) / x_obs
                    end
                    @. @views greenfunction[j, sing_idx] += wgauss[ig] * G_n * lagrange_stencil[ig]
                end

                # Second type of singularity: 𝒦ⁿ (Eq. 86: 𝒦ⁿαᵢ - δⱼᵢK⁰)
                @. @views grad_greenfunction_block[j, sing_idx] += wgauss[ig] * gradG_n * lagrange_stencil[ig]
                grad_greenfunction_block[j, j] -= gradG_0 * wgauss[ig]
            end
        end

        # Add analytic singular integral (first type) from Chance eq. 78
        if populate_greenfunction && observer isa PlasmaGeometry
            @. @views greenfunction[j, sing_idx] -= log_correction / x_obs
        end
    end

    # Normals need to point outward from vacuum region. In CCW θ convention, normal points
    # out of vacuum for plasma but inward for wall, so we multiply by -1 for wall sources
    if source isa WallGeometry
        grad_greenfunction_block .*= -1
    end

    # Since we computed 2π𝒢, divide by 2π to get 𝒢
    greenfunction ./= 2π

    # Add analytic singular integral (second type) from Table I of Chance 1997 + existing δⱼᵢ in eq. 69
    # Would need to pass in wall geometry to generalize this to open walls
    is_closed_toroidal = true
    if is_closed_toroidal # Chance eq. 89
        residue = (observer isa WallGeometry) ? 0.0 : (source isa PlasmaGeometry ? 2.0 : -2.0)
    else # Chance eq. 90
        # TODO: this line can be gotten rid of if we are never doing open walls
        residue = (typeof(observer) == typeof(source)) ? 2.0 : 0.0
    end
    # Add residue value from eq. 89/90 to block diagonal
    @inbounds for i in 1:mtheta
        grad_greenfunction_block[i, i] += residue
    end
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

    #initialize
    P = zeros(n + 2)

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

    # Constants
    sqpi = sqrt(π)
    pii = 2.0 / π
    sqtwo = sqrt(2.0)

    # Initialize output array
    P = zeros(n + 2)

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

        # Integration limits
        xl = 0.0
        xu = 5.0

        # Transform to integration interval
        agaus = 0.5 * (xu + xl)
        bgaus = 0.5 * (xu - xl)

        # Calculate integrals for P^n and P^{n+1}
        gint = 0.0
        gintp = 0.0

        for ig in 1:32
            tg0 = agaus + GAUSSIANPOINTS32[ig] * bgaus
            tg02 = tg0 * tg0
            tg1 = tg02 / (2.0 * n)
            tg1p = tg02 / (2.0 * n + 2.0)
            sinhtg1 = sinh(tg1)
            sinhtg1p = sinh(tg1p)
            sinhtg12 = sinhtg1 * sinhtg1
            sinhtg12p = sinhtg1p * sinhtg1p
            dnom = s * sinhtg12 + sinhtg1 * sqrt(1.0 + sinhtg12)
            dnomp = s * sinhtg12p + sinhtg1p * sqrt(1.0 + sinhtg12p)
            dnom = sqrt(dnom)
            dnomp = sqrt(dnomp)
            anumr = tg0 * exp(-tg02)
            gint += GAUSSIANWEIGHTS32[ig] * anumr / dnom
            gintp += GAUSSIANWEIGHTS32[ig] * anumr / dnomp
        end

        gint *= bgaus
        gintp *= bgaus

        # Calculate coefficients
        pcoef = sqrt((s - 1.0) / (s + 1.0))

        # Gamma functions: Gamma[1/2 - n] and Gamma[1/2 - (n+1)]
        gamn = sqpi
        gamp = -2.0 * sqpi

        if n != 0
            # Compute Gamma[1/2 - n] = sqpi / product(-(i-1) - 0.5 for i in 1:n)
            gamn = sqpi / prod(-(i - 1) - 0.5 for i in 1:n)
            gamp = -gamn / (n + 0.5)
        end

        # Final Legendre function values
        gint = sqtwo * pcoef^n * gint / (n * sqpi * gamn)
        gintp = sqtwo * pcoef^(n + 1) * gintp / ((n + 1.0) * sqpi * gamp)

        P[end-1] = gint   # P^n_{-1/2}
        P[end] = gintp    # P^{n+1}_{-1/2}

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
    green(x_obs, z_obs, x_source, z_source, dx_dtheta, dz_dtheta, n; uselegacygreenfunction=false)

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
"""
function green(x_obs::Float64, z_obs::Float64, x_source::Float64, z_source::Float64, dx_dtheta::Float64, dz_dtheta::Float64, n::Int; uselegacygreenfunction::Bool=false)

    x_obs2 = x_obs^2
    x_source2 = x_source^2
    x_minus2 = (x_obs - x_source)^2
    x_multiple = x_obs * x_source
    ζ = (z_obs - z_source)
    ζ2 = ζ^2

    ρ2 = x_minus2 + ζ2

    # Chance 1997 eq.(41) ℛ = R
    R4 = ρ2 * (ρ2 + 4 * x_multiple)
    R2 = sqrt(R4)
    R = sqrt(R2)
    R5 = R4 * R

    # Chance 1997 eq.(42) 𝘴 = s
    s = (x_obs2 + x_source2 + ζ2) / R2

    # Legendre functions for
    # P⁰ = p0, P¹ = p1, Pⁿ = pn, Pⁿ⁺¹ = pnp1
    if uselegacygreenfunction
        legendre = Pn_minus_half_1997(s, n)
    else
        legendre = Pn_minus_half_2007(s, n)
    end

    p0 = legendre[1]
    p1 = legendre[2]
    pnp1 = legendre[end]
    pn = legendre[end-1]

    # Chance 1997 eq.(40) 2π𝒢ⁿ = G_n
    gg = 2 * sqrt(π) * gamma(0.5 - n) / R
    G_n = gg * pn

    # Chance 1997 eq.(44) (Note this equation in the paper has an erroneous extra factor of 2π)
    grad_gg = gg / R4 / 2π

    # ∂Gⁿ/∂X' = dG_dX
    xterm1 = (n * (x_obs2 + x_source2 + ζ2) * (x_obs2 - x_source2 + ζ2) - x_source2*(x_source2-x_obs2+ζ2)) * pn
    xterm2 = (2.0 * x_source * x_obs * (x_obs2-x_source2+ζ2)) * pnp1
    dG_dX = grad_gg * (xterm1 + xterm2) / x_source

    # ∂Gⁿ/∂Z' = dG_dZ
    zterm1 = (2.0 * n + 1.0) * (x_obs2 + x_source2 + ζ2) * pn
    zterm2 = 4.0 * x_multiple * pnp1
    dG_dZ = grad_gg * (zterm1 + zterm2) * ζ

    # Chance 1997 eq.(51)
    # 𝒥 ∇'𝒢ⁿ∇'ℒ = aval
    # ∂X'/∂θ = xtp, ∂Z'/∂θ = ztp
    coupling_n = -x_source * (dz_dtheta * dG_dX - dx_dtheta * dG_dZ)

    # for 𝓃⩵0,  aval0 = 1/(2π) 𝒥 ∇'𝒢⁰∇'ℒ
    dG_dX0_R5 = ((2.0 * x_obs * (x_obs2-x_source2+ζ2)) * p1 - x_source * (x_source2-x_obs2+ζ2) * p0)
    dG_dZ0_R5 = ζ * ((x_obs2 + x_source2 + ζ2) * p0 + 4.0 * x_multiple * p1)
    coupling_0 = -x_source * (dz_dtheta * dG_dX0_R5 - dx_dtheta * dG_dZ0_R5) / R5
    return G_n, coupling_n, coupling_0
end
