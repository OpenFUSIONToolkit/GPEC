# Gaussian quadrature weights and points for 8-point integration (used for kernel! function)
const GAUSSIANWEIGHTS = [0.101228536290376, 0.222381034453374, 0.313706645877887, 0.362683783378362,
    0.362683783378362, 0.313706645877887, 0.222381034453374, 0.101228536290376]

const GAUSSIANPOINTS = [-0.960289856497536, -0.796666477413627, -0.525532409916329, -0.183434642495650,
    0.183434642495650, 0.525532409916329, 0.796666477413627, 0.960289856497536]

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

"""
    kernel!(grad_greenfunction_mat, greenfunction_mat, x_obspoints, z_obspoints, x_sourcepoints, z_sourcepoints, j1, j2, isgn, iops, inputs; xwall=nothing, zwall=nothing)

Compute kernels of integral equation for Laplace's equation in a torus.

**WARNING: This kernel only supports closed toroidal walls currently.
The residue calculation needs to be updated for open walls.**

# Arguments

  - `grad_greenfunction_mat`: Gradient Green's function matrix (output)
  - `greenfunction_mat`: Green's function matrix (output)
  - `x_obspoints`: Observer x coordinates (R coordinates)
  - `z_obspoints`: Observer z coordinates (Z coordinates)
  - `x_sourcepoints`: Source x coordinates (R coordinates)
  - `z_sourcepoints`: Source z coordinates (Z coordinates)
  - `j1/j2`: Block index for observer/source (1=plasma, 2=wall)
  - `n`: Toroidal mode number

# Returns

Modifies `grad_greenfunction_mat` and `greenfunction_mat` in place.
Note that greenfunction_mat is zeroed each time this function is called,
but grad_greenfunction_mat is not since it fills a different block of the
(2 * mtheta, 2 * mtheta) depending on the source/observer.

# Notes

  - Uses Simpson's rule for integration away from singular points
  - Uses Gaussian quadrature near singular points for improved accuracy
  - Implements analytical singularity removal following Chance 1997
"""
function kernel!(
    grad_greenfunction_mat::Matrix{Float64},
    greenfunction_mat::Matrix{Float64},
    x_obspoints::Vector{Float64},
    z_obspoints::Vector{Float64},
    x_sourcepoints::Vector{Float64},
    z_sourcepoints::Vector{Float64},
    j1::Int,
    j2::Int,
    n::Int
)

    # These used to be function arguments, but can just set inside here based on j1/j2
    plasma_plasma_block = j1 == 1 && j2 == 1 # previously iops
    plasma_is_source = j2 == 1 # previously iopw
    isgn = plasma_is_source ? -1 : 1

    mtheta = length(x_obspoints)
    dtheta = 2π / mtheta
    theta_grid = range(; start=0, length=mtheta, step=dtheta)

    # Zero out greenfunction_mat at start of each kernel call (matches Fortran behavior)
    fill!(greenfunction_mat, 0.0)

    if mtheta != length(z_obspoints) || mtheta != length(x_sourcepoints) || mtheta != length(z_sourcepoints)
        error("Length of input arrays (xobs, zobs, xsource, zsce) are different. All length should be the same")
    end

    # S₁ᵢ in Chance 1997, eq.(78)
    log_correction_0=16.0*dtheta*(log(2*dtheta)-68.0/15.0)/15.0
    log_correction_1=128.0*dtheta*(log(2*dtheta)-8.0/15.0)/45.0
    log_correction_2=4.0*dtheta*(7.0*log(2*dtheta)-11.0/15.0)/45.0

    has_zero_crossing = false
    # See if the conductor surface crosses R=0 (x=0) anywhere
    if !plasma_plasma_block
        # Initialize jbot and jtop, and figure out wall geometry based on which block we are in
        jbot = jtop = mtheta/2+1
        xwall = plasma_is_source ? x_obspoints : x_sourcepoints
        # Find if sign of wall x point crosses zero
        for i in 1:mtheta
            next_i = i == mtheta ? 1 : i + 1
            if xwall[i] * xwall[next_i] <= 0.0
                jbot = xwall[i] > 0.0 ? i : # index where the wall leaves positive R
                       jtop = xwall[i] < 0.0 ? i + 1 : jtop # index where the wall returns to positive R
                has_zero_crossing = true
            end
        end
    end

    # Used for Z'_θ and X'_θ in eq.(51)
    spline_x = cubic_spline_interpolation(theta_grid, x_sourcepoints; extrapolation_bc=Interpolations.Periodic())
    spline_z = cubic_spline_interpolation(theta_grid, z_sourcepoints; extrapolation_bc=Interpolations.Periodic())
    dx_dtheta = [Interpolations.gradient(spline_x, t)[1] for t in theta_grid]
    dz_dtheta = [Interpolations.gradient(spline_z, t)[1] for t in theta_grid]

    # Loop through observer points
    for j in 1:mtheta
        # Initialize variables
        x_obs=x_obspoints[j]
        z_obs=z_obspoints[j]
        theta_obs=theta_grid[j]
        grad_green_0 = 0.0 # simpson integral for coupling_0 (𝒥 ∇'𝒢⁰∇'ℒ)
        # Workspace = view of appropriate row of grad_greenfunction_mat for this observer point
        grad_green_work = @view(grad_greenfunction_mat[(j1-1)*mtheta+j, (j2-1)*mtheta .+ (1:mtheta)])

        # if observation point is negative, we cannot use green function
        if x_obs < 0.0
            !plasma_is_source && (grad_green_work[j] = 1.0)
            continue
        end

        # Compute istart and iend (start/end index of integration to avoid singularity)
        # If no zero crossing, istart = iend = 2
        iend = 2
        if has_zero_crossing && !plasma_is_source
            # Determine iend/istart so Simpson sweep avoids integrating across the (x=0) discontinuity near the observer index j
            if jbot - j == 1
                iend = 3
            elseif jbot - j == 0
                iend = 4
            elseif j - jtop == 0
                iend = 0
            elseif j - jtop == 1
                iend = 1
            end
        end
        istart = 4 - iend

        # Perform Simpson integration for nonsingular source points (excludes 3 singular points)
        # For cases where wall doesn't cross x=0 (iend = istart = 2), the singular points are j-1, j, j+1
        for i in 1:(mtheta-3)
            # Get source point index (ic) and ensure it is in range [1, mtheta]
            ic = i + j + istart - 1
            if ic > mtheta
                ic = ic - mtheta
            end
            x_source=x_sourcepoints[ic]
            z_source=z_sourcepoints[ic]

            # if source point is negative, we cannot use green function
            # & if source point(ic) and obs point (j) is same, it's singular
            (x_source < 0 || ic == j) && continue

            # G_n is 2pi𝒢ⁿ; coupling_n is 𝒥 ∇'𝒢ⁿ∇'ℒ; coupling_0 is 𝒥 ∇'𝒢ⁿ∇'ℒ for n=0
            G_n, coupling_n, coupling_0 = green(x_obs, z_obs, x_source, z_source, dx_dtheta[ic], dz_dtheta[ic], n)

            # Compute composite Simpson's 1/3 rule weight (https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_1/3_rule)
            # Note we set to 4 for even/2 for odd since we index from 1 while the formula assumes indexing from 0
            endpoint = (i == 1)||(i == mtheta - 3)
            wsimpson = (endpoint ? 1 : (iseven(i) ? 4 : 2)) * dtheta / 3

            # Sum contributions to Green's function matrices using Simpson weight
            grad_green_work[ic] += isgn * coupling_n * wsimpson
            greenfunction_mat[j, ic] += G_n * wsimpson
            grad_green_0 += coupling_0 * wsimpson
        end

        # Skip singularity calculation for R < 0 wall points
        if has_zero_crossing && j > jbot && j < jtop
            continue
        end

        # Perform Gaussian quadrature for singular points (source = obs point)
        # Get indices of the singularity region ([j-2, j-1, j, j+1, j+2] for iend = 2)
        js = mod.(j - iend .+ ((mtheta-1):(mtheta+3)), mtheta) .+ 1
        # Integrate region of length 2 * dtheta on left (ilr = 1)/right (ilr = 2) of singularity
        for ilr in [1, 2]
            gauss_xleft = theta_obs + (2*ilr-iend-2)*dtheta
            gauss_xright = gauss_xleft + 2 * dtheta
            gauss_xavg = (gauss_xright + gauss_xleft)/2
            theta_gauss = gauss_xavg .+ GAUSSIANPOINTS .* dtheta # tgaus is 8 point gauss points, since GAUSSIANPOINTS is for only [-1,1]
            for ig in 1:8 # 8-point Gaussian quadrature
                # Compute green function for this Gaussian point
                theta_gauss0 = mod(theta_gauss[ig], 2π)
                x_gauss = spline_x(theta_gauss0)
                dx_dtheta_gauss = Interpolations.gradient(spline_x, theta_gauss0)[1]
                z_gauss = spline_z(theta_gauss0)
                dz_dtheta_gauss = Interpolations.gradient(spline_z, theta_gauss0)[1]
                G_n, coupling_n, coupling_0 = green(x_obs, z_obs, x_gauss, z_gauss, dx_dtheta_gauss, dz_dtheta_gauss, n)

                # Add logarithm to G_n to analytically isolate the singularity (first type), Chance eq.(75)
                G_n_nonsingular = plasma_plasma_block ? G_n + log((theta_obs-theta_gauss[ig])^2)/x_obs : G_n

                # Redefine hardcoded Gaussian weights on the interval [-1, 1] to physical interval with length 2 * dtheta
                wgauss = GAUSSIANWEIGHTS[ig] * dtheta
                # Calculate p = θ/Δ = (θⱼ - θ')/Δ, 0 at observation point, ±1,±2 at other 5-point stencil nodes
                pgauss=(theta_gauss[ig]-theta_obs-(2-iend)*dtheta)/dtheta
                # Compute 5-point Lagrange basis polynomials at the Gauss point and multiply by quadrature weight
                A0 = (pgauss^2-1)*(pgauss^2-4)/4.0 * wgauss
                A1_plus = -(pgauss+1)*pgauss*(pgauss^2-4)/6.0 * wgauss
                A1_minus = -(pgauss-1)*pgauss*(pgauss^2-4)/6.0 * wgauss
                A2_plus = (pgauss^2-1)*pgauss*(pgauss+2)/24.0 * wgauss
                A2_minus = (pgauss^2-1)*pgauss*(pgauss-2)/24.0 * wgauss

                # First type of singularity: 𝒢ⁿ, occurs plasma as source only (see RHS of Chance eqs. 26/27)
                if plasma_is_source
                    greenfunction_mat[j, js[1]] += G_n_nonsingular * A2_minus
                    greenfunction_mat[j, js[2]] += G_n_nonsingular * A1_minus
                    greenfunction_mat[j, js[3]] += G_n_nonsingular * A0
                    greenfunction_mat[j, js[4]] += G_n_nonsingular * A1_plus
                    greenfunction_mat[j, js[5]] += G_n_nonsingular * A2_plus
                end

                # Second type of singularity: 𝒦ⁿ
                # Eq. 86: 𝒦ⁿαᵢ - δⱼᵢK⁰ (js[3] = j if iend=2)
                grad_green_work[js[1]] += isgn * coupling_n * A2_minus
                grad_green_work[js[2]] += isgn * coupling_n * A1_minus
                grad_green_work[js[3]] += isgn * coupling_n * A0
                grad_green_work[js[4]] += isgn * coupling_n * A1_plus
                grad_green_work[js[5]] += isgn * coupling_n * A2_plus
                # Subtract off the diverging singular n=0 component
                grad_green_work[j] -= isgn * coupling_0 * wgauss
            end
        end

        # Set residue based on logic similar to Table I of Chance 1997 + existing δⱼᵢ in eq. 69
        # Would need to pass in wall geometry to generalize this to open walls
        is_closed_toroidal = true
        if is_closed_toroidal
            residue = (j1 == 2.0) ? 0.0 : (j2 == 1 ? 2.0 : -2.0) # Chance eq. 89
        else
            # TODO: this line can be gotten rid of if we are never doing open walls
            residue = (j1 == j2) ? 2.0 : 0.0 # Chance eq. 90
        end
        # Subtract regular integral component of δⱼᵢK⁰ in eq. 83 and add residue value in eq. 89/90
        grad_green_work[j] = grad_green_work[j] - isgn * grad_green_0 + residue

        # Subtract off analytic singular integral from Chance eq.(75) if plasma-plasma block
        if plasma_plasma_block
            greenfunction_mat[j, js[1]] -= log_correction_2 / x_obs
            greenfunction_mat[j, js[2]] -= log_correction_1 / x_obs
            greenfunction_mat[j, js[3]] -= log_correction_0 / x_obs
            greenfunction_mat[j, js[4]] -= log_correction_1 / x_obs
            greenfunction_mat[j, js[5]] -= log_correction_2 / x_obs
        end
    end
    # Since we computed 2π𝒢, divide by 2π to get 𝒢
    greenfunction_mat ./= 2π
end

"""
    fourier_inverse_transform!(gll, gil, cs, m00, l00)

Perform the inverse Fourier transform of `gil` onto `gll` using Fourier coefficients stored in `cs`.

# Arguments

  - `gll`: Output matrix (mpert × mpert) updated in-place
  - `gil`: Input matrix (mtheta × mpert) containing Fourier-space data
  - `cs`: Fourier coefficient matrix (mtheta × mpert)
  - `m00`: Integer offset in the gil matrix (row offset)
  - `l00`: Integer offset in the gil matrix (column offset)

# Notes

  - Computes: `gll[l2, l1] = (2π * dth) * Σ_i cs[i, l2] * gil[i, l1]`
  - Performs the same function as fouranv in the Fortran code.

# Returns

  - gll(l2,l1) : output matrix updated in-place (mpert × mpert)
"""
function fourier_inverse_transform!(gll::Matrix{Float64}, gil::Matrix{Float64}, cs::Matrix{Float64}, m00::Int, l00::Int)

    # Zero out gll block
    mtheta, mpert = size(cs)
    fill!(view(gll, 1:mpert, 1:mpert), 0.0)

    # Inverse Fourier transform via matrix multiply: gll = cs^T * gil * (2π * dth)
    # This computes: gll[l2, l1] = (2π * dth) * Σ_i cs[i, l2] * gil[i, l1]
    dth = 2π / mtheta
    mul!(gll, cs', view(gil, (m00+1):(m00+mtheta), (l00+1):(l00+mpert)), 2π * dth, 0.0)
end

"""
    fourier_transform!(gil, gij, cs, m00, l00, mth, mpert)

    Purpose:
      This routine performs a truncated Fourier transform of gij onto gil
      using Fourier coefficients stored in cs.

    Inputs:
      gij(i,j)   : input matrix of size (mth × mth), the "physical-space" data
      cs(j,l)    : Fourier coefficient matrix (mth × mpert)
      m00, l00   : integer offsets in the gil matrix
      mth        : number of θ-grid points (dimension of gij along i, j)
      mpert      : number of Fourier modes

    Output:
      gil(i', l') : output matrix updated in-place (mth × mpert), where i' = m00 + i and l' = l00 + l
"""
function fourier_transform!(gil::Matrix{Float64}, gij::Matrix{Float64}, cs::Matrix{Float64}, m00::Int, l00::Int)

    # Zero out relevant gil block
    mtheta, mpert = size(cs)
    fill!(view(gil, (m00+1):(m00+mtheta), (l00+1):(l00+mpert)), 0.0)

    # Fourier transform via matrix multiply: gil[i, l] = Σ_j gij[i, j] * cs[j, l]
    mul!(view(gil, (m00+1):(m00+mtheta), (l00+1):(l00+mpert)), gij, cs)
end

# Returns the array of derivatives at all x points, I think this acts like difspl
# in the Fortran but need to check/consolidate spline routines later
function periodic_cubic_deriv(theta, vals)
    itp = scale(interpolate(vals, BSpline(Cubic(Periodic(OnGrid())))), theta)
    return first.(Interpolations.gradient.(Ref(itp), theta))
end

"""
    lagrange1d(ax, af, m, nl, x, iop)

Perform Lagrange interpolation and optionally compute its derivative.
Replaces `lagp`, `lagpe4`, and `lag` from Fortran code.

# Arguments

  - `ax::Vector{Float64}`: Array of x-coordinates for the interpolation points
  - `af::Vector{Float64}`: Array of y-coordinates (function values) for the interpolation points
  - `m::Int`: Number of interpolation points
  - `nl::Int`: Number of points to use for the local interpolation (polynomial degree + 1)
  - `x::Float64`: The x-value at which to evaluate the interpolated function and/or its derivative
  - `iop::Int`: Flag controlling output (0 = value only, 1 = value and derivative)

# Returns

  - `f::Float64`: The interpolated function value at `x`
  - `df::Float64`: The interpolated function derivative at `x` (0.0 if iop=0)

# Notes

  - Uses local Lagrange interpolation with `nl` points centered around `x`
  - Automatically adjusts interpolation window to stay within array bounds
"""
function lagrange1d(ax::Vector{Float64}, af::Vector{Float64}, m::Int, nl::Int, x::Float64, iop::Int)

    # --- Error fix: Initialize f and df internally to 0.0 ---
    f::Float64 = 0.0
    df::Float64 = 0.0

    jn = findfirst(i -> ax[i] >= x, 1:m)
    jn = (jn === nothing) ? m : jn
    jn = max(jn - 1, 1)
    if jn < m && abs(ax[jn+1] - x) < abs(x - ax[jn])
        jn += 1
    end

    # Determine the range of indices for interpolation
    jnmm = floor(Int, (nl - 0.1) / 2)
    jnpp = floor(Int, (nl + 0.1) / 2)

    nll = jn - jnmm
    nlr = jn + jnpp

    # Adjust for even nl when ax[jn] > x (Window shift)
    if (nl % 2 == 0) && (ax[jn] > x)
        nll -= 1
        nlr -= 1 # <--- Shifting the window (was nlr += 1)
    end

    # Clamp indices to valid array bounds
    if nlr > m
        nlr = m
        nll = nlr - nl + 1
    elseif nll < 1
        nll = 1
        nlr = nl
    end

    # Compute function value f
    for i in nll:nlr
        alag = 1.0
        for j in nll:nlr
            (i == j) && continue
            alag *= (x - ax[j]) / (ax[i] - ax[j])
        end
        f += alag * af[i]
    end

    # --- Error fix: Use the 'iop' argument ---
    (iop == 0) && return f, df # df is returned as 0.0

    # Compute derivative df
    for i in nll:nlr
        slag = 0.0
        for id in nll:nlr
            (id == i) && continue
            alag = 1.0
            for j in nll:nlr
                (j == i) && continue
                alag *= (j != id) ? ((x - ax[j]) / (ax[i] - ax[j])) : (1.0 / (ax[i] - ax[id]))
            end
            slag += alag
        end
        df += slag * af[i]
    end

    return f, df # Return value and derivative
end

"""
    interp_to_new_grid(vecin, mtheta; dx0=0.0, dx1=0.0)

Resample the input array `vecin` using a periodic cubic spline to an output array of length `mtheta`.

This function unifies the Fortran functions `trans`, `transdx`, and `transdxx` into a single
function with optional offset parameters.

# Arguments

  - `vecin::Vector{Float64}`: Input array to be resampled
  - `mtheta::Int`: Desired length of the output array
  - `dx0::Float64`: Global offset added to all x-coordinates (default 0, applied as `x += dx0 / mtheta_in`)
  - `dx1::Float64`: Fine offset added to each index (default 0, applied as `ai = (i-1) + dx1`)

# Returns

  - `vecout::Vector{Float64}`: The resampled output array (length `mtheta`)

# Notes

  - If `mtheta == length(vecin)`, returns the input vector unchanged
  - Uses periodic cubic spline interpolation for resampling
  - Input grid is normalized to [0, 1] for interpolation
"""
function interp_to_new_grid(vecin::Vector{Float64}, mtheta::Int; dx0=0.0, dx1=0.0)

    # Initialize
    mtheta_in = length(vecin)

    # If mtheta == mtheta_in, just return the input vector
    if mtheta == mtheta_in
        return vecin
    end

    # Input grids are from [0, 1] inclusive, since no interpolants will fall outside of this, we don't need periodic extrapolation
    θin = range(0.0, 1.0; length=mtheta_in)
    itp = cubic_spline_interpolation(θin, vecin)

    # Interpolate to new grid with optional offsets
    vecout = zeros(mtheta)
    for i in 1:mtheta
        x = (i - 1 + dx1) / mtheta + dx0 / mtheta_in
        x = x % 1.0  # This is for periodicity in the case of dx1/dx0 ≠ 0
        vecout[i] = itp(x)
    end
    return vecout
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

# Helper functions for compute_vacuum_field

"""
    _pickup_field(inputs, plasma_surf, grri, Bn_real, Bn_imag, R_grid, Z_grid)

Calculate the magnetic field on a specified grid using finite differencing
of the magnetic scalar potential `chi`.

This is the Julia version of the Fortran `pickup` routine. It computes the vacuum
magnetic field perturbation at a set of grid points given the plasma surface perturbation.

# Arguments

  - `inputs::VacuumInput`: Struct containing vacuum calculation parameters
  - `plasma_surf::PlasmaGeometry`: Struct with plasma surface geometry and basis functions
  - `grri::Matrix{Float64}`: Inverted Green's function response matrix from vaccal!
  - `Bn_real::Vector{Float64}`: Real part of normal field Fourier harmonics at plasma surface
  - `Bn_imag::Vector{Float64}`: Imaginary part of normal field Fourier harmonics at plasma surface
  - `R_grid::AbstractVector`: R-coordinates for output field evaluation
  - `Z_grid::AbstractVector`: Z-coordinates for output field evaluation

# Returns

  - `B_R::Matrix{ComplexF64}`: R-component of magnetic field on grid (nx × nz)
  - `B_Z::Matrix{ComplexF64}`: Z-component of magnetic field on grid (nx × nz)
  - `B_phi::Matrix{ComplexF64}`: Toroidal component of magnetic field on grid (nx × nz)
  - `grid_info::Matrix{Int}`: Grid point classification (1=inside plasma, 0=outside)

# Notes

  - Uses 5-point finite difference stencil for computing field from potential
  - Field components computed as: B_R = -∂χ/∂R, B_Z = -∂χ/∂Z, B_φ = inχ/R
"""
function _pickup_field(inputs::VacuumInput, plasma_surf::PlasmaGeometry, grri::Matrix{Float64},
    Bn_real::Vector{Float64}, Bn_imag::Vector{Float64},
    R_grid::AbstractVector, Z_grid::AbstractVector)

    nx = length(R_grid)
    nz = length(Z_grid)
    ifac = 1im

    # Create the grid of points where the potential will be calculated
    R_points, Z_points = _create_pickup_grid(R_grid, Z_grid)
    n_points = length(R_points)

    # Output arrays
    B_R_complex = zeros(ComplexF64, nx, nz)
    B_Z_complex = zeros(ComplexF64, nx, nz)
    B_phi_complex = zeros(ComplexF64, nx, nz)
    grid_info = zeros(Int, nx, nz)

    # Finite difference steps
    del_R = 1e-5 * (maximum(plasma_surf.x) - minimum(plasma_surf.x))
    del_Z = 1e-5 * (maximum(plasma_surf.x) - minimum(plasma_surf.x))

    # Calculate potential `chi` at 5 points for each grid location for finite differencing
    # 1: (R, Z + dZ), 2: (R, Z - dZ), 3: (R + dR, Z), 4: (R - dR, Z), 5: (R, Z)
    chi_r = zeros(5, n_points)
    chi_i = zeros(5, n_points)

    Threads.@threads for i in 1:n_points
        R, Z = R_points[i], Z_points[i]

        # Points for finite difference stencil
        observe_points = [
            (R, Z + del_Z),
            (R, Z - del_Z),
            (R + del_R, Z),
            (R - del_R, Z),
            (R, Z)
        ]

        for (j, (obs_R, obs_Z)) in enumerate(observe_points)
            chi_r[j, i], chi_i[j, i] = _calculate_potential_chi(
                obs_R, obs_Z, inputs, plasma_surf, grri, Bn_real, Bn_imag
            )
        end
    end

    # Calculate fields using finite differences and reshape into a grid
    for i in 1:nx
        for j in 1:nz
            idx = (i - 1) * nz + j

            # B_R = -d(chi)/dR, B_Z = -d(chi)/dZ
            br_r = -(chi_r[3, idx] - chi_r[4, idx]) / (2.0 * del_R)
            br_i = -(chi_i[3, idx] - chi_i[4, idx]) / (2.0 * del_R)

            bz_r = -(chi_r[1, idx] - chi_r[2, idx]) / (2.0 * del_Z)
            bz_i = -(chi_i[1, idx] - chi_i[2, idx]) / (2.0 * del_Z)

            # B_phi = i*n*chi / R
            # Bphi = i*n*(chi_r + i*chi_i)/R = (-n*chi_i + i*n*chi_r)/R
            bphi_r = -inputs.n * chi_i[5, idx] / R_points[idx]
            bphi_i = inputs.n * chi_r[5, idx] / R_points[idx]

            B_R_complex[i, j] = br_r + ifac * br_i
            B_Z_complex[i, j] = bz_r + ifac * bz_i
            B_phi_complex[i, j] = bphi_r + ifac * bphi_i

            # Check if point is inside the plasma
            fintjj = 0.0
            for k in 1:inputs.mtheta
                dx = R_points[idx] - plasma_surf.x[k]
                dz = Z_points[idx] - plasma_surf.z[k]
                rho2 = dx^2 + dz^2
                if rho2 > 1e-16
                    fintjj += (plasma_surf.dz_dtheta[k] * dx - plasma_surf.dx_dtheta[k] * dz) / rho2
                end
            end
            grid_info[i, j] = (fintjj > 0.1) ? 1 : 0 # 1 for interior, 0 for exterior
        end
    end

    return B_R_complex, B_Z_complex, B_phi_complex, grid_info
end

"""
    _create_pickup_grid(R_grid, Z_grid)

Create a flattened 1D list of (R, Z) coordinates from 2D grid vectors.

This is the Julia version of the Fortran `loops` subroutine.

# Arguments

  - `R_grid::AbstractVector`: Vector of R-coordinates defining the grid
  - `Z_grid::AbstractVector`: Vector of Z-coordinates defining the grid

# Returns

  - `R_points::Vector{Float64}`: Flattened array of R-coordinates (length nx*nz)
  - `Z_points::Vector{Float64}`: Flattened array of Z-coordinates (length nx*nz)

# Notes

  - Grid points are ordered as: [(R[1],Z[1]), (R[1],Z[2]), ..., (R[1],Z[nz]), (R[2],Z[1]), ...]
"""
function _create_pickup_grid(R_grid::AbstractVector, Z_grid::AbstractVector)
    nx = length(R_grid)
    nz = length(Z_grid)
    R_points = zeros(Float64, nx * nz)
    Z_points = zeros(Float64, nx * nz)

    for i in 1:nx
        for j in 1:nz
            idx = (i - 1) * nz + j
            R_points[idx] = R_grid[i]
            Z_points[idx] = Z_grid[j]
        end
    end
    return R_points, Z_points
end

"""
    _calculate_potential_chi(R_obs, Z_obs, inputs, plasma_surf, grri, Bn_real, Bn_imag)

Calculate the magnetic scalar potential chi at a single observation point (R_obs, Z_obs).

This is the Julia version of the Fortran `chi` subroutine. The potential is computed
by integrating the Green's function response with the source perturbation at the plasma surface.

# Arguments

  - `R_obs::Float64`: R-coordinate of observation point
  - `Z_obs::Float64`: Z-coordinate of observation point
  - `inputs::VacuumInput`: Struct containing vacuum calculation parameters
  - `plasma_surf::PlasmaGeometry`: Struct with plasma surface geometry
  - `grri::Matrix{Float64}`: Inverted Green's function response matrix
  - `Bn_real::Vector{Float64}`: Real part of normal field Fourier harmonics
  - `Bn_imag::Vector{Float64}`: Imaginary part of normal field Fourier harmonics

# Returns

  - `chi_real::Float64`: Real part of the magnetic scalar potential at (R_obs, Z_obs)
  - `chi_imag::Float64`: Imaginary part of the magnetic scalar potential at (R_obs, Z_obs)

# Notes

  - The potential is computed via Fourier series over poloidal modes
  - Includes coupling term from Green's function derivative
  - Factor of -0.5 * dtheta applied from Fortran convention
"""
function _calculate_potential_chi(R_obs::Float64, Z_obs::Float64,
    inputs::VacuumInput, plasma_surf::PlasmaGeometry,
    grri::Matrix{Float64},
    Bn_real::Vector{Float64}, Bn_imag::Vector{Float64})

    chi_real = 0.0
    chi_imag = 0.0

    mtheta = inputs.mtheta
    mpert = inputs.mpert
    n = inputs.n
    qa = inputs.qa
    dtheta = 2pi / mtheta

    # Pre-calculate Green's function for the observation point
    g_real = zeros(mtheta, mpert)
    g_imag = zeros(mtheta, mpert)

    l_modes = (inputs.mlow:inputs.mhigh)

    for i_theta in 1:mtheta
        R_src = plasma_surf.x[i_theta]
        Z_src = plasma_surf.z[i_theta]

        # Call the low-level Green's function calculator.
        # The `green` function returns the Green's function value itself (G_n) and
        # the coupling terms for mode n and mode 0.
        G_n, coupling_n, coupling_0 = green(R_obs, Z_obs, R_src, Z_src, plasma_surf.dx_dtheta[i_theta], plasma_surf.dz_dtheta[i_theta], n)

        # The term `aval` in the original Fortran CHI routine corresponds to the coupling term 𝒥 ∇'𝒢ⁿ∇'ℒ,
        # which is directly returned as `coupling_n` by the Julia `green` function.
        aval = coupling_n

        # Accumulate Fourier series for g_real and g_imag at this source point
        for l_idx in 1:mpert
            l = l_modes[l_idx]
            arg = l * (i_theta-1) * dtheta + n * qa * plasma_surf.delta[i_theta]
            cos_val = cos(arg)
            sin_val = sin(arg)

            g_real[i_theta, l_idx] = aval * cos_val
            g_imag[i_theta, l_idx] = aval * sin_val
        end
    end

    # Now, combine with the inverted response `grri` and the source `Bn`
    # This corresponds to the main loop in the Fortran `chi` subroutine
    for l1 in 1:mpert
        term_r = 0.0
        term_i = 0.0
        for i_theta in 1:mtheta
            # grri has structure [ (grri_cc, grri_cs), (grri_sc, grri_ss) ]
            # The indices for chiwc, chiws in Fortran map to columns of grri
            chi_wc = grri[i_theta, l1]          # Real part kernel
            chi_ws = grri[i_theta, mpert+l1]  # Imaginary part kernel

            term_r += g_real[i_theta, l1] * chi_wc - g_imag[i_theta, l1] * chi_ws
            term_i += g_imag[i_theta, l1] * chi_wc + g_real[i_theta, l1] * chi_ws
        end

        chi_real += term_r * Bn_real[l1] - term_i * Bn_imag[l1]
        chi_imag += term_i * Bn_real[l1] + term_r * Bn_imag[l1]
    end

    # The factor of 0.5 * isg * dth in Fortran
    # isg is -1 for plasma surface
    factor = -0.5 * dtheta
    return chi_real * factor, chi_imag * factor
end
