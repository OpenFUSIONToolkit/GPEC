# Gaussian quadrature weights and points for 8-point integration
const GAUSSIANWEIGHTS = [0.101228536290376, 0.222381034453374, 0.313706645877887, 0.362683783378362,
               0.362683783378362, 0.313706645877887, 0.222381034453374, 0.101228536290376]

const GAUSSIANPOINTS = [-0.960289856497536, -0.796666477413627, -0.525532409916329, -0.183434642495650,
                0.183434642495650,  0.525532409916329,  0.796666477413627,  0.960289856497536]

"""
    kernel(x_obspoints, z_obspoints, x_sourcepoints, z_sourcepoints, j1, j2, isgn, iopw, iops, ischk, params)

Compute kernels of integral equation for Laplace's equation for a torus.

    ** WARNING: This kernel only supports closed toroidal walls currently. 
        The residue calculation needs to be updated for open walls. **


# Arguments
- `grad_greenfunction_mat`: Gradient Green's function matrix, updated in-place
- `greenfunction_mat`: Green's function matrix, updated in-place
- `x_obspoints`: Observer x coordinates
- `z_obspoints`: Observer z coordinates
- `x_sourcepoints`: Source x coordinates
- `z_sourcepoints`: Source z coordinates
- `j1, j2`: Row, column block indices in the full matrix (1,1 = plasma/plasma, 1,2 = plasma/wall, etc)
- `iopw`: Wall option (0=inactive, 1=active)
- `n`: Toroidal mode number
"""
function kernel!(grad_greenfunction_mat::Matrix{Float64}, greenfunction_mat::Matrix{Float64}, x_obspoints::Vector{Float64}, z_obspoints::Vector{Float64}, x_sourcepoints::Vector{Float64}, z_sourcepoints::Vector{Float64}, j1::Int, j2::Int, iopw::Int, n::Int)

    # These used to be function arguments, but can just set inside here based on j1/j2
    isgn = 2 * j2 - 3
    iops = (j1 == 1 && j2 == 1) ? 1 : 0 # only equal to 1 for plasma/plasma block

    mtheta = length(x_obspoints)
    dtheta = 2π / mtheta
    ak0i = 0.0
    jres = 1
    theta_grid = range(start=0; length=mtheta, step=dtheta)

    if mtheta != length(z_obspoints) || mtheta != length(x_sourcepoints) || mtheta != length(z_sourcepoints)
        error("Length of input arrays (xobs, zobs, xsource, zsce) are different. All length should be the same")
    end

    # definition for solving parameters
    simpson_b1=1*dtheta/3
    simpson_b2=2*dtheta/3
    simpson_b4=4*dtheta/3

    simpson_a1=1*dtheta/3
    simpson_a2=2*dtheta/3
    simpson_a4=4*dtheta/3
    
    slog1m = 3*dtheta*(log(dtheta)-1/3)
    slog1p = slog1m
    
    log2dtheta = log(2*dtheta)
    log_correction_0=16.0*dtheta*(log2dtheta-68.0/15.0)/15.0
    log_correction_1=128.0*dtheta*(log2dtheta-8.0/15.0)/45.0
    log_correction_2=4.0*dtheta*(7.0*log2dtheta-11.0/15.0)/45.0

    has_zero_crossing = false
    # See if the conductor surface needs to be checked for singular points
    if j1 == 2 || j2 == 2
        # Initialize jbot and jtop, and figure out wall geometry based on which block we are in
        jbot = jtop = mtheta/2+1
        xwall = j2 == 2 ? x_sourcepoints : x_obspoints
        
        # Find where sign of wall x point acrosses zero.
        # has_zero_crossing means there is 0-crossing point
        for i in 1:mtheta
            next_i = i == mtheta ? 1 : i + 1
            if xwall[i] * xwall[next_i] <= 0.0
                jbot = xwall[i] > 0.0 ? i : jbot
                jtop = xwall[i] < 0.0 ? i + 1 : jtop
                has_zero_crossing = true
            end
        end
    end

    # do spline and calc derivative for Z'_θ and X'_θ in eq.(51)

    # Using interpolations.jl for now seemes ok
    spline_x = cubic_spline_interpolation(theta_grid, x_sourcepoints, extrapolation_bc=Interpolations.Periodic())
    spline_z = cubic_spline_interpolation(theta_grid, z_sourcepoints, extrapolation_bc=Interpolations.Periodic())

    gradients_x = (t -> Interpolations.gradient(spline_x, t)).(theta_grid)
    gradients_z = (t -> Interpolations.gradient(spline_z, t)).(theta_grid)
    dx_dtheta = first.(gradients_x) # d x / d theta
    dz_dtheta = first.(gradients_z) # d z / d theta

    # observer loop
    for j in 1:mtheta
        # initialize variable
        x_obs=x_obspoints[j] #observation point
        z_obs=z_obspoints[j]
        theta_obs=theta_grid[j] # theta value
        work = zeros(mtheta)
        
        # if the point of observation point is in negative, We cannot use green func
        # This is same for source point
        if x_obs < 0.0
            if j2 == 2
                work[j] = 1.0
            end
            continue
        end
            
        # set istart and iend
        iend = 2 # end point of integration
        aval1=0.0 # ∇𝒢_0
        # if wall crossing zero and wall is source, iend set.
        if has_zero_crossing && j2 == 2
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
        istart = 4 - iend # starting point of integration

        # 4-3 on mth. then mths is equals to mtheta+3 ??? I'm not sure
        mth_source=mtheta-(istart+iend-1)

        # loop for source point
        for i in 1:mth_source

            # 4.5 get source point index(ic) theta(theta), X(xt), and Z(zt)
            ic = i + j + istart - 1
            if ic ≥ mtheta + 1
                ic = ic - mtheta
            end
            # theta_source=(ic-1)*dtheta
            x_source=x_sourcepoints[ic]
            z_source=z_sourcepoints[ic]

            # if source point is in negative, we cannot use green function
            # & if source point(ic) and obs point (j) is same, it's singular
            if x_source < 0
                continue
            end
            if ic == j
                continue
            end

            # Call green function
            # G_n is 2pi𝒢ⁿ; coupling_n is 𝒥 ∇'𝒢ⁿ∇'ℒ; coupling_0 is 𝒥 ∇'𝒢ⁿ∇'ℒ for n=0
            G_n, coupling_n, coupling_0 = green(x_obs,z_obs,x_source,z_source,
                                               dx_dtheta[ic],dz_dtheta[ic],n)

            # simpson integral. 4 for odd, 2 for even, and 1 for others.
            simpson_b=simpson_b2
            if (i÷2)*2 == i
                simpson_b=simpson_b4
            end
            if (i == 1)||(i == mth_source)
                simpson_b=simpson_b1
            end
            simpson_a=simpson_a2
            if (i÷2)*2 == i
                simpson_a=simpson_a4
            end
            if (i == 1)||(i == mth_source)
                simpson_a=simpson_a1
            end

            # work and gren
            # work : simpson integral for coupling_n (𝒥 ∇'𝒢ⁿ∇'ℒ)
            # gren : log singularity values accumulated. simpson integral for G_n
            # aval1 : aval1
            work[ic] += isgn*coupling_n*simpson_a
            greenfunction_mat[j,ic] += G_n*simpson_b # integral of G_n (log singularity)
            aval1 += coupling_0 * simpson_a
        end
        
        # if it's plasma/plasma, wall/wall and in negative wall point, skip loop
        # obs : j1 = 1(plasma), wall(2)
        # src : j1 = 1(plasma), wall(2)
        if (j1+j2) != 2 && has_zero_crossing && j > jbot && j < jtop
            continue
        end

        # Get js
        js1=mod(j-iend+mtheta-1,mtheta)+1
        js2=mod(j-iend+mtheta,mtheta)+1
        js3=mod(j-iend+mtheta+1,mtheta)+1
        js4=mod(j-iend+mtheta+2,mtheta)+1
        js5=mod(j-iend+mtheta+3,mtheta)+1

        # Singular points when source point and obs point are the same
        # integration each left and right by gaussian quadrature
        for ilr in [1,2]
            gauss_xleft = theta_obs + (2*ilr-iend-2)*dtheta
            gauss_xright = gauss_xleft + 2 * dtheta
            gauss_xavg = (gauss_xright + gauss_xleft)/2
            gauss_halfdifference = (gauss_xright - gauss_xleft)/2
            tgaus = gauss_xavg .+ GAUSSIANPOINTS .* gauss_halfdifference # tgaus is 8 point gauss points, since GAUSSIANPOINTS is for only [-1,1]
            # for each of 8 gaussian points
            for ig in 1:8
                tgaus0 = tgaus[ig] #i-th value of for 8 points, in theta
                tgaus0 = mod(tgaus0, 2π)

                # get X, X', Z, Z' for gaussian point
                xt = spline_x(tgaus0)
                xtp = Interpolations.gradient(spline_x, tgaus0)[1]
                zt = spline_z(tgaus0)
                ztp = Interpolations.gradient(spline_z, tgaus0)[1]

                # call green function
                G_n, coupling_n, coupling_0 = green(x_obs,z_obs,xt,zt,xtp,ztp,n,uselegacygreenfunction=true)

                # add logarithm to G_n to analytically isolate the singularity. Chance eq.(75)
                # iops = 1
                G_n_nonsingular = G_n + iops * log((theta_obs-tgaus[ig])^2)/x_obs

                # calc GAUSSIANWEIGHTS. GAUSSIANWEIGHTS contains gaussian quadrature weights for each 8 points
                wgbg=GAUSSIANWEIGHTS[ig]*gauss_halfdifference

                # calc pgaus. below Chance eq.(77)
                # All are the noted index, times (wgbg)/(2 pgaus)
                # A0 has no extra factor
                # A1 has an extra factor of (pgaus +/- 1)
                # A2 has an extra factor of (pgaus +/- 2)
                pgaus=(tgaus[ig]-theta_obs-(2-iend)*dtheta)/dtheta
                pgaus2=pgaus*pgaus
                A0 = (pgaus2-1)*(pgaus2-4)/4.0 * wgbg
                A1_plus  = -(pgaus+1)*pgaus*(pgaus2-4)/6.0 * wgbg
                A1_minus = -(pgaus-1)*pgaus*(pgaus2-4)/6.0 * wgbg
                A2_plus  = (pgaus2-1)*pgaus*(pgaus+2)/24.0 * wgbg
                A2_minus = (pgaus2-1)*pgaus*(pgaus-2)/24.0 * wgbg

                # add up in work
                work[js1] += isgn * coupling_n * A2_minus
                work[js2] += isgn * coupling_n * A1_minus
                work[js3] += isgn * coupling_n * A0
                work[js4] += isgn * coupling_n * A1_plus
                work[js5] += isgn * coupling_n * A2_plus

                # minus diverging value
                work[j] -= isgn * coupling_0 * wgbg
                if j == jres
                    ak0i -= isgn * coupling_0 * wgbg
                end
                
                # skip when plasma, no skip when considering wall
                if iopw == 0
                    continue
                end

                # if Wall is considered(wall/wall, plasma/wall, wall/plasma), add up G_n_nonsingular value
                greenfunction_mat[j,js1] += G_n_nonsingular * A2_minus
                greenfunction_mat[j,js2] += G_n_nonsingular * A1_minus
                greenfunction_mat[j,js3] += G_n_nonsingular * A0
                greenfunction_mat[j,js4] += G_n_nonsingular * A1_plus
                greenfunction_mat[j,js5] += G_n_nonsingular * A2_plus
            end
        end

        # Set residue based on logic similar to Table I of Chance 1997
        residue = (j1 == 2) ? 2.0 : 0.0
        # Would need to pass in wall geometry to generalize this to open walls
        is_closed_toroidal = true
        if is_closed_toroidal
            residue_diagonal=(2-j1)*(2-j2)+(j1-1)*(j2-1)
            residue_k0=(2-j1)*(2-j2)+(j1-3)*(j2-1)
            residue=residue_diagonal+residue_k0
        end

        # minus residue value
        work[j] = work[j] - isgn * aval1 + residue
        if j == jres
            ak0i -= isgn * aval1
        end

        # Only when plasma/plasma, log singularity activate. (S1)
        if iops == 1 && iopw != 0
            greenfunction_mat[j,js1] -= log_correction_2 / x_obs
            greenfunction_mat[j,js2] -= log_correction_1 / x_obs
            greenfunction_mat[j,js3] -= log_correction_0 / x_obs
            greenfunction_mat[j,js4] -= log_correction_1/x_obs
            greenfunction_mat[j,js5] -= log_correction_2/x_obs
        end

        # Store all the datas of work in grdgre, gren
        grad_greenfunction_mat[(j1-1)*mtheta + j, (j2-1)*mtheta .+ (1:mtheta)] .= work
        greenfunction_mat[j, 1:mtheta] ./= 2π
    end
end

"""
    fourier_inverse_transform!(gll, gil, cs, m00, l00)

    Purpose:
      This routine performs the inverse Fourier transform of gil onto gll
      using Fourier coefficients stored in cs. Performs the same function
      as fouranv in the Fortran code.
    
    Inputs:
      gil(i,l)   : input matrix of size (mth × mpert), the Fourier-space data
      cs(j,l)    : Fourier coefficient matrix (mth × mpert)
      m00, l00   : integer offsets in the gil matrix
    
    Output:
      gll(l2,l1) : output matrix updated in-place (mpert × mpert)
"""
function fourier_inverse_transform!(gll::Matrix{Float64}, gil::Matrix{Float64}, cs::Matrix{Float64}, m00::Int, l00::Int)

    # Zero out gll block
    mtheta, mpert = size(cs)
    fill!(view(gll, 1:mpert, 1:mpert), 0.0)

    # Inverse Fourier transform via matrix multiply: gll = cs^T * gil * (2π * dth)
    # This computes: gll[l2, l1] = (2π * dth) * Σ_i cs[i, l2] * gil[i, l1]
    dth = 2π / mtheta
    mul!(gll, cs', view(gil, m00+1:m00+mtheta, l00+1:l00+mpert), 2π * dth, 0.0)
end

"""
    fourier_transform!(gil, gij, cs, m00, l00)

    Purpose:
      This routine performs a truncated Fourier transform of gij onto gil
      using Fourier coefficients stored in cs. Performs the same function
      as fouran in the Fortran code.
    
    Inputs:
      gij(i,j)   : input matrix of size (mth × mth), the "physical-space" data
      cs(j,l)    : Fourier coefficient matrix (mth × mpert)
      m00, l00   : integer offsets in the gil matrix

    Output:
      gil(i', l') : output matrix updated in-place (mth × mpert), where i' = m00 + i and l' = l00 + l
"""
function fourier_transform!(gil::Matrix{Float64}, gij::Matrix{Float64}, cs::Matrix{Float64}, m00::Int, l00::Int)

    # Zero out relevant gil block
    mtheta, mpert = size(cs)
    fill!(view(gil, m00+1:m00+mtheta, l00+1:l00+mpert), 0.0)

    # Fourier transform via matrix multiply: gil[i, l] = Σ_j gij[i, j] * cs[j, l]
    mul!(view(gil, m00+1:m00+mtheta, l00+1:l00+mpert), gij, cs)
end

# Returns the array of derivatives at all x points, I think this acts like difspl
# in the Fortran but need to check/consolidate spline routines later
function periodic_cubic_deriv(theta, vals)
    itp = scale(interpolate(vals, BSpline(Cubic(Periodic(OnGrid())))), theta)
    return first.(Interpolations.gradient.(Ref(itp), theta))
end

"""
    lagrange1d(ax, af, m, nl, x, iop)

This function performs Lagrange interpolation and optionally computes its derivative.
Replaces lagp, lagpe4, lag from Fortran code.

# Arguments
- `ax::AbstractVector{Float64}`: Array of x-coordinates for the interpolation points.
- `af::AbstractVector{Float64}`: Array of y-coordinates (function values) for the interpolation points.
- `m::Int`: Number of interpolation points.
- `nl::Int`: Number of points to use for the local interpolation (degree of polynomial + 1).
- `x::Float64`: The x-value at which to evaluate the interpolated function and/or its derivative.
- `iop::Float64`: Flag (0 = value only, 1 = value and derivative)

# Returns
- `f::Float64`: The interpolated function value at `x`
- `df::Float64`: The interpolated function derivative at `x` 

"""
function lagrange1d(ax::Vector{Float64}, af::Vector{Float64}, m::Int, nl::Int, x::Float64, iop::Int)

    # --- Error fix: Initialize f and df internally to 0.0 ---
    f::Float64 = 0.0
    df::Float64 = 0.0

    jn = findfirst(i -> ax[i] >= x, 1:m)
    jn = (jn === nothing) ? m : jn
    jn = max(jn - 1, 1)
    if jn < m && abs(ax[jn + 1] - x) < abs(x - ax[jn])
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
            if i == j
                continue
            end
            alag *= (x - ax[j]) / (ax[i] - ax[j])
        end
        f += alag * af[i]
    end

    # --- Error fix: Use the 'iop' argument ---
    if iop == 0
        return f, df # df is returned as 0.0
    end

    # Compute derivative df
    for i in nll:nlr
        slag = 0.0
        for id in nll:nlr
            if id == i
                continue
            end
            alag = 1.0
            for j in nll:nlr
                if j == i
                    continue
                end
                if j != id
                    alag *= (x - ax[j]) / (ax[i] - ax[j])
                else
                    alag /= (ax[i] - ax[id])
                end
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
This is a Fortran conversion of the functions `trans`, `transdx` and `transdxx`, which have now
been unified into a single function with optional parameters for offsets.

# Parameters
- `vecin::Vector{Float64}` : Input array to be resampled.
- `mtheta::Int`            : Desired length of the output array.
- `dx0::Float64`           : Global offset added to all x-coordinates (default 0, applied as `x += dx0 / mthin`).
- `dx1::Float64`           : Fine offset added to each index (default 0, applied as `ai = (i-1) + dx1`).

# Returns
- `vecout::Vector{Float64}` : The resampled output array with first and second points repeated (length `mtheta + 2`).
"""
function interp_to_new_grid(vecin::Vector{Float64}, mtheta::Int; dx0=0.0, dx1=0.0)

    # Initialize
    mtheta_in = length(vecin)

    # If mthin == mth, just return the input vector
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

    if m1 < 0.0 || m1 > 1.0
        throw(DomainError(m1, "Input `m1` must be in the range (0, 1]."))
    end
    log_m1 = log(m1)

    ak0 = 1.38629436112
    ak1 = 0.09666344259
    ak2 = 0.03590092383
    ak3 = 0.03742563713
    ak4 = 0.01451196    
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

    if m1 < 0.0 || m1 > 1.0
        throw(DomainError(m1, "Input `x1` must be in the range (0, 1]."))
    end
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

Compute the Legendre function of the first kind of order -1/2, Pⁿ_{-1/2}(s), recursively using Chance's equations (47)-(50).
The implementation follows the original fortran code. The paper equation 50 is incorrect, the exponent should be -1/4 instead of +1/2.

# Arguments
- `s::Real` : Legendre function parameter (s > 1)
- `n::Int`  : Maximum order n (n ≥ 0)

# Returns
- `P[end]` :  Value of P_{-1/2}^{0~n+1}(s)
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

function Pn_minus_half_2007(s::Real, n::Int)
    @error "2007 paper implementation of Pn_minus_half is not yet complete. Use old version."
    # This is a temporary alias. The new implementation should be added here.
    return Pn_minus_half_1997(s, n)
end

"""
    green(x_obs, z_obs, x_source, z_source, xtp, ztp, n; usechancebugs=false)

Compute the Green's function and related quantities for axisymmetric geometry
according to eq.s (36)-(42) of Chance 1997. Replaces green from Fortran code.

# Arguments
- `x_obs`, `z_obs`: Observation point coordinates (X,Z) (Float64)
- `x_source`, `z_source`: Source point coordinates (X',Z')(Float64)
- `xtp`, `ztp`: Derivatives ∂X'/∂θ, ∂Z'/∂θ (Float64)
- `n`: Mode number (Int)
- `uselegacygreenfunction::Bool`: Flag to use the the 1997 version of the Legendre function (default true)

# Returns
- `G_n`:   2π𝒢ⁿ(θ,θ′) — Green's function value
- `coupling_n`:   𝒥 ∇'𝒢ⁿ∇'ℒ — Coupling term for mode n
- `coupling_0`:  1/(2π) 𝒥 ∇'𝒢⁰∇'ℒ — Coupling term for mode 0
"""
function green(x_obs::Float64, z_obs::Float64, x_source::Float64, z_source::Float64, dx_dtheta::Float64, dz_dtheta::Float64, n::Int; uselegacygreenfunction::Bool=true)

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
    grad_gg = gg / R4 /2π

    # ∂Gⁿ/∂X' = dG_dX
    xterm1 = (n * (x_obs2 + x_source2 + ζ2)*(x_obs2 - x_source2 + ζ2) - x_source2*(x_source2-x_obs2+ζ2)) * pn
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
    dG_dX0 = 1/(R5*x_source) * ((2.0 * x_source * x_obs * (x_obs2-x_source2+ζ2)) * p1 - x_source2*(x_source2-x_obs2+ζ2) * p0 ) 
    dG_dZ0 = 1/(R5)* ζ * ((x_obs2 + x_source2 + ζ2) * p0 + 4.0 * x_multiple * p1) 
    coupling_0 = -x_source * (dz_dtheta * dG_dX0 - dx_dtheta * dG_dZ0)

    return G_n, coupling_n, coupling_0
end

# Helper functions for mscfld

"""
    _pickup_field(inputs, plasma_surf, grri, Bn_real, Bn_imag, R_grid, Z_grid)

Internal function to calculate the magnetic field on a specified grid.
This is a Julia version of the Fortran `pickup` routine. It uses finite differencing
of the magnetic scalar potential `chi` to find the field components.
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
            bphi_i =  inputs.n * chi_r[5, idx] / R_points[idx]
            
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

Creates a flattened 1D list of (R, Z) coordinates from 2D grid vectors.
This is a Julia version of the Fortran `loops` subroutine.
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

Calculates the magnetic scalar potential chi at a single observation point (R_obs, Z_obs).
This is a Julia version of the Fortran `chi` subroutine.
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
            chi_ws = grri[i_theta, mpert + l1]  # Imaginary part kernel
            
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