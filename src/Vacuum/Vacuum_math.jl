#############################################################
# Julia Numerical Utilities: Function Summary & Fortran Mapping
#
# Interpolation & Smoothing:
#   - spline1d(x, y, xq)           : Cubic spline interpolation (spl1d1)
#   - spline1d_deriv(x, y, xq)     : Cubic spline derivative (spl1d2)
#   - lagrange1d(x, y, xq)         : Lagrange interpolation (lagp, lagpe4, lag)
#   - smooth(y, w)                 : Moving average smoothing (smooth, smooth0)
#   - shift(y, nshift)             : Periodic array shift (shft)
#
# Green's Function & Legendre Kernel:
#   - green(xs, zs, xt, zt, n)     : Green's function (green)
#   - Pn_minus_half_1997(s, n)          : Legendre function of the first kind, order -1/2 (aleg_old)
#   - Pn_minus_half_2007(s, n)          : Updated Legendre function for certain regimes (aleg, not yet implemented)
#
# Matrix Operations:
#   - A * B                        : Matrix multiplication (mult, matmul1, matmul3)
#   - eigen(A).values              : Eigenvalues (eigen)
#   - eigen(A).vectors             : Eigenvectors (eigen)
#
# Integration & Differentiation:
#   - cumtrapz(y, dx)              : Cumulative trapezoidal integration (indef4)
#   - atan(y, x)                   : Two-argument arctangent (atan2m)
#
# Utility Functions:
#   - search(xbar, x)              : Interval search (search, searchx)
#   - interp_to_new_grid(vecin, mtheta; dx0, dx1)  : Periodic cubic spline resampling (trans, transdx, transdxx)
#
# All functions use Julia built-in or standard package features for clarity and efficiency.
#############################################################

# export spline1d, spline1d_deriv, lagrange1d, search, green
export lagrange1d, green

#############################################################
# Cubic spline and derivatives for line 1d array and return point value, 
# replacing spl1d1, spl1d2
#############################################################
# function spline1d(x, y, xq)
#     itp = CubicSplineInterpolation(x, y)
#     return itp(xq)
# end

# function spline1d_deriv(x, y, xq)
#     itp = CubicSplineInterpolation(x, y)
#     return Interpolations.gradient(itp, xq)
# end

# # Returns the array of derivatives at all x points, I think this acts like difspl
# # in the Fortran but need to check/consolidate spline routines later
# function periodic_cubic_deriv(theta, vals)
#     itp = scale(interpolate(vals, BSpline(Cubic(Periodic(OnGrid())))), theta)
#     return first.(Interpolations.gradient.(Ref(itp), theta))
# end

#############################################################
# lagrange spline for line 1d array, return point value and its derivative
# replacing lagp, lagpe4, lag
#############################################################
"""
    lagrange1d(ax, af, m, nl, x, iop)

This function performs Lagrange interpolation and optionally computes its derivative.

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

# #############################################################
# # smoothing array, replacing smooth0, smooth
# #############################################################
# function smooth(y::Vector{Float64}, w::Int)
#     n = length(y)
#     y_smooth = similar(y)
#     for i in 1:n
#         i1 = max(1, i - div(w,2))
#         i2 = min(n, i + div(w,2))
#         y_smooth[i] = mean(y[i1:i2])
#     end
#     return y_smooth
# end

# #############################################################
# # shift array, replacing shift
# #############################################################
# function shift(y::Vector, nshift::Int)
#     n = length(y)
#     nshift = mod(nshift, n)
#     return vcat(y[end-nshift+1:end], y[1:end-nshift])
# end

# #############################################################
# # cumtrapz integration for same intervals
# # replacing indef4
# #############################################################
# function cumtrapz(y::Vector{Float64}, dx::Float64)  
#     fin = zero(y)  
#     for i in eachindex(y)[2:end]  
#         fin[i] = fin[i-1] + (y[i-1] + y[i]) * dx / 2  
#     end  
#     return fin  
# end

# #############################################################
# # cubic spline for periodic 1d datas and return array
# # replacing transdx, transdxx, trans
# #############################################################
# """
#     interp_to_new_grid(vecin, mtheta; dx0=0.0, dx1=0.0)

# Resample the input array `vecin` using a periodic cubic spline to an output array of length `mtheta`.
# This is a Fortran conversion of the functions `trans`, `transdx` and `transdxx`, which have now
# been unified into a single function with optional parameters for offsets.

# # Parameters
# - `vecin::Vector{Float64}` : Input array to be resampled.
# - `mtheta::Int`            : Desired length of the output array.
# - `dx0::Float64`           : Global offset added to all x-coordinates (default 0, applied as `x += dx0 / mthin`).
# - `dx1::Float64`           : Fine offset added to each index (default 0, applied as `ai = (i-1) + dx1`).

# # Returns
# - `vecout::Vector{Float64}` : The resampled output array with first and second points repeated (length `mtheta + 2`).
# """
# function interp_to_new_grid(vecin::Vector{Float64}, mtheta::Int; dx0=0.0, dx1=0.0)

#     # Initialize
#     mtheta_in = length(vecin)

#     # If mthin == mth, just return the input vector
#     if mtheta == mtheta_in
#         return vecin
#     end

#     # Input grids are from [0, 1] inclusive, since no interpolants will fall outside of this, we don't need periodic extrapolation
#     θin = range(0.0, 1.0; length=mtheta_in)
#     itp = cubic_spline_interpolation(θin, vecin)

#     # Interpolate to new grid with optional offsets
#     vecout = zeros(mtheta)
#     for i in 1:mtheta
#         x = (i - 1 + dx1) / mtheta + dx0 / mtheta_in
#         x = x % 1.0  # This is for periodicity in the case of dx1/dx0 ≠ 0
#         vecout[i] = itp(x)
#     end
#     return vecout
# end

# #############################################################
# # Searching index , replacing search, serachx
# #############################################################
# """
#     search(xbar, x::AbstractVector{<:Real})

# Returns the 1-based index of the interval in the sorted array `x` to which `xbar` belongs.
# - Returns 0 if xbar < x[1].
# - Returns length(x)-1 if xbar ≥ x[end].
# - Otherwise, returns i such that x[i] ≤ xbar < x[i+1].
# """
# function search(xbar, x::AbstractVector{<:Real})
#     n = length(x)
#     idx = searchsortedfirst(x, xbar)
#     if xbar < x[1]
#         return 0
#     elseif xbar >= x[end]
#         return n-1
#     else
#         return idx - 1
#     end
# end

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

#############################################################
# Green function eq.(36)~(42). replacing green (verified)
#############################################################

function Pn_minus_half_2007(s::Real, n::Int)
    @error "2007 paper implementation of Pn_minus_half is not yet complete. Use old version."
    # This is a temporary alias. The new implementation should be added here.
    return Pn_minus_half_1997(s, n)
end

"""
    green(x_obs, z_obs, x_source, z_source, xtp, ztp, n; usechancebugs=false)

Compute the Green's function and related quantities for axisymmetric geometry.

# Arguments
- `x_obs`, `z_obs`: Observation point coordinates (X,Z) (Float64)
- `x_source`, `z_source`: Source point coordinates (X',Z')(Float64)
- `xtp`, `ztp`: Derivatives ∂X'/∂θ, ∂Z'/∂θ (Float64)
- `n`: Mode number (Int)
- `usechancebugs::Bool`: Flag to use the 'old' buggy version for comparison.

# Returns
- `G_n`:   2π𝒢ⁿ(θ,θ′) — Green's function value
- `coupling_n`:   𝒥 ∇'𝒢ⁿ∇'ℒ — Coupling term for mode n
- `coupling_0`:  1/(2π) 𝒥 ∇'𝒢⁰∇'ℒ — Coupling term for mode 0
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
