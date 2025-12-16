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
#   - Pn_minus_half(s, n)          : Legendre function of the first kind, order -1/2 (aleg)
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

export spline1d, spline1d_deriv, lagrange1d, search, green

#############################################################
# Cubic spline and derivatives for line 1d array and return point value, 
# replacing spl1d1, spl1d2
#############################################################
function spline1d(x, y, xq)
    itp = CubicSplineInterpolation(x, y)
    return itp(xq)
end

function spline1d_deriv(x, y, xq)
    itp = CubicSplineInterpolation(x, y)
    return Interpolations.gradient(itp, xq)
end

# Returns the array of derivatives at all x points, I think this acts like difspl
# in the Fortran but need to check/consolidate spline routines later
function periodic_cubic_deriv(theta, vals)
    itp = scale(interpolate(vals, BSpline(Cubic(Periodic(OnGrid())))), theta)
    return first.(Interpolations.gradient.(Ref(itp), theta))
end

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
-  `iop::Float64`: Flag (0 = value only, 1 = value and derivative)

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

#############################################################
# smoothing array, replacing smooth0, smooth
#############################################################
function smooth(y::Vector{Float64}, w::Int)
    n = length(y)
    y_smooth = similar(y)
    for i in 1:n
        i1 = max(1, i - div(w,2))
        i2 = min(n, i + div(w,2))
        y_smooth[i] = mean(y[i1:i2])
    end
    return y_smooth
end

#############################################################
# shift array, replacing shift
#############################################################
function shift(y::Vector, nshift::Int)
    n = length(y)
    nshift = mod(nshift, n)
    return vcat(y[end-nshift+1:end], y[1:end-nshift])
end

#############################################################
# cumtrapz integration for same intervals
# replacing indef4
#############################################################
function cumtrapz(y::Vector{Float64}, dx::Float64)
    n = length(y)
    fin = zeros(n)
    for i in 2:n
        fin[i] = fin[i-1] + (y[i-1] + y[i]) * dx / 2
    end
    return fin
end

#############################################################
# cubic spline for periodic 1d datas and return array
# replacing transdx, transdxx, trans
#############################################################
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
# Searching index , replacing search, serachx
#############################################################
"""
    search(xbar, x::AbstractVector{<:Real})

Returns the 1-based index of the interval in the sorted array `x` to which `xbar` belongs.
- Returns 0 if xbar < x[1].
- Returns length(x)-1 if xbar ≥ x[end].
- Otherwise, returns i such that x[i] ≤ xbar < x[i+1].
"""
function search(xbar, x::AbstractVector{<:Real})
    n = length(x)
    idx = searchsortedfirst(x, xbar)
    if xbar < x[1]
        return 0
    elseif xbar >= x[end]
        return n-1
    else
        return idx - 1
    end
end

#############################################################
# Legendre function of the first kind eq.(47)~(50) , replacing aleg. (verified)
#############################################################


"""
    This function is different from elliptic integral K(k). Be careful.

Returns : K(1-m1)
"""
function ellipk_old(m1)

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
function ellipe_old(m1)

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


# Chance eq.(49) (original)
function P0_minus_half_old(s)
    m1 = 2 / (s + 1)
    return 2 / π * sqrt(m1) * ellipk_old(m1)
end

# Chance eq.(50) (original)
function P0_plus_half_old(s)
    m1 = (s + sqrt(s^2 - 1))^(-2)
    return 2 / π * m1^(-1/4) * ellipe_old(m1)
end


# Chance eq.(48) (original)
function P1_minus_half_old(s)
    return 0.5 / ((s^2 - 1)^0.5) * (P0_plus_half_old(s) - s * P0_minus_half_old(s))
end


"""
    Pn_minus_half_old(s, n)

Compute the Legendre function of the first kind of order -1/2, Pⁿ_{-1/2}(s), recursively using Chance's equations (47)-(50).

# Arguments
- `s::Real` : Legendre function parameter (s > 1)
- `n::Int`  : Maximum order n (n ≥ 0)

# Returns
- `P[end]` :  Value of P_{-1/2}^{0~n+1}(s)
"""
function Pn_minus_half_old(s::Real, n::Int)

    #initialize
    P = zeros(n + 2)

    # n = 0
    P[1] = P0_minus_half_old(s)
    P[2] = P1_minus_half_old(s)
    if n == 0
        return P
    end

    # n ≥ 1
    for i in 1:n
        # Chance eq.(47)
        P[i+2] = -2 * i * s / sqrt(s^2 - 1) * P[i+1] - (i - 0.5)^2 * P[i]
    end

    return P
end

#############################################################
# Green function eq.(36)~(42). replacing green (verified)
#############################################################

function Pn_minus_half(s::Real, n::Int)
    # This is a temporary alias. The new implementation should be added here.
    return Pn_minus_half_old(s, n)
end

"""
    green(xs, zs, xt, zt, xtp, ztp, n; usechancebugs=false)

Compute the Green's function and related quantities for axisymmetric geometry.

# Arguments
- `xs`, `zs`: Observation point coordinates (X,Z) (Float64)
- `xt`, `zt`: Source point coordinates (X',Z')(Float64)
- `xtp`, `ztp`: Derivatives ∂X'/∂θ, ∂Z'/∂θ (Float64)
- `n`: Mode number (Int)
- `usechancebugs::Bool`: Flag to use the 'old' buggy version for comparison.

# Returns
- `aval`:   𝒥 ∇'𝒢ⁿ∇'ℒ — Coupling term for mode n
- `aval0`:  1/(2π) 𝒥 ∇'𝒢⁰∇'ℒ — Coupling term for mode 0
- `bval`:   2π𝒢ⁿ(θ,θ′) — Green's function value
"""
function green(xs, zs, xt, zt, xtp, ztp, n; usechancebugs::Bool=false)

    xs2 = xs^2
    xt2 = xt^2
    x_minus2 = (xs - xt)^2
    x_multiple = xs * xt
    ζ = (zs - zt)
    ζ2 = ζ^2

    ρ2 = x_minus2 + ζ2

    # Chance eq.(41) ℛ = R
    R4 = ρ2 * (ρ2 + 4 * x_multiple)
    R2 = sqrt(R4) 
    R = sqrt(R2)
    R5 = R4 * R

    # Chance eq.(42) 𝘴 = s
    s = (xs2 + xt2 + ζ2) / R2
    
    # Legendre functions for 
    # P⁰ = p0, P¹ = p1, Pⁿ = pn, Pⁿ⁺¹ = pp 
    if usechancebugs
        legendre = Pn_minus_half_old(s, n)
    else
        legendre = Pn_minus_half(s, n)
    end

    p0 = legendre[1]
    p1 = legendre[2]
    pp = legendre[end]
    pn = legendre[end-1]

    # Chance eq.(40) 2π𝒢ⁿ = G
    gg = 2 * sqrt(π) * gamma(0.5 - n) / R
    G = gg * pn

    # Chance eq.(44)
    grad_gg = gg / R4 /2π

    # ∂Gⁿ/∂X' = dG_dX
    aval1 = (n * (xs2 + xt2 + ζ2)*(xs2 - xt2 + ζ2) - xt2*(xt2-xs2+ζ2)) * pn
    aval2 = (2.0 * xt * xs * (xs2-xt2+ζ2)) * pp 
    dG_dX = grad_gg * (aval1 + aval2) / xt
    
    # ∂Gⁿ/∂Z' = dG_dZ
    aval3 = (2.0 * n + 1.0) * (xs2 + xt2 + ζ2) * pn
    aval4 = 4.0 * x_multiple * pp
    dG_dZ = grad_gg * (aval3 + aval4) * ζ
    
    # Chance eq.(51) 
    # 𝒥 ∇'𝒢ⁿ∇'ℒ = aval
    # ∂X'/∂θ = xtp, ∂Z'/∂θ = ztp
    aval = -xt * (ztp * dG_dX - xtp * dG_dZ)

    # bval. 2π𝒢ⁿ = bval
    bval = G

    # for 𝓃⩵0,  aval0 = 1/(2π) 𝒥 ∇'𝒢⁰∇'ℒ 
    dG_dX0 = 1/(R5*xt) * ((2.0 * xt * xs * (xs2-xt2+ζ2)) * p1 - xt2*(xt2-xs2+ζ2) * p0 ) 
    dG_dZ0 = 1/(R5)* ζ * ((xs2 + xt2 + ζ2) * p0 + 4.0 * x_multiple * p1) 
    aval0 = -xt * (ztp * dG_dX0 - xtp * dG_dZ0)

    return G, aval, aval0, bval
end

#############################################################
# Inverse Fourier transform
#############################################################
"""
    gll = foranv(gil, cs, dth, twopi; m00=0, l00=0, jmax1=nothing, mth=nothing)

Inverse Fourier transform from theta grid to Fourier (l) space.
- gil: θ-grid matrix (size ≥ m00+mth, l00+jmax1)
- cs:  transformation matrix (nths, jmax1)
- dth: grid spacing (Float64)
- m00, l00: starting indices (Fortran offset, usually 0)
- jmax1: number of Fourier modes
- mth: number of theta points

Returns gll: (jmax1, jmax1) matrix.
"""
function foranv(gil, cs, dth ; jmax1=size(cs,2)) 
    gll = zeros(eltype(gil), jmax1, jmax1)
    mth=size(cs,1)
    for l1 in 1:jmax1
        for l2 in 1:jmax1
            acc = zero(eltype(gil))
            for i in 1:mth
                acc += dth * cs[i, l2] * gil[m00 + i, l00 + l1] * 2π
            end
            gll[l2, l1] = acc
        end
    end
    return gll
end

#############################################################
# Utilities
#############################################################

const architecture = Sys.ARCH;
const cpu_name = Sys.CPU_NAME;
const threads = Sys.CPU_THREADS;
const username = Sys.username();


#############################################################
# Additional spline functions for vacuum structure
#############################################################
function spl1d1!(n::Int, x::Vector{Float64}, f::Vector{Float64}, w::Vector{Float64}, 
                 iop::Vector{Int}, ij::Int, a::Vector{Float64}, b::Vector{Float64}, c::Vector{Float64})
    
    zz, oz, tz, sz = 0.0, 1.0, 3.0, 6.0
    
    # Check minimum size requirement
    if n < 4
        println("spl1d1: Results incorrect because n<4.")
        return
    end
    
    k = n - 1
    a[2] = -(x[2] - x[1]) / sz
    b[2] = (x[3] - x[1]) / tz
    w[ij+1] = (f[2*ij+1] - f[ij+1]) / (x[3] - x[2]) - (f[ij+1] - f[1]) / (x[2] - x[1])
    
    # Main loop for tridiagonal system setup
    if n > 3
        for i = 3:k
            m = (i-1) * ij + 1
            j1 = m + ij
            j2 = m - ij
            con = (x[i+1] - x[i-1]) / tz
            don = (x[i] - x[i-1]) / sz
            b[i] = con - (don^2) / b[i-1]
            e = (f[j1] - f[m]) / (x[i+1] - x[i]) - (f[m] - f[j2]) / (x[i] - x[i-1])
            w[m] = e - (don * w[j2]) / b[i-1]
            a[i] = -(don * a[i-1]) / b[i-1]
        end
    end
    
    k1 = (n-2) * ij + 1
    c[n-1] = -((x[n] - x[n-1]) / sz) / b[n-1]
    w[k1] = w[k1] / b[n-1]
    a[n-1] = a[n-1] / b[n-1]
    k2 = k - 1
    
    # Backward substitution
    if n > 3
        for i = 2:k2
            j = n - i
            con = (x[j+1] - x[j]) / sz
            a[j] = (a[j] - con * a[j+1]) / b[j]
            c[j] = -(con * c[j+1]) / b[j]
            k3 = (j-1) * ij + 1
            m = k3 + ij
            w[k3] = (w[k3] - con * w[m]) / b[j]
        end
    end
    
    k4 = (n-1) * ij + 1
    
    # Boundary condition handling
    if iop[1] != 5
        c1 = w[1]
        if iop[2] != 5
            c2 = w[k4]
        else
            # Right boundary condition processing
            if n >= 4
                b1 = x[n] - x[n-3]
                b2 = x[n] - x[n-2]  
                b3 = x[n] - x[n-1]
                b4 = x[n-1] - x[n-3]
                b5 = x[n-1] - x[n-2]
                b6 = x[n-2] - x[n-3]
                l1 = k4 - ij
                l2 = l1 - ij
                l3 = l2 - ij
                w[k4] = -b2*b3*f[l3]/(b6*b4*b1) + b1*b3*f[l2]/(b6*b5*b2) - 
                        b1*b2*f[l1]/(b4*b5*b3) + f[k4]*(oz/b1+oz/b2+oz/b3)
            end
            c2 = w[k4]
        end
    else
        # Left boundary condition processing
        if n >= 4
            a1 = x[1] - x[2]
            a2 = x[1] - x[3]
            a3 = x[1] - x[4]
            a4 = x[2] - x[3]
            a5 = x[2] - x[4]
            a6 = x[3] - x[4]
            w[1] = f[1]*(oz/a1+oz/a2+oz/a3) - a2*a3*f[ij+1]/(a1*a4*a5) + 
                   a1*a3*f[2*ij+1]/(a2*a4*a6) - a1*a2*f[3*ij+1]/(a3*a5*a6)
        end
        c1 = w[1]
        
        if iop[2] != 5
            c2 = w[k4]
        else
            if n >= 4
                b1 = x[n] - x[n-3]
                b2 = x[n] - x[n-2]
                b3 = x[n] - x[n-1]
                b4 = x[n-1] - x[n-3]
                b5 = x[n-1] - x[n-2]
                b6 = x[n-2] - x[n-3]
                l1 = k4 - ij
                l2 = l1 - ij
                l3 = l2 - ij
                w[k4] = -b2*b3*f[l3]/(b6*b4*b1) + b1*b3*f[l2]/(b6*b5*b2) - 
                        b1*b2*f[l1]/(b4*b5*b3) + f[k4]*(oz/b1+oz/b2+oz/b3)
            end
            c2 = w[k4]
        end
    end
    
    # Process boundary conditions
    for i = 1:k
        m = (i-1) * ij + 1
        
        # Left boundary conditions
        bob = zz
        if iop[1] == 1
            if i == 1
                a[1] = -oz
                c[1] = zz
            end
        elseif iop[1] == 2
            if i == 1
                a[1] = -oz
                c[1] = zz
                w[1] = zz
            elseif i == 2
                bob = -c1
            end
        elseif iop[1] == 3 || iop[1] == 5
            if i == 1
                a[1] = -(x[2] - x[1]) / tz
                c[1] = zz
                w[1] = -c1 + (f[ij+1] - f[1]) / (x[2] - x[1])
            elseif i == 2
                bob = (x[2] - x[1]) / sz
            end
        elseif iop[1] == 4
            if i == 1
                a[1] = -oz
                c[1] = oz
                w[1] = zz
            end
        end
        
        # Right boundary conditions
        bill = zz
        if iop[2] == 1
            if i == 1
                a[n] = zz
                c[n] = -oz
            end
        elseif iop[2] == 2
            if i == 1
                a[n] = zz
                c[n] = -oz
                w[k4] = zz
            elseif i == k
                bill = -c2
            end
        elseif iop[2] == 3 || iop[2] == 5
            if i == 1
                a[n] = zz
                c[n] = (x[n-1] - x[n]) / tz
                w[k4] = c2 - (f[k4] - f[k4-ij]) / (x[n] - x[n-1])
            elseif i == k
                bill = (x[n] - x[n-1]) / sz
            end
        elseif iop[2] == 4
            if i == 1
                a[n] = zz
                c[n] = (x[n-1] + x[1] - x[n] - x[2]) / tz
                w[k4] = (f[ij+1] - f[1]) / (x[2] - x[1]) - (f[k4] - f[k4-ij]) / (x[n] - x[n-1])
            elseif i == 2
                bill = (x[2] - x[1]) / sz
            elseif i == k
                bill = (x[n] - x[n-1]) / sz
            end
        end
        
        # Apply boundary modifications
        if i > 1
            w[1] = w[1] - bob * w[m]
            w[k4] = w[k4] - bill * w[m]
            a[1] = a[1] - bob * a[i]
            a[n] = a[n] - bill * a[i]
            c[1] = c[1] - bob * c[i]
            c[n] = c[n] - bill * c[i]
        end
    end
    
    # Solve final system
    con = a[1] * c[n] - c[1] * a[n]
    d1 = -w[1]
    d2 = -w[k4]
    w[1] = (d1 * c[n] - c[1] * d2) / con
    w[k4] = (a[1] * d2 - d1 * a[n]) / con
    
    for i = 2:k
        m = (i-1) * ij + 1
        w[m] = w[m] + a[i] * w[1] + c[i] * w[k4]
    end
    
    return
end


function spl1d2!(n::Int, x::Vector{Float64}, f::Vector{Float64}, w::Vector{Float64}, 
                 ij::Int, y::Float64, tab::Vector{Float64})
    
    wz, sz = 2.0, 6.0
    mflag = 0
    
    # Find interval
    if y <= x[1]
        i = 1
    elseif y >= x[n]
        i = n - 1
    else
        i = search(y, x, n, 1, mflag)
    end
    
    mi = (i-1) * ij + 1
    k1 = mi + ij
    flk = x[i+1] - x[i]
    
    # Calculate spline value and derivatives
    a = (w[mi] * (x[i+1] - y)^3 + w[k1] * (y - x[i])^3) / (sz * flk)
    b = (f[k1]/flk - w[k1]*flk/sz) * (y - x[i])
    c = (f[mi]/flk - flk*w[mi]/sz) * (x[i+1] - y)
    tab[1] = a + b + c
    
    a = (w[k1] * (y - x[i])^2 - w[mi] * (x[i+1] - y)^2) / (wz * flk)
    b = (f[k1] - f[mi]) / flk
    c = flk * (w[mi] - w[k1]) / sz
    tab[2] = a + b + c
    tab[3] = (w[mi] * (x[i+1] - y) + w[k1] * (y - x[i])) / flk
    
    return
end