"""
SplineAdapter - Pure Julia spline implementations using FastInterpolations.jl

This module provides:
- CubicSpline1D: 1D cubic spline with up to 3rd derivative support
- deriv3(): Custom 3rd derivative implementation
- integrate_spline(): Cumulative integration of cubic splines

These replace the Fortran-backed splines from libspline.
"""

using FastInterpolations
using LinearAlgebra

# Re-export FastInterpolations boundary conditions
const NaturalBC = FastInterpolations.NaturalBC
const PeriodicBC = FastInterpolations.PeriodicBC

# Map old boundary condition names to FastInterpolations types
const BC_MAP = Dict(
    "natural" => NaturalBC(),
    "periodic" => PeriodicBC(),
    "extrap" => NaturalBC(),  # FastInterpolations uses extrap mode separately
    "not-a-knot" => NaturalBC(),  # Approximate with natural for now
    1 => NaturalBC(),
    2 => PeriodicBC(),
    3 => NaturalBC(),
    4 => NaturalBC()
)

# Map old boundary condition names to FastInterpolations extrapolation modes
const EXTRAP_MAP = Dict(
    "natural" => :constant,
    "periodic" => :wrap,
    "extrap" => :extension,
    "not-a-knot" => :extension,
    1 => :constant,
    2 => :wrap,
    3 => :extension,
    4 => :extension
)

"""
    CubicSpline1D{T}

A 1D cubic spline interpolator wrapping FastInterpolations.jl with additional features:

  - Support for multiple quantities (columns) in a single spline
  - 3rd derivative computation
  - Cumulative integration

# Fields

  - `xs::Vector{Float64}`: Grid points (x-coordinates)
  - `fs::Matrix{T}`: Function values at grid points (npts × nqty)
  - `fs1::Matrix{T}`: First derivatives at grid points (computed on construction)
  - `fsi::Matrix{T}`: Cumulative integrals at grid points (computed on demand)
  - `bc`: Boundary condition type
  - `extrap`: Extrapolation mode
  - `coeffs::Array{T,3}`: Spline coefficients (npts-1 × 4 × nqty) for [a, b, c, d]
"""
mutable struct CubicSpline1D{T<:Union{Float64,ComplexF64}}
    xs::Vector{Float64}
    fs::Matrix{T}
    fs1::Matrix{T}
    fsi::Matrix{T}
    bc::Any
    extrap::Symbol
    coeffs::Array{T,3}  # (nintervals, 4, nqty) for polynomial coeffs [a, b, c, d]
    _f::Vector{T}       # Work array for evaluation
    _f1::Vector{T}
    _f2::Vector{T}
    _f3::Vector{T}
end

"""
    CubicSpline1D(xs, fs; bctype="extrap")

Create a 1D cubic spline interpolator.

# Arguments

  - `xs::Vector{Float64}`: Grid points (must be sorted)
  - `fs::Union{Vector{T}, Matrix{T}}`: Function values. If Vector, treated as single quantity.
  - `bctype`: Boundary condition type ("natural", "periodic", "extrap", "not-a-knot")
"""
function CubicSpline1D(xs::Vector{Float64}, fs::Union{Vector{T},Matrix{T}};
    bctype::Union{String,Int}="extrap") where {T<:Union{Float64,ComplexF64}}
    # Ensure fs is a matrix (npts × nqty)
    fs_mat = fs isa Vector ? reshape(fs, :, 1) : fs
    npts, nqty = size(fs_mat)
    nintervals = npts - 1

    @assert length(xs) == npts "xs length must match fs rows"
    @assert issorted(xs) "xs must be sorted"

    bc = get(BC_MAP, bctype, NaturalBC())
    extrap = get(EXTRAP_MAP, bctype, :extension)

    # Compute spline coefficients for each quantity
    coeffs = zeros(T, nintervals, 4, nqty)
    fs1 = zeros(T, npts, nqty)

    for q in 1:nqty
        # Compute coefficients using tridiagonal solve
        c = _compute_spline_coeffs(xs, fs_mat[:, q], bc)

        # Store polynomial coefficients: f(x) = a + b*t + c*t² + d*t³ where t = (x - x_i)
        for i in 1:nintervals
            h = xs[i+1] - xs[i]
            coeffs[i, 1, q] = fs_mat[i, q]                      # a = f_i
            coeffs[i, 3, q] = c[i]                               # c coefficient
            coeffs[i, 4, q] = (c[i+1] - c[i]) / (3 * h)         # d = (c_{i+1} - c_i) / (3h)
            coeffs[i, 2, q] = (fs_mat[i+1, q] - fs_mat[i, q]) / h - h * (2*c[i] + c[i+1]) / 3  # b
        end

        # Compute first derivatives at grid points
        for i in 1:npts
            if i <= nintervals
                fs1[i, q] = coeffs[i, 2, q]
            else
                # Last point: use derivative from last interval at x = h
                h = xs[nintervals+1] - xs[nintervals]
                fs1[i, q] = coeffs[nintervals, 2, q] + 2*coeffs[nintervals, 3, q]*h + 3*coeffs[nintervals, 4, q]*h^2
            end
        end
    end

    # Initialize integration array (computed on demand)
    fsi = zeros(T, npts, nqty)

    # Work arrays
    _f = zeros(T, nqty)
    _f1 = zeros(T, nqty)
    _f2 = zeros(T, nqty)
    _f3 = zeros(T, nqty)

    CubicSpline1D{T}(xs, copy(fs_mat), fs1, fsi, bc, extrap, coeffs, _f, _f1, _f2, _f3)
end

"""
Compute second derivative coefficients (c) using tridiagonal system.
Standard cubic spline with natural boundary conditions.
"""
function _compute_spline_coeffs(xs::Vector{Float64}, fs::Vector{T}, bc) where {T}
    n = length(xs)

    # Build tridiagonal system for second derivatives
    h = diff(xs)

    # Right-hand side
    rhs = zeros(T, n)
    for i in 2:(n-1)
        rhs[i] = 3 * ((fs[i+1] - fs[i]) / h[i] - (fs[i] - fs[i-1]) / h[i-1])
    end

    # Tridiagonal matrix coefficients
    dl = zeros(Float64, n-1)  # sub-diagonal
    d = zeros(Float64, n)      # diagonal
    du = zeros(Float64, n-1)   # super-diagonal

    # Interior points
    for i in 2:(n-1)
        dl[i-1] = h[i-1]
        d[i] = 2 * (h[i-1] + h[i])
        du[i] = h[i]
    end

    # Boundary conditions (natural: c[1] = c[n] = 0)
    d[1] = 1.0
    d[n] = 1.0
    if n > 2
        du[1] = 0.0
        dl[n-1] = 0.0
    end

    # Handle periodic BC
    if bc isa PeriodicBC
        # Periodic: c[1] = c[n], continuity of first derivative
        d[1] = 2 * (h[end] + h[1])
        du[1] = h[1]
        dl[n-1] = h[end]
        d[n] = 1.0
        if n > 2
            du[n-1] = -1.0
        end
        rhs[1] = 3 * ((fs[2] - fs[1]) / h[1] - (fs[1] - fs[end]) / h[end])
        rhs[n] = 0
    end

    # Solve tridiagonal system
    c = zeros(T, n)
    if n == 2
        c[1] = rhs[1] / d[1]
        c[2] = rhs[2] / d[2]
    else
        # Thomas algorithm
        c_prime = zeros(Float64, n)
        d_prime = zeros(T, n)

        c_prime[1] = du[1] / d[1]
        d_prime[1] = rhs[1] / d[1]

        for i in 2:(n-1)
            denom = d[i] - dl[i-1] * c_prime[i-1]
            c_prime[i] = du[i] / denom
            d_prime[i] = (rhs[i] - dl[i-1] * d_prime[i-1]) / denom
        end
        d_prime[n] = (rhs[n] - dl[n-1] * d_prime[n-1]) / (d[n] - dl[n-1] * c_prime[n-1])

        c[n] = d_prime[n]
        for i in (n-1):-1:1
            c[i] = d_prime[i] - c_prime[i] * c[i+1]
        end
    end

    return c
end

"""
Find the interval index for evaluation point x.
Returns index i such that xs[i] <= x < xs[i+1].
"""
function _find_interval(xs::Vector{Float64}, x::Float64)
    n = length(xs)
    if x <= xs[1]
        return 1
    elseif x >= xs[end]
        return n - 1
    else
        # Binary search
        lo, hi = 1, n
        while lo < hi - 1
            mid = (lo + hi) >> 1
            if xs[mid] <= x
                lo = mid
            else
                hi = mid
            end
        end
        return lo
    end
end

"""
    evaluate!(spline, x) -> Vector{T}

Evaluate the spline at point x, returning values for all quantities.
Results stored in spline._f work array.
"""
function evaluate!(spline::CubicSpline1D{T}, x::Float64) where {T}
    i = _find_interval(spline.xs, x)
    t = x - spline.xs[i]

    for q in axes(spline.coeffs, 3)
        a, b, c, d = spline.coeffs[i, 1, q], spline.coeffs[i, 2, q],
        spline.coeffs[i, 3, q], spline.coeffs[i, 4, q]
        spline._f[q] = a + t * (b + t * (c + t * d))
    end

    return spline._f
end

"""
    deriv1!(spline, x) -> (f, f1)

Evaluate spline and first derivative at point x.
"""
function deriv1!(spline::CubicSpline1D{T}, x::Float64) where {T}
    i = _find_interval(spline.xs, x)
    t = x - spline.xs[i]

    for q in axes(spline.coeffs, 3)
        a, b, c, d = spline.coeffs[i, 1, q], spline.coeffs[i, 2, q],
        spline.coeffs[i, 3, q], spline.coeffs[i, 4, q]
        spline._f[q] = a + t * (b + t * (c + t * d))
        spline._f1[q] = b + t * (2c + 3d * t)
    end

    return spline._f, spline._f1
end

"""
    deriv2!(spline, x) -> (f, f1, f2)

Evaluate spline through second derivative at point x.
"""
function deriv2!(spline::CubicSpline1D{T}, x::Float64) where {T}
    i = _find_interval(spline.xs, x)
    t = x - spline.xs[i]

    for q in axes(spline.coeffs, 3)
        a, b, c, d = spline.coeffs[i, 1, q], spline.coeffs[i, 2, q],
        spline.coeffs[i, 3, q], spline.coeffs[i, 4, q]
        spline._f[q] = a + t * (b + t * (c + t * d))
        spline._f1[q] = b + t * (2c + 3d * t)
        spline._f2[q] = 2c + 6d * t
    end

    return spline._f, spline._f1, spline._f2
end

"""
    deriv3!(spline, x) -> (f, f1, f2, f3)

Evaluate spline through third derivative at point x.
For cubic splines, the third derivative is piecewise constant (6d within each interval).
"""
function deriv3!(spline::CubicSpline1D{T}, x::Float64) where {T}
    i = _find_interval(spline.xs, x)
    t = x - spline.xs[i]

    for q in axes(spline.coeffs, 3)
        a, b, c, d = spline.coeffs[i, 1, q], spline.coeffs[i, 2, q],
        spline.coeffs[i, 3, q], spline.coeffs[i, 4, q]
        spline._f[q] = a + t * (b + t * (c + t * d))
        spline._f1[q] = b + t * (2c + 3d * t)
        spline._f2[q] = 2c + 6d * t
        spline._f3[q] = 6d  # Third derivative is constant within interval
    end

    return spline._f, spline._f1, spline._f2, spline._f3
end

"""
    integrate!(spline)

Compute cumulative integrals at grid points and store in spline.fsi.
For each interval, integrates analytically: ∫(a + bt + ct² + dt³)dt = at + bt²/2 + ct³/3 + dt⁴/4
"""
function integrate!(spline::CubicSpline1D{T}) where {T}
    npts = length(spline.xs)
    nqty = size(spline.fs, 2)

    for q in 1:nqty
        spline.fsi[1, q] = zero(T)
        for i in 1:(npts-1)
            h = spline.xs[i+1] - spline.xs[i]
            a, b, c, d = spline.coeffs[i, 1, q], spline.coeffs[i, 2, q],
            spline.coeffs[i, 3, q], spline.coeffs[i, 4, q]
            # Integral from 0 to h
            integral = h * (a + h * (b/2 + h * (c/3 + h * d/4)))
            spline.fsi[i+1, q] = spline.fsi[i, q] + integral
        end
    end

    return spline.fsi
end

"""
    empty_CubicSpline1D(T::Type)

Create an empty/placeholder CubicSpline1D for type stability.
"""
function empty_CubicSpline1D(::Type{T}) where {T<:Union{Float64,ComplexF64}}
    xs = Float64[0.0, 1.0]
    fs = zeros(T, 2, 1)
    CubicSpline1D(xs, fs)
end

# Convenience methods for single-point evaluation returning scalar
(spline::CubicSpline1D)(x::Float64) = evaluate!(spline, x)

# Vectorized evaluation
function evaluate(spline::CubicSpline1D{T}, xs_eval::Vector{Float64}) where {T}
    npts = length(xs_eval)
    nqty = size(spline.fs, 2)
    result = zeros(T, npts, nqty)
    for (j, x) in enumerate(xs_eval)
        result[j, :] .= evaluate!(spline, x)
    end
    return result
end

# ============================================================================
# ComplexMatrixSpline - Matrix of splines for complex-valued stability matrices
# ============================================================================

"""
    ComplexMatrixSpline{S}

A matrix where each element is a separate 1D spline in psi, for complex-valued data.
Uses separate real/imaginary Float64 splines for each matrix element.

# Fields

  - `xs::Vector{Float64}`: Shared x-axis (psi grid)
  - `n1::Int`: First dimension of matrix
  - `n2::Int`: Second dimension of matrix
  - `real_splines::Matrix{S}`: Real part splines (n1 × n2)
  - `imag_splines::Matrix{S}`: Imaginary part splines (n1 × n2)
  - `_out::Matrix{ComplexF64}`: Work array for evaluation output
  - `_out1::Matrix{ComplexF64}`: Work array for first derivatives
  - `_out2::Matrix{ComplexF64}`: Work array for second derivatives
  - `_out3::Matrix{ComplexF64}`: Work array for third derivatives
"""
struct ComplexMatrixSpline{S}
    xs::Vector{Float64}
    n1::Int
    n2::Int
    real_splines::Matrix{S}
    imag_splines::Matrix{S}
    _out::Matrix{ComplexF64}
    _out1::Matrix{ComplexF64}
    _out2::Matrix{ComplexF64}
    _out3::Matrix{ComplexF64}
end

"""
    ComplexMatrixSpline(xs, data; bctype="extrap")

Create a ComplexMatrixSpline from a 3D array of complex data.

# Arguments

  - `xs::Vector{Float64}`: X-coordinates (psi grid)
  - `data::Array{ComplexF64,3}`: Complex data of shape (npsi, n1, n2)
  - `bctype`: Boundary condition type
"""
function ComplexMatrixSpline(xs::Vector{Float64}, data::Array{ComplexF64,3};
    bctype::String="extrap")
    npsi, n1, n2 = size(data)
    @assert length(xs) == npsi "xs length must match first dimension of data"

    # Create template spline to get type
    template = CubicSpline1D(xs, real.(data[:, 1, 1]); bctype=bctype)
    SplineType = typeof(template)

    # Create spline matrices
    real_splines = Matrix{SplineType}(undef, n1, n2)
    imag_splines = Matrix{SplineType}(undef, n1, n2)

    for i in 1:n1, j in 1:n2
        real_splines[i, j] = CubicSpline1D(xs, real.(data[:, i, j]); bctype=bctype)
        imag_splines[i, j] = CubicSpline1D(xs, imag.(data[:, i, j]); bctype=bctype)
    end

    # Work arrays
    _out = zeros(ComplexF64, n1, n2)
    _out1 = zeros(ComplexF64, n1, n2)
    _out2 = zeros(ComplexF64, n1, n2)
    _out3 = zeros(ComplexF64, n1, n2)

    ComplexMatrixSpline{SplineType}(xs, n1, n2, real_splines, imag_splines,
        _out, _out1, _out2, _out3)
end

"""
    ComplexMatrixSpline(xs, data_flat, n1, n2; bctype="extrap")

Create a ComplexMatrixSpline from a flattened 2D array of complex data.

# Arguments

  - `xs::Vector{Float64}`: X-coordinates (psi grid)
  - `data_flat::Matrix{ComplexF64}`: Flattened data of shape (npsi, n1*n2)
  - `n1, n2::Int`: Matrix dimensions
  - `bctype`: Boundary condition type
"""
function ComplexMatrixSpline(xs::Vector{Float64}, data_flat::Matrix{ComplexF64},
    n1::Int, n2::Int; bctype::String="extrap")
    npsi = length(xs)
    @assert size(data_flat, 1) == npsi
    @assert size(data_flat, 2) == n1 * n2

    # Reshape to 3D
    data = reshape(data_flat, npsi, n1, n2)
    ComplexMatrixSpline(xs, data; bctype=bctype)
end

"""
    evaluate!(cms, x) -> Matrix{ComplexF64}

Evaluate the ComplexMatrixSpline at point x.
"""
function evaluate!(cms::ComplexMatrixSpline, x::Float64)
    for i in 1:cms.n1, j in 1:cms.n2
        re_val = evaluate!(cms.real_splines[i, j], x)[1]
        im_val = evaluate!(cms.imag_splines[i, j], x)[1]
        cms._out[i, j] = re_val + im_val * 1im
    end
    return cms._out
end

"""
    deriv1!(cms, x) -> (f, f1)

Evaluate ComplexMatrixSpline and first derivative at point x.
"""
function deriv1!(cms::ComplexMatrixSpline, x::Float64)
    for i in 1:cms.n1, j in 1:cms.n2
        re, re1 = deriv1!(cms.real_splines[i, j], x)
        im_v, im1 = deriv1!(cms.imag_splines[i, j], x)
        cms._out[i, j] = re[1] + im_v[1] * 1im
        cms._out1[i, j] = re1[1] + im1[1] * 1im
    end
    return cms._out, cms._out1
end

"""
    deriv2!(cms, x) -> (f, f1, f2)

Evaluate ComplexMatrixSpline through second derivative at point x.
"""
function deriv2!(cms::ComplexMatrixSpline, x::Float64)
    for i in 1:cms.n1, j in 1:cms.n2
        re, re1, re2 = deriv2!(cms.real_splines[i, j], x)
        im_v, im1, im2 = deriv2!(cms.imag_splines[i, j], x)
        cms._out[i, j] = re[1] + im_v[1] * 1im
        cms._out1[i, j] = re1[1] + im1[1] * 1im
        cms._out2[i, j] = re2[1] + im2[1] * 1im
    end
    return cms._out, cms._out1, cms._out2
end

"""
    deriv3!(cms, x) -> (f, f1, f2, f3)

Evaluate ComplexMatrixSpline through third derivative at point x.
"""
function deriv3!(cms::ComplexMatrixSpline, x::Float64)
    for i in 1:cms.n1, j in 1:cms.n2
        re, re1, re2, re3 = deriv3!(cms.real_splines[i, j], x)
        im_v, im1, im2, im3 = deriv3!(cms.imag_splines[i, j], x)
        cms._out[i, j] = re[1] + im_v[1] * 1im
        cms._out1[i, j] = re1[1] + im1[1] * 1im
        cms._out2[i, j] = re2[1] + im2[1] * 1im
        cms._out3[i, j] = re3[1] + im3[1] * 1im
    end
    return cms._out, cms._out1, cms._out2, cms._out3
end

"""
    empty_ComplexMatrixSpline(n1, n2)

Create an empty/placeholder ComplexMatrixSpline for type stability.
"""
function empty_ComplexMatrixSpline(n1::Int=1, n2::Int=1)
    xs = Float64[0.0, 1.0]
    data = zeros(ComplexF64, 2, n1, n2)
    ComplexMatrixSpline(xs, data)
end
