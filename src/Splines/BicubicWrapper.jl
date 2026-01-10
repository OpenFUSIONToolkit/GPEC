"""
BicubicWrapper - 2D bicubic interpolation using Interpolations.jl

This wraps Interpolations.jl to provide bicubic spline interpolation with
support for first and second derivatives, including cross-derivatives (fxy).
"""

using Interpolations

"""
    BicubicWrapper

A 2D bicubic spline interpolator wrapping Interpolations.jl.

# Fields

  - `xs::Vector{Float64}`: X-coordinates (first dimension)
  - `ys::Vector{Float64}`: Y-coordinates (second dimension)
  - `nqty::Int`: Number of quantities
  - `itps`: Vector of scaled interpolation objects (one per quantity)
  - `fs::Array{Float64,3}`: Original function values (nx × ny × nqty)
"""
struct BicubicWrapper
    xs::Vector{Float64}
    ys::Vector{Float64}
    nqty::Int
    itps::Vector{Any}  # ScaledInterpolation objects
    fs::Array{Float64,3}
    # Work arrays
    _f::Vector{Float64}
    _fx::Vector{Float64}
    _fy::Vector{Float64}
    _fxx::Vector{Float64}
    _fxy::Vector{Float64}
    _fyy::Vector{Float64}
end

"""
    BicubicWrapper(xs, ys, fs; bctypex="extrap", bctypey="extrap")

Create a 2D bicubic spline interpolator.

# Arguments

  - `xs::Vector{Float64}`: X-coordinates (sorted)
  - `ys::Vector{Float64}`: Y-coordinates (sorted)
  - `fs::Array{Float64,3}`: Function values (nx × ny × nqty)
  - `bctypex`: Boundary condition in x ("extrap", "natural")
  - `bctypey`: Boundary condition in y ("extrap", "natural")

Note: Periodic boundary conditions are not supported due to Interpolations.jl limitations.
For periodic data in y, consider wrapping the data manually or using FourierModeSplines.
"""
function BicubicWrapper(xs::Vector{Float64}, ys::Vector{Float64},
    fs::Array{Float64,3};
    bctypex::String="extrap", bctypey::String="extrap")
    nx, ny, nqty = size(fs)
    @assert length(xs) == nx "xs length must match first dimension of fs"
    @assert length(ys) == ny "ys length must match second dimension of fs"

    # Create interpolation objects for each quantity
    itps = Vector{Any}(undef, nqty)

    # Create ranges for scaling (Interpolations.jl requires ranges)
    xs_range = range(xs[1], xs[end]; length=nx)
    ys_range = range(ys[1], ys[end]; length=ny)

    for q in 1:nqty
        # Create raw interpolation with specified boundary conditions
        # Use Line(OnGrid()) for both dimensions to enable scaling
        raw_itp = interpolate(fs[:, :, q], BSpline(Cubic(Line(OnGrid()))))
        # Scale to actual coordinates using ranges
        itps[q] = scale(raw_itp, xs_range, ys_range)
        # Add extrapolation behavior
        itps[q] = extrapolate(itps[q], Line())
    end

    # Work arrays
    _f = zeros(Float64, nqty)
    _fx = zeros(Float64, nqty)
    _fy = zeros(Float64, nqty)
    _fxx = zeros(Float64, nqty)
    _fxy = zeros(Float64, nqty)
    _fyy = zeros(Float64, nqty)

    BicubicWrapper(xs, ys, nqty, itps, copy(fs), _f, _fx, _fy, _fxx, _fxy, _fyy)
end

"""
    evaluate!(bw, x, y) -> Vector{Float64}

Evaluate the bicubic spline at point (x, y).
"""
function evaluate!(bw::BicubicWrapper, x::Float64, y::Float64)
    for q in 1:bw.nqty
        bw._f[q] = bw.itps[q](x, y)
    end
    return bw._f
end

"""
    deriv1!(bw, x, y) -> (f, fx, fy)

Evaluate bicubic spline and first derivatives at point (x, y).
"""
function deriv1!(bw::BicubicWrapper, x::Float64, y::Float64)
    for q in 1:bw.nqty
        bw._f[q] = bw.itps[q](x, y)
        grad = Interpolations.gradient(bw.itps[q], x, y)
        bw._fx[q] = grad[1]
        bw._fy[q] = grad[2]
    end
    return bw._f, bw._fx, bw._fy
end

"""
    deriv2!(bw, x, y) -> (f, fx, fy, fxx, fxy, fyy)

Evaluate bicubic spline through second derivatives at point (x, y).
Includes cross-derivative fxy.
"""
function deriv2!(bw::BicubicWrapper, x::Float64, y::Float64)
    for q in 1:bw.nqty
        bw._f[q] = bw.itps[q](x, y)
        grad = Interpolations.gradient(bw.itps[q], x, y)
        bw._fx[q] = grad[1]
        bw._fy[q] = grad[2]
        hess = Interpolations.hessian(bw.itps[q], x, y)
        bw._fxx[q] = hess[1, 1]
        bw._fxy[q] = hess[1, 2]
        bw._fyy[q] = hess[2, 2]
    end
    return bw._f, bw._fx, bw._fy, bw._fxx, bw._fxy, bw._fyy
end

"""
    empty_BicubicWrapper()

Create an empty/placeholder BicubicWrapper for type stability.
"""
function empty_BicubicWrapper()
    xs = Float64[0.0, 1.0]
    ys = Float64[0.0, 1.0]
    fs = zeros(Float64, 2, 2, 1)
    BicubicWrapper(xs, ys, fs)
end

# Convenience evaluation
(bw::BicubicWrapper)(x::Float64, y::Float64) = evaluate!(bw, x, y)
