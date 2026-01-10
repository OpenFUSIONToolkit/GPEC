"""
BicubicWrapper - 2D bicubic interpolation using Interpolations.jl

This wraps Interpolations.jl to provide bicubic spline interpolation with
support for first and second derivatives, including cross-derivatives (fxy).

# Uniform Spacing Requirement

Interpolations.jl requires uniformly spaced grids for proper scaling. This module
validates that input grids are approximately uniform (within numerical tolerance).
If non-uniform grids are required, consider using a different interpolation approach.

# Thread Safety Warning

The evaluation functions (`evaluate!`, `deriv1!`, `deriv2!`) mutate internal work
arrays and are NOT thread-safe. For multi-threaded usage:
- Use separate BicubicWrapper instances per thread, OR
- Use the non-mutating convenience function `bw(x, y)` which allocates

# References

- Interpolations.jl: https://github.com/JuliaMath/Interpolations.jl
- B-spline theory: de Boor, C. (1978). A Practical Guide to Splines.
"""

using Interpolations

# Relative tolerance for uniform spacing check
const UNIFORM_SPACING_RTOL = 1e-10

"""
    BicubicWrapper{I}

A 2D bicubic spline interpolator wrapping Interpolations.jl.

# Type Parameters

  - `I`: Type of the interpolation objects (for type stability)

# Fields

  - `xs::Vector{Float64}`: X-coordinates (first dimension)
  - `ys::Vector{Float64}`: Y-coordinates (second dimension)
  - `nqty::Int`: Number of quantities
  - `itps::Vector{I}`: Vector of scaled interpolation objects (one per quantity)
  - `fs::Array{Float64,3}`: Original function values (nx × ny × nqty)

# Work arrays (pre-allocated for efficiency)

  - `_f`, `_fx`, `_fy`, `_fxx`, `_fxy`, `_fyy`: Output vectors
"""
struct BicubicWrapper{I}
    xs::Vector{Float64}
    ys::Vector{Float64}
    nqty::Int
    itps::Vector{I}
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
    _check_uniform_spacing(coords::Vector{Float64}, name::String)

Validate that coordinates are uniformly spaced within tolerance.
Throws an error if spacing is non-uniform.
"""
function _check_uniform_spacing(coords::Vector{Float64}, name::String)
    n = length(coords)
    n < 2 && return  # Single point is trivially uniform

    # Compute spacing
    spacing = coords[2] - coords[1]
    if spacing ≤ 0
        error("$name coordinates must be strictly increasing")
    end

    # Check all spacings are equal within tolerance
    for i in 2:(n-1)
        local_spacing = coords[i+1] - coords[i]
        rel_error = abs(local_spacing - spacing) / spacing
        if rel_error > UNIFORM_SPACING_RTOL
            error(
                "$name coordinates must be uniformly spaced for Interpolations.jl. " *
                "Expected spacing $spacing, got $local_spacing at index $i " *
                "(relative error: $rel_error > tolerance $UNIFORM_SPACING_RTOL)"
            )
        end
    end
end

"""
    BicubicWrapper(xs, ys, fs)

Create a 2D bicubic spline interpolator.

# Arguments

  - `xs::Vector{Float64}`: X-coordinates (sorted, uniformly spaced)
  - `ys::Vector{Float64}`: Y-coordinates (sorted, uniformly spaced)
  - `fs::Array{Float64,3}`: Function values (nx × ny × nqty)

# Notes

Interpolations.jl requires uniformly spaced grids. This constructor validates
the spacing and throws an error if grids are non-uniform.

Periodic boundary conditions are not directly supported. For periodic data in y,
consider using FourierModeSplines instead.

# Example

```julia
xs = range(0, 1; length=10) |> collect
ys = range(0, 2π; length=20) |> collect
fs = zeros(10, 20, 2)  # 2 quantities
for i in 1:10, j in 1:20
    fs[i, j, 1] = sin(xs[i]) * cos(ys[j])
    fs[i, j, 2] = cos(xs[i]) * sin(ys[j])
end
bw = BicubicWrapper(xs, ys, fs)
f, fx, fy = deriv1!(bw, 0.5, π/4)
```
"""
function BicubicWrapper(xs::Vector{Float64}, ys::Vector{Float64},
    fs::Array{Float64,3})
    nx, ny, nqty = size(fs)
    @assert length(xs) == nx "xs length must match first dimension of fs"
    @assert length(ys) == ny "ys length must match second dimension of fs"

    # Validate uniform spacing
    _check_uniform_spacing(xs, "x")
    _check_uniform_spacing(ys, "y")

    # Create ranges for scaling (Interpolations.jl requires ranges)
    xs_range = range(xs[1], xs[end]; length=nx)
    ys_range = range(ys[1], ys[end]; length=ny)

    # Create first interpolation to determine type
    raw_itp_1 = interpolate(fs[:, :, 1], BSpline(Cubic(Line(OnGrid()))))
    scaled_itp_1 = scale(raw_itp_1, xs_range, ys_range)
    extrap_itp_1 = extrapolate(scaled_itp_1, Line())
    ItpType = typeof(extrap_itp_1)

    # Create type-stable vector
    itps = Vector{ItpType}(undef, nqty)
    itps[1] = extrap_itp_1

    # Create remaining interpolation objects
    for q in 2:nqty
        raw_itp = interpolate(fs[:, :, q], BSpline(Cubic(Line(OnGrid()))))
        scaled_itp = scale(raw_itp, xs_range, ys_range)
        itps[q] = extrapolate(scaled_itp, Line())
    end

    # Work arrays
    _f = zeros(Float64, nqty)
    _fx = zeros(Float64, nqty)
    _fy = zeros(Float64, nqty)
    _fxx = zeros(Float64, nqty)
    _fxy = zeros(Float64, nqty)
    _fyy = zeros(Float64, nqty)

    BicubicWrapper{ItpType}(xs, ys, nqty, itps, copy(fs), _f, _fx, _fy, _fxx, _fxy, _fyy)
end

"""
    evaluate!(bw, x, y) -> Vector{Float64}

Evaluate the bicubic spline at point (x, y).

Returns a reference to the internal work array `_f`.
The result is valid until the next mutating call on this wrapper.

Thread-safety: NOT thread-safe. Use separate wrappers per thread.
"""
function evaluate!(bw::BicubicWrapper{I}, x::Float64, y::Float64) where {I}
    @inbounds for q in 1:bw.nqty
        bw._f[q] = bw.itps[q](x, y)
    end
    return bw._f
end

"""
    deriv1!(bw, x, y) -> (f, fx, fy)

Evaluate bicubic spline and first derivatives at point (x, y).

Returns references to internal work arrays.
The results are valid until the next mutating call on this wrapper.

Thread-safety: NOT thread-safe. Use separate wrappers per thread.
"""
function deriv1!(bw::BicubicWrapper{I}, x::Float64, y::Float64) where {I}
    @inbounds for q in 1:bw.nqty
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

Returns references to internal work arrays.
The results are valid until the next mutating call on this wrapper.

Thread-safety: NOT thread-safe. Use separate wrappers per thread.
"""
function deriv2!(bw::BicubicWrapper{I}, x::Float64, y::Float64) where {I}
    @inbounds for q in 1:bw.nqty
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

# Convenience evaluation (allocates, but thread-safe)
(bw::BicubicWrapper)(x::Float64, y::Float64) = evaluate!(bw, x, y)
