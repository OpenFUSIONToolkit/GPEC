"""
FastCubicSpline - Lightweight wrapper around FastInterpolations.jl

This module provides a thin wrapper around FastInterpolations.jl's CubicInterpolant,
offering an API similar to SplineAdapter's CubicSpline1D but optimized for single-quantity
interpolation with FastInterpolations doing the heavy lifting.

The "extrap" boundary condition is implemented by computing endpoint derivatives
via 4-point polynomial extrapolation, then using FastInterpolations' BCPair API.

## Performance Comparison (FastInterpolations v0.2.4)

| Operation          | CubicSpline1D | FastCubicSpline1D | FastInterp direct | Notes                   |
|--------------------|---------------|-------------------|-------------------|-------------------------|
| evaluate (1 pt)    |      5.5 ns   |        9.4 ns     |        9.3 ns     | Single point            |
| deriv1 (1 pt)      |      5.5 ns   |        9.2 ns     |        9.3 ns     | Single point            |
| deriv2 (1 pt)      |      5.2 ns   |        8.2 ns     |        8.2 ns     | Single point            |
| deriv3 (1 pt)      |      N/A      |        7.4 ns     |        7.4 ns     | Single point            |
| Monotonic loop     |      4.3 ns/pt|        7.7 ns/pt  |        7.7 ns/pt  | CubicSpline1D cached    |
| Monotonic w/ hint  |      N/A      |        2.9 ns/pt  |        2.9 ns/pt  | Use hint=Ref(1)         |
| Random loop        |     19.6 ns/pt|        8.7 ns/pt  |        8.7 ns/pt  | FastCubicSpline1D wins  |

**Summary**: CubicSpline1D with cached interval search is ~1.8x faster for monotonic
access patterns (typical in ODE integration). FastCubicSpline1D is ~2.3x faster for
random access. For monotonic access, use the `hint` keyword argument for fastest
performance (2.9 ns/pt). The FastCubicSpline1D wrapper now supports all FastInterpolations
v0.2.4 features including `search` and `hint` keyword arguments.
"""

using FastInterpolations

# =============================================================================
# FastCubicSpline1D - Single Quantity (Type-Stable)
# =============================================================================

"""
    FastCubicSpline1D{T, I, D1, D2, D3}

A 1D cubic spline interpolator wrapping FastInterpolations.jl for a single quantity.

For multiple quantities, use multiple FastCubicSpline1D instances or the
multi-quantity variant FastCubicSpline1DMulti.

# Type Parameters

  - `T`: Element type (Float64 or ComplexF64)
  - `I`: Interpolant type from FastInterpolations
  - `D1`: First derivative view type
  - `D2`: Second derivative view type
  - `D3`: Third derivative view type

# Keyword Arguments for Evaluation

The main evaluation method `(spline)(x)` supports:

  - `search`: Search strategy (default: `Binary()`). Use `LinearBinary()` for monotonic access.
  - `hint`: Mutable `Ref{Int}` for interval index. Enables O(1) lookups for sequential access
    (2.9 ns/pt vs 7.7 ns/pt). FastInterpolations auto-updates the hint after each call.

Note: Derivative methods (`deriv1`, `deriv2`, `deriv3`) use FastInterpolations' DerivativeView
which doesn't support search/hint kwargs.
"""
struct FastCubicSpline1D{T<:Union{Float64,ComplexF64},I,D1,D2,D3}
    xs::Vector{Float64}
    fs::Vector{T}
    fs1::Vector{T}        # First derivatives at grid points
    fsi::Vector{T}        # Cumulative integrals
    interp::I             # CubicInterpolant (or tuple for complex)
    d1_view::D1           # deriv1 view
    d2_view::D2           # deriv2 view
    d3_view::D3           # deriv3 view
end

"""
    _estimate_endpoint_derivative_fast(xs, fs, x0)

Estimate f'(x0) using cubic Lagrange interpolation through 4 points.
This matches the GPEC "extrap" boundary condition.
"""
@inline function _estimate_endpoint_derivative_fast(xs::AbstractVector{Float64},
    fs::AbstractVector{T}, x0::Float64) where {T}
    x1, x2, x3, x4 = xs[1], xs[2], xs[3], xs[4]
    f1, f2, f3, f4 = fs[1], fs[2], fs[3], fs[4]

    d12 = x1 - x2
    d13 = x1 - x3
    d14 = x1 - x4
    d23 = x2 - x3
    d24 = x2 - x4
    d34 = x3 - x4

    L1_denom = d12 * d13 * d14
    L1_deriv = ((x0 - x2) * (x0 - x3) + (x0 - x2) * (x0 - x4) + (x0 - x3) * (x0 - x4)) / L1_denom

    L2_denom = (-d12) * d23 * d24
    L2_deriv = ((x0 - x1) * (x0 - x3) + (x0 - x1) * (x0 - x4) + (x0 - x3) * (x0 - x4)) / L2_denom

    L3_denom = (-d13) * (-d23) * d34
    L3_deriv = ((x0 - x1) * (x0 - x2) + (x0 - x1) * (x0 - x4) + (x0 - x2) * (x0 - x4)) / L3_denom

    L4_denom = (-d14) * (-d24) * (-d34)
    L4_deriv = ((x0 - x1) * (x0 - x2) + (x0 - x1) * (x0 - x3) + (x0 - x2) * (x0 - x3)) / L4_denom

    return f1 * L1_deriv + f2 * L2_deriv + f3 * L3_deriv + f4 * L4_deriv
end

"""
    _make_bc(bctype, xs, fs)

Create the appropriate BC object for FastInterpolations.
"""
function _make_bc(bctype::String, xs::Vector{Float64}, fs::Vector{Float64})
    if bctype == "natural"
        return NaturalBC()
    elseif bctype == "periodic"
        return PeriodicBC()
    elseif bctype == "extrap"
        n = length(xs)
        yp_left = _estimate_endpoint_derivative_fast(xs[1:4], fs[1:4], xs[1])
        yp_right = _estimate_endpoint_derivative_fast(xs[(n-3):n], fs[(n-3):n], xs[n])
        return BCPair(Deriv1(yp_left), Deriv1(yp_right))
    else
        error("Unknown bctype: $bctype. Valid: \"natural\", \"periodic\", \"extrap\"")
    end
end

"""
    FastCubicSpline1D(xs, fs; bctype="extrap")

Create a 1D cubic spline interpolator using FastInterpolations.jl.

# Arguments

  - `xs::Vector{Float64}`: Grid points (must be sorted ascending, at least 4 points)

  - `fs::Vector{T}`: Function values at grid points (single quantity)
  - `bctype`: Boundary condition type:

      + `"natural"`: f''(x) = 0 at endpoints
      + `"periodic"`: f and f' continuous across domain wrap
      + `"extrap"`: Endpoint derivatives from 4-point polynomial extrapolation
"""
function FastCubicSpline1D(xs::Vector{Float64}, fs::Vector{Float64};
    bctype::String="extrap")

    npts = length(xs)
    @assert length(fs) == npts "xs and fs must have same length"
    @assert npts >= 4 "Need at least 4 points for spline"
    @assert issorted(xs) "xs must be sorted ascending"

    bc = _make_bc(bctype, xs, fs)
    itp = cubic_interp(xs, fs; bc=bc)
    d1 = FastInterpolations.deriv1(itp)
    d2 = FastInterpolations.deriv2(itp)
    d3 = FastInterpolations.deriv3(itp)

    # Pre-compute first derivatives at grid points
    fs1 = [d1(x) for x in xs]
    fsi = zeros(Float64, npts)

    FastCubicSpline1D{Float64,typeof(itp),typeof(d1),typeof(d2),typeof(d3)}(
        xs, copy(fs), fs1, fsi, itp, d1, d2, d3
    )
end

# Complex version - stores separate real/imag interpolants
function FastCubicSpline1D(xs::Vector{Float64}, fs::Vector{ComplexF64};
    bctype::String="extrap")

    npts = length(xs)
    @assert length(fs) == npts "xs and fs must have same length"
    @assert npts >= 4 "Need at least 4 points for spline"
    @assert issorted(xs) "xs must be sorted ascending"

    fs_real = real.(fs)
    fs_imag = imag.(fs)

    bc_real = _make_bc(bctype, xs, fs_real)
    bc_imag = _make_bc(bctype, xs, fs_imag)

    itp_real = cubic_interp(xs, fs_real; bc=bc_real)
    itp_imag = cubic_interp(xs, fs_imag; bc=bc_imag)
    d1_real = FastInterpolations.deriv1(itp_real)
    d1_imag = FastInterpolations.deriv1(itp_imag)
    d2_real = FastInterpolations.deriv2(itp_real)
    d2_imag = FastInterpolations.deriv2(itp_imag)
    d3_real = FastInterpolations.deriv3(itp_real)
    d3_imag = FastInterpolations.deriv3(itp_imag)

    itp = (itp_real, itp_imag)
    d1 = (d1_real, d1_imag)
    d2 = (d2_real, d2_imag)
    d3 = (d3_real, d3_imag)

    fs1 = [d1_real(x) + 1im * d1_imag(x) for x in xs]
    fsi = zeros(ComplexF64, npts)

    FastCubicSpline1D{ComplexF64,typeof(itp),typeof(d1),typeof(d2),typeof(d3)}(
        xs, copy(fs), fs1, fsi, itp, d1, d2, d3
    )
end

# =============================================================================
# Evaluation Methods
# =============================================================================

# Note: FastInterpolations' DerivativeView doesn't support search/hint kwargs,
# only the main CubicInterpolant does. We provide a unified API but the kwargs
# only affect the main interpolant evaluation (not derivative views).

"""
    (spline)(x; search=nothing, hint=nothing) -> T

Evaluate the spline at point x.

# Keyword Arguments

  - `search`: Search strategy (e.g., `LinearBinary()` for monotonic access patterns).
    Default uses binary search. Only affects this method, not derivative methods.
  - `hint`: Mutable `Ref{Int}` for interval index. Use `hint=Ref(1)` and FastInterpolations
    will automatically update it after each call, enabling O(1) lookups for sequential access.
    Only affects this method, not derivative methods.

# Example with hint for fast sequential access

```julia
spline = FastCubicSpline1D(xs, fs)
hint = Ref(1)
for x in sorted_points
    y = spline(x; hint=hint)  # hint[] auto-updated after each call
end
```
"""
@inline function (spline::FastCubicSpline1D{Float64})(x::Float64; search=nothing, hint=nothing)
    if search === nothing && hint === nothing
        return spline.interp(x)
    elseif hint === nothing
        return spline.interp(x; search=search)
    elseif search === nothing
        return spline.interp(x; hint=hint)
    else
        return spline.interp(x; search=search, hint=hint)
    end
end

@inline function (spline::FastCubicSpline1D{ComplexF64})(x::Float64; search=nothing, hint=nothing)
    itp_real, itp_imag = spline.interp
    if search === nothing && hint === nothing
        return itp_real(x) + 1im * itp_imag(x)
    elseif hint === nothing
        return itp_real(x; search=search) + 1im * itp_imag(x; search=search)
    elseif search === nothing
        return itp_real(x; hint=hint) + 1im * itp_imag(x; hint=hint)
    else
        return itp_real(x; search=search, hint=hint) + 1im * itp_imag(x; search=search, hint=hint)
    end
end

"""
    evaluate(spline, x; search=nothing, hint=nothing) -> T

Non-mutating evaluation (for API compatibility).
"""
@inline evaluate(spline::FastCubicSpline1D, x::Float64; search=nothing, hint=nothing) = spline(x; search=search, hint=hint)

"""
    deriv1(spline, x) -> T

Evaluate first derivative at point x.
"""
@inline function deriv1(spline::FastCubicSpline1D{Float64}, x::Float64)
    return spline.d1_view(x)
end

@inline function deriv1(spline::FastCubicSpline1D{ComplexF64}, x::Float64)
    d1_r, d1_i = spline.d1_view
    return d1_r(x) + 1im * d1_i(x)
end

"""
    deriv2(spline, x) -> T

Evaluate second derivative at point x.
"""
@inline function deriv2(spline::FastCubicSpline1D{Float64}, x::Float64)
    return spline.d2_view(x)
end

@inline function deriv2(spline::FastCubicSpline1D{ComplexF64}, x::Float64)
    d2_r, d2_i = spline.d2_view
    return d2_r(x) + 1im * d2_i(x)
end

"""
    deriv3(spline, x) -> T

Evaluate third derivative at point x.
For cubic splines, f'''(x) is constant within each interval.
"""
@inline function deriv3(spline::FastCubicSpline1D{Float64}, x::Float64)
    return spline.d3_view(x)
end

@inline function deriv3(spline::FastCubicSpline1D{ComplexF64}, x::Float64)
    d3_r, d3_i = spline.d3_view
    return d3_r(x) + 1im * d3_i(x)
end

"""
    integrate!(spline)

Compute cumulative integrals at grid points using trapezoidal rule.
"""
function integrate!(spline::FastCubicSpline1D{T}) where {T}
    npts = length(spline.xs)
    spline.fsi[1] = zero(T)
    @inbounds for i in 1:(npts-1)
        h = spline.xs[i+1] - spline.xs[i]
        spline.fsi[i+1] = spline.fsi[i] + h * (spline.fs[i] + spline.fs[i+1]) / 2
    end
    return spline.fsi
end

"""
    empty_FastCubicSpline1D(T::Type)

Create an empty/placeholder FastCubicSpline1D for type stability.
"""
function empty_FastCubicSpline1D(::Type{T}) where {T<:Union{Float64,ComplexF64}}
    xs = collect(range(0.0, 1.0; length=5))
    fs = zeros(T, 5)
    FastCubicSpline1D(xs, fs; bctype="natural")
end
