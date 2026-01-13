"""
FastCubicSpline - Lightweight wrapper around FastInterpolations.jl

This module provides a thin wrapper around FastInterpolations.jl's CubicInterpolant,
offering an API similar to SplineAdapter's CubicSpline1D but optimized for single-quantity
interpolation with FastInterpolations doing the heavy lifting.

The "extrap" boundary condition is implemented by computing endpoint derivatives
via 4-point polynomial extrapolation, then using FastInterpolations' BCPair API.

## Performance Comparison (FastInterpolations v0.2.2)

| Operation       | CubicSpline1D | FastCubicSpline1D | Notes                    |
|-----------------|---------------|-------------------|--------------------------|
| evaluate (1 pt) |      5.1 ns   |        9.3 ns     | Single point             |
| deriv1 (1 pt)   |      5.2 ns   |        9.3 ns     | Single point             |
| deriv2 (1 pt)   |      4.9 ns   |        8.1 ns     | Single point             |
| deriv3 (1 pt)   |      4.6 ns   |        6.2 ns     | Single point             |
| Monotonic loop  |      4.7 ns/pt|        8.9 ns/pt  | Best case for cached     |
| Random loop     |     14.5 ns/pt|        8.8 ns/pt  | FastCubicSpline1D wins   |

**Summary**: CubicSpline1D with cached interval search is ~2x faster for monotonic
access patterns (typical in ODE integration). FastCubicSpline1D is faster for random
access but lacks multi-quantity support and cached interval optimization.
"""

using FastInterpolations

# =============================================================================
# FastCubicSpline1D - Single Quantity (Type-Stable)
# =============================================================================

"""
    FastCubicSpline1D{T, I, D1, D2}

A 1D cubic spline interpolator wrapping FastInterpolations.jl for a single quantity.

For multiple quantities, use multiple FastCubicSpline1D instances or the
multi-quantity variant FastCubicSpline1DMulti.

# Type Parameters

  - `T`: Element type (Float64 or ComplexF64)
  - `I`: Interpolant type from FastInterpolations
  - `D1`: First derivative view type
  - `D2`: Second derivative view type
"""
struct FastCubicSpline1D{T<:Union{Float64,ComplexF64},I,D1,D2}
    xs::Vector{Float64}
    fs::Vector{T}
    fs1::Vector{T}        # First derivatives at grid points
    fsi::Vector{T}        # Cumulative integrals
    interp::I             # CubicInterpolant (or tuple for complex)
    d1_view::D1           # deriv1 view
    d2_view::D2           # deriv2 view
    f3_coeffs::Vector{Float64}    # Pre-computed f'''(x) = (z[i+1]-z[i])/h for each interval
    f3_coeffs_imag::Vector{Float64}  # Imaginary part for complex (empty for real)
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

    # Pre-compute first derivatives at grid points
    fs1 = [d1(x) for x in xs]
    fsi = zeros(Float64, npts)

    # Pre-compute third derivative coefficients for each interval
    z = itp.z
    f3_coeffs = Vector{Float64}(undef, npts - 1)
    @inbounds for i in 1:(npts-1)
        h_i = xs[i+1] - xs[i]
        f3_coeffs[i] = (z[i+1] - z[i]) / h_i
    end
    f3_coeffs_imag = Float64[]

    FastCubicSpline1D{Float64,typeof(itp),typeof(d1),typeof(d2)}(
        xs, copy(fs), fs1, fsi, itp, d1, d2, f3_coeffs, f3_coeffs_imag
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

    itp = (itp_real, itp_imag)
    d1 = (d1_real, d1_imag)
    d2 = (d2_real, d2_imag)

    fs1 = [d1_real(x) + 1im * d1_imag(x) for x in xs]
    fsi = zeros(ComplexF64, npts)

    # Pre-compute third derivative coefficients for each interval
    z_real = itp_real.z
    z_imag_arr = itp_imag.z
    f3_coeffs = Vector{Float64}(undef, npts - 1)
    f3_coeffs_imag = Vector{Float64}(undef, npts - 1)
    @inbounds for i in 1:(npts-1)
        h_i = xs[i+1] - xs[i]
        f3_coeffs[i] = (z_real[i+1] - z_real[i]) / h_i
        f3_coeffs_imag[i] = (z_imag_arr[i+1] - z_imag_arr[i]) / h_i
    end

    FastCubicSpline1D{ComplexF64,typeof(itp),typeof(d1),typeof(d2)}(
        xs, copy(fs), fs1, fsi, itp, d1, d2, f3_coeffs, f3_coeffs_imag
    )
end

# =============================================================================
# Evaluation Methods
# =============================================================================

"""
    (spline)(x) -> T

Evaluate the spline at point x.
"""
@inline function (spline::FastCubicSpline1D{Float64})(x::Float64)
    return spline.interp(x)
end

@inline function (spline::FastCubicSpline1D{ComplexF64})(x::Float64)
    itp_real, itp_imag = spline.interp
    return itp_real(x) + 1im * itp_imag(x)
end

"""
    evaluate(spline, x) -> T

Non-mutating evaluation (for API compatibility).
"""
@inline evaluate(spline::FastCubicSpline1D, x::Float64) = spline(x)

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
    _fast_find_interval(xs, x) -> Int

Find interval index i such that xs[i] <= x < xs[i+1].
Custom binary search optimized for inlining.
"""
@inline function _fast_find_interval(xs::Vector{Float64}, x::Float64)
    n = length(xs)
    @inbounds begin
        if x <= xs[1]
            return 1
        elseif x >= xs[end]
            return n - 1
        else
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
end

"""
    deriv3(spline, x) -> T

Evaluate third derivative at point x.
For cubic splines, f'''(x) is constant within each interval (pre-computed during construction).
"""
@inline function deriv3(spline::FastCubicSpline1D{Float64}, x::Float64)
    i = _fast_find_interval(spline.xs, x)
    @inbounds return spline.f3_coeffs[i]
end

@inline function deriv3(spline::FastCubicSpline1D{ComplexF64}, x::Float64)
    i = _fast_find_interval(spline.xs, x)
    @inbounds return spline.f3_coeffs[i] + 1im * spline.f3_coeffs_imag[i]
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
