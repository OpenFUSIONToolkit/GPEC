"""
FastInterpolationsAdaptor - Helpers and types for FastInterpolations.jl in JPEC

This module provides:
- cumulative_integral: Exact spline integration (uses cubic spline coefficients)
- total_integral: Optimized final-value-only integration
- Internal helpers for BicubicSpline boundary condition handling

## Boundary Conditions

Use FastInterpolations' native BC types:
- `NaturalBC()`: f''(x) = 0 at endpoints
- `PeriodicBC()`: Periodic (requires fs[end] == fs[1])
- `CubicFit()`: Automatic endpoint derivative estimation (4-point fit)

## Performance Notes

For monotonic access patterns (typical in ODE integration), use `hint=Ref(1)` with
`search=LinearBinary()` for optimal performance (~4 ns/pt).
"""

using FastInterpolations
using LinearAlgebra

# Re-export FastInterpolations BC types for convenience
const NaturalBC = FastInterpolations.NaturalBC
const PeriodicBC = FastInterpolations.PeriodicBC
const CubicFit = FastInterpolations.CubicFit
const BCPair = FastInterpolations.BCPair
const Deriv1 = FastInterpolations.Deriv1
const Deriv2 = FastInterpolations.Deriv2

# =============================================================================
# Boundary Condition Helpers (Internal)
# =============================================================================

"""
    _estimate_endpoint_derivative(xs, fs, x0)

Estimate f'(x0) using cubic Lagrange interpolation through 4 points.
Used internally by BicubicSpline for extrap boundary conditions.
"""
@inline function _estimate_endpoint_derivative(xs::AbstractVector{Float64},
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

# =============================================================================
# Exact Spline Integration
# =============================================================================

"""
    cumulative_integral(xs, fs; bc=NaturalBC()) -> Vector/Matrix

Compute exact cumulative integral of cubic spline fitted to (xs, fs).
Returns array where result[1] = 0 and result[i+1] = result[i] + ∫_{xs[i]}^{xs[i+1]} S(x) dx,
where S(x) is the cubic spline interpolant.

For FastInterpolations' moment formulation, the exact integral over interval [x_i, x_{i+1}] is:
∫S(x)dx = h/2 * (y_i + y_{i+1}) - h³/24 * (z_i + z_{i+1})
where h is the interval width, y are function values, and z are second derivatives (moments).

This is more accurate than trapezoidal rule, matching the Fortran GPEC behavior.

# Arguments

  - `xs`: Grid points (sorted ascending)
  - `fs`: Function values (vector or matrix with columns as quantities)
  - `bc`: Boundary condition for spline fitting (default: NaturalBC())
"""
function cumulative_integral(xs::AbstractVector{Float64}, fs::AbstractVector{T}; bc=NaturalBC()) where {T}
    npts = length(xs)
    fsi = zeros(T, npts)

    # Create spline to get second derivative coefficients
    itp = cubic_interp(xs, Vector{Float64}(fs); bc=bc)
    y = itp.y
    z = itp.z
    h = itp.cache.spacing.h  # interval widths: h[i] = xs[i+1] - xs[i]

    @inbounds for i in 1:(npts-1)
        hi = h[i]
        # Exact integral: h/2*(y_i + y_{i+1}) - h³/24*(z_i + z_{i+1})
        fsi[i+1] = fsi[i] + hi/2 * (y[i] + y[i+1]) - hi^3/24 * (z[i] + z[i+1])
    end
    return fsi
end

function cumulative_integral(xs::AbstractVector{Float64}, fs::AbstractMatrix{T}; bc=NaturalBC()) where {T}
    npts, nqty = size(fs)
    fsi = zeros(T, npts, nqty)

    @inbounds for k in 1:nqty
        # Create spline for this column
        itp = cubic_interp(xs, Vector{Float64}(fs[:, k]); bc=bc)
        y = itp.y
        z = itp.z
        h = itp.cache.spacing.h

        for i in 1:(npts-1)
            hi = h[i]
            fsi[i+1, k] = fsi[i, k] + hi/2 * (y[i] + y[i+1]) - hi^3/24 * (z[i] + z[i+1])
        end
    end
    return fsi
end

"""
    integrate_spline(itp::CubicInterpolant) -> Float64

Compute exact integral of cubic spline over its entire domain.
"""
function integrate_spline(itp::FastInterpolations.CubicInterpolant{T}) where {T}
    y = itp.y
    z = itp.z
    h = itp.cache.spacing.h
    n = length(y)

    total = zero(T)
    @inbounds for i in 1:(n-1)
        hi = h[i]
        total += hi/2 * (y[i] + y[i+1]) - hi^3/24 * (z[i] + z[i+1])
    end
    return total
end

# =============================================================================
# Total Integral (Optimized for final value only)
# =============================================================================

"""
    total_integral(xs, fs; bc=NaturalBC()) -> Float64 or Vector{Float64}

Compute the total integral of a cubic spline over the entire domain [xs[1], xs[end]].
This is an optimized version of `cumulative_integral(xs, fs)[end, :]` that avoids
allocating the full cumulative array.

For the exact spline integration formula over interval [x_i, x_{i+1}]:
integral = h/2 * (y_i + y_{i+1}) - h^3/24 * (z_i + z_{i+1})
where h is the interval width, y are function values, and z are second derivatives (moments).

# Arguments

  - `xs::AbstractVector{Float64}`: Grid points (sorted ascending)
  - `fs`: Function values - vector (returns scalar) or matrix (returns vector, one value per column)
  - `bc`: Boundary condition for spline fitting (default: NaturalBC())

# Returns

  - For vector `fs`: Returns a scalar Float64
  - For matrix `fs`: Returns a Vector{Float64} of length `size(fs, 2)`

# Performance

Avoids allocating the full (npts,) or (npts, nqty) cumulative integral array.
For matrix input, uses CubicSeriesInterpolant internally for efficiency.

# Example

```julia
xs = collect(0.0:0.01:1.0)
fs = xs .^ 2
total = total_integral(xs, fs)  # Returns approx 1/3

# Matrix version
Y = hcat(xs .^ 2, sin.(xs))
totals = total_integral(xs, Y)  # Returns [1/3, 1-cos(1)] approximately    # Create spline to get second derivative coefficients
```
"""
function total_integral(xs::AbstractVector{Float64}, fs::AbstractVector{Float64}; bc=NaturalBC())
    # Create spline to get second derivative coefficients
    itp = cubic_interp(xs, fs; bc=bc)
    return integrate_spline(itp)
end

"""
    total_integral(xs, fs::AbstractMatrix; bc=NaturalBC()) -> Vector{Float64}

Matrix version: compute total integral for each column of fs.
Returns a vector of length `size(fs, 2)`.
"""
function total_integral(xs::AbstractVector{Float64}, fs::AbstractMatrix{Float64}; bc=NaturalBC())
    npts, nqty = size(fs)

    # Use CubicSeriesInterpolant for efficient multi-quantity handling
    itp = cubic_interp(xs, fs; bc=bc)
    y = itp.y   # (npts, nqty)
    z = itp.z   # (npts, nqty)
    h = itp.cache.spacing.h  # (npts-1,)

    # Allocate only the output vector, not the full (npts, nqty) cumulative array
    result = zeros(Float64, nqty)

    @inbounds for i in 1:(npts-1)
        hi = h[i]
        hi_half = hi / 2
        hi3_24 = hi^3 / 24
        for k in 1:nqty
            result[k] += hi_half * (y[i, k] + y[i+1, k]) - hi3_24 * (z[i, k] + z[i+1, k])
        end
    end

    return result
end

"""
    total_integral!(result::AbstractVector{Float64}, xs, fs::AbstractMatrix; bc=NaturalBC()) -> result

In-place matrix version: compute total integral for each column of fs into pre-allocated result.
Zero allocations beyond the spline creation.
"""
function total_integral!(result::AbstractVector{Float64}, xs::AbstractVector{Float64},
    fs::AbstractMatrix{Float64}; bc=NaturalBC())
    npts, nqty = size(fs)
    @assert length(result) >= nqty "result vector must have at least $nqty elements"

    # Use CubicSeriesInterpolant for efficient multi-quantity handling
    itp = cubic_interp(xs, fs; bc=bc)
    y = itp.y   # (npts, nqty)
    z = itp.z   # (npts, nqty)
    h = itp.cache.spacing.h  # (npts-1,)

    # Zero the result
    @inbounds for k in 1:nqty
        result[k] = 0.0
    end

    @inbounds for i in 1:(npts-1)
        hi = h[i]
        hi_half = hi / 2
        hi3_24 = hi^3 / 24
        for k in 1:nqty
            result[k] += hi_half * (y[i, k] + y[i+1, k]) - hi3_24 * (z[i, k] + z[i+1, k])
        end
    end

    return result
end
