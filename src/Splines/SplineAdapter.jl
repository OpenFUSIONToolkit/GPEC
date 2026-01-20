"""
SplineAdapter - Pure Julia spline implementations

This module provides pure Julia replacements for the Fortran-backed splines:
- CubicSpline1D: 1D cubic spline with up to 3rd derivative support
- ComplexMatrixSpline: Matrix of splines for complex-valued stability data

For MHD equilibrium and stability analysis, profile quantities (pressure, safety
factor q, flux functions) vary smoothly along the radial coordinate psi. These
splines provide efficient interpolation with analytical derivatives needed for
the Euler-Lagrange stability equations.

# Thread Safety
The `evaluate!`, `deriv1!`, `deriv2!`, and `deriv3!` functions modify internal
work arrays and are NOT thread-safe. For parallel evaluation, create separate
spline instances per thread or use the non-mutating `evaluate` function.

# References
- de Boor, "A Practical Guide to Splines", Springer (2001), Chapter 4
- Press et al., "Numerical Recipes", Chapter 3.3
"""

using LinearAlgebra

# =============================================================================
# Boundary Condition Types (for type stability)
# =============================================================================

"""
Abstract type for spline boundary conditions.
"""
abstract type SplineBoundaryCondition end

"""
Natural boundary condition: f''(x) = 0 at endpoints.
"""
struct NaturalSplineBC <: SplineBoundaryCondition end

"""
Periodic boundary condition: f(x_1) = f(x_n), f'(x_1) = f'(x_n).
"""
struct PeriodicSplineBC <: SplineBoundaryCondition end

"""
Extrapolated boundary condition: f'(x) is estimated from extrapolation of 4 nearby points.

This matches the Fortran GPEC "extrap" mode (bctype=3) which uses polynomial
extrapolation to estimate the first derivative at each endpoint.
"""
struct ExtrapSplineBC <: SplineBoundaryCondition end

"""
Map boundary condition specification to concrete type.

Accepts string names: "natural", "periodic", "extrap".
"""
function _get_boundary_condition(bctype::String)::SplineBoundaryCondition
    if bctype == "natural"
        return NaturalSplineBC()
    elseif bctype == "periodic"
        return PeriodicSplineBC()
    elseif bctype == "extrap"
        return ExtrapSplineBC()
    else
        error(
            "Unknown boundary condition type: $bctype. " *
            "Valid options: \"natural\", \"periodic\", \"extrap\""
        )
    end
end

"""
Map boundary condition to extrapolation behavior symbol.
"""
function _get_extrapolation_mode(bctype::String)::Symbol
    if bctype == "natural"
        return :constant
    elseif bctype == "periodic"
        return :wrap
    elseif bctype == "extrap"
        return :extension
    else
        return :extension
    end
end

# =============================================================================
# CubicSpline1D
# =============================================================================

"""
    CubicSpline1D{T, BC}

A 1D cubic spline interpolator with support for multiple quantities and
derivatives up to third order.

# Type Parameters

  - `T`: Element type (Float64 or ComplexF64)
  - `BC`: Boundary condition type (NaturalSplineBC or PeriodicSplineBC)

# Fields

  - `xs::Vector{Float64}`: Grid points (knots). Typically normalized flux psi_N in [0,1].
  - `fs::Matrix{T}`: Function values at grid points, shape (npts, nqty)
  - `fs1::Matrix{T}`: First derivatives at grid points (computed on construction)
  - `fsi::Matrix{T}`: Cumulative integrals at grid points (computed on demand via `integrate!`)
  - `bc::BC`: Boundary condition type
  - `extrap::Symbol`: Extrapolation mode (:constant, :extension, or :wrap)
  - `coeffs::Array{T,3}`: Polynomial coefficients (nintervals, 4, nqty) for [a, b, c, d]

# Polynomial Form

On interval [x_i, x_{i+1}], the spline is:
S_i(t) = a + b*t + c*t² + d*t³
where t = x - x_i is the local coordinate.

# Thread Safety

The mutating evaluation functions (`evaluate!`, `deriv1!`, etc.) use internal
work arrays and are NOT thread-safe. Use separate instances for parallel code.
"""
mutable struct CubicSpline1D{T<:Union{Float64,ComplexF64},BC<:SplineBoundaryCondition}
    xs::Vector{Float64}
    fs::Matrix{T}
    fs1::Matrix{T}
    fsi::Matrix{T}
    bc::BC
    extrap::Symbol
    coeffs::Array{T,3}  # (nintervals, 4, nqty) for polynomial coeffs [a, b, c, d]
    _f::Vector{T}       # Work array for evaluation
    _f1::Vector{T}
    _f2::Vector{T}
    _f3::Vector{T}
    _last_ix::Base.RefValue{Int}  # Cached interval index for sequential search
end

"""
    CubicSpline1D(xs, fs; bctype="extrap")

Create a 1D cubic spline interpolator.

# Arguments

  - `xs::Vector{Float64}`: Grid points (must be sorted in ascending order).
    Typically normalized flux psi_N in [0, 1] for radial profiles.

  - `fs::Union{Vector{T}, Matrix{T}}`: Function values at grid points.
    If Vector, treated as single quantity. If Matrix, shape is (npts, nqty).
  - `bctype`: Boundary condition type. Options:

      + `"natural"`: f''(x) = 0 at endpoints
      + `"periodic"`: f and f' continuous across domain wrap
      + `"extrap"`: Endpoint derivatives estimated from 4-point polynomial extrapolation

# Notes

  - For periodic data, ensure fs[1, :] ≈ fs[end, :] for physical consistency.

# Example

```julia
psi = collect(range(0.0, 1.0; length=101))
q_profile = 1.0 .+ 2.0 .* psi .^ 2  # Safety factor profile
q_spline = CubicSpline1D(psi, q_profile; bctype="extrap")

# Evaluate at arbitrary point
q_val = evaluate!(q_spline, 0.5)

# Get derivatives (each function returns only the requested derivative)
dq_dpsi = deriv1!(q_spline, 0.5)
d2q_dpsi2 = deriv2!(q_spline, 0.5)
d3q_dpsi3 = deriv3!(q_spline, 0.5)
```
"""
function CubicSpline1D(xs::Vector{Float64}, fs::Union{Vector{T},Matrix{T}};
    bctype::String="extrap") where {T<:Union{Float64,ComplexF64}}
    # Ensure fs is a matrix (npts × nqty)
    fs_mat = fs isa Vector ? reshape(fs, :, 1) : fs
    npts, nqty = size(fs_mat)
    nintervals = npts - 1

    @assert length(xs) == npts "xs length ($(length(xs))) must match fs rows ($npts)"
    @assert npts >= 2 "Need at least 2 points for spline interpolation"
    @assert issorted(xs) "xs must be sorted in ascending order"

    bc = _get_boundary_condition(bctype)
    extrap = _get_extrapolation_mode(bctype)

    # Compute spline coefficients for each quantity
    coeffs = zeros(T, nintervals, 4, nqty)
    fs1 = zeros(T, npts, nqty)

    for q in 1:nqty
        # Compute second derivative coefficients using tridiagonal solve
        c = _compute_spline_coeffs(xs, fs_mat[:, q], bc)

        # Store polynomial coefficients: f(t) = a + b*t + c*t² + d*t³ where t = x - x_i
        @inbounds for i in 1:nintervals
            h = xs[i+1] - xs[i]
            coeffs[i, 1, q] = fs_mat[i, q]                      # a = f_i
            coeffs[i, 3, q] = c[i]                               # c coefficient
            coeffs[i, 4, q] = (c[i+1] - c[i]) / (3 * h)         # d = (c_{i+1} - c_i) / (3h)
            coeffs[i, 2, q] = (fs_mat[i+1, q] - fs_mat[i, q]) / h - h * (2 * c[i] + c[i+1]) / 3  # b
        end

        # Compute first derivatives at grid points
        @inbounds for i in 1:npts
            if i <= nintervals
                fs1[i, q] = coeffs[i, 2, q]
            else
                # Last point: use derivative from last interval at t = h
                h = xs[nintervals+1] - xs[nintervals]
                fs1[i, q] = coeffs[nintervals, 2, q] + 2 * coeffs[nintervals, 3, q] * h +
                            3 * coeffs[nintervals, 4, q] * h^2
            end
        end
    end

    # Initialize integration array (computed on demand)
    fsi = zeros(T, npts, nqty)

    # Work arrays for thread-unsafe evaluation
    _f = zeros(T, nqty)
    _f1 = zeros(T, nqty)
    _f2 = zeros(T, nqty)
    _f3 = zeros(T, nqty)
    _last_ix = Ref(1)  # Cached interval index, starts at 1

    CubicSpline1D{T,typeof(bc)}(xs, copy(fs_mat), fs1, fsi, bc, extrap, coeffs, _f, _f1, _f2, _f3, _last_ix)
end

"""
    _estimate_endpoint_derivative(xs, fs, x0)

Estimate the first derivative f'(x0) using cubic Lagrange interpolation
through 4 points (xs, fs). This matches the Fortran `spline_get_yp` function.

For 4 points at x1, x2, x3, x4 with values f1, f2, f3, f4, the derivative
of the Lagrange interpolating polynomial at x0 is computed analytically.
"""
function _estimate_endpoint_derivative(xs::AbstractVector{Float64},
    fs::AbstractVector{T}, x0::Float64) where {T}
    @assert length(xs) == 4 && length(fs) == 4

    x1, x2, x3, x4 = xs[1], xs[2], xs[3], xs[4]
    f1, f2, f3, f4 = fs[1], fs[2], fs[3], fs[4]

    # Lagrange basis polynomial derivatives at x0
    # L_i(x) = Π_{j≠i} (x - x_j) / (x_i - x_j)
    # L_i'(x) = Σ_{k≠i} [ 1/(x_i - x_k) * Π_{j≠i,k} (x - x_j) / (x_i - x_j) ]

    # For efficiency, compute denominators
    d12 = x1 - x2
    d13 = x1 - x3
    d14 = x1 - x4
    d23 = x2 - x3
    d24 = x2 - x4
    d34 = x3 - x4

    # Compute L_1'(x0)
    L1_denom = d12 * d13 * d14
    L1_deriv = ((x0 - x2) * (x0 - x3) + (x0 - x2) * (x0 - x4) + (x0 - x3) * (x0 - x4)) / L1_denom

    # Compute L_2'(x0)
    L2_denom = (-d12) * d23 * d24
    L2_deriv = ((x0 - x1) * (x0 - x3) + (x0 - x1) * (x0 - x4) + (x0 - x3) * (x0 - x4)) / L2_denom

    # Compute L_3'(x0)
    L3_denom = (-d13) * (-d23) * d34
    L3_deriv = ((x0 - x1) * (x0 - x2) + (x0 - x1) * (x0 - x4) + (x0 - x2) * (x0 - x4)) / L3_denom

    # Compute L_4'(x0)
    L4_denom = (-d14) * (-d24) * (-d34)
    L4_deriv = ((x0 - x1) * (x0 - x2) + (x0 - x1) * (x0 - x3) + (x0 - x2) * (x0 - x3)) / L4_denom

    # f'(x0) = Σ f_i * L_i'(x0)
    return f1 * L1_deriv + f2 * L2_deriv + f3 * L3_deriv + f4 * L4_deriv
end

"""
    _compute_spline_coeffs(xs, fs, bc) -> Vector{T}

Compute second derivative coefficients (c) for cubic spline interpolation
using the standard tridiagonal system approach.

The cubic spline on interval [x_i, x_{i+1}] has the form:
S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)² + d_i(x-x_i)³

This function solves the tridiagonal system arising from continuity of the
second derivative at interior knots, with boundary conditions specified by `bc`.

# Arguments

  - `xs`: Knot positions
  - `fs`: Function values at knots
  - `bc`: Boundary condition (NaturalSplineBC or PeriodicSplineBC)

# Returns

Vector of second derivative coefficients at each knot (divided by 2).

# Reference

de Boor, "A Practical Guide to Splines", Chapter 4.
"""
function _compute_spline_coeffs(xs::Vector{Float64}, fs::Vector{T},
    bc::SplineBoundaryCondition) where {T}
    n = length(xs)

    # Build tridiagonal system for second derivatives
    h = diff(xs)

    # Right-hand side
    rhs = zeros(T, n)
    @inbounds for i in 2:(n-1)
        rhs[i] = 3 * ((fs[i+1] - fs[i]) / h[i] - (fs[i] - fs[i-1]) / h[i-1])
    end

    # Tridiagonal matrix coefficients
    dl = zeros(Float64, n - 1)  # sub-diagonal
    d = zeros(Float64, n)       # diagonal
    du = zeros(Float64, n - 1)  # super-diagonal

    # Interior points
    @inbounds for i in 2:(n-1)
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
    if bc isa PeriodicSplineBC
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

    # Handle extrap BC (clamped with extrapolated derivatives)
    # Estimate f' at endpoints using cubic polynomial fit through 4 nearby points
    if bc isa ExtrapSplineBC && n >= 4
        # Estimate f'(x_1) from first 4 points using Lagrange interpolation derivative
        yp_left = _estimate_endpoint_derivative(xs[1:4], fs[1:4], xs[1])
        # Estimate f'(x_n) from last 4 points
        yp_right = _estimate_endpoint_derivative(xs[(n-3):n], fs[(n-3):n], xs[n])

        # Modify boundary equations for clamped spline with estimated derivatives
        # Left boundary: 2*h_1*m_1 + h_1*m_2 = 3*((f_2-f_1)/h_1 - yp_left)
        # Note: Using factor of 3 (not 6) to match interior equations which solve for m = z/2
        d[1] = 2 * h[1]
        du[1] = h[1]
        rhs[1] = 3 * ((fs[2] - fs[1]) / h[1] - yp_left)

        # Right boundary: h_{n-1}*m_{n-1} + 2*h_{n-1}*m_n = 3*(yp_right - (f_n-f_{n-1})/h_{n-1})
        dl[n-1] = h[n-1]
        d[n] = 2 * h[n-1]
        rhs[n] = 3 * (yp_right - (fs[n] - fs[n-1]) / h[n-1])
    end

    # Solve tridiagonal system using Thomas algorithm
    c = zeros(T, n)
    if n == 2
        c[1] = rhs[1] / d[1]
        c[2] = rhs[2] / d[2]
    else
        # Forward elimination
        upper_elim = zeros(Float64, n)   # Eliminated upper diagonal
        rhs_elim = zeros(T, n)           # Eliminated right-hand side

        upper_elim[1] = du[1] / d[1]
        rhs_elim[1] = rhs[1] / d[1]

        @inbounds for i in 2:(n-1)
            denom = d[i] - dl[i-1] * upper_elim[i-1]
            upper_elim[i] = du[i] / denom
            rhs_elim[i] = (rhs[i] - dl[i-1] * rhs_elim[i-1]) / denom
        end
        rhs_elim[n] = (rhs[n] - dl[n-1] * rhs_elim[n-1]) / (d[n] - dl[n-1] * upper_elim[n-1])

        # Back substitution
        c[n] = rhs_elim[n]
        @inbounds for i in (n-1):-1:1
            c[i] = rhs_elim[i] - upper_elim[i] * c[i+1]
        end
    end

    return c
end

"""
Find the interval index for evaluation point x.
Returns index i such that xs[i] <= x < xs[i+1].
Uses binary search for O(log n) complexity.
"""
@inline function _find_interval(xs::Vector{Float64}, x::Float64)
    n = length(xs)
    if x <= xs[1]
        return 1
    elseif x >= xs[end]
        return n - 1
    else
        # Binary search
        lo, hi = 1, n
        @inbounds while lo < hi - 1
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
Find interval using cached index from previous evaluation.
Uses sequential search from last position - O(1) for monotonic queries.
Falls back to binary search if sequential search would be slow.
"""
@inline function _find_interval_cached!(xs::Vector{Float64}, x::Float64, last_ix::Base.RefValue{Int})
    n = length(xs)
    nmax = n - 1

    # Handle boundary cases
    @inbounds if x <= xs[1]
        last_ix[] = 1
        return 1
    elseif x >= xs[end]
        last_ix[] = nmax
        return nmax
    end

    ix = last_ix[]
    # Clamp to valid range
    ix = max(1, min(ix, nmax))

    # Sequential search from cached position (like GPEC Fortran)
    @inbounds begin
        # Search backwards if needed
        while ix > 1 && x < xs[ix]
            ix -= 1
        end
        # Search forwards if needed
        while ix < nmax && x >= xs[ix+1]
            ix += 1
        end
    end

    last_ix[] = ix
    return ix
end

"""
    evaluate!(spline, x) -> Vector{T}

Evaluate the spline at point x, returning values for all quantities.
Results stored in spline._f work array (not thread-safe).
Uses cached interval search for fast sequential evaluation.
"""
function evaluate!(spline::CubicSpline1D{T,BC}, x::Float64) where {T,BC}
    i = _find_interval_cached!(spline.xs, x, spline._last_ix)
    t = x - spline.xs[i]

    nqty = size(spline.coeffs, 3)
    @inbounds for q in 1:nqty
        a = spline.coeffs[i, 1, q]
        b = spline.coeffs[i, 2, q]
        c = spline.coeffs[i, 3, q]
        d = spline.coeffs[i, 4, q]
        # Horner's method with muladd for potential FMA optimization
        spline._f[q] = muladd(t, muladd(t, muladd(t, d, c), b), a)
    end

    return spline._f
end

"""
    deriv1!(spline, x) -> Vector{T}

Evaluate first derivative at point x.
Results stored in work array (not thread-safe).
Uses cached interval search for fast sequential evaluation.
"""
function deriv1!(spline::CubicSpline1D{T,BC}, x::Float64) where {T,BC}
    i = _find_interval_cached!(spline.xs, x, spline._last_ix)
    t = x - spline.xs[i]

    nqty = size(spline.coeffs, 3)
    @inbounds for q in 1:nqty
        b = spline.coeffs[i, 2, q]
        c = spline.coeffs[i, 3, q]
        d = spline.coeffs[i, 4, q]
        spline._f1[q] = muladd(t, muladd(3 * t, d, 2 * c), b)
    end

    return spline._f1
end

"""
    deriv2!(spline, x) -> Vector{T}

Evaluate second derivative at point x.
Results stored in work array (not thread-safe).
Uses cached interval search for fast sequential evaluation.
"""
function deriv2!(spline::CubicSpline1D{T,BC}, x::Float64) where {T,BC}
    i = _find_interval_cached!(spline.xs, x, spline._last_ix)
    t = x - spline.xs[i]

    nqty = size(spline.coeffs, 3)
    @inbounds for q in 1:nqty
        c = spline.coeffs[i, 3, q]
        d = spline.coeffs[i, 4, q]
        spline._f2[q] = muladd(6 * t, d, 2 * c)
    end

    return spline._f2
end

"""
    deriv3!(spline, x) -> Vector{T}

Evaluate third derivative at point x.

For cubic splines, the third derivative is piecewise constant within each
interval: f'''(x) = 6d, where d is the cubic coefficient.

Results stored in work array (not thread-safe).
Uses cached interval search for fast sequential evaluation.
"""
function deriv3!(spline::CubicSpline1D{T,BC}, x::Float64) where {T,BC}
    i = _find_interval_cached!(spline.xs, x, spline._last_ix)

    nqty = size(spline.coeffs, 3)
    @inbounds for q in 1:nqty
        d = spline.coeffs[i, 4, q]
        spline._f3[q] = 6 * d
    end

    return spline._f3
end

"""
    integrate!(spline)

Compute cumulative integrals at grid points and store in spline.fsi.

For each interval [x_i, x_{i+1}] with width h, integrates analytically:
∫₀ʰ (a + bt + ct² + dt³) dt = ah + bh²/2 + ch³/3 + dh⁴/4

Returns the fsi matrix containing cumulative integrals.
"""
function integrate!(spline::CubicSpline1D{T,BC}) where {T,BC}
    npts = length(spline.xs)
    nqty = size(spline.fs, 2)

    @inbounds for q in 1:nqty
        spline.fsi[1, q] = zero(T)
        for i in 1:(npts-1)
            h = spline.xs[i+1] - spline.xs[i]
            a = spline.coeffs[i, 1, q]
            b = spline.coeffs[i, 2, q]
            c = spline.coeffs[i, 3, q]
            d = spline.coeffs[i, 4, q]
            # Integral from 0 to h: ah + bh²/2 + ch³/3 + dh⁴/4
            integral = h * (a + h * (b / 2 + h * (c / 3 + h * d / 4)))
            spline.fsi[i+1, q] = spline.fsi[i, q] + integral
        end
    end

    return spline.fsi
end

"""
    empty_CubicSpline1D(T::Type)

Create an empty/placeholder CubicSpline1D for type stability in struct
initialization before actual data is available.
"""
function empty_CubicSpline1D(::Type{T}) where {T<:Union{Float64,ComplexF64}}
    xs = Float64[0.0, 1.0]
    fs = zeros(T, 2, 1)
    CubicSpline1D(xs, fs)
end

# Convenience methods for single-point evaluation returning scalar
(spline::CubicSpline1D)(x::Float64) = evaluate!(spline, x)

"""
    evaluate(spline, xs_eval; sorted=false) -> Matrix{T}

Vectorized evaluation at multiple points (allocating).
Returns matrix of shape (length(xs_eval), nqty).

When `sorted=true`, uses O(1) cached interval search (2x faster for monotonic data).
When `sorted=false`, uses O(log n) binary search (better for random access).
"""
function evaluate(spline::CubicSpline1D{T,BC}, xs_eval::Vector{Float64};
    sorted::Bool=false) where {T,BC}
    npts = length(xs_eval)
    nqty = size(spline.fs, 2)
    result = zeros(T, npts, nqty)
    evaluate!(result, spline, xs_eval; sorted=sorted)
    return result
end

"""
    evaluate!(result, spline, xs_eval; sorted=false) -> Matrix{T}

In-place vectorized evaluation at multiple points.
Result matrix must have shape (length(xs_eval), nqty).

When `sorted=true`, uses O(1) cached interval search (2x faster for monotonic data).
When `sorted=false`, uses O(log n) binary search (better for random access).
"""
function evaluate!(result::Matrix{T}, spline::CubicSpline1D{T,BC},
    xs_eval::Vector{Float64}; sorted::Bool=false) where {T,BC}
    npts = length(xs_eval)
    nqty = size(spline.fs, 2)

    if sorted
        # Use cached interval search - O(1) per point for monotonic data
        last_ix = Ref(1)
        @inbounds for j in 1:npts
            x = xs_eval[j]
            i = _find_interval_cached!(spline.xs, x, last_ix)
            t = x - spline.xs[i]

            for q in 1:nqty
                a = spline.coeffs[i, 1, q]
                b = spline.coeffs[i, 2, q]
                c = spline.coeffs[i, 3, q]
                d = spline.coeffs[i, 4, q]
                result[j, q] = muladd(t, muladd(t, muladd(t, d, c), b), a)
            end
        end
    else
        # Use binary search - O(log n) per point, better for random access
        @inbounds for j in 1:npts
            x = xs_eval[j]
            i = _find_interval(spline.xs, x)
            t = x - spline.xs[i]

            for q in 1:nqty
                a = spline.coeffs[i, 1, q]
                b = spline.coeffs[i, 2, q]
                c = spline.coeffs[i, 3, q]
                d = spline.coeffs[i, 4, q]
                result[j, q] = muladd(t, muladd(t, muladd(t, d, c), b), a)
            end
        end
    end

    return result
end

# =============================================================================
# ComplexMatrixSpline - Matrix of splines for complex-valued stability matrices
# =============================================================================

"""
    ComplexMatrixSpline{S}

A wrapper providing matrix-shaped output from a single complex CubicSpline1D,
for complex-valued MHD stability matrix coefficients.

This structure is used for interpolating the Fourier-decomposed perturbation
matrices (A, B, C, D, E, F, G, H, K stability matrices in DCON) along the radial
direction. Uses a single CubicSpline1D{ComplexF64} with nqty = n1*n2 for efficiency,
with output reshaped to (n1, n2) matrix form.

# Type Parameters

  - `S`: The concrete CubicSpline1D{ComplexF64} type

# Fields

  - `spline::S`: Single complex spline with nqty = n1*n2 quantities
  - `n1::Int`: First dimension of output matrix
  - `n2::Int`: Second dimension of output matrix
  - `_out, _out1, _out2, _out3`: Work arrays for reshaped output

# Thread Safety

The mutating evaluation functions use internal work arrays and are NOT thread-safe.
"""
struct ComplexMatrixSpline{S<:CubicSpline1D{ComplexF64}}
    spline::S
    n1::Int
    n2::Int
    _out::Matrix{ComplexF64}
    _out1::Matrix{ComplexF64}
    _out2::Matrix{ComplexF64}
    _out3::Matrix{ComplexF64}
end

"""
    ComplexMatrixSpline(xs, data; bctype="extrap")

Create a ComplexMatrixSpline from a 3D array of complex data.

# Arguments

  - `xs::Vector{Float64}`: X-coordinates (typically normalized flux psi_N)
  - `data::Array{ComplexF64,3}`: Complex data of shape (npsi, n1, n2)
  - `bctype`: Boundary condition type (passed to underlying CubicSpline1D)
"""
function ComplexMatrixSpline(xs::Vector{Float64}, data::Array{ComplexF64,3};
    bctype::String="extrap")
    npsi, n1, n2 = size(data)
    @assert length(xs) == npsi "xs length must match first dimension of data"

    # Flatten (npsi, n1, n2) -> (npsi, n1*n2) using column-major order
    data_flat = reshape(data, npsi, n1 * n2)

    # Create single complex spline with nqty = n1*n2
    spline = CubicSpline1D(xs, data_flat; bctype=bctype)

    # Work arrays for reshaped output
    _out = zeros(ComplexF64, n1, n2)
    _out1 = zeros(ComplexF64, n1, n2)
    _out2 = zeros(ComplexF64, n1, n2)
    _out3 = zeros(ComplexF64, n1, n2)

    ComplexMatrixSpline{typeof(spline)}(spline, n1, n2, _out, _out1, _out2, _out3)
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

    # Reshape to 3D then use main constructor
    data = reshape(data_flat, npsi, n1, n2)
    ComplexMatrixSpline(xs, data; bctype=bctype)
end

# Helper to reshape flat spline output to matrix
@inline function _reshape_to_matrix!(out::Matrix{ComplexF64}, flat::Vector{ComplexF64}, n1::Int)
    @inbounds for j in axes(out, 2), i in axes(out, 1)
        out[i, j] = flat[(j-1)*n1+i]
    end
end

"""
    evaluate!(cms, x) -> Matrix{ComplexF64}

Evaluate the ComplexMatrixSpline at point x.
Returns matrix of complex values (stored in work array, not thread-safe).
"""
function evaluate!(cms::ComplexMatrixSpline, x::Float64)
    f = evaluate!(cms.spline, x)
    _reshape_to_matrix!(cms._out, f, cms.n1)
    return cms._out
end

"""
    deriv1!(cms, x) -> f1

Evaluate ComplexMatrixSpline first derivative at point x.
"""
function deriv1!(cms::ComplexMatrixSpline, x::Float64)
    f1 = deriv1!(cms.spline, x)
    _reshape_to_matrix!(cms._out1, f1, cms.n1)
    return cms._out1
end

"""
    deriv2!(cms, x) -> f2

Evaluate ComplexMatrixSpline second derivative at point x.
"""
function deriv2!(cms::ComplexMatrixSpline, x::Float64)
    f2 = deriv2!(cms.spline, x)
    _reshape_to_matrix!(cms._out2, f2, cms.n1)
    return cms._out2
end

"""
    deriv3!(cms, x) -> f3

Evaluate ComplexMatrixSpline third derivative at point x.
"""
function deriv3!(cms::ComplexMatrixSpline, x::Float64)
    f3 = deriv3!(cms.spline, x)
    _reshape_to_matrix!(cms._out3, f3, cms.n1)
    return cms._out3
end

"""
    empty_ComplexMatrixSpline(n1, n2)

Create an empty/placeholder ComplexMatrixSpline for type stability.
"""
function empty_ComplexMatrixSpline(n1::Int=1, n2::Int=1)
    xs = Float64[0.0, 1.0, 2.0, 3.0, 4.0]  # Need at least 4 points for extrap BC
    data = zeros(ComplexF64, 5, n1, n2)
    ComplexMatrixSpline(xs, data)
end

# =============================================================================
# Compatibility Aliases (matching legacy Fortran API)
# =============================================================================

# These aliases allow consumer code to use the familiar Fortran-style function names
# while using the new pure Julia implementations.

"""
    spline_eval!(spline, x)

Legacy API alias for `evaluate!(spline, x)`.
"""
spline_eval!(spline::CubicSpline1D, x::Float64) = evaluate!(spline, x)

"""
    spline_deriv1!(spline, x)

Legacy API alias for `deriv1!(spline, x)`.
"""
spline_deriv1!(spline::CubicSpline1D, x::Float64) = deriv1!(spline, x)

"""
    spline_deriv2!(spline, x)

Legacy API alias for `deriv2!(spline, x)`.
"""
spline_deriv2!(spline::CubicSpline1D, x::Float64) = deriv2!(spline, x)

"""
    spline_deriv3!(spline, x)

Legacy API alias for `deriv3!(spline, x)`.
"""
spline_deriv3!(spline::CubicSpline1D, x::Float64) = deriv3!(spline, x)

"""
    spline_integrate!(spline)

Legacy API alias for `integrate!(spline)`.
"""
spline_integrate!(spline::CubicSpline1D) = integrate!(spline)

"""
    spline_eval!(out, spline, x)

Legacy in-place API: evaluate spline at x and write to preallocated output vector.
Used in hot loops where output array is reused.
"""
function spline_eval!(out::AbstractVector, spline::CubicSpline1D, x::Float64)
    result = evaluate!(spline, x)
    @inbounds for i in eachindex(out)
        out[i] = result[i]
    end
    return out
end
