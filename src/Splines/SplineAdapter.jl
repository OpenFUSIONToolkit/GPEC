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
Not-a-knot boundary condition: f'''(x) is continuous at x_2 and x_{n-1}.

This forces the spline to behave as if the first two knots (and last two knots)
belong to a single cubic polynomial.
"""
struct NotAKnotSplineBC <: SplineBoundaryCondition end

"""
Extrapolated boundary condition: f'(x) is estimated from extrapolation of 4 nearby points.

This matches the Fortran GPEC "extrap" mode (bctype=3) which uses polynomial
extrapolation to estimate the first derivative at each endpoint.
"""
struct ExtrapSplineBC <: SplineBoundaryCondition end

# Boundary condition type codes (matching legacy Fortran interface)
const BC_NATURAL = 1
const BC_PERIODIC = 2
const BC_EXTRAPOLATED = 3
const BC_NOT_A_KNOT = 4

"""
Map boundary condition specification to concrete type.

Accepts string names ("natural", "periodic", "extrap", "not-a-knot") or
legacy integer codes (1-4).

Note: "not-a-knot" is approximated using natural spline conditions.
"""
function _get_boundary_condition(bctype::Union{String,Int})::SplineBoundaryCondition
    if bctype == "natural" || bctype == BC_NATURAL
        return NaturalSplineBC()
    elseif bctype == "periodic" || bctype == BC_PERIODIC
        return PeriodicSplineBC()
    elseif bctype == "extrap" || bctype == BC_EXTRAPOLATED
        return ExtrapSplineBC()
    elseif bctype == "not-a-knot" || bctype == BC_NOT_A_KNOT
        return NotAKnotSplineBC()
    else
        error(
            "Unknown boundary condition type: $bctype. " *
            "Valid options: \"natural\", \"periodic\", \"extrap\", \"not-a-knot\" " *
            "or integers 1-4 (BC_NATURAL, BC_PERIODIC, BC_EXTRAPOLATED, BC_NOT_A_KNOT)"
        )
    end
end

"""
Map boundary condition to extrapolation behavior symbol.
"""
function _get_extrapolation_mode(bctype::Union{String,Int})::Symbol
    if bctype == "natural" || bctype == BC_NATURAL
        return :constant
    elseif bctype == "periodic" || bctype == BC_PERIODIC
        return :wrap
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

      + `"natural"`: f''(x) = 0 at endpoints (default for most physics applications)
      + `"periodic"`: f and f' continuous across domain wrap
      + `"extrap"`: Natural BC with linear extrapolation outside domain
      + `"not-a-knot"`: Approximated as natural (see note below)
      + Integer codes 1-4 for legacy compatibility

# Notes

  - "not-a-knot" boundary conditions are approximated using natural spline
    conditions. This may cause small differences near domain boundaries compared
    to true not-a-knot implementations.
  - For periodic data, ensure fs[1, :] ≈ fs[end, :] for physical consistency.

# Example

```julia
psi = collect(range(0.0, 1.0; length=101))
q_profile = 1.0 .+ 2.0 .* psi .^ 2  # Safety factor profile
q_spline = CubicSpline1D(psi, q_profile; bctype="extrap")

# Evaluate at arbitrary point
q_val = evaluate!(q_spline, 0.5)

# Get value and derivatives
q, dq_dpsi, d2q_dpsi2, d3q_dpsi3 = deriv3!(q_spline, 0.5)
```
"""
function CubicSpline1D(xs::Vector{Float64}, fs::Union{Vector{T},Matrix{T}};
    bctype::Union{String,Int}="extrap") where {T<:Union{Float64,ComplexF64}}
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

    CubicSpline1D{T,typeof(bc)}(xs, copy(fs_mat), fs1, fsi, bc, extrap, coeffs, _f, _f1, _f2, _f3)
end

"""
    _solve_not_a_knot_spline(xs, fs, h, rhs)

Solve the spline coefficient system with not-a-knot boundary conditions.

Not-a-knot requires third derivative continuity at the second and second-to-last
knots, which means d_1 = d_2 and d_{n-2} = d_{n-1}. This creates constraints
that are non-tridiagonal, requiring a more general solver.

# Implementation

Uses row elimination to reduce the pentadiagonal system to tridiagonal:

  - Left boundary: eliminate c_1 using the not-a-knot constraint
  - Right boundary: eliminate c_n using the not-a-knot constraint
  - Solve the reduced (n-2)×(n-2) tridiagonal system for c_2, ..., c_{n-1}
  - Back-substitute to find c_1 and c_n
"""
function _solve_not_a_knot_spline(xs::Vector{Float64}, fs::Vector{T},
    h::Vector{Float64}, rhs::Vector{T}) where {T}
    n = length(xs)

    # Not-a-knot constraints:
    # Left:  (c_2 - c_1)/h_1 = (c_3 - c_2)/h_2  =>  c_1 = c_2 + (h_1/h_2)*(c_2 - c_3)
    # Right: (c_{n-1} - c_{n-2})/h_{n-2} = (c_n - c_{n-1})/h_{n-1}
    #        =>  c_n = c_{n-1} + (h_{n-1}/h_{n-2})*(c_{n-1} - c_{n-2})

    # For a simpler implementation, we use an equivalent formulation:
    # Set up the interior equations for i = 2, ..., n-1 and substitute the
    # not-a-knot constraints for c_1 and c_n.

    # Interior system (n-2 equations for c_2, ..., c_{n-1})
    m = n - 2  # Number of interior unknowns
    if m < 2
        # Not enough points for not-a-knot; fall back to something reasonable
        return zeros(T, n)
    end

    # Build modified tridiagonal system
    dl = zeros(Float64, m - 1)  # sub-diagonal
    d = zeros(Float64, m)       # diagonal
    du = zeros(Float64, m - 1)  # super-diagonal
    b = zeros(T, m)             # RHS

    # Row 1 (i=2 in original): h_1*c_1 + 2*(h_1+h_2)*c_2 + h_2*c_3 = rhs_2
    # Substitute c_1 = c_2*(1 + h_1/h_2) - c_3*(h_1/h_2) from not-a-knot:
    # h_1*(c_2*(1 + h_1/h_2) - c_3*(h_1/h_2)) + 2*(h_1+h_2)*c_2 + h_2*c_3 = rhs_2
    # c_2*(h_1*(1 + h_1/h_2) + 2*(h_1+h_2)) + c_3*(h_2 - h_1^2/h_2) = rhs_2
    h1, h2 = h[1], h[2]
    d[1] = h1 * (1 + h1 / h2) + 2 * (h1 + h2)
    if m > 1
        du[1] = h2 - h1^2 / h2
    end
    b[1] = rhs[2]

    # Interior rows (i = 3 to n-2 in original, indices 2 to m-1 in reduced system)
    @inbounds for k in 2:(m-1)
        i = k + 1  # Original index
        dl[k-1] = h[i-1]
        d[k] = 2 * (h[i-1] + h[i])
        du[k] = h[i]
        b[k] = rhs[i]
    end

    # Last row (i = n-1 in original, index m in reduced system)
    # h_{n-2}*c_{n-2} + 2*(h_{n-2}+h_{n-1})*c_{n-1} + h_{n-1}*c_n = rhs_{n-1}
    # Substitute c_n = c_{n-1}*(1 + h_{n-1}/h_{n-2}) - c_{n-2}*(h_{n-1}/h_{n-2})
    hn2, hn1 = h[n-2], h[n-1]
    if m > 1
        dl[m-1] = hn2 - hn1^2 / hn2
    end
    d[m] = hn1 * (1 + hn1 / hn2) + 2 * (hn2 + hn1)
    b[m] = rhs[n-1]

    # Solve tridiagonal system using Thomas algorithm
    c_interior = zeros(T, m)
    if m == 1
        c_interior[1] = b[1] / d[1]
    else
        # Forward elimination
        upper_elim = zeros(Float64, m)
        rhs_elim = zeros(T, m)

        upper_elim[1] = du[1] / d[1]
        rhs_elim[1] = b[1] / d[1]

        @inbounds for i in 2:(m-1)
            denom = d[i] - dl[i-1] * upper_elim[i-1]
            upper_elim[i] = du[i] / denom
            rhs_elim[i] = (b[i] - dl[i-1] * rhs_elim[i-1]) / denom
        end
        rhs_elim[m] = (b[m] - dl[m-1] * rhs_elim[m-1]) / (d[m] - dl[m-1] * upper_elim[m-1])

        # Back substitution
        c_interior[m] = rhs_elim[m]
        @inbounds for i in (m-1):-1:1
            c_interior[i] = rhs_elim[i] - upper_elim[i] * c_interior[i+1]
        end
    end

    # Reconstruct full c vector
    c = zeros(T, n)
    c[2:(n-1)] .= c_interior

    # Back-substitute for c_1 and c_n using not-a-knot constraints
    # c_1 = c_2 + (h_1/h_2)*(c_2 - c_3) = c_2*(1 + h_1/h_2) - c_3*(h_1/h_2)
    c[1] = c[2] * (1 + h1 / h2) - c[3] * (h1 / h2)
    # c_n = c_{n-1}*(1 + h_{n-1}/h_{n-2}) - c_{n-2}*(h_{n-1}/h_{n-2})
    c[n] = c[n-1] * (1 + hn1 / hn2) - c[n-2] * (hn1 / hn2)

    return c
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
        # Left boundary: 2*h_1*c_1 + h_1*c_2 = 6*((f_2-f_1)/h_1 - yp_left)
        d[1] = 2 * h[1]
        du[1] = h[1]
        rhs[1] = 6 * ((fs[2] - fs[1]) / h[1] - yp_left)

        # Right boundary: h_{n-1}*c_{n-1} + 2*h_{n-1}*c_n = 6*(yp_right - (f_n-f_{n-1})/h_{n-1})
        dl[n-1] = h[n-1]
        d[n] = 2 * h[n-1]
        rhs[n] = 6 * (yp_right - (fs[n] - fs[n-1]) / h[n-1])
    end

    # Handle not-a-knot BC
    # Not-a-knot requires third derivative continuity at second and second-to-last knots.
    # This creates a pentadiagonal system; use a dedicated solver.
    if bc isa NotAKnotSplineBC && n >= 4
        return _solve_not_a_knot_spline(xs, fs, h, rhs)
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
function _find_interval(xs::Vector{Float64}, x::Float64)
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
    evaluate!(spline, x) -> Vector{T}

Evaluate the spline at point x, returning values for all quantities.
Results stored in spline._f work array (not thread-safe).
"""
function evaluate!(spline::CubicSpline1D{T,BC}, x::Float64) where {T,BC}
    i = _find_interval(spline.xs, x)
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
    deriv1!(spline, x) -> (f, f1)

Evaluate spline and first derivative at point x.
Results stored in work arrays (not thread-safe).
"""
function deriv1!(spline::CubicSpline1D{T,BC}, x::Float64) where {T,BC}
    i = _find_interval(spline.xs, x)
    t = x - spline.xs[i]

    nqty = size(spline.coeffs, 3)
    @inbounds for q in 1:nqty
        a = spline.coeffs[i, 1, q]
        b = spline.coeffs[i, 2, q]
        c = spline.coeffs[i, 3, q]
        d = spline.coeffs[i, 4, q]
        spline._f[q] = muladd(t, muladd(t, muladd(t, d, c), b), a)
        spline._f1[q] = muladd(t, muladd(3 * t, d, 2 * c), b)
    end

    return spline._f, spline._f1
end

"""
    deriv2!(spline, x) -> (f, f1, f2)

Evaluate spline through second derivative at point x.
Results stored in work arrays (not thread-safe).
"""
function deriv2!(spline::CubicSpline1D{T,BC}, x::Float64) where {T,BC}
    i = _find_interval(spline.xs, x)
    t = x - spline.xs[i]

    nqty = size(spline.coeffs, 3)
    @inbounds for q in 1:nqty
        a = spline.coeffs[i, 1, q]
        b = spline.coeffs[i, 2, q]
        c = spline.coeffs[i, 3, q]
        d = spline.coeffs[i, 4, q]
        spline._f[q] = muladd(t, muladd(t, muladd(t, d, c), b), a)
        spline._f1[q] = muladd(t, muladd(3 * t, d, 2 * c), b)
        spline._f2[q] = muladd(6 * t, d, 2 * c)
    end

    return spline._f, spline._f1, spline._f2
end

"""
    deriv3!(spline, x) -> (f, f1, f2, f3)

Evaluate spline through third derivative at point x.

For cubic splines, the third derivative is piecewise constant within each
interval: f'''(x) = 6d, where d is the cubic coefficient.

Results stored in work arrays (not thread-safe).
"""
function deriv3!(spline::CubicSpline1D{T,BC}, x::Float64) where {T,BC}
    i = _find_interval(spline.xs, x)
    t = x - spline.xs[i]

    nqty = size(spline.coeffs, 3)
    @inbounds for q in 1:nqty
        a = spline.coeffs[i, 1, q]
        b = spline.coeffs[i, 2, q]
        c = spline.coeffs[i, 3, q]
        d = spline.coeffs[i, 4, q]
        spline._f[q] = muladd(t, muladd(t, muladd(t, d, c), b), a)
        spline._f1[q] = muladd(t, muladd(3 * t, d, 2 * c), b)
        spline._f2[q] = muladd(6 * t, d, 2 * c)
        spline._f3[q] = 6 * d  # Third derivative is constant within interval
    end

    return spline._f, spline._f1, spline._f2, spline._f3
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
    evaluate(spline, xs_eval) -> Matrix{T}

Vectorized evaluation at multiple points (allocating, thread-safe).
Returns matrix of shape (length(xs_eval), nqty).
"""
function evaluate(spline::CubicSpline1D{T,BC}, xs_eval::Vector{Float64}) where {T,BC}
    npts = length(xs_eval)
    nqty = size(spline.fs, 2)
    result = zeros(T, npts, nqty)

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

    return result
end

# =============================================================================
# ComplexMatrixSpline - Matrix of splines for complex-valued stability matrices
# =============================================================================

"""
    ComplexMatrixSpline{S}

A matrix where each element is a separate 1D spline in psi (normalized flux),
for complex-valued MHD stability matrix coefficients.

This structure is used for interpolating the Fourier-decomposed perturbation
matrices (A, B, C, D, E, F, G, H, K stability matrices in DCON) along the radial
direction. Each matrix element can have different radial dependence, requiring
separate spline fits. Complex values are handled by storing separate real and
imaginary splines for each matrix element.

# Type Parameters

  - `S`: The concrete CubicSpline1D type used for real/imaginary parts

# Fields

  - `xs::Vector{Float64}`: Shared x-axis (normalized flux psi_N)
  - `n1::Int`: First dimension of matrix
  - `n2::Int`: Second dimension of matrix
  - `real_splines::Matrix{S}`: Real part splines (n1 × n2)
  - `imag_splines::Matrix{S}`: Imaginary part splines (n1 × n2)
  - `_out, _out1, _out2, _out3`: Work arrays for evaluation output

# Thread Safety

The mutating evaluation functions use internal work arrays and are NOT thread-safe.
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

  - `xs::Vector{Float64}`: X-coordinates (typically normalized flux psi_N)
  - `data::Array{ComplexF64,3}`: Complex data of shape (npsi, n1, n2)
  - `bctype`: Boundary condition type (passed to underlying CubicSpline1D)
"""
function ComplexMatrixSpline(xs::Vector{Float64}, data::Array{ComplexF64,3};
    bctype::String="extrap")
    npsi, n1, n2 = size(data)
    @assert length(xs) == npsi "xs length must match first dimension of data"

    # Create template spline to get concrete type
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
Returns matrix of complex values (stored in work array, not thread-safe).
"""
function evaluate!(cms::ComplexMatrixSpline, x::Float64)
    @inbounds for i in 1:cms.n1, j in 1:cms.n2
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
    @inbounds for i in 1:cms.n1, j in 1:cms.n2
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
    @inbounds for i in 1:cms.n1, j in 1:cms.n2
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
    @inbounds for i in 1:cms.n1, j in 1:cms.n2
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
