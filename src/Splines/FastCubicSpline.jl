"""
FastCubicSpline - Lightweight wrapper around FastInterpolations.jl

This module provides a thin wrapper around FastInterpolations.jl's CubicInterpolant,
offering an API similar to SplineAdapter's CubicSpline1D but optimized for single-quantity
interpolation with FastInterpolations doing the heavy lifting.

The "extrap" boundary condition is implemented by computing endpoint derivatives
via 4-point polynomial extrapolation, then using FastInterpolations' BCPair API.

## Performance Comparison (FastInterpolations v0.2.4)

Benchmarks on 257-point grid, median times after JIT warmup, 0 allocations for all evaluations:

| Operation          | CubicSpline1D | FastCubicSpline1D | FastInterp direct | Notes                          |
|--------------------|---------------|-------------------|-------------------|--------------------------------|
| Initialization     |      5.5 μs   |       10.2 μs     |       4.8 μs      | 257-point grid, natural BC     |
| evaluate (1 pt)    |      5.5 ns   |       12.8 ns     |      12.8 ns      | Single point, binary search    |
| deriv1 (1 pt)      |      5.5 ns   |       12.9 ns     |      12.8 ns      | Single point, binary search    |
| deriv2 (1 pt)      |      5.2 ns   |       11.8 ns     |      11.8 ns      | Single point, binary search    |
| deriv3 (1 pt)      |      4.9 ns   |       10.9 ns     |      10.9 ns      | Single point, binary search    |
| Monotonic loop     |      6.7 ns/pt|        4.2 ns/pt  |       4.2 ns/pt   | hint + search=LinearBinary()   |
| Random loop        |     37.9 ns/pt|       10.5 ns/pt  |      10.7 ns/pt   | FastInterpolations wins        |

**Summary**: FastCubicSpline1D has zero runtime overhead vs direct FastInterpolations
(identical timings within noise). For monotonic access patterns (typical in ODE
integration), use `hint=Ref(1)` with `search=LinearBinary()` for optimal performance
(~4 ns/pt vs CubicSpline1D's ~7 ns/pt). FastInterpolations is ~3.6x faster for random
access. All evaluation methods are allocation-free. The wrapper only adds overhead at
initialization (creates DerivativeViews and pre-computes grid derivatives).
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
Accepts AbstractVector to allow views without allocation.
"""
function _make_bc(bctype::String, xs::AbstractVector{Float64}, fs::AbstractVector{<:Union{Float64,ComplexF64}})
    if bctype == "natural"
        return NaturalBC()
    elseif bctype == "periodic"
        return PeriodicBC()
    elseif bctype == "extrap"
        n = length(xs)
        yp_left = _estimate_endpoint_derivative_fast(@view(xs[1:4]), @view(fs[1:4]), xs[1])
        yp_right = _estimate_endpoint_derivative_fast(@view(xs[(n-3):n]), @view(fs[(n-3):n]), xs[n])
        return BCPair(Deriv1(yp_left), Deriv1(yp_right))
    else
        error("Unknown bctype: $bctype. Valid: \"natural\", \"periodic\", \"extrap\"")
    end
end

"""
    FastCubicSpline1D(xs, fs; bctype="extrap", extrap=:none)

Create a 1D cubic spline interpolator using FastInterpolations.jl.

# Arguments

  - `xs::Vector{Float64}`: Grid points (must be sorted ascending, at least 4 points)

  - `fs::Vector{T}`: Function values at grid points (single quantity)
  - `bctype`: Boundary condition type:

      + `"natural"`: f''(x) = 0 at endpoints
      + `"periodic"`: f and f' continuous across domain wrap. FastInterpolations
        requires exact endpoint matching: fs[end] must equal fs[1].
      + `"extrap"`: Endpoint derivatives from 4-point polynomial extrapolation
  - `extrap`: Extrapolation behavior outside domain (FastInterpolations parameter):

      + `:none`: Throw DomainError for queries outside domain (default)
      + `:constant`: Return boundary value for out-of-domain queries
      + `:extension`: Use polynomial extrapolation outside domain
      + `:wrap`: Wrap around (only valid with periodic BC)
"""
function FastCubicSpline1D(xs::Vector{Float64}, fs::Vector{Float64};
    bctype::String="extrap", extrap::Symbol=:none)

    npts = length(xs)
    @assert length(fs) == npts "xs and fs must have same length"
    @assert npts >= 4 "Need at least 4 points for spline"
    @assert issorted(xs) "xs must be sorted ascending"

    bc = _make_bc(bctype, xs, fs)
    itp = cubic_interp(xs, fs; bc=bc, extrap=extrap)
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
    bctype::String="extrap", extrap::Symbol=:none)

    npts = length(xs)
    @assert length(fs) == npts "xs and fs must have same length"
    @assert npts >= 4 "Need at least 4 points for spline"
    @assert issorted(xs) "xs must be sorted ascending"

    fs_real = real.(fs)
    fs_imag = imag.(fs)

    bc_real = _make_bc(bctype, xs, fs_real)
    bc_imag = _make_bc(bctype, xs, fs_imag)

    itp_real = cubic_interp(xs, fs_real; bc=bc_real, extrap=extrap)
    itp_imag = cubic_interp(xs, fs_imag; bc=bc_imag, extrap=extrap)
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

# =============================================================================
# FastCubicSpline1DMulti - Multi-Quantity Version (CubicSpline1D-compatible interface)
# =============================================================================

"""
    FastCubicSpline1DMulti{T, N, S}

A multi-quantity 1D cubic spline interpolator wrapping multiple FastCubicSpline1D instances.
Provides the same API as CubicSpline1D (returns vectors when evaluated).

# Type Parameters

  - `T`: Element type (Float64 or ComplexF64)
  - `N`: Number of quantities
  - `S`: Type of each FastCubicSpline1D

# Fields

  - `xs::Vector{Float64}`: Shared grid points
  - `fs::Matrix{T}`: Function values, shape (npts, nqty)
  - `fs1::Matrix{T}`: First derivatives at grid points
  - `fsi::Matrix{T}`: Cumulative integrals at grid points
  - `mx::Int`: Number of grid points
  - `splines::NTuple{N, S}`: Tuple of FastCubicSpline1D for each quantity
  - `_f, _f1, _f2, _f3::Vector{T}`: Work arrays for thread-unsafe evaluation
"""
struct FastCubicSpline1DMulti{T<:Union{Float64,ComplexF64},N,S<:FastCubicSpline1D{T}}
    xs::Vector{Float64}
    fs::Matrix{T}
    fs1::Matrix{T}
    fsi::Matrix{T}
    mx::Int
    splines::NTuple{N,S}
    _f::Vector{T}
    _f1::Vector{T}
    _f2::Vector{T}
    _f3::Vector{T}
end

"""
    FastCubicSpline1DMulti(xs, fs; bctype="extrap", extrap=:none)

Create a multi-quantity 1D cubic spline from a matrix of values.

# Arguments

  - `xs::Vector{Float64}`: Grid points (must be sorted ascending, at least 4 points)
  - `fs::Matrix{T}`: Function values at grid points, shape (npts, nqty)
  - `bctype`: Boundary condition type ("natural", "periodic", "extrap").
    For periodic BC, FastInterpolations requires exact endpoint matching: fs[end, :] must equal fs[1, :].
  - `extrap`: Extrapolation behavior outside domain (`:none`, `:constant`, `:extension`, `:wrap`)
"""
function FastCubicSpline1DMulti(xs::Vector{Float64}, fs::Matrix{T};
    bctype::String="extrap", extrap::Symbol=:none) where {T<:Union{Float64,ComplexF64}}

    npts, nqty = size(fs)
    @assert length(xs) == npts "xs and fs rows must match"
    @assert npts >= 4 "Need at least 4 points for spline"
    @assert issorted(xs) "xs must be sorted ascending"

    # Create individual splines for each quantity
    splines = ntuple(q -> FastCubicSpline1D(xs, fs[:, q]; bctype=bctype, extrap=extrap), nqty)

    # Gather fs1 from individual splines
    fs1 = zeros(T, npts, nqty)
    for q in 1:nqty
        fs1[:, q] = splines[q].fs1
    end

    # Initialize fsi (computed on demand)
    fsi = zeros(T, npts, nqty)

    # Work arrays
    _f = zeros(T, nqty)
    _f1 = zeros(T, nqty)
    _f2 = zeros(T, nqty)
    _f3 = zeros(T, nqty)

    FastCubicSpline1DMulti{T,nqty,eltype(splines)}(
        xs, copy(fs), fs1, fsi, npts, splines, _f, _f1, _f2, _f3
    )
end

"""
    (spline::FastCubicSpline1DMulti)(x; search=nothing, hint=nothing) -> Vector{T}

Evaluate the multi-quantity spline at point x, returning values for all quantities.

# Keyword Arguments

  - `search`: Search strategy. Use `LinearBinary()` for monotonic access patterns.
  - `hint`: Mutable `Ref{Int}` for interval caching. For optimal performance with
    monotonic access, use `search=LinearBinary()` together with `hint=Ref(1)`.
"""
@inline function (spline::FastCubicSpline1DMulti{T,N})(x::Float64; search=nothing, hint=nothing) where {T,N}
    @inbounds for q in 1:N
        spline._f[q] = spline.splines[q](x; search=search, hint=hint)
    end
    return spline._f
end

"""
    evaluate!(spline::FastCubicSpline1DMulti, x; search=nothing, hint=nothing) -> Vector{T}

Evaluate the spline at point x, returning values for all quantities.
Results stored in work array (not thread-safe).

# Keyword Arguments

  - `search`: Search strategy. Use `LinearBinary()` for monotonic access patterns.
  - `hint`: Mutable `Ref{Int}` for interval caching. For optimal performance with
    monotonic access, use `search=LinearBinary()` together with `hint=Ref(1)`.
"""
@inline evaluate!(spline::FastCubicSpline1DMulti, x::Float64; search=nothing, hint=nothing) = spline(x; search=search, hint=hint)

"""
    deriv1!(spline::FastCubicSpline1DMulti, x) -> Vector{T}

Evaluate first derivative at point x for all quantities.
"""
@inline function deriv1!(spline::FastCubicSpline1DMulti{T,N}, x::Float64) where {T,N}
    @inbounds for q in 1:N
        spline._f1[q] = deriv1(spline.splines[q], x)
    end
    return spline._f1
end

"""
    deriv2!(spline::FastCubicSpline1DMulti, x) -> Vector{T}

Evaluate second derivative at point x for all quantities.
"""
@inline function deriv2!(spline::FastCubicSpline1DMulti{T,N}, x::Float64) where {T,N}
    @inbounds for q in 1:N
        spline._f2[q] = deriv2(spline.splines[q], x)
    end
    return spline._f2
end

"""
    deriv3!(spline::FastCubicSpline1DMulti, x) -> Vector{T}

Evaluate third derivative at point x for all quantities.
"""
@inline function deriv3!(spline::FastCubicSpline1DMulti{T,N}, x::Float64) where {T,N}
    @inbounds for q in 1:N
        spline._f3[q] = deriv3(spline.splines[q], x)
    end
    return spline._f3
end

"""
    integrate!(spline::FastCubicSpline1DMulti)

Compute cumulative integrals at grid points for all quantities.
"""
function integrate!(spline::FastCubicSpline1DMulti{T,N}) where {T,N}
    npts = spline.mx
    @inbounds for q in 1:N
        # Use the individual spline's integrate!
        integrate!(spline.splines[q])
        # Copy results to shared fsi matrix
        spline.fsi[:, q] = spline.splines[q].fsi
    end
    return spline.fsi
end

"""
    empty_FastCubicSpline1DMulti(T::Type, nqty::Int)

Create an empty/placeholder FastCubicSpline1DMulti for type stability.
"""
function empty_FastCubicSpline1DMulti(::Type{T}, nqty::Int=4) where {T<:Union{Float64,ComplexF64}}
    xs = collect(range(0.0, 1.0; length=5))
    fs = zeros(T, 5, nqty)
    FastCubicSpline1DMulti(xs, fs; bctype="natural")
end

# Aliases for compatibility
spline_eval!(spline::FastCubicSpline1DMulti, x::Float64; search=nothing, hint=nothing) = evaluate!(spline, x; search=search, hint=hint)
spline_deriv1!(spline::FastCubicSpline1DMulti, x::Float64) = deriv1!(spline, x)
spline_deriv2!(spline::FastCubicSpline1DMulti, x::Float64) = deriv2!(spline, x)
spline_deriv3!(spline::FastCubicSpline1DMulti, x::Float64) = deriv3!(spline, x)

# =============================================================================
# Compatibility with CubicSpline1D API (evaluate!, deriv1!, etc.)
# These are provided for drop-in compatibility with code written for CubicSpline1D.
# The "!" suffix is historical - FastCubicSpline1D doesn't mutate state.
# =============================================================================

"""
    evaluate!(spline, x; search=nothing, hint=nothing) -> T

CubicSpline1D-compatible API. For FastCubicSpline1D, this is equivalent to `spline(x; search=search, hint=hint)`.

# Keyword Arguments

  - `search`: Search strategy. Use `LinearBinary()` for monotonic access patterns.
  - `hint`: Mutable `Ref{Int}` for interval caching. For optimal performance with
    monotonic access, use `search=LinearBinary()` together with `hint=Ref(1)`.
"""
@inline evaluate!(spline::FastCubicSpline1D, x::Float64; search=nothing, hint=nothing) = spline(x; search=search, hint=hint)

"""
    deriv1!(spline, x) -> T

CubicSpline1D-compatible API. For FastCubicSpline1D, this is equivalent to `deriv1(spline, x)`.
Note: Derivative views in FastInterpolations do not support hint arguments.
"""
@inline deriv1!(spline::FastCubicSpline1D, x::Float64) = deriv1(spline, x)

"""
    deriv2!(spline, x) -> T

CubicSpline1D-compatible API. For FastCubicSpline1D, this is equivalent to `deriv2(spline, x)`.
Note: Derivative views in FastInterpolations do not support hint arguments.
"""
@inline deriv2!(spline::FastCubicSpline1D, x::Float64) = deriv2(spline, x)

"""
    deriv3!(spline, x) -> T

CubicSpline1D-compatible API. For FastCubicSpline1D, this is equivalent to `deriv3(spline, x)`.
Note: Derivative views in FastInterpolations do not support hint arguments.
"""
@inline deriv3!(spline::FastCubicSpline1D, x::Float64) = deriv3(spline, x)

# Aliases used in some codebases - with optional search/hint support for evaluate
spline_eval!(spline::FastCubicSpline1D, x::Float64; search=nothing, hint=nothing) = evaluate!(spline, x; search=search, hint=hint)
spline_deriv1!(spline::FastCubicSpline1D, x::Float64) = deriv1!(spline, x)
spline_deriv2!(spline::FastCubicSpline1D, x::Float64) = deriv2!(spline, x)
spline_deriv3!(spline::FastCubicSpline1D, x::Float64) = deriv3!(spline, x)
