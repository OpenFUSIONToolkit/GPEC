"""
FastInterpolationsAdaptor - Wrappers around FastInterpolations.jl for JPEC

This module provides:
- FastCubicSpline1D: Single-quantity 1D cubic spline
- FastCubicSpline1DMulti: Multi-quantity 1D cubic spline
- extrap_bc: Helper to create BCPair for 4-point endpoint derivative extrapolation
- extrap_bc_matrix: Per-column BCPairs for matrix of y-series

## Boundary Conditions

Use FastInterpolations' native BC types directly:
- `NaturalBC()`: f''(x) = 0 at endpoints
- `PeriodicBC()`: Periodic (requires fs[end] == fs[1])
- `extrap_bc(xs, fs)`: GPEC-style 4-point polynomial extrapolation at endpoints

## Performance Notes

All evaluation methods are allocation-free. For monotonic access patterns (typical in
ODE integration), use `hint=Ref(1)` with `search=LinearBinary()` for optimal performance
(~4 ns/pt).

# Thread Safety
Multi-quantity splines modify internal work arrays and are NOT thread-safe.
For parallel evaluation, create separate spline instances per thread.
"""

using FastInterpolations
using LinearAlgebra

# Re-export FastInterpolations BC types for convenience
const NaturalBC = FastInterpolations.NaturalBC
const PeriodicBC = FastInterpolations.PeriodicBC
const BCPair = FastInterpolations.BCPair
const Deriv1 = FastInterpolations.Deriv1
const Deriv2 = FastInterpolations.Deriv2

# =============================================================================
# Boundary Condition Helpers
# =============================================================================

"""
    extrap_bc(xs, fs) -> BCPair

Create a BCPair for GPEC-style "extrap" boundary conditions.

Estimates endpoint derivatives using 4-point Lagrange polynomial extrapolation,
matching the Fortran GPEC behavior. This provides better accuracy than natural
(f''=0) boundary conditions for smooth profiles.

# Arguments

  - `xs`: Grid coordinates (at least 4 points)
  - `fs`: Function values at grid points

# Returns

A `BCPair(Deriv1(yp_left), Deriv1(yp_right))` specifying first derivatives at endpoints.

# Example

```julia
xs = collect(range(0, 1; length=100))
fs = sin.(xs)
spline = FastCubicSpline1D(xs, fs; bc=extrap_bc(xs, fs))
```
"""
@inline function extrap_bc(xs::AbstractVector{Float64}, fs::AbstractVector{Float64})
    n = length(xs)
    @assert n >= 4 "Need at least 4 points for extrap BC"
    yp_left = _estimate_endpoint_derivative(@view(xs[1:4]), @view(fs[1:4]), xs[1])
    yp_right = _estimate_endpoint_derivative(@view(xs[(n-3):n]), @view(fs[(n-3):n]), xs[n])
    return BCPair(Deriv1(yp_left), Deriv1(yp_right))
end

# Complex version: returns tuple of (BCPair for real, BCPair for imag)
@inline function extrap_bc(xs::AbstractVector{Float64}, fs::AbstractVector{ComplexF64})
    n = length(xs)
    @assert n >= 4 "Need at least 4 points for extrap BC"
    fs_real = real.(fs)
    fs_imag = imag.(fs)
    yp_left_r = _estimate_endpoint_derivative(@view(xs[1:4]), @view(fs_real[1:4]), xs[1])
    yp_right_r = _estimate_endpoint_derivative(@view(xs[(n-3):n]), @view(fs_real[(n-3):n]), xs[n])
    yp_left_i = _estimate_endpoint_derivative(@view(xs[1:4]), @view(fs_imag[1:4]), xs[1])
    yp_right_i = _estimate_endpoint_derivative(@view(xs[(n-3):n]), @view(fs_imag[(n-3):n]), xs[n])
    return (BCPair(Deriv1(yp_left_r), Deriv1(yp_right_r)),
        BCPair(Deriv1(yp_left_i), Deriv1(yp_right_i)))
end

"""
    _estimate_endpoint_derivative(xs, fs, x0)

Estimate f'(x0) using cubic Lagrange interpolation through 4 points.
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

"""
    extrap_bc_matrix(xs, Y::Matrix{Float64}) -> Vector{BCPair}

Create per-column BCPairs for a matrix of y-series using extrap boundary conditions.
Returns a vector of BCPairs, one per column of Y.
"""
function extrap_bc_matrix(xs::AbstractVector{Float64}, Y::Matrix{Float64})
    n_series = size(Y, 2)
    bcs = Vector{BCPair}(undef, n_series)
    @inbounds for k in 1:n_series
        bcs[k] = extrap_bc(xs, @view(Y[:, k]))
    end
    return bcs
end


# =============================================================================
# FastCubicSpline1D - Single Quantity
# =============================================================================

"""
    FastCubicSpline1D{T, I, D1, D2, D3}

A 1D cubic spline interpolator wrapping FastInterpolations.jl for a single quantity.

# Type Parameters

  - `T`: Element type (Float64 or ComplexF64)
  - `I`: Interpolant type from FastInterpolations
  - `D1`, `D2`, `D3`: Derivative view types

# Evaluation

The callable `(spline)(x)` supports keyword arguments:

  - `search`: Search strategy. Use `LinearBinary()` for monotonic access.
  - `hint`: Mutable `Ref{Int}` for O(1) sequential lookups.
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
    FastCubicSpline1D(xs, fs; bc=NaturalBC(), extrap=:none)

Create a 1D cubic spline interpolator.

# Arguments

  - `xs::Vector{Float64}`: Grid points (sorted ascending, at least 4 points)
  - `fs::Vector{T}`: Function values at grid points
  - `bc`: Boundary condition - use `NaturalBC()`, `PeriodicBC()`, or `extrap_bc(xs, fs)`
  - `extrap`: Extrapolation behavior (`:none`, `:constant`, `:extension`, `:wrap`)

# Example

```julia
xs = collect(range(0, 2π; length=100))
fs = sin.(xs)

# Natural BC (default)
spline = FastCubicSpline1D(xs, fs)

# Periodic BC
spline = FastCubicSpline1D(xs, fs; bc=PeriodicBC())

# Extrap BC (GPEC-style)
spline = FastCubicSpline1D(xs, fs; bc=extrap_bc(xs, fs))
```
"""
function FastCubicSpline1D(xs::Vector{Float64}, fs::Vector{Float64};
    bc=NaturalBC(), extrap::Symbol=:none)

    npts = length(xs)
    @assert length(fs) == npts "xs and fs must have same length"
    @assert npts >= 4 "Need at least 4 points for spline"
    @assert issorted(xs) "xs must be sorted ascending"

    itp = cubic_interp(xs, fs; bc=bc, extrap=extrap)
    d1 = FastInterpolations.deriv1(itp)
    d2 = FastInterpolations.deriv2(itp)
    d3 = FastInterpolations.deriv3(itp)

    fs1 = [d1(x) for x in xs]
    fsi = zeros(Float64, npts)

    FastCubicSpline1D{Float64,typeof(itp),typeof(d1),typeof(d2),typeof(d3)}(
        xs, copy(fs), fs1, fsi, itp, d1, d2, d3
    )
end

# Complex version - stores separate real/imag interpolants
function FastCubicSpline1D(xs::Vector{Float64}, fs::Vector{ComplexF64};
    bc=NaturalBC(), extrap::Symbol=:none)

    npts = length(xs)
    @assert length(fs) == npts "xs and fs must have same length"
    @assert npts >= 4 "Need at least 4 points for spline"
    @assert issorted(xs) "xs must be sorted ascending"

    fs_real = real.(fs)
    fs_imag = imag.(fs)

    # Handle different BC input formats
    if bc isa Tuple{BCPair,BCPair}
        # Pre-computed (bc_real, bc_imag) tuple from extrap_bc for complex data
        bc_real, bc_imag = bc
    elseif bc isa BCPair
        # Single BCPair passed - recompute separate BCs for real/imag parts
        bc_real = extrap_bc(xs, fs_real)
        bc_imag = extrap_bc(xs, fs_imag)
    else
        # NaturalBC or PeriodicBC - use same for both
        bc_real = bc
        bc_imag = bc
    end

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
# FastCubicSpline1D Evaluation Methods
# =============================================================================

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

@inline evaluate(spline::FastCubicSpline1D, x::Float64; search=nothing, hint=nothing) = spline(x; search=search, hint=hint)

@inline function deriv1(spline::FastCubicSpline1D{Float64}, x::Float64)
    return spline.d1_view(x)
end

@inline function deriv1(spline::FastCubicSpline1D{ComplexF64}, x::Float64)
    d1_r, d1_i = spline.d1_view
    return d1_r(x) + 1im * d1_i(x)
end

@inline function deriv2(spline::FastCubicSpline1D{Float64}, x::Float64)
    return spline.d2_view(x)
end

@inline function deriv2(spline::FastCubicSpline1D{ComplexF64}, x::Float64)
    d2_r, d2_i = spline.d2_view
    return d2_r(x) + 1im * d2_i(x)
end

@inline function deriv3(spline::FastCubicSpline1D{Float64}, x::Float64)
    return spline.d3_view(x)
end

@inline function deriv3(spline::FastCubicSpline1D{ComplexF64}, x::Float64)
    d3_r, d3_i = spline.d3_view
    return d3_r(x) + 1im * d3_i(x)
end

function integrate!(spline::FastCubicSpline1D{T}) where {T}
    npts = length(spline.xs)
    spline.fsi[1] = zero(T)
    @inbounds for i in 1:(npts-1)
        h = spline.xs[i+1] - spline.xs[i]
        spline.fsi[i+1] = spline.fsi[i] + h * (spline.fs[i] + spline.fs[i+1]) / 2
    end
    return spline.fsi
end

function empty_FastCubicSpline1D(::Type{T}) where {T<:Union{Float64,ComplexF64}}
    xs = collect(range(0.0, 1.0; length=5))
    fs = zeros(T, 5)
    FastCubicSpline1D(xs, fs)
end

# Compatibility aliases
@inline evaluate!(spline::FastCubicSpline1D, x::Float64; search=nothing, hint=nothing) = spline(x; search=search, hint=hint)
@inline deriv1!(spline::FastCubicSpline1D, x::Float64) = deriv1(spline, x)
@inline deriv2!(spline::FastCubicSpline1D, x::Float64) = deriv2(spline, x)
@inline deriv3!(spline::FastCubicSpline1D, x::Float64) = deriv3(spline, x)


# =============================================================================
# FastCubicSpline1DMulti - Multi-Quantity Version
# =============================================================================

"""
    FastCubicSpline1DMulti{T, N, S}

A multi-quantity 1D cubic spline wrapping multiple FastCubicSpline1D instances.
Returns vectors when evaluated.
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
    FastCubicSpline1DMulti(xs, fs; bc=NaturalBC(), extrap=:none)

Create a multi-quantity 1D cubic spline.

# Arguments

  - `xs::Vector{Float64}`: Grid points (sorted ascending, at least 4 points)
  - `fs::Matrix{T}`: Function values (npts × nqty)
  - `bc`: Boundary condition - use `NaturalBC()`, `PeriodicBC()`, or `:extrap`
    For `:extrap`, the BC is computed per-quantity from the data.
  - `extrap`: Extrapolation behavior (`:none`, `:constant`, `:extension`, `:wrap`)
"""
function FastCubicSpline1DMulti(xs::Vector{Float64}, fs::Matrix{T};
    bc=NaturalBC(), extrap::Symbol=:none) where {T<:Union{Float64,ComplexF64}}

    npts, nqty = size(fs)
    @assert length(xs) == npts "xs and fs rows must match"
    @assert npts >= 4 "Need at least 4 points for spline"
    @assert issorted(xs) "xs must be sorted ascending"

    # Create individual splines - compute BC per quantity if :extrap
    if bc === :extrap
        if T === ComplexF64
            # For complex data, extrap_bc returns (BCPair_real, BCPair_imag) tuple
            # Pass tuple to FastCubicSpline1D which handles splitting internally
            splines = ntuple(nqty) do q
                fs_q = fs[:, q]
                bc_tuple = extrap_bc(xs, fs_q)  # Returns (bc_real, bc_imag)
                FastCubicSpline1D(xs, fs_q; bc=bc_tuple, extrap=extrap)
            end
        else
            splines = ntuple(q -> FastCubicSpline1D(xs, fs[:, q]; bc=extrap_bc(xs, fs[:, q]), extrap=extrap), nqty)
        end
    else
        splines = ntuple(q -> FastCubicSpline1D(xs, fs[:, q]; bc=bc, extrap=extrap), nqty)
    end

    fs1 = zeros(T, npts, nqty)
    for q in 1:nqty
        fs1[:, q] = splines[q].fs1
    end

    fsi = zeros(T, npts, nqty)
    _f = zeros(T, nqty)
    _f1 = zeros(T, nqty)
    _f2 = zeros(T, nqty)
    _f3 = zeros(T, nqty)

    FastCubicSpline1DMulti{T,nqty,eltype(splines)}(
        xs, copy(fs), fs1, fsi, npts, splines, _f, _f1, _f2, _f3
    )
end

@inline function (spline::FastCubicSpline1DMulti{T,N})(x::Float64; search=nothing, hint=nothing) where {T,N}
    @inbounds for q in 1:N
        spline._f[q] = spline.splines[q](x; search=search, hint=hint)
    end
    return spline._f
end

@inline evaluate!(spline::FastCubicSpline1DMulti, x::Float64; search=nothing, hint=nothing) = spline(x; search=search, hint=hint)

@inline function deriv1!(spline::FastCubicSpline1DMulti{T,N}, x::Float64) where {T,N}
    @inbounds for q in 1:N
        spline._f1[q] = deriv1(spline.splines[q], x)
    end
    return spline._f1
end

@inline function deriv2!(spline::FastCubicSpline1DMulti{T,N}, x::Float64) where {T,N}
    @inbounds for q in 1:N
        spline._f2[q] = deriv2(spline.splines[q], x)
    end
    return spline._f2
end

@inline function deriv3!(spline::FastCubicSpline1DMulti{T,N}, x::Float64) where {T,N}
    @inbounds for q in 1:N
        spline._f3[q] = deriv3(spline.splines[q], x)
    end
    return spline._f3
end

function integrate!(spline::FastCubicSpline1DMulti{T,N}) where {T,N}
    @inbounds for q in 1:N
        integrate!(spline.splines[q])
        spline.fsi[:, q] = spline.splines[q].fsi
    end
    return spline.fsi
end

function empty_FastCubicSpline1DMulti(::Type{T}, nqty::Int=4) where {T<:Union{Float64,ComplexF64}}
    xs = collect(range(0.0, 1.0; length=5))
    fs = zeros(T, 5, nqty)
    FastCubicSpline1DMulti(xs, fs)
end
