"""
FastInterpolationsAdaptor - Helpers and types for FastInterpolations.jl in JPEC

This module provides:
- extrap_bc: Helper to create BCPair for 4-point endpoint derivative extrapolation
- extrap_bc_matrix: Per-column BCPairs for matrix of y-series
- cumulative_integral: Trapezoidal integration helper
- MultiQuantityProfile: Multi-quantity 1D profile using native CubicInterpolant

## Boundary Conditions

Use FastInterpolations' native BC types directly:
- `NaturalBC()`: f''(x) = 0 at endpoints
- `PeriodicBC()`: Periodic (requires fs[end] == fs[1])
- `extrap_bc(xs, fs)`: GPEC-style 4-point polynomial extrapolation at endpoints

## Performance Notes

For monotonic access patterns (typical in ODE integration), use `hint=Ref(1)` with
`search=LinearBinary()` for optimal performance (~4 ns/pt).
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
spline = cubic_interp(xs, fs; bc=extrap_bc(xs, fs))
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
# Cumulative Integration Helper
# =============================================================================

"""
    cumulative_integral(xs, fs) -> Vector

Compute cumulative integral of fs over xs using trapezoidal rule.
Returns vector where result[1] = 0 and result[i+1] = result[i] + ∫_{xs[i]}^{xs[i+1]} f dx.

For matrix fs (npts × nqty), returns matrix of same shape with cumulative integrals per column.
"""
function cumulative_integral(xs::AbstractVector{Float64}, fs::AbstractVector{T}) where {T}
    npts = length(xs)
    fsi = zeros(T, npts)
    @inbounds for i in 1:(npts-1)
        h = xs[i+1] - xs[i]
        fsi[i+1] = fsi[i] + h * (fs[i] + fs[i+1]) / 2
    end
    return fsi
end

function cumulative_integral(xs::AbstractVector{Float64}, fs::AbstractMatrix{T}) where {T}
    npts, nqty = size(fs)
    fsi = zeros(T, npts, nqty)
    @inbounds for k in 1:nqty
        for i in 1:(npts-1)
            h = xs[i+1] - xs[i]
            fsi[i+1, k] = fsi[i, k] + h * (fs[i, k] + fs[i+1, k]) / 2
        end
    end
    return fsi
end

# =============================================================================
# MultiQuantityProfile - Native Multi-Quantity Spline
# =============================================================================

"""
    MultiQuantityProfile{N}

A multi-quantity 1D profile using native FastInterpolations CubicInterpolant.
Provides direct access to grid (xs), values (fs), and interpolation/derivative evaluation.

## Fields

  - `xs::Vector{Float64}`: Grid points (same as interps[1].cache.x)
  - `fs::Matrix{Float64}`: Function values at grid points (npts × nqty)
  - `interps::NTuple{N,CubicInterpolant}`: Native interpolants for each quantity
  - `derivs::NTuple{N,DerivativeView}`: First derivative views
  - `fsi::Matrix{Float64}`: Integrated values (computed lazily via integrate!)
  - `_eval_buf::Vector{Float64}`: Preallocated buffer for evaluation
  - `_deriv_buf::Vector{Float64}`: Preallocated buffer for derivative evaluation
"""
mutable struct MultiQuantityProfile{N,I<:FastInterpolations.CubicInterpolant,D<:FastInterpolations.DerivativeView}
    xs::Vector{Float64}
    fs::Matrix{Float64}
    interps::NTuple{N,I}
    derivs::NTuple{N,D}
    fsi::Matrix{Float64}
    _eval_buf::Vector{Float64}
    _deriv_buf::Vector{Float64}
end

"""
    MultiQuantityProfile(xs, fs; bc=:extrap, extrap=:extension)

Create a MultiQuantityProfile from grid points and values matrix.

# Arguments

  - `xs::Vector{Float64}`: Grid points (sorted ascending, at least 4 points)
  - `fs::Matrix{Float64}`: Function values (npts × nqty)
  - `bc`: Boundary condition - `:extrap` for GPEC-style extrapolation, or a BCPair/NaturalBC/PeriodicBC
  - `extrap`: Extrapolation behavior (`:none`, `:constant`, `:extension`, `:wrap`)
"""
function MultiQuantityProfile(xs::Vector{Float64}, fs::Matrix{Float64};
    bc=:extrap, extrap::Symbol=:extension)

    npts, nqty = size(fs)
    @assert length(xs) == npts "xs and fs rows must match"
    @assert npts >= 4 "Need at least 4 points for spline"

    # Create interpolants with appropriate BCs
    if bc === :extrap
        interps = ntuple(k -> cubic_interp(xs, fs[:, k]; bc=extrap_bc(xs, fs[:, k]), extrap=extrap), nqty)
    else
        interps = ntuple(k -> cubic_interp(xs, fs[:, k]; bc=bc, extrap=extrap), nqty)
    end

    # Create derivative views
    derivs = ntuple(k -> FastInterpolations.deriv1(interps[k]), nqty)

    # Initialize fsi as zeros (computed lazily)
    fsi = zeros(Float64, npts, nqty)

    # Preallocate buffers
    eval_buf = zeros(Float64, nqty)
    deriv_buf = zeros(Float64, nqty)

    MultiQuantityProfile{nqty,eltype(interps),eltype(derivs)}(
        xs, copy(fs), interps, derivs, fsi, eval_buf, deriv_buf
    )
end

# Evaluation: returns vector of all quantities at x
@inline function (mqp::MultiQuantityProfile{N})(x::Float64; search=nothing, hint=nothing) where {N}
    @inbounds for k in 1:N
        if search === nothing && hint === nothing
            mqp._eval_buf[k] = mqp.interps[k](x)
        elseif hint === nothing
            mqp._eval_buf[k] = mqp.interps[k](x; search=search)
        elseif search === nothing
            mqp._eval_buf[k] = mqp.interps[k](x; hint=hint)
        else
            mqp._eval_buf[k] = mqp.interps[k](x; search=search, hint=hint)
        end
    end
    return mqp._eval_buf
end

@inline evaluate!(mqp::MultiQuantityProfile, x::Float64; search=nothing, hint=nothing) = mqp(x; search=search, hint=hint)

# First derivative evaluation
@inline function deriv1!(mqp::MultiQuantityProfile{N}, x::Float64) where {N}
    @inbounds for k in 1:N
        mqp._deriv_buf[k] = mqp.derivs[k](x)
    end
    return mqp._deriv_buf
end

# Integration (trapezoidal, in-place)
function integrate!(mqp::MultiQuantityProfile{N}) where {N}
    npts = length(mqp.xs)
    @inbounds for k in 1:N
        mqp.fsi[1, k] = 0.0
        for i in 1:(npts-1)
            h = mqp.xs[i+1] - mqp.xs[i]
            mqp.fsi[i+1, k] = mqp.fsi[i, k] + h * (mqp.fs[i, k] + mqp.fs[i+1, k]) / 2
        end
    end
    return mqp.fsi
end

function empty_MultiQuantityProfile(nqty::Int=4)
    xs = collect(range(0.0, 1.0; length=5))
    fs = zeros(Float64, 5, nqty)
    MultiQuantityProfile(xs, fs)
end
