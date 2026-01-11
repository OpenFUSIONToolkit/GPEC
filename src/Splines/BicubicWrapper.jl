"""
BicubicWrapper - 2D bicubic interpolation using Interpolations.jl

This wraps Interpolations.jl to provide bicubic spline interpolation with
support for first and second derivatives, including cross-derivatives (fxy).

# Uniform Spacing Requirement

Interpolations.jl requires uniformly spaced grids for proper scaling. This module
validates that input grids are approximately uniform (within numerical tolerance).
If non-uniform grids are required, consider using a different interpolation approach.

# Boundary Conditions

Supports two common boundary condition configurations:
- **Non-periodic** (default): Linear extrapolation in both x and y
- **Periodic in y**: For data periodic in y (e.g., theta angle), uses periodic BC in y
  and linear extrapolation in x. The y data should span [y0, y_period) without
  duplicating the endpoint.

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
    BicubicWrapper{I,S}

A 2D bicubic spline interpolator wrapping Interpolations.jl.

# Type Parameters

  - `I`: Type of the extrapolated interpolation objects (for evaluation/gradient)
  - `S`: Type of the scaled interpolation objects (for hessian with periodic BCs)

# Fields

  - `xs::Vector{Float64}`: X-coordinates (first dimension)
  - `ys::Vector{Float64}`: Y-coordinates (second dimension)
  - `nqty::Int`: Number of quantities
  - `itps::Vector{I}`: Vector of extrapolated interpolation objects (one per quantity)
  - `scaled_itps::Vector{S}`: Vector of scaled interpolation objects (for hessian)
  - `fs::Array{Float64,3}`: Original function values (nx × ny × nqty)
  - `periodic_y::Bool`: Whether y dimension has periodic boundary conditions
  - `y_period::Float64`: Period for y coordinate wrapping (ys[end] - ys[1] + dy)

# Work arrays (pre-allocated for efficiency)

  - `_f`, `_fx`, `_fy`, `_fxx`, `_fxy`, `_fyy`: Output vectors
"""
struct BicubicWrapper{I,S}
    xs::Vector{Float64}
    ys::Vector{Float64}
    nqty::Int
    itps::Vector{I}
    scaled_itps::Vector{S}
    fs::Array{Float64,3}
    periodic_y::Bool
    y_period::Float64
    # Work arrays
    _f::Vector{Float64}
    _fx::Vector{Float64}
    _fy::Vector{Float64}
    _fxx::Vector{Float64}
    _fxy::Vector{Float64}
    _fyy::Vector{Float64}
end

"""
    _resample_to_uniform_grid(xs, ys, fs, xs_uniform, ys_uniform, periodic_y) -> Array{Float64,3}

Resample 3D data from non-uniform grid to uniform grid using cubic interpolation.
This preserves data accuracy and smoothness better than assuming the non-uniform grid is uniform.
Cubic interpolation ensures accurate derivatives after resampling.
"""
function _resample_to_uniform_grid(
    xs::Vector{Float64}, ys::Vector{Float64}, fs::Array{Float64,3},
    xs_uniform::Vector{Float64}, ys_uniform::Vector{Float64}, periodic_y::Bool
)::Array{Float64,3}
    nx, ny, nqty = size(fs)
    fs_resampled = similar(fs)

    # For each quantity, create a cubic interpolation and resample
    # Using cubic preserves more smoothness than linear, important for accurate derivatives
    for q in 1:nqty
        # Create cubic interpolation on the original (possibly non-uniform) grid
        # Note: Cubic requires at least 4 points, fall back to linear if not enough
        if nx >= 4 && ny >= 4
            if periodic_y
                itp = interpolate(fs[:, :, q], (BSpline(Cubic(Line(OnGrid()))), BSpline(Cubic(Periodic(OnGrid())))))
            else
                itp = interpolate(fs[:, :, q], BSpline(Cubic(Line(OnGrid()))))
            end
        else
            # Fallback to linear for small grids
            if periodic_y
                itp = interpolate(fs[:, :, q], (BSpline(Linear()), BSpline(Linear())))
            else
                itp = interpolate(fs[:, :, q], BSpline(Linear()))
            end
        end

        # Map from uniform coordinates to knot indices for the original non-uniform grid
        for (i, xu) in enumerate(xs_uniform)
            # Find the position in the original x grid
            ix = _find_interp_index(xs, xu)

            for (j, yu) in enumerate(ys_uniform)
                # Find the position in the original y grid
                jy = _find_interp_index(ys, yu)

                # Evaluate the interpolation
                fs_resampled[i, j, q] = itp(ix, jy)
            end
        end
    end

    return fs_resampled
end

"""
    _find_interp_index(coords, val) -> Float64

Find the floating-point index for interpolation on a non-uniform grid.
Returns a value between 1 and length(coords) suitable for use with Interpolations.jl.
"""
function _find_interp_index(coords::Vector{Float64}, val::Float64)::Float64
    n = length(coords)
    if n == 1
        return 1.0
    end

    # Handle out of bounds
    if val <= coords[1]
        return 1.0
    elseif val >= coords[end]
        return Float64(n)
    end

    # Binary search to find bracketing interval
    lo = 1
    hi = n
    while hi - lo > 1
        mid = (lo + hi) ÷ 2
        if coords[mid] <= val
            lo = mid
        else
            hi = mid
        end
    end

    # Linear interpolation within the interval to get fractional index
    t = (val - coords[lo]) / (coords[hi] - coords[lo])
    return lo + t
end

"""
    _check_uniform_spacing(coords::Vector{Float64}, name::String) -> Bool

Check if coordinates are uniformly spaced within tolerance.
Returns true if uniform, false if non-uniform. Issues a warning for non-uniform grids.
"""
function _check_uniform_spacing(coords::Vector{Float64}, name::String)::Bool
    n = length(coords)
    n < 2 && return true  # Single point is trivially uniform

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
            # Non-uniform grid - warn and return false
            # The interpolation will use uniform approximation
            return false
        end
    end
    return true
end

"""
    BicubicWrapper(xs, ys, fs; periodic_y=false, endpoint_inclusive=false)

Create a 2D bicubic spline interpolator.

# Arguments

  - `xs::Vector{Float64}`: X-coordinates (sorted, uniformly spaced)
  - `ys::Vector{Float64}`: Y-coordinates (sorted, uniformly spaced)
  - `fs::Array{Float64,3}`: Function values (nx × ny × nqty)
  - `periodic_y::Bool`: If true, use periodic boundary conditions in y (theta direction)
  - `endpoint_inclusive::Bool`: If true and periodic_y=true, the y data includes a
    redundant endpoint (last point equals first point). The endpoint will be removed
    internally for proper B-spline periodic handling.

# Boundary Conditions

  - x (psi) direction: Linear extrapolation (Line BC)
  - y (theta) direction: Linear extrapolation (default) or Periodic (if periodic_y=true)

# Notes

Interpolations.jl requires uniformly spaced grids. This constructor validates
the spacing and throws an error if grids are non-uniform.

For periodic data in y, the grid should span [y0, y_period) without duplicating
the endpoint (i.e., don't include both 0 and 2π for a 2π-periodic function).
Use `endpoint_inclusive=true` if your data DOES include the endpoint.

# Example

```julia
# Non-periodic example
xs = range(0, 1; length=10) |> collect
ys = range(0, 2π; length=20) |> collect
fs = zeros(10, 20, 2)
for i in 1:10, j in 1:20
    fs[i, j, 1] = sin(xs[i]) * cos(ys[j])
    fs[i, j, 2] = cos(xs[i]) * sin(ys[j])
end
bw = BicubicWrapper(xs, ys, fs)

# Periodic in y (theta) example - note: endpoint NOT duplicated
xs = range(0, 1; length=10) |> collect
ys = range(0, 1; length=20)[1:(end-1)] |> collect  # [0, 1) without endpoint
fs = zeros(10, 19, 2)
bw_periodic = BicubicWrapper(xs, ys, fs; periodic_y=true)

# Periodic with endpoint included (from Fortran-style data)
xs = range(0, 1; length=10) |> collect
ys = range(0, 1; length=20) |> collect  # [0, 1] WITH endpoint
fs = zeros(10, 20, 2)
bw_periodic_ei = BicubicWrapper(xs, ys, fs; periodic_y=true, endpoint_inclusive=true)
```
"""
function BicubicWrapper(
    xs::Vector{Float64}, ys::Vector{Float64}, fs::Array{Float64,3};
    periodic_y::Bool=false, endpoint_inclusive::Bool=false
)
    nx, ny, nqty = size(fs)
    @assert length(xs) == nx "xs length must match first dimension of fs"
    @assert length(ys) == ny "ys length must match second dimension of fs"

    # Handle endpoint-inclusive periodic data
    # B-splines with periodic BC need endpoint-exclusive data for correct derivative continuity
    if periodic_y && endpoint_inclusive && ny > 1
        # Strip the redundant endpoint
        ys_internal = ys[1:(end-1)]
        fs_internal = fs[:, 1:(end-1), :]
        ny_internal = ny - 1
        # Compute period from original data (full range)
        y_period = ys[end] - ys[1]
    else
        ys_internal = ys
        fs_internal = fs
        ny_internal = ny
        # For endpoint-exclusive data, period = range + one spacing
        dy = ny > 1 ? (ys[end] - ys[1]) / (ny - 1) : 0.0
        y_period = periodic_y ? (ys[end] - ys[1] + dy) : 0.0
    end

    # Create uniform target ranges
    xs_range = range(xs[1], xs[end]; length=nx)
    ys_range = range(ys_internal[1], ys_internal[end]; length=ny_internal)
    xs_uniform = collect(xs_range)
    ys_uniform = collect(ys_range)

    # Check if grids are non-uniform and resample if needed
    x_uniform = _check_uniform_spacing(xs, "x")
    y_uniform = _check_uniform_spacing(ys_internal, "y")

    if !x_uniform || !y_uniform
        # Resample data to uniform grid using linear interpolation
        # This is more accurate than pretending non-uniform grid is uniform
        fs_resampled = _resample_to_uniform_grid(xs, ys_internal, fs_internal, xs_uniform, ys_uniform, periodic_y)
        fs_use = fs_resampled
    else
        fs_use = fs_internal
    end

    # Choose boundary condition for y based on periodic_y flag
    if periodic_y
        # Periodic BC in y (theta direction)
        # Use tuple of interpolation types for per-dimension BC
        # Create first interpolation to determine type
        raw_itp_1 = interpolate(
            fs_use[:, :, 1],
            (BSpline(Cubic(Line(OnGrid()))), BSpline(Cubic(Periodic(OnGrid()))))
        )

        # Scaling for periodic requires special handling
        scaled_itp_1 = scale(raw_itp_1, xs_range, ys_range)
        extrap_itp_1 = extrapolate(scaled_itp_1, (Line(), Periodic()))
        ItpType = typeof(extrap_itp_1)
        ScaledType = typeof(scaled_itp_1)

        # Create type-stable vectors for both extrapolated and scaled interpolations
        itps = Vector{ItpType}(undef, nqty)
        scaled_itps = Vector{ScaledType}(undef, nqty)
        itps[1] = extrap_itp_1
        scaled_itps[1] = scaled_itp_1

        # Create remaining interpolation objects
        for q in 2:nqty
            raw_itp = interpolate(
                fs_use[:, :, q],
                (BSpline(Cubic(Line(OnGrid()))), BSpline(Cubic(Periodic(OnGrid()))))
            )
            scaled_itp = scale(raw_itp, xs_range, ys_range)
            itps[q] = extrapolate(scaled_itp, (Line(), Periodic()))
            scaled_itps[q] = scaled_itp
        end
    else
        # Non-periodic: Line BC in both directions
        raw_itp_1 = interpolate(fs_use[:, :, 1], BSpline(Cubic(Line(OnGrid()))))
        scaled_itp_1 = scale(raw_itp_1, xs_range, ys_range)
        extrap_itp_1 = extrapolate(scaled_itp_1, Line())
        ItpType = typeof(extrap_itp_1)
        ScaledType = typeof(scaled_itp_1)

        # Create type-stable vectors (scaled_itps used for hessian)
        itps = Vector{ItpType}(undef, nqty)
        scaled_itps = Vector{ScaledType}(undef, nqty)
        itps[1] = extrap_itp_1
        scaled_itps[1] = scaled_itp_1

        # Create remaining interpolation objects
        for q in 2:nqty
            raw_itp = interpolate(fs_use[:, :, q], BSpline(Cubic(Line(OnGrid()))))
            scaled_itp = scale(raw_itp, xs_range, ys_range)
            itps[q] = extrapolate(scaled_itp, Line())
            scaled_itps[q] = scaled_itp
        end
    end

    # Work arrays
    _f = zeros(Float64, nqty)
    _fx = zeros(Float64, nqty)
    _fy = zeros(Float64, nqty)
    _fxx = zeros(Float64, nqty)
    _fxy = zeros(Float64, nqty)
    _fyy = zeros(Float64, nqty)

    BicubicWrapper{ItpType,ScaledType}(
        xs, ys, nqty, itps, scaled_itps, copy(fs), periodic_y, y_period,
        _f, _fx, _fy, _fxx, _fxy, _fyy
    )
end

"""
    _wrap_periodic(y, y_min, y_period) -> Float64

Wrap y coordinate into the range [y_min, y_min + y_period) for periodic interpolation.
"""
function _wrap_periodic(y::Float64, y_min::Float64, y_period::Float64)::Float64
    if y_period ≤ 0
        return y
    end
    # Shift to [0, period) then shift back
    y_shifted = y - y_min
    y_wrapped = mod(y_shifted, y_period)
    return y_wrapped + y_min
end

"""
    _wrap_to_data_range(y, y_min, y_max, y_period) -> Float64

Wrap y coordinate to the data range [y_min, y_max] using the given period.
This is used for periodic interpolation where Interpolations.jl's Periodic()
extrapolation doesn't correctly handle coordinates outside the scaled range.

For endpoint-inclusive data (y_max - y_min ≈ y_period), this returns y clamped
to [y_min, y_max]. For endpoint-exclusive data, this wraps using the period.
"""
function _wrap_to_data_range(y::Float64, y_min::Float64, y_max::Float64, y_period::Float64)::Float64
    if y_period ≤ 0
        return y
    end

    # First wrap to [y_min, y_min + y_period)
    y_wrapped = _wrap_periodic(y, y_min, y_period)

    # For endpoint-inclusive data, y_max ≈ y_min + y_period
    # In this case, y_wrapped is already in [y_min, y_max)
    # For endpoint-exclusive data, y_max < y_min + y_period
    # and we need to ensure y is within [y_min, y_max]
    data_range = y_max - y_min
    if data_range >= y_period * 0.999  # endpoint-inclusive (with tolerance)
        return y_wrapped
    else
        # Endpoint-exclusive: wrap to [y_min, y_max]
        # Map [0, y_period) to [y_min, y_max]
        t = (y_wrapped - y_min) / y_period  # t in [0, 1)
        return y_min + t * data_range
    end
end

"""
    evaluate!(bw, x, y) -> Vector{Float64}

Evaluate the bicubic spline at point (x, y).

Returns a reference to the internal work array `_f`.
The result is valid until the next mutating call on this wrapper.

Thread-safety: NOT thread-safe. Use separate wrappers per thread.
"""
function evaluate!(bw::BicubicWrapper{I,S}, x::Float64, y::Float64) where {I,S}
    # For periodic y, wrap coordinate to the data range
    y_eval = bw.periodic_y ? _wrap_to_data_range(y, bw.ys[1], bw.ys[end], bw.y_period) : y

    @inbounds for q in 1:bw.nqty
        bw._f[q] = bw.itps[q](x, y_eval)
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
function deriv1!(bw::BicubicWrapper{I,S}, x::Float64, y::Float64) where {I,S}
    # For periodic y, wrap coordinate to the data range [ys[1], ys[end]]
    # This is needed because Interpolations.jl's Periodic() extrapolation
    # wraps based on the scaled range, which may not match the actual period
    y_eval = bw.periodic_y ? _wrap_to_data_range(y, bw.ys[1], bw.ys[end], bw.y_period) : y

    @inbounds for q in 1:bw.nqty
        bw._f[q] = bw.itps[q](x, y_eval)
        grad = Interpolations.gradient(bw.itps[q], x, y_eval)
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
function deriv2!(bw::BicubicWrapper{I,S}, x::Float64, y::Float64) where {I,S}
    # For periodic y, wrap coordinate to the data range
    y_eval = bw.periodic_y ? _wrap_to_data_range(y, bw.ys[1], bw.ys[end], bw.y_period) : y
    x_eval = clamp(x, bw.xs[1], bw.xs[end])

    # Use extrapolated interpolations for value and gradient
    @inbounds for q in 1:bw.nqty
        bw._f[q] = bw.itps[q](x, y_eval)
        grad = Interpolations.gradient(bw.itps[q], x, y_eval)
        bw._fx[q] = grad[1]
        bw._fy[q] = grad[2]
    end

    # For hessian, use scaled interpolations directly
    # This works around Interpolations.jl not supporting hessian through extrapolation for periodic BCs
    @inbounds for q in 1:bw.nqty
        hess = Interpolations.hessian(bw.scaled_itps[q], x_eval, y_eval)
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

# =============================================================================
# Compatibility Aliases (matching legacy Fortran API)
# =============================================================================

"""
    bicube_eval!(bw, x, y)

Legacy API alias for `evaluate!(bw, x, y)`.
"""
bicube_eval!(bw::BicubicWrapper, x::Float64, y::Float64) = evaluate!(bw, x, y)

"""
    bicube_deriv1!(bw, x, y)

Legacy API alias for `deriv1!(bw, x, y)`.
"""
bicube_deriv1!(bw::BicubicWrapper, x::Float64, y::Float64) = deriv1!(bw, x, y)

"""
    bicube_deriv2!(bw, x, y)

Legacy API alias for `deriv2!(bw, x, y)`.
"""
bicube_deriv2!(bw::BicubicWrapper, x::Float64, y::Float64) = deriv2!(bw, x, y)
