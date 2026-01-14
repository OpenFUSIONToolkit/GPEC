"""
BicubicSpline - Pure Julia 2D bicubic spline interpolation

This implementation replicates the Fortran bicube.f algorithm from GPEC/NIMROD,
using 1D cubic spline fits to compute derivatives at grid points, then
constructing local bicubic polynomials for evaluation.

# Algorithm

The bicubic spline is constructed by:
1. Fitting 1D cubic splines along the y direction to get ∂f/∂y (fsy)
2. Fitting 1D cubic splines along the x direction to get ∂f/∂x (fsx)
3. Fitting 1D cubic splines along x on fsy to get ∂²f/∂x∂y (fsxy)

Evaluation uses the standard Hermite bicubic form with the 16 coefficients
derived from f, fx, fy, fxy at the four corners of each cell.

# Boundary Conditions

Supports independent boundary conditions in x and y:
- `"extrap"`: Extrapolate derivatives from interior using 4-point Lagrange
- `"periodic"`: Periodic boundary conditions (f(end) = f(begin))
- `"natural"`: Zero second derivative at boundaries

# Thread Safety Warning

The evaluation functions (`evaluate!`, `deriv1!`, `deriv2!`) mutate internal work
arrays and are NOT thread-safe. For multi-threaded usage:
- Use separate BicubicSpline instances per thread

# Reference

H. Spaeth, "Spline Algorithms for Curves and Surfaces,"
Translated from the German by W. D. Hoskins and H. W. Sager.
Utilitas Mathematica Publishing Inc., Winnipeg, 1974.
"""

"""
    BicubicSpline{T}

Pure Julia 2D bicubic spline interpolator.

# Type Parameters

  - `T`: Element type (Float64 or ComplexF64)

# Fields

  - `xs::Vector{Float64}`: X-coordinates (length nx)
  - `ys::Vector{Float64}`: Y-coordinates (length ny)
  - `nqty::Int`: Number of quantities
  - `nx::Int`, `ny::Int`: Grid dimensions
  - `fs::Array{T,3}`: Function values (nx × ny × nqty)
  - `fsx::Array{T,3}`: ∂f/∂x at grid points
  - `fsy::Array{T,3}`: ∂f/∂y at grid points
  - `fsxy::Array{T,3}`: ∂²f/∂x∂y at grid points
  - `periodic_x::Bool`, `periodic_y::Bool`: Periodic boundary flags
"""
struct BicubicSpline{T}
    xs::Vector{Float64}
    ys::Vector{Float64}
    nqty::Int
    nx::Int
    ny::Int

    # Data arrays at grid points
    fs::Array{T,3}      # f values
    fsx::Array{T,3}     # ∂f/∂x
    fsy::Array{T,3}     # ∂f/∂y
    fsxy::Array{T,3}    # ∂²f/∂x∂y

    # Boundary condition flags
    periodic_x::Bool
    periodic_y::Bool

    # Work arrays for evaluation (not thread-safe)
    _f::Vector{T}
    _fx::Vector{T}
    _fy::Vector{T}
    _fxx::Vector{T}
    _fxy::Vector{T}
    _fyy::Vector{T}
    _last_ix::Base.RefValue{Int}
    _last_iy::Base.RefValue{Int}
end

"""
    BicubicSpline(xs, ys, fs; bctypex="extrap", bctypey="extrap", endpoint_inclusive_y=false)

Create a 2D bicubic spline interpolator.

# Arguments

  - `xs::Vector{Float64}`: X-coordinates (sorted, length nx)
  - `ys::Vector{Float64}`: Y-coordinates (sorted, length ny)
  - `fs::Array{T,3}`: Function values (nx × ny × nqty)
  - `bctypex`: Boundary condition in x ("extrap", "periodic", "natural")
  - `bctypey`: Boundary condition in y ("extrap", "periodic", "natural")
  - `endpoint_inclusive_y`: If true and bctypey="periodic", the y data includes a
    duplicated endpoint (e.g., both 0 and 2π for a 2π-periodic function). The endpoint
    is stripped internally for proper periodic spline construction.

# Example

```julia
xs = range(0, 1; length=10) |> collect
ys = range(0, 2π; length=20) |> collect
fs = zeros(10, 20, 2)
for i in 1:10, j in 1:20
    fs[i, j, 1] = sin(xs[i]) * cos(ys[j])
    fs[i, j, 2] = cos(xs[i]) * sin(ys[j])
end
bcs = BicubicSpline(xs, ys, fs; bctypex="extrap", bctypey="periodic")
```
"""
function BicubicSpline(
    xs::Vector{Float64}, ys::Vector{Float64}, fs::Array{T,3};
    bctypex::String="extrap", bctypey::String="extrap", endpoint_inclusive_y::Bool=false
) where {T<:Union{Float64,ComplexF64}}

    nx_orig, ny_orig, nqty = size(fs)
    @assert length(xs) == nx_orig "xs length must match first dimension of fs"
    @assert length(ys) == ny_orig "ys length must match second dimension of fs"
    @assert nx_orig >= 2 && ny_orig >= 2 "Need at least 2 points in each dimension"
    @assert issorted(xs) "xs must be sorted in ascending order"
    @assert issorted(ys) "ys must be sorted in ascending order"

    # Set periodicity flags
    periodic_x = (bctypex == "periodic")
    periodic_y = (bctypey == "periodic")

    # Handle endpoint-inclusive periodic data
    # For compatibility with code that accesses .ys and .fs directly, we store the
    # endpoint-inclusive data. For spline fitting, we strip the endpoint since
    # CubicSpline1D with periodic BC expects data WITHOUT endpoint duplication.
    endpoint_stripped = endpoint_inclusive_y && periodic_y && ny_orig > 2

    if endpoint_stripped
        ys_fit = ys[1:(end-1)]       # For spline fitting (no endpoint)
        fs_fit = fs[:, 1:(end-1), :] # For spline fitting
        ny_fit = ny_orig - 1         # Size for spline fitting
    else
        ys_fit = ys
        fs_fit = fs
        ny_fit = ny_orig
    end

    # Store original (endpoint-inclusive) dimensions
    nx = nx_orig
    ny = ny_orig

    # Allocate derivative arrays at ORIGINAL size (endpoint-inclusive)
    fsx = zeros(T, nx, ny, nqty)
    fsy = zeros(T, nx, ny, nqty)
    fsxy = zeros(T, nx, ny, nqty)

    # Make a working copy of fs at fit size
    fs_fit_work = copy(fs_fit)

    # Enforce periodicity if needed (for non-endpoint-inclusive data)
    if periodic_x
        fs_fit_work[nx, :, :] .= fs_fit_work[1, :, :]
    end
    if periodic_y && !endpoint_stripped
        fs_fit_work[:, ny_fit, :] .= fs_fit_work[:, 1, :]
    end

    # Fit 1D splines along y direction to get fsy (using stripped data)
    for iq in 1:nqty
        for ix in 1:nx
            y_data = fs_fit_work[ix, :, iq]
            y_spline = CubicSpline1D(ys_fit, y_data; bctype=bctypey)
            fsy[ix, 1:ny_fit, iq] .= @view y_spline.fs1[:, 1]
        end
    end

    # Fit 1D splines along x direction to get fsx (using stripped data)
    for iq in 1:nqty
        for iy in 1:ny_fit
            x_data = fs_fit_work[:, iy, iq]
            x_spline = CubicSpline1D(xs, x_data; bctype=bctypex)
            fsx[:, iy, iq] .= @view x_spline.fs1[:, 1]
        end
    end

    # Fit 1D splines along x direction on fsy to get fsxy (mixed derivative)
    for iq in 1:nqty
        for iy in 1:ny_fit
            xy_data = fsy[:, iy, iq]
            xy_spline = CubicSpline1D(xs, xy_data; bctype=bctypex)
            fsxy[:, iy, iq] .= @view xy_spline.fs1[:, 1]
        end
    end

    # For endpoint-inclusive periodic data, copy endpoint derivatives from first point
    if endpoint_stripped
        fsx[:, ny, :] .= fsx[:, 1, :]
        fsy[:, ny, :] .= fsy[:, 1, :]
        fsxy[:, ny, :] .= fsxy[:, 1, :]
    end

    # Store the original (endpoint-inclusive) fs data
    fs_work = copy(fs)

    # Allocate work arrays
    _f = zeros(T, nqty)
    _fx = zeros(T, nqty)
    _fy = zeros(T, nqty)
    _fxx = zeros(T, nqty)
    _fxy = zeros(T, nqty)
    _fyy = zeros(T, nqty)
    _last_ix = Ref(1)
    _last_iy = Ref(1)

    BicubicSpline{T}(
        xs, copy(ys), nqty, nx, ny,
        fs_work, fsx, fsy, fsxy,
        periodic_x, periodic_y,
        _f, _fx, _fy, _fxx, _fxy, _fyy, _last_ix, _last_iy
    )
end

"""
    _find_interval(coords, val, last_i, n) -> Int

Find the interval index for evaluation. Uses cached interval for efficiency.
Returns index i such that coords[i] <= val < coords[i+1] (or handles boundaries).
"""
@inline function _find_interval(
    coords::Vector{Float64}, val::Float64, last_i::Int, n::Int
)::Int
    # Clamp cached index to valid range
    ix = clamp(last_i, 1, n - 1)

    # Search downward
    while ix > 1 && val < coords[ix]
        ix -= 1
    end

    # Search upward
    while ix < n - 1 && val >= coords[ix+1]
        ix += 1
    end

    return ix
end

"""
    _bicubic_wrap_periodic(val, coord_min, period) -> Float64

Wrap a value to the periodic domain [coord_min, coord_min + period).
"""
@inline function _bicubic_wrap_periodic(val::Float64, coord_min::Float64, period::Float64)::Float64
    while val >= coord_min + period
        val -= period
    end
    while val < coord_min
        val += period
    end
    return val
end

"""
    _bicube_getco!(cmat, bcs, ix, iy)

Compute the 4×4×nqty local coefficient matrix for cell (ix, iy).
This implements the Hermite bicubic coefficient transformation.

The coefficients cmat[i,j,q] correspond to the polynomial:
f(dx, dy) = Σᵢⱼ cmat[i,j,q] * dx^(i-1) * dy^(j-1)
where dx = x - xs[ix], dy = y - ys[iy].
"""
function _bicube_getco!(
    cmat::Array{T,3}, bcs::BicubicSpline{T}, ix::Int, iy::Int
) where {T}
    # Interval sizes
    hx = bcs.xs[ix+1] - bcs.xs[ix]
    hy = bcs.ys[iy+1] - bcs.ys[iy]
    hxinv = 1 / hx
    hyinv = 1 / hy
    hxinv2 = hxinv * hxinv
    hxinv3 = hxinv2 * hxinv
    hyinv2 = hyinv * hyinv
    hyinv3 = hyinv2 * hyinv

    # Transformation matrices (from Hermite basis to polynomial)
    # gxmat transforms from [f, fx, f+, fx+] to polynomial coefficients in x
    gx31 = -3 * hxinv2
    gx32 = -2 * hxinv
    gx33 = 3 * hxinv2
    gx34 = -hxinv
    gx41 = 2 * hxinv3
    gx42 = hxinv2
    gx43 = -2 * hxinv3
    gx44 = hxinv2

    # gymat transforms from [f, fy, f+, fy+] to polynomial coefficients in y
    gy31 = -3 * hyinv2
    gy32 = -2 * hyinv
    gy33 = 3 * hyinv2
    gy34 = -hyinv
    gy41 = 2 * hyinv3
    gy42 = hyinv2
    gy43 = -2 * hyinv3
    gy44 = hyinv2

    # Build the source matrix (f, fx, fy, fxy at 4 corners)
    # Layout: smat[row, col, q] where
    #   row 1,2 = x index ix, ix+1
    #   col 1,2 = y index iy, iy+1
    #   Then row 1,2 become f/fx pairs, col 1,2 become f/fy pairs

    @inbounds for q in 1:bcs.nqty
        # Source matrix: [f, fy; fx, fxy] at each corner
        s11 = bcs.fs[ix, iy, q]
        s12 = bcs.fsy[ix, iy, q]
        s13 = bcs.fs[ix, iy+1, q]
        s14 = bcs.fsy[ix, iy+1, q]
        s21 = bcs.fsx[ix, iy, q]
        s22 = bcs.fsxy[ix, iy, q]
        s23 = bcs.fsx[ix, iy+1, q]
        s24 = bcs.fsxy[ix, iy+1, q]
        s31 = bcs.fs[ix+1, iy, q]
        s32 = bcs.fsy[ix+1, iy, q]
        s33 = bcs.fs[ix+1, iy+1, q]
        s34 = bcs.fsy[ix+1, iy+1, q]
        s41 = bcs.fsx[ix+1, iy, q]
        s42 = bcs.fsxy[ix+1, iy, q]
        s43 = bcs.fsx[ix+1, iy+1, q]
        s44 = bcs.fsxy[ix+1, iy+1, q]

        # Apply gymat^T (transform y direction)
        t11 = s11
        t12 = s12
        t13 = gy31 * s11 + gy32 * s12 + gy33 * s13 + gy34 * s14
        t14 = gy41 * s11 + gy42 * s12 + gy43 * s13 + gy44 * s14
        t21 = s21
        t22 = s22
        t23 = gy31 * s21 + gy32 * s22 + gy33 * s23 + gy34 * s24
        t24 = gy41 * s21 + gy42 * s22 + gy43 * s23 + gy44 * s24
        t31 = s31
        t32 = s32
        t33 = gy31 * s31 + gy32 * s32 + gy33 * s33 + gy34 * s34
        t34 = gy41 * s31 + gy42 * s32 + gy43 * s33 + gy44 * s34
        t41 = s41
        t42 = s42
        t43 = gy31 * s41 + gy32 * s42 + gy33 * s43 + gy34 * s44
        t44 = gy41 * s41 + gy42 * s42 + gy43 * s43 + gy44 * s44

        # Apply gxmat (transform x direction)
        cmat[1, 1, q] = t11
        cmat[1, 2, q] = t12
        cmat[1, 3, q] = t13
        cmat[1, 4, q] = t14
        cmat[2, 1, q] = t21
        cmat[2, 2, q] = t22
        cmat[2, 3, q] = t23
        cmat[2, 4, q] = t24
        cmat[3, 1, q] = gx31 * t11 + gx32 * t21 + gx33 * t31 + gx34 * t41
        cmat[3, 2, q] = gx31 * t12 + gx32 * t22 + gx33 * t32 + gx34 * t42
        cmat[3, 3, q] = gx31 * t13 + gx32 * t23 + gx33 * t33 + gx34 * t43
        cmat[3, 4, q] = gx31 * t14 + gx32 * t24 + gx33 * t34 + gx34 * t44
        cmat[4, 1, q] = gx41 * t11 + gx42 * t21 + gx43 * t31 + gx44 * t41
        cmat[4, 2, q] = gx41 * t12 + gx42 * t22 + gx43 * t32 + gx44 * t42
        cmat[4, 3, q] = gx41 * t13 + gx42 * t23 + gx43 * t33 + gx44 * t43
        cmat[4, 4, q] = gx41 * t14 + gx42 * t24 + gx43 * t34 + gx44 * t44
    end
end

"""
    evaluate!(bcs, x, y) -> Vector{T}

Evaluate the bicubic spline at point (x, y).

Returns a reference to the internal work array `_f`.
The result is valid until the next mutating call on this spline.
"""
function evaluate!(bcs::BicubicSpline{T}, x::Float64, y::Float64) where {T}
    # Handle periodic wrapping
    xx = bcs.periodic_x ? _bicubic_wrap_periodic(x, bcs.xs[1], bcs.xs[end] - bcs.xs[1]) : x
    yy = bcs.periodic_y ? _bicubic_wrap_periodic(y, bcs.ys[1], bcs.ys[end] - bcs.ys[1]) : y

    # Find intervals
    ix = _find_interval(bcs.xs, xx, bcs._last_ix[], bcs.nx)
    iy = _find_interval(bcs.ys, yy, bcs._last_iy[], bcs.ny)
    bcs._last_ix[] = ix
    bcs._last_iy[] = iy

    # Compute local offsets
    dx = xx - bcs.xs[ix]
    dy = yy - bcs.ys[iy]

    # Get local coefficients
    cmat = Array{T,3}(undef, 4, 4, bcs.nqty)
    _bicube_getco!(cmat, bcs, ix, iy)

    # Evaluate f using Horner's method in dx
    # f = Σᵢⱼ c[i,j] * dx^(i-1) * dy^(j-1)
    @inbounds for q in 1:bcs.nqty
        # Inner sum over j (dy polynomial)
        val = ((cmat[4, 4, q] * dy + cmat[4, 3, q]) * dy + cmat[4, 2, q]) * dy + cmat[4, 1, q]
        val = val * dx + ((cmat[3, 4, q] * dy + cmat[3, 3, q]) * dy + cmat[3, 2, q]) * dy + cmat[3, 1, q]
        val = val * dx + ((cmat[2, 4, q] * dy + cmat[2, 3, q]) * dy + cmat[2, 2, q]) * dy + cmat[2, 1, q]
        val = val * dx + ((cmat[1, 4, q] * dy + cmat[1, 3, q]) * dy + cmat[1, 2, q]) * dy + cmat[1, 1, q]
        bcs._f[q] = val
    end

    return bcs._f
end

"""
    deriv1!(bcs, x, y) -> (f, fx, fy)

Evaluate bicubic spline and first derivatives at point (x, y).

Returns references to internal work arrays.
"""
function deriv1!(bcs::BicubicSpline{T}, x::Float64, y::Float64) where {T}
    # Handle periodic wrapping
    xx = bcs.periodic_x ? _bicubic_wrap_periodic(x, bcs.xs[1], bcs.xs[end] - bcs.xs[1]) : x
    yy = bcs.periodic_y ? _bicubic_wrap_periodic(y, bcs.ys[1], bcs.ys[end] - bcs.ys[1]) : y

    # Find intervals
    ix = _find_interval(bcs.xs, xx, bcs._last_ix[], bcs.nx)
    iy = _find_interval(bcs.ys, yy, bcs._last_iy[], bcs.ny)
    bcs._last_ix[] = ix
    bcs._last_iy[] = iy

    # Compute local offsets
    dx = xx - bcs.xs[ix]
    dy = yy - bcs.ys[iy]

    # Get local coefficients
    cmat = Array{T,3}(undef, 4, 4, bcs.nqty)
    _bicube_getco!(cmat, bcs, ix, iy)

    @inbounds for q in 1:bcs.nqty
        # Evaluate f
        val = ((cmat[4, 4, q] * dy + cmat[4, 3, q]) * dy + cmat[4, 2, q]) * dy + cmat[4, 1, q]
        val = val * dx + ((cmat[3, 4, q] * dy + cmat[3, 3, q]) * dy + cmat[3, 2, q]) * dy + cmat[3, 1, q]
        val = val * dx + ((cmat[2, 4, q] * dy + cmat[2, 3, q]) * dy + cmat[2, 2, q]) * dy + cmat[2, 1, q]
        val = val * dx + ((cmat[1, 4, q] * dy + cmat[1, 3, q]) * dy + cmat[1, 2, q]) * dy + cmat[1, 1, q]
        bcs._f[q] = val

        # Evaluate fy = ∂f/∂y (derivative of dy polynomial)
        fy_val = ((cmat[4, 4, q] * 3 * dy + cmat[4, 3, q] * 2) * dy + cmat[4, 2, q])
        fy_val = fy_val * dx + ((cmat[3, 4, q] * 3 * dy + cmat[3, 3, q] * 2) * dy + cmat[3, 2, q])
        fy_val = fy_val * dx + ((cmat[2, 4, q] * 3 * dy + cmat[2, 3, q] * 2) * dy + cmat[2, 2, q])
        fy_val = fy_val * dx + ((cmat[1, 4, q] * 3 * dy + cmat[1, 3, q] * 2) * dy + cmat[1, 2, q])
        bcs._fy[q] = fy_val

        # Evaluate fx = ∂f/∂x (derivative of dx polynomial, Horner in dy from high to low)
        fx_val = ((cmat[4, 4, q] * 3 * dx + cmat[3, 4, q] * 2) * dx + cmat[2, 4, q])
        fx_val = fx_val * dy + ((cmat[4, 3, q] * 3 * dx + cmat[3, 3, q] * 2) * dx + cmat[2, 3, q])
        fx_val = fx_val * dy + ((cmat[4, 2, q] * 3 * dx + cmat[3, 2, q] * 2) * dx + cmat[2, 2, q])
        fx_val = fx_val * dy + ((cmat[4, 1, q] * 3 * dx + cmat[3, 1, q] * 2) * dx + cmat[2, 1, q])
        bcs._fx[q] = fx_val
    end

    return bcs._f, bcs._fx, bcs._fy
end

"""
    deriv2!(bcs, x, y) -> (f, fx, fy, fxx, fxy, fyy)

Evaluate bicubic spline through second derivatives at point (x, y).

Returns references to internal work arrays.
"""
function deriv2!(bcs::BicubicSpline{T}, x::Float64, y::Float64) where {T}
    # Handle periodic wrapping
    xx = bcs.periodic_x ? _bicubic_wrap_periodic(x, bcs.xs[1], bcs.xs[end] - bcs.xs[1]) : x
    yy = bcs.periodic_y ? _bicubic_wrap_periodic(y, bcs.ys[1], bcs.ys[end] - bcs.ys[1]) : y

    # Find intervals
    ix = _find_interval(bcs.xs, xx, bcs._last_ix[], bcs.nx)
    iy = _find_interval(bcs.ys, yy, bcs._last_iy[], bcs.ny)
    bcs._last_ix[] = ix
    bcs._last_iy[] = iy

    # Compute local offsets
    dx = xx - bcs.xs[ix]
    dy = yy - bcs.ys[iy]

    # Get local coefficients
    cmat = Array{T,3}(undef, 4, 4, bcs.nqty)
    _bicube_getco!(cmat, bcs, ix, iy)

    @inbounds for q in 1:bcs.nqty
        # Evaluate f
        val = ((cmat[4, 4, q] * dy + cmat[4, 3, q]) * dy + cmat[4, 2, q]) * dy + cmat[4, 1, q]
        val = val * dx + ((cmat[3, 4, q] * dy + cmat[3, 3, q]) * dy + cmat[3, 2, q]) * dy + cmat[3, 1, q]
        val = val * dx + ((cmat[2, 4, q] * dy + cmat[2, 3, q]) * dy + cmat[2, 2, q]) * dy + cmat[2, 1, q]
        val = val * dx + ((cmat[1, 4, q] * dy + cmat[1, 3, q]) * dy + cmat[1, 2, q]) * dy + cmat[1, 1, q]
        bcs._f[q] = val

        # Evaluate fy
        fy_val = ((cmat[4, 4, q] * 3 * dy + cmat[4, 3, q] * 2) * dy + cmat[4, 2, q])
        fy_val = fy_val * dx + ((cmat[3, 4, q] * 3 * dy + cmat[3, 3, q] * 2) * dy + cmat[3, 2, q])
        fy_val = fy_val * dx + ((cmat[2, 4, q] * 3 * dy + cmat[2, 3, q] * 2) * dy + cmat[2, 2, q])
        fy_val = fy_val * dx + ((cmat[1, 4, q] * 3 * dy + cmat[1, 3, q] * 2) * dy + cmat[1, 2, q])
        bcs._fy[q] = fy_val

        # Evaluate fx (Horner in dy from high to low)
        fx_val = ((cmat[4, 4, q] * 3 * dx + cmat[3, 4, q] * 2) * dx + cmat[2, 4, q])
        fx_val = fx_val * dy + ((cmat[4, 3, q] * 3 * dx + cmat[3, 3, q] * 2) * dx + cmat[2, 3, q])
        fx_val = fx_val * dy + ((cmat[4, 2, q] * 3 * dx + cmat[3, 2, q] * 2) * dx + cmat[2, 2, q])
        fx_val = fx_val * dy + ((cmat[4, 1, q] * 3 * dx + cmat[3, 1, q] * 2) * dx + cmat[2, 1, q])
        bcs._fx[q] = fx_val

        # Evaluate fyy = ∂²f/∂y²
        fyy_val = (cmat[4, 4, q] * 6 * dy + cmat[4, 3, q] * 2)
        fyy_val = fyy_val * dx + (cmat[3, 4, q] * 6 * dy + cmat[3, 3, q] * 2)
        fyy_val = fyy_val * dx + (cmat[2, 4, q] * 6 * dy + cmat[2, 3, q] * 2)
        fyy_val = fyy_val * dx + (cmat[1, 4, q] * 6 * dy + cmat[1, 3, q] * 2)
        bcs._fyy[q] = fyy_val

        # Evaluate fxx = ∂²f/∂x² (Horner in dy from high to low)
        fxx_val = (cmat[4, 4, q] * 6 * dx + cmat[3, 4, q] * 2)
        fxx_val = fxx_val * dy + (cmat[4, 3, q] * 6 * dx + cmat[3, 3, q] * 2)
        fxx_val = fxx_val * dy + (cmat[4, 2, q] * 6 * dx + cmat[3, 2, q] * 2)
        fxx_val = fxx_val * dy + (cmat[4, 1, q] * 6 * dx + cmat[3, 1, q] * 2)
        bcs._fxx[q] = fxx_val

        # Evaluate fxy = ∂²f/∂x∂y
        fxy_val = ((cmat[4, 4, q] * 3 * dy + cmat[4, 3, q] * 2) * dy + cmat[4, 2, q]) * 3
        fxy_val = fxy_val * dx + ((cmat[3, 4, q] * 3 * dy + cmat[3, 3, q] * 2) * dy + cmat[3, 2, q]) * 2
        fxy_val = fxy_val * dx + ((cmat[2, 4, q] * 3 * dy + cmat[2, 3, q] * 2) * dy + cmat[2, 2, q])
        bcs._fxy[q] = fxy_val
    end

    return bcs._f, bcs._fx, bcs._fy, bcs._fxx, bcs._fxy, bcs._fyy
end

"""
    empty_BicubicSpline()

Create an empty/placeholder BicubicSpline for type stability.
"""
function empty_BicubicSpline()
    xs = Float64[0.0, 1.0]
    ys = Float64[0.0, 1.0]
    fs = zeros(Float64, 2, 2, 1)
    BicubicSpline(xs, ys, fs)
end

# Convenience evaluation (allocates, but thread-safe)
(bcs::BicubicSpline)(x::Float64, y::Float64) = copy(evaluate!(bcs, x, y))
