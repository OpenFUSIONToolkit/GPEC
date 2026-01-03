# Pure Julia BicubicSpline implementation using Interpolations.jl

"""
    BicubicSpline

A 2D bicubic spline interpolation supporting multiple quantities.
Uses Interpolations.jl under the hood.

## Fields (read-only via getproperty):
- `xs`: Grid points in x-direction
- `ys`: Grid points in y-direction
- `fs`: Function values at grid points (mx+1 × my+1 × nqty array)
- `fsx`: X-derivatives at grid points
- `fsy`: Y-derivatives at grid points
- `fsxy`: Mixed derivatives at grid points
"""
mutable struct BicubicSpline
    _xs::Vector{Float64}
    _ys::Vector{Float64}
    _fs::Array{Float64,3}
    mx::Int
    my::Int
    nqty::Int
    bctypex::Int32
    bctypey::Int32

    _fsx::Array{Float64,3}
    _fsy::Array{Float64,3}
    _fsxy::Array{Float64,3}

    _f::Vector{Float64}
    _fx::Vector{Float64}
    _fy::Vector{Float64}
    _fxx::Vector{Float64}
    _fxy::Vector{Float64}
    _fyy::Vector{Float64}

    _interps::Vector{Any}  # Vector of 2D interpolation objects (one per quantity)
end

@expose_fields BicubicSpline xs ys fs fsx fsy fsxy

"""
    _get_bicube_extrap(bctype::Int32)

Get extrapolation boundary condition for Interpolations.jl.
"""
function _get_bicube_extrap(bctype::Int32)
    if bctype == 2  # Periodic
        return Periodic()
    else
        return Line()
    end
end

"""
    _vector_to_range(xs::Vector{Float64})

Convert a uniformly-spaced vector to a range for use with Interpolations.jl scale().
Returns a StepRangeLen that matches the vector's start, stop, and length.
"""
function _bicube_vector_to_range(xs::Vector{Float64})
    n = length(xs)
    @assert n >= 2 "Need at least 2 points for interpolation"
    return range(xs[1]; stop=xs[end], length=n)
end

"""
    _get_bicube_bspline_bc(n::Int)

Get appropriate BSpline boundary condition based on grid size.
Free(OnGrid()) gives best accuracy but requires n >= 4.
Falls back to Line(OnGrid()) for smaller grids.
"""
function _get_bicube_bspline_bc(n::Int)
    if n >= 4
        return Free(OnGrid())
    else
        return Line(OnGrid())
    end
end

"""
    _create_bicubic_interpolations(xs, ys, fs, bctypex, bctypey)

Create 2D interpolation objects for each quantity using cubic spline interpolation.
"""
function _create_bicubic_interpolations(xs::Vector{Float64}, ys::Vector{Float64},
    fs::Array{Float64,3}, bctypex::Int32, bctypey::Int32)

    nqty = size(fs, 3)
    interps = Vector{Any}(undef, nqty)

    extrap_x = _get_bicube_extrap(bctypex)
    extrap_y = _get_bicube_extrap(bctypey)

    # Convert vectors to ranges for scale()
    xs_range = _bicube_vector_to_range(xs)
    ys_range = _bicube_vector_to_range(ys)

    # Use appropriate boundary condition based on minimum grid dimension
    n_min = min(length(xs), length(ys))
    bc = _get_bicube_bspline_bc(n_min)

    for i in 1:nqty
        # Create 2D cubic spline interpolation
        itp = interpolate(fs[:, :, i], BSpline(Cubic(bc)))
        scaled_itp = scale(itp, xs_range, ys_range)
        interps[i] = extrapolate(scaled_itp, (extrap_x, extrap_y))
    end

    return interps
end

"""
    _eval_bicube_value(interps, x, y, qty_idx)

Evaluate a single 2D interpolation at point (x, y).
"""
function _eval_bicube_value(interps, x::Float64, y::Float64, qty_idx::Int)
    return interps[qty_idx](x, y)
end

"""
    _eval_bicube_deriv(interps, x, y, qty_idx, dx, dy)

Evaluate derivatives of 2D interpolation at point (x, y).
"""
function _eval_bicube_deriv(interps, x::Float64, y::Float64, qty_idx::Int, dx::Int, dy::Int)
    itp = interps[qty_idx]

    if dx == 0 && dy == 0
        return itp(x, y)

    elseif dx == 1 && dy == 0
        grad = Interpolations.gradient(itp, x, y)
        return grad[1]

    elseif dx == 0 && dy == 1
        grad = Interpolations.gradient(itp, x, y)
        return grad[2]

    elseif dx == 2 && dy == 0
        hess = Interpolations.hessian(itp, x, y)
        return hess[1, 1]

    elseif dx == 1 && dy == 1
        hess = Interpolations.hessian(itp, x, y)
        return hess[1, 2]

    elseif dx == 0 && dy == 2
        hess = Interpolations.hessian(itp, x, y)
        return hess[2, 2]

    else
        error("Derivative order ($dx, $dy) not supported")
    end
end

"""
    _compute_bicube_derivatives_at_grid(xs, ys, nqty, interps)

Compute derivatives at grid points for storage.
"""
function _compute_bicube_derivatives_at_grid(xs::Vector{Float64}, ys::Vector{Float64},
    nqty::Int, interps)

    mx = length(xs) - 1
    my = length(ys) - 1

    fsx = Array{Float64}(undef, mx + 1, my + 1, nqty)
    fsy = Array{Float64}(undef, mx + 1, my + 1, nqty)
    fsxy = Array{Float64}(undef, mx + 1, my + 1, nqty)

    for k in 1:nqty
        for j in 1:(my + 1)
            for i in 1:(mx + 1)
                fsx[i, j, k] = _eval_bicube_deriv(interps, xs[i], ys[j], k, 1, 0)
                fsy[i, j, k] = _eval_bicube_deriv(interps, xs[i], ys[j], k, 0, 1)
                fsxy[i, j, k] = _eval_bicube_deriv(interps, xs[i], ys[j], k, 1, 1)
            end
        end
    end

    return fsx, fsy, fsxy
end

"""
    BicubicSpline(xs, ys, fs; bctypex="not-a-knot", bctypey="not-a-knot")

Create a 2D bicubic spline interpolation.

## Arguments:
- `xs`: A vector of Float64 values representing the x-coordinates.
- `ys`: A vector of Float64 values representing the y-coordinates.
- `fs`: A 3D array of Float64 values (mx+1 × my+1 × nqty).

## Keyword Arguments:
- `bctypex`: Boundary condition type for x (default "not-a-knot")
- `bctypey`: Boundary condition type for y (default "not-a-knot")

## Returns:
- A `BicubicSpline` object ready for evaluation.
"""
function BicubicSpline(xs::Vector{Float64}, ys::Vector{Float64}, fs::Array{Float64,3};
    bctypex::Union{String,Int}="not-a-knot", bctypey::Union{String,Int}="not-a-knot")

    @assert length(xs) == size(fs, 1) "Length of xs must match number of rows in fs"
    @assert length(ys) == size(fs, 2) "Length of ys must match number of columns in fs"

    bctype_code_x = Int32(parse_bctype(bctypex))
    bctype_code_y = Int32(parse_bctype(bctypey))

    mx = length(xs) - 1
    my = length(ys) - 1
    nqty = size(fs, 3)

    # Create interpolation objects
    interps = _create_bicubic_interpolations(xs, ys, fs, bctype_code_x, bctype_code_y)

    # Compute derivatives at grid points
    fsx, fsy, fsxy = _compute_bicube_derivatives_at_grid(xs, ys, nqty, interps)

    # Initialize work arrays
    f = Vector{Float64}(undef, nqty)
    fx = Vector{Float64}(undef, nqty)
    fy = Vector{Float64}(undef, nqty)
    fxx = Vector{Float64}(undef, nqty)
    fxy = Vector{Float64}(undef, nqty)
    fyy = Vector{Float64}(undef, nqty)

    return BicubicSpline(xs, ys, fs, mx, my, nqty, bctype_code_x, bctype_code_y,
        fsx, fsy, fsxy, f, fx, fy, fxx, fxy, fyy, interps)
end

"""
    bicube_eval!(bicube::BicubicSpline, x::Float64, y::Float64)

Evaluate the bicubic spline at a single point using in-place operations.
"""
function bicube_eval!(bicube::BicubicSpline, x::Float64, y::Float64)
    @assert !isempty(bicube._interps) "BicubicSpline has not been initialized"
    f = bicube._f
    for i in 1:bicube.nqty
        f[i] = _eval_bicube_value(bicube._interps, x, y, i)
    end
    return f
end

"""
    bicube_deriv1!(bicube::BicubicSpline, x::Float64, y::Float64)

Evaluate the bicubic spline and its first derivatives at a single point.
"""
function bicube_deriv1!(bicube::BicubicSpline, x::Float64, y::Float64)
    @assert !isempty(bicube._interps) "BicubicSpline has not been initialized"
    f, fx, fy = bicube._f, bicube._fx, bicube._fy
    for i in 1:bicube.nqty
        f[i] = _eval_bicube_deriv(bicube._interps, x, y, i, 0, 0)
        fx[i] = _eval_bicube_deriv(bicube._interps, x, y, i, 1, 0)
        fy[i] = _eval_bicube_deriv(bicube._interps, x, y, i, 0, 1)
    end
    return f, fx, fy
end

"""
    bicube_deriv2!(bicube::BicubicSpline, x::Float64, y::Float64)

Evaluate the bicubic spline and its first and second derivatives at a single point.
"""
function bicube_deriv2!(bicube::BicubicSpline, x::Float64, y::Float64)
    @assert !isempty(bicube._interps) "BicubicSpline has not been initialized"
    f, fx, fy = bicube._f, bicube._fx, bicube._fy
    fxx, fxy, fyy = bicube._fxx, bicube._fxy, bicube._fyy
    for i in 1:bicube.nqty
        f[i] = _eval_bicube_deriv(bicube._interps, x, y, i, 0, 0)
        fx[i] = _eval_bicube_deriv(bicube._interps, x, y, i, 1, 0)
        fy[i] = _eval_bicube_deriv(bicube._interps, x, y, i, 0, 1)
        fxx[i] = _eval_bicube_deriv(bicube._interps, x, y, i, 2, 0)
        fxy[i] = _eval_bicube_deriv(bicube._interps, x, y, i, 1, 1)
        fyy[i] = _eval_bicube_deriv(bicube._interps, x, y, i, 0, 2)
    end
    return f, fx, fy, fxx, fxy, fyy
end

"""
    bicube_eval(bicube::BicubicSpline, xs::Vector{Float64}, ys::Vector{Float64}, derivs::Int=0)

Batch evaluation of the bicubic spline at multiple points.
"""
function bicube_eval(bicube::BicubicSpline, xs::Vector{Float64}, ys::Vector{Float64}, derivs::Int=0)
    @assert !isempty(bicube._interps) "BicubicSpline has not been initialized"
    @assert derivs in 0:2 "Invalid number of derivatives requested: $derivs. Must be 0, 1, or 2."

    n = length(xs)
    m = length(ys)

    fs = Array{Float64}(undef, n, m, bicube.nqty)
    f = bicube._f
    if derivs > 0
        fsx, fsy = similar(fs), similar(fs)
        fx, fy = bicube._fx, bicube._fy
    end
    if derivs > 1
        fsxx, fsxy, fsyy = similar(fs), similar(fs), similar(fs)
        fxx, fxy, fyy = bicube._fxx, bicube._fxy, bicube._fyy
    end

    for (i, x) in enumerate(xs)
        for (j, y) in enumerate(ys)
            for k in 1:bicube.nqty
                fs[i, j, k] = _eval_bicube_deriv(bicube._interps, x, y, k, 0, 0)
                if derivs > 0
                    fsx[i, j, k] = _eval_bicube_deriv(bicube._interps, x, y, k, 1, 0)
                    fsy[i, j, k] = _eval_bicube_deriv(bicube._interps, x, y, k, 0, 1)
                end
                if derivs > 1
                    fsxx[i, j, k] = _eval_bicube_deriv(bicube._interps, x, y, k, 2, 0)
                    fsxy[i, j, k] = _eval_bicube_deriv(bicube._interps, x, y, k, 1, 1)
                    fsyy[i, j, k] = _eval_bicube_deriv(bicube._interps, x, y, k, 0, 2)
                end
            end
        end
    end

    if derivs == 0
        return fs
    elseif derivs == 1
        return fs, fsx, fsy
    elseif derivs == 2
        return fs, fsx, fsy, fsxx, fsxy, fsyy
    end
end

"""
    empty_BicubicSpline()

Create a minimal, type-stable placeholder bicubic spline.
"""
function empty_BicubicSpline()
    xs = Float64[]
    ys = Float64[]
    fs = Array{Float64}(undef, 0, 0, 0)
    fsx = Array{Float64}(undef, 0, 0, 0)
    fsy = Array{Float64}(undef, 0, 0, 0)
    fsxy = Array{Float64}(undef, 0, 0, 0)
    f = Vector{Float64}(undef, 0)
    fx = Vector{Float64}(undef, 0)
    fy = Vector{Float64}(undef, 0)
    fxx = Vector{Float64}(undef, 0)
    fxy = Vector{Float64}(undef, 0)
    fyy = Vector{Float64}(undef, 0)
    interps = Vector{Any}()
    return BicubicSpline(xs, ys, fs, 0, 0, 0, zero(Int32), zero(Int32),
        fsx, fsy, fsxy, f, fx, fy, fxx, fxy, fyy, interps)
end
