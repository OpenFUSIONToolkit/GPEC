mutable struct BicubicSpline
    handle::Ptr{Cvoid}
    _xs::Vector{Float64}
    _ys::Vector{Float64}
    _fs::Array{Float64,3} # 3D array for bicubic spline values
    mx::Int
    my::Int
    nqty::Int
    bctypex::Int32  # Boundary condition type for x
    bctypey::Int32  # Boundary condition type for y

    _fsx::Array{Float64,3}
    _fsy::Array{Float64,3}
    _fsxy::Array{Float64,3}

    _f::Vector{Float64}
    _fx::Vector{Float64}
    _fy::Vector{Float64}
    _fxx::Vector{Float64}
    _fxy::Vector{Float64}
    _fyy::Vector{Float64}
end

@expose_fields BicubicSpline xs ys fs fsx fsy fsxy

function BicubicSpline(handle::Ptr{Cvoid}, xs::Vector{Float64}, ys::Vector{Float64}, fs::Array{Float64,3},
    mx::Int, my::Int, nqty::Int, bctypex::Int32, bctypey::Int32,
    fsx::Array{Float64,3}, fsy::Array{Float64,3}, fsxy::Array{Float64,3})
    f = Vector{Float64}(undef, nqty)
    fx = Vector{Float64}(undef, nqty)
    fy = Vector{Float64}(undef, nqty)
    fxx = Vector{Float64}(undef, nqty)
    fxy = Vector{Float64}(undef, nqty)
    fyy = Vector{Float64}(undef, nqty)
    return BicubicSpline(handle, xs, ys, fs, mx, my, nqty, bctypex, bctypey,
        fsx, fsy, fsxy, f, fx, fy, fxx, fxy, fyy)
end


function _destroy_bicubic_spline(bicube::BicubicSpline)
    if bicube.handle != C_NULL
        ccall((:bicube_c_destroy, libspline), Cvoid, (Ptr{Cvoid},), bicube.handle)
        Core.setfield!(bicube, :handle, C_NULL)
    end
end

function BicubicSpline(mx::Int, my::Int, nqty::Int, bctypex::Integer, bctypey::Integer;
    xs::Vector{Float64}=Float64[],
    ys::Vector{Float64}=Float64[],
    fs::Array{Float64,3}=Array{Float64}(undef, 0, 0, 0),
    fsx=similar(fs), fsy=similar(fs), fsxy=similar(fs))
    h = Ref{Ptr{Cvoid}}()
    ccall((:bicube_c_create, libspline), Cvoid,
        (Int64, Int64, Int64, Ref{Ptr{Cvoid}}), mx, my, nqty, h)
    return BicubicSpline(h[], xs, ys, fs, mx, my, nqty, Int32(bctypex), Int32(bctypey), fsx, fsy, fsxy)
end

function _bicube_setup(xs::Vector{Float64}, ys::Vector{Float64}, fs::Array{Float64,3}, bctypex::Integer, bctypey::Integer)
    # xs -> Float64 (mx)
    # ys -> Float64 (my)
    # fs -> Float64 (mx, my, nqty)
    @assert length(xs) == size(fs, 1) "Length of xs must match number of rows in fs"
    @assert length(ys) == size(fs, 2) "Length of ys must match number of columns in fs"
    mx = length(xs) - 1
    my = length(ys) - 1
    nqty = size(fs, 3)

    bicube = BicubicSpline(mx, my, nqty, Int32(bctypex), Int32(bctypey); xs, ys, fs)

    ccall((:bicube_c_setup, libspline), Cvoid,
        (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
        bicube.handle, xs, ys, fs)

    ccall((:bicube_c_fit, libspline), Cvoid,
        (Ptr{Cvoid}, Int32, Int32, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
        bicube.handle, bicube.bctypex, bicube.bctypey, bicube._fsx, bicube._fsy, bicube._fsxy)

    return bicube
end

"""
BicubicSpline(xs, ys, fs;
bctypex::Union{String, Int}="not-a-knot", bctypey::Union{String, Int}="not-a-knot")

    ## Arguments:
    - `xs`: A vector of Float64 values representing the x-coordinates.
    - `ys`: A vector of Float64 values representing the y-coordinates.
    - `fs`: A 3D array of Float64 values representing the function values at the (x,y) coordinates.
    ## Keyword Arguments:
    - `bctypex`: An integer specifying the boundary condition type for x (Default is 4, not a knot)
    - `bctypey`: An integer specifying the boundary condition type for y  (Default is 4, not a knot)
    ## Returns:
    - A `BicubicSpline` object containing the spline handle, x-coordinates, y-coordinates,
    function values, number of x-coordinates, number of y-coordinates, number of quantities,
    and boundary condition types.
"""
function BicubicSpline(xs::Vector{Float64}, ys::Vector{Float64}, fs::Array{Float64,3};
    bctypex::Union{String,Int}="not-a-knot", bctypey::Union{String,Int}="not-a-knot")

    bctype_code_x = parse_bctype(bctypex)
    bctype_code_y = parse_bctype(bctypey)

    bicube = _bicube_setup(xs, ys, fs, Int32(bctype_code_x), Int32(bctype_code_y))

    finalizer(_destroy_bicubic_spline, bicube)

    return bicube
end

function call_bicube_c_eval(bicube, x, y, f)
    return ccall((:bicube_c_eval, libspline), Cvoid,
        (Ptr{Cvoid}, Float64, Float64, Ptr{Float64}),
        bicube.handle, x, y, f)
end
function call_bicube_c_eval(bicube, x, y, f, fx, fy)
    return ccall((:bicube_c_eval_deriv, libspline), Cvoid,
        (Ptr{Cvoid}, Float64, Float64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
        bicube.handle, x, y, f, fx, fy)
end
function call_bicube_c_eval(bicube, x, y, f, fx, fy, fxx, fxy, fyy)
    return ccall((:bicube_c_eval_deriv2, libspline), Cvoid,
        (Ptr{Cvoid}, Float64, Float64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
        bicube.handle, x, y, f, fx, fy, fxx, fxy, fyy)
end

"""
    bicube_eval!(bicube::BicubicSpline, x::Float64, y::Float64)

Evaluate the bicubic spline at a single point using in-place operations.

## Arguments:

  - `bicube`: A `BicubicSpline` object.
  - `x`: A Float64 value representing the x-coordinate to evaluate the bicubic spline at.
  - `y`: A Float64 value representing the y-coordinate to evaluate the bicubic spline at.

## Returns:

  - A vector of Float64 values representing the function values at the (x,y) coordinate.
    The result is stored in and returned from the spline's internal work array.
"""
function bicube_eval!(bicube::BicubicSpline, x::Float64, y::Float64)
    f = bicube._f
    call_bicube_c_eval(bicube, x, y, f)
    return f
end

"""
    bicube_deriv1!(bicube::BicubicSpline, x::Float64, y::Float64)

Evaluate the bicubic spline and its first derivatives at a single point using in-place operations.

## Arguments:

  - `bicube`: A `BicubicSpline` object.
  - `x`: A Float64 value representing the x-coordinate to evaluate the bicubic spline at.
  - `y`: A Float64 value representing the y-coordinate to evaluate the bicubic spline at.

## Returns:

  - `f`: A vector of Float64 values representing the function values at the (x,y) coordinate.
  - `fx`: A vector of Float64 values representing the first derivative with respect to x at the (x,y) coordinate.
  - `fy`: A vector of Float64 values representing the first derivative with respect to y at the (x,y) coordinate.
    Results are stored in and returned from the spline's internal work arrays.
"""
function bicube_deriv1!(bicube::BicubicSpline, x::Float64, y::Float64)
    f, fx, fy = bicube._f, bicube._fx, bicube._fy
    call_bicube_c_eval(bicube, x, y, f, fx, fy)
    return f, fx, fy
end

"""
    bicube_deriv2!(bicube::BicubicSpline, x::Float64, y::Float64)

Evaluate the bicubic spline and its first and second derivatives at a single point using in-place operations.

## Arguments:

  - `bicube`: A `BicubicSpline` object.
  - `x`: A Float64 value representing the x-coordinate to evaluate the bicubic spline at.
  - `y`: A Float64 value representing the y-coordinate to evaluate the bicubic spline at.

## Returns:

  - `f`: A vector of Float64 values representing the function values at the (x,y) coordinate.
  - `fx`: A vector of Float64 values representing the first derivative with respect to x at the (x,y) coordinate.
  - `fy`: A vector of Float64 values representing the first derivative with respect to y at the (x,y) coordinate.
  - `fxx`: A vector of Float64 values representing the second derivative with respect to x at the (x,y) coordinate.
  - `fxy`: A vector of Float64 values representing the mixed second derivative at the (x,y) coordinate.
  - `fyy`: A vector of Float64 values representing the second derivative with respect to y at the (x,y) coordinate.
    Results are stored in and returned from the spline's internal work arrays.
"""
function bicube_deriv2!(bicube::BicubicSpline, x::Float64, y::Float64)
    f, fx, fy, fxx, fxy, fyy = bicube._f, bicube._fx, bicube._fy, bicube._fxx, bicube._fxy, bicube._fyy
    call_bicube_c_eval(bicube, x, y, f, fx, fy, fxx, fxy, fyy)
    return f, fx, fy, fxx, fxy, fyy
end

"""
bicube_eval(bicube::BicubicSpline, xs::Vector{Float64}, ys::Vector{Float64}, derivs::Int=0)
## Arguments:
- `bicube`: A `BicubicSpline` object.
- `x`: A vector of Float64 values representing the x-coordinates to evaluate the bicubic spline at.
- `y`: A vector of Float64 values representing the y-coordinates to evaluate the bicubic spline at.
## Returns:
- Returns a 3D array of Float64 values where each slice corresponds to the function values at
the respective (x,y) coordinates in `x` and `y`.
- Depending on the derivatives requested, it may return additional arrays for the first, second, or third derivatives.
"""
function bicube_eval(bicube::BicubicSpline, xs::Vector{Float64}, ys::Vector{Float64}, derivs::Int=0)
    # xs -> Float64 (any length)
    # ys -> Float64 (any length)
    # Returns a matrix of Float64 (length(xs), length(ys), nqty)
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
            if derivs == 0
                call_bicube_c_eval(bicube, x, y, f)
                fs[i, j, :] .= f
            elseif derivs == 1
                call_bicube_c_eval(bicube, x, y, f, fx, fy)
                fs[i, j, :] .= f
                fsx[i, j, :] .= fx
                fsy[i, j, :] .= fy
            elseif derivs == 2
                call_bicube_c_eval(bicube, x, y, f, fx, fy, fxx, fxy, fyy)
                fs[i, j, :] .= f
                fsx[i, j, :] .= fx
                fsy[i, j, :] .= fy
                fsxx[i, j, :] .= fxx
                fsxy[i, j, :] .= fxy
                fsyy[i, j, :] .= fyy
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
