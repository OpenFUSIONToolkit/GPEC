mutable struct FourierSpline
    handle::Ptr{Cvoid}
    _xs::Vector{Float64}
    _ys::Vector{Float64}
    _fs::Array{Float64,3}
    mx::Int
    my::Int
    mband::Int
    nqty::Int
    bctype::Int32
    fit_method::Int32
    cs::CubicSpline{ComplexF64}
end

@expose_fields FourierSpline xs ys fs


function _destroy_fspline(fspline::FourierSpline)
    if fspline.handle != C_NULL
        ccall((:fspline_c_destroy, libspline), Cvoid, (Ptr{Cvoid},), fspline.handle)
        Core.setfield!(fspline, :handle, C_NULL)
    end
end

function FourierSpline(mx::Int, my::Int, mband::Int, nqty::Int, bctype::Int32, fit_method::Int32)
    h = Ref{Ptr{Cvoid}}()
    ccall((:fspline_c_create, libspline), Cvoid,
        (Int64, Int64, Int64, Int64, Ref{Ptr{Cvoid}}),
        mx, my, mband, nqty, h)

    handle = h[]
    if handle == C_NULL
        error("Failed to create fspline handle in Fortran library.")
    end

    # Return a partially initialized struct with empty data arrays
    return FourierSpline(handle, Float64[], Float64[],
        Array{Float64,3}(undef, 0, 0, 0),
        mx, my, mband, nqty, bctype, fit_method, nothing)
end

function _fspline_setup(xs::Vector{Float64}, ys::Vector{Float64}, fs::Array{Float64,3},
    mband::Int, bctype::Int32, fit_method::Int32, fit_flag::Bool)

    mx = length(xs) - 1
    my = length(ys) - 1
    nqty = size(fs, 3)

    h = Ref{Ptr{Cvoid}}()
    ccall((:fspline_c_create, libspline), Cvoid,
        (Int64, Int64, Int64, Int64, Ref{Ptr{Cvoid}}),
        mx, my, mband, nqty, h)

    handle = h[]
    ccall((:fspline_c_setup, libspline), Cvoid,
        (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
        handle, xs, ys, fs)
    if fit_method == 1
        ccall((:fspline_c_fit_1, libspline), Cvoid,
            (Ptr{Cvoid}, Int32, Bool),
            handle, bctype, fit_flag)
    elseif fit_method == 2
        ccall((:fspline_c_fit_2, libspline), Cvoid,
            (Ptr{Cvoid}, Int32, Bool),
            handle, bctype, fit_flag)
    else
        error("Internal error: Invalid fit_method passed to _fspline_setup.")
    end

    # 1. Get the handle to the embedded cspline object

    cs_handle_ref = Ref{Ptr{Cvoid}}()
    ccall((:fspline_c_get_cspline_handle, libspline), Cvoid,
        (Ptr{Cvoid}, Ref{Ptr{Cvoid}}),
        handle, cs_handle_ref)
    cs_handle = cs_handle_ref[]

    # 2. Get the dimensions and data for the cspline
    cs_mx = mx
    cs_nqty = (mband + 1) * nqty
    cs_fs = Matrix{ComplexF64}(undef, cs_mx + 1, cs_nqty)
    ccall((:fspline_c_get_cspline_fs, libspline), Cvoid,
        (Ptr{Cvoid}, Ptr{ComplexF64}),
        handle, cs_fs)
    unmanaged_cspline = CubicSpline(cs_handle, xs, cs_fs, cs_mx, cs_nqty)

    # 4. Assign it to the parent object
    return FourierSpline(handle, xs, ys, fs, mx, my, mband, nqty, bctype, fit_method, unmanaged_cspline)
end

"""
FourierSpline(xs::Vector{Float64}, ys::Vector{Float64}, fs::Array{Float64, 3},
mband::Int; bctype::Union{String, Int}="not-a-knot", fit_method::Int=1, fit_flag::Bool=true)

    Creates and fits a function of two variables, f(x, y), to a cubic spline
    in the x-direction and a Fourier series in the y-direction. The y-direction
    is assumed to be periodic.

    ## Arguments:
    - `xs`: Vector of x-coordinates (length `mx`+1).
    - `ys`: Vector of y-coordinates (length `my`+1, periodic direction).
    - `fs`: 3D array of function values with dimensions (`mx`+1, `my`+1, `nqty`).
    - `mband`: Number of Fourier modes (harmonics) to keep, from 0 to `mband`.

    ## Keyword Arguments:
    - `bctype`: Boundary condition type for the cubic spline in `x`.
        - 1: Natural spline (default)
        - 2: Periodic spline
        - 3: Extrapolated spline
        - 4: "Not-a-knot" spline
    - `fit_method`: Algorithm for computing Fourier coefficients.
        - 1: Integration method (for non-uniform `y` grids).
        - 2: Fast Fourier Transform (FFT) method (requires `length(ys)-1` to be a power of 2).

    ## Returns:
    - A `FourierSpline` object ready for evaluation.
"""
function FourierSpline(xs::Vector{Float64}, ys::Vector{Float64}, fs::Array{Float64,3},
    mband::Int; bctype::Union{String,Int}="not-a-knot", fit_method::Int=1, fit_flag::Bool=true)

    @assert length(xs) == size(fs, 1) "Length of xs must match number of rows in fs"
    @assert length(ys) == size(fs, 2) "Length of ys must match number of columns in fs"

    @assert fit_method in (1, 2) "Invalid `fit_method`. Choose 1 or 2."
    if fit_method == 2
        my = length(ys) - 1
        @assert ispow2(my) "For `fit_method=2` (FFT), `length(ys)-1` (is $my) must be a power of 2."
        @assert 2 * mband <= my - 1 "For `fit_method=2` (FFT), `2*mband` must not be greater than `(length(ys)-1) - 1`. Got 2*$(mband) > $(my-1)."
    end

    bctype_code = parse_bctype(bctype)

    @assert bctype_code != 1 "Fourier spline doesn't have natural spline. (bctype = 1/natural)"
    # Call the internal setup function that creates and fits the spline
    fspline = _fspline_setup(xs, ys, fs, mband, Int32(bctype_code), Int32(fit_method), fit_flag)

    # Add a finalizer to ensure the Fortran object is deallocated when the Julia object is garbage collected.
    finalizer(_destroy_fspline, fspline)

    return fspline
end

function call_fspline_c_eval(spl, x, y, f)
    return ccall((:fspline_c_eval, libspline), Cvoid,
        # (handle,      x,       y,      f_out)
        (Ptr{Cvoid}, Float64, Float64, Ptr{Float64}),
        spl.handle, x, y, f)
end

function call_fspline_c_eval_deriv(spl, x, y, f, fx, fy)
    return ccall((:fspline_c_eval_deriv, libspline), Cvoid,
        # (handle,      x,       y,      f_out,      fx_out,      fy_out)
        (Ptr{Cvoid}, Float64, Float64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
        spl.handle, x, y, f, fx, fy)
end

function call_fspline_c_eval_deriv2(spl, x, y, f, fx, fy, fxx, fxy, fyy)
    return ccall((:fspline_c_eval_deriv2, libspline), Cvoid,
        # (handle,      x,       y,      f_out,      fx_out,      fy_out,      fxx_out,     fxy_out,     fyy_out)
        (Ptr{Cvoid}, Float64, Float64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
        spl.handle, x, y, f, fx, fy, fxx, fxy, fyy)
end

"""
fspline_eval(fspline::FourierSpline, x::Float64, y::Float64, derivs::Int=0)

    Evaluates a fitted Fourier-Spline at given coordinates.

    ## Arguments:
    - `fspline`: A `FourierSpline` object.
    - `x`: A `Float64` x-coordinate.
    - `y`: A `Float64` y-coordinate.

    ## Returns:
    - If `x`, `y` are scalars: A tuple containing the function value(s) and any requested derivatives. If nqty=1, the results are scalars, otherwise they are vectors.
"""
function fspline_eval(fspline::FourierSpline, x::Float64, y::Float64, derivs::Int=0)

    @assert (derivs in 0:2) "Invalid number of derivatives requested: $derivs. Must be 0, 1, or 2."

    f = Vector{Float64}(undef, fspline.nqty)
    if derivs == 0
        call_fspline_c_eval(fspline, x, y, f)
        results = f

    elseif derivs == 1
        fx, fy = similar(f), similar(f)
        call_fspline_c_eval_deriv(fspline, x, y, f, fx, fy)
        results = (f, fx, fy)

    elseif derivs == 2
        fx, fy = similar(f), similar(f)
        fxx, fxy, fyy = similar(f), similar(f), similar(f)
        call_fspline_c_eval_deriv2(fspline, x, y, f, fx, fy, fxx, fxy, fyy)
        results = (f, fx, fy, fxx, fxy, fyy)

    end

    if fspline.nqty == 1
        if isa(results, Tuple)
            return map(vec -> vec[1], results)
        else
            return results[1]
        end
    end

    return results
end


"""
fspline_eval(fspline::FourierSpline, xs::Vector{Float64}, ys::Vector{Float64}, derivs::Int=0)

    Evaluates a fitted Fourier-Spline at given coordinates.

    ## Arguments:
    - `fspline`: A `FourierSpline` object.
    - `x`: A `Vector{Float64}` of x-coordinates.
    - `y`: A `Vector{Float64}` of y-coordinates.

    ## Returns:
    - If `x`, `y` are vectors: A tuple of 3D arrays for the function values and derivatives on the grid defined by `x` and `y`.
"""
function fspline_eval(fspline::FourierSpline, xs::Vector{Float64}, ys::Vector{Float64}, derivs::Int=0)
    # xs -> Float64 (any length)
    # ys -> Float64 (any length)
    # Returns a matrix of Float64 (length(xs), length(ys), nqty)

    @assert (derivs in 0:2) "Invalid number of derivatives requested: $derivs. Must be 0, 1, or 2."

    n = length(xs)
    m = length(ys)
    fs = Array{Float64}(undef, n, m, fspline.nqty)
    f = Vector{Float64}(undef, fspline.nqty)
    if derivs > 0
        fsx, fsy = similar(fs), similar(fs)
        fx, fy = similar(f), similar(f)
    end
    if derivs > 1
        fsxx, fsxy, fsyy = similar(fs), similar(fs), similar(fs)
        fxx, fxy, fyy = similar(f), similar(f), similar(f)
    end
    for (i, x) in enumerate(xs)
        for (j, y) in enumerate(ys)
            if derivs == 0
                call_fspline_c_eval(fspline, x, y, f)
                fs[i, j, :] .= f
            elseif derivs == 1
                call_fspline_c_eval_deriv(fspline, x, y, f, fx, fy)
                fs[i, j, :] .= f
                fsx[i, j, :] .= fx
                fsy[i, j, :] .= fy
            elseif derivs == 2
                call_fspline_c_eval_deriv2(fspline, x, y, f, fx, fy, fxx, fxy, fyy)
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