# Pure Julia CubicSpline implementation using Interpolations.jl

"""
    CubicSpline{T<:Union{Float64,ComplexF64}}

A cubic spline interpolation supporting multiple quantities (columns).
Uses Interpolations.jl under the hood.

## Fields (read-only via getproperty):
- `xs`: Grid points (x-coordinates)
- `fs`: Function values at grid points (mx+1 × nqty matrix)
- `fsi`: Integrated function values (filled by `spline_integrate!`)
- `fs1`: First derivatives at grid points (filled during construction)
"""
mutable struct CubicSpline{T<:Union{Float64,ComplexF64}}
    _xs::Vector{Float64}
    _fs::Matrix{T}
    mx::Int
    nqty::Int
    bctype::Int32
    _fsi::Matrix{T}  # Integrated function values at gridpoints
    _fs1::Matrix{T}  # First derivatives at gridpoints
    _f::Vector{T}    # Work array for evaluation
    _f1::Vector{T}   # Work array for 1st derivative
    _f2::Vector{T}   # Work array for 2nd derivative
    _f3::Vector{T}   # Work array for 3rd derivative
    _interps::Vector{Any}  # Vector of interpolation objects (one per quantity)
end

@expose_fields CubicSpline xs fs fsi fs1

"""
    _vector_to_range(xs::Vector{Float64})

Convert a uniformly-spaced vector to a range for use with Interpolations.jl scale().
Returns a StepRangeLen that matches the vector's start, stop, and length.
"""
function _vector_to_range(xs::Vector{Float64})
    n = length(xs)
    @assert n >= 2 "Need at least 2 points for interpolation"
    return range(xs[1]; stop=xs[end], length=n)
end

"""
    CubicExtrapolatingInterp

Wrapper that provides cubic polynomial extrapolation outside the interpolation range.
Stores the scaled interpolation object and boundary derivatives for Taylor expansion.
"""
struct CubicExtrapolatingInterp{I}
    itp::I                  # Scaled interpolation (no extrapolation wrapper)
    x_min::Float64
    x_max::Float64
    # Derivatives at left boundary (x_min): f, f', f'', f'''
    f_left::Float64
    df_left::Float64
    d2f_left::Float64
    d3f_left::Float64
    # Derivatives at right boundary (x_max)
    f_right::Float64
    df_right::Float64
    d2f_right::Float64
    d3f_right::Float64
    is_periodic::Bool
end

# Make CubicExtrapolatingInterp callable
function (e::CubicExtrapolatingInterp)(x::Float64)
    if e.is_periodic
        # Wrap x into [x_min, x_max]
        period = e.x_max - e.x_min
        x_wrapped = e.x_min + mod(x - e.x_min, period)
        return e.itp(x_wrapped)
    elseif x < e.x_min
        # Taylor expansion at left boundary
        dx = x - e.x_min
        return e.f_left + e.df_left*dx + 0.5*e.d2f_left*dx^2 + (1/6)*e.d3f_left*dx^3
    elseif x > e.x_max
        # Taylor expansion at right boundary
        dx = x - e.x_max
        return e.f_right + e.df_right*dx + 0.5*e.d2f_right*dx^2 + (1/6)*e.d3f_right*dx^3
    else
        return e.itp(x)
    end
end

"""
    _compute_boundary_derivs(itp, x_min, x_max)

Compute derivatives at boundaries for cubic extrapolation.
Returns (f, f', f'', f''') at x_min and x_max.

Note: We sample slightly inside the range to get better derivative estimates,
since BSpline boundary conditions may constrain derivatives at exact boundaries.
"""
function _compute_boundary_derivs(itp, x_min::Float64, x_max::Float64)
    span = x_max - x_min

    # Sample a small fraction inside to avoid boundary effects
    # Use 1% of the span or one step if grid is coarse
    offset = 0.01 * span

    # Left boundary - sample just inside
    x_left = x_min + offset
    f_at_offset = itp(x_left)
    df_at_offset = Interpolations.gradient(itp, x_left)[1]
    d2f_at_offset = Interpolations.hessian(itp, x_left)[1]

    # Third derivative by finite differences
    h = 0.001 * span
    d2f_plus = Interpolations.hessian(itp, x_left + h)[1]
    d2f_minus = Interpolations.hessian(itp, x_left - h)[1]
    d3f_left = (d2f_plus - d2f_minus) / (2h)

    # Extrapolate back to boundary: f(x_min) = f(x_left) - df*offset + 0.5*d2f*offset^2 - (1/6)*d3f*offset^3
    f_left = f_at_offset - df_at_offset*offset + 0.5*d2f_at_offset*offset^2 - (1/6)*d3f_left*offset^3
    df_left = df_at_offset - d2f_at_offset*offset + 0.5*d3f_left*offset^2
    d2f_left = d2f_at_offset - d3f_left*offset

    # Right boundary - sample just inside
    x_right = x_max - offset
    f_at_offset = itp(x_right)
    df_at_offset = Interpolations.gradient(itp, x_right)[1]
    d2f_at_offset = Interpolations.hessian(itp, x_right)[1]

    d2f_plus = Interpolations.hessian(itp, x_right + h)[1]
    d2f_minus = Interpolations.hessian(itp, x_right - h)[1]
    d3f_right = (d2f_plus - d2f_minus) / (2h)

    # Extrapolate forward to boundary
    f_right = f_at_offset + df_at_offset*offset + 0.5*d2f_at_offset*offset^2 + (1/6)*d3f_right*offset^3
    df_right = df_at_offset + d2f_at_offset*offset + 0.5*d3f_right*offset^2
    d2f_right = d2f_at_offset + d3f_right*offset

    return (f_left, df_left, d2f_left, d3f_left,
            f_right, df_right, d2f_right, d3f_right)
end

"""
    _create_cubic_extrapolating_interp(itp, x_min, x_max, is_periodic)

Create a CubicExtrapolatingInterp wrapper around a scaled interpolation.
"""
function _create_cubic_extrapolating_interp(itp, x_min::Float64, x_max::Float64, is_periodic::Bool)
    if is_periodic
        # For periodic, just wrap - no need for boundary derivatives
        return CubicExtrapolatingInterp(itp, x_min, x_max,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, true)
    else
        f_l, df_l, d2f_l, d3f_l, f_r, df_r, d2f_r, d3f_r = _compute_boundary_derivs(itp, x_min, x_max)
        return CubicExtrapolatingInterp(itp, x_min, x_max,
            f_l, df_l, d2f_l, d3f_l, f_r, df_r, d2f_r, d3f_r, false)
    end
end

"""
    _get_bspline_bc(n::Int)

Get appropriate BSpline boundary condition based on grid size.
Free(OnGrid()) gives best accuracy but requires n >= 4.
Falls back to Line(OnGrid()) for smaller grids.
"""
function _get_bspline_bc(n::Int)
    if n >= 4
        return Free(OnGrid())
    else
        return Line(OnGrid())
    end
end

"""
    _create_cubic_interpolations(xs, fs, bctype, ::Type{T}) where T

Create interpolation objects for each quantity column using BSpline cubic interpolation.
Uses interpolate + scale with range-based scaling, and custom cubic extrapolation.
"""
function _create_cubic_interpolations(xs::Vector{Float64}, fs::Matrix{T}, bctype::Int32, ::Type{T}) where T
    nqty = size(fs, 2)
    interps = Vector{Any}(undef, nqty)

    is_periodic = (bctype == 2)
    xs_range = _vector_to_range(xs)
    x_min, x_max = xs[1], xs[end]
    n = length(xs)
    bc = _get_bspline_bc(n)

    for i in 1:nqty
        if T === ComplexF64
            # Interpolate real and imaginary parts separately
            real_itp = interpolate(real.(fs[:, i]), BSpline(Cubic(bc)))
            imag_itp = interpolate(imag.(fs[:, i]), BSpline(Cubic(bc)))
            real_scaled = scale(real_itp, xs_range)
            imag_scaled = scale(imag_itp, xs_range)
            real_extrap = _create_cubic_extrapolating_interp(real_scaled, x_min, x_max, is_periodic)
            imag_extrap = _create_cubic_extrapolating_interp(imag_scaled, x_min, x_max, is_periodic)
            interps[i] = (real_extrap, imag_extrap)
        else
            itp = interpolate(fs[:, i], BSpline(Cubic(bc)))
            scaled = scale(itp, xs_range)
            interps[i] = _create_cubic_extrapolating_interp(scaled, x_min, x_max, is_periodic)
        end
    end

    return interps
end

"""
    _eval_spline_value(interps, x, qty_idx, ::Type{T}) where T

Evaluate a single interpolation at point x.
"""
function _eval_spline_value(interps, x::Float64, qty_idx::Int, ::Type{T}) where T
    itp = interps[qty_idx]
    if T === ComplexF64
        return complex(itp[1](x), itp[2](x))
    else
        return itp(x)
    end
end

"""
    _eval_gradient(itp::CubicExtrapolatingInterp, x)

Evaluate gradient for CubicExtrapolatingInterp, using stored derivatives for extrapolation.
"""
function _eval_gradient(e::CubicExtrapolatingInterp, x::Float64)
    if e.is_periodic
        period = e.x_max - e.x_min
        x_wrapped = e.x_min + mod(x - e.x_min, period)
        return Interpolations.gradient(e.itp, x_wrapped)[1]
    elseif x < e.x_min
        # Derivative of Taylor expansion: f' + f''*dx + 0.5*f'''*dx^2
        dx = x - e.x_min
        return e.df_left + e.d2f_left*dx + 0.5*e.d3f_left*dx^2
    elseif x > e.x_max
        dx = x - e.x_max
        return e.df_right + e.d2f_right*dx + 0.5*e.d3f_right*dx^2
    else
        return Interpolations.gradient(e.itp, x)[1]
    end
end

"""
    _eval_hessian(itp::CubicExtrapolatingInterp, x)

Evaluate hessian for CubicExtrapolatingInterp, using stored derivatives for extrapolation.
"""
function _eval_hessian(e::CubicExtrapolatingInterp, x::Float64)
    if e.is_periodic
        period = e.x_max - e.x_min
        x_wrapped = e.x_min + mod(x - e.x_min, period)
        return Interpolations.hessian(e.itp, x_wrapped)[1]
    elseif x < e.x_min
        # Second derivative of Taylor expansion: f'' + f'''*dx
        dx = x - e.x_min
        return e.d2f_left + e.d3f_left*dx
    elseif x > e.x_max
        dx = x - e.x_max
        return e.d2f_right + e.d3f_right*dx
    else
        return Interpolations.hessian(e.itp, x)[1]
    end
end

"""
    _eval_third_deriv(itp::CubicExtrapolatingInterp, x)

Evaluate third derivative for CubicExtrapolatingInterp.
"""
function _eval_third_deriv(e::CubicExtrapolatingInterp, x::Float64)
    if e.is_periodic
        period = e.x_max - e.x_min
        x_wrapped = e.x_min + mod(x - e.x_min, period)
        # Use finite differences on hessian
        h = max(1e-6, abs(x_wrapped) * 1e-6)
        h2_plus = Interpolations.hessian(e.itp, x_wrapped + h)[1]
        h2_minus = Interpolations.hessian(e.itp, x_wrapped - h)[1]
        return (h2_plus - h2_minus) / (2h)
    elseif x < e.x_min
        # Third derivative is constant in cubic extrapolation
        return e.d3f_left
    elseif x > e.x_max
        return e.d3f_right
    else
        # Inside range: use finite differences on hessian
        h = max(1e-6, abs(x) * 1e-6)
        h2_plus = Interpolations.hessian(e.itp, x + h)[1]
        h2_minus = Interpolations.hessian(e.itp, x - h)[1]
        return (h2_plus - h2_minus) / (2h)
    end
end

"""
    _eval_spline_deriv(interps, x, qty_idx, deriv_order, ::Type{T}) where T

Evaluate derivative of interpolation at point x.
"""
function _eval_spline_deriv(interps, x::Float64, qty_idx::Int, deriv_order::Int, ::Type{T}) where T
    itp = interps[qty_idx]

    if deriv_order == 0
        if T === ComplexF64
            return complex(itp[1](x), itp[2](x))
        else
            return itp(x)
        end
    elseif deriv_order == 1
        if T === ComplexF64
            grad_real = _eval_gradient(itp[1], x)
            grad_imag = _eval_gradient(itp[2], x)
            return complex(grad_real, grad_imag)
        else
            return _eval_gradient(itp, x)
        end
    elseif deriv_order == 2
        if T === ComplexF64
            hess_real = _eval_hessian(itp[1], x)
            hess_imag = _eval_hessian(itp[2], x)
            return complex(hess_real, hess_imag)
        else
            return _eval_hessian(itp, x)
        end
    elseif deriv_order == 3
        if T === ComplexF64
            d3_real = _eval_third_deriv(itp[1], x)
            d3_imag = _eval_third_deriv(itp[2], x)
            return complex(d3_real, d3_imag)
        else
            return _eval_third_deriv(itp, x)
        end
    else
        error("Derivative order $deriv_order not supported")
    end
end

"""
    _compute_derivatives_at_grid(xs, nqty, interps, ::Type{T}) where T

Compute first derivatives at grid points.
"""
function _compute_derivatives_at_grid(xs::Vector{Float64}, nqty::Int, interps, ::Type{T}) where T
    mx = length(xs) - 1
    fs1 = Matrix{T}(undef, mx + 1, nqty)

    for j in 1:nqty
        for i in 1:(mx + 1)
            fs1[i, j] = _eval_spline_deriv(interps, xs[i], j, 1, T)
        end
    end

    return fs1
end

function CubicSpline(xs::Vector{Float64}, fs::Vector{<:Union{Float64,ComplexF64}}, bctype::Int32)
    @assert length(xs) == length(fs) "Length of xs must match length of fs"
    fs_matrix = reshape(fs, length(xs), 1)
    return CubicSpline(xs, fs_matrix, bctype)
end

function CubicSpline(xs::Vector{Float64}, fs::Matrix{T}, bctype::Int32) where {T<:Union{Float64,ComplexF64}}
    @assert length(xs) == size(fs, 1) "Length of xs must match number of rows in fs"

    mx = length(xs) - 1
    nqty = size(fs, 2)

    # Create interpolation objects
    interps = _create_cubic_interpolations(xs, fs, bctype, T)

    # Compute derivatives at grid points
    fs1 = _compute_derivatives_at_grid(xs, nqty, interps, T)

    # Initialize other arrays
    fsi = Matrix{T}(undef, mx + 1, nqty)
    f = Vector{T}(undef, nqty)
    f1 = Vector{T}(undef, nqty)
    f2 = Vector{T}(undef, nqty)
    f3 = Vector{T}(undef, nqty)

    return CubicSpline{T}(xs, fs, mx, nqty, bctype, fsi, fs1, f, f1, f2, f3, interps)
end

"""
    CubicSpline(xs, fs; bctype="not-a-knot")

Create a cubic spline interpolation.

## Arguments:
- `xs`: A vector of Float64 values representing the x-coordinates.
- `fs`: A vector or matrix of Float64/ComplexF64 values representing the function values.

## Keyword Arguments:
- `bctype`: Boundary condition type for the cubic spline.
    - 1 or "natural": Natural spline
    - 2 or "periodic": Periodic spline
    - 3 or "extrap": Extrapolated spline
    - 4 or "not-a-knot": Not-a-knot spline (default)

## Returns:
- A `CubicSpline` object ready for evaluation.
"""
function CubicSpline(xs::Vector{Float64}, fs::Union{Vector{T},Matrix{T}};
    bctype::Union{String,Int}="not-a-knot") where {T<:Union{Float64,ComplexF64}}

    bctype_code = parse_bctype(bctype)

    if T === ComplexF64 && bctype_code == 1
        error("Complex spline doesn't have natural spline. (bctype = 1/natural)")
    end

    return CubicSpline(xs, fs isa Vector ? reshape(fs, :, 1) : fs, Int32(bctype_code))
end

"""
    spline_eval!(spline::CubicSpline{T}, x::Float64) where {T}

Evaluate the cubic spline at a single point using in-place operations.
"""
function spline_eval!(spline::CubicSpline{T}, x::Float64) where {T<:Union{Float64,ComplexF64}}
    @assert !isempty(spline._interps) "CubicSpline has not been initialized"
    f = spline._f
    for i in 1:spline.nqty
        f[i] = _eval_spline_value(spline._interps, x, i, T)
    end
    return f
end

"""
    spline_deriv1!(spline::CubicSpline{T}, x::Float64) where {T}

Evaluate the cubic spline and its first derivative at a single point.
"""
function spline_deriv1!(spline::CubicSpline{T}, x::Float64) where {T<:Union{Float64,ComplexF64}}
    @assert !isempty(spline._interps) "CubicSpline has not been initialized"
    f, f1 = spline._f, spline._f1
    for i in 1:spline.nqty
        f[i] = _eval_spline_deriv(spline._interps, x, i, 0, T)
        f1[i] = _eval_spline_deriv(spline._interps, x, i, 1, T)
    end
    return f, f1
end

"""
    spline_deriv2!(spline::CubicSpline{T}, x::Float64) where {T}

Evaluate the cubic spline and its first two derivatives at a single point.
"""
function spline_deriv2!(spline::CubicSpline{T}, x::Float64) where {T<:Union{Float64,ComplexF64}}
    @assert !isempty(spline._interps) "CubicSpline has not been initialized"
    f, f1, f2 = spline._f, spline._f1, spline._f2
    for i in 1:spline.nqty
        f[i] = _eval_spline_deriv(spline._interps, x, i, 0, T)
        f1[i] = _eval_spline_deriv(spline._interps, x, i, 1, T)
        f2[i] = _eval_spline_deriv(spline._interps, x, i, 2, T)
    end
    return f, f1, f2
end

"""
    spline_deriv3!(spline::CubicSpline{T}, x::Float64) where {T}

Evaluate the cubic spline and its first three derivatives at a single point.
"""
function spline_deriv3!(spline::CubicSpline{T}, x::Float64) where {T<:Union{Float64,ComplexF64}}
    @assert !isempty(spline._interps) "CubicSpline has not been initialized"
    f, f1, f2, f3 = spline._f, spline._f1, spline._f2, spline._f3
    for i in 1:spline.nqty
        f[i] = _eval_spline_deriv(spline._interps, x, i, 0, T)
        f1[i] = _eval_spline_deriv(spline._interps, x, i, 1, T)
        f2[i] = _eval_spline_deriv(spline._interps, x, i, 2, T)
        f3[i] = _eval_spline_deriv(spline._interps, x, i, 3, T)
    end
    return f, f1, f2, f3
end

"""
    spline_eval(spline::CubicSpline{T}, xs::Vector{Float64}, derivs::Int=0) where {T}

Batch evaluation of the cubic spline at multiple points.
"""
function spline_eval(spline::CubicSpline{T}, xs::Vector{Float64}, derivs::Int=0) where {T<:Union{Float64,ComplexF64}}
    @assert !isempty(spline._interps) "CubicSpline has not been initialized"
    @assert (derivs in 0:3) "Invalid number of derivatives requested: $derivs. Must be 0, 1, 2, or 3."

    n = length(xs)
    fs = Matrix{T}(undef, n, spline.nqty)
    f = spline._f
    if derivs > 0
        fs1 = similar(fs)
        f1 = spline._f1
    end
    if derivs > 1
        fs2 = similar(fs)
        f2 = spline._f2
    end
    if derivs > 2
        fs3 = similar(fs)
        f3 = spline._f3
    end

    for (i, x) in enumerate(xs)
        for j in 1:spline.nqty
            fs[i, j] = _eval_spline_deriv(spline._interps, x, j, 0, T)
            if derivs > 0
                fs1[i, j] = _eval_spline_deriv(spline._interps, x, j, 1, T)
            end
            if derivs > 1
                fs2[i, j] = _eval_spline_deriv(spline._interps, x, j, 2, T)
            end
            if derivs > 2
                fs3[i, j] = _eval_spline_deriv(spline._interps, x, j, 3, T)
            end
        end
    end

    if derivs == 0
        return fs
    elseif derivs == 1
        return fs, fs1
    elseif derivs == 2
        return fs, fs1, fs2
    elseif derivs == 3
        return fs, fs1, fs2, fs3
    end
end

"""
    spline_eval!(f, spline, x; derivs=0, f1=nothing, f2=nothing, f3=nothing)

In-place evaluation of `CubicSpline` at a single point `x`.
"""
function spline_eval!(f::Vector{T}, spline::CubicSpline{T}, x::Float64;
    derivs::Int=0, f1=nothing, f2=nothing, f3=nothing) where {T<:Union{Float64,ComplexF64}}
    @assert !isempty(spline._interps) "CubicSpline has not been initialized"
    @assert (derivs in 0:3) "Invalid number of derivatives requested: $derivs"

    for i in 1:spline.nqty
        f[i] = _eval_spline_deriv(spline._interps, x, i, 0, T)
        if derivs >= 1 && f1 !== nothing
            f1[i] = _eval_spline_deriv(spline._interps, x, i, 1, T)
        end
        if derivs >= 2 && f2 !== nothing
            f2[i] = _eval_spline_deriv(spline._interps, x, i, 2, T)
        end
        if derivs >= 3 && f3 !== nothing
            f3[i] = _eval_spline_deriv(spline._interps, x, i, 3, T)
        end
    end
    return nothing
end

"""
    spline_integrate!(spline::CubicSpline{T}) where {T}

Compute cumulative integrals at grid points.
Updates `spline._fsi` in place so that `spline.fsi[i, :]` equals
`∫_{xs[1]}^{xs[i]} f(x) dx` for each component.
"""
function spline_integrate!(spline::CubicSpline{T}) where {T<:Union{Float64,ComplexF64}}
    @assert !isempty(spline._interps) "CubicSpline has not been initialized"

    xs = spline._xs
    fsi = spline._fsi

    # Initialize first point to zero
    fsi[1, :] .= zero(T)

    # Use Simpson's rule with spline values for integration
    for i in 2:(spline.mx + 1)
        h = xs[i] - xs[i-1]
        x_mid = (xs[i-1] + xs[i]) / 2
        for j in 1:spline.nqty
            f_left = _eval_spline_value(spline._interps, xs[i-1], j, T)
            f_mid = _eval_spline_value(spline._interps, x_mid, j, T)
            f_right = _eval_spline_value(spline._interps, xs[i], j, T)
            # Simpson's rule: (h/6) * (f_left + 4*f_mid + f_right)
            fsi[i, j] = fsi[i-1, j] + (h / 6) * (f_left + 4 * f_mid + f_right)
        end
    end

    return nothing
end

"""
    empty_CubicSpline(::Type{T}=ComplexF64) where {T<:Union{Float64,ComplexF64}}

Create a minimal, type-stable placeholder spline.
"""
function empty_CubicSpline(::Type{T}=ComplexF64) where {T<:Union{Float64,ComplexF64}}
    xs = Float64[]
    fs = Matrix{T}(undef, 0, 0)
    fsi = Matrix{T}(undef, 0, 0)
    fs1 = Matrix{T}(undef, 0, 0)
    f = Vector{T}(undef, 0)
    f1 = Vector{T}(undef, 0)
    f2 = Vector{T}(undef, 0)
    f3 = Vector{T}(undef, 0)
    interps = Vector{Any}()
    return CubicSpline{T}(xs, fs, 0, 0, zero(Int32), fsi, fs1, f, f1, f2, f3, interps)
end
