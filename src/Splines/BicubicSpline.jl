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

    itps::Vector{CubicInterpolantND{T,T,2}}

    # Work arrays for evaluation (not thread-safe)
    _f::Vector{T}
    _fx::Vector{T}
    _fy::Vector{T}
    _fxx::Vector{T}
    _fxy::Vector{T}
    _fyy::Vector{T}
end

"""
Boundary condition type for BicubicSpline: NaturalBC(), PeriodicBC(), or :extrap
"""
const BicubicBC = Union{NaturalBC,PeriodicBC,Symbol}

"""
    BicubicSpline(xs, ys, fs, bcx, bcy)

Create a 2D bicubic spline interpolator.

# Arguments

  - `xs::Vector{Float64}`: X-coordinates (sorted, length nx)
  - `ys::Vector{Float64}`: Y-coordinates (sorted, length ny)
  - `fs::Array{T,3}`: Function values (nx × ny × nqty)
  - `bcx`: Boundary condition in x - use `NaturalBC()`, `PeriodicBC()`, or `:extrap`
  - `bcy`: Boundary condition in y - use `NaturalBC()`, `PeriodicBC()`, or `:extrap`

# Example

```julia
xs = range(0, 1; length=10) |> collect
ys = range(0, 2π; length=20) |> collect
fs = zeros(10, 20, 2)
for i in 1:10, j in 1:20
    fs[i, j, 1] = sin(xs[i]) * cos(ys[j])
    fs[i, j, 2] = cos(xs[i]) * sin(ys[j])
end
bcs = BicubicSpline(xs, ys, fs, :extrap, PeriodicBC())
```
"""
function BicubicSpline(
    xs::Vector{Float64}, ys::Vector{Float64}, fs::Array{T,3},
    bcx::BicubicBC=:extrap, bcy::BicubicBC=:extrap
) where {T<:Union{Float64,ComplexF64}}

    nx_orig, ny_orig, nqty = size(fs)
    @assert length(xs) == nx_orig "xs length must match first dimension of fs"
    @assert length(ys) == ny_orig "ys length must match second dimension of fs"
    @assert nx_orig >= 2 && ny_orig >= 2 "Need at least 2 points in each dimension"
    @assert issorted(xs) "xs must be sorted in ascending order"
    @assert issorted(ys) "ys must be sorted in ascending order"

    nx = nx_orig
    ny = ny_orig

    # Allocate derivative arrays
    fsx = zeros(T, nx, ny, nqty)
    fsy = zeros(T, nx, ny, nqty)
    fsxy = zeros(T, nx, ny, nqty)

    # Set boundary conditions and extrapolation types
    bc2d = ( 
        bcx isa AbstractBC ? bcx : CubicFit(),
        bcy isa AbstractBC ? bcy : CubicFit()
    )   

    extrap2d = (
        bcx isa PeriodicBC ? :wrap : :extension,
        bcy isa PeriodicBC ? :wrap : :extension
    )

    itps = Vector{CubicInterpolantND{T,T,2}}(undef, nqty)

    for q in 1:nqty
        itps[q] = cubic_interp((xs, ys), fs[:,:,q]; extrap=extrap2d, bc=bc2d)
        fsx[:,:,q] .= itps[q].nodal_derivs.partials[2,:,:]
        fsy[:,:,q] .= itps[q].nodal_derivs.partials[3,:,:]
        fsxy[:,:,q] .= itps[q].nodal_derivs.partials[4,:,:]
    end

    # Store a copy of fs for the struct
    fs_stored = copy(fs)

    # Allocate work arrays
    _f = zeros(T, nqty)
    _fx = zeros(T, nqty)
    _fy = zeros(T, nqty)
    _fxx = zeros(T, nqty)
    _fxy = zeros(T, nqty)
    _fyy = zeros(T, nqty)

    BicubicSpline{T}(
        copy(xs), copy(ys), nqty, nx, ny,
        fs_stored, fsx, fsy, fsxy,
        itps,
        _f, _fx, _fy, _fxx, _fxy, _fyy
    )
end


"""
    evaluate!(bcs, x, y) -> Vector{T}

Evaluate the bicubic spline at point (x, y).

Returns a reference to the internal work array `_f`.
The result is valid until the next mutating call on this spline.
"""
function evaluate!(bcs::BicubicSpline{T}, x::Float64, y::Float64) where {T}
    query = (x, y)
    @inbounds for q in 1:bcs.nqty
        bcs._f[q] = bcs.itps[q](query)
    end

    return bcs._f
end

"""
    deriv1!(bcs, x, y) -> (f, fx, fy)

Evaluate bicubic spline and first derivatives at point (x, y).

Returns references to internal work arrays.
"""
function deriv1!(bcs::BicubicSpline{T}, x::Float64, y::Float64) where {T}
    query = (x,y)
    @inbounds for q in 1:bcs.nqty
        bcs._f[q] = bcs.itps[q](query)
        bcs._fx[q] = bcs.itps[q](query; deriv=(1,0))
        bcs._fy[q] = bcs.itps[q](query; deriv=(0,1))
    end

    return bcs._f, bcs._fx, bcs._fy
end

"""
    deriv2!(bcs, x, y) -> (f, fx, fy, fxx, fxy, fyy)

Evaluate bicubic spline through second derivatives at point (x, y).

Returns references to internal work arrays.
"""
function deriv2!(bcs::BicubicSpline{T}, x::Float64, y::Float64) where {T}
    query = (x,y)
    @inbounds for q in 1:bcs.nqty
        bcs._f[q] = bcs.itps[q](query)
        bcs._fx[q] = bcs.itps[q](query; deriv=(1,0))
        bcs._fy[q] = bcs.itps[q](query; deriv=(0,1))
        bcs._fxx[q] = bcs.itps[q](query; deriv=(2,0))
        bcs._fxy[q] = bcs.itps[q](query; deriv=(1,1))
        bcs._fyy[q] = bcs.itps[q](query; deriv=(0,2))
    end

    return bcs._f, bcs._fx, bcs._fy, bcs._fxx, bcs._fxy, bcs._fyy
end

"""
    empty_BicubicSpline()

Create an empty/placeholder BicubicSpline for type stability.
"""
function empty_BicubicSpline()
    # Need at least 4 points for extrap BC (endpoint derivative estimation)
    xs = collect(range(0.0, 1.0; length=4))
    ys = collect(range(0.0, 1.0; length=4))
    fs = zeros(Float64, 4, 4, 1)
    BicubicSpline(xs, ys, fs)
end

# Convenience evaluation (allocates, but thread-safe)
(bcs::BicubicSpline)(x::Float64, y::Float64) = copy(evaluate!(bcs, x, y))
