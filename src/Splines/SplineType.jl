#=
    SplineType.jl
    
    Simple data structure mirroring Fortran spline_type for compatibility.
    Works with the C API wrapper functions via opaque handles.
=#

"""
    SplineType(mx::Int, nqty::Int)

A Julia struct for holding spline data compatible with C API interface.

## Creation:
```julia
spl = SplineType(10, 2)  # 10 grid points, 2 quantities
spl.xs = range(0, 1, length=11) |> collect
spl.fs[:, 1] = y_data_1
spl.fs[:, 2] = y_data_2
spline_fit!(spl, "periodic")
y = spline_eval!(spl, 0.5)
roots = spline_roots(spl, 1)
```

## Fields:
  - `handle::Ptr{Cvoid}` - opaque handle to Fortran spline object
  - `_xs::Vector{Float64}` - grid points (length mx+1)
  - `_fs::Matrix{Float64}` - function values (mx+1 × nqty)
  - `mx::Int` - maximum grid index
  - `nqty::Int` - number of quantities
  - `bctype::Int32` - boundary condition type (1=natural, 2=periodic, 3=extrap, 4=not-a-knot)
  - `_fs1::Matrix{Float64}` - derivatives (computed by spline_fit!)
"""
mutable struct SplineType
    handle::Ptr{Cvoid}
    _xs::Vector{Float64}
    _fs::Matrix{Float64}
    mx::Int
    nqty::Int
    bctype::Int32
    _fs1::Matrix{Float64}
end

"""Create a new SplineType with allocated arrays."""
function SplineType(mx::Int, nqty::Int)
    # Create handle by calling spline_c_create
    h = Ref{Ptr{Cvoid}}()
    ccall((:spline_c_create, libspline), Cvoid, 
        (Int64, Int64, Ref{Ptr{Cvoid}}), 
        Int64(mx), Int64(nqty), h)
    
    xs = zeros(Float64, mx + 1)
    fs = zeros(Float64, mx + 1, nqty)
    fs1 = zeros(Float64, mx + 1, nqty)
    
    return SplineType(h[], xs, fs, mx, nqty, Int32(2), fs1)  # Default bctype=2 (periodic)
end

# Clean up when GC'd
function Base.finalizer(spl::SplineType)
    if spl.handle != C_NULL
        ccall((:spline_c_destroy, libspline), Cvoid, (Ptr{Cvoid},), spl.handle)
    end
end

# Make xs and fs accessible as properties (read-only view)
function Base.getproperty(spl::SplineType, sym::Symbol)
    if sym === :xs
        return getfield(spl, :_xs)
    elseif sym === :fs
        return getfield(spl, :_fs)
    elseif sym === :fs1
        return getfield(spl, :_fs1)
    else
        return getfield(spl, sym)
    end
end

function Base.setproperty!(spl::SplineType, sym::Symbol, val)
    if sym === :xs
        setfield!(spl, :_xs, val)
    elseif sym === :fs
        setfield!(spl, :_fs, val)
    elseif sym === :fs1
        setfield!(spl, :_fs1, val)
    else
        setfield!(spl, sym, val)
    end
end

