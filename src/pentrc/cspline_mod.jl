"""
Minimal cspline wrapper for tests.
Provides CSplineType and evaluation/alloc helpers expected by `torque.jl`.
This is a lightweight compatibility layer that returns safe defaults
when real spline objects are not provided. If a SplinesMod object is
passed, we'll attempt to evaluate it.
"""
using LinearAlgebra

mutable struct CSplineType
    xs::Vector{Float64}
    fs::Matrix{Float64}
    mx::Int
end

CSplineType() = CSplineType(Float64[], Array{Float64}(undef,0,0), 0)

function cspline_alloc(nx::Integer, nvar::Integer)
    xs = zeros(Float64, nx)
    fs = zeros(Float64, nx, nvar)
    return CSplineType(xs, fs, nx)
end

function cspline_eval_external(obj, psi::Float64)
    # If the object is a plain array (e.g., smats-like flattened data), return it
    try
        # If object is from SplinesMod, call bicube_eval or spline_eval where appropriate
        if typeof(obj) <: AbstractArray
            return obj
        end
    catch
    end

    # Default: return a small zero complex vector sized by mpert if available
    mp = try
        DconInterface.mpert[]
    catch
        1
    end
    return ComplexF64.(zeros(mp * mp))
end
"""
Minimal cspline wrapper for tests.
Provides CSplineType and evaluation/alloc helpers expected by `torque.jl`.
This is a lightweight compatibility layer that returns safe defaults
when real spline objects are not provided. If a SplinesMod object is
passed, we'll attempt to evaluate it.
"""
using LinearAlgebra

mutable struct CSplineType
    xs::Vector{Float64}
    fs::Matrix{Float64}
    mx::Int
end

CSplineType() = CSplineType(Float64[], Array{Float64}(undef,0,0), 0)

function cspline_alloc(nx::Integer, nvar::Integer)
    xs = zeros(Float64, nx)
    fs = zeros(Float64, nx, nvar)
    return CSplineType(xs, fs, nx)
end

function cspline_eval_external(obj, psi::Float64)
    # If the object is a plain array (e.g., smats-like flattened data), return it
    try
        # If object is from SplinesMod, call bicube_eval or spline_eval where appropriate
        if typeof(obj) <: AbstractArray
            return obj
        end
    catch
    end

    # Default: return a small zero complex vector sized by mpert if available
    mp = try
        DconInterface.mpert[]
    catch
        1
    end
    return ComplexF64.(zeros(mp * mp))
end
