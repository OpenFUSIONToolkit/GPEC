"""
Minimal spline_mod stub for tests.
Provides use_classic_splines() placeholder.
"""
"""
Minimal spline_mod stub for tests.
Provides use_classic_splines() placeholder.
"""
function use_classic_splines()
    # no-op placeholder
    return nothing
end

# Minimal spline types and helpers used by torque.jl/tests

mutable struct SplineType
    xs::Vector{Float64}
    fs::Matrix{Float64}
    mx::Int
    fsi::Union{Matrix{Float64}, Nothing}
end

function SplineType(nx::Integer, nvar::Integer)
    xs = zeros(Float64, nx)
    fs = zeros(Float64, nx, nvar)
    return SplineType(xs, fs, nx, nothing)
end

function spline_alloc(nx::Integer, nvar::Integer)
    return SplineType(nx, nvar)
end

function spline_fit!(spl::SplineType, _mode::AbstractString)
    # no-op: pretend spline is fitted
    return nothing
end

function spline_int!(spl::SplineType)
    # create an integral storage field if requested
    if spl.fsi === nothing
        spl.fsi = zeros(Float64, spl.mx, size(spl.fs,2))
    end
    return nothing
end

function spline_dealloc!(_spl)
    return nothing
end

function spline_eval_external(obj, psi::Float64; deriv::Integer=0)
    # Return dummy kinetic vector (length 9) and its derivative if requested
    out = zeros(Float64, 9)
    if deriv == 1
        return out, out
    else
        return out
    end
end
