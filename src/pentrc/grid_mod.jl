"""
Minimal grid utilities used by torque.jl for tests.
Provides powspace and linspace_sub! used to build lambda/kappa grids.
"""
function powspace(a::Float64, b::Float64, mode::Integer, n::Integer, _side::AbstractString)
    # Simple geometric-like spacing between a and b
    return collect(range(a, stop=b, length=n))
end

function linspace_sub!(a::Float64, b::Float64, n::Integer, out::AbstractVector{Float64})
    if length(out) != n
        resize!(out, n)
    end
    out .= collect(range(a, stop=b, length=n))
    return out
end
