"""
EqfunSplines - Simple container for eqfun data with pre-computed first derivatives.

Just stores the grid data and derivatives as arrays. No fancy wrapper methods.
Access the arrays directly: eqfun.fs, eqfun.fs_x, eqfun.fs_y.
"""

"""
    EqfunSplines

Simple container holding eqfun values and pre-computed first derivatives on the grid.

# Fields

  - `xs::Vector{Float64}`: Psi grid points
  - `ys::Vector{Float64}`: Theta grid points (endpoint-inclusive)
  - `fs::Array{Float64,3}`: Function values (npsi, ntheta, nqty)
  - `fs_x::Array{Float64,3}`: d/dpsi
  - `fs_y::Array{Float64,3}`: d/dtheta
"""
struct EqfunSplines
    xs::Vector{Float64}
    ys::Vector{Float64}
    fs::Array{Float64,3}
    fs_x::Array{Float64,3}
    fs_y::Array{Float64,3}
end

"""
    empty_EqfunSplines()

Create an empty placeholder for type stability.
"""
function empty_EqfunSplines()
    xs = Float64[0.0, 0.5, 1.0]
    ys = Float64[0.0, 0.5, 1.0]
    fs = zeros(Float64, 3, 3, 3)
    return EqfunSplines(xs, ys, fs, copy(fs), copy(fs))
end
