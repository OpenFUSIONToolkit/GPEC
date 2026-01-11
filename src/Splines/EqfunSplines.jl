"""
EqfunSplines - Simple container for eqfun data with pre-computed first derivatives.

Just stores the grid data and derivatives as arrays. No wrapper methods.
Access the arrays directly: `eqfun.fs[ipsi, itheta, iq]`, `eqfun.fs_x`, `eqfun.fs_y`.

# Quantity Indices

The 3 quantities stored in the `fs` array (third dimension) are:
  - `fs[:,:,1]`: |B| - magnetic field magnitude
  - `fs[:,:,2]`: C1 - gyrokinetic coefficient (v·∇B terms)
  - `fs[:,:,3]`: C2 - gyrokinetic coefficient (toroidal terms)

These are computed by `compute_eqfun_nodes` in DirectEquilibrium.jl.
"""

"""
    EqfunSplines

Simple container holding eqfun values and pre-computed first derivatives on the grid.

# Fields

  - `xs::Vector{Float64}`: Normalized psi grid points (0 at axis, 1 at edge)
  - `ys::Vector{Float64}`: Theta grid points in radians (endpoint-inclusive, 0 to 2π)
  - `fs::Array{Float64,3}`: Function values (npsi, ntheta, nqty)
  - `fs_x::Array{Float64,3}`: d/dpsi derivatives
  - `fs_y::Array{Float64,3}`: d/dtheta derivatives
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
