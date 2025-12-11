module SplinesMod

const libdir = joinpath(@__DIR__, "..", "..", "deps")
const libspline = joinpath(libdir, "libspline")

include("Helper.jl")

include("CubicSpline.jl")
include("BicubicSpline.jl")
include("FourierSpline.jl")

export spline_eval!, spline_deriv1!, spline_deriv2!, spline_deriv3!, spline_eval, spline_integrate!, CubicSpline
export bicube_eval!, bicube_deriv1!, bicube_deriv2!, bicube_eval, BicubicSpline
export fspline_eval, FourierSpline

end
