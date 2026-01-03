module SplinesMod

using Interpolations

include("Helper.jl")

include("CubicSpline.jl")
include("BicubicSpline.jl")

export spline_eval!, spline_deriv1!, spline_deriv2!, spline_deriv3!, spline_eval, spline_integrate!, CubicSpline, empty_CubicSpline
export bicube_eval!, bicube_deriv1!, bicube_deriv2!, bicube_eval, BicubicSpline, empty_BicubicSpline

end
