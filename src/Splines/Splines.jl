module SplinesMod

const libdir = joinpath(@__DIR__, "..", "..", "deps")
const libspline = joinpath(libdir, "libspline")

include("Helper.jl")

# Legacy Fortran-backed splines (to be deprecated)
include("CubicSpline.jl")
include("BicubicSpline.jl")
include("FourierSpline.jl")

# New pure Julia spline implementations
include("SplineAdapter.jl")
include("BicubicWrapper.jl")
include("FourierModeSplines.jl")

# Legacy exports (Fortran-backed)
export spline_eval!, spline_deriv1!, spline_deriv2!, spline_deriv3!, spline_eval, spline_integrate!, CubicSpline, empty_CubicSpline
export bicube_eval!, bicube_deriv1!, bicube_deriv2!, bicube_eval, BicubicSpline, empty_BicubicSpline
export fspline_eval, FourierSpline, empty_FourierSpline

# New pure Julia exports
export CubicSpline1D, empty_CubicSpline1D
export evaluate!, deriv1!, deriv2!, deriv3!, integrate!
export BicubicWrapper, empty_BicubicWrapper
export FourierModeSplines, empty_FourierModeSplines
export ComplexMatrixSpline, empty_ComplexMatrixSpline

end
