module SplinesMod

# Pure Julia spline implementations
include("SplineAdapter.jl")
include("FastCubicSpline.jl")
include("BicubicSpline.jl")
include("BicubicWrapper.jl")
include("FourierModeSplines.jl")
include("RZPhiSplines.jl")
include("EqfunSplines.jl")

# Exports
export CubicSpline1D, empty_CubicSpline1D
export FastCubicSpline1D, empty_FastCubicSpline1D
export evaluate!, deriv1!, deriv2!, deriv3!, integrate!
export BicubicSpline, empty_BicubicSpline
export BicubicWrapper, empty_BicubicWrapper
export FourierModeSplines, empty_FourierModeSplines, get_complex_coeff, get_complex_coeffs!
export ComplexMatrixSpline, empty_ComplexMatrixSpline
export RZPhiSplines, empty_RZPhiSplines
export EqfunSplines, empty_EqfunSplines

# Compatibility aliases for legacy API
export bicube_eval!, bicube_deriv1!, bicube_deriv2!

end
