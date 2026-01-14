module SplinesMod

# Pure Julia spline implementations
include("SplineAdapter.jl")
include("FastCubicSpline.jl")
include("BicubicSpline.jl")
include("FourierModeSplines.jl")
include("EqfunSplines.jl")

# Exports
export CubicSpline1D, empty_CubicSpline1D
export FastCubicSpline1D, empty_FastCubicSpline1D
export evaluate!, deriv1!, deriv2!, deriv3!, integrate!
export BicubicSpline, empty_BicubicSpline
export FourierModeSplines, empty_FourierModeSplines, get_complex_coeff, get_complex_coeffs!
export ComplexMatrixSpline, empty_ComplexMatrixSpline
export EqfunSplines, empty_EqfunSplines

end
