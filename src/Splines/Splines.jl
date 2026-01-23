module SplinesMod

# Pure Julia spline implementations
# FastCubicSpline.jl must come first as other modules depend on FastCubicSpline1D/Multi
include("FastCubicSpline.jl")
include("SplineAdapter.jl")
include("BicubicSpline.jl")
include("FourierModeSplines.jl")

# Re-export FastInterpolations search strategies for use with FastCubicSpline1D hint/search kwargs
using FastInterpolations: LinearBinary, Binary, HintedBinary

# Exports
export CubicSpline1D, empty_CubicSpline1D
export FastCubicSpline1D, empty_FastCubicSpline1D
export FastCubicSpline1DMulti, empty_FastCubicSpline1DMulti
export evaluate!, deriv1!, deriv2!, deriv3!, integrate!
export deriv1, deriv2, deriv3  # Non-mutating versions for FastCubicSpline1D
export LinearBinary, Binary, HintedBinary  # Search strategies for hint-based evaluation
export BicubicSpline, empty_BicubicSpline
export FourierModeSplines, empty_FourierModeSplines, get_complex_coeff, get_complex_coeffs!
export ComplexMatrixSpline, empty_ComplexMatrixSpline

end
