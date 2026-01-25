module SplinesMod

# Pure Julia spline implementations
# FastInterpolationsAdaptor.jl provides FastCubicSpline1D, FastCubicSpline1DMulti, and ComplexMatrixSpline
include("FastInterpolationsAdaptor.jl")
include("BicubicSpline.jl")
include("FourierModeSplines.jl")

# Re-export FastInterpolations search strategies for use with FastCubicSpline1D hint/search kwargs
using FastInterpolations: LinearBinary, Binary, HintedBinary

# Exports
export FastCubicSpline1D, empty_FastCubicSpline1D
export FastCubicSpline1DMulti, empty_FastCubicSpline1DMulti
export evaluate!, deriv1!, deriv2!, deriv3!, integrate!
export deriv1, deriv2, deriv3  # Non-mutating versions for FastCubicSpline1D
export LinearBinary, Binary, HintedBinary  # Search strategies for hint-based evaluation
export BicubicSpline, empty_BicubicSpline, BicubicBC
export FourierModeSplines, empty_FourierModeSplines, get_complex_coeff, get_complex_coeffs!
export ComplexMatrixSpline, empty_ComplexMatrixSpline

# Export boundary condition types and helpers
export extrap_bc, NaturalBC, PeriodicBC, BCPair, Deriv1, Deriv2

end
