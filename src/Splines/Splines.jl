module SplinesMod

# Pure Julia spline implementations
# FastInterpolationsAdaptor.jl provides helpers for FastInterpolations CubicInterpolant/CubicSeriesInterpolant
# The codebase uses native interpolants from FastInterpolations directly
include("FastInterpolationsAdaptor.jl")

# Re-export FastInterpolations search strategies for use with CubicInterpolant hint/search kwargs
using FastInterpolations: LinearBinary, Binary, HintedBinary

# Exports
export evaluate!, deriv1!, integrate!, cumulative_integral, integrate_spline, total_integral, total_integral!
export LinearBinary, Binary, HintedBinary  # Search strategies for hint-based evaluation

# Export boundary condition types
export NaturalBC, PeriodicBC, CubicFit, BCPair, Deriv1, Deriv2

end
