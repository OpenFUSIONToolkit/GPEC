module SplinesMod

# Pure Julia spline implementations
# FastInterpolationsAdaptor.jl provides helpers and MultiQuantityProfile
# The codebase uses native CubicInterpolant from FastInterpolations directly
include("FastInterpolationsAdaptor.jl")
include("BicubicSpline.jl")

# Re-export FastInterpolations search strategies for use with CubicInterpolant hint/search kwargs
using FastInterpolations: LinearBinary, Binary, HintedBinary

# Exports
export MultiQuantityProfile, empty_MultiQuantityProfile
export evaluate!, deriv1!, integrate!, cumulative_integral, integrate_spline, total_integral, total_integral!
export LinearBinary, Binary, HintedBinary  # Search strategies for hint-based evaluation
export BicubicSpline, empty_BicubicSpline, BicubicBC

# Export boundary condition types and helpers
export extrap_bc, extrap_bc_matrix, NaturalBC, PeriodicBC, BCPair, Deriv1, Deriv2

end
