module SplinesMod

# Fortran library path (for Fortran-backed splines)
const libdir = joinpath(@__DIR__, "..", "..", "deps")
const libspline = joinpath(libdir, "libspline")

# Helper utilities
include("Helper.jl")

# Pure Julia spline implementations
include("SplineAdapter.jl")
include("BicubicWrapper.jl")
include("FourierModeSplines.jl")
include("RZPhiSplines.jl")

# Fortran-backed bicubic spline (for compatibility and comparison)
include("BicubicSplineFortran.jl")

# Exports - Pure Julia implementations
export CubicSpline1D, empty_CubicSpline1D
export evaluate!, deriv1!, deriv2!, deriv3!, integrate!
export BicubicWrapper, empty_BicubicWrapper
export FourierModeSplines, empty_FourierModeSplines, get_complex_coeff, get_complex_coeffs!
export ComplexMatrixSpline, empty_ComplexMatrixSpline
export RZPhiSplines, empty_RZPhiSplines

# Exports - Fortran-backed splines
export BicubicSpline, empty_BicubicSpline

# Compatibility aliases for legacy API
export bicube_eval!, bicube_deriv1!, bicube_deriv2!

end
