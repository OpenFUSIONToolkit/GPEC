"""
    Utilities

Shared mathematical and computational utilities for GPEC modules.

This module provides common functionality used across multiple GPEC modules,
including efficient Fourier transforms, numerical integration, and other
mathematical utilities.

# Submodules

  - `FourierTransforms`: Efficient Fourier transforms with pre-computed basis functions
  - `PhysicalConstants`: SI physical constants matching Fortran GPEC/SLAYER values
"""
module Utilities

include("FourierTransforms.jl")
include("FourierCoefficients.jl")
include("PhysicalConstants.jl")

using .FourierTransforms
export FourierTransform, inverse, compute_fourier_coefficients
export transform!, inverse_transform!
export fourier_transform!, fourier_inverse_transform!

export FourierCoefficients, empty_FourierCoefficients, get_complex_coeff, get_complex_coeffs!

using .PhysicalConstants
export PhysicalConstants
export MU_0, M_E, M_P, E_CHG, K_B, EPS_0

end # module Utilities
