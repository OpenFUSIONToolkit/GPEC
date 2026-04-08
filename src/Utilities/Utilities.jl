"""
    Utilities

Shared mathematical and computational utilities for GPEC modules.

This module provides common functionality used across multiple GPEC modules,
including efficient Fourier transforms, numerical integration, and other
mathematical utilities.

# Submodules

  - `FourierTransforms`: Efficient Fourier transforms with pre-computed basis functions
"""
module Utilities

include("FourierTransforms.jl")
include("FourierCoefficients.jl")

using .FourierTransforms
export FourierTransform, inverse, compute_fourier_coefficients
export transform!, inverse_transform!
export fourier_transform!, fourier_inverse_transform!

export FourierCoefficients, empty_FourierCoefficients, get_complex_coeff, get_complex_coeffs!

end # module Utilities
