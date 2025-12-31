"""
    Utilities

Shared mathematical and computational utilities for JPEC modules.

This module provides common functionality used across multiple JPEC modules,
including efficient Fourier transforms, numerical integration, and other
mathematical utilities.

# Submodules

- `FourierTransforms`: Efficient Fourier transforms with pre-computed basis functions
"""
module Utilities

include("FourierTransforms.jl")

# Re-export key functionality
using .FourierTransforms
export FourierTransform, inverse, compute_fourier_coefficients

end # module Utilities
