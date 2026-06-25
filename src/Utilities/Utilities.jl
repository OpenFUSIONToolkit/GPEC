"""
    Utilities

Shared mathematical and computational utilities for GPEC modules.

This module provides common functionality used across multiple GPEC modules,
including efficient Fourier transforms, numerical integration, and other
mathematical utilities.

# Submodules

  - `FourierTransforms`: Efficient Fourier transforms with pre-computed basis functions
  - `PhysicalConstants`: SI physical constants matching Fortran GPEC/SLAYER values
  - `NeoclassicalResistivity`: Spitzer/Sauter/Redl resistivity closures shared by
    the GGJ and SLAYER inner-layer models
"""
module Utilities

include("FourierTransforms.jl")
include("FourierCoefficients.jl")
include("PhysicalConstants.jl")
include("KineticProfiles.jl")
include("NeoclassicalResistivity.jl")
include("GridUtilities.jl")

using .FourierTransforms
export FourierTransform, inverse, compute_fourier_coefficients
export transform!, inverse_transform!
export fourier_transform!, fourier_inverse_transform!

export FourierCoefficients, empty_FourierCoefficients, get_complex_coeff, get_complex_coeffs!

using .PhysicalConstants
export PhysicalConstants
export MU_0, M_E, M_P, E_CHG, K_B, EPS_0

export KineticProfiles

using .NeoclassicalResistivity
export NeoclassicalResistivity
export NeoResistivityModel, SpitzerModel, SpitzerHarmModel, SauterNeoModel, RedlNeoModel
export coulomb_log_e, eta_spitzer, tau_ee_spitzer_harm, eta_spitzer_harm
export trapped_fraction, trapped_fraction_eps
export nu_star_e, eta_neoclassical

end # module Utilities
