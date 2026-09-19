"""
    Analysis

Post-processing and visualization utilities for GPEC simulation outputs.

## Submodules

  - `ForceFreeStates`: Plotting functions for ForceFreeStates (DCON-style ideal MHD stability) results
  - `Equilibrium`: Plotting functions for equilibrium profiles and flux surfaces
  - `CoilForcing`: Plotting functions for coil forcing spectra
  - `PerturbedEquilibrium`: Plotting functions for perturbed equilibrium and singular coupling results
  - `PerturbedEquilibriumModes`: Data helpers to convert modal GPEC output to (ψ, θ) and (ψ, θ, φ) grids
  - `ErrorFields`: Plotting functions for the error-field assessment (coil sensitivities, tolerance Monte Carlo, locking risk, phasing maps)
"""
module Analysis

include("ForceFreeStates.jl")
include("Equilibrium.jl")
include("CoilForcing.jl")
include("PerturbedEquilibrium.jl")
include("PerturbedEquilibriumModes.jl")
include("ErrorFields.jl")

end # module Analysis
