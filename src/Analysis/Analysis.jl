"""
    Analysis

Post-processing and visualization utilities for GPEC simulation outputs.

## Submodules

  - `ForceFreeStates`: Plotting functions for ForceFreeStates (DCON-style ideal MHD stability) results
  - `Equilibrium`: Plotting functions for equilibrium profiles and flux surfaces
  - `PerturbedEquilibrium`: Plotting functions for perturbed equilibrium and singular coupling results
"""
module Analysis

include("ForceFreeStates.jl")
include("Equilibrium.jl")
include("PerturbedEquilibrium.jl")

end # module Analysis
