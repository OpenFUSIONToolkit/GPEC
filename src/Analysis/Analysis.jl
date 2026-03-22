"""
    Analysis

Post-processing and visualization utilities for GPEC simulation outputs.

## Submodules

  - `ForceFreeStates`: Plotting functions for DCON-style ideal MHD stability results
  - `Equilibrium`: Plotting functions for equilibrium objects
"""
module Analysis

include("ForceFreeStates.jl")
include("Equilibrium.jl")
include("CoilForcing.jl")

end # module Analysis
