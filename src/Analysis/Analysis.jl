"""
    Analysis

Post-processing and visualization utilities for GPEC simulation outputs.

Every registered plot takes a result in memory or the path of the `gpec.h5` it was saved to, and throws an
`ArgumentError` naming what to change in the run when that source lacks the data it draws. `available_plots(src)`
lists the plots for a result, a path, or the named tuple `main` returns.

## Submodules

  - `ForceFreeStates`: Plotting functions for ForceFreeStates (DCON-style ideal MHD stability) results
  - `Equilibrium`: Plotting functions for equilibrium profiles and flux surfaces
  - `CoilForcing`: Plotting functions for coil forcing spectra
  - `PerturbedEquilibrium`: Plotting functions for perturbed equilibrium and singular coupling results
  - `PerturbedEquilibriumModes`: Data helpers to convert modal GPEC output to (ψ, θ) and (ψ, θ, φ) grids
  - `ErrorFields`: Plotting functions for the error-field assessment (coil sensitivities, tolerance Monte Carlo, locking risk, phasing maps)
"""
module Analysis

include("Registry.jl")
include("ForceFreeStates.jl")
include("Equilibrium.jl")
include("CoilForcing.jl")
include("PerturbedEquilibrium.jl")
include("PerturbedEquilibriumModes.jl")
include("ErrorFields.jl")

export available_plots

end # module Analysis
