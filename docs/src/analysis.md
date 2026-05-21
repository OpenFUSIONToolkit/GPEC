# Analysis Module

The Analysis module provides post-processing and visualization utilities for GPEC simulation outputs.

## Submodules

- `ForceFreeStates`: Plotting functions for DCON-style ideal MHD stability results
- `Equilibrium`: Plotting functions for equilibrium objects
- `CoilForcing`: Plotting functions for coil geometry and normal field spectra
- `PerturbedEquilibrium`: Plotting functions for perturbed equilibrium results
- `PerturbedEquilibriumModes`: Data helpers to convert modal GPEC output to (ψ, θ) and (ψ, θ, φ) grids

```@docs
GeneralizedPerturbedEquilibrium.Analysis
```

## ForceFreeStates

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Analysis.ForceFreeStates]
```

## Equilibrium

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Analysis.Equilibrium]
```

## CoilForcing

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Analysis.CoilForcing]
```

## PerturbedEquilibrium

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Analysis.PerturbedEquilibrium]
```

## PerturbedEquilibriumModes

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Analysis.PerturbedEquilibriumModes]
```
