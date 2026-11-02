# Utilities

`Utilities` holds the helpers shared by the physics modules: Fourier transforms and adaptive sampling, physical constants, the neoclassical resistivity closures, the HDF5 annotation layer of `gpec.h5`, the Hermite finite-element basis, and the IMAS export.

## Fourier transforms and sampling

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Utilities, GeneralizedPerturbedEquilibrium.Utilities.FourierTransforms, GeneralizedPerturbedEquilibrium.Utilities.AdaptiveSampling]
```

## Physical constants

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Utilities.PhysicalConstants]
```

## Neoclassical resistivity

Parallel-resistivity closures (Spitzer, Spitzer-Härm, and the Sauter and Redl
neoclassical models) used to set the Lundquist number in the tearing stack.

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Utilities.NeoclassicalResistivity]
```

## HDF5 annotations

Self-describing metadata for `gpec.h5` (long_name/units/dims attributes and HDF5
Dimension Scales); see the metadata contract in `docs/development/hdf5-conventions.md`.

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Utilities.HDF5Annotations]
```

## Hermite FEM

The Hermite-cubic basis and packed grid shared by the outer-region and inner-layer Galerkin solvers.

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Utilities.HermiteFEM]
```

## IMAS output

```@docs
GeneralizedPerturbedEquilibrium.write_imas
```
