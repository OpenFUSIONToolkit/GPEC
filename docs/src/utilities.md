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

## Bootstrap current

The flux-surface-averaged bootstrap current ``\langle j_{bs}\cdot B\rangle`` from the Redl et al. (2021) closure, assembled as OpenFUSIONToolkit TokaMaker assembles it: exact Redl coefficients with ``L_{34}=L_{31}``, the Koh ion collisionality, NRL Coulomb logarithms floored at 10, and the Sauter circulating fraction from surface moments of ``|B|``.
ψ is the poloidal flux per radian, increasing from the axis to the edge, so ``\psi = \psi_o\hat\psi``; ``I(\psi)=RB_\phi`` is signed and the result is co-directed with the equilibrium ``\langle j\cdot B\rangle`` of the same convention.
Temperatures are in eV and densities in m⁻³.
See Redl et al., Phys. Plasmas **28**, 022502 (2021) and, for the circulating fraction, Sauter et al., Phys. Plasmas **6**, 2834 (1999).

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Utilities.BootstrapCurrent]
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
