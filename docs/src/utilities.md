# Utilities Module

The Utilities module provides helper functions and data structures used across GeneralizedPerturbedEquilibrium.

## Overview

The Utilities module currently provides:
- `FourierCoefficients`: Lightweight FFT-based Fourier decomposition for periodic data
- Helper functions for accessing Fourier coefficients at grid points

## API Reference

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Utilities, GeneralizedPerturbedEquilibrium.Utilities.FourierTransforms]
```

## Physical Constants

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Utilities.PhysicalConstants]
```

## Neoclassical Resistivity

Parallel-resistivity closures (Spitzer, Spitzer-Härm, and the Sauter and Redl
neoclassical models) used to set the Lundquist number in the tearing stack.

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Utilities.NeoclassicalResistivity]
```

## HDF5 Annotations

Self-describing metadata for `gpec.h5` (long_name/units/dims attributes and HDF5
Dimension Scales); see the metadata contract in `docs/development/hdf5-conventions.md`.

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Utilities.HDF5Annotations]
```

## IMAS Output

```@docs
GeneralizedPerturbedEquilibrium.write_imas
```

## Example Usage

```julia
using GeneralizedPerturbedEquilibrium

# Create sample periodic data
xs = range(0, 1; length=100) |> collect
ys = range(0, 2π; length=64) |> collect
fs = zeros(100, 64, 2)

# Fill with periodic function
for i in 1:100, j in 1:64
    fs[i, j, 1] = exp(-xs[i]) * cos(3*ys[j])
    fs[i, j, 2] = exp(-xs[i]) * sin(5*ys[j])
end

# Compute Fourier coefficients, keeping 10 modes
fc = GeneralizedPerturbedEquilibrium.Utilities.FourierCoefficients(xs, ys, fs, 10)

# Access individual coefficient
# Get mode 3 at radial index 50 for quantity 1
c = GeneralizedPerturbedEquilibrium.Utilities.get_complex_coeff(fc, 50, 3, 1)

# Get all coefficients for a given radial index and quantity
coeffs = Vector{ComplexF64}(undef, fc.mmax + 1)
GeneralizedPerturbedEquilibrium.Utilities.get_complex_coeffs!(coeffs, fc, 50, 1)
```
