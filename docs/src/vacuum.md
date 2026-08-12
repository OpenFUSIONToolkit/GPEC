# Vacuum Module

The Vacuum module provides magnetostatic vacuum field calculations with plasma-wall interactions.
The 2D vacuum calculations follow the approach outlined in [Chance Phys. Plasmas 1997, Chance J. Comp. Phys. 2007] with a pure Julia implementation.

## Overview

The module provides:

- Vacuum response calculations (`compute_vacuum_response`, `compute_vacuum_field`)
- Support for various wall geometries (conformal, elliptical, dee-shaped, or custom)
- Pre-computed Legendre functions using Bulirsch elliptic integrals for improved accuracy

## Key Structures

### VacuumInput
Contains plasma boundary data and calculation parameters including:
- Plasma boundary coordinates (r, z) on GPEC theta grid
- Free toroidal angle parameter (ν) where ϕ = 2πζ + ν(ψ, θ)
- Poloidal mode numbers (mlow, mpert)
- Toroidal mode number (n)
- Grid resolution (mtheta) for vacuum calculations
- Optional kernel sign and symmetry flags

### WallShapeSettings
Specifies wall geometry configuration with options for:
- No wall, conformal, elliptical, dee-shaped, or custom walls
- Geometric parameters (a, aw, bw, cw, dw, tw)
- Equal arc length spacing option

## API Reference

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Vacuum]
```

## Functions

### compute_vacuum_response
```@docs
GeneralizedPerturbedEquilibrium.Vacuum.compute_vacuum_response
```

### compute_vacuum_field
```@docs
GeneralizedPerturbedEquilibrium.Vacuum.compute_vacuum_field
```

## Example Usage

### Basic Vacuum Response Calculation

```julia
using GeneralizedPerturbedEquilibrium

# Create VacuumInput struct with plasma boundary data
# Note: ν is the free toroidal angle parameter where ϕ = 2πζ + ν(ψ, θ)
inputs = GeneralizedPerturbedEquilibrium.Vacuum.VacuumInput(
    r = plasma_r_coords,      # Plasma R coordinates on GPEC theta grid
    z = plasma_z_coords,      # Plasma Z coordinates on GPEC theta grid
    ν = nu_array,             # Toroidal angle parameter (formerly delta/qa)
    mlow = 1,                 # Lowest poloidal mode number
    mpert = 10,               # Number of poloidal modes
    n = 1,                    # Toroidal mode number
    mtheta = 256,             # Number of poloidal grid points
    kernelsign = 1.0          # Kernel sign (+1 or -1)
)

# Define wall settings
wall_settings = GeneralizedPerturbedEquilibrium.Vacuum.WallShapeSettings(
    shape = "conformal",      # Wall shape type
    a = 0.3,                  # Wall distance parameter
    equal_arc_wall = true     # Use equal arc length spacing
)

# Compute vacuum response; returns a VacuumResponse with wv, grri, grre, plasma_pts, wall_pts
vac = GeneralizedPerturbedEquilibrium.Vacuum.compute_vacuum_response(inputs, wall_settings)
```

### Vacuum Field Calculation at Observation Points

```julia
# Compute vacuum field at specific observation points
# xi, eta are the real and imaginary parts of the perturbation amplitudes
R_obs = 2.0  # Major radius of observation point
Z_obs = 0.0  # Height of observation point

chi = GeneralizedPerturbedEquilibrium.Vacuum.compute_vacuum_field(R_obs, Z_obs, inputs, xi, eta, plasma_surf)
```

## Notes

- The Julia implementation uses Bulirsch's algorithm for elliptic integrals, providing improved accuracy over polynomial approximations
- For large mode numbers (nρ̂ ≥ 0.1), 32-point Gaussian quadrature is used for Legendre function evaluation
- For n=0 modes with closed walls, automatic regularization is applied
- Wall shapes support: nowall, conformal, elliptical, dee, mod_dee, or custom from file
- The vacuum response matrix wv is scaled by the singular factor (m - nq)(m' - nq) per [Chance Phys. Plasmas 1997]
