# Vacuum Module

The Vacuum module provides magnetostatic vacuum field calculations with plasma-wall interactions.
Refactored/interfaced from/with VACUUM by M.S. Chance.

## Overview

The module includes:

- Interface to Fortran vacuum field calculations (`mscvac`, `set_dcon_params`)
- Pure Julia implementation of vacuum response calculations (`compute_vacuum_response`, `compute_vacuum_field`)
- Support for various wall geometries and configurations

## Key Structures

### VacuumInput
Contains plasma boundary data and calculation parameters including:
- Plasma boundary coordinates (R, Z)
- Poloidal mode numbers (mlow, mhigh, mpert)
- Toroidal mode number (n)
- Grid resolution (mtheta, mtheta_eq)
- Safety factor (qa)

### WallShapeSettings
Specifies wall geometry configuration with options for:
- No wall, conformal, elliptical, dee-shaped, or custom walls
- Geometric parameters (a, aw, bw, cw, dw, tw)
- Equal arc length spacing option

## API Reference

```@autodocs
Modules = [JPEC.Vacuum]
```

## Functions

### set_dcon_params
```@docs
JPEC.Vacuum.set_dcon_params
```

### mscvac
```@docs
JPEC.Vacuum.mscvac
```

### compute_vacuum_response
```@docs
JPEC.Vacuum.compute_vacuum_response
```

### compute_vacuum_field
```@docs
JPEC.Vacuum.compute_vacuum_field
```

## Example Usage

### Basic Vacuum Calculation with Fortran Interface

```julia
using JPEC

# Set DCON parameters
mtheta, lmin, lmax, nnin = Int32(4), Int32(1), Int32(4), Int32(2)
qa1in = 1.23
xin = rand(Float64, lmax - lmin + 1)
zin = rand(Float64, lmax - lmin + 1)
deltain = rand(Float64, lmax - lmin + 1)

# Initialize DCON interface
JPEC.Vacuum.set_dcon_params(mtheta, lmin, lmax, nnin, qa1in, xin, zin, deltain)

# Set up vacuum calculation parameters
mpert = 5
mtheta_plasma = 256
mtheta_vacuum = 256
wv = zeros(ComplexF64, mpert, mpert)
complex_flag = true
kernelsignin = -1.0
wall_flag = false
farwall_flag = true
grrio = rand(Float64, 2*(mtheta_vacuum+5), mpert*2)
xzptso = rand(Float64, mtheta_vacuum+5, 4)

# Perform vacuum calculation
JPEC.Vacuum.mscvac(
    wv, mpert, mtheta_plasma, mtheta_vacuum,
    complex_flag, kernelsignin,
    wall_flag, farwall_flag,
    grrio, xzptso
)
```

### Julia Implementation Example

```julia
using JPEC

# Create VacuumInput struct
inputs = JPEC.Vacuum.VacuumInput(
    r = plasma_r_coords,
    z = plasma_z_coords,
    delta = delta_array,
    mlow = 1,
    mhigh = 10,
    mpert = 10,
    n = 1,
    qa = 2.5,
    mtheta_eq = 128,
    mtheta = 256,
    kernelsign = 1.0,
    force_wv_symmetry = true
)

# Define wall settings
wall_settings = JPEC.Vacuum.WallShapeSettings(
    shape = "conformal",
    a = 0.3,
    equal_arc_wall = true
)

# Compute vacuum response
wv, grri, xzpts = JPEC.Vacuum.compute_vacuum_response(inputs, wall_settings)
```

## Notes

- Requires proper initialization of DCON parameters before using the Fortran interface
- The pure Julia implementation (`compute_vacuum_response`) provides equivalent functionality
- For n=0 modes with closed walls, automatic regularization is applied
