# Vacuum Module

The Vacuum module provides magnetostatic vacuum field calculations with plasma-wall interactions. Refactored/interfaced from/with VACUUM by M.S. Chance.

## Overview

The module includes:
- Interface to Fortran vacuum field calculations
- Julia refactored version of the VACUUM code

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

## Example Usage

### Basic Vacuum Calculation
```julia
using JPEC

# Set DCON parameters
mthin, lmin, lmax, nnin = Int32(4), Int32(1), Int32(4), Int32(2)
qa1in = 1.23
xin = rand(Float64, lmax - lmin + 1)
zin = rand(Float64, lmax - lmin + 1)
deltain = rand(Float64, lmax - lmin + 1)

# Initialize DCON interface
JPEC.Vacuum.set_dcon_params(mthin, lmin, lmax, nnin, qa1in, xin, zin, deltain)

# Set up vacuum calculation parameters
mpert = 5
mtheta = 256
mthvac = 256
wv = zeros(ComplexF64, mpert, mpert)
complex_flag = true
kernelsignin = -1.0
wall_flag = false
farwall_flag = true
grrio = rand(Float64, 2*(mthvac+5), mpert*2)
xzptso = rand(Float64, mthvac+5, 4)

# Perform vacuum calculation
JPEC.Vacuum.mscvac(
    wv, mpert, mtheta, mthvac,
    complex_flag, kernelsignin,
    wall_flag, farwall_flag,
    grrio, xzptso
)
```

## Notes

- Requires proper initialization of DCON parameters before use
- Supports both complex and real arithmetic depending on the application
