# Splines Module

The Splines module provides helpers and utilities for working with FastInterpolations.jl cubic splines, used for smooth representation of MHD equilibria.

## Overview

JPEC uses [FastInterpolations.jl](https://github.com/ThummeTo/FastInterpolations.jl) for high-performance cubic spline interpolation. The SplinesMod provides:
- Helper functions for exact spline integration (`cumulative_integral`, `total_integral`)
- Re-exports of FastInterpolations boundary condition types
- Re-exports of FastInterpolations search strategies for optimized evaluation

## API Reference

```@autodocs
Modules = [JPEC.SplinesMod]
```

## Boundary Condition Types

The module re-exports FastInterpolations boundary condition types:
- `NaturalBC()`: Natural boundary condition (f''(x) = 0 at endpoints)
- `PeriodicBC()`: Periodic boundary condition (requires data[end] == data[1])
- `CubicFit()`: Automatic endpoint derivative estimation using 4-point fit
- `BCPair(bc_left, bc_right)`: Different BCs at left and right boundaries
- `Deriv1(value)`: First derivative boundary condition
- `Deriv2(value)`: Second derivative boundary condition

## Search Strategies

For optimized evaluation (especially with monotonic access patterns):
- `LinearBinary()`: Linear search + binary fallback (~4 ns/point for monotonic access)
- `Binary()`: Binary search
- `HintedBinary()`: Binary search with hint

## Integration Functions

### cumulative_integral
```@docs
JPEC.SplinesMod.cumulative_integral
```

### total_integral
```@docs
JPEC.SplinesMod.total_integral
```

### integrate_spline
```@docs
JPEC.SplinesMod.integrate_spline
```

## Example Usage

### 1D Cubic Spline
```julia
using JPEC
using FastInterpolations: cubic_interp, deriv1

# Create data points
xs = collect(range(0.0, 2π, length=21))
fs = sin.(xs)

# Create spline with natural boundary conditions
spline = cubic_interp(xs, fs; bc=JPEC.SplinesMod.NaturalBC())

# Evaluate at a single point
x = 1.0
f = spline(x)

# Evaluate derivative at a point
d1_spline = deriv1(spline)
f_deriv = d1_spline(x)

# Evaluate at vector of new points
xs_fine = collect(range(0.0, 2π, length=100))
fs_fine = spline.(xs_fine)

# Compute cumulative integral
fs_integral = JPEC.SplinesMod.cumulative_integral(xs, fs; bc=JPEC.SplinesMod.NaturalBC())
```

### 2D Cubic Spline (CubicInterpolantND)
```julia
using FastInterpolations: cubic_interp

# Create 2D grid
xs = collect(range(0.0, 0.99, length=20))
ys = collect(range(0.0, 2π, length=20))

# Create 2D function data
fs = zeros(20, 20)
for i in 1:20, j in 1:20
    fs[i, j] = sqrt(xs[i]) * cos(ys[j])
end

# Set up 2D cubic interpolant
# Use PeriodicBC on second dimension (θ direction)
spline_2d = cubic_interp((xs, ys), fs; 
    bc=(JPEC.SplinesMod.CubicFit(), JPEC.SplinesMod.PeriodicBC()),
    extrap=(:extension, :wrap))

# Evaluate spline at a point
x_eval, y_eval = 0.5, π/4
f = spline_2d((x_eval, y_eval))

# Evaluate with derivative (specify which dimension)
fx = spline_2d((x_eval, y_eval); deriv=(1, 0))  # ∂f/∂x
fy = spline_2d((x_eval, y_eval); deriv=(0, 1))  # ∂f/∂y
```

### Multi-series 1D Spline (CubicSeriesInterpolant)
```julia
using FastInterpolations: cubic_interp

# Create multiple data series
xs = collect(range(0.0, 1.0, length=50))
fs = hcat(sin.(2π*xs), cos.(2π*xs), xs.^2)  # 3 series

# Create series interpolant (more efficient than separate splines)
spline_series = cubic_interp(xs, fs; bc=JPEC.SplinesMod.CubicFit())

# Evaluate all series at once
x = 0.5
f_all = spline_series(x)  # Returns vector of length 3

# Compute total integral for all series
integrals = JPEC.SplinesMod.total_integral(xs, fs; bc=JPEC.SplinesMod.CubicFit())
```
