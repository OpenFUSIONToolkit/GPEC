# Splines Module

The Splines module provides cubic, bicubic, and fourier spline interpolation functionality, used for smooth representation of MHD equilibria.

## Overview

The module includes:
- Cubic spline interpolation for 1D data
- Bicubic spline interpolation for 2D data
- Fourier spline interpolation for decomposed data
- Derivative evaluation capabilities
- Support for both real and complex-valued data

## API Reference

```@autodocs
Modules = [JPEC.SplinesMod]
```

## Types

### CubicSplineType
Represents a cubic spline interpolation object.

### BicubicSplineType
Represents a bicubic spline interpolation object for 2D data.

### FourierSplineType
Represents a Fourier spline interpolation object for decomposed data.

## Functions

### CubicSpline
```@docs
JPEC.SplinesMod.CubicSpline
```

### spline_eval
```@docs
JPEC.SplinesMod.spline_eval!
JPEC.SplinesMod.spline_deriv1!
JPEC.SplinesMod.spline_deriv2!
JPEC.SplinesMod.spline_deriv3!
JPEC.SplinesMod.spline_eval
```

### BicubicSpline
```@docs
JPEC.SplinesMod.BicubicSpline
```

### bicube_eval
```@docs
JPEC.SplinesMod.bicube_eval!
JPEC.SplinesMod.bicube_deriv1!
JPEC.SplinesMod.bicube_deriv2!
JPEC.SplinesMod.bicube_eval
```

### FourierSpline
```@docs
JPEC.SplinesMod.FourierSpline
```

### fspline_eval
```@docs
JPEC.SplinesMod.fspline_eval
```

## Example Usage

### 1D Cubic Spline
```julia
using JPEC

# Create data points
xs = collect(range(0.0, stop=2π, length=21))
fs = sin.(xs)

# Set up spline (1 quantity)
spline = JPEC.SplinesMod.CubicSpline(xs, hcat(fs), 1)

# Evaluate at value (and derivatives) at single point
x = 1.0
f = JPEC.SplinesMod.spline_eval!(spline, x)
f, f1, f2, f3 = JPEC.SplinesMod.spline_deriv3!(spline, x)

# Evaluate at vector of new points
xs_fine = collect(range(0.0, stop=2π, length=100))
fs_fine = JPEC.SplinesMod.spline_eval(spline, xs_fine)
```

### 2D Bicubic Spline
```julia
# Create 2D grid
xs = collect(range(0.0, stop=2π, length=20))
ys = collect(range(0.0, stop=2π, length=20))

# Create 2D function data
fs = zeros(20, 20, 1)
for i in 1:20, j in 1:20
    fs[i, j, 1] = sin(xs[i]) * cos(ys[j])
end

# Set up bicubic spline
bcspline = JPEC.SplinesMod.BicubicSpline(xs, ys, fs, 1, 1)

# Evaluate spline
x_eval, y_eval = π/2, π/4
f = JPEC.SplinesMod.bicube_eval!(bcspline, x_eval, y_eval) # just the value
f, fx, fy = JPEC.SplinesMod.bicube_deriv1!(bcspline, x_eval, y_eval, 1) #include first derivative
```
