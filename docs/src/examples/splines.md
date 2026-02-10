# Spline Examples

This page demonstrates the usage of cubic splines with FastInterpolations and JPEC helper functions.

## 1D Cubic Spline Interpolation

### Basic Usage

```julia
using JPEC, Plots
using FastInterpolations: cubic_interp, deriv1

# Create sample data
xs = collect(range(0.0, 2π, length=21))
fs_sin = sin.(xs)
fs_cos = cos.(xs)

# Create splines with CubicFit boundary conditions
spline_sin = cubic_interp(xs, fs_sin; bc=JPEC.SplinesMod.CubicFit())
spline_cos = cubic_interp(xs, fs_cos; bc=JPEC.SplinesMod.CubicFit())

# Evaluate on fine grid
xs_fine = collect(range(0.0, 2π, length=100))
fs_sin_fine = spline_sin.(xs_fine)
fs_cos_fine = spline_cos.(xs_fine)

# Plot results
plot(xs_fine, fs_sin_fine, label="sin(x) spline", legend=:topright)
plot!(xs_fine, fs_cos_fine, label="cos(x) spline")
scatter!(xs, fs_sin, label="sin(x) data")
scatter!(xs, fs_cos, label="cos(x) data")
```

### Multi-series Splines (More Efficient)

```julia
using FastInterpolations: cubic_interp

# Create sample data
xs = collect(range(0.0, 2π, length=21))
fs_matrix = hcat(sin.(xs), cos.(xs))  # Each column is a series

# Create series interpolant (more efficient than separate splines)
spline_series = cubic_interp(xs, fs_matrix; bc=JPEC.SplinesMod.CubicFit())

# Evaluate on fine grid
xs_fine = collect(range(0.0, 2π, length=100))
fs_fine = spline_series.(xs_fine)  # Returns matrix with 100 rows, 2 columns

# Plot results
plot(xs_fine, fs_fine[:, 1], label="sin(x) spline")
plot!(xs_fine, fs_fine[:, 2], label="cos(x) spline")
scatter!(xs, fs_matrix[:, 1], label="sin(x) data")
scatter!(xs, fs_matrix[:, 2], label="cos(x) data")
```

### Computing Derivatives

```julia
using FastInterpolations: cubic_interp, deriv1

# Create data
xs = collect(range(0.0, 2π, length=21))
fs = sin.(xs)

# Create spline
spline = cubic_interp(xs, fs; bc=JPEC.SplinesMod.CubicFit())

# Create derivative view
d1_spline = deriv1(spline)

# Evaluate function and derivative on fine grid
xs_fine = collect(range(0.0, 2π, length=100))
fs_fine = spline.(xs_fine)
d1_fs_fine = d1_spline.(xs_fine)

# Plot
plot(xs_fine, fs_fine, label="sin(x)", legend=:topright)
plot!(xs_fine, d1_fs_fine, label="d/dx sin(x) ≈ cos(x)")
plot!(xs_fine, cos.(xs_fine), label="cos(x) exact", linestyle=:dash)
```

### Integration

```julia
using JPEC

# Create data
xs = collect(range(0.0, 2π, length=21))
fs = sin.(xs)

# Compute cumulative integral (exact spline integration)
fs_integral = JPEC.SplinesMod.cumulative_integral(xs, fs; bc=JPEC.SplinesMod.CubicFit())

# Should be approximately -cos(x) + constant
plot(xs, fs_integral, label="∫sin(x)dx", legend=:topright)
plot!(xs, -cos.(xs) .+ cos(0.0), label="-cos(x) + 1 (exact)", linestyle=:dash)
```

## 2D Cubic Spline Interpolation

### Basic 2D Function

```julia
using JPEC, Plots
using FastInterpolations: cubic_interp

# Create 2D grid
xs = collect(range(0.0, 2π, length=20))
ys = collect(range(0.0, 2π, length=20))

# Create 2D function data (note: no need for transpose with this convention)
fs = [sin(x) * cos(y) + 1.0 for x in xs, y in ys]

# Set up 2D cubic interpolant with periodic BC in y direction
spline_2d = cubic_interp((xs, ys), fs; 
    bc=(JPEC.SplinesMod.CubicFit(), JPEC.SplinesMod.PeriodicBC()),
    extrap=(:extension, :wrap))

# Evaluate on fine grid
xs_fine = collect(range(0.0, 2π, length=100))
ys_fine = collect(range(0.0, 2π, length=100))
fs_fine = [spline_2d((x, y)) for x in xs_fine, y in ys_fine]

# Create contour plot
contourf(xs_fine, ys_fine, fs_fine',
         title="2D Cubic Spline: sin(x)cos(y) + 1")
```

### With Derivatives

```julia
using FastInterpolations: cubic_interp

# Create 2D interpolant
xs = collect(range(0.0, 2π, length=20))
ys = collect(range(0.0, 2π, length=20))
fs = [sin(x) * cos(y) for x in xs, y in ys]

spline_2d = cubic_interp((xs, ys), fs; 
    bc=(JPEC.SplinesMod.CubicFit(), JPEC.SplinesMod.CubicFit()))

# Evaluate with derivatives on fine grid
xs_fine = collect(range(0.0, 2π, length=100))
ys_fine = collect(range(0.0, 2π, length=100))

fs_fine = [spline_2d((x, y)) for x in xs_fine, y in ys_fine]
fsx_fine = [spline_2d((x, y); deriv=(1, 0)) for x in xs_fine, y in ys_fine]  # ∂f/∂x
fsy_fine = [spline_2d((x, y); deriv=(0, 1)) for x in xs_fine, y in ys_fine]  # ∂f/∂y

# Plot function and derivatives
p1 = contourf(xs_fine, ys_fine, fs_fine', title="f(x,y)")
p2 = contourf(xs_fine, ys_fine, fsx_fine', title="∂f/∂x")
p3 = contourf(xs_fine, ys_fine, fsy_fine', title="∂f/∂y")

plot(p1, p2, p3, layout=(1,3), size=(1200, 400))
```

### Equilibrium Example

This example shows spline usage with the Solov'ev equilibrium:

```julia
using FastInterpolations: cubic_interp

# Create equilibrium parameters
kappa = 1.8  # elongation
a = 1.0      # minor radius
r0 = 3.5     # major radius
q0 = 1.25    # safety factor

# Create spatial grid
mr, mz = 40, 43
rmin, rmax = r0 - 1.5*a, r0 + 1.5*a
zmin, zmax = -1.5*kappa*a, 1.5*kappa*a

rs = collect(range(rmin, rmax, length=mr))
zs = collect(range(zmin, zmax, length=mz))

# Create Solov'ev equilibrium psi field
f0 = r0 * 1.0  # b0fac = 1.0
psio = kappa * f0 * a^2 / (2 * q0 * r0)
psifac = psio / (a*r0)^2
efac = 1/kappa^2

psifs = [psio - psifac * (efac * (r * z)^2 + (r^2-r0^2)^2/4) for r in rs, z in zs]

# Set up 2D cubic interpolant for psi
psi_spline = cubic_interp((rs, zs), psifs; 
    bc=(JPEC.SplinesMod.CubicFit(), JPEC.SplinesMod.CubicFit()),
    extrap=(:extension, :extension))

# Evaluate on fine grid for plotting
rs_fine = collect(range(rmin, rmax, length=110))
zs_fine = collect(range(zmin, zmax, length=100))
psi_fine = [psi_spline((r, z)) for r in rs_fine, z in zs_fine]

# Create contour plot
contourf(rs_fine, zs_fine, psi_fine',
         title="Ψ: Solov'ev Equilibrium",
         xlabel="R", ylabel="Z",
         aspect_ratio=:equal)
```

## Performance Tips

1. **Use LinearBinary search for monotonic access**: When evaluating splines with monotonically increasing/decreasing x values (common in ODE integration), use `search=LinearBinary()` for ~4 ns/point performance
   ```julia
   spline = cubic_interp(xs, fs; search=LinearBinary())
   ```

2. **Use CubicSeriesInterpolant for multiple quantities**: When interpolating multiple quantities on the same grid, use a matrix and create a single `CubicSeriesInterpolant` rather than multiple separate splines

3. **Reuse interpolants**: Spline construction is relatively expensive; reuse interpolant objects when possible

4. **Use hint for sequential access**: For sequential evaluation, pass a hint to avoid searching:
   ```julia
   hint = Ref(1)
   for x in xs_eval
       f = spline(x; hint=hint)
   end
   ```
