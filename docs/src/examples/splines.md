# Spline Examples

This page demonstrates the usage of the Splines module with practical examples.

## 1D Cubic Spline Interpolation

### Basic Usage

```julia
using JPEC, Plots

# Create sample data
xs = collect(range(0.0, stop=2π, length=21))
fs = sin.(xs)
fc = cos.(xs)

# Combine into matrix (each column is a different quantity)
fs_matrix = hcat(fs, fc)

# Set up spline for 2 quantities
spline = JPEC.SplinesMod.CubicSpline(xs, fs_matrix, 2)

# Evaluate on fine grid
xs_fine = collect(range(0.0, stop=2π, length=100))
fs_fine = JPEC.SplinesMod.spline_eval(spline, xs_fine)

# Plot results
plot(xs_fine, fs_fine[:, 1], label="sin(x) spline", legend=:topright)
plot!(xs_fine, fs_fine[:, 2], label="cos(x) spline")
scatter!(xs, fs, label="sin(x) data")
scatter!(xs, fc, label="cos(x) data")
```

### Complex-valued Splines

```julia
# Create complex exponential data
xs = collect(range(0.0, stop=2π, length=20))
fm = exp.(-im .* xs)  # e^(-ix)
fp = exp.(im .* xs)   # e^(ix)

# Combine complex data
fs_matrix = hcat(fm, fp)

# Set up complex spline
spline = JPEC.SplinesMod.CubicSpline(xs, fs_matrix, 2)

# Evaluate
xs_fine = collect(range(0.0, stop=2π, length=100))
fs_fine = JPEC.SplinesMod.spline_eval(spline, xs_fine)

# Plot real and imaginary parts
plot(xs_fine, real.(fs_fine[:, 1]), label="Re(e^(-ix))")
plot!(xs_fine, imag.(fs_fine[:, 1]), label="Im(e^(-ix))")
```

## 2D Bicubic Spline Interpolation

### Basic 2D Function

```julia
using JPEC, Plots

# Create 2D grid
xs = collect(range(0.0, stop=2π, length=20))
ys = collect(range(0.0, stop=2π, length=20))

# Create 2D function data
fs1 = sin.(xs') .* cos.(ys) .+ 1.0
fs2 = cos.(xs') .* sin.(ys) .+ 1.0

# Prepare data array (nx × ny × nquantities)
fs = zeros(20, 20, 2)
fs[:, :, 1] = fs1
fs[:, :, 2] = fs2

# Set up bicubic spline
bcspline = JPEC.SplinesMod.bicube_setup(xs, ys, fs, 2, 2)

# Evaluate on fine grid
xs_fine = collect(range(0.0, stop=2π, length=100))
ys_fine = collect(range(0.0, stop=2π, length=100))
fs_fine = JPEC.SplinesMod.bicube_eval(bcspline, xs_fine, ys_fine)

# Create contour plot
contourf(xs_fine, ys_fine, fs_fine[:, :, 1]',
         title="Bicubic Spline: sin(x)cos(y) + 1")
```

### With Derivatives

```julia
# Evaluate with first derivatives
fs_fine, fsx_fine, fsy_fine = JPEC.SplinesMod.bicube_eval(bcspline, xs_fine, ys_fine, 1)

# Plot function and derivatives
p1 = contourf(xs_fine, ys_fine, fs_fine[:, :, 1]', title="f(x,y)")
p2 = contourf(xs_fine, ys_fine, fsx_fine[:, :, 1]', title="∂f/∂x")
p3 = contourf(xs_fine, ys_fine, fsy_fine[:, :, 1]', title="∂f/∂y")

plot(p1, p2, p3, layout=(1,3), size=(1200, 400))
```

### Equilibrium Example

This example shows spline usage with the Solov'ev equilibrium:

```julia
# Create equilibrium parameters
kappa = 1.8  # elongation
a = 1.0      # minor radius
r0 = 3.5     # major radius
q0 = 1.25    # safety factor

# Create spatial grid
mr, mz = 40, 43
rmin, rmax = r0 - 1.5*a, r0 + 1.5*a
zmin, zmax = -1.5*kappa*a, 1.5*kappa*a

rs = collect(range(rmin, stop=rmax, length=mr))
zs = collect(range(zmin, stop=zmax, length=mz))

# Create Solov'ev equilibrium psi field
f0 = r0 * 1.0  # b0fac = 1.0
psio = kappa * f0 * a^2 / (2 * q0 * r0)
psifac = psio / (a*r0)^2
efac = 1/kappa^2

psifs = zeros(mr, mz, 1)
for i in 1:mr, j in 1:mz
    psifs[i, j, 1] = psio - psifac * (efac * (rs[i] * zs[j])^2 + (rs[i]^2-r0^2)^2/4)
end

# Set up bicubic spline for psi
psi_spline = JPEC.SplinesMod.bicube_setup(rs, zs, psifs, 3, 3)

# Evaluate on fine grid for plotting
rs_fine = collect(range(rmin, stop=rmax, length=110))
zs_fine = collect(range(zmin, stop=zmax, length=100))
psi_fine = JPEC.SplinesMod.bicube_eval(psi_spline, rs_fine, zs_fine)

# Create contour plot
contourf(rs_fine, zs_fine, psi_fine[:, :, 1]',
         title="Ψ: Solov'ev Equilibrium",
         xlabel="R", ylabel="Z",
         aspect_ratio=:equal)
```

## Performance Tips

1. **Batch evaluations**: Evaluate multiple points simultaneously for better performance
2. **Memory management**: Splines should not be re-created frequently; reuse existing spline objects
