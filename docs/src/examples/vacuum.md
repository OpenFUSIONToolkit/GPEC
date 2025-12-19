# Vacuum Examples

This page demonstrates the usage of the Vacuum module for magnetostatic calculations.

## Basic Vacuum Field Calculation

### Setting up DCON Parameters

The vacuum module requires initialization with DCON (Displacement CONtinuum) parameters:

```julia
using JPEC

# Define DCON parameters
mthin = Int32(4)     # Number of theta points
lmin = Int32(1)      # Minimum poloidal mode number
lmax = Int32(4)      # Maximum poloidal mode number
nnin = Int32(2)      # Toroidal mode number
qa1in = 1.23         # Safety factor parameter

# Create geometry arrays
n_modes = lmax - lmin + 1
xin = rand(Float64, n_modes)      # Radial coordinates
zin = rand(Float64, n_modes)      # Vertical coordinates
deltain = rand(Float64, n_modes)  # Displacement data

# Initialize DCON interface
JPEC.VacuumMod.set_dcon_params(mthin, lmin, lmax, nnin, qa1in, xin, zin, deltain)
```

### Vacuum Matrix Calculation

```julia
# Set up vacuum calculation parameters
mpert = 5              # Number of perturbation modes
mtheta = 256           # Number of theta points for plasma
mthvac = 256           # Number of theta points for vacuum

# Initialize result matrix
wv = zeros(ComplexF64, mpert, mpert)

# Calculation flags
complex_flag = true     # Use complex arithmetic
kernelsignin = -1.0    # Kernel sign
wall_flag = false      # Include wall effects
farwall_flag = true     # Far wall approximation

# Geometry and source data
grrio = rand(Float64, 2*(mthvac+5), mpert*2)  # Geometry data
xzptso = rand(Float64, mthvac+5, 4)           # Source points

# Perform vacuum matrix calculation
JPEC.VacuumMod.mscvac(
    wv, mpert, mtheta, mthvac,
    complex_flag, kernelsignin,
    wall_flag, farwall_flag,
    grrio, xzptso
)

println("Vacuum matrix calculation completed")
println("Result matrix dimensions: ", size(wv))
```

### Analyzing Results

```julia
using Plots

# Plot magnitude of vacuum matrix elements
heatmap(abs.(wv),
        title="Vacuum Matrix |W| Elements",
        xlabel="Mode j",
        ylabel="Mode i",
        color=:plasma)

# Plot phase information
heatmap(angle.(wv),
        title="Vacuum Matrix Phase",
        xlabel="Mode j",
        ylabel="Mode i",
        color=:phase)
```

## Advanced Usage

### Including Wall Effects

```julia
# Enable wall calculations
wall_flag = true
farwall_flag = false  # Do not use far wall approximation

# Additional wall parameters might be needed
# (specific implementation depends on geometry)

JPEC.VacuumMod.mscvac(
    wv, mpert, mtheta, mthvac,
    complex_flag, kernelsignin,
    wall_flag, farwall_flag,
    grrio, xzptso
)
```

### Multi-mode Analysis

```julia
# Analyze eigenvalues of vacuum matrix
using LinearAlgebra

eigenvals = eigvals(wv)
eigenvecs = eigvecs(wv)

# Plot eigenvalue spectrum
scatter(real.(eigenvals), imag.(eigenvals),
        title="Vacuum Matrix Eigenvalues",
        xlabel="Real part",
        ylabel="Imaginary part",
        legend=false)

# Identify most unstable mode
max_growth_idx = argmax(imag.(eigenvals))
println("Most unstable eigenvalue: ", eigenvals[max_growth_idx])
```

### Coupling with Equilibrium Data

```julia
# This example shows how vacuum calculations integrate with equilibrium

# 1. Set up equilibrium (using spline example from equilibrium page)
# ... equilibrium setup code ...

# 2. Extract boundary data for vacuum calculation
# boundary_data = extract_boundary(psi_spline, flux_surface)

# 3. Convert to DCON format
# xin, zin, deltain = process_boundary_data(boundary_data)

# 4. Perform vacuum calculation
# JPEC.VacuumMod.set_dcon_params(mthin, lmin, lmax, nnin, qa1in, xin, zin, deltain)
# ... vacuum calculation ...

# 5. Analyze stability
# stability_analysis(wv, eigenvals)
```

## Troubleshooting

### Common Issues

1. **Initialization Error**: Ensure `set_dcon_params` is called before `mscvac`
2. **Memory Issues**: Large `mtheta`/`mthvac` values require significant memory
3. **Convergence**: Check that geometry arrays are properly normalized
4. **Complex Arithmetic**: Ensure `complex_flag=true` for stability analysis

### Performance Optimization

```julia
# For large problems, consider:
# - Reducing mtheta/mthvac if possible
# - Using real arithmetic (complex_flag=false) when appropriate
# - Parallelization (if available in Fortran backend)

# Monitor memory usage
using Pkg
Pkg.add("BenchmarkTools")
using BenchmarkTools

@time JPEC.VacuumMod.mscvac(wv, mpert, mtheta, mthvac,
                           complex_flag, kernelsignin,
                           wall_flag, farwall_flag,
                           grrio, xzptso)
```
