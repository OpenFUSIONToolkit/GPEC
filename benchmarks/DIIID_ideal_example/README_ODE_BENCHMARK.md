# ODE Solver Benchmark

This benchmark tests different OrdinaryDiffEq.jl solver methods for the DCON ODE integration to determine optimal performance for problems approaching singularities.

## Purpose

The DCON module integrates the Euler-Lagrange equations for MHD stability analysis, which involves approaching rational surface singularities. This benchmark helps determine which ODE solver methods are most efficient for this specific problem.

## Solver Methods Tested

1. **Tsit5** - Current default, Tsitouras 5/4 Runge-Kutta method (non-stiff)
2. **AutoTsit5** - Automatic stiffness detection with Tsit5/Rosenbrock23 switching
3. **Vern6** - 6th order Verner method (efficient, high accuracy)
4. **Vern7** - 7th order Verner method (higher accuracy)
5. **Vern8** - 8th order Verner method (very high accuracy)
6. **DP5** - Dormand-Prince 5th order (classic method)
7. **BS5** - Bogacki-Shampine 5th order (good for moderate accuracy)

These solvers were chosen based on:
- Explicit methods suitable for non-stiff or moderately stiff problems
- Methods with adaptive step sizing (important for singularities)
- Methods with higher-order accuracy options
- AutoTsit5 for automatic stiffness detection near singularities

## Tolerance Scan

The benchmark scans the following tolerance values (with `tol_r = tol_nr` for each run):
- 1e-3
- 1e-4
- 1e-5
- 1e-6
- 1e-8

## Running the Benchmark

From the `benchmarks/DIIID_ideal_example/` directory:

```bash
julia --project=../.. ode_solver_benchmark.jl
```

Note: This benchmark can take a significant amount of time (potentially several hours) as it runs multiple solvers at multiple tolerance levels with statistical sampling.

## Output Files

The benchmark generates:

1. **ode_solver_steps_vs_tolerance.png** - Plot showing number of integration steps vs tolerance for each solver
2. **ode_solver_timing_vs_tolerance.png** - Plot showing integration time vs tolerance for each solver
3. **ode_solver_efficiency.png** - Scatter plot showing time vs steps, colored by tolerance level
4. **ode_solver_benchmark_results.jld2** - Raw results data for further analysis

## Interpreting Results

### Step Count Plot
- Lower is better (fewer steps = more efficient stepping)
- Shows how aggressively each solver adapts step sizes
- Expect increasing steps as tolerance tightens

### Timing Plot
- Lower is better (faster integration)
- Trade-off between step count and per-step cost
- Optimal solver minimizes total time, not necessarily steps

### Efficiency Plot
- Points closer to bottom-left are more efficient
- Compare solvers at same tolerance level (same marker shape)
- Identifies sweet spot between accuracy and performance

## Example Analysis

After running, look for:
1. Which solver has best timing at your target tolerance (likely 1e-6 or 1e-8)
2. Whether higher-order methods (Vern7, Vern8) are more efficient at tight tolerances
3. Whether AutoTsit5 adapts well near singularities
4. Any solver failures (appears as missing data points)

## Modifying the Benchmark

To add more solvers, edit the `solver_configs` array in `ode_solver_benchmark.jl`:

```julia
solver_configs = [
    # ... existing solvers ...
    (name="NewSolver", solver=NewSolver(), description="Description"),
]
```

To change tolerance values, edit:

```julia
tolerance_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-8]
```

## Related Files

- `src/DCON/Ode.jl` - Main ODE integration code (line 301 contains current solver choice)
- `examples/DIIID-like_ideal_example/dcon.toml` - DCON configuration with tolerance settings
- Issue #115 - Original feature request for this benchmark
