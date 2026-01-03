# Benchmarking Absolute Tolerance Implementation

This document explains how to benchmark the absolute tolerance improvements to the ODE solver.

## Quick Test

To see the current implementation in action:

```bash
julia --project=. compare_ode_tolerances.jl
```

This will run one test case and show:
- Total integration steps
- Steps spent in deep core (ψ < 0.1)
- Lowest eigenvalue
- The `atol_scale` value used

## Full Benchmark Comparison

To properly compare performance with and without absolute tolerance:

### Step 1: Create Baseline (No Absolute Tolerance)

Temporarily disable absolute tolerance:

1. Edit `src/DCON/Ode.jl`, line 445:
   ```julia
   # Change this line:
   atol = odet.atol_scale > 0 ? odet.atol_scale * tol : 0.0

   # To:
   atol = 0.0  # Disable atol for baseline
   ```

2. Run benchmark and save results:
   ```bash
   julia --project=. benchmark_ode_atol.jl > baseline_results.txt
   ```

### Step 2: Run with Absolute Tolerance

1. Revert the change from Step 1 (restore the original line 445)

2. Run benchmark again:
   ```bash
   julia --project=. benchmark_ode_atol.jl > atol_results.txt
   ```

### Step 3: Compare Results

Compare the two output files:
```bash
diff baseline_results.txt atol_results.txt
```

## Expected Improvements

Based on the issue description (#122), we expect:

1. **Fewer total steps**: 10-30% reduction
2. **Significantly fewer deep core steps**: 40-60% reduction in steps where ψ < 0.1
3. **Same physics results**: Eigenvalues should match within numerical precision (< 0.1% difference)

## Key Metrics to Check

1. **Total integration steps**: Should decrease
2. **Deep core percentage**: Should decrease from ~25% to ~10-15%
3. **Lowest eigenvalue**: Should remain essentially unchanged (validates that accuracy is preserved)
4. **Integration time**: Should decrease proportionally to step reduction

## Running Full Test Suite

To verify all tests still pass:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Or run specific test files:
```bash
julia --project=. test/runtests_ode.jl
julia --project=. test/runtests_fullruns.jl
```

## Implementation Details

The absolute tolerance strategy works by:

1. **Sampling solution magnitudes** at "safe" locations (half-integer q values between rational surfaces)
2. **Avoiding the deep core** (q < 1.0) where solutions are tiny
3. **Setting atol_scale** to a fraction (1e-6) of the sampled magnitude
4. **Updating periodically** every 5 rational surfaces to adapt as solutions grow

This allows the solver to take larger steps in regions where solutions are legitimately small (core) without sacrificing accuracy in regions where they matter (edge).
