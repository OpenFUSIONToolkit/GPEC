# Type Stability Tests

This document describes the unit tests added to verify the type stability improvements made to DCON and Equilibrium structs.

## Overview

Type stability in Julia means that the compiler can infer the concrete types of all variables at compile time. When struct fields use abstract `Union` types like `Union{Missing, T}` or `Union{Nothing, T}`, the compiler cannot determine the actual type until runtime, leading to performance overhead.

The changes in PR #88 eliminate these abstract unions by providing concrete default initializations.

## Test Files

### `runtests_spline.jl`

Extended with new test suites for empty spline constructors:

1. **Empty Spline Constructors**: Verifies that `empty_CubicSpline`, `empty_BicubicSpline`, and `empty_FourierSpline` create properly initialized splines with:
   - `C_NULL` handles
   - Empty arrays with correct dimensions
   - Correct type parameters
   - All work arrays allocated

2. **Assertion Guards**: Tests that uninitialized (C_NULL) splines throw `AssertionError` when evaluated, preventing silent failures

3. **Spline Replacement**: Tests that empty splines can be replaced with real splines, demonstrating the intended usage pattern

### `runtests_type_stability.jl`

DCON-specific test suite that verifies:
- `SingType` uses concrete array types (no `Union{Missing, ...}`)
- `DconInternal` uses concrete spline types (no `Union{Missing, ...}`)
- `FourFitVars` has all 9 spline fields as concrete types
- `OdeState` has `wvmat_spline` as a concrete type
- All fields can be queried with `fieldtype()` to confirm no Union types

### `test_type_stability_standalone.jl`

Standalone test that can run without building the Fortran libraries. Uses mock structs to verify:
- Empty constructor patterns work correctly
- Struct initialization removes Union types
- Type inference can determine concrete types at compile time

This is useful for quick validation during development.

## Running the Tests

### With Full Build

```bash
# Run all tests including type stability
julia --project=. test/runtests.jl

# Run only type stability tests
julia --project=. test/runtests.jl runtests_type_stability.jl
```

### Standalone (No Build Required)

```bash
# Run standalone mock tests
julia test/test_type_stability_standalone.jl
```

## What These Tests Verify

### Before the Changes

```julia
@kwdef mutable struct FourFitVars
    amats::Union{Missing, Spl.CubicSpline{ComplexF64}} = missing
    # ^^^ Abstract Union type - runtime dispatch required
end

ffv = FourFitVars(mpert=5, mband=4)
# Compiler doesn't know if ffv.amats is Missing or CubicSpline
```

### After the Changes

```julia
@kwdef mutable struct FourFitVars
    amats::Spl.CubicSpline{ComplexF64} = Spl.empty_CubicSpline(ComplexF64)
    # ^^^ Concrete type - compiler knows exact type
end

ffv = FourFitVars(mpert=5, mband=4)
# Compiler knows ffv.amats is CubicSpline{ComplexF64}

# Empty spline has C_NULL handle and will error if used
@assert ffv.amats.handle != C_NULL "CubicSpline has not been initialized"
```

## Benefits Verified by Tests

1. **Type Inference**: `fieldtype()` tests confirm no Union types remain
2. **Type Safety**: Assertion tests confirm uninitialized splines fail fast
3. **Correctness**: Constructor tests verify proper initialization
4. **Usability**: Replacement tests show the intended usage pattern

## Related Changes

These tests cover the changes made in:
- `src/Splines/CubicSpline.jl` - Added `empty_CubicSpline()` and assertions
- `src/Splines/BicubicSpline.jl` - Added `empty_BicubicSpline()` and assertions
- `src/Splines/FourierSpline.jl` - Added `empty_FourierSpline()` and assertions
- `src/DCON/DconStructs.jl` - Removed Union types from multiple structs
- `src/Splines/Splines.jl` - Exported new empty constructor functions
