# Equilibrium Module

The Equilibrium module provides tools to read, construct, and analyze
magnetohydrodynamic (MHD) plasma equilibria. It wraps a variety of
equilibrium file formats (EFIT, CHEASE, SOL, LAR, and others), runs the
appropriate direct or inverse solver, and returns a processed
`PlasmaEquilibrium` object ready for downstream calculations.

## Overview

Key responsibilities of the module:

- Read equilibrium input files and TOML configuration (see `EquilibriumConfig`).
- Provide convenient constructors for analytic / model equilibria (Large
	Aspect Ratio, Solev'ev).
- Build spline representations used throughout the code (1D cubic and
	2D bicubic splines).
- Run the direct or inverse equilibrium solver and post-process results
	(global parameters, q-profile finding, separatrix finding, GSE checks).

The module exposes a small public API that covers setup, configuration,
and common analyses used by other GPEC components (e.g. `ForceFreeStates`, vacuum
interfaces).

## EFIT solver strategies

When `eq_type` is one of the EFIT-based options, three solver strategies are available:

| `eq_type` | Method | Best for |
|-----------|--------|----------|
| `"efit"` | Geometric-angle field-line ODE | Default; fast and robust for standard cases |
| `"efit_arclength"` | Arc-length field-line ODE | Highly elongated plasmas where geometric-angle ODE has singularities near X-points |
| `"efit_by_inversion"` | Contour.jl marching-squares → inverse solver | Highest geometric accuracy; avoids ODE singularities entirely |

`efit_by_inversion` traces flux surface level sets directly from ψ(R,Z) using marching
squares, resamples each closed curve to a uniform geometric-angle grid, and feeds the
result into the inverse equilibrium solver (the same path used by CHEASE input). It is
~70× more accurate in roundtrip error than the ODE methods and handles near-separatrix
surfaces (psihigh up to 0.999) without modification.

**Caveat:** `efit_by_inversion` requires `psilow ≤ 0.01`. For larger psilow values the
innermost grid surfaces are too far from the axis for the inverse solver's boundary
extrapolation to be reliable.

**`refine` parameter:** `efit_by_inversion` evaluates ψ on a Cartesian grid at
`refine×` the EFIT resolution (default `refine = 4`). Higher values improve accuracy
near the separatrix at increased cost; `refine = 2` is sufficient for most production
runs and `refine = 8` gives near-machine-precision roundtrip error.

<details>
<summary><strong>Benchmark results (DIIID-like example, 8 threads)</strong></summary>

**Method comparison at default parameters (psihigh=0.99, psilow=1e-4, mpsi=128, mtheta=256, refine=4):**

| Method | Runtime | Roundtrip max error | q monotonicity violations |
|--------|---------|---------------------|--------------------------|
| `efit` | 0.047 s | 3.15e-04 | 2 (interior, none at edge) |
| `efit_arclength` | 0.056 s | 2.51e-04 | 2 (interior, none at edge) |
| `efit_by_inversion` | 0.096 s | 4.58e-06 | 1 (interior, none at edge) |

**psihigh robustness (all methods succeed through psihigh = 0.999):**

| psihigh | `efit` rt_edge | `efit_arclength` rt_edge | `efit_by_inversion` rt_edge |
|---------|---------------|--------------------------|------------------------------|
| 0.980 | 2.03e-04 | 2.52e-04 | 3.42e-06 |
| 0.990 | 2.28e-04 | 2.67e-04 | 3.84e-06 |
| 0.995 | 1.75e-04 | 2.00e-04 | 3.04e-06 |
| 0.999 | 1.87e-04 | 3.10e-04 | 4.12e-06 |

Zero q monotonicity violations at the edge for all methods at all psihigh values.
q grows as expected toward the separatrix (q ≈ 4.9 at ψ=0.98, q ≈ 6.2 at ψ=0.999).

**psilow near-axis accuracy (q0 extrapolated to axis, true value ≈ 1.21):**

| psilow | `efit` q0 | `efit_arclength` q0 | `efit_by_inversion` q0 |
|--------|-----------|---------------------|------------------------|
| 0.10 | 1.199 | 1.199 | ❌ 11.4 (too far from axis) |
| 0.05 | 1.206 | 1.207 | ❌ 11.7 (too far from axis) |
| 0.01 | 1.209 | 1.209 | ✓ 1.225 |
| 0.001 | 1.210 | 1.210 | ✓ 1.230 |
| 1e-04 | 1.210 | 1.210 | ✓ 1.205 |

**Contour grid refinement sweep (`efit_by_inversion`, mpsi=128, mtheta=256):**

| refine | Runtime | Roundtrip max |
|--------|---------|---------------|
| 2 | 0.09 s | 1.98e-05 |
| 4 (default) | 0.18 s | 6.16e-06 |
| 8 | 0.88 s | 1.17e-06 |

</details>

## API Reference

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Equilibrium]
```

## Important types

- `EquilibriumConfig` — top-level configuration container parsed from a
	TOML file (outer constructor `EquilibriumConfig(path::String)` is
	provided).
- `EquilibriumControl` — low-level control parameters (grid, jacobian
	type, tolerances, etc.).
- `PlasmaEquilibrium` — the runtime structure containing spline fields,
	geometry, profiles, and computed diagnostics (q-profile, separatrix,
	etc.).
- `LargeAspectRatioConfig`, `SolevevConfig` — convenience structs to
	construct analytic/model equilibria when using `eq_type = "lar"` or
	`eq_type = "sol"`.

## Key functions

- `setup_equilibrium(path::String = "equil.toml")`
	— main entry point that reads configuration, builds the equilibrium,
	runs the solver, and returns a `PlasmaEquilibrium` instance.
- `equilibrium_separatrix_find!(pe::PlasmaEquilibrium)` — locate
	separatrix and related boundary geometry in-place.
- `equilibrium_global_parameters!(pe::PlasmaEquilibrium)` — populate
	common global parameters (major radius, magnetic axis, volumes, etc.).
- `equilibrium_qfind!(pe::PlasmaEquilibrium)` — compute safety factor
	(q) information across the grid.
- `equilibrium_gse!(pe::PlasmaEquilibrium)` — diagnostics on the
	Grad–Shafranov solution.

## Example usage

Basic example: read a TOML config and build an equilibrium

```julia
using GeneralizedPerturbedEquilibrium

# Build from a TOML file (searches relative paths if needed)
pe = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium("docs/examples/ForceFreeStates.toml")

println("Magnetic axis: ", pe.params.r0, ", ", pe.params.z0)
println("q(0) = ", pe.params.q0)

# Find separatrix (in-place) and inspect results
GeneralizedPerturbedEquilibrium.Equilibrium.equilibrium_separatrix_find!(pe)
println("rsep = ", pe.params.rsep)
```

Analytic / testing example: construct a large-aspect-ratio model

```julia
using GeneralizedPerturbedEquilibrium

# Create a LAR config from a small TOML fragment or file
larcfg = GeneralizedPerturbedEquilibrium.Equilibrium.LargeAspectRatioConfig(lar_r0=10.0, lar_a=1.0, beta0=1e-3)
pe = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(control=Dict("eq_filename"=>"unused","eq_type"=>"lar")), larcfg)

println("Built LAR equilibrium with a = ", lorcfg.lar_a)
```

## Notes and Caveats

- `EquilibriumConfig` is constructed from the `[Equilibrium]` section of `gpec.toml`.
  Paths that are not absolute are resolved relative to the TOML file location.
- The Equilibrium module contains readers for EFIT and CHEASE formats. Ensure the
  required data files are present and paths are set correctly in `gpec.toml`.
- For `efit_by_inversion`, use `psilow ≤ 0.01` (see benchmark table above).

## See also

- `docs/src/vacuum.md` — coupling between equilibrium and vacuum solvers
