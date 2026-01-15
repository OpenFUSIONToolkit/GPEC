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
and common analyses used by other JPEC components (e.g. `DCON`, vacuum
interfaces).

## API Reference

```@autodocs
Modules = [JPEC.Equilibrium]
```

## Important types

- `EquilibriumConfig` — top-level configuration container parsed from a
	TOML file (outer constructor `EquilibriumConfig(path::String)` is
	provided).
- `EquilibriumControl` — low-level control parameters (grid, jacobian
	type, tolerances, etc.).
- `EquilibriumOutput` — options controlling what output is written.
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
using JPEC

# Build from a TOML file (searches relative paths if needed)
pe = JPEC.Equilibrium.setup_equilibrium("docs/examples/dcon.toml")

println("Magnetic axis: ", pe.params.r0, ", ", pe.params.z0)
println("q(0) = ", pe.params.q0)

# Find separatrix (in-place) and inspect results
JPEC.Equilibrium.equilibrium_separatrix_find!(pe)
println("rsep = ", pe.params.rsep)
```

Analytic / testing example: construct a large-aspect-ratio model

```julia
using JPEC

# Create a LAR config from a small TOML fragment or file
larcfg = JPEC.Equilibrium.LargeAspectRatioConfig(lar_r0=10.0, lar_a=1.0, beta0=1e-3)
pe = JPEC.Equilibrium.setup_equilibrium(JPEC.Equilibrium.EquilibriumConfig(control=Dict("eq_filename"=>"unused","eq_type"=>"lar")), larcfg)

println("Built LAR equilibrium with a = ", lorcfg.lar_a)
```

Notes:

- The `EquilibriumConfig(path::String)` constructor parses TOML and
	expects `[EQUIL_CONTROL]` to contain at minimum `eq_filename` and
	`eq_type` fields. Paths that are not absolute are resolved relative
	to the TOML file location.
- When `eq_type == "inverse_testing"` a small example inverse run is
	constructed (useful in tests and examples).

## Notes and Caveats

- Many routines rely on spline representations; the `Splines` module is
	used heavily and should be initialized where appropriate.
- The Equilibrium module contains several reader routines for external
	formats (EFIT/CHEASE) and also interfaces to older Fortran helpers —
	ensure required data files are present for those backends.
- For programmatic usage, prefer constructing `EquilibriumConfig` from a
	TOML file to ensure all path resolution and defaults are handled.

## See also

- `docs/src/splines.md` — spline helpers used by equilibrium routines
- `docs/src/vacuum.md` — coupling between equilibrium and vacuum solvers

```
