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
	Aspect Ratio, Solovev).
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
| `"efit_arclength"` | Arc-length field-line ODE | Near-separatrix surfaces where the geometric-angle ODE becomes singular |
| `"efit_by_inversion"` | Contour.jl marching-squares → inverse solver | Highest geometric accuracy; avoids ODE singularities entirely |

**`efit`** integrates field lines using the geometric angle η (polar angle from the
magnetic axis, 0→2π) as the independent variable. Position at each step is computed as
`R = R₀ + rfac·cos(η)`, so the RHS denominator is `Bz·cos(η) − Br·sin(η)` — the
projection of B_pol onto the radial direction. Near an X-point where Bp → 0 this
denominator passes through zero, causing a coordinate singularity: the ODE steps
become extremely small or the solver fails to converge.

**`efit_arclength`** uses arc length along the flux surface as the independent variable
instead. The tangent direction `(dR/ds, dZ/ds) = ∇ψ⊥/|∇ψ|` has unit magnitude by
construction, so the position equations have no denominator and remain well-behaved
all the way to the separatrix. The 1/Bp factors in the accumulated integrals still
diverge near X-points, but their solver tolerances are set large so they do not
restrict the step size — only the position tracking drives adaptivity. The two methods
produce identical results when both succeed; `efit_arclength` extends the reliable
range of psihigh closer to 1.

**`efit_by_inversion`** traces flux surface level sets directly from ψ(R,Z) using marching
squares, resamples each closed curve to a uniform geometric-angle grid, and feeds the
result into the inverse equilibrium solver (the same path used by CHEASE input).
The Cartesian evaluation grid is clipped to the separatrix bounding box and its
resolution is set adaptively from a bilinear interpolation error bound, so no manual
tuning is needed.

## Radial grid packing

The default `grid_type = "log_asymptotic"` uses a three-region grid that respects
the asymptotic behavior of q near both the magnetic axis and the separatrix:

- **Core** (ψ < 0.15): geometric spacing in log(ψ) — handles the axis where profiles
  behave as ψⁿ
- **Middle** (0.15 ≤ ψ ≤ 0.95): uniform spacing — preserves resolution through the
  pedestal region
- **Edge** (ψ > 0.95): geometric spacing in log(1−ψ) — tracks the logarithmic
  divergence q ~ −A·ln(1−ψ) near a diverted separatrix

With `mpsi = 0` (the default), the number of radial knots is chosen automatically
from the `psi_accuracy` parameter (target absolute error in q). Two probe field-line
integrations near psihigh estimate the local log-slope A, and the knot count is set so
the cubic spline error stays below `psi_accuracy` throughout the domain. The legacy
`grid_type = "ldp"` (sin²-spaced) and explicit `mpsi` are still supported.

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
- `LargeAspectRatioConfig`, `SolovevConfig` — convenience structs to
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

println("Built LAR equilibrium with a = ", larcfg.lar_a)
```

## Notes and Caveats

- `EquilibriumConfig` is constructed from the `[Equilibrium]` section of `gpec.toml`.
  Paths that are not absolute are resolved relative to the TOML file location.
- The Equilibrium module contains readers for EFIT and CHEASE formats. Ensure the
  required data files are present and paths are set correctly in `gpec.toml`.

## See also

- `docs/src/stability.md` — ideal MHD stability analysis built on top of the equilibrium
