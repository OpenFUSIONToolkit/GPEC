# Vacuum

`Vacuum` computes the magnetostatic response of the region between the plasma boundary and a conducting wall, or of free space when there is no wall, following [Chance (1997)](citations.md#Vacuum) and [Chance et al. (2007)](citations.md#Vacuum).
It runs inside `ForceFreeStates` whenever `vac_flag = true`, and its matrices are shared with `PerturbedEquilibrium`.

## What it computes

For a normal displacement ``\xi`` of the plasma boundary, expanded in the poloidal harmonics of the run, the perturbed vacuum field is found from a Green's-function integral over the boundary and the wall, and the vacuum energy is the quadratic form ``\delta W_v = \xi^\dagger W_v \xi``.
``W_v`` is the vacuum part of the free-boundary energy, ``W = W_\mathrm{plasma} + W_\mathrm{vacuum}``, whose eigenvalues decide the ideal kink stability on the [Stability](stability.md) page: a close conducting wall raises ``W_v`` and stabilizes external modes, no wall gives the free-space limit.
The stored matrix `wv` is scaled by the resonance factors ``(m - nq)(m' - nq)`` of its row and column harmonics, the convention of Chance (1997).
When the run needs it, the module also returns the vacuum surface-current matrix ``I_v``, which `PerturbedEquilibrium` inverts into the surface inductance ``L`` of the plasma response.

## Running it

The wall is the optional `[Wall]` section of the deck; omit it for no wall.
`shape` selects one of five shapes, each with its own keys, and `equal_arc_wall` (default `true`) distributes the wall nodes by equal arc length:

```toml
# one [Wall] section per deck; the five shapes:
[Wall]
shape = "conformal"       # the plasma boundary expanded by standoff_fraction minor radii
standoff_fraction = 0.3

[Wall]
shape = "elliptical"      # an ellipse of the given minor radius
minor_radius = 0.3

[Wall]
shape = "dee"             # a D-shaped wall: standoff_fraction, half_thickness, elongation, center_offset, triangularity, corner_sharpness
standoff_fraction = 0.3
elongation = 1.5
triangularity = 0.5

[Wall]
shape = "mod_dee"         # the same shape placed by minor_radius instead of standoff_fraction
minor_radius = 0.3

[Wall]
shape = "file"            # a wall contour read from a file
file = "wall.dat"
```

`examples/Solovev_ideal_example` holds a near-marginal kink by a conformal wall; the DIII-D-like examples run without one.
In a script the shapes are `ConformalWall`, `EllipticalWall`, `DeeWall`, `ModDeeWall` and `FileWall`, passed as the `wall` keyword of the solve (`nothing` for no wall), and the response itself is

```julia
using GeneralizedPerturbedEquilibrium

inputs = Vacuum.VacuumInput(; r=plasma_r, z=plasma_z, ν=nu, mlow=1, mpert=10, n=1, mtheta=256, kernelsign=1.0)
vac = Vacuum.compute_vacuum_response(inputs, Vacuum.ConformalWall(; standoff_fraction=0.3))
vac.wv, vac.I_v, vac.plasma_pts, vac.wall_pts
```

where `ν(ψ, θ)` is the toroidal-angle offset of the straight-field-line coordinates, ``\phi = 2\pi\zeta + \nu`` with ``\phi`` the internal toroidal angle of the [Conventions Reference](conventions.md) (the laboratory angle carries the helicity sign).
With `nzvac > 1` the response is evaluated on a toroidal grid, with the singular correction of the [3D kernel](citations.md#3D-vacuum-singular-correction) (`examples/Solovev_ideal_example_3D`).

## Outputs

| Dataset | Meaning |
|---|---|
| `ForceFreeStates/FreeBoundaryStability/W_vacuum`, `vacuum_eigenvalue` | the vacuum energy matrix on the control surface and its least eigenvalue |
| `PerturbedEquilibrium/ResponseMatrices/surface_inductance` | the surface inductance ``L`` derived from ``I_v`` |
| `SurfaceGeometries/Plasma/x`, `y`, `z` and `SurfaceGeometries/Wall/x`, `y`, `z` | the plasma boundary and wall point clouds, for visualization |

## Numerical notes

- The Legendre functions of the Green's function are evaluated from Bulirsch's elliptic integrals, and by 32-point Gaussian quadrature for large arguments (``n\hat\rho \geq 0.1``).
- For n = 0 with a closed wall the kernel is regularized automatically.

## API reference

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Vacuum]
```

```@docs
GeneralizedPerturbedEquilibrium.Vacuum.compute_vacuum_response
```
