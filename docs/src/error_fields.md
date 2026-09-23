# ErrorFields Module

The `ErrorFields` module quantifies how sensitive a perturbed equilibrium's resonant drive is
to the placement of each coil set. It is the foundation of an error-field tolerance
assessment: once one plasma solve exists, the question *"how much resonant field does a
misaligned coil produce?"* is linear in the coil's motion, so the module linearizes it once
and everything downstream (tolerance Monte Carlo, locking risk, allowable-tolerance scans)
becomes linear algebra on a small table.

## What is computed

For every named coil set of a coil-forced run, the module evaluates the root-area-weighted
control-surface spectrum ``\tilde{b}`` of the set as built and its central-difference
derivatives with respect to the six rigid-body degrees of freedom: Cartesian shifts
``(\Delta x, \Delta y, \Delta z)`` in metres and rotations ``(\theta_x, \theta_y, \theta_z)``
about the machine axes in degrees. The spectra are placed on the run's ``(m, n)`` ordering and
conformed with the control-surface operator, so they are exactly the vectors the resonant
coupling matrix and its singular vectors act on
(see [Dominant resonant-coupling mode](perturbed_equilibrium.md#Dominant-resonant-coupling-mode)).

The overlap of a coil set with singular mode ``k`` of the coupling matrix, normalized by the
axis toroidal field, is the dimensionless

```math
\delta = \frac{V_k^{\mathrm{H}}\,\tilde{b}}{B_{T0}},
```

and because the projection is linear, the derivatives of ``\tilde{b}`` project to the
derivatives of ``\delta``. A rigid shift ``(\Delta x, \Delta y)`` therefore moves the overlap
by ``S_x \Delta x + S_y \Delta y`` with complex ``S_x = \partial\delta/\partial\Delta x`` and
``S_y = \partial\delta/\partial\Delta y``, which is exact for any coil shape; for an
axisymmetric coil ``S_y = \pm i S_x`` and the response reduces to a single magnitude with a
free phase, the model the OMFIT tolerance tool used.

The stored primitive is the linearization of the spectrum, not a scalar, so the resonant
surfaces retained, the singular mode, and the normalization are all post-hoc choices.

## Running it

Add an `[ErrorFields]` section to a deck whose `[ForcingTerms]` uses
`forcing_data_format = "coil"` and whose `[PerturbedEquilibrium]` computes the singular
coupling:

```toml
[ErrorFields]
fd_step_shift_m = 1e-3          # Central-difference step for the rigid shifts [m]
fd_step_tilt_deg = 0.1          # Central-difference step for the rigid tilts [degrees]
rotation_center = "conductor"   # Tilt pivot: each conductor's own centre ("conductor") or the whole set's ("set")
write_outputs_to_HDF5 = true    # Write ErrorFields/CoilSensitivities/ to the output file
verbose = false                 # Log per-coil-set progress and linearity diagnostics
```

The stage runs after the perturbed equilibrium and writes `ErrorFields/CoilSensitivities/`:
the spectra `nominal_field`, `shift_sensitivity`, `tilt_sensitivity`, a finite-difference
curvature diagnostic per tap, the current pattern the spectra were evaluated at, and a
`DominantMode/` summary projected onto the run's full-window dominant mode (`delta_nominal`,
its shift and tilt sensitivities, their direction-averaged in-plane magnitudes, and the in-plane
shift and tilt that would cancel `delta_nominal`).

`examples/DIIID-like_error_field_example/` is a complete case: the DIII-D-like equilibrium with
the C-coil as the nominal n = 1 source and the eighteen DIII-D F coils added as single-filament
hoops at the centroids of their winding packs (from OpenFUSIONToolkit's TokaMaker
`DIIID_geom.json`). An axisymmetric hoop drives no n = 1 field as built, so each F coil's
error-field content is entirely its sensitivity to misalignment; `analyze_example.jl` ranks the
coils by error field per millimetre of shift and per tenth of a degree of tilt, over all
rational surfaces and over the edge only.

The tilt pivot matters for multi-filament winding packs: `"conductor"` rotates each filament
about its own arc-length centre, the Fortran `coil_read` convention inherited by the OMFIT
tolerance tool, while `"set"` rotates the pack rigidly about its common centre, which is what
an engineering axis-line tolerance constrains. Single-conductor sets give identical results
either way.

## Analysis after the run

Window the coupling to any range of rational surfaces and project onto any singular mode
without re-running anything, from memory or from the file:

```julia
using GeneralizedPerturbedEquilibrium
EF = GeneralizedPerturbedEquilibrium.ErrorFields

# From the file: the coupling is rebuilt, windowed to 0.5 ≤ ψ_N ≤ 1, and projected onto mode 1
table = EF.sensitivity_table("gpec.h5"; psi_low=0.5)
table.delta_nominal            # complex overlap of each coil set
table.delta_per_mm_shift       # direction-averaged |∂δ/∂Δ| per millimetre of shift, per coil set
table.delta_per_mm_rim         # the same for tilt, as rim displacement at the coil's major radius
table.tilt[1, :]               # ∂δ/∂θx per degree, per coil set

# In memory, from the run's returned state (no I/O)
rc  = PerturbedEquilibrium.ResonantCoupling(run.pe, run.ffs)
dom = PerturbedEquilibrium.dominant_coupling(rc; psi_low=0.5)
table = EF.sensitivity_table(run.coil_sensitivities, dom; mode=1)
```

## Comparing coil revisions

Assessing a *new* coil design against an existing run costs one Biot-Savart pass per coil set and
no plasma solve: the equilibrium, the control surface and the resonant coupling all come from the
stored file. Gather them once with `ResonantDriveContext` and hand it whatever geometry you like.

```julia
ctx = EF.ResonantDriveContext("gpec.h5")                    # nzeta_coil=…, dat_dir=… override the deck
old = EF.coil_overlaps(ctx, ForcingTerms.load_coil_sets(old_cfg, 1))
new = EF.coil_overlaps(ctx, ForcingTerms.load_coil_sets(new_cfg, 1))

new[1].delta               # dimensionless overlap δ = Vᴴb̃ / B_T0
new[1].fraction_percent    # how much of this coil's own spectrum is resonant
new[1].spectrum            # b̃ itself, for the diagnostics below
```

Every normalization the quantity is quoted in travels on the `CoilOverlap`, so a caller
never has to work out which one a bare number was in.

A revision can rename or split coils. Because the field is linear in the currents, a design current
pattern is applied afterwards rather than by re-running the geometry, and a split is written as the
combination it is:

```julia
sets = ForcingTerms.conductors(ForcingTerms.read_coil_dat("old_pair.dat"))   # one file, two coils
ovs  = EF.coil_overlaps(ctx, sets)                                          # each at 1 kA, say

EF.combine_overlaps(ovs, "old_pair_1" => 20.0, "old_pair_2" => -15.0)       # weights in kA
```

Weights multiply each spectrum as it was evaluated; they are not absolute currents, since the sets
may have been swept at any current. Evaluating every conductor at 1 kA and weighting by the design
current in kA is the idiom that makes them read as currents. An unmatched name raises rather than
contributing zero, so a coil renamed between revisions cannot quietly drop out of a comparison.

For sensitivities to rigid motion as well, `compute_coil_sensitivities` takes the same context.
It costs thirteen spectrum evaluations per coil set instead of one, so reach for `coil_overlaps`
when only the overlaps are wanted.

## API Reference

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.ErrorFields]
```

```@docs
GeneralizedPerturbedEquilibrium.equilibrium_from_h5
```
