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
table.shift_rms                # direction-averaged |∂δ/∂Δ| per metre of shift, per coil set
table.tilt[1, :]               # ∂δ/∂θx per degree, per coil set

# In memory, from the run's returned state (no I/O)
rc  = PerturbedEquilibrium.ResonantCoupling(run.pe, run.ffs)
dom = PerturbedEquilibrium.dominant_coupling(rc; psi_low=0.5)
table = EF.sensitivity_table(run.coil_sensitivities, dom; mode=1)
```

To assess a *new* coil design against an existing run — a moved or re-wound coil — sweep the
new geometry on the stored solve's own control surface. The equilibrium is rebuilt from the
file; no plasma solve is repeated:

```julia
new_sets = ForcingTerms.load_coil_sets(cfg, 1)   # any CoilSet vector
sens  = EF.compute_coil_sensitivities("gpec.h5", new_sets; rotation_center="set")
table = EF.sensitivity_table(sens, PerturbedEquilibrium.dominant_coupling(PerturbedEquilibrium.ResonantCoupling("gpec.h5")))
```

A coil set's spectrum is proportional to its currents, so the table is specific to the current
pattern it was evaluated with; `peak_current` and `winding_multiplier` are stored so a user can
renormalize per ampere-turn when comparing designs.

## API Reference

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.ErrorFields]
```

```@docs
GeneralizedPerturbedEquilibrium.equilibrium_from_h5
```
