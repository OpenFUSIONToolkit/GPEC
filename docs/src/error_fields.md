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

## Tolerance input

Manufacturing tolerances are engineering data, reviewed and versioned independently of any
run, so they live in their own TOML file named by `tolerance_file` in `[ErrorFields]` (a path
relative to the run directory). The run validates the file, echoes its text into
`Input/RawInputs/ErrorFields/tolerance_toml_raw` for replay, and leaves the device numbers
where they belong: outside the repository. The sampling and Monte Carlo stages that consume
these tolerances follow in later releases; this release fixes the format.

```toml
# Fallbacks for keys a coil block leaves out (every key optional)
[ErrorFields.defaults]
shift_sigma_mm = 0.0             # Gaussian uncertainty added to the sampled shift [mm]
tilt_sigma = 0.0                 # Gaussian uncertainty added to the sampled tilt, in tilt_units
tilt_units = "deg"               # "deg", or "m" for a rim displacement converted through the nominal radius
radial_shape = "hollow"          # Radial sampling density on the disk: flat, uniform_area, hollow, or ring
tolerance_model = "additive"     # "additive" (independent shift and tilt) or "cylinder" (correlated axis line)
cylinder_half_height_m = 0.0     # Cylinder half-height for the cylinder model [m]

# One block per coil set that moves on its own
[[ErrorFields.coil]]
name = "PF1U"                    # Coil set name; must match a [[ForcingTerms.coil_set]] of the run
shift_tol_mm = 0.5               # Radius of the in-plane displacement disk the coil centre may lie in [mm]
tilt_tol = 0.019                 # Tilt tolerance, in tilt_units
tolerance_model = "cylinder"     # Axis line confined to a cylinder; shift and tilt drawn together
cylinder_half_height_m = 1.5     # Cylinder half-height; sets the tilt reachable at the given radius [m]

# Coil sets that move together: one shared draw per sample
[[ErrorFields.coherent_group]]
name = "upper_pf_brace"          # Label of the coherently moving group
members = ["PF1U", "PF2U"]       # Coil sets sharing the draw
shift_tol_mm = 2.0               # Coherent shift amplitude shared by the group [mm]
tilt_tol = 0.0                   # Coherent tilt amplitude shared by the group, in tilt_units
phase_group = "braces"           # Groups naming the same phase_group share the random direction
rotation_center_z_m = 0.0        # Height of the pivot of the group's rigid rotation on the machine axis [m]

[ErrorFields.correctability]
uncorrectable_coils = ["EFCC_U"] # Coil sets the error-field correction cannot reduce
efc_factor = 2.0                 # Divisor applied to correctable contributions under error-field correction

[ErrorFields.other_field]
magnitude = 8.2e-6               # Overlap budget of sources not attributed to any coil
sigma = 0.0                      # Gaussian uncertainty on that budget
radial_shape = "ring"            # Radial sampling density of the budget magnitude
```

The shift tolerance is one number, the radius of the disk the coil centre may sit in; the
direction is sampled. A tilt may be given in degrees or, as legacy tolerance tables do, as a
rim displacement in metres, which `tilt_tolerance_deg` converts through the coil set's
arc-length-weighted major radius exactly as the coil loader's `tilt_in_meters` does. A coherent
group's tilt is a rigid rotation of all its members about a pivot on the machine axis at
`rotation_center_z_m`, so a member at height `z` also shifts laterally by `(z − z_pivot)·θ`.
Groups sharing a `phase_group` label draw the same random direction with independent
amplitudes. A group is correctable only if none of its members is listed as uncorrectable.
Coil sets of the run without a tolerance block contribute their nominal overlap only.

Every key is checked against the schema, so a misspelled key is an error rather than a silent
default. `read_tolerance_toml` parses a file into a `ToleranceSet`, and `validate_tolerances`
checks its names against a run's coil sets.

## Sampling a tolerance

A tolerance is one number, the radius of the disk the coil centre may lie in; the direction is
random and the radial density is a `RadialDistribution`: `Flat` (uniform in radius, the OMFIT
tool's `flat`), `UniformArea` (uniform over the disk), `Hollow` (peaked toward the edge, the
OMFIT default), `Ring` (always on the edge), or `PowerLaw(p)`. `sample_disk` draws a point as
`Δx + iΔy`, `sample_uncertainty` adds the Gaussian uncertainty on where the coil actually sits,
and the two tolerance models combine them: `sample_additive` draws a shift (metres) and a tilt
(degrees) independently, while `sample_cylinder` confines the coil's axis line to a cylinder
of radius `R` and half-height `z_top`, drawing its two endpoints and deriving the correlated
midplane shift and lean. Tilts are the rotation angles `θx + iθy` about the machine axes in
the sense `apply_transforms` uses. Every sampler takes the random generator and optional fixed
directions, so coherent groups sharing a direction pass one phase and draw their own radii.

```julia
rng = Random.Xoshiro(1)
Δ = EF.sample_disk(rng, 0.5e-3, EF.Hollow())          # a point in a 0.5 mm disk, m
Δ, θ = EF.sample_cylinder(rng, 0.5e-3, 1.5, EF.randpow(EF.Flat()))   # correlated shift [m] and tilt [deg]
```

## Tolerance Monte Carlo

With a `tolerance_file` named, the run samples every coil set's misalignment within its
tolerance and histograms the dominant-mode overlap `|δ|`: per sample and coil set the overlap
moves by `S·(Δ + u) + T·(θ + v)` for the coil's own draw (`Δ`, `θ`) and Gaussian placement
uncertainties (`u`, `v`); coherent groups add one shared draw per group, with the lateral shift
a rigid rotation about the group pivot gives each member; the unattributed budget adds a random
direction. Because the overlap is linear in the misalignments, a million samples take about a
second and no field is recomputed. Two histograms are written to `ErrorFields/MonteCarlo/`: the
intrinsic `|δ|` and the corrected one, in which every correctable term (coil sets and groups not
listed as uncorrectable, and the unattributed budget) is divided by `efc_factor`. Batches are
seeded individually, so results are bit-identical for any thread count, and their spread is the
statistical error bar of anything derived from them.

```toml
[ErrorFields]
tolerance_file = "tolerances.toml"      # Manufacturing-tolerance TOML, relative to the run directory

[ErrorFields.MonteCarlo]
nsample = 1000000               # Samples per batch
nbatch = 10                     # Independent batches; their spread is the statistical error bar
seed = 1                        # Base seed; batch b uses Xoshiro(hash((seed, b)))
nbins = 300                     # Histogram bins, linear on [0, delta_max]
delta_max = 0.0                 # Upper histogram edge; 0 = 1.5 × the worst-case alignment bound
tolerance_scale = 1.0           # Multiplies every shift and tilt tolerance (for tolerance scans)
coil_subset = []                # Coil sets whose tolerances are sampled; empty = all
```

The run's histogram is the full-window, dominant-mode summary. Any other window or mode, a
tolerance scale, or a coil subset is a post-hoc re-run of the same kernel:

```julia
mc = EF.run_monte_carlo("gpec.h5"; psi_low=0.5, tolerance_scale=2.0, coil_subset=["PF1U", "PF2U"])
mc.pdf, mc.bin_edges           # intrinsic |δ| density
mc.pdf_efc                     # corrected
mc.mean_abs_delta, mc.delta_nominal
```

## Locking risk and allowable tolerance

An overlap distribution becomes a locking risk through the empirical ITPA penetration-threshold
scalings (n = 1: Logan et al., *Plasma Phys. Control. Fusion* **62**, 084001 (2020); n = 2:
Logan et al., *Nucl. Fusion* **60**, 086010 (2020)):
`δ_thresh = 10^α_c · n_e^α_n · B_T^α_B · R_0^α_R · (β_N/l_i)^α_β`. Sampling the fitted exponents
within their standard errors turns the threshold into a distribution; its cumulative
distribution is the probability that an overlap `δ` locks, and the locking probability of the
assembled machine is `100 ∫ pdf(δ) P(lock|δ) dδ` over the Monte Carlo bins, per batch. The
operating point is an `[ErrorFields.scenario]` table: density must be given (it is not an
equilibrium output); field, major radius, β_N and l_i default from the equilibrium.

```toml
[ErrorFields.scenario]
n_e = 5.0                       # Electron density for the threshold scaling [1e19 m^-3]

[ErrorFields.Risk]
dataset = "O,L"                 # ITPA dataset of the threshold fit: "O,L" or "O,L,H" (n = 1); "O,L", "O,L,-C", "O,L,N" (n = 2)
fit = "WLS"                     # Fitting method: "OLS", "DSOLS", or "WLS"
distribution = "normal"         # How the fit exponents are sampled: "normal", "flat", or "normal_truncated"
nsample_threshold = 1000000     # Threshold samples
seed = 1                        # Seed of the threshold sampling
scan_scales = [0.25, 0.5, 1.0, 2.0, 4.0]   # Tolerance multipliers of the allowable-tolerance scan (empty: no scan)
```

`ErrorFields/Risk/` holds the threshold density and `P(lock|δ)` on the Monte Carlo grid, the
locking probability of the intrinsic and corrected distributions (with per-batch values), the
as-designed risk, and the sharp-threshold risk; `ErrorFields/Risk/ToleranceScan/` the risk
against tolerance scale. The scan is the stored quantity; the allowable tolerance for a target
risk is a post-hoc inversion, and every window or fit choice is re-evaluated from the file:

```julia
scan = EF.ToleranceScan("gpec.h5")
EF.allowable_tolerance(scan, 1.0)                  # tolerance multiplier at 1 % locking risk
EF.allowable_tolerance(scan, 1.0; corrected=true)  # with error-field correction
risk = EF.locking_risk("gpec.h5"; n_e=5.0, psi_low=0.7, risk_ctrl=EF.RiskControl(; dataset="O,L,H"))
scan2 = EF.tolerance_scan("gpec.h5"; n_e=5.0, scales=[0.5, 1, 2, 4], coil_subset=["F6A", "F7A"])
```

## Plots and coil-array phasing

`Analysis.ErrorFields` plots everything above from `gpec.h5`, and every function takes a list
of `label => path` pairs so coil-design revisions overplot on one axis:

```julia
AEF = GeneralizedPerturbedEquilibrium.Analysis.ErrorFields
AEF.plot_coil_sensitivities(["rev A" => "revA/gpec.h5", "rev B" => "revB/gpec.h5"]; quantity=:shift)
AEF.plot_tolerance_pdf("gpec.h5"; corrected=true)
AEF.plot_locking_risk("gpec.h5"; target_percent=1.0)     # marks the allowable scale
AEF.plot_threshold_scaling("gpec.h5")
AEF.plot_dominant_mode_spectrum("gpec.h5")
AEF.plot_error_field_summary("gpec.h5"; save_path="error_field_summary.png")
```

When several independently powered coil arrays share the job of correcting the error field,
the relative phases of their current patterns decide how much dominant-mode field they can
drive per ampere-turn. `phasing_map` evaluates `|Σ_k δ_k e^{iφ_k}|` per kilo-ampere-turn and
the resonant fraction of the applied field on a grid of the `N − 1` relative phases from the
stored nominal spectra, a closed form with no optimizer, and `plot_phasing_map` draws it (a
line for two arrays, a contour for three):

```julia
pmap = EF.phasing_map("gpec.h5", ["EFCC_L", "EFCC_M", "EFCC_U"]; psi_low=0.5)
EF.extreme_phasing(pmap)                        # best |δ| per kAt and the phases giving it
AEF.plot_phasing_map(pmap; quantity=:overlap_percent)
```

## NTV limits of error-field correction

A correction coil cancels the dominant-mode overlap at `C_c` per kilo-ampere-turn, but the
non-resonant remainder of its field drives a neoclassical toroidal viscosity (NTV) torque
`T·I²` that a perfect correction does not remove. With a torque budget `T_0` and the threshold
taken to fall in proportion to the torque spent, the current that corrects an intrinsic overlap
`δ_EF` solves `δ_EF − C_c I = s δ_thresh (1 − T_residual I²/T_0)`, and real roots exist only up
to a largest correctable overlap. `[ErrorFields.NTV]` names the correction arrays; the run
evaluates each one's `C_c`, resonant fraction, and NTV torque per kAt² for its whole field and
for its field with the dominant mode projected out — two plasma-response evaluations of the
unit-current spectrum followed by the kinetic torque, which needs a `[KineticForces]` section —
and writes `ErrorFields/NTV/`. The spectrum is evaluated on the run's `[ForcingTerms]` boundary
grid and normalized by the magnitude of the array's ampere-turns, `|nw| × max|I|`, so the winding
sense and current pattern of the deck stay the phase reference; the torques are stored with their
sign, and the limits consume the budget with their magnitude (the sign depends on the rotation
and on conventions, so a negative torque is never read as no torque). Torque budget, threshold
and safety factor are analysis choices:

```toml
[ErrorFields.NTV]
efc_coils = ["d3d_c"]           # Coil set names of the correction arrays to evaluate
method = "fgar"                 # KineticForces torque method (must be enabled in [KineticForces])
```

```julia
couplings = EF.read_efc_couplings("gpec.h5")
curve = EF.efc_current_curve(couplings[1]; delta_threshold=1.4e-4, torque_budget=4.0)
EF.max_correctable_overlap(couplings[1]; delta_threshold=1.4e-4, torque_budget=4.0)
AEF.plot_efc_ntv_limits("gpec.h5"; torque_budget=4.0)   # threshold from the run's Risk/ group
```

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
