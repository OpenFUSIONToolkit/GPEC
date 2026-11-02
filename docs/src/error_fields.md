# Error Fields

`ErrorFields` turns a coil-driven perturbed equilibrium into an error-field tolerance assessment: how much resonant field each coil set drives when it is misaligned, how likely the assembled machine is to lock for given manufacturing tolerances, and how much correction current an error-field correction array may and must carry.
Once one plasma solve exists the resonant drive is linear in each coil's motion, so the module linearizes it once and everything downstream is linear algebra on a small table.

## What it computes

1. **Coil sensitivities** (`[ErrorFields]`, `ErrorFields/CoilSensitivities/`): the resonant spectrum of each coil set as built and its derivatives with respect to rigid shifts and tilts.
2. **Tolerances** (`tolerance_file`): the manufacturing tolerances of each coil set, in their own TOML file.
3. **Monte Carlo** (`[ErrorFields.MonteCarlo]`, `ErrorFields/MonteCarlo/`): the distribution of the dominant-mode overlap ``|\delta|`` over sampled misalignments, intrinsic and with error-field correction.
4. **Locking risk** (`[ErrorFields.scenario]`, `[ErrorFields.Risk]`, `ErrorFields/Risk/`): the overlap distribution against the empirical penetration-threshold scaling, giving the locking probability and its scan against tolerance scale, from which the allowable tolerance follows.
5. **Correction limits** (`[ErrorFields.NTV]`, `ErrorFields/NTV/`): how much correction current each array may carry before the NTV torque of its non-resonant field spends the rotation budget, and how much it needs.

For every named coil set of a coil-forced run the stage evaluates the root-area-weighted control-surface spectrum ``\tilde b`` of the set as built and its central-difference derivatives with respect to the six rigid-body degrees of freedom, the shifts ``(\Delta x, \Delta y, \Delta z)`` in metres and the rotations ``(\theta_x, \theta_y, \theta_z)`` about the machine axes in degrees, placed on the run's (m, n) ordering so that they are exactly the vectors the resonant coupling matrix and its singular vectors act on (see [Dominant resonant-coupling mode](perturbed_equilibrium.md#Dominant-resonant-coupling-mode)).
The stored primitive is the linearization of the spectrum, not a scalar, so the resonant surfaces retained, the singular mode and the normalization are post-hoc choices, and every later step can be re-evaluated from `gpec.h5` without a new plasma solve.
In the snippets below `EF` is `GeneralizedPerturbedEquilibrium.ErrorFields` and `AEF` is `GeneralizedPerturbedEquilibrium.Analysis.ErrorFields`.

## Governing equations

The overlap of a coil set with singular mode ``k`` of the coupling matrix, normalized by the axis toroidal field, is the dimensionless

```math
\delta = \frac{V_k^{\mathrm{H}}\,\tilde{b}}{B_{T0}},
```

and because the projection is linear, a rigid shift ``(\Delta x, \Delta y)`` moves it by ``S_x \Delta x + S_y \Delta y`` with the complex sensitivities ``S_x = \partial\delta/\partial\Delta x``, ``S_y = \partial\delta/\partial\Delta y``, exact for any coil shape; for an axisymmetric coil ``S_y = \pm i S_x`` and the response reduces to one magnitude with a free phase.
Over sampled misalignments the overlap of the machine is the coherent sum over coil sets of ``S\,(\Delta + u) + T\,(\theta + v)`` for each coil's draw ``(\Delta, \theta)`` and placement uncertainty ``(u, v)``, plus one shared draw per coherent group and a random-direction unattributed budget; no field is recomputed, so a million samples take about a second.

An overlap distribution becomes a locking risk through the empirical ITPA penetration-threshold scalings of [Logan et al. (2020)](citations.md#ErrorFields) for n = 1 and n = 2 and [Bursch et al. (2026)](citations.md#ErrorFields) for n = 1,

```math
\delta_\mathrm{thresh} = 10^{\alpha_c}\, n_e^{\alpha_n}\, B_T^{\alpha_B}\, R_0^{\alpha_R}\, (\beta_N/l_i)^{\alpha_\beta}\, I_p^{\alpha_I},
```

with ``n_e`` in 10¹⁹ m⁻³, ``B_T`` in T, ``R_0`` in m and ``I_p`` in MA (the current term only in the 2026 fits).
Sampling the fitted exponents within their standard errors turns the threshold into a distribution whose cumulative distribution is the probability ``P(\mathrm{lock}\,|\,\delta)`` that an overlap locks, and the locking probability of the assembled machine is ``100 \int \mathrm{pdf}(\delta)\, P(\mathrm{lock}\,|\,\delta)\, d\delta`` over the Monte Carlo bins, per batch.

A correction array cancels the dominant-mode overlap at ``C_c`` per kilo-ampere-turn, but the non-resonant remainder of its field drives an NTV torque ``T I^2`` that a perfect correction does not remove.
With a torque budget ``T_0`` and the threshold taken to fall in proportion to the torque spent, the current that corrects an intrinsic overlap ``\delta_\mathrm{EF}`` solves

```math
\delta_\mathrm{EF} - C_c I = s\,\delta_\mathrm{thresh}\left(1 - \frac{T_\mathrm{residual} I^2}{T_0}\right),
```

the model of [Logan et al. (2026)](citations.md#ErrorFields) for SPARC and [Leuthold et al. (2026)](citations.md#ErrorFields) for ARC.
It holds only while the plasma still rotates, so the current is capped at ``I_\max = \sqrt{T_0/T_\mathrm{residual}}``, and the largest correctable overlap is the peak of the quadratic or the overlap cancelled at ``I_\max``, whichever comes first; the two zero-rotation limits ``C_c\sqrt{T_0/T_\mathrm{residual}}`` and ``C_c\sqrt{T_0/T_\mathrm{full}}`` are reported beside it.

## Running it

The stage needs a `[PerturbedEquilibrium]` run with `forcing_data_format = "coil"` and the singular coupling computed; `examples/DIIID-like_error_field_example` is the complete case, with the DIII-D C-coil as the n = 1 source and the eighteen F coils as single-filament hoops whose error-field content is entirely their sensitivity to misalignment.
From its deck:

```toml
[ErrorFields]
fd_step_shift_m = 1e-3          # central-difference step of the rigid shifts [m]
fd_step_tilt_deg = 0.1          # central-difference step of the rigid tilts [degrees]
rotation_center = "conductor"   # tilt pivot: each conductor's own centre, or "set" for the whole set's
exclude_coils = []              # coil sets that are not error-field sources (the NTV efc_coils are excluded automatically)
tolerance_file = "tolerances.toml"   # manufacturing tolerances, relative to the deck directory

[ErrorFields.MonteCarlo]
nsample = 1000000               # samples per batch
nbatch = 10                     # independent batches; their spread is the statistical error bar
seed = 1
nbins = 300                     # histogram bins on [0, delta_max]; delta_max = 0 is 1.5 × the worst-case bound
tolerance_scale = 1.0           # multiplies every tolerance (for scans); scale_subset / scale_map restrict it
coil_subset = []                # coil sets sampled; empty = all (the others get zero tolerance)

[ErrorFields.scenario]
n_e = 5.0                       # electron density of the threshold scaling [1e19 m^-3]; B_T, R_0, beta_N, l_i and I_p default from the equilibrium

[ErrorFields.Risk]
year = 2020                     # threshold fit: 2020 (n = 1 and 2) or 2026 (n = 1)
dataset = "O,L"                 # ITPA dataset of the fit ("O,L", "O,L,H", ...)
fit = "WLS"                     # "OLS", "DSOLS" or "WLS"
distribution = "normal"         # how the fit exponents are sampled: "normal", "flat", "normal_truncated"
nsample_threshold = 1000000
scan_scales = [0.25, 0.5, 1.0, 2.0, 4.0]   # tolerance multipliers of the allowable-tolerance scan

[ErrorFields.NTV]
efc_coils = ["d3d_c"]           # the correction arrays; never error-field sources
method = "fgar"                 # KineticForces torque method, enabled in [KineticForces]
```

The tilt pivot matters for multi-filament winding packs: `"conductor"` rotates each filament about its own centre, `"set"` rotates the pack rigidly about its common centre, which is what an engineering axis-line tolerance constrains.
A coil set that is not an error-field source (a correction array or anything in `exclude_coils`) is swept like the others but left out of the as-designed field, the Monte Carlo and the scans; a tolerance that names it is an error.
The NTV limits need a `[KineticForces]` section, since each array's torque per kAt² is one kinetic torque evaluation of its unit-current spectrum, with and without the dominant mode projected out.

The tolerances are engineering data, reviewed independently of any run, so they live in their own file, echoed into `Input/RawInputs/ErrorFields/tolerance_toml_raw` for replay; every key is checked against the schema.
Its tables are `[ErrorFields.defaults]` (fallbacks for every coil block), one `[[ErrorFields.coil]]` per coil set that moves on its own, `[[ErrorFields.coherent_group]]` for coil sets that move together, `[ErrorFields.correctability]` and `[ErrorFields.other_field]`:

```toml
[ErrorFields.defaults]
shift_sigma_mm = 0.0             # Gaussian placement uncertainty added to the sampled shift [mm]
tilt_sigma = 0.0                 # the same for the tilt, in tilt_units
tilt_units = "deg"               # "deg", or "m" for a rim displacement converted through the coil's major radius
radial_shape = "hollow"          # radial sampling density on the disk: flat, uniform_area, hollow or ring
tolerance_model = "additive"     # "additive" (independent shift and tilt) or "cylinder" (correlated axis line)
current_factor = 1.0             # current as built relative to the run's; 0 switches a coil off

[[ErrorFields.coil]]
name = "PF1U"                    # must match a [[ForcingTerms.coil_set]] of the run
shift_tol_mm = 0.5               # radius of the disk the coil centre may lie in [mm]
tilt_tol = 0.019                 # tilt tolerance, in tilt_units
tolerance_model = "cylinder"
cylinder_half_height_m = 1.5     # sets the tilt reachable at the given radius

[[ErrorFields.coherent_group]]
name = "upper_pf_brace"
members = ["PF1U", "PF2U"]       # one shared draw per sample
shift_tol_mm = 2.0
tilt_tol = 0.0
phase_group = "braces"           # groups naming the same phase_group share the random direction
rotation_center_z_m = 0.0        # pivot of the group's rigid rotation on the machine axis [m]

[ErrorFields.correctability]
uncorrectable_coils = ["EFCC_U"] # coil sets the error-field correction cannot reduce
efc_factor = 2.0                 # divisor applied to correctable contributions under correction

[ErrorFields.other_field]
magnitude = 8.2e-6               # overlap budget of sources not attributed to any coil
sigma = 0.0
radial_shape = "ring"
```

A coherent group's tilt is a rigid rotation of its members about the pivot, so a member at height ``z`` also shifts by ``(z - z_\mathrm{pivot})\,\theta``; coil sets without a tolerance block contribute their as-designed overlap only; a coil's `current_factor` scales its overlap and sensitivities together, the knob for a risk against the current in an uncorrectable coil.

From a script the stage is one call on the plasma response and the coil sets, and every product is re-evaluated post hoc from the file for any surface window, mode, fit, tolerance scale or coil subset:

```julia
ef = error_fields(resp, pe.coil_sets, ForcingTerms.CoilConfig(ft_ctrl);   # ft_ctrl: the [ForcingTerms] control
                  tolerances=EF.read_tolerance_toml("tolerances.toml"),
                  scenario=EF.ScenarioParameters(ffs.equil; n_e=5.0))
save_to_hdf5("gpec.h5", ef)

table = EF.sensitivity_table("gpec.h5"; psi_low=0.5)                        # overlaps and sensitivities per coil set, edge window
mc    = EF.run_monte_carlo("gpec.h5"; psi_low=0.5, tolerance_scale=2.0)      # |δ| distribution, intrinsic and corrected
risk  = EF.locking_risk("gpec.h5"; n_e=5.0, risk_ctrl=EF.RiskControl(; year=2026, fit="OLS"))
EF.allowable_tolerance(EF.ToleranceScan("gpec.h5"), 1.0; corrected=true)    # tolerance multiplier at 1% locking risk
EF.max_correctable_overlap(EF.read_efc_couplings("gpec.h5")[1]; delta_threshold=1.4e-4, torque_budget=4.0)
AEF.plot_error_field_summary("gpec.h5"; save_path="error_field_summary.png")
```

`correction_requirement` gives the current each array needs, by the dominant mode or jointly over every rational surface in the least-squares sense, and `uncorrectable_probability` the fraction of the Monte Carlo overlap beyond what the NTV-limited array can correct; `phasing_map` evaluates the dominant-mode field per kilo-ampere-turn over the relative phases of several arrays; `ResonantDriveContext` and `coil_overlaps` assess a new coil design against the stored run at one Biot–Savart pass per coil set, with `combine_overlaps` applying a design current pattern afterwards.
`Analysis.ErrorFields` plots each product, and every plotting function takes `label => path` pairs so coil-design revisions overplot on one axis.

## Outputs

| Dataset | Meaning |
|---|---|
| `ErrorFields/CoilSensitivities/field_as_designed`, `shift_sensitivity_per_m`, `tilt_sensitivity_per_deg` | the control-surface spectrum of each coil set as built and its derivatives with respect to the rigid shifts and tilts, with the current pattern they were evaluated at and the finite-difference linearity residual per tap |
| `ErrorFields/CoilSensitivities/DominantMode/delta_as_designed`, … | the overlaps and sensitivities projected onto the run's full-window dominant mode, their direction-averaged in-plane magnitudes, and the shift and tilt that would cancel the as-designed overlap |
| `ErrorFields/MonteCarlo/abs_delta_pdf`, `abs_delta_efc_pdf`, `abs_delta_bin_edges` | the intrinsic and corrected ``\lvert\delta\rvert`` histograms over the sampled tolerances, per batch and averaged |
| `ErrorFields/MonteCarlo/abs_delta_total_as_designed`, `abs_delta_worst_case` | the overlap of the design as built and the worst-case bound with every term at its tolerance edge and in phase |
| `ErrorFields/Risk/locking_probability_given_delta`, threshold density | ``P(\mathrm{lock}\,|\,\delta)`` and the threshold distribution on the Monte Carlo grid |
| `ErrorFields/Risk/locking_probability_percent`, `locking_probability_efc_percent`, `*_batches_percent`, `locking_probability_as_designed_percent`, `locking_probability_fit_threshold_percent` | the locking probability of the intrinsic and corrected distributions with per-batch values, of the design as built, and at the sharp fitted threshold |
| `ErrorFields/Risk/ToleranceScan/` | the same against `tolerance_scale`; the allowable tolerance at a target risk is its inversion |
| `ErrorFields/NTV/` | each correction array's dominant-mode coupling ``C_c``, resonant fraction and NTV torque per kAt² for its full field and with the dominant mode projected out |

Every dataset name follows one grammar, `<quantity>[_<instance>][_efc][_<statistic>][_<unit>]`: `delta` is the overlap and `abs_delta` its magnitude; `_as_designed`, `_sampled`, `_worst_case` name the geometry; `_efc` marks error-field correction applied; a probability stored as a percentage carries `_percent`.

## Numerical notes

- The sensitivities assume the overlap is linear in the misalignments out to the tolerance edge; the run checks it only at the finite-difference step, and `linearity_check` displaces each coil set by `±scale × tolerance` and recomputes its overlap to report the relative error of the linear model at the largest scale a tolerance scan will use (one Biot–Savart pass per row).
- Monte Carlo batches are seeded individually, so results are bit-identical for any thread count; `risk_convergence` repeats the Monte Carlo over sample and bin counts, and the batch spread must fall as ``1/\sqrt{n_\mathrm{sample}}`` while the value must not move with the bin count by more than the spread.
- A threshold scaling is fitted for one toroidal mode number, so the risk refuses a run that spans several; project onto one n's coupling first.
- A coil set whose field at the run's toroidal mode is round-off (an axisymmetric hoop at n = 1) has no resonant fraction and reports `NaN` rather than the ratio of two noise vectors.
- The NTV torques are stored with their sign and the limits consume the budget with their magnitude; the torque budget is the plasma's intrinsic torque and an analysis choice.

## API reference

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.ErrorFields]
```

```@docs
GeneralizedPerturbedEquilibrium.equilibrium_from_h5
```
