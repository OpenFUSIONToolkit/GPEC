# Workflow

GPEC reads one deck, `gpec.toml`, runs every stage whose section the deck contains, and writes every result into one output file, `gpec.h5`.
This page is the map: what each stage computes physically, which deck sections run it, and which datasets to read first.
The module pages give the governing equations and the full key and output tables; the [Scripting API](api.md) page shows the same pipeline called from Julia.

```
 equilibrium file ──► Equilibrium ──► ForceFreeStates ──► PerturbedEquilibrium ──► KineticForces
 kinetic profiles                       │   (Vacuum)        ▲         │              ErrorFields
                                        ├──► LocalStability │         │
                                        ├──► Match          │         └────────────► gpec.h5
                                        └──► Tearing   ForcingTerms (coils or file)
```

`Vacuum` runs inside `ForceFreeStates` whenever the solve is free-boundary, and `ForcingTerms` is read when `PerturbedEquilibrium` needs a drive, so neither is a stage of its own.
Every stage after `Equilibrium` reads the force-free states, and `KineticForces` and `ErrorFields` read the plasma response, so a section runs only when the sections it depends on are present.

## Which stage do I need

| Question | Deck sections | Read first |
|---|---|---|
| Is the equilibrium read and mapped correctly; what are q and β_N? | `[Equilibrium]` (+ `[ForceFreeStates]`, which triggers the write) | `Equilibrium/q_95`, `Equilibrium/beta_N`, `Equilibrium/Profiles/` |
| Is the plasma ideal-kink stable, with or without a wall? | `[ForceFreeStates]` (+ `[Wall]`) | `ForceFreeStates/FreeBoundaryStability/W_freeboundary`, `…/eigenmode_energies` |
| Is it Mercier, resistive-interchange or ballooning unstable? | `[LocalStability]` (+ `[Ballooning]`) | `LocalStability/D_I`, `LocalStability/D_R`, `LocalStability/ballooning_Delta_prime` |
| What is the tearing stability index Δ′ at each rational surface? | `[ForceFreeStates] integrator = "Riccati"` or `"Galerkin"` (+ `[Riccati]` or `[Galerkin]`) | `SingularSurfaces/Delta_prime_matrix` |
| How does the plasma respond to an RMP or error field; do islands overlap? | `[ForcingTerms]` + `[PerturbedEquilibrium]` | `PerturbedEquilibrium/SingularCoupling/resonant_area_weighted_field`, `…/island_half_width`, `…/chirikov_parameter` |
| What NTV torque does that field drive? | `[KineticForces]` (+ `[Equilibrium] kinetic_file`) | `KineticForces/<method>/total_torque` |
| How tight must the coil tolerances be to avoid locking? | `[ErrorFields]` | `ErrorFields/Risk/locking_probability_percent` |
| What are the tearing growth rate and critical resonant field? | `[Tearing]` (+ `[SLAYER]` or `[GGJ]`) | `Tearing/Roots/gamma`, `Tearing/Roots/omega`, `Tearing/CriticalResonantField/` |
| What does a resistive inner layer do to the response? | `[Match]` (+ `[GGJ]`) | `SingularSurfaces/Match/` |

`[PerturbedEquilibrium]` and the stages built on it need a `[ForceFreeStates]` solve with `integrator = "Forward"` and no `[Wall]`, while the Δ′ matrix, `[Tearing]` and wall stability need `"Riccati"` or `"Galerkin"` or a `[Wall]`, so one deck cannot do both: run the response and the Δ′ or wall cases as separate decks (`DIIID-like_ideal_example` and `DIIID-like_riccati_deltaprime_example` are such a pair).

### Headline outputs

- **Free-boundary energy** `ForceFreeStates/FreeBoundaryStability/W_freeboundary` is the power-normalized free-boundary energy matrix, the sum of `W_plasma` and `W_vacuum` for the wall of the deck (no wall when `[Wall]` is absent). `eigenmode_energies` lists its generalized eigenvalues, the ideal δW per unit displacement of each free-boundary eigenmode, sorted most unstable first, so a negative first entry means the plasma is ideal-kink unstable and a positive one is the stability margin.
- **Tearing index** `SingularSurfaces/Delta_prime_matrix` is the Δ′ matching matrix in the PEST-3 convention, whose diagonal is the outer-region Δ′ of each rational surface; a positive Δ′ means the outer region drives the tearing mode and the inner layer (`Tearing`) sets its growth rate.
- **Resonant field** `PerturbedEquilibrium/SingularCoupling/resonant_area_weighted_field` is the area-weighted resonant normal field b̄ʳ `[T]` at each rational surface after the plasma response, the quantity that drives islands and sets locking thresholds. `island_half_width` is the vacuum-superposition island each resonant flux would open, w_s ∝ √|δψ_s|, and `chirikov_parameter` is that half-width over the half-distance to the neighbouring rational surface, so σ > 1 means the islands overlap.
- **Torque** `KineticForces/<method>/total_torque` is the neoclassical toroidal viscosity (NTV) torque T_φ `[N·m]` that the perturbed equilibrium exerts on the plasma, with the sign convention of the [Conventions Reference](conventions.md) page.

## What each stage computes

### Equilibrium

Reads or constructs the axisymmetric equilibrium, maps it to straight-field-line coordinates (`[Equilibrium] jac_type`), builds the flux-surface splines every later stage evaluates, and locates the rational surfaces q = m/n.
`[Equilibrium] eq_type` selects a file format: `efit`, `efit_arclength` and `efit_by_inversion` (EFIT g-files, three flux-surface tracing strategies), `chease` and `chease2` (CHEASE binary and ASCII), `ifile` (inverse i-file, as written by TokaMaker) and `imas` (an `IMASdd.dd` passed in from a script).
An analytic equilibrium is selected by its own section instead: `[Solovev]`, `[LargeAspectRatio]` or `[TJAnalytic]`.
The optional `[Equilibrium] kinetic_file` adds the density, temperature and rotation profiles that the kinetic solve, `KineticForces` and the `SLAYER` layer model need; `[[Equilibrium.ion_species]]` declares the ion composition.
Outputs: `Equilibrium/` scalars (`beta_N`, `q_axis`, `q_95`, …), `Equilibrium/Profiles/` and `Equilibrium/Geometry/`.
Page: [Equilibrium](equilibrium.md).

### ForceFreeStates

Determines ideal MHD stability by the Newcomb criterion of the Fortran DCON: the Euler–Lagrange equations of δW are integrated in ψ for all poloidal harmonics at once, crossing each rational surface with the prescribed small-solution jump, to give the force-free states ξ(ψ, θ) and the plasma energy matrix.
With `vac_flag = true` the `Vacuum` module adds the vacuum energy for the wall of the `[Wall]` section (shapes `conformal`, `elliptical`, `dee`, `mod_dee` and `file`; no wall when the section is absent), and the free-boundary eigenvalue problem gives the kink stability.
`[ForceFreeStates] integrator` selects the radial method and its section: `"Riccati"` (the default; `[Riccati]`) also produces the Δ′ matrix, `"Forward"` (`[Forward]`) is the plain ODE integration, and `"Galerkin"` (`[Galerkin]`) is the resistive outer-region solve on a finite-element basis.
`kinetic = "KineticForces"` replaces the ideal energy matrices by their drift-kinetic form, which needs a kinetic file.
Outputs: `ForceFreeStates/FreeBoundaryStability/`, `ForceFreeStates/Solutions/`, `ForceFreeStates/EdgeScan/`, `SingularSurfaces/` and `SurfaceGeometries/`.
Key references: [Glasser (2016)](citations.md#ForceFreeStates), [Glasser (2018)](citations.md#ForceFreeStates), [Chance (1997)](citations.md#Vacuum).
Pages: [Stability](stability.md), [Galerkin Solver](galerkin.md), [Vacuum](vacuum.md).

### LocalStability

Evaluates the Mercier criterion D_I, the resistive-interchange criterion D_R and the high-n ballooning Δ′ on every flux surface of the solve, and with `[LocalStability] alpha_boundary_scan` the critical normalized pressure gradient α_crit.
Runs only with `[ForceFreeStates]`, on the same equilibrium grid; `[Ballooning]` sets the ballooning integration.
Outputs: `LocalStability/`.
Page: [Local Stability](ballooning.md).

### Match

Matches the outer-region Δ′ to a resistive inner layer (`inner_model = "GGJ"`, the only model, with its options in an optional `[GGJ]` section) and rebuilds the force-free states with the resulting layer response, so that the later stages see a resistive rather than perfectly shielded plasma; `ideal = true` keeps the shielded limit for comparison.
Outputs: `SingularSurfaces/Match/`.
Page: [Tearing](inner_layer.md).

### PerturbedEquilibrium

Computes the plasma response to an external non-axisymmetric field: `plasma_response` builds the permeability, inductance and reluctance operators on the control surface from the force-free states and the vacuum matrices, and `perturbed_equilibrium` applies them to the drive that `ForcingTerms` supplies, either a normal-field spectrum from a file or the Biot–Savart field of `[[ForcingTerms.coil_set]]` coils.
The resonant fields, island half-widths and Chirikov parameter at each rational surface follow from the singular coupling (`compute_singular_coupling`), and `regularization_width` controls the regularization of the ideal response there.
The response is built only from a `\"Forward\"` solve without a wall: a `\"Riccati\"` result (the default) makes `plasma_response` skip the operators with a warning, and a `\"Galerkin\"` or matched result or a wall is an error.
Outputs: `PerturbedEquilibrium/ForcingModes/`, `Response/`, `ResponseMatrices/`, `SingularCoupling/`, `Energies/`.
Key references: [Park et al. (2007a)](citations.md#PerturbedEquilibrium), [Park et al. (2009)](citations.md#PerturbedEquilibrium).
Pages: [Forcing Terms](forcing_terms.md), [Perturbed Equilibrium](perturbed_equilibrium.md).

### KineticForces

Integrates the NTV torque and kinetic energy of the perturbed equilibrium over ψ for each torque `method`, by species when the ion composition has more than one.
Runs after `PerturbedEquilibrium` and needs n_e, T_i, T_e and ω_E in the kinetic file; without `[PerturbedEquilibrium]` the section only configures the kinetic solve of `ForceFreeStates`.
Outputs: `KineticForces/<method>/` and `KineticForces/PerSpecies/`.
Key references: [Park and Logan (2017)](citations.md#KineticForces), [Logan et al. (2013)](citations.md#KineticForces).
Page: [Kinetic Forces](kinetic_forces.md).

### ErrorFields

Assesses error-field tolerances for a coil-driven response: the sensitivity of the resonant field to the shift and tilt of each coil set, a Monte Carlo sample of the tolerances of `[ErrorFields.MonteCarlo]`, the locking probability of `[ErrorFields.Risk]` against a threshold scenario, and the NTV coupling of the correction coils of `[ErrorFields.NTV]`.
Needs `[PerturbedEquilibrium]` driven by `[[ForcingTerms.coil_set]]` coils.
Outputs: `ErrorFields/CoilSensitivities/`, `MonteCarlo/`, `Risk/`, `NTV/`.
Page: [Error Fields](error_fields.md).

### Tearing

Solves the tearing dispersion relation of each rational surface by matching the outer-region Δ′ to an inner-layer model: the two-fluid slab `SLAYER` (`[SLAYER]`) or the toroidal resistive-MHD `GGJ` (`[GGJ]`), selected by `[Tearing] inner_model`.
`[Tearing.search]` sets the root search in the complex frequency plane, and `[Tearing.CriticalResonantField]` adds the resonant field at which the layer's shielding is overcome.
Needs the Δ′ matrix of `"Riccati"` with `vac_flag = true` or `"Galerkin"` with `[Galerkin] rpec_flag = true`, and the kinetic profiles of `[Equilibrium] kinetic_file` (`SLAYER` reads its densities, temperatures, rotation and diffusivities there).
Outputs: `Tearing/PerSurface/`, `Roots/`, `LayerWidths/`, `Diagnostics/`, `CriticalResonantField/`.
Key references: [Glasser, Wang and Park (2016)](citations.md#Tearing), [Park (2022)](citations.md#Tearing).
Page: [Tearing](inner_layer.md).

## The deck

A stage runs when its section is present, so a deck with only `[Equilibrium]` and `[ForceFreeStates]` stops after the stability solve, and one without `[ForceFreeStates]` writes nothing.
The abridged `examples/DIIID-like_ideal_example/gpec.toml` below runs the equilibrium, the ideal free-boundary solve with local stability, and the coil-driven perturbed equilibrium:

```toml
[Run]
verbose = true

[Equilibrium]
eq_type = "efit"
eq_filename = "TkMkr_D3Dlike_Hmode.geqdsk"
kinetic_file = "TkMkr_D3Dlike_Hmode_kinetic.h5"
jac_type = "hamada"
psihigh = 0.995

[LocalStability]
alpha_boundary_scan = true

[ForceFreeStates]
nn = 1
vac_flag = true
integrator = "Forward"   # "Forward": [PerturbedEquilibrium] needs the dense ξ solution and no [Wall] ("Riccati" is the default)

[ForcingTerms]
forcing_data_format = "coil"
machine = "d3d"

[[ForcingTerms.coil_set]]
name = "c"
currents = [1000.0, 500.0, -500.0, -1000.0, -500.0, 500.0]

[PerturbedEquilibrium]
compute_singular_coupling = true
regularization_width = 0.05
```

Every key is documented on the module page of its section, and every example deck carries an inline comment per key.

### Deck section index

| Section | Runs or configures | Page |
|---|---|---|
| `[Run]` | `output_file` (default `gpec.h5`, written next to the deck) and `verbose` | this page |
| `[Equilibrium]` | the equilibrium stage: file, format, coordinates, grid, kinetic file | [Equilibrium](equilibrium.md) |
| `[[Equilibrium.ion_species]]` | the ion composition of the kinetic profiles | [Equilibrium](equilibrium.md) |
| `[Solovev]`, `[LargeAspectRatio]`, `[TJAnalytic]` | an analytic equilibrium in place of a file | [Equilibrium](equilibrium.md) |
| `[Wall]` | the conducting wall of the vacuum response; omit for no wall | [Vacuum](vacuum.md) |
| `[ForceFreeStates]` | the stability stage: `nn`, the mode window, `integrator`, `vac_flag`, `kinetic` | [Stability](stability.md) |
| `[Forward]`, `[Riccati]` | the radial ODE integrators | [Stability](stability.md) |
| `[Galerkin]` | the finite-element outer-region solver | [Galerkin Solver](galerkin.md) |
| `[LocalStability]`, `[Ballooning]` | the local stability stage and its ballooning integration | [Local Stability](ballooning.md) |
| `[Match]`, `[GGJ]` | the inner-layer matching stage and the options of its GGJ layer model | [Tearing](inner_layer.md) |
| `[ForcingTerms]`, `[[ForcingTerms.coil_set]]` | the external field: a file or coil sets | [Forcing Terms](forcing_terms.md) |
| `[PerturbedEquilibrium]` | the plasma response stage and its singular-coupling options | [Perturbed Equilibrium](perturbed_equilibrium.md) |
| `[KineticForces]` | the NTV torque stage and the kinetic solve of `ForceFreeStates` | [Kinetic Forces](kinetic_forces.md) |
| `[Tearing]`, `[Tearing.search]`, `[Tearing.CriticalResonantField]`, `[SLAYER]` | the tearing stage, its root search, the critical field and the SLAYER layer model | [Tearing](inner_layer.md) |
| `[ErrorFields]`, `[ErrorFields.MonteCarlo]`, `[ErrorFields.scenario]`, `[ErrorFields.Risk]`, `[ErrorFields.NTV]` | the error-field tolerance stage | [Error Fields](error_fields.md) |

## Running it

From a shell, on a directory holding a `gpec.toml`:

```bash
./gpec examples/DIIID-like_ideal_example
```

From Julia, the same pipeline is `GeneralizedPerturbedEquilibrium.main(["examples/DIIID-like_ideal_example"])`, which returns the stage results as a named tuple; each stage is also a `solve` call on a problem type, so a script can run any subset. See the [Scripting API](api.md).
`examples/*/run_example.jl` and `analyze_example.jl` show the scripted run and the [Analysis](analysis.md) plots of its output.

### Examples

| Directory | Exercises |
|---|---|
| `DIIID-like_ideal_example` | DIII-D-like H-mode, n = 1: free-boundary ideal stability, local stability, C-coil-driven perturbed equilibrium and NTV torque; the default benchmark case |
| `DIIID-like_ideal_multi_n_example` | the same plasma with n = 1 and 2 solved together and driven by mixed-n coil currents |
| `DIIID-like_ideal_example_IMAS` | the same equilibrium read from an `IMASdd.dd`, run from `run_imas_example.jl` |
| `DIIID-like_riccati_deltaprime_example` | the Δ′ matrix on the `"Riccati"` integrator, no forcing |
| `DIIID-like_gal_resistive_example` | the `"Galerkin"` outer-region Δ′ solve, no forcing |
| `DIIID-like_gal_resistive_pe_example` | Galerkin Δ′ with `[Match]` inner-layer matching and coil forcing |
| `DIIID-like_SLAYER_example` | tearing growth rates and critical resonant fields with the `SLAYER` layer model |
| `DIIID-like_truncation_example` | the resistive-layer overlap cap on the edge of the integration domain |
| `DIIID-like_error_field_example` | error-field sensitivities, Monte Carlo tolerances and locking risk for the DIII-D poloidal-field coils |
| `Solovev_ideal_example` | analytic Solov'ev equilibrium from `[Solovev]`: a near-marginal n = 1 kink held by a conformal wall |
| `Solovev_ideal_example_3D` | the same plasma with the vacuum response on a toroidal grid |
| `Solovev_ideal_example_multi_n` | the same plasma swept over n = 1 and 2 in one run, with single-n cross-checks |
| `LAR_reversed_shear_example` | a `[LargeAspectRatio]` equilibrium with reversed shear, q = 2 crossed twice |
| `LAR_ideal_match_test` | a `[TJAnalytic]` equilibrium with `[Match] ideal = true`, the shielded reference for the resistive case |
| `LAR_resistive_match_test` | the same equilibrium with resistive GGJ matching |
| `LAR_beta_scan`, `LAR_epsilon_scan` | `run_scan.jl` sweeps of the `[TJAnalytic]` pressure factor and inverse aspect ratio |
| `TJ_ideal_example` | a `[TJAnalytic]` equilibrium: n = 1 stability and a file-driven perturbed equilibrium |
| `TJ_kinetic_NTV_example` | adds `[KineticForces]`: the NTV torque of the perturbed equilibrium |
| `TJ_kinetic_calculated_example` | the drift-kinetic stability solve (`kinetic = "KineticForces"`) with a D–T main ion |
| `TJ_kinetic_multiion_example` | NTV summed over D, T, the impurity and electrons |
| `a10_kinetic_example` | drift-kinetic stability of a low-aspect-ratio circular tokamak |
| `efit_fixedbdy_separatrix_example` | a fixed-boundary EFIT file whose box hugs the separatrix, for the separatrix search |

## The output file

All results are written to one HDF5 file (`[Run] output_file`, default `gpec.h5`, next to the deck), with top-level groups by physics topic; the schema conventions are in `docs/development/hdf5-conventions.md` of the repository.

| Group | Contents |
|---|---|
| `Info/` | Run metadata: git version, mode-number ranges, `psilim`, `qlim`, `Runtimes/` (per-stage wall-clock seconds) |
| `Input/` | Rerun snapshot: the recorded deck and the raw equilibrium, forcing and coil inputs |
| `Equilibrium/` | Scalars (`beta_N`, `q_axis`, `q_95`, …), 1-D profiles (`Profiles/`), 2-D geometry (`Geometry/`) |
| `ForceFreeStates/` | `Solutions/`, `FreeBoundaryStability/`, `EdgeScan/`, and `EulerLagrangeMatrices/` when `write_el_matrices = true` |
| `LocalStability/` | `D_I`, `D_R`, `ballooning_Delta_prime` on `psi`; the α boundary on `ballooning_psi` |
| `SingularSurfaces/` | Per-rational-surface data: `rational_psi`, `rational_q`, GGJ coefficients, `Delta_prime_matrix` and the PEST-3 matrices, `Match/`, `Kinetic/` |
| `PerturbedEquilibrium/` | `ForcingModes/`, `Response/`, `ResponseMatrices/`, `SingularCoupling/`, `Energies/` |
| `KineticForces/` | `<method>/` torque and energy profiles; `PerSpecies/<species>/<method>/` for multi-ion runs |
| `ErrorFields/` | `CoilSensitivities/`, `MonteCarlo/`, `Risk/`, `NTV/` |
| `Tearing/` | `PerSurface/`, `Roots/`, `LayerWidths/`, `Diagnostics/`, `CriticalResonantField/` |
| `SurfaceGeometries/` | Plasma and wall surface point clouds for visualization |

Rerunning `main(["gpec.h5"])` on an output file reproduces the run from its `Input/` snapshot; see the [Scripting API](api.md).
