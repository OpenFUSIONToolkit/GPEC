# HDF5 Output Schema Conventions

Conventions for the structure and naming of the `gpec.h5` output file. Every writer that adds a group or dataset must follow these rules; the naming is enforced by `test/runtests_h5_schema.jl`.

## Governing principle: physics-first organization

**The schema must be intuitive to a plasma physicist who is not a developer of this code.** The top level largely mirrors the TOML sections / major `src/` modules — which are themselves organized along physics lines — but **physics intuition wins whenever the two diverge**. Worked examples of that rule:

- All per-rational-surface stability results consolidate under `SingularSurfaces/`, regardless of which algorithm produced them: the GGJ coefficients sit beside the Δ′/PEST-3 matching results, and the Riccati BVP and the Galerkin outer-region solve write the *same* `Delta_prime_matrix`/`Delta_prime_raw`/`Delta_coil`/`pest3_*` paths rather than each owning a producer-named subgroup.
- `EulerLagrangeMatrices` over a bare `Matrices` — group names must say what the data *is*, not which array it came from. Same reasoning renamed `records/` → `EnergyIntegrals/` and `matrices_<method>/` → `KineticMatrices/`.

Physics-topic groups elevated to top level (rather than nested under their producer): `Info/`, `Input/`, `SingularSurfaces/`, `LocalStability/`, `SurfaceGeometries/`.

## Naming rules

These rules govern `gpec.h5` (and any future GPEC-produced HDF5 output); harness-internal synthetic fixtures (e.g. the `ggj/*` reference files written by `regression-harness/src/runner.jl`) are out of scope.

- **Groups are CamelCase at every level** (`ForceFreeStates/`, `PerSurface/`, `GalerkinIntegration/`).
- **Datasets (leaves) are snake_case** (`eigenmode_energies`, `delta_prime_matrix`). Established physics symbols keep their natural case (`E`, `F`, `Q_root`, `pest3_Delta`, `2piF`).
- **Data-driven tokens are stored verbatim**: coil-set names under `Input/RawInputs/Coils/`, KineticForces method tokens (`fgar`, …), NTV species labels under `KineticForces/PerSpecies/` (`ion_z1_m2`, `impurity_z6_m12`, `electron` — charge and mass identify the species, with a numeric discriminator appended only if a run repeats a `(z, m)` pair), scan indices (`Surface_<k>`, `psi_<i>`).
- **Word-valued names and boolean flags**: multi-word dataset names are snake_case English (`resonance_psi`, `trajectory_offsets`, `layer_widths`), never CamelCase — CamelCase is reserved for groups. A boolean flag is named for the state it asserts when true, with an `is_` prefix only where the bare word would read as a noun or collide with a data family: `is_rational` (bare `rational` would clash with the `rational_*` coordinate family) versus `enabled`, `truncated`, `no_root`, which already read as predicates.
- **Literature capitalization for physics symbols**: names match the standard literature — `D_I`, `D_R`, `Delta_prime`, `tau_R`, `tau_A` (not `di`, `dr`, `delta_prime`, `taur`); lowercase stays where the literature is lowercase (`alpha`, `q`, `beta*`, `delta_s`).
- **Scalar equilibrium parameters spell out the physics**: `R_axis`, `Z_axis`, `B_T_axis`, `a_mean`, `aspect_ratio`, `I_p`, `q_edge`, `beta_N`, `delta_upper`/`delta_lower`. The qualifier is a trailing subscript (`_axis`, `_edge`, `_wall`, `_min`, `_max`, `_upper`, `_lower`, `_extremum`), and a numbered literature definition keeps its number as the last subscript (`beta_p_1`, `l_i_2`). Fortran-era contractions (`bt0`, `amean`, `crnt`, `qa`, `betan`, `li1`) survive only as `Equilibrium.EquilibriumParameters` struct fields: `EQUIL_H5_NAMES` in `src/HDF5Schema.jl` maps each field to its dataset name, and `EQUIL_H5_SKIP` drops fields that duplicate another dataset or echo a control flag.
- **Derivatives are `d<x>dpsi`** (`dTdpsi`, `dVdpsi`, `dqdpsi`, `dxidpsi`) — never Fortran `<x>1` suffixes or `<x>_deriv`.
- **"rational" over "singular"** in dataset names (`rational_psi`, `rational_q`, `rational_m`, `rational_n`, `rational_index`, `rational_count`) — kinetic/resistive runs are not singular at the rationals. Specifier order is standardized specifier-first (`rational_psi`, never `psi_rational`).
- **Vector components** follow `[d]<variable>[_<representation>]_<coordinate>[dpsi]` — the variable always comes first and the coordinate is always the trailing subscript. A bare coordinate suffix is the **contravariant** component (`xi_psi` = ξ^ψ), `_cov_` marks the **covariant** one (`b_cov_theta` = b_θ), a leading `J` marks a **Jacobian-weighted** component (`Jxi_theta` = J·ξ^θ), and other representations sit in the same slot (`xi_clebsch_psi`, `dxi_clebsch_psidpsi`). Never drop the variable: `clebsch_psi` is wrong because it does not say *what* is being represented. There is no HDF5/netCDF standard for super- vs subscripts — flat `_` names are universal — so the typeset form always appears in the dataset's `long_name`.
- **Coordinates**: the radial abscissa is `psi` (normalized poloidal flux ψ_N) and the poloidal one is `theta` in every group; never `psi_n`, `xs`, or `ys`.
- **One name per physical quantity**: a quantity written in several groups carries the identical leaf name everywhere (`rational_psi` in `SingularSurfaces/`, `Solutions/GalerkinIntegration/`, and `SingularCoupling/`; `Delta_prime_matrix` in `SingularSurfaces/` and `Tearing/PerSurface/`; `dVdpsi` in `Profiles/`, `SingularSurfaces/`, and `KineticForces/<method>/`) — the group supplies the context, the leaf supplies the identity.

## Inputs live only under `Input/`

`Input/gpec_toml_raw` stores the full merged TOML, and `Input/RawInputs/` stores the raw equilibrium/forcing/coil data — together they make `gpec.h5` a self-contained rerun snapshot (`Rerun.jl` reconstructs every control struct from them; the writer and rerun reader carry cross-reference comments marking the mirrored path pair). **Never echo TOML flags or control-struct values into any other group** — every group outside `Input/` is derived output. (The former `kinetic/` and `slayer/settings/` echoes were removed under this rule.)

## Schema

Top level (10 groups):

| Group | Contents |
|---|---|
| `Info/` | Run metadata: `git_version`, mode-number ranges (`mpert`, `mlow`, …, `mn_index`), `psilim`, `qlim` |
| `Input/` | Rerun snapshot: `gpec_toml_raw`, `RawInputs/{Equilibrium, ForcingTerms, Coils/<name>}` |
| `Equilibrium/` | Scalars (`beta_N`, `q_axis`, `q_95`, `I_p`, …) plus `Profiles/` (1-D on `psi`: 2piF, mu0p, dVdpsi, q) and `Geometry/` (2-D on `psi`×`theta`: rcoords, offset, nu, jac) |
| `ForceFreeStates/` | `Solutions/ForwardIntegration/` (u-solutions), `Solutions/GalerkinIntegration/` (closed ξ profiles in the shared layout, `Match/` diagnostics, the gal surface list, debug-gated `Basis/`), `EulerLagrangeMatrices/{Ideal,Kinetic}`, `FreeBoundaryStability/`, `EdgeScan/` |
| `LocalStability/` | Mercier `D_I`, resistive interchange `D_R`, `ballooning_Delta_prime` on `psi`; the ballooning α boundary on `ballooning_psi` |
| `SingularSurfaces/` | Per-rational-surface data: `rational_psi`/`rational_q`/`rational_m`/`rational_n`, GGJ coefficients, `Delta_prime_matrix`/`Delta_prime_raw`/`Delta_coil`/`pest3_A`/`pest3_B`/`pest3_Gamma` (Riccati or Galerkin alike), `Kinetic/` |
| `PerturbedEquilibrium/` | `ForcingModes/`, `Response/`, `ResponseMatrices/`, `SingularCoupling/`, `Energies/`, control-surface spectra |
| `KineticForces/` | `<method>/` (torque/energy profiles, `EnergyIntegrals/`, `KineticMatrices/`); multi-ion runs add `PerSpecies/<species>/<method>/` with the same per-method layout, summing to the top-level total |
| `Tearing/` | `PerSurface/` (+ `DpMatrix/`), `Roots/`, `LayerWidths/`, `Diagnostics/{ValidRoots,Poles,FilteredRoots}`, `Scan/Surface_<k>/` |
| `SurfaceGeometries/` | `{Plasma,Wall}/{x,y,z}` point clouds |

Reserved (documented, not yet written): `ForceFreeStates/Solutions/RiccatiIntegration/` — the third integrator backend slot alongside `ForwardIntegration` and `GalerkinIntegration`.

## Metadata contract (self-describing datasets)

Every dataset outside `Input/` (raw snapshot) and `GalerkinIntegration/Match/` (debug-only) must answer "what is this, in what units, plotted against what" without opening the source — enforced by `test/runtests_h5_schema.jl`:

- **`long_name`** — plain-text physics description.
- **`units`** — SI string (`"T"`, `"Wb/rad"`, `"A"`, `"m"`, `"J"`, `"N*m"`, `"Hz"`, `"Ohm*m"`); `"1"` for dimensionless (CF convention). Normalized quantities state the normalization in `long_name` (e.g. the power-normalized stability energies are per unit ⟨|ξ|²⟩, not joules).
- **`dims`** — required on rank ≥ 2 datasets: a greppable string like `"(psi, m)"` listing axis names in **Julia (column-major) order, axis 1 first**. Note h5py/HDFView users see file dimensions in the reversed (row-major) order. Square-matrix axes use distinct `_row`/`_col` names (`(mode_row, mode_col)`): netCDF permits repeated dimension names, but xarray mangles them on load.
- **HDF5 Dimension Scales** (the netCDF-4 coordinate mechanism): shared coordinate datasets (the `psi` grids, rational-surface `rational_psi`, geometry `psi`/`theta`) are marked with `h5ds_set_scale` and attached per-axis with `h5ds_attach_scale`/`h5ds_set_label`, so h5py `.dims`, xarray, and HDFView resolve axes natively. The H5DS C API indexes file (row-major) dimensions: Julia axis `k` of an `N`-d dataset is C index `N - k`.

Root-level file attributes: `schema_version` (currently `"2.0"`; bump on breaking schema changes — readers dispatch on it), `Conventions = "GPEC-HDF5-2.0"`, `references`, `title` (run description), `date_created` (ISO 8601 UTC). The code version stays in `Info/git_version`.

Mechanism: writers stay table-driven — each writer keeps a `path => (; long_name, units, dims, scale, attach)` table next to it (`scale` marks a coordinate dataset as a dimension scale; `attach` binds axes to scales — see the `Utilities.HDF5Annotations.annotate!` docstring) (`src/HDF5Schema.jl` for the main writer; alongside `write_galerkin!`, the PerturbedEquilibrium writer, `KineticForces/Output.jl`, and `Tearing/Runner/HDF5Output.jl` for the rest) and applies it post-write via `Utilities.HDF5Annotations.annotate!`. Entries for conditionally-written datasets are simply skipped when absent. When adding a dataset, add its table entry in the same commit — the schema test fails otherwise.

## File-wide conventions

- Complex quantities are stored as the native HDF5.jl compound type (readable by h5py as a compound dtype) — **never split into `*_real`/`*_imag` dataset pairs**. Sole sanctioned exception: `Input/RawInputs/ForcingTerms/amplitude_{real,imag}`, which mirrors the external forcing ingest-file format and keeps pre-existing snapshots replayable.
- `NaN` is the not-computed sentinel in numeric datasets (e.g. auto-derived settings, rootless growth-rate entries).
- A **zero-extent array** is the not-computed sentinel for whole datasets that a given run never produces (e.g. `SingularSurfaces/ca_left`/`ca_right` on kinetic or galerkin-matched runs, the free-boundary energies when `vac_flag=false`, the on-demand derivative stores). Never write unpopulated (`undef`) memory.
- Ragged (variable-length) data uses the flat-plus-`offsets` companion pattern (`offsets[k+1] - offsets[k]` = length of row `k`) rather than HDF5 VLEN types, e.g. `KineticForces/<method>/EnergyIntegrals/` and `Tearing/Diagnostics/*`.

## Back-compatibility policy

Schema renames are clean breaks everywhere — no dual-path reads, no legacy-path translation layers. When renaming a path, update the writer, all readers, and the regression-harness case TOMLs in the same PR. Cross-commit harness comparisons across a rename boundary work only from already-cached quantities of the pre-rename refs (values are stored by quantity *name*, which renames preserve); fresh extraction of a pre-rename output reports the quantity as missing, and `--ref-range` scans crossing the boundary do the same. Developers should re-baseline old refs with `--force` before a rename lands if they need that history, and treat a schema rename as a `schema_version` bump readers can dispatch on.
