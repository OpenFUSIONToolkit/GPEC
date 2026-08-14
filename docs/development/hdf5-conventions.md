# HDF5 Output Schema Conventions

Conventions for the structure and naming of the `gpec.h5` output file. Every writer that adds a group or dataset must follow these rules; the naming is enforced by `test/runtests_h5_schema.jl`.

## Governing principle: physics-first organization

**The schema must be intuitive to a plasma physicist who is not a developer of this code.** The top level largely mirrors the TOML sections / major `src/` modules — which are themselves organized along physics lines — but **physics intuition wins whenever the two diverge**. Worked examples of that rule:

- All per-rational-surface stability results consolidate under `SingularSurfaces/`, regardless of which algorithm produced them: the ideal BVP `delta_prime_matrix`, the GGJ coefficients, and the Galerkin outer-region Δ′/PEST-3 results (`GalerkinDeltaPrime/`) live side by side. Provenance is recorded in the subgroup name, not by scattering results across producer-owned groups.
- `EulerLagrangeMatrices` over a bare `Matrices` — group names must say what the data *is*, not which array it came from. Same reasoning renamed `records/` → `EnergyIntegrals/` and `matrices_<method>/` → `KineticMatrices/`.

Physics-topic groups elevated to top level (rather than nested under their producer): `Info/`, `Input/`, `SingularSurfaces/`, `LocalStability/`, `SurfaceGeometries/`.

## Naming rules

These rules govern `gpec.h5` (and any future GPEC-produced HDF5 output); harness-internal synthetic fixtures (e.g. the `ggj/*` reference files written by `regression-harness/src/runner.jl`) are out of scope.

- **Groups are CamelCase at every level** (`ForceFreeStates/`, `PerSurface/`, `GalerkinDeltaPrime/`).
- **Datasets (leaves) are snake_case** (`eigenmode_energies`, `delta_prime_matrix`). Established physics symbols keep their natural case (`E`, `F`, `Q_root_real`, `pest3_Delta`, `2piF`).
- **Data-driven tokens are stored verbatim**: coil-set names under `Input/RawInputs/Coils/`, KineticForces method tokens (`fgar`, …), scan indices (`Surface_<k>`, `psi_<i>`).

## Inputs live only under `Input/`

`Input/gpec_toml_raw` stores the full merged TOML, and `Input/RawInputs/` stores the raw equilibrium/forcing/coil data — together they make `gpec.h5` a self-contained rerun snapshot (`Rerun.jl` reconstructs every control struct from them; the writer/reader path pair is locked by shared `H5_*` consts in `GeneralizedPerturbedEquilibrium.jl`). **Never echo TOML flags or control-struct values into any other group** — every group outside `Input/` is derived output. (The former `kinetic/` and `slayer/settings/` echoes were removed under this rule.)

## Schema

Top level (10 groups):

| Group | Contents |
|---|---|
| `Info/` | Run metadata: `git_version`, mode-number ranges (`mpert`, `mlow`, …, `mn_index`), `psilim`, `qlim` |
| `Input/` | Rerun snapshot: `gpec_toml_raw`, `RawInputs/{Equilibrium, ForcingTerms, Coils/<name>}` |
| `Equilibrium/` | Scalars (β, q₀, q95, …) plus `Profiles/` (1-D: xs, 2piF, mu0p, dVdpsi, q) and `Geometry/` (2-D: rcoords, offset, nu, jac) |
| `ForceFreeStates/` | `Solutions/ForwardIntegration/` (u-solutions), `Solutions/GalerkinIntegration/` (`Solution/`, `Match/`, `msing`), `EulerLagrangeMatrices/{Ideal,Kinetic}`, `FreeBoundaryStability/`, `EdgeScan/` |
| `LocalStability/` | Mercier `di`, resistive interchange `dr`, `ballooning_Delta_prime`, ballooning α boundary |
| `SingularSurfaces/` | Per-rational-surface data: ψ, q, m/n, GGJ coefficients, `delta_prime_matrix`/`delta_prime_raw`/`delta_coil`, `GalerkinDeltaPrime/`, `Kinetic/` |
| `PerturbedEquilibrium/` | `ForcingModes/`, `Response/`, `ResponseMatrices/`, `SingularCoupling/`, `Energies/`, control-surface spectra |
| `KineticForces/` | `<method>/` (torque/energy profiles, `EnergyIntegrals/`, `KineticMatrices/`) |
| `Tearing/` | `PerSurface/` (+ `DpMatrix/`), `Roots/`, `LayerWidths/`, `Diagnostics/{ValidRoots,Poles,FilteredRoots}`, `Scan/Surface_<k>/` |
| `SurfaceGeometries/` | `{Plasma,Wall}/{x,y,z}` point clouds |

Reserved (documented, not yet written): `ForceFreeStates/Solutions/RiccatiIntegration/` — the third integrator backend slot alongside `ForwardIntegration` and `GalerkinIntegration`.

## File-wide conventions

- Complex numbers are stored as the native HDF5.jl compound type (readable by h5py as a compound dtype).
- `NaN` is the not-computed sentinel in numeric datasets (e.g. auto-derived settings, rootless growth-rate entries).
- Ragged (variable-length) data uses the flat-plus-`offsets` companion pattern (`offsets[k+1] - offsets[k]` = length of row `k`) rather than HDF5 VLEN types, e.g. `KineticForces/<method>/EnergyIntegrals/` and `Tearing/Diagnostics/*`.

## Back-compatibility policy

Schema renames are clean breaks everywhere — no dual-path reads, no legacy-path translation layers. When renaming a path, update the writer, all readers, and the regression-harness case TOMLs in the same PR. Cross-commit harness comparisons across a rename boundary work only from already-cached quantities of the pre-rename refs (values are stored by quantity *name*, which renames preserve); fresh extraction of a pre-rename output reports the quantity as missing, and `--ref-range` scans crossing the boundary do the same. Developers should re-baseline old refs with `--force` before a rename lands if they need that history, and treat a schema rename as a `schema_version` bump readers can dispatch on.
