# KineticForces Module

Kinetic torque and energy calculations for perturbed equilibria.
Implements neoclassical toroidal viscosity (NTV) from the PENTRC formulation
([Logan & Park, 2013](citations.md); [Logan, 2015](citations.md)).

## Kinetic profile file formats

Kinetic profiles are read by `read_kinetic_file`, which dispatches on the file
extension:

  - **HDF5 (`.h5`/`.hdf5`)** — the GPEC kinetic schema (recommended). Datasets
    at the file root: required `psi` (normalized poloidal flux), `n_e`, `T_i`,
    `T_e`, `omega_E` (and `n_i`, defaulting to `n_e` if omitted); optional
    `omega_tor`, `chi_e` (perpendicular heat diffusivity ``\chi_\perp``), and
    `chi_phi` (toroidal momentum diffusivity ``\chi_\phi``). Each dataset
    carries a `units` attribute; the root carries `schema_version` and
    `provenance`. Densities are m⁻³, temperatures eV, frequencies rad/s,
    diffusivities m²/s. Additional datasets named `n_*` (e.g. `n_D`, `n_T`)
    are named per-species densities for multi-ion runs; names outside `n_*`
    are reserved for future schema fields and ignored. Write files with
    `write_kinetic_h5` (which round-trips the per-species densities).
  - **ASCII (`.gpeckf`/`.kin`/`.dat`)** — legacy six-column whitespace table
    `psi_n n_i n_e T_i[eV] T_e[eV] omega_E`, retained for backward
    compatibility. Header rows are skipped.

The NTV calculation consumes `n_i, n_e, T_i, T_e, omega_E`; `chi_e`/`chi_phi`,
when present, are carried for the resistive-layer (SLAYER) analysis and ignored
here.

## Multi-ion runs

A plasma may declare an arbitrary list of main-ion species; the NTV is computed
per species under one shared full-composition ``Z_\mathrm{eff}`` and summed
(``\tau = \sum_s \tau_s``) over the main ions, the quasineutrality-closing
impurity, and (with `electron = true`) the electrons. This applies to both NTV
paths: the post-PE ψ-quadrature diagnostic and the self-consistent
`kinetic_source = "calculated"` matrices.

```toml
[KineticForces]
kinetic_file = "kinetic.h5"    # n_i column/dataset = TOTAL main-ion density
electron = true                # add electron NTV in addition to the ion species
zimp = 6                       # impurity charge (closes quasineutrality)
mimp = 12                      # impurity mass

[[KineticForces.ion_species]]
z = 1
m = 2
fraction = 0.5                 # this species' share of the total n_i
[[KineticForces.ion_species]]
z = 1
m = 3
density = "n_T"                # or: explicit n_* dataset from the HDF5 kinetic file
```

Each species sets exactly one of `fraction` (share of the file's total `n_i`)
or `density` (a named `n_*` dataset). An all-fraction list must sum to 1; in a
mixed list the impurity content is set by the file's `n_i`/`n_e` deficit, not
by a fraction shortfall, and fractions may sum below (never above) 1. Every
main-ion charge must satisfy `z < zimp`. An empty `ion_species` list runs the
single main ion from `zi`/`mi` (with `electron = true` still adding the
electron species — the electron flag always means *in addition to* the ions).

Per-species results are written to HDF5 groups `kinetic_forces_<label>`
(e.g. `kinetic_forces_ion1_z1_m2`, `kinetic_forces_impurity_z6_m12`,
`kinetic_forces_electron`) alongside the summed total in `kinetic_forces`. The
summed cumulative torque profile is a diagnostic (linear interpolation onto the
union grid); the summed `total_torque` scalar is the exact Gauss-Kronrod value.

The profile-scaling knobs below are not supported together with a multi-ion
`ion_species` list (they error).

## Profile Scaling Knobs

Seven scaling factors are available on `KineticForcesControl` to modify kinetic
profiles and physics parameters for sensitivity studies:

| Knob | Default | Stage | Description |
|---|---|---|---|
| `density_factor` | 1.0 | Profile loader | Density scaling (ni, ne) |
| `temperature_factor` | 1.0 | Profile loader | Temperature scaling (Ti, Te) |
| `ExB_rotation_factor` | 1.0 | Profile loader | ExB rotation scaling (omegaE) |
| `toroidal_rotation_factor` | 1.0 | Profile loader | Total toroidal rotation scaling (wphi) |
| `wdfac` | 1.0 | Evaluation time | Magnetic drift frequency scaling |
| `nufac` | 1.0 | Evaluation time | Collisionality scaling |
| `divxfac` | 1.0 | Evaluation time | ``\nabla \cdot \xi_\perp`` scaling |

### Profile-loader knobs (`density_factor`, `temperature_factor`, `ExB_rotation_factor`, `toroidal_rotation_factor`)

These four knobs are applied during kinetic profile loading in
`load_kinetic_profiles`, before any physics evaluation. The physical model is:

```math
\omega_\phi = \omega_E + \omega_{*n,i} + \omega_{*T,i}
```

where ``\omega_\phi`` is the user's measured total toroidal rotation,
``\omega_E`` is the ExB rotation (the input profile), and the diamagnetic
frequencies are computed from cubic spline derivatives of the unscaled profiles:

```math
\omega_{*n,i} = -\frac{2\pi T_i}{\chi_1 Z_i e} \frac{1}{n_i}\frac{dn_i}{d\psi}, \qquad
\omega_{*T,i} = -\frac{2\pi}{\chi_1 Z_i e} \frac{dT_i}{d\psi}
```

The scaling sequence is:

1. Build first-pass cubic splines from original (unscaled) profiles
2. Compute ``\omega_{*n,i}``, ``\omega_{*T,i}``, and ``\omega_\phi`` at each grid point
3. Apply `density_factor` to density, `temperature_factor` to temperature, and update diamagnetic terms:
   ``\omega_{*n,\text{new}} = \texttt{temperature\_factor} \cdot \omega_{*n,i}`` (`density_factor` cancels in ``T \cdot (dn/d\psi)/n``),
   ``\omega_{*T,\text{new}} = \texttt{temperature\_factor} \cdot \omega_{*T,i}``
4. Reform ExB rotation: ``\omega_E = \texttt{toroidal\_rotation\_factor} \cdot \omega_\phi - \omega_{*n,\text{new}} - \omega_{*T,\text{new}}``
5. Apply ExB scaling: ``\omega_E \mathrel{*}= \texttt{ExB\_rotation\_factor}``
6. Recompute collisionality from scaled density and temperature
7. Build final splines from scaled arrays

### Evaluation-time knobs (`wdfac`, `nufac`, `divxfac`)

These three knobs are applied during the bounce-averaged kinetic matrix and
torque calculations in `KineticForces/Torque.jl` and related modules. They
multiply the magnetic drift, collisionality, and ``\nabla \cdot \xi_\perp``
terms respectively, and do not modify the stored kinetic profile splines.

!!! warning "Differences from Fortran PENTRC"
    1. **Collisionality from scaled profiles:** Julia recomputes collisionality
       (``\nu_i``, ``\nu_e``) from the scaled density and temperature arrays.
       Fortran PENTRC (`inputs.f90:237-246`) computes collisionality from
       **unscaled** profiles. If you need independent collisionality scaling
       without changing the density/temperature profiles, use `nufac`.

    2. **Consistent derivative ordering:** Fortran's `inputs.f90:269-272`
       mixes pre-scaling spline derivatives with post-scaling array values
       when computing the `toroidal_rotation_factor` back-solve. Julia uses a
       clean reimplementation with consistent pre-scaling derivatives throughout.

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.KineticForces]
```
