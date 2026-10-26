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
    `omega_tor`, `chi_e` (perpendicular heat diffusivity ``\chi_\perp``),
    `chi_phi` (toroidal momentum diffusivity ``\chi_\phi``), and `zeff`
    (effective charge ``Z_\mathrm{eff}``, dimensionless). Each dataset
    carries a `units` attribute; the root carries `schema_version` and
    `provenance`. Densities are m⁻³, temperatures eV, frequencies rad/s,
    diffusivities m²/s. Additional datasets named `n_*` (e.g. `n_D`, `n_T`)
    are named per-species densities for multi-ion runs; names outside `n_*`
    are reserved for future schema fields and ignored. Write files with
    `write_kinetic_h5` (which round-trips the per-species densities).
  - **ASCII (`.gpeckf`/`.kin`/`.dat`)** — legacy six-column whitespace table
    `psi_n n_i n_e T_i[eV] T_e[eV] omega_E`, retained for backward
    compatibility. Header rows are skipped.

The NTV calculation consumes `n_i, n_e, T_i, T_e, omega_E`; `chi_e`/`chi_phi`/`zeff`,
when present, are carried for the resistive-layer (SLAYER) analysis and ignored
here. That analysis takes ``Z_\mathrm{eff}(\psi)`` from the `zeff` dataset when the file
has one, otherwise from `n_i/n_e` with the `[SLAYER]` `impurity_charge`, otherwise
from the scalar `[SLAYER] zeff`. Supply `zeff` whenever the impurity content is not a
single species fixed by `n_i/n_e`.

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
impurity_charge = 6            # impurity charge (closes quasineutrality)
impurity_mass = 12             # impurity mass

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
main-ion charge must satisfy `z < impurity_charge`. An empty `ion_species` list runs the
single main ion from `ion_charge`/`ion_mass` (with `electron = true` still adding the
electron species — the electron flag always means *in addition to* the ions).

The summed total is written to `KineticForces/<method>/` exactly as in a single-ion
run, and each species' own contribution to `KineticForces/PerSpecies/<label>/<method>/`
— labels are `ion_z<z>_m<m>`, `impurity_z<z>_m<m>`, and `electron` (a numeric suffix
is appended only if a run repeats a `(z, m)` pair). Every per-species group carries the
same datasets and metadata as the total, so `dTdpsi` and `T` are available per species. The summed cumulative torque profile is a diagnostic (linear
interpolation onto the union grid); the summed `total_torque` scalar is the exact
Gauss-Kronrod value.

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
| `magnetic_drift_factor` | 1.0 | Evaluation time | Magnetic drift frequency scaling |
| `collisionality_factor` | 1.0 | Evaluation time | Collisionality scaling |
| `divxi_factor` | 1.0 | Evaluation time | ``\nabla \cdot \xi_\perp`` scaling |

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

### Evaluation-time knobs (`magnetic_drift_factor`, `collisionality_factor`, `divxi_factor`)

These three knobs are applied during the bounce-averaged kinetic matrix and
torque calculations in `KineticForces/Torque.jl` and related modules. They
multiply the magnetic drift, collisionality, and ``\nabla \cdot \xi_\perp``
terms respectively, and do not modify the stored kinetic profile splines.

!!! warning "Differences from Fortran PENTRC"
    1. **Collisionality from scaled profiles:** Julia recomputes collisionality
       (``\nu_i``, ``\nu_e``) from the scaled density and temperature arrays.
       Fortran PENTRC (`inputs.f90:237-246`) computes collisionality from
       **unscaled** profiles. If you need independent collisionality scaling
       without changing the density/temperature profiles, use `collisionality_factor`.

    2. **Consistent derivative ordering:** Fortran's `inputs.f90:269-272`
       mixes pre-scaling spline derivatives with post-scaling array values
       when computing the `toroidal_rotation_factor` back-solve. Julia uses a
       clean reimplementation with consistent pre-scaling derivatives throughout.

## Quadrature tolerances

The torque is three nested adaptive Gauss–Kronrod integrals: over ψ (paneled at rational and kinetic-resonance surfaces), over pitch λ at each ψ, and over normalized energy x at each λ. Each inner integral is the integrand of the one outside it, so its tolerance is derived from the outer one by a fixed margin unless set explicitly:

| level | control | default |
|---|---|---|
| ψ | `rtol_psi` | 1e-2 (the primary knob; ~2 significant figures matches the NTV model's validity) |
| λ | `rtol_pitch` | `psi_tolerance_margin × rtol_psi` = 1e-3 |
| x | `rtol_x` | `nested_tolerance_margin × rtol_pitch` = 1e-5 |

QuadGK bounds the error of the whole integral, not of each step as the LSODE integrators of Fortran PENTRC did, so a level tighter than its margin only multiplies evaluations: the ψ quadrature cannot see λ-level noise below `rtol_psi/10`. The calculated kinetic matrix build (`kinetic_source = "calculated"`) has no ψ quadrature; it uses `rtol_pitch` when set and 1e-3 otherwise. Tightening `rtol_psi` tightens the whole ladder.

## Regularization: off by default, and never needed by kinetic runs

`[PerturbedEquilibrium] regularization_width` smooths the displacements before they drive the NTV
integrand, multiplying ``\xi^{\psi\prime}`` and ``\xi^\alpha`` by
``Q^2/(Q^2 + \mathrm{regularization\_width}^2)`` with ``Q = m - nq``. It exists because **ideal** MHD is
singular at the rationals: ``\bar F_\mathrm{ideal} = Q F Q`` has ``\det \bar F = 0`` there, so
those two components diverge as ``1/Q`` and the torque integral does not converge.

The **self-consistent kinetic** workflow has no such singularity. Park & Logan
([Phys. Plasmas 24, 032505 (2017)](https://doi.org/10.1063/1.4978562), §III D) decompose the
kinetic composite matrix as ``F_k = Q \bar F_k Q - P_l^\dagger Q - Q P_u + R_1`` where
``R_1 \neq 0`` at ``Q = 0``; with finite torque ``\det \bar F`` is complex, its zeros leave the
real ``\psi`` axis, and the singularity is removed from both the solution and the torque
integral. (Torque-free kinetic energy principles instead *shift and split* the zeros to
``\psi_r \mp r_{L,R}``, where the singularity is logarithmic and integrable — still not a case
for smoothing.) Logan, *Electromagnetic Torque in Tokamaks with Toroidal Asymmetries* (PhD thesis,
Princeton, 2015), Ch. 7 Eq. 7.46 states the same resonance shifting/splitting result in GPEC's own
notation.

GPEC therefore **defaults `regularization_width` to 0** for every result, as the Fortran `reg_flag = f`
template does. Set it, typically to 0.05, only to compute the NTV torque of an ideal solve; a kinetic run
should leave it at 0.
Leaving it on suppresses a finite physical response and does so inconsistently — ``\xi^\psi``
is never regularized, so damping the other two breaks their near-resonance cancellation in
``\delta B/B`` and leaves a spurious residue driving the NTV integrand.

The check that this is right is that the NTV torque and the Euler–Lagrange solution's own
dissipation ``-2n\,\mathrm{Im}\langle \xi, u_2\rangle/4\mu_0`` — two independent calculations
of the same quantity — agree to a fraction of a percent with the regularization off, and disagree
by tens of percent with it on.

## HDF5 outputs: complex torque convention

The method level of `KineticForces/<method>/` reports the two physical scalars a user
wants first: `total_torque` = ``T_\phi`` (N·m) and `total_energy` = ``\delta W_k`` (J).
Internally both are halves of one complex quantity ``T = T_\phi + 2in\,\delta W_k``
(Logan 2013), and the ψ-profiles `dTdpsi` and `T` store that complex ``T`` directly —
so `imag(T)` carries the ``2n`` factor while `total_energy` has it divided out.

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.KineticForces]
```
