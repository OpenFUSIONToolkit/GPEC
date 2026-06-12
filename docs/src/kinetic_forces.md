# KineticForces Module

Kinetic torque and energy calculations for perturbed equilibria.
Implements neoclassical toroidal viscosity (NTV) from the PENTRC formulation
([Logan & Park, 2013](citations.md); [Logan, 2015](citations.md)).

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
