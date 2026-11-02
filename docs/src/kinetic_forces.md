# Kinetic Forces

`KineticForces` computes the neoclassical toroidal viscosity (NTV) of a perturbed equilibrium: the toroidal torque and the kinetic energy that a non-axisymmetric field exerts on the plasma through the drift-kinetic response of its particles.
It implements the PENTRC formulation of [Park et al. (2009)](citations.md#KineticForces), [Logan et al. (2013)](citations.md#KineticForces) and [Park and Logan (2017)](citations.md#KineticForces).

## What it computes

A non-axisymmetric perturbation ``\delta B`` breaks the toroidal symmetry that conserves the canonical angular momentum of each particle.
The resulting non-ambipolar radial transport, dominated by trapped particles whose bounce and precession motion resonates with the E×B rotation, carries toroidal momentum out of the plasma: the NTV torque ``T_\phi``, which brakes or spins up the rotation, together with a kinetic contribution ``\delta W_k`` to the perturbed energy that stabilizes or destabilizes the response.
Both are halves of one complex quantity, ``T = T_\phi + 2in\,\delta W_k``, and the module integrates it over the plasma for one or more torque `method`s.

The torque enters the pipeline in two ways:

| Workflow | Deck | What it gives |
|---|---|---|
| Torque of a perturbed equilibrium | `[KineticForces]` after `[PerturbedEquilibrium]` | ``T(\psi)`` integrated over the stored response; the stage of the [Workflow](workflow.md) |
| Self-consistent kinetic solve | `[ForceFreeStates] kinetic = "KineticForces"` | the drift-kinetic energy matrices replace the ideal ones inside the stability solve, so the response itself carries the torque (see [Stability](stability.md) and `TorqueResponse/` on the [Perturbed Equilibrium](perturbed_equilibrium.md) page) |

Both need the kinetic profiles of `[Equilibrium] kinetic_file` (n_e, T_i, T_e, ω_E, and n_i when it differs from n_e) and the ion composition of `[[Equilibrium.ion_species]]`; the file schema is on the [Equilibrium](equilibrium.md) page.

## Governing equations

The torque is a sum over species of three nested integrals: over flux ``\psi``, over the pitch variable ``\Lambda`` separating trapped from passing particles, and over the normalized energy ``x = \varepsilon/T``, with a sum over the bounce harmonics ``\ell`` up to `bounce_harmonic_max`.
The integrand weights the bounce-averaged perturbation of the field strength, ``\langle \delta B_\parallel / B \rangle_\ell`` evaluated from the displacement ``\xi`` of the perturbed equilibrium, by the Maxwellian and by the resonance factor ([Logan et al. (2013)](citations.md#KineticForces)):

```math
T = \sum_s \sum_{\ell} \int d\psi \int d\Lambda \int_0^\infty dx\;
\frac{\big(\omega_E + \omega_{*n} + (x - \tfrac32)\,\omega_{*T}\big)\, x^{5/2} e^{-x}}{i\,\Omega(x) - \nu}\;
\mathcal{K}_{s,\ell}(\psi, \Lambda)\,\big|\langle \delta B_\parallel / B \rangle_\ell\big|^2 ,
\qquad
\Omega(x) = n\,\omega_E + \ell_\mathrm{eff}\,\omega_b\sqrt{x} + n\,\omega_D\,x ,
```

with ``\mathcal{K}`` the geometric kernel of the species and harmonic, ``\nu`` the energy-dependent effective collision frequency, and ``\omega_b``, ``\omega_D`` the bounce and precession frequencies at thermal energy (``\ell_\mathrm{eff} = \ell + nq`` for passing and ``\ell`` for trapped particles).
The rotation that enters is the E×B rotation ``\omega_E`` of the kinetic file, related to the measured toroidal rotation by

```math
\omega_\phi = \omega_E + \omega_{*n,i} + \omega_{*T,i}, \qquad
\omega_{*n,i} = -\frac{2\pi T_i}{\chi_1 Z_i e}\,\frac{1}{n_i}\frac{dn_i}{d\psi}, \qquad
\omega_{*T,i} = -\frac{2\pi}{\chi_1 Z_i e}\,\frac{dT_i}{d\psi},
```

with the diamagnetic frequencies from the spline derivatives of the profiles; the sign conventions are under [KineticForces rotation conventions](conventions.md#KineticForces-rotation-conventions).
The collision operator is selected by `nutype`: `"harmonic"` (the default), `"krook"`, `"small"` or `"zero"` (the collisionless limit), with the collision frequencies computed from the density and temperature profiles; `f0type` selects the equilibrium distribution, `"maxwellian"` (the default), `"park"` or `"cgl"`.

### Torque methods

Each `method` is one evaluation of the same integral under a different model, switched on by its `<method>_flag` in `[KineticForces]`; the first letter selects the particle population, `f` full, `t` trapped only, `p` passing only.

| Method | Model |
|---|---|
| `fgar` (default), `tgar`, `pgar` | general-aspect-ratio calculation of the torque and energy |
| `fcgl` | fluid Chew–Goldberger–Low limit |
| `fwmm`, `twmm`, `pwmm` | energy from the Euler–Lagrange kinetic matrix |
| `ftmm`, `ttmm`, `ptmm` | torque from the Euler–Lagrange kinetic matrix |
| `fkmm`, `tkmm`, `pkmm` | norm of the kinetic energy matrix |
| `frmm`, `trmm`, `prmm` | norm of the kinetic torque matrix |

## Running it

From `examples/DIIID-like_ideal_example/gpec.toml`, after the `[PerturbedEquilibrium]` section (`examples/TJ_kinetic_NTV_example` is the same stage on an analytic equilibrium):

```toml
[KineticForces]
bounce_harmonic_max = 6          # bounce harmonics summed over ℓ = -max:max
electron = false                 # add the electron NTV to the ion species
nutype = "harmonic"              # collision operator: "zero", "small", "krook", "harmonic"
f0type = "maxwellian"            # distribution function: "maxwellian", "park", "cgl"
atol_pitch = 1e-9                # absolute tolerance of the pitch integration
axis_validity_suppression = true # suppress the near-axis terms where the model is not valid
```

In a script the stage is one call on the perturbed equilibrium:

```julia
using GeneralizedPerturbedEquilibrium

kinetic = kinetic_torque(pe, KineticForcesControl(; bounce_harmonic_max=6, nutype="harmonic"))
```

The torque is three nested adaptive Gauss–Kronrod integrals, and each inner tolerance is derived from the outer one unless set: `rtol_psi` (1e-2, the primary knob; two significant figures match the model's validity), `rtol_pitch` (`psi_tolerance_margin × rtol_psi`) and `rtol_x` (`nested_tolerance_margin × rtol_pitch`).
Tightening `rtol_psi` tightens the whole ladder; tightening an inner level alone only multiplies evaluations.
The kinetic matrix build of a `kinetic = "KineticForces"` solve has no ψ quadrature and uses `rtol_pitch` (1e-3 when unset).

### Profile scaling

Seven factors on `KineticForcesControl` scale the kinetic inputs for sensitivity studies.
`density_factor`, `temperature_factor`, `ExB_rotation_factor` and `toroidal_rotation_factor` act on the profiles before any physics evaluation: the diamagnetic frequencies are formed from the unscaled profiles, ``\omega_E`` is rebuilt from the scaled total rotation ``\omega_\phi`` minus the scaled diamagnetic terms and then multiplied by `ExB_rotation_factor`, and the collision frequencies are recomputed from the scaled density and temperature.
`magnetic_drift_factor`, `collisionality_factor` and `divxi_factor` multiply the magnetic drift, the collisionality and ``\nabla\cdot\xi_\perp`` at evaluation time; use `collisionality_factor` to scale the collisionality alone.
The scaling factors are not supported together with a multi-ion `ion_species` list.

### Multi-ion runs

With several main-ion species the NTV is computed per species under one shared ``Z_\mathrm{eff}`` and summed over the main ions, the quasineutrality-closing impurity and, with `electron = true`, the electrons (`electron` always means in addition to the ions).
The sum is written to `KineticForces/<method>/` as in a single-ion run and each species to `KineticForces/PerSpecies/<label>/<method>/`, with labels `ion_z<z>_m<m>`, `impurity_z<z>_m<m>` and `electron`; the summed `total_torque` is the exact quadrature value, the summed profiles a linear interpolation onto the union grid.

### Regularization

`[PerturbedEquilibrium] regularization_width` multiplies ``\xi^{\psi\prime}`` and ``\xi^\alpha`` by ``Q^2/(Q^2 + w^2)`` with ``Q = m - nq`` before they drive the NTV integrand.
It exists because the ideal response diverges as ``1/Q`` at every rational surface, so the torque integral of an ideal solve does not converge; the self-consistent kinetic solve has no such singularity, since the finite torque moves the zeros of the kinetic coefficient off the real ``\psi`` axis ([Park and Logan (2017)](citations.md#KineticForces)).
The default is 0; set it, typically to 0.05, only for the torque of an ideal solve, and leave it at 0 for a kinetic run, where it would suppress a finite physical response.

## Outputs

| Dataset | Meaning | Units |
|---|---|---|
| `KineticForces/<method>/total_torque` | the NTV torque ``T_\phi`` on the plasma | N·m |
| `KineticForces/<method>/total_energy` | the kinetic energy ``\delta W_k = \operatorname{Im}(T)/(2n)`` | J |
| `KineticForces/<method>/n`, `psi_nsteps` | the toroidal mode number and the ψ quadrature count | |
| `KineticForces/<method>/dTdpsi`, `T` | the complex torque density and its cumulative integral in ψ, ``T = T_\phi + 2in\,\delta W_k`` | N·m |
| `KineticForces/<method>/dTdpsi_ell` | the torque density resolved by bounce harmonic on the `ell` scale | N·m |
| `KineticForces/<method>/KineticMatrices/` | the kinetic energy and torque matrices of the matrix methods | |
| `KineticForces/PerSpecies/<label>/<method>/` | the same layout per species in a multi-ion run | |

The `TorqueResponse/` matrices of a kinetic forward solve, which refer the torque back to the applied field, are described on the [Perturbed Equilibrium](perturbed_equilibrium.md) page.

## API reference

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.KineticForces]
```
