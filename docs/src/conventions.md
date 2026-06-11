# Conventions Reference

This page is a comprehensive reference for the sign, coordinate, and field-amplitude conventions
used throughout GPEC. Understanding them is essential for interpreting outputs, interfacing with
other codes, and correctly setting up perturbed-equilibrium and kinetic calculations.

These conventions are inherited from the Fortran GPEC suite and are GPEC-native: they predate
and do not fully coincide with the COCOS standard (see [COCOS Compatibility](@ref) below).
Each section names the Julia source that establishes the convention so the documentation stays
traceable to the implementation.

## Quick Reference

| Quantity | Convention | Forced? |
|----------|-----------|---------|
| ``\psi`` (poloidal flux) | Normalized 0 (axis) to 1 (edge), no ``2\pi`` | Yes, positive |
| ``\theta`` (poloidal angle) | Upward outboard | Fixed |
| ``\phi`` (toroidal angle) | CCW for LH, CW for RH | By helicity |
| ``F = R B_\phi`` | Always positive | Yes, `abs` |
| ``q`` (safety factor) | Computed (direct) / input (inverse), positive | Effectively positive |
| ``n`` (toroidal mode) | Always positive (``\ge 1``) | By convention |
| Resonant ``m`` | Always positive (since ``n>0``, ``q>0``) | By construction |
| helicity | +1 RH, ``-1`` LH | Computed |
| ``\omega_E`` | Positive = direction of ``\zeta`` | No |

## Coordinate System

GPEC uses right-handed magnetic coordinates ``(\psi, \theta, \zeta)`` with the Fourier kernel

```math
\exp\!\big(i(m\theta - n\zeta)\big),
```

matching the Fortran ``\exp(im\theta - in\phi)`` convention. The forward analysis transform that
produces the mode coefficients ``f_m`` from a real-space profile ``f(\theta)`` uses the conjugate
kernel ``\exp(-im\theta)``; the inverse transform reconstructs with ``\exp(+im\theta)``
(`src/Utilities/FourierTransforms.jl`).

### Poloidal Flux ``\psi``

- Normalized from 0 (magnetic axis) to 1 (plasma boundary).
- ``\psi_0 = |\psi_\mathrm{bry} - \psi_\mathrm{axis}|`` is **forced positive** on read
  (`read_eq_efit` in `src/Equilibrium/ReadEquilibrium.jl`). If the EQDSK has
  ``\psi_\mathrm{bry} < \psi_\mathrm{axis}``, the 2D flux array is sign-flipped so that ``\psi``
  increases from axis to edge.
- The internal ``\psi`` carries **no** factor of ``2\pi`` (poloidal flux per radian). IMAS inputs,
  which use the full poloidal flux, are divided by ``2\pi`` on read — see
  [COCOS Compatibility](@ref).
- Occasionally ``\rho = \sqrt{\psi}`` is used as a radius-like variable.

### Poloidal Angle ``\theta``

- "Upward outboard" convention, always: ``\theta = 0`` on the outboard midplane, increasing
  upward.
- ``\theta`` is normalized to increase by 1 (not ``2\pi``) over one poloidal circuit. This is fixed
  regardless of helicity or working-coordinate choice.

### Toroidal Coordinate ``\zeta`` and ``\phi``

- The ignorable toroidal coordinate is ``\zeta = \phi/(2\pi) + \nu(\psi,\theta)``, where ``\nu`` is
  a single-valued straight-field-line offset that depends on the working coordinate. PEST
  coordinates have ``\nu = 0``.
- The physical toroidal angle is reconstructed as
  ``\phi = -\,\mathrm{helicity}\,(2\pi\zeta + \nu)`` (`sample_boundary_grid` in
  `src/ForcingTerms/CoilFourier.jl`). Thus ``\phi`` is effectively **counter-clockwise** (viewed
  from above) for left-handed (LH) configurations and **clockwise** for right-handed (RH)
  configurations.

### Working Coordinate Options

Controlled by `jac_type` in the `[Equilibrium]` section of `gpec.toml`. The Jacobian is
``J \propto B_p^{p_{bp}}\, B^{p_b}\, R^{-p_r}``:

| Name | `power_bp` | `power_b` | `power_r` |
|------|-----------|-----------|-----------|
| Hamada (default) | 0 | 0 | 0 |
| PEST | 0 | 0 | 2 |
| Boozer | 0 | 2 | 0 |
| Equal-arc | 1 | 0 | 0 |

The powers are set automatically from `jac_type` (`EquilibriumConfig` in
`src/Equilibrium/EquilibriumTypes.jl`) and applied in the field-line Jacobian
(`direct_fieldline_der!` in `src/Equilibrium/DirectEquilibrium.jl`).

## Helicity and Handedness

Helicity is derived from the equilibrium current and toroidal-field directions:

```julia
helicity = bt_sign * Int(sign(crnt))   # src/ForcingTerms/CoilFourier.jl
```

- `bt_sign` is the toroidal-field direction and `crnt` is the signed plasma current.
- **helicity = +1**: right-handed (RH) --- ``B_t`` and ``I_p`` in the same direction.
- **helicity = -1**: left-handed (LH) --- ``B_t`` and ``I_p`` opposed.
- Helicity sets the toroidal-grid direction ``\phi_j = -\mathrm{helicity}\times 2\pi j/n_\zeta`` and
  the output sign flips (see [Spectrum Output Sign Conventions](@ref)). For DIII-D
  (``B_t < 0``, ``I_p > 0`` ⇒ helicity = ``-1``) ``\phi`` increases with ``j``.

## ``F = R B_\phi``

The poloidal-current function ``F = R B_\phi`` from the Grad-Shafranov equation is **forced
positive** via `abs` (`read_eq_efit` in `src/Equilibrium/ReadEquilibrium.jl`):

```julia
abs.(fpol_data)
```

The code always works with ``|F|``; the sign of ``B_t`` is carried separately (`fpol_sign`, and the
`bt_sign`/`crnt` helicity inputs).

## Safety Factor ``q``

The direct solver **computes** ``q`` by field-line integration for direct (EFIT) equilibria and 
overwrites the file profile (`direct_run` in `src/Equilibrium/DirectEquilibrium.jl`):

```math
q = \frac{F}{2\pi}\oint \frac{J}{B_p}\,dl_p .
```

- Because the Jacobian ``J > 0``, the closed-loop integral is positive, and ``F = |F|`` is forced
  positive, **``q`` is positive** in standard operation. The solver emits a warning if the on-axis
  extrapolation yields ``q_0 \le 0``, treating it as a spline artifact rather than a physical
  result.
- **Inverse** equilibria instead use the ``q`` profile supplied with the input
  (`src/Equilibrium/InverseEquilibrium.jl`).

## Mode Numbers ``m`` and ``n``

### Toroidal Mode Number ``n``

- Set as a **positive integer** via `nn_low`/`nn_high` in `gpec.toml`. ``n < 1`` is not supported
  and is clamped to 1 with a warning (`src/GeneralizedPerturbedEquilibrium.jl`). Results for
  multiple ``n`` can be superposed.

### Poloidal Mode Range

The poloidal spectrum spans `mlow` to `mhigh` (`src/GeneralizedPerturbedEquilibrium.jl`):

```math
m_\mathrm{low} = \min(n\,q_\mathrm{min},\,0) - 4 - \Delta m_\mathrm{low}, \qquad
m_\mathrm{high} = n\,q_\mathrm{max} + \Delta m_\mathrm{high}.
```

`delta_mlow` and `delta_mhigh` widen the range beyond the resonant modes.

### Why Positive ``m`` Is Always Resonant

Resonant surfaces are found by `sing_find!` (`src/ForceFreeStates/Sing.jl`), which locates flux
surfaces where ``m = n\,q(\psi)``. Since ``n > 0`` by convention and ``q > 0`` for a standard
tokamak, the resonant ``m`` is always positive:

```math
m_\mathrm{res} = n\,q > 0 .
```

Negative-``m`` modes are always non-resonant. This is by design: the ``m = 2`` displacement shows
resonant behavior at ``q = 2``, while ``m = -2`` never does.

## Spectrum Output Sign Conventions

For the real-space representation, GPEC takes the complex conjugate for **right-handed**
configurations. This is implemented in the mode reconstruction
(`src/Analysis/PerturbedEquilibriumModes.jl`):

```julia
if helicity > 0
    theta_data[:, :, k] .= conj.(theta_data[:, :, k])
end
```

The straight-field-line offset enters as ``\exp(i\,n\,\nu)`` before the conjugation. For the full
Fourier representation ``\exp(i(m\theta - n\zeta))`` the conjugate operation flips both the
toroidal direction and the up/down (poloidal) sense.

### Interfacing with SURFMN

SURFMN expands in ``\exp(-im\theta - in\phi)`` with CCW ``\phi``. To convert from GPEC:

```python
m_surfmn = helicity * m_gpec
b_surfmn = real(b_m) - 1j * helicity * imag(b_m)
```

For LH configurations only the sign of ``m`` is flipped; for RH, ``m`` is unchanged but the complex
conjugate is taken.

### Interfacing with VACUUM

The VACUUM code uses CCW ``\phi`` and downward-outboard ``\theta``. GPEC uses the complex conjugate
of RH configurations when interfacing with VACUUM.

## Field Amplitudes and Units

GPEC reports perturbed magnetic fields as Fourier spectra on flux surfaces. The spectrum of a field
on a surface depends on in general on the coordinate system used. However, certain resonant components on rational surfaces or the 2-norm of the Fourier coefficient vector on any surface can be made coordinate-independent if the proper area weighting is used within the Fourier decomposition. The
coordinate-invariance of the weightings used in GPEC is established in Pharr (2026) (see [Citations](citations.md)). In each case, we normalize the harmonics by scalar area factors in such a way that the units are always Tesla. 

### Resonant Field

The resonant field is the pitch-resonant (``m = nq``) Fourier component of the perturbed flux at a
rational surface, divided by the **scalar area** ``A^r`` of that surface:

```math
b^r = \frac{\Phi^r}{A^r} \qquad [\mathrm{T}].
```

Dividing the resonant flux ``\Phi^r`` by the surface area turns it into a genuine field amplitude
that is invariant under changes of the poloidal-angle (working) coordinate. This is the quantity
reported as `resonant_flux`.

### Power-Normalized Field

The power-normalized field is the Fourier spectrum of the square-root-area-weighted normal field
``\sqrt{\mathcal{J}\,|\nabla\psi|}\,(\mathbf{b}\cdot\hat{\mathbf{n}})``, divided by the scalar
``\sqrt{A}`` of the surface:

```math
\tilde{b}_m = \frac{1}{\sqrt{A}}\oint \sqrt{\mathcal{J}\,|\nabla\psi|}\;
(\mathbf{b}\cdot\hat{\mathbf{n}})\, e^{-i m\theta}\, d\theta \qquad [\mathrm{T}].
```

The name comes from Parseval's theorem: the sum of its squared mode amplitudes equals the
area-averaged squared normal field — the surface-averaged field "power",

```math
\sum_m |\tilde{b}_m|^2 = \big\langle (\mathbf{b}\cdot\hat{\mathbf{n}})^2 \big\rangle.
```

The ``\sqrt{A}`` weighting is the unique one for which the spectrum transforms unitarily between
working coordinates, so the power-normalized amplitudes — and quantities derived from them, such as
the singular values of the resonant coupling matrix — are coordinate-invariant.


## Rotation Velocity Conventions (KineticForces)

The `KineticForces` module (the Julia port of PENTRC) conventions below are used in
`src/KineticForces/Torque.jl`, `EnergyIntegration.jl`, and `PitchIntegration.jl`.

### ``\omega_E`` (E × B Rotation)

- ``\omega_E`` is the input E × B toroidal-rotation profile, evaluated as
  `welec = omegaE_spline(psi)` (`Torque.jl`). Positive ``\omega_E`` means rotation in the direction
  of the toroidal coordinate ``\zeta``.
- The mapping to current direction follows from the ``\zeta``/``\phi`` convention fixed by helicity
  ([Helicity and Handedness](@ref)); it is not re-derived inside the kinetic module, which only
  reads the supplied profile.

### Diamagnetic Frequencies

Computed at evaluation time from the kinetic-profile gradients (Logan & Park 2013, Eq. 7;
`Torque.jl`):

```math
\omega_{*n} = -\frac{2\pi\, T_i}{e\, Z_i\, \chi_1\, n_i}\frac{dn_i}{d\psi_n}, \qquad
\omega_{*T} = -\frac{2\pi}{e\, Z_i\, \chi_1}\frac{dT_i}{d\psi_n},
```

where ``\chi_1 = 2\pi\psi_0`` (`chi1 = 2π·psio`). The negative signs mean a negative
(outward-decreasing) density or temperature gradient yields a positive diamagnetic frequency. The
total toroidal rotation is the sum

```math
\omega_\phi = \omega_E + \omega_{*n} + \omega_{*T} .
```

### Energy Integral Resonance

In the kinetic energy integral (`integrate_energy` in `EnergyIntegration.jl`) the resonance
denominator is ``i\,\Omega(x) - \nu``, where the resonance condition is

```math
\Omega(x) = n\,\omega_E + \ell_\mathrm{eff}\,\omega_b\sqrt{x} + n\,\omega_D\,x ,
```

with ``x = E/T`` the normalized energy and ``\nu`` the (energy-dependent) effective collisionality.
Here:

- ``\omega_b`` and ``\omega_D`` are the bounce and magnetic-precession frequency **coefficients at
  thermal energy** (``x = 1``): ``\omega_b \propto v_\mathrm{th}`` and
  ``\omega_D \propto v_\mathrm{th}^2`` (`wbhat`, `wdhat` in `Torque.jl`). Their velocity-space
  dependence enters explicitly as ``\sqrt{x}`` (bounce ``\propto v``) and ``x`` (precession
  ``\propto E``).
- ``\ell_\mathrm{eff}`` is the effective bounce harmonic (`PitchIntegration.jl`):

```math
\ell_\mathrm{eff} =
\begin{cases}
\ell + n q & \text{circulating (passing) particles},\\
\ell & \text{trapped particles}.
\end{cases}
```

The sign of ``\omega_E`` shifts the resonance location in velocity space.


## COCOS Compatibility

The conventions above are GPEC-native and predate the COCOS standard, as defined in
[Sauter, Comp. Phys. Comm. 2013](https://doi.org/10.1016/j.cpc.2012.09.010). What is **certain**
about the internal convention:

- Right-handed magnetic coordinates ``(\psi, \theta, \zeta)``.
- ``\psi`` excludes the ``2\pi`` factor (poloidal flux per radian) and increases from axis to edge.
- ``q > 0`` and ``n > 0`` in standard operation; ``F = |F|``.

The code labels this internal convention **COCOS 2** (`read_imas` in
`src/Equilibrium/ReadEquilibrium.jl`) and converts IMAS inputs accordingly:

```julia
imas_cocos = 11   # default: IMAS standard, ψ divided by 2π on read
imas_cocos = 2    # data already in the internal convention, no conversion
```

!!! warning "The COCOS index is not yet certified"
    The internal "COCOS 2" label is a working assumption, not a verified index. The IMAS reader
    converts COCOS 11 → "COCOS 2" by dividing ``\psi`` by ``2\pi`` **only**, whereas in the COCOS
    standard the pure ``2\pi`` partner of COCOS 11 is COCOS **1** (COCOS 2 differs additionally in
    the cylindrical/poloidal handedness ``\sigma_{R\phi Z}`` and in ``\mathrm{sign}(q)``). A
    rigorous classification would require auditing the ``\sigma_{B_p}``, ``\sigma_{R\phi Z}``, and
    ``\sigma_{\rho\theta\phi}`` signs against the GPEC field representation. Until that audit is
    done, treat the **practical input requirement** as the reliable statement: provide IMAS
    equilibria in COCOS 11 (default) or COCOS 2, selected via `imas_cocos`. Full COCOS conformance
    is tracked as a separate effort.

## See also

- [Workflow](workflow.md) --- where each convention enters the analysis pipeline
- [Citations](citations.md) --- theoretical references for GPEC's algorithms
