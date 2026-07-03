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

- The magnetic coordinate toroidal angle is ``\zeta = \phi/(2\pi) + \nu(\psi,\theta)``, where ``\nu`` is
  a single-valued straight-field-line offset that depends on the working coordinate. PEST
  coordinates have ``\nu = 0``.
- The physical toroidal angle is reconstructed as
  ``\phi = -\,\mathrm{helicity}\,(2\pi\zeta + \nu)`` (`sample_boundary_grid` in
  `src/ForcingTerms/CoilFourier.jl`). Thus ``\phi`` is effectively **counter-clockwise** (viewed
  from above) for left-handed (LH) configurations and **clockwise** for right-handed (RH)
  configurations.

### Working Coordinate Options

Controlled by `jac_type` in the `[Equilibrium]` section of `gpec.toml`. The Jacobian is
``J \propto B_p^{p_{bp}}\, B^{p_b}\, R^{-p_r}\, r_c^{-p_{rc}}``, where ``B_p`` is the poloidal
field magnitude, ``B`` is the total field magnitude, ``R`` is the major radius, and
``r_c = \sqrt{(R-R_0)^2 + (Z-Z_0)^2}`` is the minor radius (distance from the magnetic axis
``(R_0, Z_0)``):

| Name | `power_bp` | `power_b` | `power_r` | `power_rc` |
|------|-----------|-----------|-----------|------------|
| Hamada (default) | 0 | 0 | 0 | 0 |
| PEST | 0 | 0 | 2 | 0 |
| Boozer | 0 | 2 | 0 | 0 |
| Equal-arc | 1 | 0 | 0 | 0 |
| Park | 0 | 1 | 0 | 0 |

The powers are set automatically from `jac_type` (`EquilibriumConfig` in
`src/Equilibrium/EquilibriumTypes.jl`) and applied in the field-line Jacobian
(`direct_fieldline_der!` in `src/Equilibrium/DirectEquilibrium.jl`). Setting `jac_type = "custom"`
lets you specify the four exponents manually via `jac_custom_power_bp`, `jac_custom_power_b`,
`jac_custom_power_r`, and `jac_custom_power_rc`.

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
- **Inverse** equilibria, for which only CHEASE and analytic options are currently supported, instead use the ``q`` profile supplied with the input
  (`src/Equilibrium/InverseEquilibrium.jl`). This follows the fortran GPEC CHEASE reading convention.

## Mode Numbers ``m`` and ``n``

### Toroidal Mode Number ``n``

- Set as a **positive integer** via `nn_low`/`nn_high` in `gpec.toml`. ``n < 1`` is not supported
  and is clamped to 1 with a warning (`src/GeneralizedPerturbedEquilibrium.jl`). Results for
  multiple ``n`` can be superposed.

### Poloidal Mode Range

The poloidal spectrum spans `mlow` to `mhigh` (`src/GeneralizedPerturbedEquilibrium.jl`):

```math
m_\mathrm{low} = \min(\lfloor n\,q_\mathrm{min}\rfloor,\,0) - 4 - \Delta m_\mathrm{low}, \qquad
m_\mathrm{high} = \lfloor n\,q_\mathrm{max}\rfloor + 2\,\Delta m_\mathrm{high}.
```

`delta_mlow` and `delta_mhigh` widen the range beyond the resonant modes. Note the asymmetric
factor of 2 on ``\Delta m_\mathrm{high}``: `delta_mhigh` is doubled before being applied (for
consistency with the Fortran DCON convention), so the user-supplied value adds twice as many
modes on the high side as `delta_mlow` does on the low side.

### Why Positive ``m`` Is Always Resonant

Resonant surfaces are found by `sing_find!` (`src/ForceFreeStates/Sing.jl`), which locates flux
surfaces where ``m = n\,q(\psi)``. Since ``n > 0`` by convention and ``q > 0`` for a standard
tokamak, the resonant ``m`` is always positive:

```math
m_\mathrm{res} = n\,q > 0 .
```

Negative-``m`` modes are always non-resonant. This is by design: the ``m = 2`` displacement shows
resonant behavior at ``q = 2``, etc.

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

### Interfacing with Vacuum

The Vacuum code uses CCW ``\phi`` and downward-outboard ``\theta``. GPEC uses the complex conjugate
of RH configurations when interfacing with Vacuum.

## Field Amplitudes and Units

GPEC reports perturbed magnetic fields as Fourier spectra on flux surfaces. The spectrum of a field
on a surface depends in general on the coordinate system used. However, certain resonant components
on rational surfaces or the 2-norm of the Fourier coefficient vector on any surface can be made
coordinate-independent if the proper area weighting is used within the Fourier decomposition. The
coordinate-invariance of the weightings used in GPEC is established in Pharr (2026) (see
[Citations](citations.md)). In each case, we normalize the harmonics by scalar area factors in such
a way that the units are always Tesla.

### Resonant Field

The resonant field is the pitch-resonant (``m = nq``) Fourier component of the perturbed flux at a
rational surface, divided by the **scalar area** ``A^r`` of that surface:

```math
b^r = \frac{\Phi^r}{A^r} \qquad [\mathrm{T}].
```

Dividing the resonant flux ``\Phi^r`` by the surface area turns it into a field amplitude in Tesla
that is invariant under changes of the poloidal-angle (working) coordinate. This is the quantity
reported as `resonant_flux`. The need for this normalization is the central point of Park, Boozer,
and Menard (2008) (see [Citations](citations.md)): the raw Fourier spectrum of the perturbed field is
*spectrally asymmetric* and changes with the magnetic working coordinate, so only the resonant
harmonic scaled by the scalar surface area ``A^r`` is a coordinate-independent physical measure of
the resonant field.

Note that there is strictly NO resonant field in ideal MHD perturbed equilibrium calculations, and
only the effective resonant field is non-zero in these cases. Effective resonant fields are computed
as above but for the resonant component of flux that is being shielded by the shielding current layer
on the rational surface. Resistive and kinetic MHD perturbed equilibria have, in general, finite
resonant fields as well as effective resonant fields.

### Root-Area Normalized Field

The root-area normalized field is the Fourier spectrum of the square-root-area-weighted normal field
``\sqrt{\mathcal{J}\,|\nabla\psi|}\,(\mathbf{b}\cdot\hat{\mathbf{n}})``, divided by the scalar
``\sqrt{A}`` of the surface:

```math
\tilde{b}_m = \frac{1}{\sqrt{A}}\oint \sqrt{\mathcal{J}\,|\nabla\psi|}\;
(\mathbf{b}\cdot\hat{\mathbf{n}})\, e^{-i m\theta}\, d\theta \qquad [\mathrm{T}].
```

According to Parseval's theorem, the sum of its squared mode amplitudes equals the
area-averaged squared normal field — the surface-averaged field "power",

```math
\sum_m |\tilde{b}_m|^2 = \big\langle (\mathbf{b}\cdot\hat{\mathbf{n}})^2 \big\rangle.
```

The ``\sqrt{A}`` weighting is the unique one for which the spectrum transforms unitarily between
working coordinates, so the power-normalized amplitudes — and quantities derived from them, such as
the singular values of the resonant coupling matrix — are coordinate-invariant.

### The Three Field Amplitudes

Following Pharr (2026), *Coordinate-invariant flux-surface Fourier analysis in tokamaks* (see
[Citations](citations.md)), GPEC works with **three** Fourier decompositions of the normal field, **all
in field units (tesla)**, distinguished only by the **area weight applied to the Fourier integrand**.
These are the authoritative names used in prose, in code, and in the HDF5 output:

| Pharr symbol | FT integrand weight ``W`` | English name | HDF5 token |
|---|---|---|---|
| ``b`` (bare) | ``1`` | **normal field** | `b_n` |
| ``\bar b`` (bar) | ``\mathcal{J}\,\lvert\nabla\psi\rvert`` | **area-weighted field** | `area` |
| ``\tilde b`` (tilde) | ``\sqrt{\mathcal{J}\,\lvert\nabla\psi\rvert}`` | **root-area-weighted field** | `root_area` |

All three carry units of tesla because each integral is divided by the appropriate power of the
coordinate-invariant scalar surface area ``A``:

```math
b_m = \oint (\mathbf{b}\cdot\hat{\mathbf n})\, e^{-i m\theta}\, d\theta, \qquad
\bar b_m = \frac{1}{A}\oint \mathcal{J}\lvert\nabla\psi\rvert\,(\mathbf{b}\cdot\hat{\mathbf n})\, e^{-i m\theta}\, d\theta, \qquad
\tilde b_m = \frac{1}{\sqrt{A}}\oint \sqrt{\mathcal{J}\lvert\nabla\psi\rvert}\,(\mathbf{b}\cdot\hat{\mathbf n})\, e^{-i m\theta}\, d\theta .
```

Only the **square-root-area weighting** ``\tilde b`` transforms unitarily between working coordinates, so
its 2-norm (and quantities derived from it, such as the singular values of the resonant coupling matrix)
are coordinate-invariant. Note that the "root-area-weighted field" sometimes appears as "power-normalized
field" in early GPEC literature, as the latter names a Parseval *property* of ``\tilde b``. The **full-area weighting** ``\bar b`` is coordinate-invariant for the
pitch-resonant ``m = nq`` harmonic on a rational surface — the **resonant area-weighted field**
``\bar b^{\,r} = \Phi^r/A^r`` as described above.

GPEC operates and outputs **only in these field representations — poloidal flux ``\Phi`` (weber) is never
stored.** Flux appears, briefly, only internally when a user supplies flux-valued forcing, and is
recovered on demand as the scalar product ``\Phi = A\,\bar b``.

### Translation Operators

With ``\Sigma`` ≡ `sqrtamat` (the mode-space ``\sqrt{}``weight convolution) and the scalar surface area
``A`` ≡ `jarea`, the three fields are related by (`Equilibrium/CoordinateInvariant.jl`):

```math
\tilde b = \Sigma\, b, \qquad
\bar b = (\Sigma/\sqrt{A})\,\tilde b, \qquad
\Phi = A\,\bar b = \Sigma\sqrt{A}\;\tilde b .
```

- **`rootarea_to_area_weight`** ``= \Sigma/\sqrt{A}`` maps ``\tilde b \to \bar b``.
- **`area_to_rootarea_weight`** ``= \sqrt{A}\,\Sigma^{-1}`` is its inverse (``\bar b \to \tilde b``).

The internal flux-conform operator is just ``R = \Sigma\sqrt{A} = `` `rootarea_to_area_weight` ``\cdot A``.

### Field Representations in GPEC Output

- **`forcing_b` / `forcing_b_root_area` / `forcing_b_area`** (and the `response_*` triplet) — the
  control-surface forcing and response spectra in the bare (``b``), root-area-weighted (``\tilde b``) and
  area-weighted (``\bar b``) representations, under `perturbed_equilibrium/`.
- **`b_n`** — the bare normal field ``\mathbf{b}\cdot\hat{\mathbf n}`` (and the area-weighted radial field
  `b_psi_area_weighted`), under `perturbed_equilibrium/response/`.
- **`resonant_area_weighted_field`** / **`C_resonant_area_weighted_field`** — the resonant area-weighted
  field ``\bar b^{\,r} = \Phi^r/A^r`` and its coupling matrix, under
  `perturbed_equilibrium/singular_coupling/`. The sibling `penetrated_area_weighted_field` follows the
  same convention.
- **Root-area-weighted (``\tilde b``) space** — the control-surface response matrices (`permeability`,
  `reluctance`, `plasma_inductance`, `surface_inductance`) are stored in this coordinate-invariant space
  under `perturbed_equilibrium/response_matrices/`. The stored `rootarea_to_area_weight_operator` ``S``
  recovers the area-weighted field forms (``L_{\bar b} = S\,\tilde L\,S^\dagger``) and the scalar
  `surface_area` ``A`` recovers flux (``\Phi = A\,\bar b``). The coordinate-invariant ideal-MHD energies
  written by the stability stage (`FreeBoundaryStability/eigenmode_energies`) are the ``\tilde b``
  quadratic form scaled by the scalar ``c = A``: ``\mathrm{d}W = c\,\tilde b^\dagger W_t\,\tilde b``.


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
[Sauter, Comp. Phys. Comm. 2013](https://doi.org/10.1016/j.cpc.2012.09.010). The internal convention:

- Right-handed magnetic coordinates ``(\psi, \theta, \zeta)``.
- ``\psi`` excludes the ``2\pi`` factor (poloidal flux per radian) and increases from axis to edge.
- ``q > 0`` and ``n > 0`` in standard operation; ``F = |F|``.

The code labels this internal convention **COCOS 2** (`read_imas` in
`src/Equilibrium/ReadEquilibrium.jl`) and converts IMAS inputs accordingly:

```julia
imas_cocos = 11   # default: IMAS standard, ψ divided by 2π on read
imas_cocos = 2    # data already in the internal convention, no conversion
```

!!! warning "The COCOS indexing is still a beta feature"
    Note that the IMAS reader converts COCOS 11 → "COCOS 2" by simply dividing ``\psi`` by ``2\pi``
    (i.e. conversion from COCOS 11 to 1) and relying on the standardly enforced sign conventions
    above to eventually conform to the COCOS 2 conventions.

## See also

- [Workflow](workflow.md) --- where each convention enters the analysis pipeline
- [Citations](citations.md) --- theoretical references for GPEC's algorithms
