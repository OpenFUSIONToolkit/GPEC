# Conventions Reference

The sign, coordinate and field-amplitude conventions used throughout GPEC, for reading its output and interfacing it with other codes.
They are inherited from the Fortran GPEC suite and predate the COCOS standard (see [COCOS Compatibility](@ref)); each section names the source that establishes the convention.

## Quick Reference

| Quantity | Convention | Forced? |
|----------|-----------|---------|
| ``\psi`` (poloidal flux) | Normalized 0 (axis) to 1 (edge), no ``2\pi`` | Yes, positive |
| ``\theta`` (poloidal angle) | Upward outboard, one circuit = 1 | Fixed |
| ``\phi`` (toroidal angle) | CCW for LH, CW for RH | By helicity |
| ``F = R B_\phi`` | Always positive | Yes, `abs` |
| ``q`` (safety factor) | Computed (direct) / input (inverse), positive | Effectively positive |
| ``n`` (toroidal mode) | Always positive (``\ge 1``) | By convention |
| Resonant ``m`` | Always positive (since ``n>0``, ``q>0``) | By construction |
| helicity | +1 RH, ``-1`` LH | Computed |
| ``\omega_E`` | Positive = direction of ``\zeta`` | No |
| Field amplitudes | ``b``, ``\bar b``, ``\tilde b``, all in tesla | By area weighting |

## Coordinate System

GPEC uses right-handed magnetic coordinates ``(\psi, \theta, \zeta)`` with the Fourier kernel

```math
\exp\!\big(i(m\theta - n\zeta)\big),
```

matching the Fortran ``\exp(im\theta - in\phi)`` convention.
The analysis transform that produces the mode coefficients ``f_m`` of a profile ``f(\theta)`` uses the conjugate kernel ``\exp(-im\theta)``, and the synthesis reconstructs with ``\exp(+im\theta)`` (`Utilities.FourierTransforms`).

### Poloidal Flux ``\psi``

- Normalized from 0 (magnetic axis) to 1 (plasma boundary); ``\psi_0 = |\psi_\mathrm{bry} - \psi_\mathrm{axis}|`` is forced positive on read, and an EQDSK with ``\psi_\mathrm{bry} < \psi_\mathrm{axis}`` has its flux array sign-flipped so that ``\psi`` increases outward (`Equilibrium.Sources.GFile`).
- The internal ``\psi`` carries no factor of ``2\pi`` (poloidal flux per radian); IMAS inputs, which use the full flux, are divided by ``2\pi`` on read.
- ``\rho = \sqrt{\psi}`` serves as a radius-like variable where one is needed.

### Poloidal Angle ``\theta``

- Upward outboard, always: ``\theta = 0`` on the outboard midplane, increasing upward.
- Normalized to increase by 1, not ``2\pi``, over one poloidal circuit, whatever the helicity or working coordinate.

### Toroidal Coordinate ``\zeta`` and ``\phi``

- The magnetic toroidal angle is ``\zeta = (\phi - \nu(\psi,\theta))/(2\pi)``, with ``\phi`` the internal toroidal angle and ``\nu`` the single-valued straight-field-line offset of the working coordinate (``\nu = 0`` in PEST coordinates).
- The physical toroidal angle is ``\phi = -\,\mathrm{helicity}\,(2\pi\zeta + \nu)`` (`ForcingTerms`), so ``\phi`` runs counter-clockwise viewed from above for left-handed configurations and clockwise for right-handed ones.

### Working Coordinate Options

`[Equilibrium] jac_type` selects the Jacobian ``J \propto B_p^{p_{bp}}\, B^{p_b}\, R^{-p_r}\, r_c^{-p_{rc}}``, with ``B_p`` the poloidal field, ``B`` the total field, ``R`` the major radius and ``r_c`` the distance from the magnetic axis:

| Name | `power_bp` | `power_b` | `power_r` | `power_rc` |
|------|-----------|-----------|-----------|------------|
| Hamada (default) | 0 | 0 | 0 | 0 |
| PEST | 0 | 0 | 2 | 0 |
| Boozer | 0 | 2 | 0 | 0 |
| Equal-arc | 1 | 0 | 0 | 0 |
| Park | 0 | 1 | 0 | 0 |

`jac_type = "custom"` sets the four exponents by hand through `jac_custom_power_bp`, `jac_custom_power_b`, `jac_custom_power_r` and `jac_custom_power_rc`.

## Helicity and Handedness

Helicity is the product of the toroidal-field and plasma-current directions, `helicity = bt_sign * sign(crnt)` (`ForcingTerms`): +1 is right-handed (``B_t`` and ``I_p`` parallel), ``-1`` left-handed (opposed).
It sets the toroidal-grid direction ``\phi_j = -\mathrm{helicity}\times 2\pi j/n_\zeta`` and the output sign flips of [Spectrum Output Sign Conventions](@ref); for DIII-D (``B_t < 0``, ``I_p > 0``, helicity ``-1``) ``\phi`` increases with ``j``.

## ``F = R B_\phi``

The poloidal-current function is forced positive on read, ``F = |F|`` (`Equilibrium.Sources.GFile`); the sign of ``B_t`` is carried separately by `fpol_sign` and the helicity inputs.

## Safety Factor ``q``

The direct solver computes ``q`` by field-line integration and overwrites the file profile (`Equilibrium.Axisymmetric.Direct`),

```math
q = \frac{F}{2\pi}\oint \frac{J}{B_p}\,dl_p ,
```

which is positive because ``J > 0`` and ``F = |F|``; a warning marks an on-axis extrapolation with ``q_0 \le 0`` as a spline artifact.
Inverse equilibria use the ``q`` profile supplied with the input (`Equilibrium.Axisymmetric.Inverse`), the convention of the Fortran GPEC CHEASE reader.

## Mode Numbers ``m`` and ``n``

### Toroidal Mode Number ``n``

`[ForceFreeStates] nn` is a positive integer or a list of consecutive integers.
``n < 1`` is not supported: a run whose modes are all below 1 is an error, and a lowest mode below 1 is clamped to 1 with a warning (`ForceFreeStates.Run`).
Results for several ``n`` can be superposed.

### Poloidal Mode Range

The poloidal spectrum spans `mlow` to `mhigh` (`ForceFreeStates.ForceBalance`):

```math
m_\mathrm{low} = \min(\lfloor n_\mathrm{low}\,q_\mathrm{min}\rfloor,\,0) - 4 - \Delta m_\mathrm{low}, \qquad
m_\mathrm{high} = \lfloor n_\mathrm{high}\,q_\mathrm{max}\rfloor + 2\,\Delta m_\mathrm{high}.
```

`delta_mlow` and `delta_mhigh` widen the range beyond the resonant modes; `delta_mhigh` is doubled before use, the Fortran DCON convention, so it adds twice as many modes on the high side as `delta_mlow` does on the low side.

### Why Positive ``m`` Is Always Resonant

Resonant surfaces are the flux surfaces where ``m = n\,q(\psi)`` (`ForceFreeStates.Surfaces`).
Since ``n > 0`` by convention and ``q > 0`` for a standard tokamak, the resonant ``m`` is always positive, ``m_\mathrm{res} = n\,q > 0``, and negative-``m`` modes are never resonant: the ``m = 2`` displacement resonates at ``q = 2``.

## Spectrum Output Sign Conventions

For the real-space representation GPEC takes the complex conjugate of the mode amplitudes for right-handed configurations (`Analysis.PerturbedEquilibriumModes`), after the straight-field-line offset has entered as ``\exp(i\,n\,\nu)``.
For the full Fourier representation ``\exp(i(m\theta - n\zeta))`` the conjugation flips both the toroidal direction and the up/down sense.

### Interfacing with SURFMN

SURFMN expands in ``\exp(-im\theta - in\phi)`` with CCW ``\phi``; to convert from GPEC,

```python
m_surfmn = helicity * m_gpec
b_surfmn = real(b_m) - 1j * helicity * imag(b_m)
```

so that for LH configurations only the sign of ``m`` is flipped and for RH the complex conjugate is taken.

### Interfacing with Vacuum

The `Vacuum` module works in the same angles as the rest of GPEC, ``\theta`` upward-outboard and the basis ``\exp(-i(m\theta - 2\pi n\zeta))``, unlike the Fortran VACUUM code whose ``\theta`` and ``\phi`` both run the other way; a clockwise plasma contour is rejected and a clockwise wall file is reversed with a warning.
The ``\nu`` passed to it is GPEC's ``\nu = \phi - 2\pi\zeta``, which Chance (1997, Eq. 97) writes as ``\zeta = \phi + \nu``.

## Field Amplitudes and Units

GPEC reports perturbed fields as Fourier spectra on flux surfaces.
A raw spectrum depends on the working coordinate ([Park, Boozer and Menard (2008)](citations.md)), but the resonant component on a rational surface and the 2-norm of the spectrum on any surface become coordinate-invariant with the right area weighting in the Fourier integrand ([Pharr et al. (2026)](citations.md)).
GPEC therefore works with three decompositions of the normal field, all in tesla, distinguished only by that weight:

| Pharr symbol | Integrand weight | Name | HDF5 token |
|---|---|---|---|
| ``b`` (bare) | ``1`` | normal field | `b_n` |
| ``\bar b`` (bar) | ``\mathcal{J}\,\lvert\nabla\psi\rvert`` | area-weighted field | `area` |
| ``\tilde b`` (tilde) | ``\sqrt{\mathcal{J}\,\lvert\nabla\psi\rvert}`` | root-area-weighted field | `root_area` |

```math
b_m = \oint (\mathbf{b}\cdot\hat{\mathbf n})\, e^{-i m\theta}\, d\theta, \qquad
\bar b_m = \frac{1}{A}\oint \mathcal{J}\lvert\nabla\psi\rvert\,(\mathbf{b}\cdot\hat{\mathbf n})\, e^{-i m\theta}\, d\theta, \qquad
\tilde b_m = \frac{1}{\sqrt{A}}\oint \sqrt{\mathcal{J}\lvert\nabla\psi\rvert}\,(\mathbf{b}\cdot\hat{\mathbf n})\, e^{-i m\theta}\, d\theta ,
```

with ``A`` the scalar surface area.
The root-area weighting is the unique one under which the spectrum transforms unitarily between working coordinates, so ``\sum_m |\tilde b_m|^2 = \langle (\mathbf{b}\cdot\hat{\mathbf n})^2 \rangle`` (Parseval) and the quantities derived from ``\tilde b``, such as the singular values of the resonant coupling matrix, are coordinate-invariant; "power-normalized field" in early GPEC literature names this property.
The area weighting is coordinate-invariant for the pitch-resonant ``m = nq`` harmonic on a rational surface: the **resonant field** ``\bar b^{\,r} = \Phi^r/A^r``, the resonant flux divided by the surface area, stored as `PerturbedEquilibrium/SingularCoupling/resonant_area_weighted_field`.
In an ideal perturbed equilibrium the surface is perfectly shielded and the stored value is the effective resonant field, the resonant flux the shielding current cancels; resistive and kinetic equilibria have finite resonant fields as well.

With ``\Sigma`` the mode-space square-root-weight convolution (`sqrtamat`) and ``A`` the surface area (`jarea`), the three fields and the flux are related by (`Equilibrium.CoordinateInvariant`)

```math
\tilde b = \Sigma\, b, \qquad
\bar b = (\Sigma/\sqrt{A})\,\tilde b, \qquad
\Phi = A\,\bar b = \Sigma\sqrt{A}\;\tilde b ,
```

where `rootarea_to_area_weight` ``= \Sigma/\sqrt{A}`` maps ``\tilde b \to \bar b``, `area_to_rootarea_weight` is its inverse, and ``R = \Sigma\sqrt{A}`` is the internal flux-conform operator.
Flux ``\Phi`` is never stored: it appears only when a user supplies flux-valued forcing and is recovered on demand as ``\Phi = A\,\bar b``.

### Field Representations in GPEC Output

- `PerturbedEquilibrium/forcing_b`, `forcing_b_root_area`, `forcing_b_area` and the `response_*` triplet: the control-surface spectra in the bare, root-area-weighted and area-weighted representations.
- `PerturbedEquilibrium/Response/b_n` and `b_psi_area_weighted`: the bare normal field and the area-weighted radial field profiles.
- `PerturbedEquilibrium/SingularCoupling/resonant_area_weighted_field`, `C_resonant_area_weighted_field` and `penetrated_area_weighted_field`: the resonant area-weighted field, its coupling matrix and the penetrated field.
- `PerturbedEquilibrium/ResponseMatrices/`: the response matrices in root-area-weighted space, with `rootarea_to_area_weight_operator` ``S`` (``L_{\bar b} = S\,\tilde L\,S^\dagger``) and `surface_area` ``A`` (``\Phi = A\,\bar b``) to recover the other forms.
- `ForceFreeStates/FreeBoundaryStability/eigenmode_energies`: energies per unit surface-averaged squared displacement, a normalization described on the [Stability](stability.md) page.

## KineticForces rotation conventions

The rotation profiles enter the `KineticForces` integrals as follows.

### ``\omega_E`` (E × B Rotation)

``\omega_E`` is the input E × B toroidal rotation profile, positive in the direction of the toroidal coordinate ``\zeta``; its relation to the current direction follows from the ``\zeta``/``\phi`` convention fixed by helicity ([Helicity and Handedness](@ref)) and is not re-derived by the kinetic module.

### Diamagnetic Frequencies

Computed at evaluation time from the kinetic-profile gradients (Logan et al. (2013), Eq. 7):

```math
\omega_{*n} = -\frac{2\pi\, T_i}{e\, Z_i\, \chi_1\, n_i}\frac{dn_i}{d\psi_n}, \qquad
\omega_{*T} = -\frac{2\pi}{e\, Z_i\, \chi_1}\frac{dT_i}{d\psi_n},
```

with ``\chi_1 = 2\pi\psi_0``, so that an outward-decreasing density or temperature gives a positive diamagnetic frequency; the total toroidal rotation is ``\omega_\phi = \omega_E + \omega_{*n} + \omega_{*T}``.

### Energy Integral Resonance

The resonance denominator of the energy integral is ``i\,\Omega(x) - \nu`` with

```math
\Omega(x) = n\,\omega_E + \ell_\mathrm{eff}\,\omega_b\sqrt{x} + n\,\omega_D\,x ,
\qquad
\ell_\mathrm{eff} = \begin{cases} \ell + n q & \text{passing particles},\\ \ell & \text{trapped particles}, \end{cases}
```

where ``x = E/T`` is the normalized energy, ``\nu`` the energy-dependent effective collisionality, and ``\omega_b``, ``\omega_D`` the bounce and precession frequencies at thermal energy (``\omega_b \propto v_\mathrm{th}``, ``\omega_D \propto v_\mathrm{th}^2``).
The sign of ``\omega_E`` shifts the resonance in velocity space.

## COCOS Compatibility

The conventions above are GPEC-native and predate the COCOS standard of [Sauter and Medvedev (2013)](https://doi.org/10.1016/j.cpc.2012.09.010): right-handed ``(\psi, \theta, \zeta)``, ``\psi`` without the ``2\pi`` and increasing outward, ``q > 0``, ``n > 0`` and ``F = |F|``.
The code labels this internal convention COCOS 2 (`Equilibrium.Sources.IMAS`) and accepts IMAS input in two forms:

```julia
imas_cocos = 11   # default: IMAS standard, ψ divided by 2π on read
imas_cocos = 2    # data already in the internal convention, no conversion
```

The conversion from COCOS 11 divides ``\psi`` by ``2\pi`` and relies on the forced sign conventions above (positive ``\psi``, ``F`` and ``q``) for the rest; no other COCOS index is converted.

## See also

- [Workflow](workflow.md), where each convention enters the pipeline
- [Citations](citations.md), the references behind the conventions
