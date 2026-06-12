# Field Conventions

GPEC reports perturbed magnetic fields on flux surfaces as Fourier spectra. Following
Pharr (2026), *Coordinate-invariant flux-surface Fourier analysis in tokamaks* (see
[Citations](citations.md)), there are three Fourier decompositions of the normal field,
**all in field units (tesla)**, distinguished only by the **area weight applied to the
Fourier integrand**. This page is the authoritative in-repository source for how those
three quantities are named — in prose, in code, and in the HDF5 output.

!!! note "Authoritative field terminology"
    These names are canonical for the Julia GPEC codebase. The standalone Conventions
    Reference page (added by a separate documentation effort) should review and conform to
    the terminology in the table below — in particular, use **"root-area-weighted field"**
    rather than "power-normalized field", since "power-normalized" is a Parseval *property*
    and is not the term used by Pharr (2026).

## The three field amplitudes

| Pharr symbol | FT integrand weight ``W`` | English name | HDF5 token |
|---|---|---|---|
| ``b`` (bare) | ``1`` | **normal field** | `b_n` |
| ``\bar b`` (bar) | ``\mathcal{J}\,\lvert\nabla\psi\rvert`` | **area-weighted field** | `area` |
| ``\tilde b`` (tilde) | ``\sqrt{\mathcal{J}\,\lvert\nabla\psi\rvert}`` | **root-area-weighted field** | `root_area` |

All three carry units of tesla because each integral is divided by the appropriate power of
the coordinate-invariant scalar surface area ``A``:

```math
b_m = \oint (\mathbf{b}\cdot\hat{\mathbf n})\, e^{-i m\theta}\, d\theta, \qquad
\bar b_m = \frac{1}{A}\oint \mathcal{J}\lvert\nabla\psi\rvert\,(\mathbf{b}\cdot\hat{\mathbf n})\, e^{-i m\theta}\, d\theta, \qquad
\tilde b_m = \frac{1}{\sqrt{A}}\oint \sqrt{\mathcal{J}\lvert\nabla\psi\rvert}\,(\mathbf{b}\cdot\hat{\mathbf n})\, e^{-i m\theta}\, d\theta .
```

Only the **square-root-area weighting** ``\tilde b`` transforms unitarily between working
coordinates, so its 2-norm (and quantities derived from it, such as the singular values of
the resonant coupling matrix) are coordinate-invariant. The **full-area weighting**
``\bar b`` is coordinate-invariant for the pitch-resonant ``m = nq`` harmonic on a rational
surface — the **resonant area-weighted field** ``\bar b^{\,r} = \Phi^r/A^r``.

## Where each appears in GPEC output

- **`b_n`** — the bare normal field ``\mathbf{b}\cdot\hat{\mathbf n}`` (and the area-weighted
  radial field `b_psi_area_weighted`), under `perturbed_equilibrium/response/`.
- **`resonant_area_weighted_field`** / **`C_resonant_area_weighted_field`** — the resonant
  area-weighted field ``\bar b^{\,r}`` and its coupling matrix, under
  `perturbed_equilibrium/singular_coupling/`. The sibling `penetrated_area_weighted_field`
  follows the same convention.
- **Root-area-weighted (``\tilde b``) space** — the control-surface response matrices
  (`permeability`, `reluctance`, `plasma_inductance`, `surface_inductance`) are stored in
  this coordinate-invariant space under `perturbed_equilibrium/response_matrices/`; the
  `rootarea_field_to_flux_operator` recovers the flux-space forms. The
  coordinate-invariant ideal-MHD energies written by the stability stage
  (`FreeBoundaryStability/eigenmode_energies`) are likewise obtained through the
  root-area-weighted transform.

## See also

- [Citations](citations.md) — Pharr (2026) and the other theoretical references.
- [Perturbed Equilibrium](perturbed_equilibrium.md) — where these quantities are produced.
