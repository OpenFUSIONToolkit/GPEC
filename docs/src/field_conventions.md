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

GPEC operates and outputs **only in these field representations — poloidal flux ``\Phi`` (weber)
is never stored.** Flux appears, briefly, only internally when a user supplies flux-valued forcing,
and is recovered on demand as the scalar product ``\Phi = A\,\bar b``.

## Translation operators

With ``\Sigma`` ≡ `sqrtamat` (the mode-space ``\sqrt{}``weight convolution) and the scalar surface
area ``A`` ≡ `jarea`, the three fields are related by (`Equilibrium/CoordinateInvariant.jl`):

```math
\tilde b = \Sigma\, b, \qquad
\bar b = (\Sigma/\sqrt{A})\,\tilde b, \qquad
\Phi = A\,\bar b = \Sigma\sqrt{A}\;\tilde b .
```

- **`rootarea_to_area_weight`** ``= \Sigma/\sqrt{A}`` maps ``\tilde b \to \bar b``.
- **`area_to_rootarea_weight`** ``= \sqrt{A}\,\Sigma^{-1}`` is its inverse (``\bar b \to \tilde b``).

The internal flux-conform operator is just ``R = \Sigma\sqrt{A} = `` `rootarea_to_area_weight` ``\cdot A``.

## Where each appears in GPEC output

- **`forcing_b` / `forcing_b_root_area` / `forcing_b_area`** (and the `response_*` triplet) — the
  control-surface forcing and response spectra in the bare (``b``), root-area-weighted (``\tilde b``)
  and area-weighted (``\bar b``) representations, under `perturbed_equilibrium/`.
- **`b_n`** — the bare normal field ``\mathbf{b}\cdot\hat{\mathbf n}`` (and the area-weighted
  radial field `b_psi_area_weighted`), under `perturbed_equilibrium/response/`.
- **`resonant_area_weighted_field`** / **`C_resonant_area_weighted_field`** — the resonant
  area-weighted field ``\bar b^{\,r} = \Phi^r/A^r`` and its coupling matrix, under
  `perturbed_equilibrium/singular_coupling/`. The sibling `penetrated_area_weighted_field`
  follows the same convention.
- **Root-area-weighted (``\tilde b``) space** — the control-surface response matrices
  (`permeability`, `reluctance`, `plasma_inductance`, `surface_inductance`) are stored in
  this coordinate-invariant space under `perturbed_equilibrium/response_matrices/`. The stored
  `rootarea_to_area_weight_operator` ``S`` recovers the area-weighted field forms
  (``L_{\bar b} = S\,\tilde L\,S^\dagger``) and the scalar `surface_area` ``A`` recovers flux
  (``\Phi = A\,\bar b``). The coordinate-invariant ideal-MHD energies written by the stability
  stage (`FreeBoundaryStability/eigenmode_energies`) are the ``\tilde b`` quadratic form scaled by
  the scalar ``c = A``: ``\mathrm{d}W = c\,\tilde b^\dagger W_t\,\tilde b``.

## See also

- [Citations](citations.md) — Pharr (2026) and the other theoretical references.
- [Perturbed Equilibrium](perturbed_equilibrium.md) — where these quantities are produced.
