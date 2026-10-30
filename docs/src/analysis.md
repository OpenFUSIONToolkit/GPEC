# Analysis Module

The Analysis module provides post-processing and visualization utilities for GPEC simulation outputs.

## Submodules

- `ForceFreeStates`: Plotting functions for DCON-style ideal MHD stability results
- `Equilibrium`: Plotting functions for equilibrium objects
- `CoilForcing`: Plotting functions for coil geometry and normal field spectra
- `PerturbedEquilibrium`: Plotting functions for perturbed equilibrium results
- `PerturbedEquilibriumModes`: Data helpers to convert modal GPEC output to (ψ, θ) and (ψ, θ, φ) grids
- `EnergyDecomposition`: Decomposition of the plasma energy of free-boundary eigenmodes on the (ψ, θ) grid

```@docs
GeneralizedPerturbedEquilibrium.Analysis
```

## ForceFreeStates

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Analysis.ForceFreeStates]
```

## Equilibrium

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Analysis.Equilibrium]
```

## CoilForcing

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Analysis.CoilForcing]
```

## PerturbedEquilibrium

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Analysis.PerturbedEquilibrium]
```

## PerturbedEquilibriumModes

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Analysis.PerturbedEquilibriumModes]
```

## ErrorFields

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Analysis.ErrorFields]
```

## EnergyDecomposition

`Analysis.EnergyDecomposition` evaluates the ideal-MHD plasma potential energy ``\delta W_p`` of the
free-boundary eigenmodes of a forward ForceFreeStates solve on the ``(\psi,\theta)`` grid and splits it
into its physical terms. It is a scripted diagnostic, the Julia counterpart of the Fortran GPEC
`gpout_recon` with `effective_field_flag` and `standard_form_flag`:

```julia
using GeneralizedPerturbedEquilibrium
const ED = GeneralizedPerturbedEquilibrium.Analysis.EnergyDecomposition
ffs = GeneralizedPerturbedEquilibrium.main(["examples/Solovev_ideal_example"]).ffs
ed = ED.decompose_plasma_energy(ffs; eigenmodes=[1, 2])
ED.plot_energy_decomposition(ed; eigenmode=1, save_path="energy_decomposition.png")
```

It needs `integrator = "forward"`, `vac_flag = true`, a single toroidal mode number and an ideal solve.

### The two forms

The energy principle gives the *standard form*

```math
2\,\delta W_p = \int d\psi \oint d\theta\, \mathcal J \left[ \frac{|\delta\mathbf B|^2}{\mu_0}
- \mathbf j\cdot(\delta\mathbf B\times\boldsymbol\xi^*) + (\nabla\cdot\boldsymbol\xi)^*\,(\boldsymbol\xi\cdot\nabla p) \right],
```

and completing the square with the equilibrium current gives the *effective-field form*

```math
2\,\delta W_p = \int d\psi \oint d\theta\, \mathcal J \left[ \frac{|\delta\mathbf B_{\rm eff}|^2}{\mu_0} - K\,|\xi_n|^2 \right],
\qquad \delta\mathbf B_{\rm eff} = \delta\mathbf B + \xi_n\,(\mu_0 \mathbf j\times\hat n),
```

```math
K = K_1 + K_2 + K_3,\qquad K_1 = |\nabla\psi|^2\,\sigma S,\quad K_2 = \mu_0 B^2 \sigma^2,\quad K_3 = 2 p'\,\kappa_\psi,
\qquad \sigma = \frac{\mathbf j\cdot\mathbf B}{B^2},
```

with ``S`` the local shear and ``\kappa_\psi`` the curvature projection: the shear–current,
parallel-current and pressure–curvature drives. Both forms omit the compressional
``\gamma p|\nabla\cdot\boldsymbol\xi|^2`` term, as the Fortran does. The code keeps the shorthand `b` for
``\delta\mathbf B``.

The result holds the seven integrands `(b_eff_sq, K1, K2, K3, b_sq, j_b_xi, divxi_gradp)` per
``(\psi,\theta)`` node, their surface integrals and the two totals. Everything is scaled by
``2\mu_0/\psi_0^2``, so the totals compare directly with the DCON plasma energy
``e_{p,k}`` = `ffs.free_boundary.ep[k]` of the unit-power eigenvector ``\mathbf w_k``.

### Grid and accuracy

The radial nodes are the equilibrium ``\psi`` knots inside the solution domain, plus each end of the
domain that falls between knots: the start of the ODE sweep (`max(psilow, first knot)`, or a `qlow`
crossing) and the solution edge. On the knots all geometry is read from `ffs.metric`, and the
standard-form surface integral equals ``2/\psi_0^2`` times the DCON Lagrangian ``\xi'^\dagger F\xi' + 2\,\mathrm{Re}(\xi'^\dagger K\xi) + \xi^\dagger G\xi``
to roundoff, so the standard-form total differs from ``e_{p,k}`` only by the ``\psi`` quadrature of the knot
grid. ``\xi`` at each node is integrated from the nearest saved ODE step that no singular surface
separates from it (`ForceFreeStates.solution_state_at`), not splined between saved steps.

Choices that differ from the Fortran: ``B^2`` is the metric field strength
``(\chi'/\mathcal J)^2(g_{\theta\theta} + 2q g_{\theta\zeta} + q^2 g_{\zeta\zeta})``, which makes ``K_2|\xi_n|^2``
cancel against ``|\delta\mathbf B_{\rm eff}|^2``; the covariant field, ``\xi_n`` and ``\delta\mathbf B_{\rm eff}``
are formed pointwise in ``\theta`` rather than band-truncated; ``\partial_\theta(\mathcal J\xi^\theta)`` is
spectral; the reconstruction is unregularized; and the totals integrate over the knots rather than the ODE
`psifac` grid.

### Success checks

Every call reports two checks per eigenmode and warns when either fails:

- `relative_error_effective`, `relative_error_standard`: the mismatch of each total against
  ``e_{p,k}``, warned above 5 %. It is the ``\psi`` quadrature error of the knot grid, so a large value
  means `mpsi` is too coarse; it converges as in the table below. It does not check the algebra: an
  error that moves a total by a few percent, such as a factor 2 lost in ``K_3`` (4.8 % on the test
  fixture), stays below 5 %.
- `curl_residual`: the largest ``(\nabla\times\delta\mathbf B_{\rm eff})\cdot\nabla\psi`` over the knots,
  relative to the largest ``(|\partial_\theta \delta B_{{\rm eff},\zeta}| + |\partial_\zeta \delta B_{{\rm eff},\theta}|)/\mathcal J``
  it cancels, warned above 1e-8; normalization and threshold are those of the Fortran `gpeq_cveri`. It is
  an algebraic identity on the knots, so it checks how ``\delta\mathbf B_{\rm eff}`` is assembled from
  ``\delta\mathbf B``, ``\xi_n``, the metric and the equilibrium current, not the radial ``\xi`` solution,
  the kernel ``K`` or the radial integral. It is roundoff (4e-14) on the Solovev test fixture and 4e-12
  on the DIII-D-like example, but that example reaches 2e-5 with a uniform `mpsi = 512` grid and 5e-5 in
  PEST coordinates, all from knots inside ``\psi = 0.1``, without an error in ``\delta\mathbf B_{\rm eff}``.

Relative error of both forms against ``e_{p,k}`` on the Solovev test fixture (`mpert = 8`):

| `mpsi` | mode 1 effective / standard | mode 2 effective / standard |
|---|---|---|
| 16 | +1.38e-2 / +1.38e-2 | −1.99e-2 / −1.98e-2 |
| 32 | −1.53e-3 / −1.53e-3 | +1.01e-2 / +1.01e-2 |
| 64 | +2.40e-4 / +2.40e-4 | +8.36e-4 / +8.36e-4 |
| 128 | +1.66e-4 / +1.66e-4 | −1.63e-5 / −1.63e-5 |

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Analysis.EnergyDecomposition]
```
