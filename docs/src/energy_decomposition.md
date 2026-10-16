# Energy Decomposition of Free-Boundary Eigenmodes

`EnergyDecomposition.jl` evaluates the ideal-MHD plasma potential energy ``\delta W_p`` of a
free-boundary eigenmode of the ForceFreeStates solve on the ``(\psi,\theta)`` grid and splits
it into its physical terms. It is the Julia counterpart of the reconstruction diagnostics on the
`recon` branch of the Fortran GPEC (its `docs/tex/recon/main.tex` note carries the full
derivation); "Fortran" on this page and in the code comments refers to that branch. This page
keeps only what is needed to read the output. The stage covers the free-boundary DCON
eigenmodes only: the edge condition is the free-boundary eigenvector and the reference energy is
its ``e_{p,k}``, which fixed-boundary or forced-response perturbations do not supply.

## The two forms of ``\delta W_p``

The ideal-MHD energy principle (Bernstein et al. 1958; Boozer 2004, Sec. V.C) writes the fluid
potential energy of a displacement ``\boldsymbol\xi`` with perturbed field
``\delta\mathbf B = \nabla\times(\boldsymbol\xi\times\mathbf B)`` in the *standard form*

```math
2\,\delta W_p = \int d\psi \oint d\theta\, \mathcal J \left[ \frac{|\delta\mathbf B|^2}{\mu_0}
- \mathbf j\cdot(\delta\mathbf B\times\boldsymbol\xi^*) + (\nabla\cdot\boldsymbol\xi)^*\,(\boldsymbol\xi\cdot\nabla p) \right],
```

and, after completing the square with the equilibrium current (Bernstein 1983; Chance et al.
1992, Eq. 12c; Boozer 2004, Eq. 100), in the *effective-field form*

```math
2\,\delta W_p = \int d\psi \oint d\theta\, \mathcal J \left[ \frac{|\delta\mathbf B_{\rm eff}|^2}{\mu_0}
- K\,|\xi_n|^2 \right],
\qquad
\delta\mathbf B_{\rm eff} = \delta\mathbf B + \xi_n\,(\mu_0 \mathbf j\times\hat n),
\quad \xi_n = \boldsymbol\xi\cdot\hat n .
```

Both drop the compressional ``\gamma p|\nabla\cdot\boldsymbol\xi|^2`` term, a reconstruction
assumption shared with the Fortran. The destabilizing kernel
``K = 2\,(\mathbf j\times\hat n)\cdot(\mathbf B\cdot\nabla\hat n)`` is split as in Chance et al.
(1992, Eqs. 15–17) into

```math
K = K_1 + K_2 + K_3,\qquad
K_1 = |\nabla\psi|^2\,\sigma S,\quad
K_2 = \mu_0 B^2 \sigma^2,\quad
K_3 = 2 p'\,\kappa_\psi,\qquad
\sigma = \frac{\mathbf j\cdot\mathbf B}{B^2},
```

with ``S`` the local magnetic shear (Chance et al. 1992, Eq. 17; its surface average is the
global shear ``q'``), ``\kappa_\psi = \boldsymbol\kappa\cdot\nabla\psi`` the curvature projection
and ``p'`` the pressure gradient. The three ``K`` terms are the shear–current coupling, the
parallel-current drive and the pressure–curvature (ballooning) drive.

The effective field is the vector ``\mathbf C`` of Boozer (2004, Eq. 98): its curl is the current
of the perturbed equilibrium, which has to lie in the perturbed flux surfaces, so
``\nabla\psi\cdot\nabla\times\delta\mathbf B_{\rm eff} = \nabla\cdot(\delta\mathbf B_{\rm eff}\times\nabla\psi) = 0``
(Boozer 2004, Eq. 99; Park 2009, §2.2). The decomposition evaluates this identity on every
equilibrium knot and reports the residual; a non-zero value means the reconstructed field
components are inconsistent. It is an algebraic identity there, so the residual sits at roundoff
(1e-13). It validates that ``\delta\mathbf B_{\rm eff}`` was assembled consistently from
``\delta\mathbf B``, ``\xi_n`` and the metric on each surface; it does not test the radial
``\xi`` solution, the curvature kernel ``K``, the radial integral or the convergence of the totals.

## Normalization and the reference energy

The eigenmode is built from the free-boundary eigenvector ``\mathbf w_k`` of the
``(W,N)`` pencil with unit power norm, exactly as `FreeBoundaryStability/eigenmode_plasma_energies`
is. Every stored energy is scaled by ``2\mu_0/\psi_0^2`` so that `dW_plasma` reads in the same
normalization, and `dW_plasma_reference` holds the corresponding entry ``e_{p,k}``. The relative
mismatch of each form against the reference is stored and, above 5 %, reported with a warning.
On the Solovev test fixture (`mpsi = 16`, `mpert = 8`) both forms reproduce ``e_{p,k}`` to ≲2 %
(effective / standard form +1.7 % / +1.4 % for the least stable mode, −2.0 % / −2.0 % for the
second), converging with resolution (+5.0e-3 / +7.9e-5 and +5.4e-4 / −1.1e-4 at `mpsi = 128`).
The effective-field form needs the metric-consistent
``B^2 = (\chi'/\mathcal J)^2(g_{\theta\theta} + 2q g_{\theta\zeta} + q^2 g_{\zeta\zeta})`` for
``\sigma``, ``K_2`` and ``\kappa_\psi`` rather than the equilibrium's separate ``|B|`` spline.
The standard-form remainder is pure ``\psi``-quadrature error of the DCON Lagrangian (converging
roughly as ``h^2``–``h^3``). The remaining offset between the two forms is equilibrium noise from
the fixture's loose `etol = 1e-7`: with `etol = 1e-10` it drops 50–300× to ≲1e-4 for both modes at
`mpsi` = 64 and 128. The pressure-curvature term ``K_3`` (through ``\partial_\psi B^2``) carries
most of it for the least stable mode but not all of it for the second, and at `etol = 1e-7` every
term is rough near the edge.

## Running it

```toml
[EnergyDecomposition]
eigenmodes = [1]              # Free-boundary eigenmodes to decompose (1 = least stable)
effective_field_form = true   # Decompose δW_p in the effective-field form ½∫[|δB_eff|²/μ₀ − K|ξ_n|²]
standard_form = true          # Decompose δW_p in the standard form ½∫[|δB|²/μ₀ − j·(δB×ξ*) + (∇·ξ)*(ξ·∇p)]
write_densities = false       # Also store the per-(ψ,θ) energy densities with R,Z for 2-D maps
```

The stage runs right after ForceFreeStates, also with `force_termination = true`, and needs the
dense ``\xi`` solution (`integrator = "forward"`), the free-boundary eigenmodes
(`vac_flag = true`), a single toroidal mode number and an ideal solve; otherwise it warns and is
skipped. From a script:

```julia
result = GeneralizedPerturbedEquilibrium.main(["examples/Solovev_ideal_example"])
energy = GeneralizedPerturbedEquilibrium.energy_decomposition(result.ffs; eigenmodes=[1, 2])
```

The radial nodes are the equilibrium ``\psi`` knots inside the solution domain plus the solution
edge: on the knots the metric, the Euler–Lagrange matrices and the geometry are all exact, which
is what keeps the curl identity at roundoff. ``\xi`` and ``\xi'`` are splined there from the
solution grid and ``\xi_s`` is recomputed from the matrices at each node. The Fortran evaluates
its profiles on the same knots but integrates its totals over the ODE `psifac` grid.

## Output

Everything lands in `PerturbedEquilibrium/EnergyDecomposition/` of `gpec.h5`, profiles on the
decomposition ``\psi`` nodes with one column per decomposed eigenmode; `eigenmode_index` is the
coordinate of that axis (the free-boundary eigenmode number, 1 = least stable):

| Dataset | Content |
|---|---|
| `effective_b_squared` (+ `_psi`, `_theta`, `_zeta`) | ``\oint \mathcal J\,\lvert\delta\mathbf B_{\rm eff}\rvert^2/\mu_0\,d\theta`` and its component split |
| `shear_current`, `parallel_current_squared`, `pressure_curvature` | ``\oint \mathcal J\,K_i\lvert\xi_n\rvert^2\,d\theta`` for ``K_1, K_2, K_3`` |
| `b_squared`, `current_coupling`, `pressure_compression` | the three standard-form pieces |
| `dW_cumulative`, `dW_cumulative_standard_form` | running ``\tfrac12\int d\psi`` of each form |
| `dW_plasma`, `dW_plasma_standard_form`, `dW_plasma_reference` | the totals and ``e_{p,k}`` |
| `dW_plasma_relative_error`, `dW_plasma_standard_form_relative_error` | mismatch against the reference |
| `effective_b_curl_residual_max`, `_rms`, `_relative` | the ``\nabla\psi\cdot\nabla\times\delta\mathbf B_{\rm eff} = 0`` check |
| `Densities/…` | per-``(\psi,\theta)`` integrands with `R`, `Z`, only with `write_densities` |

A disabled form leaves its datasets zero-extent. The dataset names and the API docstrings below keep the
code's shorthand `b` for the perturbed field ``\delta\mathbf B``.

## References

- I. B. Bernstein, E. A. Frieman, M. D. Kruskal, R. M. Kulsrud, "An energy principle for hydromagnetic stability problems", *Proc. R. Soc. London A* **244**, 17 (1958), [doi:10.1098/rspa.1958.0023](https://doi.org/10.1098/rspa.1958.0023) — the energy principle.
- I. B. Bernstein, in *Basic Plasma Physics I*, Handbook of Plasma Physics Vol. 1, edited by A. A. Galeev and R. N. Sudan (North-Holland, Amsterdam, 1983), p. 421 — the derivation of the effective-field form.
- M. S. Chance, Y.-C. Sun, S. C. Jardin, C. E. Kessel, M. Okabayashi, "MHD stability of tokamak plasmas", PPPL-CFP-2687, 2nd Symposium on Plasma Dynamics, Trieste (1992), [OSTI 7234275](https://www.osti.gov/biblio/7234275) — Eqs. (12a)–(12c) for the forms of ``\delta W_p``, Eqs. (15)–(17) for ``K = K_1 + K_2 + K_3`` and the local shear ``S``.
- A. H. Boozer, "Physics of magnetically confined plasmas", *Rev. Mod. Phys.* **76**, 1071 (2004), Sec. V.C, [doi:10.1103/RevModPhys.76.1071](https://doi.org/10.1103/RevModPhys.76.1071) — the energy principle, the effective field ``\mathbf C`` (Eq. 98), the ``(\nabla\times\mathbf C)\cdot\nabla\psi = 0`` constraint (Eq. 99) and the split ``w = C^2/2\mu_0 - w_d`` (Eq. 100).
- J.-K. Park, *Ideal Perturbed Equilibria in Tokamaks*, PhD thesis, Princeton University (2009), §2.2 — the effective field as the surface-current field.
- A. H. Glasser, "The direct criterion of Newcomb for the ideal MHD stability of an axisymmetric toroidal plasma", *Phys. Plasmas* **23**, 072505 (2016), [doi:10.1063/1.4958328](https://doi.org/10.1063/1.4958328) — the DCON coordinates, matrices and shear used here.

## API

```@docs
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.EnergyDecompositionControl
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.EnergyDecompositionResult
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.decompose_energy
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.write_energy_decomposition!
```
