# Energy Decomposition of Free-Boundary Eigenmodes

`EnergyDecomposition.jl` evaluates the ideal-MHD plasma potential energy ``\delta W_p`` of a
free-boundary eigenmode of the ForceFreeStates solve on the ``(\psi,\theta)`` grid and splits
it into its physical terms. It is the Julia counterpart of the reconstruction diagnostics of
the Fortran GPEC (its `docs/tex/recon/main.tex` note carries the full derivation); this page
keeps only what is needed to read the output.

## The two forms of ``\delta W_p``

Bernstein's energy principle (Bernstein et al. 1958) writes the fluid potential energy of a
displacement ``\boldsymbol\xi`` with perturbed field ``\mathbf b = \nabla\times(\boldsymbol\xi\times\mathbf B)``
in the *standard form*

```math
2\,\delta W_p = \int d\psi \oint d\theta\, \mathcal J \left[ \frac{|\mathbf b|^2}{\mu_0}
- \mathbf j\cdot(\mathbf b\times\boldsymbol\xi^*) + (\nabla\cdot\boldsymbol\xi)^*\,(\boldsymbol\xi\cdot\nabla p) \right],
```

and, after completing the square with the equilibrium current, in the *effective-field form*

```math
2\,\delta W_p = \int d\psi \oint d\theta\, \mathcal J \left[ \frac{|\mathbf b_{\rm eff}|^2}{\mu_0}
- K\,|\xi_n|^2 \right],
\qquad
\mathbf b_{\rm eff} = \mathbf b + \xi_n\,(\mu_0 \mathbf j\times\hat n),
\quad \xi_n = \boldsymbol\xi\cdot\hat n .
```

Both drop the compressional ``\gamma p|\nabla\cdot\boldsymbol\xi|^2`` term, a reconstruction
assumption shared with the Fortran. The destabilizing kernel is

```math
K = K_1 + K_2 + K_3,\qquad
K_1 = |\nabla\psi|^2\,\sigma S,\quad
K_2 = \mu_0 B^2 \sigma^2,\quad
K_3 = 2 p'\,\kappa_\psi,\qquad
\sigma = \frac{\mathbf j\cdot\mathbf B}{B^2},
```

with ``S`` the DCON magnetic shear, ``\kappa_\psi = \boldsymbol\kappa\cdot\nabla\psi`` the curvature
projection and ``p'`` the pressure gradient. The three ``K`` terms are the shear–current
coupling, the parallel-current drive and the pressure–curvature (ballooning) drive.

The effective field is the field whose curl is a genuine surface current: the equilibrium
current is tangent to the flux surfaces, so ``\nabla\psi\cdot\nabla\times\mathbf b_{\rm eff} = 0``
(Park 2009, §2.2). The decomposition evaluates this identity on every equilibrium knot and
reports the residual; a non-zero value means the reconstructed field components are
inconsistent. It is an algebraic identity there, so the residual sits at roundoff (1e-13).

## Normalization and the reference energy

The eigenmode is built from the free-boundary eigenvector ``\mathbf w_k`` of the
``(W,N)`` pencil with unit power norm, exactly as `FreeBoundaryStability/eigenmode_plasma_energies`
is. Every stored energy is scaled by ``2\mu_0/\psi_0^2`` so that `dW_plasma` reads in the same
normalization, and `dW_plasma_reference` holds the corresponding entry ``e_{p,k}``. The relative
mismatch of each form against the reference is stored and, above 5 %, reported with a warning.
On the Solovev example the standard form reproduces ``e_{p,1}`` to 1e-5 and the effective-field
form to 1 %, the same order the Fortran reports for DIII-D.

## Running it

```toml
[EnergyDecomposition]
eigenmodes = [1]              # Free-boundary eigenmodes to decompose (1 = least stable)
effective_field_form = true   # Decompose δW_p in the effective-field form ½∫[|b_eff|²/μ₀ − K|ξ_n|²]
standard_form = true          # Decompose δW_p in the standard form ½∫[|b|²/μ₀ − j·(b×ξ*) + (∇·ξ)*(ξ·∇p)]
write_densities = false       # Also store the per-(ψ,θ) energy densities with R,Z for 2-D maps
```

The stage runs right after ForceFreeStates, also with `force_termination = true`, and needs the
dense ``\xi`` solution (`integrator = "forward"`), the free-boundary eigenmodes
(`vac_flag = true`) and an ideal solve; otherwise it warns and is skipped. From a script:

```julia
result = GeneralizedPerturbedEquilibrium.main(["examples/Solovev_ideal_example"])
energy = GeneralizedPerturbedEquilibrium.energy_decomposition(result.ffs; eigenmodes=[1, 2])
```

The radial nodes are the equilibrium ``\psi`` knots inside the solution domain plus the solution
edge: on the knots the metric, the Euler–Lagrange matrices and the geometry are all exact, which
is what keeps the curl identity at roundoff. ``\xi`` and ``\xi'`` are splined there from the
solution grid and ``\xi_s`` is recomputed from the matrices at each node.

## Output

Everything lands in `ForceFreeStates/EnergyDecomposition/` of `gpec.h5`, profiles on the
decomposition ``\psi`` nodes with one column per decomposed eigenmode:

| Dataset | Content |
|---|---|
| `effective_b_squared` (+ `_psi`, `_theta`, `_zeta`) | ``\oint \mathcal J\,\lvert\mathbf b_{\rm eff}\rvert^2/\mu_0\,d\theta`` and its component split |
| `shear_current`, `parallel_current_squared`, `pressure_curvature` | ``\oint \mathcal J\,K_i\lvert\xi_n\rvert^2\,d\theta`` for ``K_1, K_2, K_3`` |
| `b_squared`, `current_coupling`, `pressure_compression` | the three standard-form pieces |
| `dW_cumulative`, `dW_cumulative_standard_form` | running ``\tfrac12\int d\psi`` of each form |
| `dW_plasma`, `dW_plasma_standard_form`, `dW_plasma_reference` | the totals and ``e_{p,k}`` |
| `dW_plasma_relative_error`, `dW_plasma_standard_form_relative_error` | mismatch against the reference |
| `effective_b_curl_residual_max`, `_rms`, `_relative` | the ``\nabla\psi\cdot\nabla\times\mathbf b_{\rm eff} = 0`` check |
| `Densities/…` | per-``(\psi,\theta)`` integrands with `R`, `Z`, only with `write_densities` |

A disabled form leaves its datasets zero-extent.

## References

- I. B. Bernstein, E. A. Frieman, M. D. Kruskal, R. M. Kulsrud, *Proc. R. Soc. A* **244**, 17 (1958) — the energy principle and its effective-field form.
- J.-K. Park, *Ideal Perturbed Equilibria in Tokamaks*, PhD thesis, Princeton (2009), §2.2 — the effective field as the surface-current field.
- A. H. Glasser, *Phys. Plasmas* **23**, 112506 (2016) — the DCON coordinates, matrices and shear used here.

## API

```@docs
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.EnergyDecompositionControl
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.EnergyDecompositionResult
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.decompose_energy
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.write_energy_decomposition!
```
