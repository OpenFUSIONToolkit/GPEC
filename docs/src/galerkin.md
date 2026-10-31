# Galerkin Δ′ Solver (RDCON outer region)

The Galerkin solver is the Julia port of RDCON's `gal.f` — Dewar's singular Galerkin method for
the outer-region Δ′ computation.  Instead of integrating the Euler-Lagrange ODE radially, it
discretizes the displacement on a packed radial grid of Hermite-cubic finite elements and solves
a single global banded system.  Cells adjacent to each rational surface ("resonant" and
"extension" cells) carry the large/small power-law asymptotic solutions analytically, so the
singular behavior is built into the basis rather than resolved numerically.

The ideal coefficient is reconstructed as ``F = (QL)(QL)^\dagger`` from the interpolated
Cholesky factor ``L``, with ``Q = \operatorname{diag}(m-nq)``. This uses the same coefficient
representation as the Euler–Lagrange integration and singular-surface expansions, including
between spline knots, and preserves Hermitian positive semidefiniteness of ``F``.

The solve produces the inter-surface Δ′ matrix and the PEST-3 matching blocks
(``A'``, ``B'``, ``\Gamma'``, ``\Delta'``), written to the HDF5 output under the same
`SingularSurfaces/` paths the Riccati Δ′ uses (`Delta_prime_matrix`, `Delta_prime_raw`,
`Delta_coil`, `pest3_A`, `pest3_B`, `pest3_Gamma`) — one set of names per quantity,
whichever formalism produced it.  These are the outer-region inputs to resistive matched-asymptotic stability analysis
[Glasser 2016, Phys. Plasmas **23**, 072505].

Select it with `integrator = "Galerkin"` in `[ForceFreeStates]`; its settings go in `[Galerkin]`. It replaces the radial ODE
integration rather than supplementing it: the run computes its own vacuum response at the
control surface (when `vac_flag`) and produces no free-boundary energies or ODE trace.
A `[Match]` section additionally matches the inner layer, giving a driven ξ solution. The
composite inner-region profiles need the cut solution, `[Galerkin] cut_solution = true` (or
`Galerkin(; cut_solution=true)` followed by `solve(MatchProblem(...))`).

The banded solve automatically tries a Hermitian Cholesky
factorization of the lower band (LAPACK `zpbtrf`/`zpbtrs`, with the Dirichlet rows' columns moved
to the right-hand side). The free block's off-diagonal entries must be Hermitian within
`1e-10 * sqrt(abs(A_ii)) * sqrt(abs(A_jj))`; the diagonal's imaginary part is deliberately
discarded. Unequal bandwidths, off-diagonal asymmetry, or failure of positive definiteness
trigger a reported fallback to Jacobi-scaled LU. The factorization actually used is
recorded as `ForceFreeStates/Solutions/GalerkinIntegration/solver` in `gpec.h5`.

The implementation lives in `src/ForceFreeStates/Galerkin/`:

| File | Content |
|------|---------|
| `GalerkinStructs.jl` | Cell, interval, and workspace types mirroring the Fortran derived types |
| `GalerkinGrid.jl` | Packed grid construction and local→global DOF mapping |
| `GalerkinAssembly.jl` | Element-level assembly: Hermite basis, Gauss-Lobatto stiffness, resonant and extension cells, boundary conditions |
| `GalerkinSolution.jl` | Reconstruct ξ(ψ) and analytic ξ′(ψ) on the gal-native grid |
| `GalerkinSolve.jl` | Top-level driver `galerkin_solve`, banded solve, Δ′ extraction, PEST-3 blocks, HDF5 output |

## API Reference

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.ForceFreeStates]
Pages = ["Galerkin/GalerkinStructs.jl", "Galerkin/GalerkinGrid.jl", "Galerkin/GalerkinAssembly.jl", "Galerkin/GalerkinSolution.jl", "Galerkin/GalerkinSolve.jl"]
```

## See also

- `docs/src/stability.md` — the ForceFreeStates module this solver belongs to (EL system, metric matrices, singular surfaces)
- `docs/src/inner_layer.md` — inner-layer solutions to be matched against the outer-region Δ′
