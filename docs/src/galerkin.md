# Galerkin Solver

The Galerkin solver is the third `[ForceFreeStates] integrator`: the singular Galerkin method of the Fortran RDCON for the outer-region Δ′ of resistive stability ([Glasser, Wang and Park (2016)](citations.md#Tearing)).

## What it computes

Instead of integrating the Euler–Lagrange system radially, it discretizes the displacement on a packed radial grid of Hermite-cubic finite elements and solves one global banded system.
The cells adjacent to each rational surface carry the large and small power-law asymptotic solutions analytically, so the singular behavior is built into the basis rather than resolved numerically, and the solve yields the Δ′ matrix and the PEST-3 matching blocks of every surface at once.

## Governing equations

The ideal coefficient is reconstructed as ``F = (QL)(QL)^\dagger`` from the interpolated Cholesky factor ``L`` with the resonance factor ``Q = \operatorname{diag}(m - nq)``, the same representation the Euler–Lagrange integration and the singular-surface expansions use, so ``F`` stays Hermitian positive semidefinite between spline knots.
The weak form of the same system as on the [Stability](stability.md) page is assembled on the Hermite basis, with the resonant and extension cells next to each rational surface carrying the asymptotic solutions ``|\psi - \psi_s|^{-1/2 \pm \sqrt{-D_I}}``.

## Running it

Select it in `[ForceFreeStates]` and set its options in `[Galerkin]`, as in `examples/DIIID-like_gal_resistive_example` (abridged):

```toml
[ForceFreeStates]
integrator = "Galerkin"

[Galerkin]
nx = 256                # Hermite-cubic elements per interval between singular surfaces
pfac = 0.03             # grid packing ratio toward the singular surfaces
rpec_flag = true        # edge coil-response columns (Delta_coil), needed by [Match] and [Tearing]
```

`cut_solution = true` (default `false`) adds the cut solution that composite inner-region profiles need.

It replaces the radial integration rather than supplementing it: the run computes its own vacuum response at the control surface (with `vac_flag`) but produces no free-boundary energies, no ODE trace and no `crit` scan, it does not support the kinetic solve, and its result cannot drive `[PerturbedEquilibrium]`, which needs the free-boundary δW of a `"Forward"` solve.
A `[Match]` section matches the inner layer to its Δ′ and gives a driven ``\xi`` solution; the composite inner-region profiles need `cut_solution = true` (`Galerkin(; cut_solution=true)` followed by `solve(MatchProblem(ffs))` in a script).

## Outputs

| Dataset | Meaning |
|---|---|
| `SingularSurfaces/Delta_prime_matrix` | the Δ′ matrix in the PEST-3 convention, on the same path as the Riccati result, so downstream stages never branch on which formalism ran |
| `SingularSurfaces/Delta_prime_raw`, `Delta_coil` | the side-major matching matrix and the coil-response columns (`rpec_flag = true`) |
| `SingularSurfaces/pest3_A`, `pest3_B`, `pest3_Gamma` | the PEST-3 interchange and mixed blocks ``A'``, ``B'``, ``\Gamma'`` |
| `ForceFreeStates/Solutions/GalerkinIntegration/` | the closed ``\xi`` profiles on the Galerkin grid, the surface list, the factorization used (`solver`) and, with `basis_output = true`, the `Basis/` group |

Nothing is written under `ForceFreeStates/FreeBoundaryStability/`.

## Numerical notes

- The radial grid is packed toward the rational surfaces and the edge, and the banded system is factorized by a Hermitian Cholesky decomposition, falling back to a Jacobi-scaled LU when the band is not Hermitian positive definite; the factorization used is recorded as `solver`.
- The singular cells make the solution insensitive to the resolution at the rational surfaces, so convergence is set by the smooth part of the grid (`nx`).

## API reference

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.ForceFreeStates]
Pages = ["Galerkin/GalerkinStructs.jl", "Galerkin/GalerkinGrid.jl", "Galerkin/GalerkinAssembly.jl", "Galerkin/GalerkinSolution.jl", "Galerkin/GalerkinSolve.jl"]
```

See [Stability](stability.md) for the module this solver belongs to and [Tearing](inner_layer.md) for the inner-layer models matched to its Δ′.
