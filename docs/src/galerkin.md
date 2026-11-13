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
error_field_response = true        # error-field (EF) columns (Delta_EF): the ideal ξ solution and W_plasma, needed by [Match]
```

`cut_solution = true` (default `false`) adds the cut solution that composite inner-region profiles need.

It replaces the radial integration rather than supplementing it: the run computes its own vacuum response at the control surface (with `vac_flag`), produces no ODE trace and no `crit` scan, and does not support the kinetic solve. With `error_field_response = true` and `vac_flag = true` it also produces the free-boundary energies of its ideal solution, so it can drive `[PerturbedEquilibrium]` like a `"Forward"` solve; a `[Match]`ed result cannot yet.
With `error_field_response = true` the error-field (EF) columns, one per edge mode with unit edge displacement and no inner-layer drive, are solved with the edge held fixed; their small-solution coefficients are `Delta_EF`, they are the ideal (perfectly shielded) ``\xi`` solution, and the force they exert on the edge is the fixed-boundary plasma energy matrix ``W_P`` of Eq. 66 of [Glasser (2016)](citations.md#ForceFreeStates).
A `[Match]` section matches the inner layer to its Δ′ and replaces them with the driven ``\xi`` solution, whose edge response ``W_P`` is complex: its anti-Hermitian part is the torque absorbed in the layers. The composite inner-region profiles need `cut_solution = true` (`Galerkin(; cut_solution=true)` followed by `solve(MatchProblem(ffs))` in a script).

## Outputs

| Dataset | Meaning |
|---|---|
| `SingularSurfaces/Delta_prime_matrix` | the Δ′ matrix in the PEST-3 convention, on the same path as the Riccati result, so downstream stages never branch on which formalism ran; free-boundary with `vac_flag = true` (with `error_field_response = true` a second factorization re-solves the big-solution columns with the vacuum edge, so the EF columns never enter it; `free_boundary_delta_prime = false` skips it for matching-only scans, leaving no tearing Δ′) and fixed-boundary otherwise |
| `SingularSurfaces/Delta_prime_raw_freeboundary`, `Delta_prime_raw_fixedboundary`, `Delta_EF`, `edge_response` | the side-major matching matrices with the vacuum and the fixed edge, the error-field (EF) response (`error_field_response = true`), and the edge momentum of each fixed-edge big-solution column |
| `SingularSurfaces/pest3_A`, `pest3_B`, `pest3_Gamma` (and `_fixedboundary`) | the PEST-3 interchange and mixed blocks ``A'``, ``B'``, ``\Gamma'`` |
| `ForceFreeStates/Solutions/GalerkinIntegration/` | the closed ``\xi`` profiles on the Galerkin grid, the surface list, the factorization used (`solver`) and, with `basis_output = true`, the `Basis/` group |
| `ForceFreeStates/FreeBoundaryStability/` | `W_plasma`, the plasma energy matrix ``W_P`` of the closed solution (`error_field_response = true`), and with `vac_flag = true` the vacuum and free-boundary energies and eigenmodes as for a `"Forward"` solve |

## Numerical notes

- The radial grid is packed toward the rational surfaces and the edge, and the banded system is factorized by a Hermitian Cholesky decomposition, falling back to a Jacobi-scaled LU when the band is not Hermitian positive definite; the factorization used is recorded as `solver`.
- The singular cells make the solution insensitive to the resolution at the rational surfaces, so convergence is set by the smooth part of the grid (`nx`).

## API reference

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.ForceFreeStates]
Pages = ["Galerkin/GalerkinStructs.jl", "Galerkin/GalerkinGrid.jl", "Galerkin/GalerkinAssembly.jl", "Galerkin/GalerkinSolution.jl", "Galerkin/GalerkinSolve.jl"]
```

See [Stability](stability.md) for the module this solver belongs to and [Tearing](inner_layer.md) for the inner-layer models matched to its Δ′.
