# Galerkin Δ′ Solver (RDCON outer region)

The Galerkin solver is the Julia port of RDCON's `gal.f` — Dewar's singular Galerkin method for
the outer-region Δ′ computation.  Instead of integrating the Euler-Lagrange ODE radially, it
discretizes the displacement on a packed radial grid of Hermite-cubic finite elements and solves
a single global banded system.  Cells adjacent to each rational surface ("resonant" and
"extension" cells) carry the large/small power-law asymptotic solutions analytically, so the
singular behavior is built into the basis rather than resolved numerically.

The solve produces the inter-surface Δ′ matrix and the PEST-3 matching blocks
(``A'``, ``B'``, ``\Gamma'``, ``\Delta'``), written to the HDF5 output under the `galerkin/`
group.  These are the outer-region inputs to resistive matched-asymptotic stability analysis
[Glasser 2016, Phys. Plasmas **23**, 072505].

The implementation lives in `src/ForceFreeStates/Galerkin/`:

| File | Content |
|------|---------|
| `GalerkinStructs.jl` | Cell, interval, and workspace types mirroring the Fortran derived types |
| `GalerkinGrid.jl` | Packed grid construction and local→global DOF mapping |
| `GalerkinAssembly.jl` | Element-level assembly: Hermite basis, Gauss-Lobatto stiffness, resonant and extension cells, boundary conditions |
| `GalerkinSolve.jl` | Top-level driver `galerkin_solve`, banded solve, Δ′ extraction, PEST-3 blocks, HDF5 output |

## API Reference

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.ForceFreeStates]
Pages = ["Galerkin/GalerkinStructs.jl", "Galerkin/GalerkinGrid.jl", "Galerkin/GalerkinAssembly.jl", "Galerkin/GalerkinSolve.jl"]
```

## See also

- `docs/src/stability.md` — the ForceFreeStates module this solver belongs to (EL system, metric matrices, singular surfaces)
- `docs/src/inner_layer.md` — inner-layer solutions to be matched against the outer-region Δ′
