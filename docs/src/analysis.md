# Analysis

`Analysis` holds the plotting and post-processing functions for GPEC results, grouped by the stage they read; each works on the in-memory result or on `gpec.h5`.

## Submodules

- `ForceFreeStates`: the eigenmode energies, the `crit` scan and the displacement profiles of a stability solve
- `Equilibrium`: flux surfaces, profiles and the radial grid of an equilibrium
- `CoilForcing`: coil geometry and the normal-field spectra it drives
- `PerturbedEquilibrium`: the plasma response, the resonant fields, island widths and Chirikov parameter against ψ_N or q
- `PerturbedEquilibriumModes`: helpers that convert the modal output to (ψ, θ) and (ψ, θ, φ) grids for real-space plots
- `ErrorFields`: coil sensitivities, tolerance distributions, locking risk and correction limits (see [Error Fields](error_fields.md))
- `EnergyDecomposition`: the plasma energy of free-boundary eigenmodes split into effective-field and standard-form terms on the (ψ, θ) grid

`examples/DIIID-like_ideal_example/analyze_example.jl`, `examples/Solovev_ideal_example/analyze_example.jl` and `examples/DIIID-like_error_field_example/analyze_example.jl` are worked examples of each; the s–α maps of local stability use `salpha_reference`, `compute_ballooning_stability!` and `scan_delta_prime_map`.
Figure conventions for spectrum plots are in `docs/development/plotting.md` of the repository, and `write_imas` on the [Utilities](utilities.md) page exports results to an IMAS data dictionary.

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

A scripted diagnostic, the Julia counterpart of the Fortran `gpout_recon` with `effective_field_flag` and `standard_form_flag`. It needs `integrator = "forward"`, `vac_flag = true`, one toroidal mode number and an ideal solve.

```julia
using GeneralizedPerturbedEquilibrium
const ED = GeneralizedPerturbedEquilibrium.Analysis.EnergyDecomposition
ffs = GeneralizedPerturbedEquilibrium.main(["examples/Solovev_ideal_example"]).ffs
ed = ED.decompose_plasma_energy(ffs; eigenmodes=[1, 2])
ED.plot_energy_decomposition(ed; eigenmode=1, save_path="energy_decomposition.png")
```

The ψ integrals run over every saved ODE step as well as the equilibrium knots, since the knots alone miss the structure of ξ at the rational surfaces. The effective-field total can differ from the standard-form one by a grid-independent amount whose cause is still open (−0.4 % on the DIII-D-like example).

Relative error against `ffs.free_boundary.ep` on the Solovev test fixture (PEST, `mpert = 8`):

| `mpsi` | mode 1 effective / standard | mode 2 effective / standard |
|---|---|---|
| 16 | −4.44e-3 / −4.50e-3 | −3.19e-2 / −3.23e-2 |
| 32 | −2.17e-6 / −8.19e-6 | −3.35e-5 / −6.86e-5 |
| 64 | −1.07e-6 / −1.37e-6 | −9.14e-6 / −1.11e-5 |
| 128 | −2.01e-6 / −1.98e-6 | −1.72e-5 / −1.72e-5 |

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Analysis.EnergyDecomposition]
```
