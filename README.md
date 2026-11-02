# GPEC — Generalized Perturbed Equilibrium Code

This Julia implementation of GPEC continues the legacy of the Fortran [GPEC suite](https://github.com/PrincetonUniversity/GPEC) for magnetohydrodynamic (MHD) analysis of fusion plasmas. The Fortran suite (DCON, VACUUM, STRIDE, GPEC, PENTRC and SLAYER) has long been used to model the response of tokamak plasmas to small non-axisymmetric magnetic perturbations, from error-field correction and resonant magnetic perturbation optimization to kink and tearing stability. This implementation consolidates them into one code base of cooperating physics modules that share data in memory behind a unified, modern interface, which removes the hand-offs between separate codes and makes parameter scans, uncertainty quantification and integrated modeling practical.

One input deck runs the stages it names, in order, into one output file:

1. **Equilibrium** — reads an EFIT or CHEASE reconstruction, an IMAS data dictionary or an analytic model, maps it to straight-field-line flux coordinates, and finds the rational surfaces q = m/n.
2. **Stability** — ideal MHD stability by Newcomb's criterion (the DCON integration, or the STRIDE Riccati and RDCON Galerkin formulations): the force-free states ξ(ψ,θ), the free-boundary kink energies with the VACUUM response, and the tearing stability matrix Δ′.
3. **Perturbed equilibrium** — the plasma response to external non-axisymmetric fields (RMPs, error fields, correction coils): resonant fields, island half-widths and the Chirikov overlap parameter.
4. **Torque, error fields and tearing** — the neoclassical toroidal viscosity torque (PENTRC), coil-tolerance and locking-risk analysis, and tearing growth rates and critical resonant fields from inner-layer models (SLAYER, GGJ).

Full documentation: [https://openfusiontoolkit.github.io/GPEC/dev/](https://openfusiontoolkit.github.io/GPEC/dev/)

## Quick Start

GPEC is configured via a `gpec.toml` file. To run an analysis on a directory containing that file:

```bash
./gpec path/to/directory
```

To use GPEC programmatically from Julia:

```julia
using GeneralizedPerturbedEquilibrium
GeneralizedPerturbedEquilibrium.main(["path/to/directory"])
```

See [Setup](https://openfusiontoolkit.github.io/GPEC/dev/set_up/) for full installation instructions.

## Citing

If you use GPEC, please cite it as given in [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" button), along with the key papers behind the modules you use, listed on the [citations page](https://openfusiontoolkit.github.io/GPEC/dev/citations/).

The 3D vacuum singular correction in `src/Vacuum/Kernel3D.jl` is derived from [BIEST](https://github.com/dmalhotra/BIEST) and licensed under Apache-2.0 (see [`NOTICE`](NOTICE) and [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt)); its papers are listed with the [Vacuum module citations](https://openfusiontoolkit.github.io/GPEC/dev/citations/#3D-vacuum-singular-correction). The rest of GPEC is MIT-licensed.

## Contributing

The Git workflow, commit message conventions, and the regression-testing harness are documented in the [Developer Notes](https://openfusiontoolkit.github.io/GPEC/dev/developer_notes/).
