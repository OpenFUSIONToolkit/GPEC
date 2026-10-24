# GPEC — Generalized Perturbed Equilibrium Code

GPEC is a Julia implementation of the [GPEC suite](https://github.com/PrincetonUniversity/GPEC) for magnetohydrodynamic (MHD) analysis of fusion plasmas. The Fortran suite (DCON, VACUUM, STRIDE, GPEC, PENTRC and SLAYER) has long been used to model the response of tokamak plasmas to small non-axisymmetric magnetic perturbations, from error-field correction and resonant magnetic perturbation optimization to kink and tearing stability. This implementation consolidates them into one code base of cooperating physics modules that share data in memory behind a unified, modern interface, which removes the hand-offs between separate codes and makes parameter scans, uncertainty quantification and integrated modeling practical.

It performs a three-stage analysis pipeline:

1. **Equilibrium reconstruction** from experimental data (EFIT, CHEASE) or analytical models — solves the Grad-Shafranov equation, computes flux surfaces, safety factor q(ψ), and builds field quantity splines.
2. **Ideal MHD stability analysis** via Newcomb's criterion (DCON-style ODE integration) — identifies singular surfaces where q = m/n, computes force-free eigenfunctions ξ(ψ,θ), evaluates the tearing stability parameter Δ' at each rational surface, and checks fixed/free boundary stability.
3. **Perturbed equilibrium** — quantifies the self-consistent plasma response to external magnetic perturbations (RMPs, error fields, correction coils). Key outputs include resonant flux, island half-widths, and the Chirikov overlap parameter — the principal diagnostics for ELM suppression, error field correction, and disruption avoidance.

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
