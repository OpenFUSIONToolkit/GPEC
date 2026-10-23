# GPEC — Generalized Perturbed Equilibrium Code

GPEC is a Julia implementation of the [GPEC suite](https://github.com/PrincetonUniversity/GPEC) for magnetohydrodynamic (MHD) analysis of fusion plasmas. It performs a three-stage analysis pipeline:

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

GPEC is MIT-licensed, except `src/Vacuum/Kernel3D.jl`, which is derived from [BIEST](https://github.com/dmalhotra/BIEST) and licensed under Apache-2.0 (see [`NOTICE`](NOTICE) and [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt)). If you use the 3D vacuum calculation, please cite:

- D. Malhotra, BIEST: Boundary Integral Equation Solver for Taylor states, https://github.com/dmalhotra/BIEST, doi:[10.5281/zenodo.23511944](https://doi.org/10.5281/zenodo.23511944).
- D. Malhotra, A. Cerfon, L.-M. Imbert-Gérard, M. O'Neil, "Taylor states in stellarators: A fast high-order boundary integral solver," J. Comput. Phys. 397 (2019) 108791, doi:[10.1016/j.jcp.2019.06.067](https://doi.org/10.1016/j.jcp.2019.06.067).
- D. Malhotra, A. J. Cerfon, M. O'Neil, E. Toler, "Efficient high-order singular quadrature schemes in magnetic fusion," Plasma Phys. Control. Fusion 62 (2020) 024004, doi:[10.1088/1361-6587/ab57f4](https://doi.org/10.1088/1361-6587/ab57f4).

See the [citations page](https://openfusiontoolkit.github.io/GPEC/dev/citations/) for the references behind the other modules.

## Contributing

The Git workflow, commit message conventions, and the regression-testing harness are documented in the [Developer Notes](https://openfusiontoolkit.github.io/GPEC/dev/developer_notes/).
