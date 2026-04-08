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

## Developer Notes

### Git Workflow

This project uses [GitFlow](http://nvie.com/posts/a-successful-git-branching-model):

- Two permanent branches: `main` and `develop`
- `main` is updated only at release-ready stages
- Feature branches should come off `develop` and merge back with `--no-ff`

### Commit Message Format

```
CODE - TAG - Detailed message
```

Where CODE is the module name (EQUIL, ForceFreeStates, VAC, PERTURBED EQUILIBRIUM, etc.) and TAG describes the type of change (WIP, MINOR, IMPROVEMENT, BUG FIX, NEW FEATURE, REFACTOR, CLEANUP, etc.). This format is used for compiling release notes — tags should be human-readable but are not enforced to a fixed set.
