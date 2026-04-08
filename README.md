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

### Regression Testing


#### Regression Harness: Quick usage guide

Set up an alias for convenience (optional):
```bash
alias regress='julia --project=regression-harness regression-harness/regress.jl'
```

**List available cases:**
```bash
regress --list-cases
```
```
Available regression cases:
----------------------------------------------------------------
  diiid_n1                 DIII-D-like equilibrium, n=1, ideal + perturbed equilibrium
                           dir: examples/DIIID-like_ideal_example  (24 quantities)
  solovev_multi_n          Solovev analytical equilibrium, multi-n, ideal stability
                           dir: examples/Solovev_ideal_example_multi_n  (12 quantities)
  solovev_n1               Solovev analytical equilibrium, n=1, ideal stability
                           dir: examples/Solovev_ideal_example  (18 quantities)
```

**Compare two branches/commits:**
```bash
regress --cases diiid_n1 --refs develop,feature/kinetic-damping
```
```
================================================================
Case: diiid_n1 — DIII-D-like equilibrium, n=1, ideal + perturbed equilibrium
================================================================
[ Info: Cached: diiid_n1 @ 0a905a7d (2026-04-06T23:41:50+09:00)
[ Info: Cached: diiid_n1 @ 44b2494f (2026-04-08T18:30:46+09:00)

Regression Report: diiid_n1
==================================================================================================================
Ref 1: develop  @ 0a905a7d (2026-04-06)
Ref 2: feature/kinetic-damping  @ 44b2494f (2026-04-08)
------------------------------------------------------------------------------------------------------------------
Quantity                     develop                  feature/kinetic-damping  Diff                  Status
------------------------------------------------------------------------------------------------------------------
beta_n                       -1.376214e+00            -1.376214e+00            0.0e+00               OK
beta_t                       1.322850e-02             1.322850e-02             0.0e+00               OK
Chirikov parameter           [4 elements]             [4 elements]             0.0e+00               OK
delta prime                  [4 elements]             [4 elements]             0.0e+00               OK
plasma energy Re(ep[1])      -8.809610e-01            -8.809610e-01            0.0e+00               OK
total energy Im(et[1])       6.175834e-05             6.175834e-05             0.0e+00               OK
total energy Re(et[1])       1.199597e+00             1.199597e+00             0.0e+00               OK
vacuum energy Re(ev[1])      2.080558e+00             2.080558e+00             0.0e+00               OK
island half-widths           [4 elements]             [4 elements]             0.0e+00               OK
mpert                        34                       34                       0.0e+00               OK
# singular surfaces          4                        4                        0.0e+00               OK
npert                        1                        1                        0.0e+00               OK
ODE steps (saved)            740                      740                      0.0e+00               OK
ODE steps (total)            1348                     1348                     0.0e+00               OK
PE plasma energy             0.000000e+00             0.000000e+00             0.0e+00               OK
PE total energy              0.000000e+00             0.000000e+00             0.0e+00               OK
pressure profile (checksum)  657ad2329d7b...          657ad2329d7b...          identical             OK
q0                           1.209710e+00             1.209710e+00             0.0e+00               OK
q95                          4.505007e+00             4.505007e+00             0.0e+00               OK
q profile (checksum)         75912afcc351...          75912afcc351...          identical             OK
||resonant flux||            4.523707e+02             4.523707e+02             0.0e+00               OK
Runtime (s)                  50.9s                    52.0s                                          --
singular psi locations       [4 elements]             [4 elements]             0.0e+00               OK
singular q values            [4 elements]             [4 elements]             0.0e+00               OK
==================================================================================================================
Summary: 23 unchanged, 3 missing/N/A
```

**Compare your uncommitted working tree against develop:**
```bash
regress --cases solovev_n1 --refs develop,local
```

**Track a specific quantity across cached commits:**
```bash
regress --show et_real --case solovev_n1
```
```
History: et_real — solovev_n1
================================================================================
Commit      Date          Value                 Δ from prev           Status
--------------------------------------------------------------------------------
edff6e86    2026-04-02    -4.624928e-01         --                    --
0a905a7d    2026-04-06    -4.624928e-01         0.0e+00               OK
================================================================================
```

**Scan across a range of commits (git-bisect style):**
```bash
regress --cases solovev_n1 --ref-range develop~10..develop
```

**Other useful flags:**
- `--force` — re-run even if cached
- `--verbose` — print GPEC subprocess output
- `--no-instantiate` — skip `Pkg.instantiate()` (faster if deps are already resolved)