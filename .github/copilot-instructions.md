# Copilot instructions for JPEC

## Project overview
- JPEC is a Julia port of GPEC-style MHD equilibrium and stability analysis. Core modules live in [src](src): Utilities, Equilibrium, Vacuum, ForceFreeStates, ForcingTerms, PerturbedEquilibrium (see [CLAUDE.md](CLAUDE.md)).
- Data flow: equilibrium setup → vacuum response → stability analysis (documented in [CLAUDE.md](CLAUDE.md)).

## Architecture and entry points
- Equilibrium: `setup_equilibrium(path|config)`; types in [src/Equilibrium](src/Equilibrium).
- Vacuum: `compute_vacuum_response()`; code in [src/Vacuum](src/Vacuum).
- DCON stability: entry points in [src/DCON/Main.jl](src/DCON/Main.jl).

## Data flow and key structures
- Equilibrium: TOML config → read equilibrium → solve → diagnostics (gse*.h5) when relevant.
- Vacuum: initialize plasma/wall surfaces → compute response matrix → return wv, grri, xzpts.
- Stability: equilibrium + vacuum response → integrate ODEs → compute energies.
- Core types: `PlasmaEquilibrium` and `EquilibriumConfig` in [src/Equilibrium/EquilibriumTypes.jl](src/Equilibrium/EquilibriumTypes.jl); `DconControl` in [src/DCON/DconStructs.jl](src/DCON/DconStructs.jl).

## Tests and docs
- Run all tests:
  ```bash
  julia --project=. test/runtests.jl
  ```
- Run specific tests:
  ```bash
  julia --project=. test/runtests.jl test/runtests_spline.jl
  ```
- Build docs locally:
  ```bash
  julia --project=. build_docs_local.jl
  ```
  (See [docs/README.md](docs/README.md) for the long form.)

## Project conventions
- GitFlow workflow; develop is the active integration branch (see [README.md](README.md)).
- Commit message format: `CODE - TAG - Detailed message` (examples in [CLAUDE.md](CLAUDE.md)).
- Avoid step numbering in comments; instructions should be unnumbered (see [CLAUDE.md](CLAUDE.md)).
- Many routines use 0-based indexing for historical consistency with the original GPEC Fortran code before converting to 1-based Julia indexing (see [CLAUDE.md](CLAUDE.md)).

## Configuration examples
- TOML configs: `equil.toml` uses `[EQUIL_CONTROL]` and `[EQUIL_OUTPUT]`; `dcon.toml` uses `[DCON_CONTROL]` and `[WALL]`.
- Example configs in [examples/DIIID-like_ideal_example](examples/DIIID-like_ideal_example) and [examples/Solovev_ideal_example](examples/Solovev_ideal_example).

## Development tips
- Use Revise.jl in your global environment (not Project.toml); see [CLAUDE.md](CLAUDE.md).
- Target Julia version: 1.11.

## Benchmarks
- Default benchmark case: [examples/DIIID-like_ideal_example](examples/DIIID-like_ideal_example).
- Report metrics: least stable eigenmode energy (first `et`), solver step count, and wall-clock time (details in [CLAUDE.md](CLAUDE.md)).
