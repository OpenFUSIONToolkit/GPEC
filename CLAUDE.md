# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

GPEC (Generalized Perturbed Equilibrium Code, Julia implementation) is a comprehensive Julia reimplementation of the GPEC suite for MHD analysis of fusion plasmas. The code performs equilibrium reconstruction, ideal MHD stability analysis, and perturbed equilibrium calculations including plasma response and singular surface coupling diagnostics.

**Relationship to Fortran GPEC**: This Julia GPEC is an evolution of the Fortran GPEC code suite, available at https://github.com/PrincetonUniversity/GPEC. When users reference "the Fortran code", "the original GPEC", "Fortran GPEC", or "legacy VACUUM", they are referring to that Fortran codebase — not a runtime dependency of this Julia implementation. This Julia implementation reimplements and extends GPEC's functionality with improved performance and maintainability.

**Local GPEC Repository**: For code conversion or comparison with the original Fortran implementation, check for a local GPEC repository at `~/Code/gpec`. If not found at this location, ask the user for the correct path.

## Key References

**IMPORTANT**: The papers in `docs/resources/` provide the theoretical foundation for GPEC's algorithms. **Citing equations from these papers in code comments and annotations is strongly encouraged** to maintain traceability between theory and implementation. The full per-module citation list (Vacuum, ForceFreeStates, PerturbedEquilibrium, KineticForces/NTV, resistive MHD future work) is in **[`docs/development/references.md`](docs/development/references.md)** — read it before touching a physics kernel.

## Common Commands

### Building and Testing

```bash
# Run all tests
julia --project=. -e 'using Pkg; Pkg.activate("."); Pkg.instantiate(); include("test/runtests.jl")'

# Run specific test file
julia --project=. test/runtests.jl test/runtests_solovev.jl

# Available test files:

# - test/runtests_vacuum_julia.jl   # Julia vacuum module
# - test/runtests_solovev.jl        # Analytical equilibrium
# - test/runtests_ode.jl            # ODE integration
# - test/runtests_sing.jl           # Singular surface handling
# - test/runtests_fullruns.jl       # End-to-end tests
```

### Building Documentation

```bash
# Build documentation locally
julia --project=. build_docs_local.jl

# Documentation hosted at: https://openfusiontoolkit.github.io/GPEC/dev/
```

### Development with Revise

For faster recompilation during development, use Revise.jl (installed in global environment, not in Project.toml):

```julia
using Revise
using GeneralizedPerturbedEquilibrium
```

### Benchmarking

Use the generic benchmarking tool at `benchmarks/benchmark_git_branches.jl` to compare performance between branches or commits (default case: `examples/DIIID-like_ideal_example`). Full usage, flags, and benchmark-script conventions (never duplicate example inputs into `benchmarks/`; outputs go under `benchmarks/`, not committed) are in **[`docs/development/benchmarking.md`](docs/development/benchmarking.md)**.

### Regression Harness

***This should be used at least once every single pull request before merging into develop. This test harness is what tracks values as they evolve across changes to the code, and must be both kept up to date and used consistently. Do not forget this and make sure to suggest any new regression cases or updates to existing ones as needed. Remind the user of its existence and report back the output regression report you get when modifying the code significantly. This is extremely important, do not forget this tidbit.***

```bash
alias regress='julia --project=regression-harness regression-harness/regress.jl'
regress --list-cases
regress --cases diiid_n1 --refs develop,local   # working tree vs develop
```

Full command reference (comparing branches/commits, tracking a quantity's history, git-bisect-style scans, sample report output) is in **[`docs/development/regression-harness.md`](docs/development/regression-harness.md)**.

## Architecture

GPEC follows a three-stage pipeline — **Equilibrium** (Grad-Shafranov solve, q-profile) → **Stability Analysis** (ideal MHD eigenvalue problem, DCON-style, singular surfaces) → **Perturbed Equilibrium** (plasma response, singular coupling, island formation) — implemented across seven modules in `src/`: `Splines`, `Utilities`, `Equilibrium`, `Vacuum`, `ForceFreeStates`, `ForcingTerms`, `PerturbedEquilibrium`. All modules are configured via the single `gpec.toml` file (see `examples/*/gpec.toml`).

Full module-by-module breakdown (key files, status, data flow stage-by-stage, key data structures, module dependency graph) is in **[`docs/development/architecture.md`](docs/development/architecture.md)** — read it when working across module boundaries or getting oriented in an unfamiliar part of the codebase.

## Git Workflow

This project uses GitFlow (http://nvie.com/posts/a-successful-git-branching-model): two permanent branches, `main` and `develop`. **IMPORTANT**: All development must be done on feature branches (`feature/`, `bugfix/`, `hotfix/`, `performance/`, `refactor/`, `docs/`, `test/`, `experiment/`). No commits directly to `develop` or `main`. Branch from `develop`, open a PR back into `develop`.

Commit messages follow `CODE - TAG - Detailed message` (e.g. `VAC - IMPROVEMENT - Add dual Green's function computation`).

Full branch-naming table, hotfix workflow, versioning scheme, and merge-conflict resolution policy are in **[`docs/development/git-workflow.md`](docs/development/git-workflow.md)**.

## Agent Team

This repo ships a small team of specialized Claude Code subagents in `.claude/agents/`. They are **stateless reviewers** — each runs in its own context on a specific deliverable and reports back; **the main session is the integrator**, not a delegator. Invoke an agent by name for the matching job (e.g. *"review this with the fortran-physics-reviewer"*); don't reflexively consult all of them.

Roster, the recommended review pipeline (physics fidelity → readability → performance → regression), and per-consultation budget rules are in **[`.claude/agents/README.md`](.claude/agents/README.md)**.

## Important Notes

### General
- **Julia version**: 1.11 is the target version
- **Never remove packages from Project.toml** - If a package fails to load or resolve, run `Pkg.add(...)` or `Pkg.instantiate()` to fix the local environment. Do NOT remove the package from `Project.toml`. The developer works across multiple branches and machines, so environment drift is expected — the right fix is always to update the environment to satisfy the toml, not to trim the toml to match the current environment state.
- **Indexing**: The codebase uses 0-based indexing in many places to match Fortran conventions, then converts to 1-based Julia indexing
- **No step numbering in code comments** - Avoid annotations like "Step 1: do this" followed by "Step 2: do that". These get out of sync as code changes. Just describe the action without numbering.
- **Documentation coverage** - When adding a new module or submodule with public docstrings, add a corresponding `@autodocs` block in `docs/src/`. Documenter CI will fail with a `missing_docs` error if any exported docstring is not covered. The analysis submodule docs live in `docs/src/analysis.md`.
- **Docstrings are rendered as Markdown** - Documenter parses docstrings as CommonMark, so `[text](...)` patterns become hyperlinks and will fail CI with `invalid local link/image` if the target doesn't exist. Common pitfall: unit annotations like `[degrees] (or [m] if ...)` parse as `[degrees](or [m] if ...)` — a broken link. Use plain words (`in degrees`) or backticks (`` `[m]` ``) for unit labels inside docstrings. Same rule for bracketed array-shape hints (`[ncoil]`) followed by parentheses. When in doubt, preview with `julia --project=docs docs/make.jl` before pushing.
- **Keep code comments concise** - A comment should be one line where possible. Do not write multi-line block comments explaining the current session's investigation, what was tried, what was wrong before, or why a specific file/path behaves differently. State what the code does and why at a general level. Example of too much detail: a 6-line block explaining that efit_by_inversion uses psilow>0 while CHEASE starts at 0, that the old code was removed, and that spline spikes result. Preferred: `# Replicate Fortran inverse.f: overwrite deta at axis (r²=0) by extrapolating from innermost surfaces.`
- **No PR/issue references in source** - Do not cite specific GitHub PR or issue numbers (e.g. `#171`, `#202`) in source or test comments/docstrings. During dynamic development these references quickly go stale and pollute the code. Just concisely state the objective or what the code does; the git history and PR threads track the provenance.
- **Document struct fields in the docstring, not inline** - When adding a field to a struct, put its description in the struct's `## Fields` docstring rather than (or in addition to) a trailing inline comment, and document any non-obvious contract (e.g. which keys a `Dict`/ingest carries per variant). Do not annotate outer constructors with positional-argument counts ("passes 11 positional args") — those rot when a field is added; describe the intent instead.

### Minimal-change discipline
- **Reuse native ops and existing utilities before writing new ones.** FastInterpolations splines integrate and differentiate natively (`integrate`, `cumulative_integrate`, `deriv1`); the Equilibrium module already has flux-surface integration/average patterns. Do not reimplement spline integration, quadrature, or differentiation — grep for the existing idiom first.
- **Size the change to the problem.** A small numerical correction (e.g. a ~1% fix) should be a handful of lines, not new general-purpose machinery. Resist faithfully porting Fortran scaffolding (custom integrators, power-law spline bases) when a native call plus a one-line correction gives the same numbers — verify equivalence instead of assuming the elaborate version is needed.
- **Don't commit throwaway artifacts for minor fixes.** No in-repo benchmark scripts/outputs or agent-memory churn for a small change — these accumulate and outsize `src`. Verify with a scratch script (e.g. under `/tmp`) and the regression harness; the regression harness is the durable record of numerical behavior.

### Output Files
- **Default output**: `gpec.h5` (previously `euler.h5` in older versions)

### Plotting
Spectrum-plot (`seriestype=:steppre` / `step_series`) and figure-saving conventions are in **[`docs/development/plotting.md`](docs/development/plotting.md)** — read before writing any plotting code.

### Subagent Consultations

When delegating to specialized agents (julia-performance-optimizer, fast-interpolations-optimizer, fortran-physics-reviewer, clean-code-reviewer, etc.), every prompt **must include an explicit budget** because runaway agents silently consume the user's daily token quota. A single consultation that explores instead of editing has been measured at 167 tool calls / 55 minutes / no return — that is a session-killer. Defaults:

- **Hard cap: ≤ 30 tool uses and ≤ 10 minutes wall time per consultation.** State both numbers in the prompt verbatim ("Budget: ≤30 tool uses, ≤10 min").
- **Single concrete deliverable.** One file, one function, or one named hotspot list. Not "audit the module."
- **No exploration phase.** The prompt must hand the agent the file paths and line numbers; the agent's job is to edit, not to map the codebase.
- **Require an interim status if the work might exceed budget.** Tell the agent: "If you cannot finish within budget, stop and report what was changed and what remains."
- **Prefer two short focused agents over one open-ended one.** If an investigation needs both performance and interpolation review, run them sequentially with separate ≤30-tool budgets — don't chain them in one long prompt.
- **Never re-launch a runaway agent.** If an agent hits the API rate limit before returning, do not retry; report the partial state to the user and switch to hand-implementation.

These rules apply to **every** Agent tool invocation, not just performance work.

### Code Formatting

Pre-commit hooks enforce formatting via JuliaFormatter (v1.0.62) and general file hygiene. **All code you write or modify must already conform to these standards before committing**, so the hooks have nothing to fix. Failing to do this creates noisy diffs in PRs where formatting changes leak into unrelated files.

The project's `.JuliaFormatter.toml` settings:
- **Line width**: 180 characters max (`margin = 180`)
- **`for` loops**: always use `in` (not `=` or `∈`)
- **Keyword arguments**: no spaces around `=` in kwargs (`f(x; a=1)` not `f(x; a = 1)`)
- **Keyword separator**: use semicolons to separate kwargs (`f(x; a=1, b=2)`)
- **No trailing commas** in argument lists
- **Docstrings**: formatted according to JuliaFormatter rules
- **No extra blank line removal**: `remove_extra_newlines = false`
- **Join short lines**: `join_lines_based_on_source = true` — don't arbitrarily split lines that fit within the margin

Additional file hygiene (enforced by pre-commit hooks):
- No trailing whitespace on any line
- Files must end with exactly one newline
- LF line endings only (no CRLF)

### TOML Annotation Conventions

Config-style TOML files (`examples/*/gpec.toml`, `examples/*/sol.toml`, `test/test_data/*` fixtures, `regression-harness/cases/*.toml`) follow one shared annotation style (header comment block, inline `# description` on every variable line sourced from the matching config struct's docstring, no Fortran references, no deprecated variables). **Do not invent a new convention** — read **[`docs/development/toml-conventions.md`](docs/development/toml-conventions.md)** in full before adding or editing one of these files.

### Performance
- Pure Julia implementations are available for all major components and offer comparable or better performance than Fortran
- Benchmarks available in `benchmark/` directory for Fourier transforms and vacuum calculations
- Pre-commit hooks are configured for notebook cleaning and Julia formatting (see `docs/src/set_up.md` for developer setup)
