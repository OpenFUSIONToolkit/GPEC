# Developer Notes

## Git Workflow

This project uses [GitFlow](http://nvie.com/posts/a-successful-git-branching-model):

- Two permanent branches: `main` and `develop`
- `main` is updated only at release-ready stages
- Feature branches should come off `develop` and merge back with `--no-ff`

## Commit Message Format

```
Area[.Submodule] - TAG[!] - Imperative summary
```

`Area` is a module name such as `Equilibrium` or `ForceFreeStates`, or a repository area such as `Docs` or `CI`. `TAG` is one of `FEATURE`, `BUGFIX`, `PERF`, `API`, `DEPRECATION`, `DOCS` (which appear in release notes) or `MINOR`, `REFACTOR`, `TEST` (which do not). Append `!` when results move or an interface changes, for example `Equilibrium - BUGFIX! - Correct bp0 edge quadrature`.

Both vocabularies are closed — do not invent new ones. Pull request titles use the same grammar and are checked in CI; commit subjects follow it by convention rather than enforcement. Pull request bodies carry a release-note block, which is what the changelog is compiled from. The full grammar, the abbreviations to use in prose, and the release-note block are documented in
[`docs/development/naming.md`](https://github.com/OpenFUSIONToolkit/GPEC/blob/develop/docs/development/naming.md).

## Documentation standard

Module reference pages follow a shared *Journal of Computational Physics*-style
structure (governing equations → numerical method → validation figures → API),
and every documentation figure is committed together with the script that made
it and a provenance stamp. The full policy — page template, figure
organization under `docs/src/figures/<module>/`, provenance, and the
regenerate-only-when-`depends`-change rule — is in
[`docs/DOC_STANDARD.md`](https://github.com/OpenFUSIONToolkit/GPEC/blob/develop/docs/DOC_STANDARD.md).
The `Local Stability` and `Tearing` pages are the reference exemplars.

## Deck and API parity

Every capability must be reachable both from a `gpec.toml` deck (through `main`) and from the
scripting API (`solve` and the problem and result types). A feature that exists on only one path
is not finished. A new feature, deck key, or API keyword also adds a case to the `"deck and API parity"`
testset in `test/runtests_solve_api.jl`. That case runs the workflow once from a deck and once
from the same inputs through the API, and checks that the two results agree. A known drift
between the paths is marked `@test_broken`, not dropped, so the change that fixes it shows up
as an unexpected pass.

## Regression Testing

The regression harness **must be run on every pull request before merging into `develop`**. It is the project's primary safeguard for tracking how numerical results evolve across changes, so it is only useful if every PR exercises it. When you open a PR, paste the regression report into the PR thread so reviewers can see what moved (and what did not). If your change touches a quantity that is not yet tracked, add a new regression case — or extend an existing one — in the same PR.

### Pinned Δ′ values

The Δ′ values pinned in the tests and tracked by the harness are single-grid snapshots, not converged values: the extraction is grid-sensitive, and on the DIII-D-like deck the q = 2 diagonal converges only on a fixed `core_edge_packed` grid, while the automatic grid leaves it moving by several percent between refinements.
Treat a pinned diagonal as a sign and order-of-magnitude regression signal, expect to re-pin it when the equilibrium grid changes, and see `docs/development/regression-harness.md` for the convergence scans behind this.

### Regression Harness: Quick usage guide

Set up an alias for convenience (optional):
```bash
alias regress='julia --project=regression-harness regression-harness/regress.jl'
```

**List available cases:**
```bash
regress --list-cases
```
**Compare two branches/commits:**
```bash
regress --cases diiid_n1 --refs develop,feature/kinetic-damping
```
The report lists every tracked quantity of the case for each ref with its difference and an `OK`/changed status.

**Compare your uncommitted working tree against develop:**
```bash
regress --cases solovev_n1 --refs develop,local
```

**Track a specific quantity across cached commits:**
```bash
regress --show et_real --case solovev_n1
```
**Scan across a range of commits (git-bisect style):**
```bash
regress --cases solovev_n1 --ref-range develop~10..develop
```

**Other useful flags:**
- `--force` — re-run even if cached
- `--verbose` — print GPEC subprocess output
- `--no-instantiate` — skip `Pkg.instantiate()` (faster if deps are already resolved)

The full flag list, the environment-pinning behaviour that keeps source code the only variable in
a comparison, and the exit-status contract are documented in
[`docs/development/regression-harness.md`](https://github.com/OpenFUSIONToolkit/GPEC/blob/develop/docs/development/regression-harness.md).

## Developer tips

### Precompile workload

Building the package runs a small Solovev case through the solver so that a fresh Julia process
skips most first-call compilation, at the cost of a longer one-time build. Code compiled into the
package image can round differently in the last bit from code compiled on first use, and
near-marginal quantities such as Δ′ amplify that, so results are reproducible only within one
build mode. The regression harness runs with the workload disabled. To disable it yourself (for a
faster build, or to match harness numbers), set the preference by UUID from the project, without
loading GPEC (which would first build it with the workload on). `Preferences` must be installed,
e.g. in your global environment:

```julia
using Preferences, UUIDs
set_preferences!(UUID("462872dd-e066-4d2e-b993-6468b5239634"), "precompile_workload" => false; force=true)
```

This writes `LocalPreferences.toml` next to the active `Project.toml` (the file is gitignored);
delete the entry to re-enable. The regression harness overrides this file for its own runs.

The workload runs only when Julia rebuilds the package image, not at the start of each run. That
happens after any change under `src/` (a pull, a branch switch or a local edit) and after a change
of dependency versions, Julia version or this preference. Repeated runs on an unchanged checkout
reuse the cached image. Julia keeps a separate image per set of compile flags, so the first
`Pkg.test()` after such a change (which checks array bounds) runs the workload once more. If you edit `src/` and restart Julia often, disabling the workload keeps
each rebuild short; within one Revise session nothing is rebuilt.

### Revise.jl

When iterating on code, use [Revise.jl](https://timholy.github.io/Revise.jl/stable/) to avoid full recompilation on every change. It tracks source file modifications and recompiles only the affected code.

Install Revise in your **global** Julia environment (not in this project's `Project.toml`):

```julia
# In the Julia REPL (no --project flag)
using Pkg
Pkg.add("Revise")
```

Then load it before `GeneralizedPerturbedEquilibrium` in any session:

```julia
using Revise
using GeneralizedPerturbedEquilibrium
```

For it to load automatically, add `using Revise` to your [Julia startup file](https://timholy.github.io/Revise.jl/stable/config/#Using-Revise-by-default):

```julia
# ~/.julia/config/startup.jl
try
    using Revise
catch e
    @warn "Could not load Revise" exception=e
end
```

### Pre-commit hooks

The repository uses pre-commit hooks to maintain code quality. They run automatically on `git commit`, and
the same hooks run in CI on every pull request (the `pre-commit` check), which fails if they would change a file.
They:
- Format Julia code according to `.JuliaFormatter.toml` settings
- Remove trailing whitespace and fix line endings
- Validate YAML/TOML syntax
- Prevent accidentally committing large files (>5MB)
- Check config-style TOML annotation conventions (header block, inline `# description`
  comments, no deprecated keys — see `docs/development/toml-conventions.md`)

**Installation:**

```bash
# Install pre-commit (requires Python/pip)
pip install pre-commit

# Install JuliaFormatter 1.0.62 in the default environment: the hook runs whatever version is there,
# and CI pins this one, so another version formats differently and fails the check
julia -e 'using Pkg; Pkg.add(PackageSpec(name="JuliaFormatter", version="1.0.62"))'

# Install the git hooks in your local repository
cd /path/to/GPEC
pre-commit install
```

You can also run the hooks manually without committing:

```bash
pre-commit run --all-files   # run on all files
pre-commit run               # run on staged files only
```

In rare cases where you need to bypass the hooks:

```bash
git commit --no-verify
```

This is discouraged as it may introduce formatting inconsistencies.
