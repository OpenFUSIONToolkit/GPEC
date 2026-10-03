# Git Workflow

This project uses GitFlow (http://nvie.com/posts/a-successful-git-branching-model):

- Two permanent branches: `main` and `develop`
- `main` is updated only at release-ready stages via pull request from `develop`
- `develop` is the integration branch — all feature branches merge here

**IMPORTANT**: All development must be done on feature branches. No commits should be made directly to `develop` or `main`. Always create a branch from `develop`, do all work there, and open a pull request back into `develop`.

## Branch Naming

Branches use a typed prefix and a lowercase hyphen-separated description:

| Prefix | Purpose | Branches from | Merges into |
|---|---|---|---|
| `feature/` | New functionality | `develop` | `develop` |
| `bugfix/` | Non-critical bug fixes | `develop` | `develop` |
| `hotfix/` | Critical production fix | `main` | `main` + `develop` |
| `performance/` | Performance improvements | `develop` | `develop` |
| `refactor/` | Refactoring without behavior change | `develop` | `develop` |
| `docs/` | Documentation only | `develop` | `develop` |
| `test/` | Test additions/improvements | `develop` | `develop` |
| `experiment/` | Exploratory work, may not merge | `develop` | — |

Examples: `bugfix/sing-lim-bounds-error`, `feature/kinetic-damping`, `performance/green-function-prefactor`

Author-named branches (e.g. `jmh/`, `nlogan/`) are not used — git history already records authorship on every commit.

## Hotfix Workflow

Hotfixes address critical bugs in production (`main`) that cannot wait for the next release cycle:

1. Branch `hotfix/description` from the current tagged `main` commit
2. Fix the bug with one or more commits
3. Merge into `main` via pull request; tag the merge commit with a new patch version (e.g. `v0.1.1`)
4. Merge the same branch into `develop` so the fix is not lost in the next release

## Versioning

This project uses semantic versioning: `v{major}.{minor}.{patch}`

- **major**: breaking API or file-format changes
- **minor**: new features, backward-compatible
- **patch**: bug fixes (typically via hotfix branches)

Tags are applied to merge commits on `main`.

## Commit Message Format

```
Area[.Submodule] - TAG[!] - Imperative summary
```

Examples:

- `PerturbedEquilibrium - FEATURE - Implement singular coupling diagnostics`
- `Vacuum - PERF - Add dual Green's function computation`
- `Equilibrium - BUGFIX! - Fix separatrix finding for high kappa`
- `ForceFreeStates - REFACTOR - Unify singular surface data structure`

The Area and TAG vocabularies are closed — do not invent new ones. Pull request titles use the same grammar and are checked in CI; commit subjects are not, so follow this by hand. Every pull request body also carries a release-note block, which is what the release notes are compiled from. The full grammar, the Area list, and the abbreviations to use in prose are in [`naming.md`](naming.md).

## Merge Conflict Resolution Policy

- When resolving git conflicts, do not simply accept one side.
- Analyze what each side changed and WHY before producing a resolution.
- Produce a merged version incorporating both sets of changes.
- If both sides renamed the same symbol differently, prefer the current (ours) branch convention.
- When a rename on one side conflicts with a logic change on the other, apply the logic change using the renamed symbol.
- If a conflict involves changes to numerical parameters (tolerances, boundary conditions, grid sizes), flag for human review rather than guessing.
- Flag any conflicts where the combination is ambiguous for human review.
