# Naming and Commit Conventions

Commit subjects, pull request titles, and issue titles share one grammar. It exists so that an expert can tell at a glance whether a change touches their niche, and so release notes can be assembled from the history without anyone re-reading every diff.

## Subject line

```
Area[.Submodule] - TAG[!] - Imperative summary
```

Used verbatim for commit subjects, PR titles, and (without the TAG) issue titles. The separator is a space-hyphen-space; the Area is a single token; the TAG is a single word.

## TAG

Two blocks. Decide first whether the change is noteworthy, then what kind it is.

Release-note tags — each names a section of the release notes:

| TAG | Section | Use when |
|---|---|---|
| `FEATURE` | New capabilities | A user can do something they could not before |
| `BUGFIX` | Bug fixes | Something was wrong and now is not |
| `PERF` | Performance | Same answers, less time or memory |
| `API` | Interface & format changes | A config key, output dataset, or exported name changed |
| `DEPRECATION` | Deprecations | Something still works but should not be relied on |
| `DOCS` | Documentation | Documentation only |

Excluded tags — never appear in release notes:

| TAG | Use when |
|---|---|
| `MINOR` | You judge the change not worth reporting: work in progress, trivia, tidying |
| `REFACTOR` | Deliberate restructuring that preserves behavior |
| `TEST` | Tests only |

`MINOR` is an author's judgement that a commit can be skipped, not a category of work. That judgement is the useful signal — keep making it.

## The `!` mark

Append `!` to the TAG when the change is visible to a user: results move, an output dataset changes, or a config key changes meaning.

```
Equilibrium - BUGFIX! - Correct bp0 edge quadrature
PerturbedEquilibrium - API! - Rename Clebsch datasets to xi_clebsch_*
ForceFreeStates - REFACTOR! - Reassociate Riccati sums
```

**`!` promotes a change into the release notes whatever its tag.** Every marked change is listed under `Changed results & breaking changes` at the top of the release, ahead of the per-tag sections. This is why `REFACTOR!` is worth having: a restructuring that genuinely perturbs numbers is exactly what a user needs told, and the honest label should exist.

`MINOR!`, `TEST!`, and `DOCS!` are rejected — none of those can move a result. Reaching for one means the change is really a `BUGFIX`, `API`, or `FEATURE`.

## Area

Modules, with the abbreviation to use in prose:

| Area | Prose | Source |
|---|---|---|
| `Analysis` | — | `src/Analysis/` |
| `Equilibrium` | EQUIL | `src/Equilibrium/` |
| `ForceFreeStates` | FFS | `src/ForceFreeStates/` |
| `ForcingTerms` | FT | `src/ForcingTerms/` |
| `HDF5Schema` | — | `src/HDF5Schema.jl` |
| `InnerLayer` | IL | `src/InnerLayer/` |
| `KineticForces` | KF | `src/KineticForces/` |
| `LocalStability` | LS | `src/LocalStability/` |
| `PerturbedEquilibrium` | PE | `src/PerturbedEquilibrium/` |
| `Rerun` | — | `src/Rerun.jl` |
| `Tearing` | — | `src/Tearing/` |
| `Utilities` | — | `src/Utilities/` |
| `Vacuum` | VAC | `src/Vacuum/` |

`EQUIL`, `VAC`, `FFS`, and `PE` were already in common use; the rest are new. Short module names take no abbreviation.

A submodule may be named where it helps triage: `ForceFreeStates.Galerkin`, `InnerLayer.GGJ`, `InnerLayer.SLAYER`, `Tearing.Dispersion`, `Tearing.Runner`. Only directories holding Julia sources qualify, so data directories are not Areas.

Areas outside `src/`: `Benchmarks`, `Build`, `CI`, `Docs`, `Examples`, `Regression`, `Repo`, `Test`.

A change spanning exactly two areas may name both, `Equilibrium/Vacuum`. Three or more is `Repo`.

## Abbreviations in prose

Titles always carry the full name. In the body of an issue, PR, or comment, expand on first use and abbreviate after:

> The ForceFreeStates (FFS) solver disagreed with PerturbedEquilibrium (PE) at the q=2 surface. FFS now matches PE to 1e-10.

This keeps a title searchable and a thread readable without assuming the reader has this table memorized.

## The release-note block

Every PR body carries one:

```markdown
## Release note
- **Audience:** users | developers
- **Numerical impact:** none | <what moved> _(harness @ <sha>)_
- **Migration:** none | <what a user must change>

<1-3 sentences in user-facing language.>
```

These three fields are the things a reader of the diff cannot work out for themselves, which is why they are asked for rather than inferred.

**Numerical impact** comes from the regression harness, which every PR must run anyway (see [`regression-harness.md`](regression-harness.md)). Record the commit it ran at. Review changes code, and a report from five commits ago may no longer be true — so CI re-checks on every push and fails if anything under `src/` changed after the stamped commit. Commits touching only docs, tests, or examples never trip it.

**Migration** must say something real when the title carries `!`.

## Issue titles

`Area - Short description`, with no TAG — an issue is a request, not a change, and GitHub labels carry its type. Using the same Area vocabulary means one scan covers issues, PRs, and history alike.

## Pull request metadata

Labels are applied by CI from the TAG in the title, so there is nothing to do by hand.

An assignee and at least one human reviewer are the author's job, and a pull request without them should be opened as a draft. CI comments when either is missing but does not block the merge. [`contributors.md`](contributors.md) lists lead developers and their handles as a starting point for who to ask; it is a suggestion list, not the full set of people who can review.

## Enforcement

`ci/conventions/check_subject.py` implements all of the above. What it blocks:

- the `pr-conventions` CI workflow checks the **pull request title**, the release-note block, and the freshness of the harness stamp;
- the `naming-table-in-sync` pre-commit hook checks the Area table above against `src/`.

**Individual commit subjects are not enforced.** They follow the same grammar, and should, but the release notes are compiled from pull requests rather than from `git log` — so a rejected commit mid-flow would cost more than the consistency is worth. Merge commits carry the pull request title, which means `git log --merges` reads correctly whatever happens inside a branch.

Valid Areas are derived from `src/` on disk rather than listed in the script, so adding or removing a module updates the vocabulary on its own. Every rejection prints the replacement to use, including for the spellings this convention replaced (`EQUIL`, `VACUUM`, `IMPROVEMENT`, `CLEANUP`, and the rest).

To check a subject by hand, or to wire it into your own clone as a `commit-msg` hook:

```bash
python3 ci/conventions/check_subject.py --title "Equilibrium - BUGFIX! - Correct bp0 edge quadrature"
```
