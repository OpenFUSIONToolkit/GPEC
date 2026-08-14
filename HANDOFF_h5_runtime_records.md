# HANDOFF — `feature/h5-runtime-records`

**Temporary working document. Delete it before this PR merges — see "Remove this file" at the bottom.**

Written 2026-08-14 when the working session moved to another machine. Everything described here is
pushed to `origin/feature/h5-runtime-records`.

## What this branch does

`main_from_inputs` already timed every pipeline stage but only logged the durations. This branch
persists them to `gpec.h5` under `Info/Runtimes/<stage>` (Float64 seconds, annotated with
`long_name` + `units="s"`), and teaches the branch-benchmark tool to report per-stage deltas.

Stages recorded, only when they actually run: `equilibrium`, `galerkin`, `force_free_states`,
`slayer`, `perturbed_equilibrium`, `kinetic_forces`, plus `total`.

**Runtimes are informational only** — machine- and load-dependent. They are not regression
quantities and must never be added to the regression harness.

Target: PR into `refactor/hdf5-metadata` (stack: #363 → #364 → this).

## Commits

| Commit | Contents |
|---|---|
| `df82c9e1` | `GPEC - NEW FEATURE` — `_write_runtimes!` + `_RUNTIME_LONG_NAMES`, six stage captures, both h5-bearing exit paths, doc schema row, two `runtests_fullruns.jl` assertions |
| `e695e890` | `BENCHMARKS - IMPROVEMENT` — `read_stage_runtimes` / `average_stage_runtimes` / `print_stage_comparison` in `benchmarks/benchmark_git_branches.jl` |
| `eb944f9b` | Merge of `origin/refactor/hdf5-metadata` (branch was ~60 commits behind) |

### Design points worth knowing

- The banner at both exit paths was rewritten to reuse `total_dt`, so the log line and the recorded
  value cannot disagree.
- `_write_runtimes!` guards on `isfile` (no-op when `write_outputs_to_HDF5=false`) and deletes an
  existing `Info/Runtimes` group before writing, so reruns into an existing file stay idempotent.
- Annotation goes through the existing `Utilities.HDF5Annotations.annotate!` — no new annotation
  path was written. `test/runtests_h5_schema.jl` enforces this automatically: its metadata walker
  fails if any `Info/Runtimes/*` dataset lacks `long_name`/`units`.
- The equilibrium-only early exit is deliberately untouched — no h5 exists at that point.
- In the benchmark tool the `Info/Runtimes` read happens **inside** the warm-run loop, because each
  run overwrites `gpec.h5`. Refs predating the feature yield an empty dict and the per-stage table
  is skipped silently, so old-vs-new comparisons still work.

### Merge resolution note

Upstream `refactor/hdf5-metadata` removed the shared `H5_*` path consts from
`src/GeneralizedPerturbedEquilibrium.jl` in favour of inline literal paths with cross-reference
comments. The merge conflicted there; it was resolved **upstream's way** — the `H5_RUNTIMES` const
was dropped and `_write_runtimes!` now uses the `"Info/Runtimes"` literal directly. The second
conflict was `galerkin_solve`'s changed signature (`wv=` replaced `vac_data=`); upstream's call was
kept with the timing capture layered on top.

## Verification already done

All of the following ran **before** the upstream merge unless noted:

| Check | Result |
|---|---|
| `runtests_h5_schema.jl` | pass, **re-run after the merge** (14/14 + 6/6 group-name rule) |
| `runtests_fullruns.jl` | 19/19 pass, **re-run after the merge** (17 before; +2 new runtime assertions) |
| Manual DIII-D run | all five recorded values match their `completed in` log lines exactly; `units`/`long_name` present; only stages that ran appear |
| Benchmark dry-check | real read, missing-group fallback, missing-file fallback, shared-key averaging, `total`-last ordering, silent skip on empty/disjoint sides — all correct |
| Regression harness, `diiid_n1`, `0ece9c4f` vs `e695e890` | **48 unchanged, 0 changed** — zero movement |
| Package load after merge | clean |

### Regression-harness caveat — read before rerunning

The first harness run was done against `origin/refactor/hdf5-metadata` and reported **38 changed
quantities**. That was a wrong-baseline artifact: at the time this branch's parent (`0ece9c4f`) was
~60 commits behind that ref, so the report measured *upstream* physics changes (auto psi-grid Δ′
convergence fix, on-demand solution derivatives, coil `psilim` fix) — not this work. Rerun against
the branch's actual merge-base, not the remote branch tip.

There is a second, subtler trap: **comparing a ref against `local` compares environments as well as
code.** Non-`local` refs run in a freshly instantiated harness worktree; `local` runs the working
tree with its own resolved environment. Run `0ece9c4f` vs `local` and 14 quantities come back
flagged `** CHANGED **` — all at 0.00% (1e-9 to 1e-16), plus `ODE steps (total)` 1960 → 1968 and
three profile checksums. That is environment drift, not code.

This was chased to ground and is now **closed**:

- *Is the case nondeterministic?* No. Two `--force` runs of the same commit `0ece9c4f` produced
  byte-identical values for all 49 quantities; only the wall-clock line differed.
- *Do these commits move any number?* No. Running the parent and the feature tip through the **same**
  worktree path — `--refs 0ece9c4f,e695e890` — gives **48 unchanged, 0 changed**.

So: compare worktree-to-worktree (two commit refs), not commit-vs-`local`, whenever the numbers
need to be trusted at last-bit precision.

This is a known class of artifact — see `docs/development/regression-harness.md`, "Making source
code the only variable": an unpinned `Manifest.toml` lets a worktree resolve different package
versions, and the adaptive ODE step controller amplifies machine-epsilon library differences into
apparent regressions. The harness merged in from upstream now pins the working tree's Manifest into
every worktree; the misleading run above was made with the pre-merge harness, which predates that.

## Outstanding work

1. **Optional:** a real two-branch benchmark run to see the per-stage table print end-to-end. Only a
   dry-check against synthetic and real h5 files was done, by explicit choice — the full run costs
   ~20–40 min of compute.
2. **Remove this file** (below) and get human review.

Everything else is done: both test files pass on the merged tree and the regression comparison is
clean.

## Environment note

If a `julia` invocation hits manifest errors on the new machine:
`julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'`.
**Never** remove a package from `Project.toml` — the developer works across several machines and
environment drift is expected; fix the environment, not the manifest.

## Remove this file

This document is scaffolding for a machine switch and must not land in `refactor/hdf5-metadata`:

```bash
git rm HANDOFF_h5_runtime_records.md
git commit -m "DOCS - CLEANUP - Remove the machine-switch handoff document"
git push
```

Do this once the outstanding work above is finished and before the PR is approved for merge.

---

# ⚠️ **MERGE GATE** ⚠️

# **NO PULL REQUEST IS EVER MERGED INTO `develop` WITHOUT A THIRD-PARTY HUMAN REVIEWER'S APPROVAL.**

# **THIS IS NON-NEGOTIABLE. NO EXCEPTIONS.**
