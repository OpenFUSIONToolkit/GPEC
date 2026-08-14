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
| `runtests_h5_schema.jl` | 28/28 pass — confirms `Info/Runtimes/*` carries required metadata |
| `runtests_fullruns.jl` | 19/19 pass (17 before; +2 new runtime assertions) |
| Manual DIII-D run | all five recorded values match their `completed in` log lines exactly; `units`/`long_name` present; only stages that ran appear |
| Benchmark dry-check | real read, missing-group fallback, missing-file fallback, shared-key averaging, `total`-last ordering, silent skip on empty/disjoint sides — all correct |
| Regression harness, `diiid_n1`, `0ece9c4f` vs working tree | every physical quantity agrees to 1e-9–1e-16 (0.00%); see caveat below |
| Package load after merge | clean |

### Regression-harness caveat — read before rerunning

The first harness run was done against `origin/refactor/hdf5-metadata` and reported **38 changed
quantities**. That was a wrong-baseline artifact: at the time this branch's parent (`0ece9c4f`) was
~60 commits behind that ref, so the report measured *upstream* physics changes (auto psi-grid Δ′
convergence fix, on-demand solution derivatives, coil `psilim` fix) — not this work. Rerun against
the branch's actual merge-base, not the remote branch tip.

Against the correct baseline, 14 quantities were still flagged `** CHANGED **` but every one at
**0.00%** (1e-9 to 1e-16 — floating-point noise; the harness flags any last-bit difference, and the
profile "checksum" quantities flag on any bit change at all). Two residuals were being investigated
when the session ended:

- `ODE steps (total)` 1960 → 1968 (0.41%); `ODE steps (saved)` identical at 1327.
- Three profile checksums (Mercier `D_I`, resistive interchange `D_R`, ballooning Δ′) differ.

The diff cannot affect numerics — it computes `time()` differences and appends to the h5 *after* all
computation — so the working hypothesis is run-to-run nondeterminism (threaded BLAS reassociation)
and/or environment drift between the harness worktree and the local environment. A determinism check
was running at handoff time: two `--force` runs of the *same* commit `0ece9c4f`, diffed against each
other. **Its result was never seen.** Re-run it to close this out:

```bash
julia --project=regression-harness regression-harness/regress.jl --cases diiid_n1 --refs 0ece9c4f --force > a.log 2>&1
julia --project=regression-harness regression-harness/regress.jl --cases diiid_n1 --refs 0ece9c4f --force > b.log 2>&1
diff a.log b.log   # differences here ⇒ the case is nondeterministic ⇒ the residuals above are noise
```

## Outstanding work

1. **Re-run the full test suite on the merged tree.** Only the package load and (in flight at
   handoff) `runtests_h5_schema.jl` were checked after the merge. `runtests_fullruns.jl` (~23 min)
   has not been re-run post-merge:
   ```bash
   julia --project=. test/runtests.jl runtests_h5_schema.jl runtests_fullruns.jl
   ```
2. **Close out the determinism check** above, and post the corrected regression table on the PR.
3. **Optional:** a real two-branch benchmark run to see the per-stage table print end-to-end. Only a
   dry-check against synthetic and real h5 files was done, by explicit choice — the full run costs
   ~20–40 min of compute.
4. **Remove this file** (below) and get human review.

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
