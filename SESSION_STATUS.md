# Session Status — Parallel FFS Thread Scaling

**Branch:** `feature/unified-ustore-reconstruction`
**Last updated:** 2026-05-23
**State:** Phase 1 done & pushed; Phase 2 blocked on a multi-day refactor.

---

## Pushed (Phase 1, commit `c6c3ef15`)

Tactical fixes to `parallel_eulerlagrange_integration`:

1. **Concurrent Phase A + Phase B** — `Threads.@spawn` the serial
   `standard_eulerlagrange_pass` while the threaded chunk-FM loop runs.
2. **BLAS pin** around the threaded chunk loop (restored in `finally`).
3. **Manual worker pool** with `(julia_nthreads − 1)` `@spawn`'d tasks sharing an
   atomic work counter — replaces `@threads :static` (which saturates all workers
   and starves the concurrent Phase A) and `:dynamic` (same issue). The `−1`
   reserves one worker for Phase A.
4. **Honest `parallel_threads` docstring** — it never actually capped at the
   field's value; `@threads` always used all `Threads.nthreads()`.

### Measured (DIII-D-like, `julia -t N`, 5 BenchmarkTools samples)

| n | path | pre-Phase 1 @ 4 thr | **Phase 1** @ 4 thr | vs standard EL |
|---|---|---|---|---|
| 1 | parallel | 7.73 s | **6.34 s** | 14% slower (5.55 s) |
| 4 | parallel | 68.9 s | **53.3 s** | 29% slower (41.3 s) |

Speedup saturates at 2 threads — beyond that, the serial Phase A is the wall
(predicted exactly by Amdahl). Memory unchanged; parallel still allocates ~2×
standard. Figure: `benchmarks/figures/ffs_thread_scaling_DIIID.png`.

### Pre-existing test failures (unchanged by Phase 1)

`test/runtests_parallel_integration.jl` has **6 pre-existing failures** on the
branch tip prior to Phase 1 (verified by stashing the Phase 1 changes and
re-running). Three-path PE agreement (22/22) is unaffected. The 6 failures are
in:
- "Parallel FM integration matches standard ODE" — 3 fails on regression pins.
- "delta_prime_matrix STRIDE BVP" — 3 fails on regression pins.

These are independent of the thread-scaling work and should be triaged
separately.

---

## Phase 2 attempt — **not pushed** (working tree at this status doc's commit
reflects the Phase 2 *helpers* but the parallel branch is reverted to Phase 1)

The goal was to break the serial Phase A wall by reconstructing `u_store`
directly from the parallel-FM chunk propagators. Resurrected (from
`git show 3a7e35d3^:src/ForceFreeStates/Riccati.jl`):

- `solve_chunk_fm` — solves `Φ_chunk · x = rhs` for the chunk-entry-state mapping.
- `gr_right_multiply!` — O(1) `cbase → cbase · G` after a GR fixup.
- `reconstruct_u_store_via_gr!` — chunk-walking u_store builder.
- `_uratio_of`, `_fire_gr_at!` — Phase 2.2 helpers for partition-invariant GR
  firing via dense bisection on saved chunk history.

These are all left in `src/ForceFreeStates/Riccati.jl` as **inert
infrastructure** for the next session — they compile, the docstring on
`reconstruct_u_store_via_gr!` says clearly "WIP / not used in production", but
nothing calls them in production code.

### What was tried and what blocked it

| Attempt | Result |
|---|---|
| Pure FM reconstruction over balanced sub-chunks | **Solovev**: factor-4 errors in PE outputs. **DIIID**: 4× → 2–35% per element. Root cause: backward-chunk `solve_chunk_fm` is ill-conditioned (decayed FM at `psi_start`) → small-solution info lost. |
| + Phase 2.2 dense-bisection GR firing | Improved DIIID partition-invariance but Solovev unchanged (no GR firings to bisect). |
| Hybrid: FM forward chunks + `integrate_el_region!` backward chunks | Solovev element 1 fixed (4× → 1%), element 2 still ~3× off. Forward FM chunks still lose small solutions to roundoff when `uratio` exceeds `ucrit` inside the chunk (the chunk FM has no intra-chunk GR fixups). |
| `integrate_el_region!` on ALL **balanced** sub-chunks | Solovev still ~3× off, DIIID 5-30% per element. Root cause: the GR `ContinuousCallback`'s `unorm0` anchoring resets at sub-chunk boundaries, drifting from the natural-chunk firing pattern. |
| `integrate_el_region!` on **natural** chunks (debug experiment) | Three-path 22/22 ✅. But this is functionally equivalent to running `standard_eulerlagrange_pass`, and sequentially (after Phase B), so it's strictly worse than Phase 1's concurrent `@spawn`. No production value. |

### The structural blocker

For a real Phase 2 win (parallel path < standard EL), we need per-chunk
parallel standard-EL with precomputed `cbase` for every chunk start. That
needs:

1. **Intra-chunk GR fixups in `integrate_propagator_chunk!`** — so the chunk FM
   preserves small-solution info, and a recorded list of intra-chunk GR
   matrices is available for the reconstruction to compose into `cbase`.
2. **Joint 2N-column ODE integration** (instead of two separate N-column
   solves for upper/lower ICs) so GR fires consistently across all 2N columns.
3. **BVP path adapted** — `apply_propagator!` and `compute_delta_prime_matrix!`
   currently assume raw (no-GR) chunk propagators. Post-GR propagators are
   in a different basis; the BVP assembly needs to be checked / adjusted.
4. **S-gauge pass adapted** — `assemble_riccati_s_gauge!` similarly.

This is a multi-day refactor that touches the working BVP path. Doing it
incrementally with WIP commits is sensible; doing it in a single session is
not realistic.

---

## To resume on another computer

```bash
git fetch origin
git checkout feature/unified-ustore-reconstruction
git pull
```

Phase 1 commit is on `origin`. Working tree at the head of this branch holds
the inert Phase 2 helpers and this status doc.

### Recommended next-session plan

1. **Triage the 6 pre-existing test failures first** — they may be hiding a
   real issue, and Phase 2 work will be hard to validate while regression pins
   are already broken.
2. **Design step**: decide between joint 2N-column integration (cleanest) vs
   two-N-column integration with synchronised GR firings (less invasive). Both
   need a clear plan for the BVP / S-gauge adaptation before any code is
   written.
3. **Implementation in two PRs**:
   - PR-A: modify `integrate_propagator_chunk!` to do GR-augmented chunk FMs,
     adapt `apply_propagator!`, `compute_delta_prime_matrix!`, and
     `assemble_riccati_s_gauge!`. Tests for BVP path stay green.
   - PR-B: wire `reconstruct_u_store_via_gr!` in `parallel_eulerlagrange_integration`,
     run per-chunk `integrate_el_region!` in parallel using precomputed cbases.
     Re-measure thread scaling.

---

## Key files

- `src/ForceFreeStates/Riccati.jl` —
  `parallel_eulerlagrange_integration` (Phase 1 concurrent A+B wiring),
  `integrate_propagator_chunk!` (raw FM today, target of next session's PR-A),
  `reconstruct_u_store_via_gr!` (WIP / inert), `solve_chunk_fm`,
  `gr_right_multiply!`, `_uratio_of`, `_fire_gr_at!`.
- `src/ForceFreeStates/EulerLagrange.jl` —
  `standard_eulerlagrange_pass`, `integrate_el_region!` (the
  `ContinuousCallback` GR pattern next session's per-chunk parallel pass would
  reuse).
- `src/ForceFreeStates/ForceFreeStatesStructs.jl` — `parallel_threads`
  docstring updated to match reality.
- `benchmarks/benchmark_ffs_thread_scaling.jl` — full thread-scaling sweep
  driver; `benchmarks/figures/ffs_thread_scaling_DIIID.png` shows Phase 1.
- `test/runtests_parallel_integration.jl` — 22/22 Three-path PE agreement
  enforced at `rtol = 1e-10`; this is the test that determined the Phase 2
  attempts above were wrong (factor-3+ errors, not tolerance-loosenable).
