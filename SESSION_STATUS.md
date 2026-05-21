# Session Status — Unified Canonical `u_store` for ForceFreeStates Integration Paths

**Branch:** `feature/unified-ustore-reconstruction` (forked from `perf/riccati` @ `8073c126`)
**Last updated:** 2026-05-21
**State:** Phases A–B complete; Phase C partially complete; follow-ups identified.

---

## Goal

`PerturbedEquilibrium` consumes `OdeState.u_store[:,:,1,:]` — the eigenmode radial-displacement
fundamental matrix ξ_ψ in the Gaussian-reduction (GR) axis basis. `ForceFreeStates` has three
integration paths (standard EL, serial Riccati `use_riccati`, parallel-FM `use_parallel`). All
three must produce the **identical canonical `u_store`** so PE works regardless of path.

## What this branch added so far

### Phase A — origin/develop merged (commit `06ac9374`)

`origin/develop` (76 commits ahead of our fork point) was merged in. The merge brings
PR #196's full perturbed-equilibrium pipeline online — `ResponseMatrices`,
`FieldReconstruction`, `SingularCoupling` with Δ′, ||Φ_res||, island widths, Chirikov —
plus IMAS integration, the GGJ inner-layer scaffolding, and develop's `ud` recompute fix
in `cross_ideal_singular_surf!` (commit `ebecc423`). Develop's PE module is now the
canonical consumer of `u_store`.

Conflict resolutions (3 by-hand): `EquilibriumTypes.jl` (kept both `use_galgrid` and
`imas_cocos`), `EulerLagrange.jl` (kept the develop `ud` recompute alongside our note on
why the standard path skips Δ′), `GeneralizedPerturbedEquilibrium.jl` (combined the
analytic-equilibrium embedded-TOML path and develop's IMAS dd path into a single
`additional_input` slot). Post-merge fixes: `DirectRunInput` gained a trailing `bt_sign`
field on develop — wired through `read_imas` and the TJ-analytic direct path. Re-pinned
four Δ′ regression values shifted by develop's `ud` fix; loose tolerances kept (Phase C/D
tightens later).

All four FFS-side tests + IMAS + coils + utilities + vacuum + equilibrium + sing
suites pass: 924 tests passed, 0 failed.

### Phase B — PE-output cross-path benchmark (commit `e2738a84`)

`benchmarks/benchmark_parallel_u_store.jl` now drives the full PerturbedEquilibrium
pipeline (`compute_perturbed_equilibrium`) on each of `:standard`, `:riccati`, `:parallel`
and prints both per-path PE outputs (||Φ_res||, per-surface Δ′, island width, Chirikov)
and cross-path relative differences. Per-surface arrays are written to a TOML snapshot
per example for trend tracking.

A new testset "Three-path PerturbedEquilibrium ||Φ_res|| agreement" in
`test/runtests_parallel_integration.jl`:
- pins inter-path bit-identity between :riccati and :parallel at rtol = 1e-10 (they
  share the FM propagator code path; this *passes* today);
- pins the data-flow contract (PE runs cleanly on all three paths, emits aligned
  per-surface arrays);
- marks :standard-vs-:parallel and :standard-vs-:riccati ||Φ_res|| agreement as
  `@test_broken` (4 broken total) until Phase C completes the unification.

Observed divergence today:
- **Solovev** ||Φ_res||: standard = 7.47e-6, parallel = riccati = 7.81e-5 (~10× ratio).
- **DIIID** ||Φ_res||: standard = 113, parallel = riccati = 9.3e-4 (paths trivially
  different — the chunk-balancing GR-trigger bug, exactly as predicted).

### Phase C — partition-invariant GR firing in standard EL (partial, commit `a9c19948`)

`integrate_el_region!` was rewritten to decouple the GR firing decision from the
integrator's adaptive step grid:

- **`save_callback!` (DiscreteCallback)** handles step counting, the first-sample-after-
  fixup `unorm0` initialization, and the existing save heuristic (near segment boundaries,
  `save_interval`, edge-scan band).
- **`gr_continuous_callback` (ContinuousCallback)** fires GR exactly where
  `max(unorm)/min(unorm) − ucrit` crosses zero. OrdinaryDiffEq bisects the dense
  interpolant to find this root, so the firing ψ depends only on the continuous solution
  u(ψ), not on the adaptive step grid.
- The new affect routine resets `unorm0` to post-GR column norms at the exact firing ψ
  (replacing zero-pivot post-reduction columns by the geometric mean of survivors), so
  the next `gr_condition` eval is anchored at the same ψ regardless of chunking.

Full suite remains green (924 pass + 4 `@test_broken`). Solovev `et[1] ≈ 16.480` and
DIIID `et[1] ≈ −0.18` pins still hold within rtol.

---

## Remaining work

Two follow-ups are needed before the `@test_broken` ||Φ_res|| assertions can flip to
`@test` and Phase D's tolerance tightening makes sense.

### C.2 — Align `reconstruct_u_store_via_gr!` (Riccati.jl:1619)

The parallel/Riccati path still fires GR sample-discretely on the propagator's
`psi_history`. Two reasonable implementations:
- **Bisect between samples**: walk `psi_history` linearly; when `uratio` crosses `ucrit`
  between samples `k−1` and `k`, linearly interpolate `u` to find the approximate
  crossing ψ and fire GR there. Cheapest, but linear interp of `u` over solver
  intervals is inaccurate for exponential growth — would need a denser `subsample`.
- **Store solver dense interpolant**: extend `ChunkPropagator` to keep `sol_upper`/
  `sol_lower` objects (not just discrete samples) and use them to evaluate `u` at
  arbitrary ψ. Accurate, but adds a few MB per chunk × per-thread memory.

### C.3 — Make the save grid partition-invariant

`integrate_el_region!`'s save heuristic uses `odet.total_steps % save_interval`
(cumulative, partition-dependent) and `near_q_frac * q_range` (per-chunk, where `q_range`
differs between natural chunks and balanced sub-chunks). A direct `test_partition_inv.jl`
experiment confirmed that even with the ContinuousCallback fix in place, running the
standard EL with `balance_integration_chunks(chunk_el_integration_bounds(...))` still
gives a different result from running it with the natural chunks (DIIID min wᵗ ≈ −0.99
vs −0.91 — ~8% drift driven by the save-grid divergence).

Fix sketch: thread `psi_natural_start`/`psi_natural_end` through `IntegrationChunk` so
balanced sub-chunks know their parent natural chunk. Inside `integrate_el_region!`,
build a canonical saveat grid relative to the natural-chunk bounds; the sub-chunk's
`saveat` is then the subset of that canonical grid in `[psi_start, psi_end]`. Both
the standard path and the parallel path's `integrate_propagator_chunk!` would share the
same canonical grid.

### Phase D — tighten coverage once C.2 + C.3 land

- Re-pin and tighten the loose `rtol = 0.05` Δ′ assertions in
  `runtests_parallel_integration.jl` (Solovev sing[2], DIIID sing[4], dpm[2,2], dpm[4,4])
  — these are the values that shifted with develop's `ud` recompute fix and should
  shift again once the GR firing is fully unified.
- Flip the 4 `@test_broken` ||Φ_res|| assertions to `@test` with `rtol = 1e-4`
  (Solovev) and `rtol = 1e-3` (DIIID).
- Add a chunk-balancing-sensitivity regression: run the standard EL with both
  `chunk_el_integration_bounds` and `balance_integration_chunks(chunk_el_integration_bounds(...))`,
  assert results agree to machine epsilon.
- Run the regression harness (`regression-harness/regress.jl --cases
  diiid_n1,solovev_n1,solovev_multi_n,ggj_reference --refs 4f614917,local`) and confirm
  no PE quantity drifts beyond its `noise_threshold` against develop's standard-EL
  truth on any of the three integration paths.

---

## How to reproduce / verify

```bash
# Phase A merge + standard-path GR fix in place; full suite green
julia --project=. test/runtests.jl

# Phase B benchmark — shows per-path PE outputs and cross-path divergence
julia --project=. benchmarks/benchmark_parallel_u_store.jl Solovev_ideal_example
julia --project=. benchmarks/benchmark_parallel_u_store.jl DIIID-like_ideal_example

# Local-tree harness baseline (used as Phase D ground truth once C.2 + C.3 land)
julia --project=regression-harness regression-harness/regress.jl \
    --cases diiid_n1,solovev_n1,solovev_multi_n,ggj_reference --refs local --no-instantiate
```

Note: regression-harness against `--refs 4f614917,local` currently fails on the 4f614917
worktree side due to a FastInterpolations `PeriodicBC(:inclusive)` strict-endpoint check;
this is a known precompile drift on the develop snapshot, not a fault in our merge.
Local-tree extraction works.

## Key files

- `src/ForceFreeStates/EulerLagrange.jl` — `integrate_el_region!` (`save_callback!`,
  `gr_condition`, `gr_affect!`, ContinuousCallback bisection), `compute_solution_norms!`,
  `cross_ideal_singular_surf!`, `apply_gaussian_reduction!`, `finalize_canonical_u_store!`.
- `src/ForceFreeStates/Riccati.jl` — `reconstruct_u_store_via_gr!` (the C.2 fix site),
  `gr_right_multiply!`, `parallel_eulerlagrange_integration`, `integrate_propagator_chunk!`,
  `assemble_riccati_s_gauge!`.
- `src/PerturbedEquilibrium/SingularCoupling.jl` — primary consumer of `u_store`;
  computes Δ′, ||Φ_res||, island widths.
- `benchmarks/benchmark_parallel_u_store.jl` — 3-path FFS + PE comparison; emits
  TOML snapshot to `benchmarks/figures/parallel_u_store_pe_<example>.toml`.
- `test/runtests_parallel_integration.jl` — 3-path PE @test_broken testset; the C.2/C.3
  completion gate.
