# Session Status — Unified Canonical `u_store` for ForceFreeStates Integration Paths

**Branch:** `feature/unified-ustore-reconstruction` (from `perf/riccati` @ `8073c126`)
**Last updated:** 2026-05-21
**State:** WIP checkpoint. Architecture implemented; one root-caused bug remains unfixed.

---

## Goal

`PerturbedEquilibrium` consumes `OdeState.u_store[:,:,1,:]` — the eigenmode radial-displacement
fundamental matrix ξ_ψ in the Gaussian-reduction (GR) axis basis. `ForceFreeStates` has three
integration paths (standard EL, serial Riccati `use_riccati`, parallel-FM `use_parallel`). All
three must produce the **identical canonical `u_store`** so PE works regardless of path.

## What is implemented

- **`finalize_canonical_u_store!`** (`EulerLagrange.jl`) — shared post-integration tail
  (`trim_storage!`, edge-dW scan, `evaluate_stability_criterion!`, `transform_u!`). All three
  paths call it.
- **`reconstruct_u_store_via_gr!`** (`Riccati.jl`) — replays the parallel-FM chunk propagators
  through the GR machinery to build `u_store` (post-multiply: `u(ψ_k) = Φ_chunk(ψ_k)·cbase`).
- **`gr_right_multiply!`** (`Riccati.jl`) — exact O(1) re-anchor `cbase → cbase·G` after a GR
  fixup. Replaced an ill-conditioned 2N×2N chunk-FM solve (which blew `u_store` up 152× on
  large chunks). **This is a genuine fix — keep it.**
- **`parallel_eulerlagrange_integration`** rewired into a GR pass (`u_store`) + an S-gauge pass
  (`Δ'`, `ca_l`/`ca_r`, `S_at_surface_left`) on a scratch `OdeState`.
- **`use_riccati`** routes through `parallel_eulerlagrange_integration` with `parallel_threads=1`.
- **`benchmarks/benchmark_parallel_u_store.jl`** — 3-path comparison + eigenmode ξ_m(ψ) overlay.

## What works / is verified

- **Solovev**: all three paths agree to ~1e-5 on energies; `S(ψ)=U₁U₂⁻¹` to ~2%.
- Serial Riccati and parallel paths produce **bit-identical** results.
- Test suites: parallel 111/111 pass. (riccati / eulerlagrange / fullruns — see "Test status".)

---

## THE REMAINING BUG (root-caused, not fixed)

On **DIIID** (and any high-growth equilibrium) the parallel/riccati paths give a wrong result
— the `wt` eigenvalue spectrum is ~10% off the standard path, `et[1]` off ~7–15%.

### Root cause — DEFINITIVELY localized: `balance_integration_chunks`

Running the **identical standard EL integration loop** on two chunk partitions:

| Chunk partition                                  | DIIID min `wt` eigenvalue |
|--------------------------------------------------|---------------------------|
| Standard (`chunk_el_integration_bounds`)         | **−1.101**                |
| Balanced (`+ balance_integration_chunks`)        | **−0.9917**               |

Same equilibrium, integrator, `cross_ideal` — only the chunk *partition* differs. −0.9917 is
**exactly** the parallel path's result. The chunk **subdivision corrupts the EL integration
itself**; the parallel path inherits it because it builds chunks via `balance_integration_chunks`.

- **Growth-dependent**: corrupts steep DIIID (~1e4 solution growth); negligible on gentle
  Solovev (~1e-3) — which is why Solovev matched 1e-5.
- **Pre-existing**: the parallel path was already ~7% off for DIIID's `et[1]`, masked by a
  hardcoded loose (`rtol=0.05`) test "Parallel FM integration matches standard ODE — large N"
  (`test/runtests_parallel_integration.jl`).

### Mechanism (leading hypothesis)

`balance_integration_chunks` inserts *artificial* chunk boundaries inside smooth regions. Each
boundary forces `integrate_el_region!` to stop/restart the ODE solver, changing the adaptive
step grid. The GR trigger `uratio > ctrl.ucrit` in `compute_solution_norms!` is evaluated **at
ODE steps**, so GR fires at partition-dependent ψ. On a steep problem the solution spreads
enough between GR events that this shifts the result ~10%. The boundary handoff *should* be
transparent — that it isn't is the bug.

### Ruled out (do NOT re-investigate these)

- **NOT** the FM post-multiply small-solution loss — re-integrating *every* chunk still gives
  −0.992 ≠ −1.101.
- **NOT** chunk-FM growth/conditioning — 6× more chunks did not fix it.
- **NOT** the bidirectional crossing-chunk bounds — identical to standard (only the `direction`
  field differs).
- **NOT** the gauge / `transform_u!` — the parallel `u_store` is single-gauge (smooth ξ curves).
- **NOT** `truncate_at_dW_peak` — off for DIIID (diagnostic-only edge scan, restores state).

---

## Next steps — the fix

**Option 1 (recommended, fundamental):** make the GR-firing decision partition-invariant. The
`uratio > ucrit` test in `compute_solution_norms!` must not depend on where ODE-step samples
land — e.g. evaluate it on a fixed ψ-grid, or via the solver's dense interpolant. Fixes the
integration for *any* chunking and de-risks the standard path too.

**Option 2 (contained):** keep the GR / `u_store` reconstruction off balanced chunks.
`balance_integration_chunks` exists only to load-balance the parallel FM-propagator BVP;
`reconstruct_u_store_via_gr!` could walk the un-subdivided `chunk_el_integration_bounds`
partition while the propagators stay balanced for the Δ' BVP.

## How to reproduce / verify

```bash
# Shows the bug: standard −1.101 vs parallel/riccati ≈ −0.9 (DIIID)
julia --project=. benchmarks/benchmark_parallel_u_store.jl DIIID-like_ideal_example
# Solovev — all three paths agree (the bug is growth-dependent)
julia --project=. benchmarks/benchmark_parallel_u_store.jl Solovev_ideal_example
```

**Decisive isolation test:** in `eulerlagrange_integration` (`EulerLagrange.jl`, ~line 197)
temporarily change `chunks = chunk_el_integration_bounds(odet, ctrl, intr)` to
`chunks = balance_integration_chunks(chunk_el_integration_bounds(odet, ctrl, intr), ctrl, intr)`
and run the standard path on DIIID → −0.9917 (= parallel). Revert afterward.

## Key files

- `src/ForceFreeStates/EulerLagrange.jl` — `finalize_canonical_u_store!`,
  `integrate_el_region!`, `chunk_el_integration_bounds`, `balance_integration_chunks`,
  **`compute_solution_norms!` (the GR trigger — the bug site)**, `transform_u!`.
- `src/ForceFreeStates/Riccati.jl` — `reconstruct_u_store_via_gr!`, `gr_right_multiply!`,
  `parallel_eulerlagrange_integration`, `assemble_riccati_s_gauge!`.
- `benchmarks/benchmark_parallel_u_store.jl` — 3-path benchmark + overlay plots.

## Current code state

Clean baseline: `reconstruct_u_store_via_gr!` uses the post-multiply approach for all chunks.
This session's crossing-only / conditioning-based / all-reintegrate experiments were all
reverted (each was based on a hypothesis the diagnostics disproved). The DIIID bug is present
but masked by the loose test tolerance.

Note: the regression harness baseline (`8073c126`) fails to run in a worktree due to a
FastInterpolations precompile issue — could not compare against it this session.

## Test status

All four suites pass on this checkpoint:

- `test/runtests_parallel_integration.jl` — 111/111 pass.
- `test/runtests_riccati.jl`, `test/runtests_eulerlagrange.jl`, `test/runtests_fullruns.jl`
  — pass (exit 0, no failures).

The DIIID bug is **not** caught by the suites — it is masked by the loose `rtol=0.05`
tolerance in the "Parallel FM integration matches standard ODE — large N" test. Tightening
that test (or adding a standard-vs-balanced-chunk regression) is advisable once the bug is
fixed, so it cannot silently regress again.
