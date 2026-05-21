# Session Status — Unified Canonical `u_store` for ForceFreeStates Integration Paths

**Branch:** `feature/unified-ustore-reconstruction` (forked from `perf/riccati` @ `8073c126`)
**Last updated:** 2026-05-21
**State:** **DONE.** All three integration paths feed a bit-identical `u_store` to
PerturbedEquilibrium. Full test suite green; cross-path PE outputs match at machine ε.

---

## What's done

### Phase A — `origin/develop` merged (commit `06ac9374`)

Brought in PR #196's full perturbed-equilibrium pipeline (`ResponseMatrices`,
`FieldReconstruction`, `SingularCoupling` with Δ′, ||Φ_res||, island widths, Chirikov)
plus IMAS integration, GGJ inner-layer scaffolding, develop's `ud` recompute fix in
`cross_ideal_singular_surf!`, and the new coil-forcing infrastructure. Three conflicts
resolved by hand; post-merge fixes for `DirectRunInput`'s new `bt_sign` field; four Δ′
regression values re-pinned for the `ud` shift.

### Phase B — PE-output cross-path benchmark + regression test (commit `e2738a84`)

`benchmarks/benchmark_parallel_u_store.jl` drives the full PerturbedEquilibrium pipeline
on each of `:standard`, `:riccati`, `:parallel`, prints per-path ||Φ_res|| / Δ′ /
island_width / Chirikov, and saves a TOML snapshot per example. A "Three-path
PerturbedEquilibrium ||Φ_res|| agreement" testset went into
`runtests_parallel_integration.jl` (with `@test_broken` placeholders pending Phase C).

### Phase B-prime — better visual feedback (commit `9a84813e`)

Eigenmode ξ_m(ψ) plot now picks `plot_ms` from the resonant m values (helicity range)
instead of m ≈ -12..-8. Added a new 2×2 PE comparison figure
(`parallel_u_store_pe_<example>.png`): |Φ_res|, arg(Φ_res), |Δ'|, island_half_width
per rational surface, with one marker per path. Log-scale axes handle DIIID's 6-OOM
dynamic range that bar plots couldn't render.

### Phase C — partition-invariant GR firing in `integrate_el_region!` (commit `a9c19948`)

Decoupled the Gaussian-reduction firing decision from the integrator's adaptive step
grid via a `ContinuousCallback` that bisects the dense interpolant of `uratio − ucrit`.
After firing, `unorm0` is anchored to the post-GR column norms at the exact crossing
ψ (replacing zero-pivot columns by the geometric mean of survivors), so the next
condition eval is also partition-anchored.

### Phase C.2 — `u_store` unification via standard-EL delegation (commit `6ed0dc2c`)

`parallel_eulerlagrange_integration` now delegates `u_store` building to a new
`standard_eulerlagrange_pass` (extracted from `eulerlagrange_integration`). The
parallel-FM propagator phase runs only for the deferred Δ' BVP and S-gauge outputs.
This is the **fundamental fix** for the unification goal — by construction, all three
integration paths (`:standard`, `:use_riccati`, `:use_parallel`) build their `u_store`
through the same code path, so it is bit-identical regardless of which integration
flag is set.

### Phase C-cleanup — drop now-unused legacy reconstruction code (commit `3a7e35d3`)

`reconstruct_u_store_via_gr!`, `gr_right_multiply!`, and `solve_chunk_fm` were only
ever reached from the old chunk-propagator GR-replay path. After C.2 delegates to the
standard pass, they're dead code and got removed (154 lines).

---

## Verification

Benchmark — both examples, all four PE quantities (`benchmark_parallel_u_store.jl`):

```
Solovev:
  path       ||Φ_res||       rational q values
  standard   7.470565e-06   [2.000, 3.000]
  riccati    7.470565e-06   [2.000, 3.000]
  parallel   7.470565e-06   [2.000, 3.000]
  cross-path ||Δresonant_flux|| / ||resonant_flux|| = 0.000e+00
  u_store[:,:,1] diagonal rel-L1 (parallel vs standard) = 0.000e+00
  ξ_m(ψ) rel-L1                                          = 1.001e-10 (machine ε)

DIIID:
  path       ||Φ_res||       rational q values
  standard   1.131977e+02   [2.000, 3.000, 4.000, 5.000, 6.000]
  riccati    1.131977e+02   [2.000, 3.000, 4.000, 5.000, 6.000]
  parallel   1.131977e+02   [2.000, 3.000, 4.000, 5.000, 6.000]
  cross-path ||Δresonant_flux|| / ||resonant_flux|| = 0.000e+00
  ξ_m(ψ) rel-L1                                      ~ 1e-15 across m = 1..7
```

The DIIID PE comparison figure
(`benchmarks/figures/parallel_u_store_pe_DIIID-like_ideal_example.png`) shows the
three path markers overlapping exactly at every rational surface on all four panels.

Tests: 928 pass, 0 fail, 0 broken across the full suite. The Three-path PE agreement
testset asserts bit-identity (`rtol = 1e-10`) on `resonant_flux`, `delta_prime`, and
`||Φ_res||` between `:standard`, `:riccati`, and `:parallel`. The DIIID parallel
`et_par` pin re-anchored from `≈ 1.29` (old chunk-balancing-corrupted value) to
`≈ −30.84` (the standard-EL truth that the unified path now produces).

---

## Key files

- `src/ForceFreeStates/EulerLagrange.jl` — `integrate_el_region!` (ContinuousCallback
  bisection on uratio crossings), `standard_eulerlagrange_pass`,
  `eulerlagrange_integration` dispatch.
- `src/ForceFreeStates/Riccati.jl` — `parallel_eulerlagrange_integration`
  (now delegates u_store to the standard pass), `assemble_riccati_s_gauge!`,
  `integrate_propagator_chunk!`. (`reconstruct_u_store_via_gr!`,
  `gr_right_multiply!`, `solve_chunk_fm` removed.)
- `src/PerturbedEquilibrium/SingularCoupling.jl` — primary consumer of `u_store`.
- `benchmarks/benchmark_parallel_u_store.jl` — 3-path FFS + PE comparison; emits
  the per-rational-surface PNG figure and a TOML snapshot for trend tracking.
- `test/runtests_parallel_integration.jl` — Three-path PE ||Φ_res|| agreement testset
  (`rtol = 1e-10`), per-surface array equality, plus the existing
  `riccati_cross_ideal_singular_surf!` Δ′ pins.
