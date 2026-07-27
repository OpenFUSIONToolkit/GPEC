# Islands benchmark ladder (docs/05)

Benchmark scripts for the Islands verification ladder
(`docs/src/islands/design/05-verification.md`). Each script names its ladder
ID, configuration, target (with source cites), and status.

**Status policy (the `[VERIFY]` discipline, `src/Islands/CLAUDE.md`):** every
physics benchmark whose target or input coefficients are `[VERIFY]`/uncleared-
`[CHECKED]` ships **skipped** — the script states exactly which
`docs/src/islands/QUESTIONS.md` entries gate it and exits without running.
Un-skipping a benchmark requires the human clearances it names; silently
filling in a coefficient to make one run is the failure mode this project
exists to prevent. The structural A-ladder (A1–A8) runs in CI via
`test/runtests_islands_*.jl`, not here.

**Targets are tiered (Decision D9, docs/05 "Target tiers").** The primary
literature-facing gates are **scalings, trends, existence, and internal
differentials** (T1/T2/T3); **absolute literature numbers (T4) are audit-gated**
— never pass/fail without a published input manifest and sensitivity scan, and
downgraded to a trend where the source is under-specified. Each script's header
labels its targets by tier.

Figure scripts (docs/07 pipeline) live in `figures/` and read archived
benchmark data only.

| Script | Ladder ID | Primary tier(s) | Status |
|---|---|---|---|
| `benchmark_B2_large_w_limits.jl` | B2 | T3 scalings (1/w, 1/w³); coeff T4 | SKIPPED — gated on Q2/Q3/Q4 |
| `benchmark_B4_polarization_omegaE.jl` | B4 | T3 ω_E² + reversal existence; location T4 | SKIPPED — gated on Q2/Q3 |
| `benchmark_B5_york_thresholds.jl` | B5a/b/c | T2 toggle ratio + T3 existence/trend; absolutes T4 | SKIPPED — gated on Q2/Q3/Q4 |
