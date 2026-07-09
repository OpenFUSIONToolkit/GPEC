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

Figure scripts (docs/07 pipeline) live in `figures/` and read archived
benchmark data only.

| Script | Ladder ID | Status |
|---|---|---|
| `benchmark_B2_large_w_limits.jl` | B2 | SKIPPED — gated on Q2/Q3/Q4 |
| `benchmark_B4_polarization_omegaE.jl` | B4 | SKIPPED — gated on Q2/Q3 |
| `benchmark_B5_york_thresholds.jl` | B5a/B5b/B5c | SKIPPED — gated on Q2/Q3/Q4 |
