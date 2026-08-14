# Handoff — issue #376 mpsi/ForceFreeStates performance investigation

**Date:** 2026-08-14. **Base:** `performance/regression-harness-threading` @ e7bb7cb8.
**Status:** investigation essentially complete; one discriminator run pending; issue comment not
yet drafted/posted.

## What was established (details + tables in RESULTS.md)

1. FastInterpolations series-spline eval is flat in knot count (~890 ns/eval, 676 series,
   hinted, zero-alloc). Search/hints are not the problem.
2. The EL slowdown at large mpsi is **step-count growth at flat per-step cost**: accepted step
   size stays a constant fraction (~0.12 median) of the *local knot spacing*, so nstep tracks
   knot count where knots pack (dominantly ψ<0.1 on log-family grids; edge secondarily).
   Singular-surface step counts are mpsi-flat.
3. Uniform-grid discriminator (mpsi=512): near-axis steps 1891→559, nstep 3010→2237 —
   confirms knot-density slaving over axis physics.
4. The two-pass auto grid (mpsi=0) reaches converged et[1]≈0.80 with 558 knots and ~half the
   EL time of a fixed 1024 log grid.

## Loose ends for the next session

1. **Tolerance discriminator landed** (see RESULTS.md §5): tol 1e-8 at mpsi=512 → nstep
   1629 vs 3010, identical et[1]. Both mechanisms confirmed: dψ ∝ knotΔ at fixed tol,
   dψ ∝ tol^{1/9} at fixed grid. A tolerance-sweep (1e-6 … 1e-12) checking et[1]/Δ′ stability
   would firm up a recommendation to relax the default `eulerlagrange_tolerance`.
2. **Draft the issue #376 comment** (do not post without the user's review — user chose
   draft-for-review). Content = RESULTS.md conclusion + tables + ranked follow-ups.
3. Optional cheap experiment from the ranked list: swap Vern9 → Vern7/Tsit5 in
   `integrate_el_region!` (`src/ForceFreeStates/EulerLagrange.jl:766`) and in the Riccati
   chunk solves (`src/ForceFreeStates/Riccati.jl:1442,1454`), rerun the ladder point at
   mpsi=1024, compare EL wall time at equal physics (et[1], Δ′). Regression harness required
   for any such change: `regress --cases diiid_n1 --refs develop,local`.

## How to reproduce the runs

```bash
# per mpsi: copy example inputs into a fresh dir, edit gpec.toml (mpsi = N), then
julia --project=. handoff/issue376/run_one.jl <run_dir>   # cold+warm, TSMARK timestamps
julia --project=. handoff/issue376/parse_ladder.jl        # expects mpsi_<N>.log + run_<N>/ in its dir
julia --project=. handoff/issue376/step_regions.jl        # step distribution vs psi
julia --project=. handoff/issue376/step_vs_knot.jl        # dpsi / knot-spacing ratios
julia --project=. handoff/issue376/microbench_mpsi.jl     # standalone, no inputs needed
```

The parse/step scripts look for logs and run dirs in their own directory (`@__DIR__`); either
run them from a scratch dir containing `mpsi_<N>.log` + `run_<N>/`, or adjust `LADDER_DIR`.
Warm-run numbers (RUN2) are the meaningful ones. Note: run GPEC comparisons single-threaded or
all with `-t auto`, consistently — the ladder above was single-threaded.

## Remove this handoff directory when done

This directory (`handoff/issue376/`) is a transfer vehicle, not repo content — per repo policy
benchmark artifacts do not get committed. Once the investigation lands (issue comment posted,
any follow-up implemented on its own branch):

```bash
git rm -r handoff/issue376
git commit -m "FFS - CHORE - Remove issue #376 handoff artifacts"
git push
```

Then close the associated PR **without merging** (it exists only for machine transfer), or if
any part of the PR is to be kept, strip this directory from it first. Never merge the handoff
directory into develop.
