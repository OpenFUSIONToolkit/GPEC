# Handoff — issue #376 mpsi/ForceFreeStates performance investigation

**Date:** 2026-08-14, updated 2026-08-15. **Base:** §2–§5 measured on
`performance/regression-harness-threading` @ e7bb7cb8; §6 onward on this branch's tree (post
on-demand-solution-derivatives merge) — see the note at the end of RESULTS.md §6.
**Status:** investigation complete. Tolerance sweep done (RESULTS.md §6). Issue comment drafted in
`ISSUE_COMMENT.md` — **not posted**; awaiting the user's review.

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

1. **Tolerance sweep — done** (RESULTS.md §6). et[1] is invariant across 1e-6…1e-12; Δ′ is the
   discriminator; 1e-8 is the sweet spot. EL wall time is sub-linear in nstep at fixed grid.
2. **Issue comment — drafted, not posted.** `ISSUE_COMMENT.md` holds the full text. Posting needs
   the user's explicit go-ahead (`gh issue comment 376 --body-file handoff/issue376/ISSUE_COMMENT.md`).
3. **Integrator-order experiment — done** (RESULTS.md §7). Result recorded there; the `src/` edits
   were reverted, so this branch still touches no source. If it is worth pursuing, it lands on its
   own branch with `regress --cases diiid_n1 --refs develop,local`.
4. Remaining ranked follow-ups (smoothing splines for F/K/G, equilibrium `etol` noise floor, and
   any per-deck `eulerlagrange_tolerance` change) are unstarted — each needs its own branch and a
   regression-harness run.

## How to reproduce the runs

```bash
# per mpsi: copy example inputs into a fresh dir, edit gpec.toml (mpsi = N), then
julia --project=. handoff/issue376/run_one.jl <run_dir>   # cold+warm, TSMARK timestamps
julia --project=. handoff/issue376/parse_ladder.jl        # expects mpsi_<N>.log + run_<N>/ in its dir
julia --project=. handoff/issue376/step_regions.jl        # step distribution vs psi
julia --project=. handoff/issue376/step_vs_knot.jl        # dpsi / knot-spacing ratios
julia --project=. handoff/issue376/microbench_mpsi.jl     # standalone, no inputs needed

# tolerance sweep: one run dir per tolerance (run_<tol>/ with mpsi=512 and that
# eulerlagrange_tolerance) plus its tol_<tol>.log, then
julia --project=. handoff/issue376/parse_tolsweep.jl <sweep_dir>
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
