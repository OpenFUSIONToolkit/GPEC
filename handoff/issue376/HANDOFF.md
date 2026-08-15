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
2. **Issue comment — posted and kept current.** `ISSUE_COMMENT.md` is the source of truth; edit it
   and PATCH comment 5302684530 rather than appending corrections.
3. **Integrator-order experiment — done** (RESULTS.md §7). Result recorded there; the `src/` edits
   were reverted, so this branch still touches no source. If it is worth pursuing, it lands on its
   own branch with `regress --cases diiid_n1 --refs develop,local`.
4. **Mechanism established** (RESULTS.md §8–§9, new `roughness.jl`): a ~1e-5 relative node-error
   floor in the C/E/H coefficient matrices, amplified as ε/Δ² by spline interpolation. `etol`,
   `mtheta` and the input equilibrium were each tested and ruled out as the source.
5. **Source traced** (§10): the floor is in the per-surface geometry (nu/offset/rcoords), and it
   reaches C/E/H through the psi-derivative channel (g11/g12/g31 use `partials[2]`; the clean
   matrices use only theta-derivatives). 1D profiles cleared. Do **not** re-test
   `etol`/`mtheta`/input resolution/`abstol` — sections 8 and 10 closed all four.
6. **Repairing it is worth much less than expected** (§11): capping the psi-derivative resolution
   cleans C/E/H by 20-100x but recovers only 11% of steps at mpsi=512 and 20% at mpsi=1024.
   The posted issue comment was **reworked in place** to say this (no separate correction comment:
   `gh api -X PATCH repos/OpenFUSIONToolkit/GPEC/issues/comments/5302684530 -F body=@<file>`).
7. **The real open question**: what accounts for the other ~80% of the step growth. Candidate
   worth testing first: how much of the near-axis step count is legitimate resolution of real
   structure (post-repair near-axis |f''|/|f| for K is 2.1e7, a ~2e-4 curvature scale against a
   1.7e-4 median knot spacing) versus still-unexplained grid slaving.

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

# mechanism (RESULTS.md 8-9). Stripped decks run in ~31 s instead of ~220 s: drop the
# [ForcingTerms]/[PerturbedEquilibrium]/[KineticForces] sections and set
# local_stability_flag = false. Everything else stays as the example ships it.
julia --project=. handoff/issue376/roughness.jl <rundir> [<rundir> ...]         # F/K/G
GPEC_ROUGHNESS_MATS=A,D,C,E,H,F,K,G julia --project=. handoff/issue376/roughness.jl <rundir>
```

Read the **mid-plasma (0.3-0.7)** rows for the clean diagnosis: near the axis the r1 statistic is
unreliable on the strongly graded grid (A and D read r1 < -2/3 there despite a 2e-7 residual), and
genuine psi->0 structure is mixed in with the noise. `PREDICTIONS.md` records what each hypothesis
predicted, written before the runs.

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
