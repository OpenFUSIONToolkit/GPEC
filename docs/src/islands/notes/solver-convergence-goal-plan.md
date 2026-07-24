# Goal plan — make the physical Level-0 solve converge (resolve the resmax~1e-3 wall)

**Owner:** autonomous goal-mode loop. **Branch:** `feature/islands`.
**Created:** 2026-07-24. Read this + `LOG.md` + `QUESTIONS.md` at the start of every
iteration; update the Progress checklist at the end of every iteration.

## Goal (definition of done — all must hold)

1. A **physical** Level-0 config on a **physically-resolved grid** (`nE ≥ 3`, `ny ≥ 9`)
   — the DIII-D `scenario_from_equilibrium` scenario **and** the hand-set
   `ρ̂_θi=0.05` config — cold-or-globalized-solves to **`resmax ≤ 1e-9`**.
2. The converging solve is captured by a **fast unit/regression test** (so it can't
   silently regress) and the full `test/runtests_islands_*.jl` suite stays green.
3. Any physics-adjacent change (esp. the `y_c` treatment) has a **physics-verifier
   PASS** and its derivation/design doc updated in the same commit.
4. Stretch (not required to close the goal): `Δ_neo` computed on the converged solve
   and shown **resolution-stable** across `K`/`nE`, on a physical bounded box.

**Hard rules (never violate to reach "done"):** never weaken a solver tolerance or
re-baseline a target; never guess a coefficient/sign/normalization (→ `QUESTIONS.md`);
never commit a physics change in `src/Islands/` without a physics-verifier pass.

## What is already known (from LOG cont. 9 + the 2026-07-24 diagnostics)

- The stall is **BC-independent, grid-type-independent, collisionality-independent**.
- `cond(J) ~ 1e5` at the stall (σ_min~2e-4, **not** near-singular); **same** for
  stalling (w=0.05) and converging (w=0.03) configs.
- `PlaneJacobi` reduces `cond(M⁻¹J)` by **≤2.3×** on physical resolved grids (it only
  gives >1000× in the special high-`ρ̂_θi`/`w=0.5` regime).
- **`newton_direct` (exact Jacobian + exact LU) also fails, and worse** — even at nE=2
  where inexact `newton_krylov` converges to 1e-12. ⇒ **not** the linear
  preconditioner; it is the **nonlinear Newton iteration**.
- `cond(J)` grows ~4× per energy node; cold convergence flips off at **nE=3**.
- Leading hypothesis: **`y_c`-layer residual non-smoothness** (the
  `_try_drift_brackets`/`_try_pitch_brackets` "graceful miss" → coeff 0 as a node
  crosses `y_c`) breaks Newton; more `(y,E)` nodes ⇒ more kinks ⇒ worse.

## Environment (do this every run)

- Run Julia as `env -u LD_LIBRARY_PATH julia --project=. …` (the omfit conda env leaks
  SuiteSparse 5 → CHOLMOD init failure otherwise).
- Explicit background jobs get SIGTERM-reaped on turn-yield; run **foreground** (long
  `timeout`) or `nohup … &`. Poll the output file.
- The node is shared — check `free -g` / competing julia before launching; **right-size**
  every experiment (small `nx,ny,nE`, dense LA only at `N ≲ 6000`). Never launch an
  ~80k-unknown solve for a diagnostic.
- Scratch scripts under `/tmp/claude-*/…` (not committed). Durable record = LOG +
  QUESTIONS + tests + this plan.

## Option A — resolution / homotopy continuation (do FIRST; cheapest, no physics change)

Hypothesis: the cold init is outside the Newton basin at `nE≥3`; a warm start from an
easier problem crosses the wall.

- A1. **Same-grid coefficient homotopy** (preferred — no cross-grid interpolation).
  Add a homotopy scalar `λ∈[0,1]` that scales the **energy-coupling** stiffness (the
  `√E`-dependent magnetic-drift / streaming contribution, which the nE sweep showed
  drives `cond(J)` up ~4×/node) from a mild value (`λ=0`, converges cold) to physical
  (`λ=1`). Drive it through `Solvers.natural_continuation`, warm-starting each step —
  exactly the `globalized_level0_solve` pattern but continuing in `λ`, not `ν̂`.
  Implement as an **opt-in solver utility** (numerics only — the endpoints are the same
  physical config; intermediate `λ` are a numerical path, no coefficient invented).
- A2. If a clean scalar hook is awkward, do **grid prolongation**: solve at `nE=2`
  (converges), interpolate the state onto the `nE=3` energy grid (Lagrange in the
  energy nodes), use as the `newton_krylov` init.
- **Gate A**: physical `nE≥3` solve reaches `resmax ≤ 1e-9` → Option A succeeds → add a
  test, verify on both physical configs, update LOG, commit, **goal likely met** (go to
  Closeout). If continuation stalls at the *same* wall for *every* path (init-independent)
  → it is not a basin problem → go to Option B.

## Option B — `y_c`-layer residual smoothness (physics-adjacent → physics-verifier)

Hypothesis: the residual is kinked at the trapped–passing boundary `y_c`.

- B1. **Diagnose smoothness**: build the coefficient tables (drift/pitch brackets, the
  collision operator) and plot each coefficient vs `y` across `y_c` on a refined `y`
  grid; confirm/deny a jump or a `nothing→0` cliff. Also check the residual/Jacobian
  entries for rows whose `y`-node is nearest `y_c`. Quantify: does the stall onset
  correlate with a node landing in the `y_c` "graceful miss" zone?
- B2. If a discontinuity is confirmed, **fix the numerical treatment of the integrable
  `y_c` singularity faithfully** — this is the standard bounce-average near-separatrix
  treatment (cf. Logan 2015 App. D; the `orbit_average_*` brackets). Options: a `C¹`
  blend across `y_c`, or the correct principal-value/endpoint handling of the bracket
  integral instead of the hard `nothing→0`. **This is physics — do NOT guess the
  regularization form.** If the correct treatment is uncertain, write it up in
  `docs/src/islands/derivations/` as `[DERIVED]`, escalate the open choice to
  `QUESTIONS.md`, and get a **physics-verifier PASS** before implementing.
- **Gate B**: with the `y_c` fix, `newton_krylov`/`newton_direct` converge on `nE≥3`
  (`resmax ≤ 1e-9`) → success → physics-verifier PASS, test, doc update, commit → goal
  likely met. Else → Option C.

## Option C — trust-region globalization (numerics only)

Hypothesis: residual is smooth enough but strongly nonlinear; the full Newton step
overshoots and Armijo backtracking stagnates.

- C1. Add a **trust-region** corrector (dogleg, or Levenberg–Marquardt damping
  `(J + μI)δu = −F` with `μ` adapted on the actual/predicted-reduction ratio) as a new
  option in `Solvers.jl` — **do not** replace the existing `newton_krylov`/`newton_direct`.
- C2. Test on the stalling `nE=3,4,6` configs and both physical scenarios.
- **Gate C**: trust-region reaches `resmax ≤ 1e-9` → success → test, commit → goal met.
  Else → the problem is deeper than these three options: record the full evidence in
  LOG + QUESTIONS with the disproven hypotheses, and stop with a crisp hand-off (do not
  thrash further).

## Closeout (when a Gate passes)

1. Add/greenlight the converging-solve **test**; run the full `runtests_islands_*` suite.
2. Verify the fix on **both** physical configs (DIII-D scenario + hand-set), not one.
3. (Stretch) compute `Δ_neo` on the converged solve; check `K`/`nE`/domain stability.
4. Update `LOG.md` (what worked, evidence), close/annotate `QUESTIONS.md` Q7, tick the
   Progress checklist below.
5. `regression-guardian` if any tracked quantity could have moved; push `feature/islands`.

## Per-iteration protocol

Read LOG + QUESTIONS + this plan → pick the current step → run right-sized diagnostic
(env-clean, foreground/nohup) → record result in LOG → (physics change? physics-verifier
first) → commit + push → update Progress → decide next step. Never end an iteration with
uncommitted work or without a LOG line.

## Progress checklist (update every iteration)

- [~] A1 same-grid coefficient homotopy — SKIPPED (A2 showed init-independence; A1 would hit the same λ=1 wall). Revisit only if B+C fail.
- [x] A2 grid-prolongation warm start tried — prolonged nE=3 still fails (1.8e-3), no better than cold ⇒ init-independent
- [x] Gate A decision recorded — Option A ruled out (not a basin problem); advance to B
- [x] B1 `y_c` smoothness diagnosed — CONFIRMED non-smooth: trapped brackets MISS→0 erratically (quadrature bug on the integrable turning-point singularity) + genuine T(y) log-divergence at y_c=1. Evidence in LOG cont. 11.
- [ ] B2a half-angle-substitution bounce quadrature (fixes MISSES; numerics; physics-verifier before commit) — prototyping
- [ ] B2b genuine y_c=1 divergence treatment (only if B2a insufficient; physics-adjacent)
- [ ] Gate B decision recorded
- [ ] C1 trust-region corrector implemented
- [ ] C2 tested on nE=3,4,6 + both physical scenarios
- [ ] Gate C decision recorded
- [ ] Closeout: test added, suite green, both configs, LOG/QUESTIONS/push done
