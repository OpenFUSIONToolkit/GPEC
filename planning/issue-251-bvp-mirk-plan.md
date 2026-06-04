# Issue #251 — BVP/MIRK reformulation of the ForceFreeStates Euler–Lagrange integration

> **STATUS (2026-06-04, end of first overnight pass).**
> - Phase 0 ✅ — branch, `BoundaryValueDiffEqMIRK` dep, native-complex MIRK6 smoke test (3/3), draft PR #256.
> - Phase 1 ✅ — per-column native-complex MIRK6 path implemented end-to-end (`BvpEulerLagrange.jl`),
>   `use_bvp` opt-in (default false), reuses `sing_der!`, materializes `OdeState` for `free_run!`.
> - Stage-1 (no-crossing Solovev) ✅ **VALIDATED**: `et[1]` matches the parallel-FM reference to **7.7e-5**
>   (ep 1.2e-3, ev 6.9e-3). Formulation + materialization + downstream integration are correct.
> - Stage-2 crossing implemented (`_bvp_cross_singular_surf!`, faithful to `cross_ideal_singular_surf!`)
>   but **not validated**: blocked by conditioning (below).
> - **Blocker (the prototype's empirical answer):** the unscaled fundamental-matrix collocation is
>   ill-conditioned (the FM growth that shooting's Gaussian reduction / Riccati renormalization control).
>   The Newton solve hits a conditioning-limited residual floor — `Success` at loose tol on small/low-growth
>   domains, `Stalled` on full segments. Dense per-column Jacobian is also `O((2N·n)³)` slow. So a naive
>   linear-FM MIRK collocation is **not production-viable**; parallel-FM stays default.
> - **Recommended next step:** collocate the bounded Riccati variable `S=U₂U₁⁻¹` (or axis-regular
>   `W=U₁U₂⁻¹`, `W(axis)=0`) → well-conditioned; recover `u_store` by back-substitution. Add an
>   `nlsolve`-level iteration cap (fail-fast) and a banded/sparse collocation linear solve.
> - Default path regression-checked unchanged (`use_bvp=false`).



**Branch:** `feature/bvp-mirk-eulerlagrange`, off `origin/perf/riccati` (PR #178; safe to assume it merges to `develop` before this lands).
**Owner of execution:** autonomous Claude Code + sub-agents, overnight, no user interrupts.
**Driving doc for the whole run.** Re-read this after any context compaction.

---

## 1. Goal, in priority order (this is the "definition of done" ladder)

The single most important deliverable is a **faster, more robust way to produce `u_store`** (the
fundamental-matrix eigenmode basis) via collocation, validated by two independent physics checks.

| Lvl | Target | Pass signal (automated) | Ship? |
|----|--------|--------------------------|-------|
| **P0** | Stage‑1 Solovev: FM via MIRK reproduces `et[1]`,`ep[1]`,`ev[1]` | `regress --cases solovev_n1 --refs perf/riccati,local` → `et_real/ep_real/ev_real` = OK; same on `solovev_multi_n` | yes |
| **P1** (primary) | Stage‑2 DIIID: same `u_store` interface reproduces **`et[1]` AND `‖resonant flux‖`** | `regress --cases diiid_n1 --refs perf/riccati,local` → `et_real`, `||resonant flux||` = OK (plus as many of the 32 tracked qty as possible) | yes |
| **P2** (win) | Walltime ≤ standard shooting path; `u_store` ψ-grid no larger at equal `et[1]` accuracy (ideally smaller → less disk) | benchmark CSV+plot in `benchmarks/` | yes |
| **P3** (stretch) | `Δ'` extracted from the BVP matches the Riccati `delta_prime_matrix` | `regress` `delta prime` = OK | optional |

**Key physics rationale to keep in mind:** `wp = U₂U₁⁻¹` and all downstream resonant-flux
projections are **invariant under right-multiplication of the fundamental matrix by any invertible
matrix**. So we do NOT need to reproduce the legacy basis vectors — only a correct *spanning of the
regular-at-axis solution subspace*. `et[1]` is a necessary check; `‖resonant flux‖` is the
independent confirmation that the subspace (hence `u_store`) is right. Matching both = success.

Δ' is explicitly **optional**: if it stays on the Riccati path, that is acceptable per the issue,
*provided* this work improves the old shooting-path walltime/robustness for `u_store`.

---

## 2. The three make-or-break unknowns — front-load them in Phase 0/Stage 1

1. **Complex support (DEFAULT = native complex).** BoundaryValueDiffEq sets
   `allowscomplex = true` for *all* solvers and auto-selects `AutoFiniteDiff` + a complex-safe
   `NewtonRaphson`/`GaussNewton` the moment `u0::ComplexF64` is passed. **So the default is to keep
   everything `ComplexF64` end-to-end** (operator, IC, `bc!`, initial guess) and solve with native
   complex `MIRK6` — no real/imag splitting. Likely gotcha: pass
   `jac_alg = BVPJacobianAlgorithm(; bc_diffmode=AutoFiniteDiff(), nonbc_diffmode=AutoSparse(AutoFiniteDiff()))`
   (and, belt-and-suspenders, `nlsolve=NewtonRaphson(; autodiff=AutoFiniteDiff())`). Native complex
   is wired for collocation but only regression-*tested* for Shooting — hence the Phase‑0 smoke test
   exercises it on MIRK6 first. **Real/imag stacking `[[ReL,−ImL];[ImL,ReL]]` is a LAST-RESORT
   fallback**, used only if native complex MIRK6 demonstrably fails the smoke test and the jac_alg/
   nlsolve overrides don't fix it (then try `Shooting(Vern7())`/`MultipleShooting` complex — the
   CI-proven complex path — before any splitting).
2. **Conditioning / mode growth.** Legacy shooting needs Gaussian reduction / renormalization
   because `U₁` grows exponentially. A global collocation solve of the raw FM may or may not stay
   well-conditioned. Validate on Solovev (Stage 1); if accuracy fails, escalate through the
   contingency ladder (Riccati-variable collocation → per-segment renormalization).
3. **Downstream basis-covariance.** Assumed: `et` and resonant flux match for *any* correct
   subspace basis. The Stage‑2 `‖resonant flux‖` check is exactly this test. If `et` matches but
   flux does not, it is a grid-density / derivative issue at crossings (see C2), not a covariance
   failure.

The plan is sequenced so these are answered on the cheapest possible case first.

---

## 3. Architecture decisions (pre-decided defaults + fallback)

- **What we solve:** the full fundamental matrix `Y(ψ) ∈ ℂ^{2N×N}`, `Y' = L(ψ)Y`, axis IC
  `Y(ψ_low) = [0; I]`. This is the object `u_store` stores; it is what downstream consumes. We are
  NOT posing an eigenvalue BVP for the single least-stable mode (that would not yield the full
  `u_store` resonant flux needs).
- **Decomposition (DEFAULT = per-column):** solve N independent column BVPs, each state dim
  `4N`≈136 (real-stacked `2N` complex). Cheap Jacobian blocks, embarrassingly parallel, columns
  share `L(ψ)` and mesh. State is **`ComplexF64`** (native), so per-column dim is `2N`≈68 complex,
  not `4N` real. *Fallback:* single vectorized matrix BVP (dim `2N²` complex) only if per-column
  proves inadequate — flagged costly, use as last resort.
- **Solver/formulation (DEFAULT = native-complex `MIRK6` on the first-order `Y'=L(ψ)Y`).**
  *Alternative kept open:* recast the EL equation as a second-order system in the displacement
  (`u'' = f(u,u',ψ)`, `SecondOrderBVProblem` + `DynamicalBVPFunction`) and solve with **`MIRKN6`** —
  this halves the state and exploits the natural second-order EL structure. MIRKN is fixed-step (no
  defect adaptivity) and also native-complex-but-untested, so it is a Stage‑5 experiment to try if
  MIRK6 accuracy/cost disappoints, not the first attempt.
  - Note: with all BCs at the axis each column is an IVP-posed-as-BVP; MIRK must accept all-BC-at-
    one-end. **Confirm this in the Phase‑0 smoke test** (try an all-left-BC linear problem).
- **Crossings (DEFAULT = Option B, sequential per-segment collocation + explicit crossing):**
  split `[ψ_low, ψ_high]` at each ψ_s; solve each segment by MIRK collocation; between segments
  apply the **existing** asymptotic-basis crossing (`compute_sing_asymptotics` / `sing_get_ua` /
  `sing_get_ca` / the inject-small-solution step from `riccati_cross_ideal_singular_surf!`) to set
  the next segment's IC. This reuses validated crossing code, isolates risk, and still replaces
  shooting-within-segments with robust collocation → the `u_store` win.
  - *Stretch (Option A):* one global multipoint BVP with interface residuals coupling segments
    (BoundaryValueDiffEq has no native jump API → coupled augmented system). Only attempt for the
    Δ' angle (P3) and only if Option B is green and time remains.
- **u_store adapter:** evaluate the MIRK interpolant on a chosen output ψ-grid and materialize
  `u_store`,`ud_store`(=`L·u`),`psi_store`,`q_store` in the exact `(N,N,2,nsteps)` layout, including
  the small `±spot_psi` bracket points near each ψ_s that `SingularCoupling` interpolates. No
  downstream code changes required if the adapter matches the existing contract.
- **Integration path flag (DEFAULT false):** add `use_bvp::Bool=false` to `ForceFreeStatesControl`,
  wired into the `eulerlagrange_integration` dispatcher as a third opt-in path. Keeps parallel‑FM
  the production default (the issue's fallback design); nothing existing regresses.

---

## 4. Phased execution plan

Each phase: **action → automated success check → contingency → commit/PR update.** Every phase is
time-boxed; on exhaustion, commit WIP, record status in the PR, and proceed. Partial credit is
acceptable — the deliverable is "a full attempt with the ladder climbed as far as it goes."

### Phase 0 — Setup, dependency, GO/NO-GO smoke test (target ≤ 1–1.5 h)
1. `git fetch`; branch `feature/bvp-mirk-eulerlagrange` from `origin/perf/riccati`.
2. Add `BoundaryValueDiffEqMIRK` (≥ 1.17.0) to `Project.toml` `[deps]`+`[compat]`; `Pkg.resolve()`
   + `Pkg.instantiate()`.
   - *Contingency:* if it conflicts with `OrdinaryDiffEq`/SciML pins, try the umbrella
     `BoundaryValueDiffEq` instead, or relax/align compat bounds. **Never remove an existing dep**
     (CLAUDE.md). If unresolvable, pin via Manifest and document; do not block.
3. **Smoke test** (`test/scratch_bvp_probe.jl`, scratch): solve a tiny **`ComplexF64`** linear
   system with (a) all BCs at the left end, (b) one interior-point residual via `u(t_interior)`,
   with native-complex `MIRK6(; dt, abstol, reltol)` and a `ComplexF64` `u0`. Confirms install,
   **native complex** end-to-end, all-left-BC acceptance, and interior-condition API. This is the
   GO/NO-GO for the whole approach.
   - *If native complex errors:* add `jac_alg=BVPJacobianAlgorithm(bc_diffmode=AutoFiniteDiff(),
     nonbc_diffmode=AutoSparse(AutoFiniteDiff()))` and `nlsolve=NewtonRaphson(autodiff=AutoFiniteDiff())`;
     if still failing, try `Shooting(Vern7())` complex (CI-proven), and only then the real/imag
     split as last resort. Record which worked.
   - *If MIRK rejects all-left-BC:* pose columns as genuine two-point (axis regularity + a benign
     edge normalization) or switch to FIRK; record finding.
4. Open **DRAFT PR** `feature/bvp-mirk-eulerlagrange → perf/riccati` (retarget to `develop` once
   #178 merges) titled `ForceFreeStates - NEW FEATURE - BVP/MIRK collocation path for u_store
   (issue #251)`. Body = this plan's checklist + "Phase 0: done" + smoke-test result.

### Phase 1 — Infrastructure: operator + real packing + u_store adapter (target ≤ 2–3 h)
- New file `src/ForceFreeStates/BvpEulerLagrange.jl`.
- `el_operator!`: wrap existing `sing_der!` coefficient evaluation to expose the per-column linear
  action; add a thin real⇄complex packing layer (`pack!/unpack!`). Reuse `sing_der!` directly — do
  not re-derive the physics.
- `f!(du,u,p,t)` real RHS for one column (dim `4N`).
- `build_u_store_from_bvp(sols, ψ_grid, ...)`: materialize `u_store/ud_store/psi_store/q_store` in
  the `(N,N,2,nsteps)` contract; `ud = L·u`.
- Unit tests: packing round-trips; adapter shape; `ud` matches finite-difference of `u`; operator
  matches `sing_der!` on a random `u` at a sample ψ (bit-level agreement modulo packing).

### Phase 2 — Stage 1: Solovev, no in-domain crossings (target ≤ 3–4 h; CORE VALIDATION)
- Solve all N columns on `[ψ_low, ψ_high]`; assemble `Y(ψ_high)`; `wp = U₂U₁⁻¹/psio²`; run the
  existing Free.jl extraction → `et/ep/ev`.
- **Success:** `regress --cases solovev_n1 --refs perf/riccati,local` → `et_real,ep_real,ev_real`
  = OK; repeat `solovev_multi_n`.
- **Tolerance scan:** sweep `abstol/reltol ∈ {1e-6…1e-11}`; record `et[1]`, `#mesh points`,
  walltime → `benchmarks/bvp_solovev_tolscan.{csv,png}` (spectrum/scan plotting per CLAUDE.md;
  print absolute output paths).
- **Contingency C1 (accuracy fails):** ladder — (a) tighten tol; (b) reformulate on the bounded
  Riccati variable `S=U₂U₁⁻¹` (nonlinear, watch poles); (c) per-segment renormalization reusing
  `renormalize_riccati*`. Take the first that passes; document. If none reach tolerance, keep BVP
  as opt-in and still deliver the Stage‑1 report — do not block.
- **Milestone:** PR update with Stage‑1 table + convergence plot.

### Phase 3 — Stage 2: DIIID, 4 crossings, multipoint matching (target ≤ 5–7 h; PRIMARY)
- Split domain at each ψ_s (reuse `chunk_el_integration_bounds` offsets, `singfac_min/|n·q'|`).
  Solve each segment by collocation; cross with the existing asymptotic machinery (Option B).
- Build merged `u_store` across segments incl. the `±spot_psi` bracket points near each ψ_s.
- **Success:** `regress --cases diiid_n1 --refs perf/riccati,local` → `et_real` **and**
  `||resonant flux||` = OK; maximize agreement across all 32 tracked qty (`delta prime`,
  `island half-widths`, `Chirikov parameter`, `singular psi/q`, counts).
- **Contingency C2 (flux mismatch, `et` OK):** grid-density/derivative issue near ψ_s. Densify the
  output grid around each surface; supply analytic `ud=L·u`; mirror the SingularCoupling chord-slope
  expectation. Iterate grid density (bounded loop).
- **Contingency C3 (a crossing is brittle):** reuse the exact `ca_l/ca_r` capture; if one surface
  fails, fall back to the legacy crossing for *that surface only* (hybrid) and keep going. Never
  hard-stop.
- **Milestone:** PR update with full Stage‑2 regression report pasted in.

### Phase 4 — Δ' from the BVP (stretch, P3; time-box 1 iteration block)
- If Option B yields `ca_l/ca_r` + segment propagators compatible with `compute_delta_prime_matrix!`
  (PEST3 combination), wire them in and compare `delta_prime_matrix` to Riccati. Otherwise mark
  "Δ' remains on Riccati path" in the PR and move on (acceptable).

### Phase 5 — Optimization loop (agent-driven; time-box, gated on green regression)
- Benchmark BVP vs parallel‑FM vs standard on DIIID & Solovev: walltime, `#mesh points`, `et[1]`
  error → `benchmarks/`.
- Spawn `julia-performance-optimizer` (and `fast-interpolations-optimizer` for the spline-hint
  evaluation): analytic `colorvec`/`jac_prototype`, sparse linear solver, exploit linearity
  (≈1 Newton step), mesh tuning, Threads over columns/segments. **Every change gated by: regression
  still OK + walltime not worse.** Loop until diminishing returns or time budget.
- `fortran-physics-reviewer`: confirm the matching equations faithfully reproduce Glasser
  2016/2018 and the legacy crossing. `clean-code-reviewer` + `/code-review`: fix findings.

### Phase 6 — Tests, docs, finalize (target ≤ 2 h)
- `test/runtests_bvp_mirk.jl`: Stage‑1/Stage‑2 regression values, adapter unit tests, smoke BVP;
  wire into `test/runtests.jl`. Run full suite.
- `use_bvp` flag through the dispatcher (default false). Docstrings + `@autodocs` block in
  `docs/src/`; note third path in CLAUDE.md.
- Full `regress --cases diiid_n1,solovev_n1,solovev_multi_n --refs perf/riccati,local --force`;
  paste report into PR. Suggest a `bvp`-variant regression case.
- JuliaFormatter (v1.0.62, margin 180) + file hygiene clean so pre-commit has nothing to fix.
- Flip PR **draft → ready**; final summary: walltime, accuracy, grid-size, code-delta, Δ' status,
  ladder level reached.

---

## 5. Autonomy guardrails (so it runs through the night unattended)

- **No user prompts.** At every fork take the documented DEFAULT; switch to a contingency only on a
  concrete automated failure (regression CHANGED, test fail, solver error, timeout). Never wait.
- **Time-box + WIP-commit.** Each phase has a wall-clock/iteration cap. On exhaustion: commit WIP,
  write status to the PR, proceed to the next phase. Bound every iterate-until loop with a max count.
- **Self-verify cadence.** Run the relevant `regress` case after each substantive change; commit on
  green with `CODE - TAG - message` + `Co-Authored-By: Claude Opus 4.8 (1M context)` trailer.
- **Logging:** `@info/@debug/@warn` in `src/`, never `println` (memory: logging style).
- **Scope:** consolidate perf + wiring into THIS PR; no "follow-up PR" deferrals (memory:
  pre-merge scope).
- **Deps:** never remove a `Project.toml` entry; fix env via `Pkg.add`/`instantiate` (CLAUDE.md).
- **Branch hygiene:** all work on `feature/bvp-mirk-eulerlagrange`; nothing committed to
  `develop`/`perf/riccati`. If `perf/riccati` advances, merge it in per the merge-conflict policy
  (flag numeric-parameter conflicts for human review rather than guessing).
- **Parallel agents** use worktree isolation to avoid clobbering the working tree.
- **Examples:** benchmark inputs come from `examples/` (copy to temp for modified TOMLs); all
  outputs land under `benchmarks/`; print absolute paths for every figure.

---

## 6. Risk register

| Risk | Trigger | Mitigation |
|------|---------|------------|
| Native complex MIRK misbehaves | Phase‑0 smoke fails | jac_alg/nlsolve → AutoFiniteDiff; then Shooting(Vern7) complex; real/imag split only as last resort |
| FM ill-conditioned (mode growth) | Stage‑1 `et` off | C1 ladder: tol → Riccati var → per-segment renorm |
| Jacobian cost at N≈34 | Stage‑1 walltime ≫ shooting | per-column solves (default), `colorvec`/`jac_prototype`, sparse solve, linearity, Threads |
| Interface matching brittle | a DIIID crossing fails | C3 hybrid: legacy crossing for that surface only |
| Flux mismatch despite `et` OK | Stage‑2 `‖flux‖` CHANGED | C2: densify grid near ψ_s, analytic `ud`, chord-slope parity |
| Dep resolution conflict | `Pkg.resolve` fails | umbrella pkg / align compat / Manifest pin; never trim deps |
| #178 changes under us | merge drift | merge `perf/riccati` in, follow merge policy, re-run regress |

---

## 7. First concrete actions on approval
1. `git fetch && git switch -c feature/bvp-mirk-eulerlagrange origin/perf/riccati`
2. Add `BoundaryValueDiffEqMIRK` to Project.toml; instantiate.
3. Write & run `test/scratch_bvp_probe.jl` (GO/NO-GO smoke).
4. Open the draft PR with this plan as the checklist.
Then proceed Phase 1 → 6, climbing the P0→P3 ladder, committing on green, updating the PR at each
milestone, never pausing for input.
