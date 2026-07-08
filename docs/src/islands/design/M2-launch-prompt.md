# M2 launch prompt (Islands overnight autonomous run)

> This file is fed verbatim to the overnight loop:
> `claude --permission-mode dontAsk --continue -p "$(cat docs/src/islands/design/M2-launch-prompt.md)"`.
> It is the milestone contract (design doc `06 §2.1`). Keep it stable across
> relaunches so `--continue` resumes the same objective.

You are working milestone **M2** of the Islands module (`src/Islands/`), a
steady-state drift-kinetic island/layer solver, autonomously and unattended. M1
(phase-space grids + operator-stack skeleton + MMS/AD harness) is complete
(PR #320); M2 builds the **Level-0 solve machinery** on top of it.

## Read first (every session)

`src/Islands/CLAUDE.md` (module conventions + the [VERIFY] policy),
`docs/src/islands/LOG.md` and `docs/src/islands/QUESTIONS.md` (session memory +
open blockers), then the design docs
`docs/src/islands/design/{00-roadmap,01-physics-level0,02-species-and-eps,03-architecture,04-numerics,05-verification,06-autonomy-and-tooling,07-documentation-and-papers}.md`.
The repo-root `CLAUDE.md` governs GPEC-wide conventions.

## The gating reality (read this before you plan)

M2's roadmap headline is "L0 single-species solve, Δ moments, **York gates** →
Paper I." **The York gates are NOT reachable in this run** and reaching one is a
policy violation, not completion. Every L0 physics coefficient the gates need
(the `ω̂_D`/`L̂_B` toggle, the collision kernel, `k=−1.173`, `f_p=1−1.46√ε`,
`⟨ν̂_ii⟩_u`, the quasineutrality closure, the York numbers `8.73`/`1.46` ρ_bi) is
at most **`[CHECKED]`** — none is human-cleared — and Decisions **D7** (implement
L0 from an independent re-derivation vs the L23-amended set) and **D8** (pin the
B5a/b/c benchmark triangle) are **unratified** (`docs/00` Decision Log). `[CHECKED]`
is *not* permission to hardcode a number (`docs/01` header; `CLAUDE.md` policy).

So M2 here = **build the full L0 solve *structure* with every physics coefficient
a `[VERIFY]`-gated parameter, close the physics-free structural gates, and escalate
a prioritized clearance queue** for the human. A thin follow-up run fills the
numbers and hits the York gates *after* a human clears the tags.

## Goal (set this as your `/goal` completion condition)

**M2 is done when all of the following hold:**

1. **The L0 solve machinery exists** as AD-compatible, allocation-free structure,
   with **every physics coefficient a supplied `[VERIFY]`-gated parameter (no
   literal in `src/`)** — new submodules per design `03 §1`:
   - `solvers/` — matrix-free **Newton–Krylov** (GMRES on the M1 ForwardDiff JVP;
     inexact-Newton / Eisenstat–Walker forcing; line search; a pseudo-arclength
     continuation scaffold that detects folds from day one; a physics-block
     preconditioner that treats the `y_c` matching block explicitly), `04 §3, §5`.
   - `moments/` — `Δ_cos`/`Δ_sin` assembly: species-charge-weighted velocity moment
     of `g` → `J̄_∥(x, ξ)` → the `∮ J̄_∥ {cos, sin} ξ dξ ∫ dψ` projections (`01 §4`),
     plus the bootstrap/polarization channel decomposition *structure*. The `μ₀R/2ψ̃`
     prefactor and `ψ̃` stay **gated** (open `[VERIFY]`); the projection + quadrature
     is pure numerics.
   - `frames/` — THE `ω`/normalization conversion module (`01 §5`): the conversion
     *forms* with every sign and normalization flagged and gated. This module owns
     all `ω`-sign conventions (the polarization sign disputes are frame disputes —
     do not reproduce them elsewhere). Unit-test only the mechanical, sign-free
     identities.
   - `fields/` — formalize the quasineutrality residual (already stubbed in
     `operators/`); the flattened-electron closure *structure* gated.
   - `species/` — `Species`, `AbstractBackground`, `SpeciesRole {Bulk, Trace}`
     plumbing (`02 §1`, Decision D3) — pure data structures, fully unblocked. The L0
     test is a single bulk ion + a trace-deuterium copy.
   - **neoclassical-matching far-field BCs** (`01 §3`, `04 §1`): structure only; the
     no-island neoclassical far-field solution is gated. **Never bare Neumann** (the
     L23 "winged" spurious-branch failure).

2. **Physics-free structural gates green** (`05 §A`), in `test/runtests_islands_*.jl`,
   passing in the full suite:
   - **A5** zero-drive null: gradients and `Φ̃` off ⇒ `g ≡ 0` exactly, residual =
     machine zero, Newton converges trivially.
   - **Assembled solve-MMS**: a manufactured `g*` with `GradientDrive` set so `g*` is
     the exact Newton solution; the solver recovers `g*` at design order (extends
     M1's residual-MMS to a full converged solve).
   - **A8** `y_c` matching-block smallest-singular-value conditioning monitor
     (regression = the silent-noise failure mode of L23 §4.2).
   - **A4** (L0): particle conservation + discrete entropy sign `∫ g C[g]/F_M ≤ 0`
     of the discretized collision operator (structural — holds for any valid
     discretization, independent of the physical `ν` value).
   - **A3** parity: `Δ_cos` even / `Δ_sin` odd under the appropriate `(ξ, σ, ω_E)`
     reflection, tested on a manufactured `J̄_∥` (coefficient-free).
   - **A7** the coefficient-free identity `⟨∂²h/∂x²⟩_Ω = 0` (defer the number-bearing
     `k`, `f_p`, `⟨ν̂_ii⟩` identities to a post-clearance run).

3. The full suite passes: `julia --project=. test/runtests.jl`.

4. **`docs/src/islands/papers/paper-1/OUTLINE.md`** written as the Paper-I figure
   contract (`07 §3`): claims → figures → ladder IDs (B5a/b/c, B2, B4). This is the
   level-*start* deliverable, not the level-*gate*.

5. **The clearance queue** (the parallel-human deliverable): a consolidated,
   prioritized set of `QUESTIONS.md` entries covering every coefficient the York
   gates need + D7/D8 ratification + the open `[VERIFY]`s, **each paired with a
   skipped B-series benchmark** in `benchmarks/islands/` that references its tag — so
   the human can clear in parallel and the follow-up run fills numbers and hits gates.

6. A PR is open onto `feature/islands` (a sub-branch → `feature/islands`, as M1's
   PR #320 did).

7. **No `[VERIFY]`/uncleared-`[CHECKED]` coefficient has been assigned a value**
   without a `docs/src/islands/QUESTIONS.md` entry.

**Explicitly NOT in the M2 DoD (gated on human clearance — do not attempt):** the
York gates B5a/b/c, B2, B4; B1 (needs an external NEO/NCLASS run); any physics
number. A "green York gate" reached by hardcoding is the exact failure this project
exists to prevent.

## Scope

- **Reuse existing GPEC machinery; do not reimplement** (repo-root CLAUDE.md
  minimal-change discipline): `FastInterpolations.integrate` /
  `cumulative_integrate` for the ψ/flux integrals (`src/Equilibrium/Equilibrium.jl`,
  `src/ForceFreeStates/Resist.jl`); the `src/KineticForces/BounceAveraging.jl`
  velocity-space λ-averaging pattern for the `(y, E, σ)` moments; the allocation-free
  `QuadGK.quadgk!` pattern (`src/ForceFreeStates/Galerkin/GalerkinAssembly.jl`) for
  hot quadrature; the `EquilibriumConfig` / `build_inputs_from_toml` TOML-config
  pattern (`src/Equilibrium/EquilibriumTypes.jl`,
  `src/GeneralizedPerturbedEquilibrium.jl`) for an `[Islands]` `gpec.toml` section in
  a new `io/`.
- **Add `Krylov.jl`** to `Project.toml` `[deps]` + `[compat]` (matrix-free GMRES;
  `04 §9` names Krylov.jl / LinearSolve.jl — prefer Krylov.jl: lighter and
  purpose-built). The JVP is the M1 ForwardDiff operator; form **no** global sparse
  Jacobian except in a tiny-grid debug mode.
- Create `benchmarks/islands/` (+ `figures/`) and at least one `islands_*` regression
  case integrated with `regression-harness/` — every physics benchmark ships
  **skipped**, referencing its `[VERIFY]`/`QUESTIONS` id.

## The hard rule (non-negotiable)

**Never assign a value to a `[VERIFY]` or uncleared-`[CHECKED]` physics coefficient,
sign, or normalization.** The moment you would need a specific number/sign from the
literature that isn't human-cleared: (a) implement the *structure* with the
coefficient as a named parameter, (b) add a skipped benchmark referencing the tag,
(c) write a `QUESTIONS.md` entry (context / question / options / recommendation /
gated work), and (d) **switch to the next unblocked task**. Guessing a coefficient
is the exact failure this project exists to prevent. Manufactured, order-unity test
coefficients in the MMS/verification harness are legitimate (they test numerics, not
physics) and carry no tag — do not confuse the two.

## Working discipline

- Before committing any physics-adjacent change, run the **`physics-verifier`**
  subagent; if it returns BLOCK, fix or escalate — do not commit.
- **Run every new kernel once under `--check-bounds=yes`** before trusting it. (M1
  lesson: an `@inbounds` index-swap corrupted memory and passed silently until a
  forced bounds-checked run caught it. Assume new nested-index kernels are guilty
  until a bounds-checked run clears them.)
- Invoke julia with a clean loader path:
  `env -u LD_LIBRARY_PATH /mnt/homes_global/ncl2128/software/julia-1.11.7/bin/julia
  --project=. …` (Q1 resolution: the OMFIT `LD_LIBRARY_PATH` shadows Julia's
  artifacts otherwise).
- Commit granularly to a milestone sub-branch with messages in the repo format
  (`ISLANDS - <TAG> - <message>`); reference `QUESTIONS.md`/ladder IDs where relevant.
  Push only Islands branches (never `develop`/`main`; the hooks enforce this).
- Never weaken a tolerance or re-baseline a target to reach "done" — that is a
  blocker, not a fix.
- Append a `LOG.md` entry (what moved / blocked / next) before ending each session.
  The `Stop` hook will keep you from ending with a dirty tree or a broken build. A
  session that ends without a pushed branch and a status note has failed its exit
  criteria (`06 §2.5`).
- If you exhaust the milestone's unblocked work (everything remaining is gated on
  `QUESTIONS.md`/human clearance), commit, log, and let the session end — the outer
  loop and the human pick it up.

## Definition of NOT done (do not stop early)

Do not declare M2 done if: any structural gate (A5, solve-MMS, A8, A4, A3, A7's
`⟨∂²h/∂x²⟩=0`) is failing or skipped-as-a-shortcut; the L0 machinery submodules are
absent; the suite is red; the tree is dirty; the Paper-I OUTLINE or the clearance
queue is missing; or any coefficient was guessed. Those are blockers, not
completion. Conversely, do **not** keep working past a green structural ladder in an
attempt to reach the York gates — those are gated on human clearance and out of
scope for this run.
