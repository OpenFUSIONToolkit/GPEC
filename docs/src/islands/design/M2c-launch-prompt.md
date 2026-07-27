# M2c milestone contract (Islands) — Level-0 assembly + audit + docs infrastructure (autonomous)

> The milestone contract for Islands M2c. Fed to the autonomous loop
> (`claude --permission-mode dontAsk --continue -p "$(cat …/M2c-launch-prompt.md)"`)
> or set as a `/goal` completion condition. Keep it stable across relaunches.

You are working milestone **M2c** of the Islands module, **autonomously and
unattended**. The M2b derivation lane is complete: all six main Level-0
coefficient families (`ψ̃`, `ω̂_D` + drift toggle, collision operator, `h(Ω)`
closure, quasineutrality, `Δ` prefactors) are human-signed-off and cleared into
`Coefficients`/`Moments`. M2c **assembles** that cleared physics into a runnable
Level-0 configuration, produces the input-completeness audit, and stands up the
`docs/07` infrastructure — all **without any new human sign-off**.

## Read first (every session)

`src/Islands/CLAUDE.md`, `docs/src/islands/LOG.md` + `QUESTIONS.md`, then design
docs `00-roadmap`, `01-physics-level0`, `03-architecture`, `04-numerics`,
`05-verification` (**especially "Target tiers", Decision D9**), `06-autonomy-and-tooling`,
`07-documentation-and-papers`, and the derivations `docs/src/islands/derivations/`
(the cleared physics). Repo-root `CLAUDE.md` governs GPEC conventions.

## The hard constraint (you are unattended)

**Human sign-off is unavailable this milestone.** Therefore:

- **Un-gate nothing new.** Only the six already-cleared coefficient families may
  enter `src/` (via `Coefficients.*`). The deferred sub-constants —
  `⟨ν̂_ii⟩_u`, the Hirshman–Sigmar `k ≃ −1.173`, the `1.46` in `f_p` — stay
  `[CHECKED]`-gated. Anywhere the assembly needs them, use a **named, supplied
  parameter (or NaN-gate)** and record a `QUESTIONS.md` entry; never guess.
- **Draft, don't clear.** You may *draft* the deferred-constant derivations
  (`⟨ν̂_ii⟩_u`, `k`, `f_p`) as `[DERIVED]` chapters *awaiting sign-off* — but only
  if you can derive them rigorously from the in-repo sources (read L23 Eq. 4.1.6
  for `⟨ν̂_ii⟩_u`; the trapped-fraction integral for `f_p`; the parallel-viscosity
  moment problem for `k`). **If a derivation is not rigorous, do not fake it** —
  write a `QUESTIONS.md` entry stating exactly what's missing and move on
  (policy rule 4). Physics-verifier every draft.
- **Never weaken a tolerance or re-baseline to reach "done."**

## Goal (set this as your `/goal` completion condition)

**M2c is done when all of the following hold:**

1. **Level-0 coefficient assembly.** A named-configuration builder in `src/Islands`
   (design `03 §2`, e.g. `configure_level0(grid, params, species; variant)` →
   an `IslandStack` + far-field BCs + `Δ`-prefactors) that populates every
   operator-stack coefficient from the **cleared** `Coefficients.*` functions on
   the phase-space grid (orbit-averaged `ω̂_D` per `(y,E,σ)`; the mimetic
   collision `P(y)` from `pitch_diffusivity` + `deflection_frequency`; the
   `Quasineutrality` coefficient; the `GradientDrive` from the cleared closure;
   the `Δ` prefactors). The deferred `⟨ν̂_ii⟩_u`/`k`/`f_p` enter as clearly-named
   gated parameters. Allocation-free, AD-compatible (the M1 discipline). **Tests:**
   the assembly builds; the assembled residual/`newton_krylov` **runs and
   converges structurally** on the `:imada2019`/`:dudkovskaia2021` configs (with
   the deferred constants set to documented placeholder values *for the
   structural run only*, flagged in the test as non-physics); no cleared
   coefficient is a literal in `src/` (all via `Coefficients.*`).
2. **Input-completeness audit** (Decision D9, `docs/05` "Target tiers"): per-source
   input manifests (I19, D21, D23a/b, L23) in a new
   `docs/src/islands/design/09-input-manifests.md`, each required input cited to
   where the paper states it or marked "unspecified → assumption + sensitivity
   needed". This is the D9/Paper-I C9 deliverable and pins which B5/C4 numbers are
   even T4-attemptable.
3. **`docs/07` infrastructure:** the auto-generated **STATE dashboard**
   (`docs/src/islands/state/STATE.md`, ladder status from test/benchmark
   artifacts — never hand-edited) and the **anchor-sync CI check** (every operator
   cites its Physics-Book section; every equation names its implementing symbol).
4. **B-ladder scaffolding wired to the assembly:** the skipped `benchmarks/islands/`
   scripts updated so that, the moment the deferred constants clear, the **T2/T3
   primary gates** (the `:original→:improved` `w_c` toggle differential; the
   `1/w`, `1/w³`, `ν^{1/2}`, `ω_E²` scalings) run against the assembled solve. Keep
   them skipped (gated on QUESTIONS) but make un-skipping a one-line change.
5. **Deferred-constant derivations** drafted where rigorously possible (§ hard
   constraint), each `[DERIVED]`, physics-verifier PASS, **awaiting sign-off** (not
   cleared); the rest escalated in `QUESTIONS.md` with the specific missing piece.
6. Full suite green (`julia --project=. test/runtests.jl`); the Physics-Book /
   `numerics.md` updated for the assembly (docs/07); a PR open onto
   `feature/islands`; physics-verifier PASS on every physics-adjacent commit.

**Explicitly NOT in the M2c DoD (needs human sign-off / clearance):** un-gating
`⟨ν̂_ii⟩_u`/`k`/`f_p`; the B-ladder physics gates; any York number. A physics
threshold result is *not* reachable until a human clears the last constants.

## Working discipline (autonomous)

- Reuse existing GPEC machinery (`FastInterpolations.integrate`, `QuadGK.quadgk!`,
  the `KineticForces/BounceAveraging` λ-averaging, the `EquilibriumConfig` TOML
  pattern) — don't reimplement (repo-root CLAUDE.md).
- Run every new kernel once under `--check-bounds=yes` (the M1 corruption lesson).
- Invoke julia with `env -u LD_LIBRARY_PATH /mnt/homes_global/ncl2128/software/julia-1.11.7/bin/julia --project=. …`.
- Any new Islands submodule needs its `@autodocs` block in `docs/src/islands.md`
  (checkdocs=:exports — this silently reddened docs CI once; verify with a local
  `build_docs_local.jl`).
- Commit granularly (`ISLANDS - <TAG> - …`), reference QUESTIONS/ladder IDs; append
  a `LOG.md` entry per session; end with the branch pushed.
- When blocked on anything requiring human judgment (a sign-off, a coefficient, a
  convention), write a `QUESTIONS.md` entry and switch to the next unblocked task —
  never stall, never guess.

## Definition of NOT done

Any deferred constant un-gated without sign-off; any coefficient guessed; a
deferred derivation faked (presented as rigorous when it isn't); the assembly not
running structurally; the suite red; the tree dirty; the audit or `docs/07` infra
missing. A "physics result" reached by placeholder constants is a structural
check, not a milestone — label it so.
