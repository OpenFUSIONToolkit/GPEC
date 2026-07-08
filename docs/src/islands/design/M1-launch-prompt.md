# M1 launch prompt (Islands overnight autonomous run)

> This file is fed verbatim to the overnight loop:
> `claude --permission-mode dontAsk --continue -p "$(cat docs/src/islands/design/M1-launch-prompt.md)"`.
> It is the milestone contract (design doc `06 §2.1`). Keep it stable across
> relaunches so `--continue` resumes the same objective.

You are working milestone **M1** of the Islands module (`src/Islands/`), a
steady-state drift-kinetic island/layer solver, autonomously and unattended.

## Read first (every session)

`src/Islands/CLAUDE.md` (module conventions + the [VERIFY] policy),
`docs/src/islands/LOG.md` and `docs/src/islands/QUESTIONS.md` (session memory +
open blockers), then the design docs
`docs/src/islands/design/{00-roadmap,03-architecture,04-numerics,05-verification}.md`.
The repo-root `CLAUDE.md` governs GPEC-wide conventions.

## Goal (set this as your `/goal` completion condition)

**M1 is done when all of the following hold:**
1. `test/runtests_islands_*.jl` exist and are included in `test/runtests.jl`, and
   they implement ladder **A1** (MMS: per-operator + assembled-system convergence
   at design order) and **A2** (JVP vs. finite-difference residual), plus an
   allocation-regression test for the `apply!` hot paths — all **green**.
2. The full suite passes: `julia --project=. test/runtests.jl`.
3. A PR is open onto `feature/islands` with the changes.
4. **No `[VERIFY]` coefficient has been assigned a value** without a
   `docs/src/islands/QUESTIONS.md` entry.

## Scope

- **M1 core** (design `03 §1–2`, `04`): the phase-space grids `(x, ξ, λ, E, σ)`
  with the layer-clustered mappings; the operator-stack skeleton (`AbstractTerm`,
  `apply!`, and the term structs from `03 §2`) as **AD-compatible,
  allocation-free stubs**; the MMS harness and per-term AD-vs-FD JVP checks
  (`04 §3`). Build the structure and the tests, not the physics numbers.
- **Early-M2 structure, best-effort after M1 core is green**: wire the
  term/moment/field structure toward the L0 solve — but **every physics
  coefficient stays a parameterized, `[VERIFY]`-tagged stub with a skipped
  benchmark** (CLAUDE.md rule 1).

## The hard rule (non-negotiable)

**Never assign a value to a `[VERIFY]` physics coefficient, sign, or
normalization.** The moment you would need a specific number/sign from the
literature that isn't already `[CHECKED]`-cleared: (a) implement the *structure*
with the coefficient as a named parameter, (b) add a skipped benchmark
referencing the `[VERIFY]` tag, (c) write a `QUESTIONS.md` entry (context /
question / options / recommendation / gated work), and (d) **switch to the next
unblocked task**. Guessing a coefficient is the exact failure this project
exists to prevent.

## Working discipline

- Before committing any physics-adjacent change, run the **`physics-verifier`**
  subagent; if it returns BLOCK, fix or escalate — do not commit.
- Commit granularly to `feature/islands` with messages in the repo format
  (`ISLANDS - <TAG> - <message>`); reference `QUESTIONS.md` IDs where relevant.
  Push only to `feature/islands` (never `develop`/`main`; the hooks enforce this).
- Never weaken a tolerance or re-baseline a target to reach "done" — that is a
  blocker, not a fix.
- Append a `LOG.md` entry (what moved / blocked / next) before ending each
  session. The `Stop` hook will keep you from ending with a dirty tree or a
  broken build.
- If you exhaust the milestone's unblocked work (everything remaining is gated on
  `QUESTIONS.md`), commit, log, and let the session end — the outer loop and the
  human will pick it up.

## Definition of NOT done (do not stop early)

Do not declare M1 done if any A1/A2/allocation test is failing or skipped-as-a-
shortcut, if the suite is red, if the tree is dirty, or if any coefficient was
guessed. Those are blockers, not completion.
