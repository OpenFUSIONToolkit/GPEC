# CLAUDE.md — ISLET project conventions

ISLET is a steady-state, multi-species drift-kinetic solver for the resonant
island/layer region in tokamaks, generalizing the Modified Rutherford Equation
the way SLAYER generalized linear layer theory. Read `README.md`, then
`docs/00-roadmap.md`. The `docs/` directory is **normative**: code must not
contradict it; when physics or design must change, amend the doc in the same PR
(doc-first workflow) and append to the Decision Log in `docs/00-roadmap.md`.

## The [VERIFY] policy (most important rule)

Equations and numeric targets transcribed from literature carry `[VERIFY: source]`
tags in docs and in code comments. Rules:

1. Never implement physics against a [VERIFY]-tagged expression as if it were
   confirmed. Implement the *structure*, parameterize the uncertain coefficient,
   and add a failing/skipped benchmark referencing the tag.
2. Never silently "fix" a coefficient to make a benchmark pass. Flag the
   discrepancy for human review with the source citation.
3. Only a human clears a [VERIFY] tag, after checking the source paper. Record
   the clearance (paper, equation number) in the doc.
4. If you (Claude) derive an expression yourself, mark it `[DERIVED: date]` with
   the derivation in `docs/derivations/` — never present a derivation as a
   literature transcription or vice versa.
5. `[CHECKED: source, Eq./p.]` is the intermediate state: the expression was
   transcribed from a PDF in the in-repo reference library (docs/08) with an
   exact equation/page cite and machine-checked against it, but has not yet
   received the human sign-off of rule 3. [CHECKED] expressions may guide
   design but are implemented under the same rule-1 discipline as [VERIFY].
6. **The policy is not paranoia — it is calibrated to this literature.** Leigh
   2023 §2.6 documents concrete coefficient/sign errors in the *published*
   Imada 2019 equation set (docs/01 header). Published equations in this
   lineage are re-derived before implementation, full stop.

## Physics conventions (pinned; changing any of these is a Decision Log entry)

- Island width `w` = **half**-width; Ω = 2x²/w² − cos ξ; O-point Ω = −1,
  separatrix Ω = +1. (Matches every source in the York lineage — [CHECKED:
  I19 Eq. 7; Diss19 Eq. 2.7; L23 Eq. 2.1.8]. Thresholds are always reported
  as half-widths with the gyroradius unit stated; docs/05 reporting rule 5.)
- Primary field representation: A_∥(x, ξ) on the grid. Island coordinates Ω are
  diagnostics only (Decision D1). Any PR introducing Ω as a solve coordinate
  outside the RDK cross-check mode is rejected.
- All frame/frequency conversions live in `src/frames/`. No other module may
  contain an ω sign convention. The polarization-current sign disputes in the
  literature are largely frame disputes; we will not reproduce them internally.
- Normalizations per `docs/01-physics-level0.md §5`. SI only at I/O boundaries.
- Species lists are first-class everywhere; no function may assume a single ion
  species (Decision D3). Trace species go through the linear post-pass
  (`docs/02 §1.2`), with trace-criteria checks that warn, never silently degrade.

## Code conventions

- Julia. Style: 4-space indent, explicit imports, no `using X` wildcard in
  `src/`. Public API documented with docstrings; physics functions cite the doc
  section they implement (`# docs/01 §2.1`).
- Operator stack rules (`docs/03 §2`): terms are independent (no term inspects
  other terms), generic over `eltype` (AD compatibility is tested), and
  allocation-free in `apply!` hot paths (allocation regression test in CI).
- No regime-specific branches in physics code. If you find yourself writing
  `if collisional ... else banana ...` inside an operator, stop: that is the
  disease this project treats. Regime logic is allowed only in `verify/`
  (choosing analytic comparison targets) and in preconditioners (approximations
  there are legitimate).
- Threading: per-thread preallocated caches; no shared mutable state in kernels.

## Testing gates

- `test/`: fast unit + symmetry + conservation + MMS-at-low-resolution + AD
  checks. Must pass on every commit.
- `benchmarks/`: the docs/05 ladder. CI runs the fast subset; the full ladder
  runs before any tag/paper. A PR that changes physics must state which ladder
  IDs it affects and show them green (or explicitly re-baselined with
  justification).
- New operators/terms ship with: MMS test, at least one analytic-limit hook in
  `verify/`, and an AD-compatibility test.

## Merge conflict policy

Synthesize both branch sides rather than choosing one; preserve current-branch
naming conventions; flag ambiguous cases for human review rather than guessing.
Conflicts touching `docs/` or physics conventions always go to human review.

## Documentation policy (docs/07)

Documentation is a build artifact; stale docs are broken builds.

- Any PR changing physics behavior updates the corresponding Physics Book
  section (`docs/physics/`) **in the same PR**, and regenerates affected
  verification figures if outputs change. Otherwise the PR description carries
  `docs-not-needed:` with justification.
- Bidirectional anchors are mandatory: every operator cites its Physics Book
  section (`# physics: <file>#<anchor>`); every Physics Book equation block
  names its implementing symbol. The anchor-sync CI check must stay green.
- The Physics Book documents equations **as implemented** (code normalization,
  code sign conventions). Aspirational physics belongs in docs/00–02.
- Never hand-edit `docs/state/STATE.md` or gallery pages — they regenerate
  from benchmark artifacts.
- Paper figures are generated only by pinned scripts in `benchmarks/figures/`
  reading archived benchmark data; a figure that can't be regenerated by
  `make figures` is a release-blocking bug.
- Each level begins with the paper OUTLINE.md (claims → figures → ladder IDs);
  implementing agents treat it as the figure contract.

## Autonomous-session protocol (docs/06)

- Definition of done for any milestone run = its docs/05 ladder IDs green with
  convergence artifacts + CI passing + PR opened. Never weaken a tolerance or
  re-baseline a target to reach done — that is a blocker.
- When blocked on anything CLAUDE.md forbids guessing ([VERIFY] clearances,
  coefficients, signs, conventions, doc contradictions): append an entry to
  `QUESTIONS.md` (context, question, options, recommendation, gated work) and
  continue with the next unblocked task. Never stall waiting for a human;
  never resolve silently.
- Read `LOG.md` and `QUESTIONS.md` at session start; append a LOG.md entry
  (what moved / blocked / next) before session end. End every session with the
  branch pushed.
- This file governs work inside `islet/`; the repo-root CLAUDE.md governs
  GPEC-wide conventions. Flag genuine contradictions via QUESTIONS.md.
- GPD (`/gpd:*` commands, if installed) is for derivation/verification/
  literature tasks feeding `docs/derivations/` and [VERIFY] proposals — not
  for implementation work, which follows this file and docs/03–05.

## What to do when uncertain

Prefer: (a) parameterize and flag, (b) add a skipped test documenting the
uncertainty, (c) ask — in that order. Do not guess coefficients, sign
conventions, or normalizations; in this project those are the entire failure
mode of the field we're trying to fix.
