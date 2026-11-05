Read HOW-TO-WORK-THIS.md in full before doing anything else.

# Work order: ntv-orbit — orbit-averaged kinetic energy and NTV torque

Branch `feature/ntv-orbit`. The project directory is three levels up
(`../../../`): `CLAUDE.md`, `PLAN.md`, `JOURNAL.md`, `QUESTIONS.md`,
`refs/`, `paper/`, `codes/`, `handoff/`, `results/`. This packet is the
code-facing part and is deleted by the graduate milestone.

## Goal

Add to `KineticForces` a second way of evaluating the action variation δJ
and the kinetic energy δW_K: along the real guiding-center orbit in
constants of motion (E, μ, P_φ), instead of along a flux surface. Reuse the
resonance operator, the energy integral and the torque identity. Reproduce
the existing path as the orbit width goes to zero. Add an isotropic
slowing-down alpha distribution with analytic derivatives. Produce two
results for the paper (PLAN.md KR1, KR2) and one methods figure (KR3).

## Scope boundary

In scope: trapped and passing orbits in an axisymmetric equilibrium; the
loss boundary at the last closed surface; the ∂f/∂P_φ drive; Maxwellian
and slowing-down distributions; 3D fields with plasma response as GPEC
already computes them; the fixed-MHD field scan via TokaMaker.

Out of scope: orbit following in the 3D field (the perturbation is first
order along the unperturbed orbit); stochastic ripple loss; finite Larmor
radius; collisional orbit-width effects beyond the existing collision
operator; anisotropic alpha distributions; any change to
`ForceFreeStates`, `PerturbedEquilibrium` or `Vacuum`.

## Code-side milestone ladder

See PLAN.md for the full ladder with physics checks. Code-side summary:

| M | Change in `gpec/` | Lines (target) | Exit |
|---|---|---|---|
| M0 | none | 0 | report; physics.md signed (card) |
| M1 | `src/KineticForces/Orbits.jl`, HDF5 group, config flag off by default | < 400 | M1 checks green |
| M2 | orbit-averaged δJ into the existing operator chain, behind the flag | < 400 | ZOW limit green |
| M3 | ∂f/∂P_φ drive, lost orbits, slowing-down species | < 400 | M3 checks green |
| M4 | scripts under `examples/` or the project `results/`; no module change | < 200 | KR1 figure in report |
| M5 | scripts | < 200 | KR2 figure in report |
| M6 | none (paper repo) | 0 | paper draft, co-author gate |
| M7 | docs, regression cases, delete this packet | — | merge PR open |

## Budget

8 milestones, 15 sessions. This is a limit.

## Stop rules (this project's numbers)

- A milestone not green after **2 sessions** (M4: 3): write the Decisions
  card, the report, and stop. Do not start the next milestone.
- Blocked on a human decision: card it, report, stop that milestone.
- A card unanswered for one whole milestone is a stop; say so in the next
  report's status strip and wait.
- The physics sign-off card from M0 unanswered: nothing after M0 starts.
- The DIII-D kinetic EFIT and the ASCOT package are external dependencies,
  not blockers; use the fallbacks in PLAN.md and say so in the report.
- If the ZOW limit (M2 check 1) cannot be made green in two sessions, the
  premise that the new path is a strict generalization is in question; say
  that in the report rather than widening the tolerance.
- Never more than three Decisions cards in a report.

## Conventions

GPEC's own `CLAUDE.md` governs code style, tests, the regression harness
and the commit grammar for every change under `gpec/`. New code goes behind
a config flag that is off by default until M7. No change to any existing
regression case's output before M7.
