Read HOW-TO-WORK-THIS.md in full before doing anything else.

# mre-islands — configurable modified Rutherford equation and saturated-island flattening

Project `logan-nc/2026_Claude_NTM_SPARC_ARC`; this packet lives on `feature/mre-islands` of OpenFUSIONToolkit/GPEC and is deleted at the graduate milestone. GPEC issues #629 and #630.

## Goal

Add to the `Tearing` module a modified Rutherford equation (MRE) whose terms — classical Δ′(w) with the Mercier correction, bootstrap with the transport threshold w<sub>d</sub>, Glasser–Greene–Johnson curvature, ion polarization with island rotation as input, and an external vacuum-island drive — can each be switched off with its coefficient exposed; return the marginal and saturated widths per rational surface; recover the Fortran 1.6.0 Callen features (critical vacuum width, Φ<sub>res,crit</sub>, the cubic for w<sub>min</sub> and w<sub>sat</sub>) as a preset; and flatten the pressure profile across any set of saturated islands as a static perturbed equilibrium, reporting ΔW/W, the pedestal-height change and Δj<sub>bs</sub>. The equations are in `physics.md` and are authoritative.

## Scope boundary

In: everything in `physics.md` §1–§5 and §7; outputs under `Tearing/ModifiedRutherford/` in `gpec.h5`; the `[Tearing.ModifiedRutherford]` deck section with deck/API parity tests; TJ-analytic and DIII-D-like regression cases.

Out (binding):
- No time evolution of the island, the profiles or the bootstrap current. GPEC is a perturbed-equilibrium code.
- No coupled multi-surface MRE; each surface is single helicity.
- No new transport or rotation solver; the rotation profile is an input built as `physics.md` §7 states.
- No change to how Δ′, D<sub>I</sub>, D<sub>R</sub>, H or the trapped fraction are computed in `ForceFreeStates`; the MRE consumes them.
- No change to SLAYER or `CriticalResonantField` beyond reading their results.

## Code-side milestone ladder

The full ladder, spanning code and paper, is in `../../../PLAN.md`. The code-facing part:

| M | Product in `src/` | Exit physics checks |
|---|---|---|
| M0 | none (report, helpers, current-behavior plot) | rational-surface locations; D<sub>R</sub> < 0; Riccati vs Galerkin Δ′ sign agreement; Δ′ normalization (physics.md V1) |
| M1 | `src/Tearing/ModifiedRutherford/`: terms and F(w) to `gpec.h5`, deck section, parity tests | classical-only zero at Δ′(w) = 0; bootstrap limits; polarization sign change with ω(ω − ω<sub>\*pi</sub>) |
| M2 | roots, Callen preset, h<sub>v,crit</sub>, Φ<sub>res,crit</sub> | Fortran 1.6.0 Callen roots reproduced; w<sub>m</sub> monotonic in Δ′ and D<sub>nc</sub>; w<sub>m</sub> ∝ ρ<sub>θi</sub> with polarization on |
| M3 | TJ benchmarks, harness cases, PR of M1–M3 | TJ large-aspect-ratio limits; TPS fallback ordering |
| M4 | flattening and its consequences | ΔW → 0 as w → 0; ΔW ∝ w for one narrow island; overlaps counted once |
| M7 | graduate | packet deleted; physics.md → docs/src/ |

Target under ~400 lines of diff per milestone. The GPEC `CLAUDE.md` governs code conventions, the commit grammar (`Tearing - FEATURE - ...`), tests and the regression harness; a PR into `develop` needs third-party human review.

## Budget and stop rules

Budget: 8 milestones (M0–M7), 14 attempts in total, at most 2 attempts per milestone. An attempt is one pass that ends in a report.

- A milestone not green after 2 attempts: write to `../../../QUESTIONS.md` and STOP the project.
- Blocked on anything Nik must decide: Decisions card in the report, recorded in QUESTIONS.md, STOP that milestone.
- The tearing-physics-suite request (M3) is an external dependency, not a blocker; use the fallback in `../../../refs/` if it has not arrived.
- A question unanswered for a whole milestone is itself a stop.
- The same blocker in two milestones: say plainly that the premise may be wrong.

## State at hand-off

`develop` at e08c719c (2026-11-05). Modules read: `ForceFreeStates` (`delta_prime`, `Surfaces/ResistEval.jl`), `Utilities/NeoclassicalResistivity.jl`, `Equilibrium.KineticProfileData`, `PerturbedEquilibrium` (`island_half_width`, resonant flux, singular current), `Tearing` (new submodule beside `CriticalResonantField/`). Reference Fortran: PrincetonUniversity/GPEC v1.6.0 `rdcon/mercier.f` and `gpec/gpout.f`, excerpted in `../../../refs/notes/`.
