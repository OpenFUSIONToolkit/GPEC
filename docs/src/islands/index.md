# Islands — drift-kinetic island/layer solver

> GPEC submodule (`src/Islands/`, `module Islands`). The name is a plain
> description of the domain — the resonant magnetic island/layer region — per the
> repo module-naming convention (simple, intuitive names, not standalone-code
> acronyms; repo-root CLAUDE.md). Earlier drafts used the working acronym
> "ISLET"; that has been retired.

## One-paragraph pitch

The Modified Rutherford Equation (MRE) is assembled, by decades-long practice, from
regime-specific analytic terms (Rutherford resistive drive, Sauter/Hegna–Callen
bootstrap, GGJ curvature, Wilson–Connor/Smolyakov polarization, Fitzpatrick's
transport threshold w_d), each valid only in its asymptotic corner of parameter
space. This mirrors the pre-SLAYER state of linear error-field penetration theory,
where Fitzpatrick's and Cole & Fitzpatrick's regime-specific thresholds coexisted
until Park's SLAYER (Phys. Plasmas 29, 2022) solved the underlying layer equations
numerically for arbitrary parameters and recovered every analytic limit. Islands is
the nonlinear analog: a steady-state, multi-species drift-kinetic solver for the
resonant island/layer region that returns the growth moment Δ_cos(w, ω; p) and
torque moment Δ_sin(w, ω; p) for arbitrary parameters, replacing the sum of
regime-specific MRE terms with a single calculation — and, in its small-amplitude
limit, reducing to the linear layer response so that error-field shielding,
penetration, seeded-NTM onset, and island saturation become faces of one solution
manifold.

## Key deliverables (priority order)

1. **Unification of small (SLAYER) and large (MRE) island response.** One inner-region
   framework spanning shielded linear response → penetration bifurcation → saturated
   island. The transition regime w ~ δ_layer has no kinetic treatment in the
   literature; it is the flagship result.
2. **Toggleable ordering stack.** Every approximation in the Imada 2019 /
   Dudkovskaia 2021–2023 / Leigh 2023 (DK-NTM / RDK-NTM / kokuchou) lineage is a
   runtime toggle, so their theory is a *benchmark configuration* of Islands, and
   the impact of each ordering is measurable all the way up to the unreduced
   problem.
3. **Multi-species physics.** High-Z minority impurities (W: mixed-collisionality
   bootstrap/friction physics, radiative island destabilization) and energetic
   particles (alphas: finite-orbit-width nonlocality, slowing-down backgrounds,
   precession resonance with island rotation).
4. **Δ(w, ω; p) surfaces + emulator.** Precomputed response surfaces over
   nondimensional parameter space for integrated modeling, the way SLAYER is used
   for penetration thresholds.

## Relationship to the existing toolchain

Islands develops **inside the OpenFUSIONToolkit GPEC repository** as a
subdirectory Julia package (see docs/06 §1). Status of the GPEC-side assets
(checked 2026-07-07):

- **Outer region + linear layer:** everything Islands consumes arrives with the
  Tearing module work (PR #238, `feature/tearing-growthrates`, sequenced to
  land before Islands starts): SLAYER Δ(Q) and GGJ inner-layer models
  (`src/Tearing/InnerLayer/`), the dispersion/root-finding layer, and the
  outer-region Δ′ as a full 2m×2m matrix (`delta_prime_raw` from
  ForceFreeStates with the new Riccati ideal solver). Islands never recomputes
  global ideal-MHD physics; the SLAYER Δ(Q) is a Level-3 *verification target*
  called directly in CI (docs/05 D1). If Islands work begins before #238 merges,
  branch from `feature/tearing-growthrates`, not `develop`.
- **EP corrections to Δ'** (fast-ion pressure in the outer region) stay on the
  GPEC side (`src/KineticForces`, the PENTRC/NTV machinery — also the natural
  NTV restoring-torque source for Level 4 torque balance); Islands handles
  resonant/orbit-width EP physics at the island.

## Document map

Design docs live in `docs/src/islands/design/`; module conventions in
`src/Islands/CLAUDE.md`. The `docs/NN` shorthand used throughout means design
doc `NN` (`docs/src/islands/design/NN-*.md`).

| File | Contents |
|---|---|
| `src/Islands/CLAUDE.md` | Module conventions for Claude Code: layout map, style, testing gates, [VERIFY] policy, merge policy |
| `design/00-roadmap.md` | Level 0–4 plan, milestones, risk register, decision log |
| `design/01-physics-level0.md` | Level 0 equation set: coordinates, DKE, quasineutrality, moments, nondimensionalization |
| `design/02-species-and-eps.md` | Species abstraction; tungsten (Level 1) and energetic particles (Level 2) physics specs |
| `design/03-architecture.md` | In-repo module layout, operator stack, toggles, AD strategy, coupling interfaces |
| `design/04-numerics.md` | Discretization, boundary layers, Newton–Krylov, continuation, performance model |
| `design/05-verification.md` | The benchmark ladder: every analytic limit and published number Islands must recover, per level |
| `design/06-autonomy-and-tooling.md` | GPEC-repo integration, autonomous Claude Code workflows, GPD, skills, subagents, setup checklist |
| `design/07-documentation-and-papers.md` | Living documentation system (Physics Book, State Dashboard, figure pipeline) and the paper series; level gates are manuscripts |
| `design/08-reference-library.md` | Map of the in-repo PDF sources (`docs/resources/Drift_Kinetic_Island_References/`), per-source load-bearing content, and known cross-source inconsistencies |

## Reading order for a new contributor (human or Claude)

`00-roadmap` → `01-physics-level0` → `03-architecture` → `05-verification`, then
`02` and `04` as needed. Nothing in `src/Islands/` may contradict these documents;
when it must, the document is amended *first* (doc-first workflow, see
`src/Islands/CLAUDE.md`).

## Canonical references

The York drift-kinetic lineage is in-repo as PDFs
(`docs/resources/Drift_Kinetic_Island_References/`) — see **docs/08** for the
per-source map, abbreviations, and known cross-source inconsistencies:

- K. Imada et al., PRL 121, 175001 (2018); JPCS 1125, 012013 (2018); **Nucl.
  Fusion 59, 046016 (2019)** — DK-NTM: 4D nonlinear drift-kinetic island
  theory. The 2019 paper is the complete reference; **its published equation
  set carries known errata (see Leigh 2023 §2.6)**.
- A. V. Dudkovskaia, PhD dissertation, York (2019) — full RDK derivation chain
  and solver coefficient sets (Appendices C–E).
- A. V. Dudkovskaia et al., PPCF 63, 054001 (2021); Nucl. Fusion 63, 016020
  (2023); Nucl. Fusion 63, 126040 (2023) — RDK-NTM v.1–v.3: improved drift
  model, shaping/finite-β, separatrix layer & polarization / ω-dependence.
- S. Leigh, PhD thesis, York (Dec 2023) — `kokuchou`: amended DK-NTM equation
  set, finite-ν★ threshold surface w_c(ρ̂_θi, ν_★), and the most complete
  forensic record of the numerical failure modes (singular trapped-passing
  matching, separatrix-layer resolution, Picard non-convergence, spurious
  Neumann branches).
- A. V. Dudkovskaia et al., JPCS 1125, 012009 (2018) — *phase-space* island
  stability (bump-on-tail); methodological antecedent and EP background only,
  not an NTM threshold source.

Classical MRE-term and penetration literature (not yet in the PDF library):

- R. Fitzpatrick, Nucl. Fusion 33, 1049 (1993); Phys. Plasmas 5, 3325 (1998) — linear penetration thresholds, torque balance.
- A. Cole & R. Fitzpatrick, Phys. Plasmas 13, 032503 (2006) — drift-MHD regime-specific penetration thresholds.
- J.-K. Park, Phys. Plasmas 29 (2022) — SLAYER: parametric layer responses across linear two-fluid drift-MHD regimes.
- P. H. Rutherford, Phys. Fluids 16, 1903 (1973) — nonlinear island evolution.
- O. Sauter et al., Phys. Plasmas 6, 2834 (1999) — bootstrap coefficients (arbitrary ν*).
- A. H. Glasser, J. M. Greene, J. L. Johnson, Phys. Fluids 18, 875 (1975) — curvature (GGJ).
- H. R. Wilson, J. W. Connor, R. J. Hastie & C. C. Hegna, Phys. Plasmas 3, 248 (1996) — the analytic electron closure and large-w limits (load-bearing for Level 0; acquire the PDF); F. L. Waelbroeck & R. Fitzpatrick, PRL 78, 1703 (1997); A. I. Smolyakov — polarization current (regime-dependent).
- R. Fitzpatrick, Phys. Plasmas 2, 825 (1995) — transport threshold w_d.

Equation transcriptions from the in-repo PDFs carry [CHECKED: source, Eq./p.]
tags (AI-verified against the PDF, one human sign-off pending); everything
else carries [VERIFY] until the source is in hand (see CLAUDE.md).
