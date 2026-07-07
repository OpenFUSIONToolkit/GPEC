# 00 — Roadmap: the Level structure

Each Level relaxes specific orderings of the Imada 2019 / Dudkovskaia / Leigh
lineage (reference library: docs/08). The code never branches on "which level"
— levels are *configurations* of the operator stack (docs/03), so any
intermediate combination of toggles is legal. A Level is "done" when: (i) its
verification gate below is green (docs/05), (ii) the Physics Book chapters
covering its equations are complete and [VERIFY]-cleared, and (iii) its
manuscript in the paper series is submission-ready, with every claim backed by
a ladder ID and every figure regenerable from archived data (docs/07). Papers
I–VI map to gates as defined in docs/07 §3; each level *starts* by writing the
paper outline (the figure contract), not ends with it.

---

## Level 0 — DK-NTM reproduction (the benchmark configuration)

**Orderings retained** (the I19/L23 set):
- O1. Large aspect ratio ε ≪ 1, circular concentric surfaces, low β.
- O2. Radially local: w ≪ r_s, constant background gradients across the domain.
- O3. Prescribed island: single-harmonic, constant-ψ, fixed w (half-width
      convention, docs/01 §1). No Ampère solve.
- O4. Fixed equilibrium-E_r parameter ω_E (≡ −ω₀, the island propagation
      frequency in the zero-E_r frame; docs/01 §5). The published York
      thresholds sit at ω_E = 0; Islands treats ω_E as a scanned input from day
      one (D23b already does), because Δ_pol ∝ ω_E² with a sign reversal near
      −0.89 ω_dia,e makes single-ω_E polarization values misleading.
- O5. Timescale ordering ω, ω_*, ω_D ≪ ω_bounce (orbit-averaged leading order
      at fixed p_φ → 4D).
- O6. Momentum-conserving pitch-angle collision model, banana regime ν_★ ≪ 1
      (exact operator + the energy-dependence sub-toggle: docs/01 §2.3).
- O7. Ions drift-kinetic; electrons via the WCHH96 analytic closure (flattened
      h(Ω) profile + coupled parallel-flow relation, docs/01 §2.4), with
      `electrons = :kinetic` (the RDK-NTM treatment) available as the E4
      toggle.
- O8. No perpendicular transport operator (w_d physics external).
- O9. Maxwellian backgrounds, single bulk ion species *in the physics* — but the
      species list is a first-class array from day one (docs/02). Multi-species
      *plumbing* is a Level 0 requirement even though multi-species *physics*
      is Level 1+.

**Critical architectural decision made at Level 0 even though it only pays at
Level 3:** discretize in (x, ξ), not island coordinates Ω or drift-surface
coordinates S. Island/drift coordinates presuppose a separatrix and cannot
represent shielded linear states; (x, ξ) representation makes shielding,
penetration, and saturated islands points on one solution manifold. Island
flux-surface averages are *diagnostics*. The RDK S-coordinate solve path
exists as a cross-check mode (its full coefficient set is published: Diss19
Eqs. D.60–D.62, D23b Eq. 19 + App. A), never as the primary representation.

**Prior-art baseline to beat (new since the original plan):** kokuchou (L23)
is a direct 4D implementation of this exact level and documents where it
breaks: ν_★ floor 5×10⁻³ and ŵ ceiling 0.75 ρ̂_θi set by memory + separatrix
resolution, Picard non-convergence, a singular trapped-passing matching
matrix, and a spurious solution branch from Neumann far-field BCs (docs/04
§§2–3, 6). Islands' architecture (matrix-free Newton–Krylov, adaptive
layer-packed grids, neoclassical-matching BCs) is chosen point-by-point
against that failure list. Getting *below* kokuchou's ν_★ floor while matching
its thresholds is the headline Level-0 numerics deliverable.

**Outputs:** Δ_cos(w, ω_E; p), Δ_sin(w, ω_E; p); flux-surface profiles (n, T,
Φ, flows) across the island; J_∥(x, ξ) with species/channel partitions.

**Gate:** (i) large-w analytic limits (ladder B2: Δ_bs+Δ_cur ∝ 1/w vs. WCHH96
Eq. (85), Δ_pol ∝ 1/w³); (ii) York thresholds with exact configurations
(B5a: w_c ≃ 2.76 ρ_θi ≡ 8.73 ρ_bi, :original drift model; B5b: w_c ≈ 0.45
ρ_θi ≡ 1.46 ρ_bi half-width, :improved model; B5c: kokuchou's
w_c ≈ 0.440 ρ̂_θi + 0.0178 ν_★ − 7.54×10⁻⁵ finite-ν_★ surface); (iii)
separatrix-layer polarization structure and the Δ_pol(ω_E) sign-reversal curve
of D23b (B4, B6).

---

## Level 1 — Arbitrary collisionality + multi-species collisions

**Relaxes:** O6.

- Full linearized multi-species Fokker–Planck operator (momentum- and
  energy-conserving; field-particle terms between all species pairs).
  Bootstrap physics is unforgiving about non-conservative collision operators —
  this is a correctness requirement, not a nicety.
- Brute-force numerical resolution of the trapped-passing dissipation layer and
  the separatrix layer at arbitrary ν (both widths ∝ ν^{1/2} at low ν, with the
  E×B-dominated regime where the layer width tracks the iterating potential —
  docs/04 §2). These are the layers RDK-NTM handles analytically and that set
  kokuchou's operating floor; Level 1 owns beating them numerically.
- **Tungsten physics lands here** (docs/02 §W): mixed-regime neoclassics (W in
  Pfirsch–Schlüter/plateau while bulk ions are banana), ion–impurity friction
  modification of the bootstrap drive, Z_eff effect on electron channel.
  Deliverable: predicted in-island impurity density asymmetries, comparable to
  AUG/DIII-D W-accumulation measurements.

**Gate:** (i) in the no-island limit, bootstrap current and multi-species
neoclassical flows vs. NEO/NCLASS across ν_★ (this doubles as the single most
powerful global correctness check of the velocity-space discretization);
(ii) Sauter coefficients recovered in the large-w limit across banana–plateau–PS;
(iii) smooth connection of Level-0 low-ν results to collisional regimes,
including closing the gap between kokuchou's ν_★ ≥ 5×10⁻³ window and
RDK-NTM's ν_★ ≤ 10⁻³ window — the two prior codes never overlapped cleanly
(ladder B7).

---

## Level 2 — General geometry, drop orbit averaging, energetic particles

**Relaxes:** O1, O5, O2 (partially), O9 (backgrounds).

- Retain poloidal angle: the kinetic problem becomes 5D (x, ξ, θ; λ, E).
- Geometry arrives in two steps: **Miller analytic parametrization first**
  (κ, δ, s_κ, s_δ, Shafranov shift — exactly D23a's geometry, giving direct
  benchmark access to its shaping results), then equilibrium ingested from the
  same numerical representations DCON/GPEC use (docs/03 §interfaces); shaped,
  finite-β (the finite-β drift terms are the D23a Eq. 28–31 extension).
  Analytic circular remains a regression toggle.
- Widened, potentially nonlocal radial domain (several ρ_θα), with background
  profile *variation* across the domain as a toggle (relaxing strict locality).
- **Energetic particles land here** (docs/02 §EP): slowing-down F₀ (touches drive
  terms and collision drag), alpha finite-orbit-width nonlocality (the drift-island
  shift is no longer perturbative), and the precession resonance ω ~ ω_D,α — a
  collisionless polarization-type contribution to Δ with no fluid analog. Highest
  physics novelty in the program for burning plasmas (ITER/SPARC NTM thresholds
  with self-consistent alpha kinetics).
- Because alphas and W are trace in density, their response solves are *linear in
  the trace species* given the bulk fields — cheap post-processing passes with an
  additive contribution to Δ_cos/Δ_sin (SpeciesRole mechanism, docs/02 §roles).

**Gate:** (i) general-geometry neoclassics vs. NEO; (ii) orbit frequencies
(bounce, transit, precession) vs. analytic large-aspect-ratio formulas and a
standalone orbit integrator; (iii) Level-0 results recovered when the circular
low-β toggle set is applied; (iv) D23a shaping/finite-β targets with exact
numbers (ladder C4: triangularity 2w_c 1.82 → 2.90 ρ_bi across δ = +0.42 →
−0.5; the ε ≈ 0.3 crossover from w_c ∝ ε^{1/2}ρ_θi to ∝ ρ_θi; β_θ trend vs.
EAST 91972).

---

## Level 3 — Self-consistent electromagnetics: the unification level

**Relaxes:** O3 (and enables retiring O7).

- Solve Ampère's law for the resonant helical harmonic(s) of A_∥ alongside the
  kinetics. Island width/shape becomes an *output*; the external drive enters as
  a boundary condition carrying Δ'(w) and/or the error-field amplitude from the
  outer-region code.
- Multiple ξ-harmonics; island deformation.
- Kinetic (or reduced-fluid) electrons become necessary here: shielding currents
  are carried by electrons; the flattened-electron closure O7 cannot shield.
  (The RDK-NTM kinetic-electron machinery, already the E4 toggle, is the
  starting point.)
- **Small-amplitude limit = linear layer problem.** Verification against SLAYER's
  Δ(Q) across its drift-MHD regimes. **Dependency note:** the SLAYER Δ(Q)
  implementation arrives with GPEC's Tearing module work (PR #238,
  `feature/tearing-growthrates`: `src/Tearing/InnerLayer/SLAYER/` Riccati layer
  model + GGJ under the same interface, dispersion root-finding, and the
  `delta_prime_raw` outer-region Δ′). That PR is sequenced to land before Islands
  work begins (docs/06 §1); ladder D1's in-CI form then calls it directly.
  The transition regime w ~ δ_layer (penetration bifurcation, kinetic) is the
  flagship new-physics deliverable.
- De-risking sub-track (strongly recommended, can start during Level 1): a
  fluid-electron reduced configuration of Level 3 — essentially a kinetic-ion
  analog of nonlinear cylindrical two-fluid codes (TM1-class) — to shake out the
  (x, ξ) electromagnetic solve before full kinetics arrive.

**Gate:** (i) linear limit vs. SLAYER Δ(Q) curves in shared regimes; (ii)
constant-ψ recovery at small Δ' and w ≫ δ_layer; (iii) fluid-limit toggles vs.
an established nonlinear cylindrical code (TM1 / XTOR-2F-class case); (iv)
qualitative reproduction of Fitzpatrick's penetration bifurcation.

---

## Level 4 — Closures: rotation, transport, radiation

**Relaxes:** O4, O8.

- **Torque balance:** ω_E becomes an unknown closed by the Δ_sin = 0 root (the
  sin ξ Ampère projection *is* the torque-balance condition — Diss19 Eq. 2.10)
  plus flux-surface-averaged momentum balance against viscous/NTV restoring
  torques (the in-repo KineticForces module is the natural NTV source),
  appended to the Newton system — the nonlinear analog of SLAYER's
  torque-balance closure. Multiple roots exist (Diss19 Fig. 4.18 found ±0.93,
  ±1.28 ω_dia,e); continuation must track root branches, not assume
  uniqueness. Quasi-static evolution: dw/dt from the MRE assembly, solved as a
  sequence of steady states (arclength continuation in time-like parameter).
- **Perpendicular transport operator:** model χ_⊥ (and D_⊥) as an explicit
  operator, bringing Fitzpatrick's w_d threshold *inside* the fundamental
  equation. Documented honestly as a closure knob (turbulence–island interaction
  is not first-principles here).
- **Radiative/thermo-resistive channel for W:** energy transport closure with a
  radiation sink L_Z(T_e) n_W n_e inside the island and η(T_e) coupling — the
  radiation-driven island / density-limit mechanism (Gates & Delgado-Aparicio
  class). Requires Level 3 (η enters Ampère/Ohm) + Level 1 W transport.

**Gate:** (i) w_d scaling vs. Fitzpatrick 1995; (ii) penetration thresholds with
self-consistent torque balance vs. SLAYER-based thresholds in the linear limit
and vs. empirical scalings in trend (La Haye database context, ladder B9);
(iii) radiation-driven island growth vs. published thermo-resistive island
models.

---

## Milestone sequencing (dependency graph, not a schedule)

```
M0  repo + CLAUDE.md + docs (this)                     ──┐
M1  phase-space grids + operator stack skeleton + AD     │ Level 0
M2  L0 single-species solve, Δ moments, York gates       │
    → Paper I                                          ──┘
M3  FP collision operator + NEO cross-check            ──┐ Level 1
M4  W minority: friction/bootstrap + asymmetry result    │
    → Paper II                                         ──┘
M5  Miller + general geometry + 5D + orbit benchmarks  ──┐ Level 2
M6  slowing-down F0 + alpha trace response + ω_D res.    │
    → Paper III                                        ──┘
M7  fluid-electron (x,ξ) EM solve  [start after M2]    ──┐
M8  kinetic-electron Ampère; SLAYER-limit gate           │ Level 3
M9  w ~ δ_layer transition study                         │
    → Paper IV (flagship)                              ──┘
M10 torque balance + χ⊥ + w_d gate                     ──┐
M11 radiative W channel                                  │ Level 4
    → Paper V                                            │
M12 Δ-surface dataset + emulator release → Paper VI    ──┘
```

Documentation infrastructure (Physics Book skeleton, anchor-sync CI, figure
pipeline, STATE dashboard — docs/07) is part of M0–M1, not deferred: the first
operator merged is the first operator anchored.

Parallelism: M3–M4 and M7 are independent of each other; M5–M6 depends on M3
(collision operator) but not M7.

**Sequencing against GPEC:** the Tearing module PR (#238,
`feature/tearing-growthrates` — SLAYER + GGJ inner layers, dispersion solver,
Δ′ machinery) lands **before** M0. If M0 starts while #238 is still open, the
Islands branch is cut from `feature/tearing-growthrates` rather than `develop`,
so the SLAYER/Δ′ interfaces Islands consumes are in hand from the first commit.

## Risk register

| Risk | Level | Mitigation |
|---|---|---|
| Separatrix + trapped-passing layers unresolvable at low ν without RDK-style analytics — **confirmed by kokuchou hitting a ν_★ = 5×10⁻³ floor** | 0–1 | Mapped/adaptive grids clustered at both layers using the now-known ∝ν^{1/2} width estimates (docs/04 §2); matrix-free removes kokuchou's memory wall; RDK reduction retained as cross-check; accept and document a ν floor per resolution tier |
| Trapped-passing matching block is intrinsically singular; plain linear algebra yields silent noise, not errors | 0+ | Explicit regularized (TSVD-style) treatment of the y_c block in the preconditioner; smallest-singular-value CI monitor (ladder A8); basis-change spike in M1 (docs/04 §3) |
| Separatrix-layer width depends on Φ̂ and moves between nonlinear iterations (E×B-dominated regime) | 0–1 | Pack from lower-bound width estimates over the expected Φ̂ range; post-solve validation of layer resolution; re-mesh-and-continue fallback (docs/04 §2) |
| Far-field BCs admit spurious solution branches (kokuchou's "winged" states under Neumann) | 0+ | Neoclassical-matching far-field BCs, never bare Neumann; continuation warm-starts; spurious-branch detection via far-field flow comparison against no-island neoclassics (docs/01 §3) |
| Published equation sets in the lineage contain errors (L23 §2.6 amendment list against I19 Eq. A.1) | 0 | Independent re-derivation before implementation ([VERIFY]/[DERIVED] policy); benchmark against L23-amended physics; standing triage category in docs/05 reporting rules |
| (x, ξ) small-amplitude limit fails to reproduce delicate linear layer structure | 3 | This is *the* physics risk. De-risk via M7 fluid track; verify against SLAYER regime-by-regime; budget the painful months here |
| In-repo SLAYER not merged yet (on `develop` it is a placeholder; the implementation lives on PR #238 `feature/tearing-growthrates`) | 3 | Sequence #238 before M0; branch Islands from `feature/tearing-growthrates` if starting earlier; fall back to published Park 2022 curves only if that branch stalls |
| Non-conservative collision discretization poisons bootstrap | 1 | NEO no-island cross-check as a CI-level gate; conservation tests as unit tests |
| ω_E sensitivity of polarization makes single-point Δ misleading (Δ_pol ∝ ω_E², sign flip near −0.89 ω_dia,e; L23's anomalous electron Δ_pol at ω_E = 0) | 0–4 | Always publish Δ as surfaces over (w, ω_E), never single points, until Level 4 closes ω_E; E4/E6 toggle studies address the open electron-Δ_pol question |
| 5D cost explosion at Level 2 | 2 | Orbit-averaged (4D) mode retained as toggle; trace-species linear passes; emulator strategy assumes expensive solves |
| Turbulence–island interaction hiding in χ⊥ | 4 | Explicit closure-knob documentation; sensitivity scans part of every Level-4 result |

## Decision log (append-only)

- D1 (adopted): primary representation (x, ξ), never Ω or S. Rationale: Level-3
  unification; island coordinates cannot represent shielded states.
- D2 (adopted): steady-state Newton–Krylov + continuation, not initial-value
  time-stepping and not nested Picard loops. Rationale: Δ-surface generation is
  the product; continuation produces it as a byproduct; bifurcation tracking
  needs it; kokuchou's documented Picard non-convergence (L23 §6.1.1) is the
  empirical case against the alternative.
- D3 (adopted): species list first-class at Level 0. Rationale: retrofit cost ≫
  upfront cost; trace-role machinery needed by both W and EP tracks.
- D4 (adopted): Julia, as a submodule of the GeneralizedPerturbedEquilibrium
  package (`src/Islands/`, `module Islands` — no separate Project.toml).
  Rationale: AD through the operator stack (exact Jacobians + ∂Δ/∂p
  sensitivities), and GPEC-stack affinity (direct calls to the Δ′/SLAYER/
  equilibrium machinery, docs/06 §1).
- D5 (open): velocity coordinates (λ, E) vs. (v_∥, v_⊥) vs. (θ_b-aligned).
  Default (λ, E) with σ = sgn(v_∥); revisit at Level 2 when θ is retained.
  Note: prior art all uses y = λB_max with the y_c = 1 boundary; the singular
  matching block (docs/04 §3) is a point against inheriting it unexamined.
- D6 (open): kinetic electron treatment at Level 3 — full DKE vs. reduced
  (parallel-kinetic) electron model. Decide after M7 results. The RDK-NTM
  kinetic-electron formulation (Diss19 Eq. D.61) is the full-DKE candidate.
- D7 (proposed 2026-07-07, needs human ratification): implement Level-0
  physics from an independent re-derivation cross-checked against the
  L23-amended equation set, treating I19 Eq. (A.1) as printed as known-errata;
  ω_E enters as a scanned input parameter at Level 0 (not deferred to L4).
  Rationale: L23 §2.6 amendment list; D23b ω_E-parametric formulation.
- D8 (proposed 2026-07-07, needs human ratification): benchmark grid = the
  three-code triangle (DK-NTM published numbers, RDK-NTM improved-model
  numbers, kokuchou finite-ν_★ surface) with the B5a/B5b/B5c configurations
  pinned in docs/05, superseding the single "York thresholds" gate item.
