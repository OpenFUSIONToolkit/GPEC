# 05 — Verification ladder

The project's credibility is this document. Every claim of the form "ISLET
generalizes X" is backed by "ISLET reproduces X in its limit." Benchmarks are
code in `benchmarks/`, each with: named configuration, target (formula/number/
dataset), tolerance, grid-convergence requirement, and status. CI runs a fast
subset; full ladder runs before any tagged release or paper submission.

Source abbreviations and the [CHECKED]/[VERIFY] tag semantics per docs/01
header; the reference-library map is docs/08. Targets below marked [CHECKED]
have been transcribed from the in-repo PDFs with equation/page cites and await
one human sign-off; remaining [VERIFY] items state exactly what is missing.

**Standing triage rule for York-lineage targets**: L23 §2.6 documents errors
in the published I19 equation set (docs/01 header warning). Where ISLET
disagrees with a published DK-NTM number but agrees with the L23-amended
physics, the triage outcome "their published equation set" is available — but
only after the [VERIFY] resolution is logged with the specific amended term.

## A. Structural (pre-physics)

| ID | Target |
|---|---|
| A1 | MMS: per-operator and assembled-system convergence at design order |
| A2 | JVP vs. finite-difference residual directional derivatives |
| A3 | Symmetry/parity relations of Δ_cos, Δ_sin (docs/01 §6) |
| A4 | Conservation: particles (L0); +momentum/energy per collision pair (L1); discrete entropy sign |
| A5 | Zero-drive null test: g ≡ 0, residual = machine zero |
| A6 | 4D orbit-averaged mode = θ-average of 5D mode (once L2 exists) |
| A7 | Closure identities: ⟨∂²h/∂x²⟩_Ω = 0; k → −1.173; f_p → 1−1.46√ε; analytic ⟨ν̂_ii⟩_v [CHECKED: L23 Ch. 4 — kokuchou's unit set, which caught inherited DK-NTM bugs] |
| A8 | Trapped-passing block conditioning monitor: smallest singular value of the y_c matching block tracked; regression = silent-noise failure mode of L23 §4.2 |

## B. Level 0–1 physics

| ID | Target | Source / configuration |
|---|---|---|
| B1 | **No-island limit**: bootstrap current & neoclassical flows vs. NEO/NCLASS across ν_★; single- and multi-species | NEO; Sauter PoP 6, 2834 (1999) |
| B2 | Large-w limit: Δ_bs+Δ_cur ∝ 1/w with coefficient, against WCHH96 Eq. (85) mapped to the island frame (Diss19 p. 86 frame caveat); scaling ε^{1/2}(L_q/L_p)(β_θ/w). Δ_pol ∝ 1/w³ tail | [CHECKED: Diss19 pp. 84–86; D21 Fig. 8-class curves] |
| B3 | Curvature: GGJ/D_R contribution in the appropriate fluid-ish limit; note Δ_cur = O(ε²), negligible in the York configs — pick a configuration where it isn't [VERIFY accessible configuration — may require L4 transport toggle] | Glasser–Greene–Johnson 1975; in-repo GGJ inner-layer model (src/InnerLayer/GGJ) as cross-check |
| B4 | Polarization: (i) Wilson–Connor collisionless and Smolyakov collisional scalings; (ii) the frame-pinned sign structure: Δ_pol ∝ ω_E² away from zero, **sign reversal at ω_E ≈ −0.89 ω_dia,e**, reversal point insensitive to w/ρ_θi; (iii) torque-balance roots at discrete ω̂_E (Diss19 benchmark root ω₀ = −0.93 ω_dia,e) | WCHH96; Waelbroeck & Fitzpatrick PRL 78 (1997); [CHECKED: D23b Fig. 8; Diss19 Fig. 4.18] |
| B5a | **DK-NTM threshold**: w_c ≃ 2.76 ρ_θi (half-width) ≡ 8.73 ρ_bi at ε = 0.1 (ρ_bi = ε^{1/2}ρ_θi; the "8.73" is a unit conversion, not printed in I19). Config: circular, ε = 0.1, L̂_q = 1, m/n = 2/1, T_e = T_i, ω_E = 0, **:original** drift model. [VERIFY: I19 run collisionality — I19 §4.2 states ν_★ = 0.01; L23 p. 82 quotes DK-NTM at ν_★ = 10⁻³ — resolve before pinning tolerance] | [CHECKED: I19 Fig. 9; PRL p. 4] |
| B5b | **RDK-NTM improved-model threshold**: w_c ≈ 0.45 ρ_θi (0.46 fit) ≡ 1.41–1.47 ρ_bi half-width ≡ 2.85 ρ_bi full width. Config: as B5a but **:improved** drift model (L̂_B⁻¹ = 0 proxy), ν_i★ = 10⁻³–10⁻⁴, Φ′_eqm = 0, η_j = 1. The 8.73 → 1.46 pair is authoritative in the D23a abstract | [CHECKED: D21 abstract + Fig. 8; D23a abstract] |
| B5c | **kokuchou finite-ν_★ threshold surface**: w_c[r_s] ≈ 0.440 ρ̂_θi + 0.0178 ν_★ − 7.54×10⁻⁵ (2D OLS, R² = 0.9916), and per-ν_★ fits w_c/ρ̂_θi = {0.397, 0.427, 0.451, 0.487} at ν_★ = {5, 10, 15, 20}×10⁻³. Validity: ρ̂_θi ∈ [1,5]×10⁻³, ε = 0.1, ω_E = 0, m/n = 2/1, **L23-amended equation set**. The ν_★-dependence of the threshold is the new physics here | [CHECKED: L23 Eqs. 6.3.1–6.3.2, Figs. 6.10–6.12] |
| B6 | Δ vs. w curves and separatrix-layer structure vs. published RDK-NTM figures: D23b Fig. 3 (layer-resolved g at both σ), Fig. 4 (g(p_φ) separatrix zoom), Fig. 6 (u_∥i contours with/without layer), Fig. 7 (δΦ contours), Fig. 9 (Δ_neo, Δ*_bs, Δ_pol vs. w against 1/w and 1/w³); layer effect on threshold: w_c 0.78 → 0.52 ρ_θi at ω_E = 0, ν_i = 10⁻³ | [CHECKED: D23b §3–4] |
| B7 | DK mode vs. RDK cross-check mode agreement above the documented ν̂ floor; prior-art agreement window ν_★ ~ 10⁻³–10⁻⁴ (D21 Appendix C benchmarked DK-NTM against RDK-NTM there); note kokuchou could not operate below ν_★ = 5×10⁻³ — beating that floor is an ISLET numerics deliverable | internal; [CHECKED: D21 App. C; L23 §5.3] |
| B8 | W minority: multi-species neoclassical fluxes & bootstrap modification vs. NEO multi-species; PS-impurity/banana-bulk mixed regime | NEO |
| B9 | Threshold-vs-experiment context check (not pass/fail): La Haye NSTX+DIII-D fit w_c = 0.26 ρ̂_θi ≈ 0.955 ρ_bi half-width (quoted as full width 1.91 ρ_bi in the source), vs. B5b/B5c — the residual gap (rotation, shaping, finite ε) is the Level 2–4 motivation | [CHECKED: L23 pp. 131–132, Fig. 6.12; La Haye 2012] |

## C. Level 2 physics

| ID | Target |
|---|---|
| C1 | General-geometry no-island neoclassics vs. NEO on a shaped numerical equilibrium |
| C2 | Orbit frequencies ω_b, ω_t, ω_D(λ, E) vs. standalone guiding-center integrator and large-ε analytic formulas; includes the :original/:improved ω̂_D forms (docs/01 §2.1) as analytic checks |
| C3 | Circular/low-β toggle set reproduces Level 0–1 results on the same equilibria |
| C4 | Shaping/finite-β trends vs. D23a, now with exact targets: (i) triangularity — threshold FULL width 2w_c from 1.82 ρ_bi at δ = +0.42 to 2.90 ρ_bi at δ = −0.5 (ν_★ = 10⁻⁴, m/n = 2/1, Miller geometry), destabilizing trend confirmed against the DIII-D 2/1 onset database; (ii) aspect ratio — w_c ∝ ε^{1/2}ρ_θi up to ε ≈ 0.3, crossing to w_c ∝ ρ_θi beyond; (iii) β_θ — w_c grows with β_θ, benchmarked vs. EAST discharge 91972 RMP-triggered 2/1 onset | [CHECKED: D23a §6.3, Figs. 5–9] |
| C5 | Slowing-down F₀: analytic slowing-down flux/current limits in no-island geometry [identify best analytic target — candidate: classical alpha-driven current / electron shielding results] |
| C6 | Precession resonance: trace-EP Δ contribution vs. a reduced analytic resonant-response model in a controlled limit [derive companion analytic limit as part of the study — publishable on its own; the bounce/angle-variable + separatrix-layer machinery of Dudkovskaia JPCS 1125 012009 (2018) is the methodological antecedent] |

## D. Level 3–4 physics

| ID | Target |
|---|---|
| D1 | **Linear limit vs. SLAYER**: (Δ_cos + iΔ_sin)(Q) across drift-MHD regimes; frame/Q mapping documented. **Prerequisite**: the in-repo SLAYER Δ(Q) lands with the Tearing module PR (#238, `src/Tearing/InnerLayer/SLAYER/`), sequenced before ISLET M0 (docs/00, docs/06 §1); this benchmark then calls it directly in CI. Published Park 2022 curves remain the independent cross-check. [VERIFY: Park PoP 29 (2022) conventions — paper not in the reference library; acquire] |
| D2 | Constant-ψ recovery: prescribed-island results re-derived as the small-Δ′, w ≫ δ_layer limit of the self-consistent solve |
| D3 | Fluid-limit toggles vs. an established nonlinear cylindrical two-fluid code (TM1-class case) for island growth and penetration |
| D4 | Penetration bifurcation: fold structure and hysteresis qualitatively vs. Fitzpatrick 1998; thresholds with L4 torque balance vs. SLAYER-derived thresholds in the linear limit |
| D5 | w_d: threshold island width scaling vs. Fitzpatrick PoP 2, 825 (1995) with the χ⊥ operator on [VERIFY target formula] |
| D6 | Radiative island: growth/threshold trends vs. published thermo-resistive island model (Gates–Delgado-Aparicio class) [select specific reference case] |
| D7 | Torque moment: linear-limit Δ_sin ↔ SLAYER torque; NTV-side consistency check against the in-repo KineticForces (PENTRC) module in the appropriate limit [scope carefully — may be a paper, not a benchmark] |

## E. The toggle-impact studies (deliverable science, run on the ladder)

Not pass/fail — measured differences, each a figure or paper section:

- E1: drift-frequency model :original vs. :improved — reproduce the
  8.73 → 1.46 ρ_bi finding (B5a/B5b configs), then extend across parameter
  space. The toggle is one term (L̂_B⁻¹ handling, docs/01 §2.1) — the cheapest
  high-impact study in the program.
- E2: orbit-averaged 4D vs. full 5D (the O5 ordering's cost).
- E3: pitch-angle vs. full FP collisions across ν̂ (the O6 cost); plus the
  energy-dependence sub-toggle ν(v) Chandrasekhar-form vs. V⁻³ (I19 vs. D21
  operator variants, docs/01 §2.3) — quantifies a known inconsistency *within*
  the York lineage.
- E4: flattened vs. kinetic electrons (O7) — the DK-NTM/kokuchou closure vs.
  the RDK-NTM kinetic treatment; mandatory before trusting anything at small w
  at L3. Includes reproducing (or refuting) L23's unexplained stabilizing
  electron Δ_pol at ω_E = 0 (L23 §6.2.2/§7) — a live physics question the
  ω_E scan can settle.
- E5: local vs. profile-variation domains for EPs (O2 cost at large ρ_θα).
- E6: fixed-ω_E vs. torque-balance ω_E (O4) — the polarization-term
  sensitivity study; publish Δ surfaces both ways. D23b Fig. 8 (sign reversal
  at ω_E ≈ −0.89 ω_dia,e) is the anchor curve.

## Reporting rules

1. No benchmark "passes" on a single grid: convergence demonstrated, tolerance
   stated, both archived with the result.
2. Every figure in every paper names its configuration (docs/03 §2) and git SHA.
3. Disagreements with published targets are triaged as {our bug, their
   approximation, **their published-equation error** (see standing triage
   rule), transcription error} — with [VERIFY] resolution logged in this
   file's history before the triage concludes anything other than "our bug."
4. Every ladder benchmark ships with a figure script per the docs/07 §2
   pipeline; the same script feeds CI artifacts, the state gallery, and paper
   panels. Ladder status renders automatically into docs/state/STATE.md — this
   file defines targets; the dashboard reports reality.
5. Threshold numbers are always reported as **half-widths with the unit
   stated** (ρ_θi and ρ_bi = ε^{1/2}ρ_θi both given at the run's ε), because
   the literature mixes half/full widths and both gyroradius units (docs/01 §1).
