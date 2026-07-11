# Paper I — OUTLINE (the Level-0 figure contract)

> Created at level *start* per design doc `07 §3`: claims → figures → ladder
> IDs. This outline is the figure contract — agents implementing benchmarks
> know which figures are paper figures from day one. Every claim must be backed
> by a ladder ID; claims lacking one are flagged. Status reflects the
> `[VERIFY]` gating of `docs/src/islands/QUESTIONS.md` (Q2–Q4) — **no
> submission until every tag in the paper's equation set is cleared**.

**Working title:** Formulation and verification of a generalized drift-kinetic
solver for magnetic-island stability (Islands, Level 0).

**Indicative venue:** Phys. Plasmas (methods paper). **Gate:** Level 0
(design `00`), i.e. ladder A + B1/B2/B4/B5a–c green with convergence artifacts.

## Claims and figures

| # | Claim | Figure(s) | Ladder ID(s) | Status |
|---|---|---|---|---|
| C1 | The `(x, ξ; y, E, σ)` discretization converges at design order: spectral in `ξ`, 4th-order FD on layer-clustered `x`/`y` grids, per operator and for the assembled solve | F1: MMS convergence panels (per-operator + assembled residual + assembled solve error vs. resolution) | A1, A2 | **green** (M1/M2; CI artifacts) |
| C2 | The steady-state Newton–Krylov solve is exact-Jacobian (AD), globally convergent from zero states, and its conditioning is monitored at the trapped–passing boundary (no silent-noise regime, contra L23 §4.2) | F2: Newton/GMRES convergence histories with and without the physics-block preconditioner; `σ_min(y_c)` track | A5, A8 (+ solver gates) | **green** (M2) |
| C3 | The discretized collision operator conserves particles exactly and has definite entropy sign — bootstrap-relevant structure holds discretely, not just asymptotically | F3: conservation/entropy residuals vs. resolution and profile | A4 | **green** (L0 parts, M2) |
| C4 | In the no-island limit the solver reproduces standard local neoclassics (bootstrap current vs. Sauter/NEO) — the strongest global check of the velocity-space discretization | F4: `J_bs` vs. `ν_★` against NEO/Sauter | B1 | **gated** — needs cleared L0 coefficient set (Q2, Q3) + external NEO runs |
| C5 | **(T3, primary)** At large `w` the solver recovers the analytic MRE *scalings* `Δ_bs + Δ_cur ∝ 1/w` and `Δ_pol ∝ 1/w³`; **(T4)** the 1/w coefficient vs. WCHH96 Eq. 85 (frame-mapped) is audit-gated | F5: `Δ` channels vs. `w` with fitted exponents and analytic asymptotes | B2 | **gated** — Q2, Q3, Q4 (`ψ̃` [VERIFY]); WCHH96 acquired |
| C6 | **(T2, robust headline)** The `:original → :improved` drift-model **toggle differential** — a ~×6 reduction in `w_c` in an otherwise identical configuration, measured within Islands (the reproducible form of the sources' `8.73 → 1.46 ρ_bi` story) — plus **(T3)** threshold *existence* at `w_c ~ O(ρ_θi)` and kokuchou's `dw_c/dν_★ > 0` trend. **(T4, audit-gated)** the absolute triangle values (`2.76`/`0.45 ρ_θi`, the `0.440…` fit surface) reported only with input manifests | F6: `w_c` vs. configuration with tier labels; F7: the E1 toggle *ratio* scan | B5a, B5b, B5c (E1) | **gated** — Q2 (D7/D8), Q3, Q4 |
| C7 | **(T3, primary)** The polarization `Δ_pol ∝ ω_E²` away from zero with a sign reversal *existing* at an `ω_E` of order `−ω_dia,e`, reversal location insensitive to `w/ρ_θi`; single-`ω_E` `Δ` values are misleading (surfaces over `(w, ω_E)` are the deliverable). **(T4)** the reversal location (`≈ −0.89 ω_dia,e`) audit-gated | F8: `Δ_pol(ω_E)` reversal curve (morphology vs. D23b Fig. 8); F9: `Δ(w, ω_E)` surface | B4 | **gated** — Q2, Q3 (frame-convention signs) |
| C8 (candidate headline) | **(T2, internal)** Resolution of L23's open question: the stabilizing *electron* `Δ_pol` at `ω_E = 0` — reproduced or refuted by the `ω_E` scan with kinetic vs. flattened electrons (E4), a channel-decomposition comparison we control end-to-end | F10: electron-channel `Δ_pol(ω_E)` decomposition | B4 + E4 | **gated** — Q2, Q3; kinetic-electron toggle is M2+/E4 work |
| C9 (methods) | **Input-completeness audit** of the DK-NTM/RDK-NTM/kokuchou configurations: a per-source manifest of what the published NTM-threshold scenarios actually pin down — itself a reproducibility contribution that frames every T4 comparison and pre-empts benchmark-provenance questions | Table: input manifests per source (docs/05 "input-manifest" template) | — (methods) | **in progress** (M2b deliverable) |

## Verification-artifact rules (docs/05 reporting)

- Every figure names its configuration (docs/03 §2) and git SHA; no benchmark
  "passes" on a single grid — convergence + tolerance archived with the result.
- Threshold numbers are reported as **half-widths** with both `ρ_θi` and
  `ρ_bi = ε^{1/2} ρ_θi` stated at the run's `ε` (docs/05 rule 5).
- Disagreements with published targets are triaged per the standing rule
  (docs/05: our bug / their approximation / their published-equation error /
  transcription error / **under-specified source configuration**) with
  `[VERIFY]` resolution logged first.
- **Targets are tiered (Decision D9, docs/05).** The paper's quantitative
  physics claims are T1 (exact math), T2 (internal differentials/cross-checks —
  the sharpest), and T3 (scalings/trends/existence vs. literature). Absolute
  literature numbers (T4) appear only with their input manifests and
  sensitivity scans; where the source is under-specified they are reported as
  order-of-magnitude + trend, not agreement claims.

## Dependencies for un-gating (the human clearance queue)

`QUESTIONS.md` **Q2** (ratify D7/D8 — done), **Q3** (clear the L0 `[CHECKED]`
coefficient set via re-derivation), **Q4** (resolve the `ψ̃` and
B5a-collisionality `[VERIFY]`s; sources acquired). C1–C3 are already green as
CI artifacts. The physics claims un-gate as the M2b derivation lane and the
input-completeness audit (C9) proceed — the T2/T3 scaling gates need only the
cleared coefficients, while the T4 absolute comparisons additionally need the
per-source input manifests.
