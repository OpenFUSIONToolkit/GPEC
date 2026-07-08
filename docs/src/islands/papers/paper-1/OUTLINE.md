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
| C5 | At large `w` the solver recovers the analytic MRE limits: `Δ_bs + Δ_cur ∝ 1/w` (WCHH96 Eq. 85, frame-mapped) and `Δ_pol ∝ 1/w³` | F5: `Δ` channels vs. `w` with analytic asymptotes | B2 | **gated** — Q2, Q3, Q4 (WCHH96 acquisition; `ψ̃` [VERIFY]) |
| C6 | The three-code threshold triangle is reproduced in its exact configurations: DK-NTM `w_c ≃ 2.76 ρ_θi ≡ 8.73 ρ_bi` (`:original` drift model), RDK-NTM `w_c ≈ 0.45 ρ_θi ≡ 1.46 ρ_bi` (`:improved`), kokuchou's finite-`ν_★` surface `w_c(ρ̂_θi, ν_★)` — and the 8.73 → 1.46 shift is a single drift-model toggle (E1) | F6: threshold `w_c` vs. configuration (the triangle); F7: the E1 toggle scan | B5a, B5b, B5c (E1) | **gated** — Q2 (D7/D8), Q3, Q4 (B5a collisionality) |
| C7 | The polarization contribution is frame-pinned: `Δ_pol ∝ ω_E²` away from zero with sign reversal at `ω_E ≈ −0.89 ω_dia,e`, insensitive to `w/ρ_θi`; single-`ω_E` `Δ` values are misleading (surfaces over `(w, ω_E)` are the deliverable) | F8: `Δ_pol(ω_E)` reversal curve vs. D23b Fig. 8; F9: `Δ(w, ω_E)` surface | B4 | **gated** — Q2, Q3 (frame-convention signs) |
| C8 (candidate headline) | Resolution of L23's open question: the stabilizing *electron* `Δ_pol` at `ω_E = 0` — reproduced or refuted by the `ω_E` scan with kinetic vs. flattened electrons (E4) | F10: electron-channel `Δ_pol(ω_E)` decomposition | B4 + E4 | **gated** — Q2, Q3; kinetic-electron toggle is M2+/E4 work |

## Verification-artifact rules (docs/05 reporting)

- Every figure names its configuration (docs/03 §2) and git SHA; no benchmark
  "passes" on a single grid — convergence + tolerance archived with the result.
- Threshold numbers are reported as **half-widths** with both `ρ_θi` and
  `ρ_bi = ε^{1/2} ρ_θi` stated at the run's `ε` (docs/05 rule 5).
- Disagreements with published targets are triaged per the standing rule
  (docs/05: our bug / their approximation / their published-equation error /
  transcription error) with `[VERIFY]` resolution logged first.

## Dependencies for un-gating (the human clearance queue)

`QUESTIONS.md` **Q2** (ratify D7/D8), **Q3** (clear the L0 `[CHECKED]`
coefficient set), **Q4** (resolve the `ψ̃` and B5a-collisionality `[VERIFY]`s;
acquire WCHH96 + Park 2022). C1–C3 are already green as CI artifacts; C4–C8
un-gate in that order of effort once the queue clears.
