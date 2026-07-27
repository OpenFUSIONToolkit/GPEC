# 09 — Input-completeness audit (the source input manifests)

**Decision D9 (docs/05 "Target tiers"), Paper-I claim C9.** Reproducing an
*absolute* threshold number from a publication (a T4 target) requires **every
input** of that publication's exact scenario. This file audits what each
DK-NTM/RDK-NTM/kokuchou source actually pins down. The rule (docs/05): a T4
comparison is attemptable only where the manifest is complete; where a required
input is **unspecified**, the target permanently downgrades to T3 (scaling +
trend) and any residual gap is reported with an input-sensitivity scan.

**This is itself a reproducibility result** — it documents that the published
NTM-threshold configurations are, in several places, under-specified or
internally contradictory (the type specimen: I19's own run collisionality).

## Manifest template

Each required input, per source: **value + where the paper states it**, or
**"unspecified → assumption + sensitivity needed"**. Fields:
`ε` · `q_s` · shear `ŝ`/`L̂_q` · density gradient `L̂_n` · temperature gradient
`L̂_T`/`η_j` · `τ=T_e/T_i` · `ν_★` per species · `ω_E` · `m/n` · domain `L_x` ·
resolution / convergence · far-field BC · threshold-extraction procedure ·
`Δ′` convention · frame · species list.

---

## I19 — Imada et al., NF 59, 046016 (2019) — DK-NTM (B5a)

Audited first-hand (print pp. 2–6, 10–11). The flagship manifest.

| Input | Value | Source | Status |
|---|---|---|---|
| `ε = r_s/R₀` | 0.1 | §4 (figures) | ✅ specified |
| `m/n` | 2/1 | §4 | ✅ |
| `q_s` | 2 (= m/n) | Eq. 6 def. | ✅ |
| shear `L̂_q` | 1 (`L̂_q⁻¹ = (ψ_s/q_s)dq/dψ = 1`) | §4 normalization | ✅ |
| `τ = T_e/T_i` | 1 (`T_e = T_i`) | §3 ("hydrogenic, quasi-neutral") | ✅ |
| density gradient `L̂_n` | present (`F'_M` drive), value **unspecified** | §2 | ⚠️ unspecified |
| temperature gradient `η_j` | **unspecified** (η appears in Eq. 22 but run value not given) | §3 | ⚠️ unspecified |
| **`ν_★`** | **CONTRADICTORY: §4.2 states `ν_★ = 0.01`; L23 p. 82 quotes this same DK-NTM run at `ν_★ = 10⁻³`** | §4.2 vs L23 | ❌ **contradictory** |
| `ω_E` | 0 (island rest frame, no equilibrium E_r) | §2 | ✅ |
| domain `L_x` | **unspecified** (shooting method; far-field "away from the island") | Appendix | ⚠️ unspecified |
| resolution / convergence | shooting (`n_ξ`, `n_p`, `n_y` grids, Fig. A1); no convergence table | Appendix, Eq. A.2 | ⚠️ partial |
| far-field BC | localized: `ĥ` radial gradient → 0 away; equilibrium gradient present | Appendix (after A.1) | ✅ described |
| threshold-extraction | `w_c` where `dw/dt = 0` (MRE Eq. 1 with a given `Δ′`); the `Δ′` value assumed is not stated | Eq. 1, §5 | ⚠️ `Δ′` value unspecified |
| `Δ′` convention | standard tearing index, jump in `A_∥` log-derivative | Eq. 1 | ✅ convention / ⚠️ value |
| frame | island rest frame | §2 | ✅ |
| species | 1 bulk ion (DK) + flattened electrons (WCHH96) | §2–3 | ✅ |
| **Reported T4 number** | `w_c ≃ 2.76 ρ_θi` half-width (`≡ 8.73 ρ_bi` at ε=0.1) | Fig. 9 | — |

**Verdict (I19 / B5a): T4 NOT cleanly attemptable.** Two required inputs are
missing or contradictory: the run **`ν_★` is internally inconsistent** (0.01 vs
10⁻³ — a factor of 10), and the **`Δ′` value** behind the `dw/dt=0` threshold is
not stated. B5a therefore stays **T3** (threshold *exists* at `w_c ~ O(ρ_θi)`);
any absolute comparison must scan `ν_★ ∈ [10⁻³, 10⁻²]` and report the sensitivity
`∂w_c/∂ν_★` (which is exactly what kokuchou's B5c surface quantifies).

## D21 — Dudkovskaia et al., PPCF 63, 054001 (2021) — RDK-NTM v.1 (B5b)

Audited: abstract + App. A/B (first-hand); config from the abstract/§7.

| Input | Value | Source | Status |
|---|---|---|---|
| `ε` | 0.1 (ρ_bi = ε^{1/2}ρ_θi context) | abstract | ✅ |
| `m/n`, `q_s`, `L̂_q` | 2/1, 2, 1 (as B5a) | §7 | ✅ |
| `τ` | 1 | (lineage) | ✅ |
| `η_j` | `η_j = 1` | (D23a config; carried) | ⚠️ confirm per-run |
| `ν_i★` | `10⁻³–10⁻⁴` (a *range*, not a point) | abstract/App. C | ⚠️ range not point |
| `Φ′_eqm` (`ω_E`) | 0 | abstract | ✅ |
| drift model | **:improved** (`L̂_B⁻¹ = 0` proxy) | App. A, footnote 10 | ✅ |
| domain / resolution | S-space reduction; analytic layers | §3 | ⚠️ different method |
| **Reported T4 number** | `w_c ≈ 0.45 ρ_θi ≡ 1.41–1.47 ρ_bi` half-width | abstract, Fig. 8 | — |

**Verdict (D21 / B5b): T4 partial (ν_★ a range).** The **T2 primary gate is the
`:original→:improved` toggle differential** — a within-code ratio that needs no
absolute manifest and is the reproducible form of the `8.73→1.46` story. The
absolute `0.45 ρ_θi` is a *range* over `ν_i★`, so report it as a T3 band, not a
point.

## D23a — Dudkovskaia et al., NF 63, 016020 (2023) — finite-β, shaped (C4)

| Input | Value | Source | Status |
|---|---|---|---|
| `ν_★` | `10⁻⁴` | §6.3 | ✅ |
| `m/n` | 2/1 | §6.3 | ✅ |
| geometry | Miller (κ, δ, s_κ, s_δ, Shafranov) | Eq. 33 | ✅ parametrized |
| triangularity `δ` scan | +0.42 → −0.5 | Figs. 5–9 | ✅ |
| other Miller shape params (κ, s_κ, s_δ) at each δ | **unspecified in the transcription** — need first-hand read | §6.3 | ⚠️ unaudited |
| `β_θ` | scanned; EAST 91972 context | §6.3 | ⚠️ partial |
| **Reported T4 numbers** | `2w_c` full width 1.82 ρ_bi (δ=+0.42) → 2.90 ρ_bi (δ=−0.5) | §6.3 | — |

**Verdict (D23a / C4): Level-2 milestone; T3 primary** (triangularity
*destabilizing trend*, the ε-crossover, β_θ trend). Absolute widths T4, pending a
first-hand audit of the full Miller parameter set at each δ.

## D23b — Dudkovskaia et al., NF 63, 126040 (2023) — separatrix/polarization (B4, B6)

| Input | Value | Source | Status |
|---|---|---|---|
| `ε`, `m/n`, `ν_i` | 0.1, 2/1, `10⁻³` | §3–4 | ✅ |
| `ω_E` | **scanned** (the point — Δ_pol vs ω_E) | §4, Fig. 8 | ✅ (scan) |
| layer resolution | separatrix-layer resolved (Figs. 3–4) | §3 | ✅ described |
| **Reported T4** | reversal at `ω_E ≈ −0.89 ω_dia,e`; layer effect `w_c 0.78→0.52 ρ_θi` | Fig. 8–9 | — |

**Verdict (D23b / B4): T3 primary** (ω_E² scaling + reversal *existence*; layer
threshold *reduction ratio* ~⅓ is T2). The −0.89 location is T4 (a curve
feature, comparable in morphology).

## L23 — Leigh PhD thesis, York (2023) — kokuchou (B5c)

The most fully-documented source (a thesis) — the **best T4 candidate**.

| Input | Value | Source | Status |
|---|---|---|---|
| `ε` | 0.1 | §6.3 | ✅ |
| `m/n`, `ω_E` | 2/1, 0 | §6.3 | ✅ |
| `ρ̂_θi` range | `[1,5]×10⁻³` | Eq. 6.3.2 validity | ✅ |
| `ν_★` grid | `{5,10,15,20}×10⁻³` | Figs. 6.10–12 | ✅ (grid) |
| equation set | **L23-amended** (the §2.6 corrections) | §2.6 | ✅ (explicit) |
| resolution / `ν_★` floor | `ν_★ ≥ 5×10⁻³` (memory-bound); `ŵ ≤ 0.75 ρ̂_θi` | §5.3, §6.1.2 | ✅ documented |
| far-field BC | neoclassical-matching (not Neumann — §5.3 forensics) | §5.3, §7.1 | ✅ |
| threshold-extraction | `w_c[r_s]` 2D OLS fit, R²=0.9916 | Eq. 6.3.1–2 | ✅ (with fit quality) |
| **Reported T4 surface** | `w_c ≈ 0.440 ρ̂_θi + 0.0178 ν_★ − 7.54×10⁻⁵` | Eq. 6.3.2 | — |

**Verdict (L23 / B5c): T4 attemptable** — the manifest is essentially complete
(a thesis documents its numerics + fit quality + the amended equation set). This
is the source Islands should target for a genuine absolute comparison **after the
deferred constants clear**, reporting grid-convergence + the `∂w_c/∂ν_★` slope
against the fitted `0.0178`.

## Summary — what the audit changes

| Ladder | Reported number | Manifest verdict | Gate tier |
|---|---|---|---|
| B5a (I19) | `2.76 ρ_θi` | contradictory `ν_★`, `Δ′` unspecified | **T3** (existence) — absolute needs ν_★ scan |
| B5b (D21) | `0.45 ρ_θi` | `ν_★` a range | **T2** toggle ratio primary; absolute a T3 band |
| B5c (L23) | `0.440 ρ̂_θi + …` | essentially complete | **T4 attemptable** (best candidate) |
| B4 (D23b) | reversal `−0.89` | scan documented | **T3** (existence/scaling); location T4 |
| C4 (D23a) | `1.82→2.90 ρ_bi` | Miller params unaudited | **T3** trend (L2 milestone) |

The primary quantitative physics gates remain the **T2 internal differentials**
(the drift-model toggle) and **T3 scalings/trends** — none of which need a
complete input manifest. Absolute comparisons are attempted only for L23/B5c
(complete manifest) with a sensitivity scan, per D9.
