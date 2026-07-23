# Physical-parameter audit of the Islands Level-0 code and tests

Audit date 2026-07-23, prompted by the observation that the stall used unphysical
choices (island/domain sizes ≥ the plasma). Scope: every place a physics parameter
or grid domain is set, in `src/Islands/`, `test/`, and `benchmarks/islands/`.
**Verdict up front**: the *physics* parameter values (ε, q, m/n, τ, ν_★, ρ̂_θi) are
mostly reasonable and match York's regime; the *domains* are grossly unphysical
everywhere (Lx = 6–8 in `x=(ψ−ψ_s)/ψ_s` units, i.e. 6–8× the whole plasma), the
model has **no physical-input layer** (nothing is derived from T_i/B/R/a — all
normalized numbers are hand-set independent knobs), and **York replication is not
demonstrated** (the B5 benchmark is a gated stub).

## The coordinate, and why "Lx" matters

`x = (ψ−ψ_s)/ψ_s` (docs/01 §5). The magnetic **axis is at x = −1** (ψ=0); the edge
is at `x = ψ_edge/ψ_s − 1` (≈ +1 for a mid-radius surface). So the physical range is
`x ∈ [−1, ~+1]`, and a **local layer model needs |x| ≪ 1** (a thin layer around the
rational surface matched to the outer ideal region). Any symmetric domain with
`halfwidth_x > 1` extends past the axis (`x=−6 ⇒ ψ=−5ψ_s < 0`, nonexistent flux).

## Table 1 — hardcoded `Level0Physics` parameters (every site)

| param | meaning (docs/01 §5) | configure test | solve test | **B5 benchmark** | physical? |
|---|---|---|---|---|---|
| `epsilon` ε | `r_s/R_0` (inverse aspect ratio) | 0.1 | 0.1 | **0.1** | ✅ York uses ε=0.1 (L23: "typically ε=0.1") |
| `q_s` | safety factor at r_s | 2.0 | 2.0 | **2.0** | ✅ consistent with m/n=2/1 |
| `m` | poloidal mode number | 2.0 | 2.0 | **2.0** | ✅ the 2/1 NTM |
| `tau` | `T_e/T_i` | 1.0 | 1.0 | **1.0** | ✅ sources assume T_e=T_i |
| `nu_star` ν_★ | `ν_jj Rq/(ε^{3/2}v_th)` | 0.01 | 0.01 | **0.01** | ✅ **low collisionality (banana, ν_★≪1)**; York window 5e-3–2e-2 |
| `rho_hat_theta_i` ρ̂_θi | `ρ_θi/r_s` | **1.0** | 0.05 | **0.05** | 0.05 ✅ (⇒ ρ_i/r_s≈2.5e-3, physical); **1.0 ✗ (ρ_θi = r_s, absurd — a structural-test value)** |
| `w_psi` ŵ | island **half**-width | 0.05 | **0.5** | **0.05** | 0.05 ✅ (threshold regime ŵ/ρ̂_θi=1); **0.5 ✗ (≈½ the plasma; large-island)** |
| `inv_Lq` L̂_q⁻¹ | `(ψ_s/q)dq/dψ` (shear) | 1.0 | 1.0 | 1.0 | ⚠️ O(1), plausible; not tied to a real ŝ |
| `inv_LB` L̂_B⁻¹ | ∇B length (0 if `:improved`) | 1.0 | 0.7 | 1.0 | ⚠️ O(1), plausible |
| `inv_Ln0` L̂_n0⁻¹ | density-gradient length | 1.0 | 1.0 | 1.0 | ⚠️ L_n≈r_s, plausible |
| `eta_i` η_i | `L_n/L_Ti` | 0.5 | 0.5 | 1.0 | ⚠️ 0.5–1; physical η_i~1–3 (a bit low) |
| `dq_dpsi` | `dq/dψ` (feeds ψ̃, shear) | 0.5 | 0.8 | 0.5 | ⚠️ O(1); not independently grounded |
| `mu0_R` | `μ₀R` in the Δ prefactor | 1.0 | 3.0 | 1.0 | ⚠️ a normalization factor, not pinned to a case |

**⚠️ = the value is O(1)/plausible but is a free hand-set knob, not derived from or
checked against a real equilibrium.** The five ✅ physics knobs (ε, q, m/n, τ, ν_★) and
`ρ̂_θi=0.05` are genuinely in York's physical regime.

## Table 2 — grid **domains** (the unphysical part)

| site | `halfwidth_x` = Lx | ŵ used | Lx in plasma units | physical? |
|---|---|---|---|---|
| operators test | 6.0 | — | 6× plasma | structural test (physics-neutral) — tolerable |
| configure test | 6.0 | 0.05 | 6× plasma | structural test — tolerable |
| solve test | 6.0 | 0.5 | 6× plasma | structural test — tolerable |
| MMS (`Verify.jl`) | 6.0 | — | 6× plasma | structural test — tolerable |
| `resolved_island_grid` default | `Lx_over_w·w`, `Lx_over_w=6` | any | 6ŵ | ⚠️ far field only 6 island-widths; and 6w can exceed the plasma for large w |
| **B5 physics benchmark** | **8.0** | 0.05 | **8× plasma** | ❌ a *physics* threshold on an 8×-plasma domain |
| scratch physics runs (this stall) | up to **20w = 1.0** | 0.05 | **whole inner plasma** | ❌ unphysical (reached the axis) |

The `halfwidth_x=6–8` in the unit/MMS tests is harmless (they check discretization,
not physics). It becomes a real error the moment it's used for a **physics** result —
the B5 benchmark and the scratch Δ_neo runs. There is **no** physically-bounded
(`|x|≲0.3`) physics configuration anywhere in the repo.

## Answers to the specific questions

- **Collisionality regime?** **Low (banana), ν_★ = 0.01 ≪ 1** — the correct NTM/drift-
  kinetic regime and squarely in York's window (5×10⁻³–2×10⁻²). ✅
- **Is ρ_i self-consistently computed from an ion temperature?** **No.** `ρ̂_θi` is a
  direct hand-set input. There is **no T_i, B, R, or a in the model** to compute it
  from. `ρ̂_θi=0.05` ⇒ (with ε=0.1, q=2) ρ_i/r_s = ρ̂_θi·ε/q ≈ 2.5×10⁻³ — a physical
  ρ_*, but chosen, not derived.
- **What ρ_i / ρ_θi is being used?** ρ̂_θi = ρ_θi/r_s = 0.05 in the physics configs
  (⇒ ρ_i ≈ 0.0025 r_s); ρ̂_θi = **1.0** in the configure test (ρ_θi = r_s — a
  deliberately non-physical structural value, flagged in that test's comment).
- **Major/minor radius?** **Not inputs.** Only the inverse aspect ratio `ε = r_s/R_0
  = 0.1` appears (⇒ R_0/r_s = 10, a large-aspect ordering — the York choice). No
  absolute R, a, B_0.
- **Are the parameters self-consistent?** **No.** `inv_Ln0`, `eta_i`, `ν_★`,
  `ρ̂_θi`, `inv_Lq`, `dq_dpsi`, `mu0_R` all derive from the same physical
  equilibrium + profiles in reality, but here they are **independent knobs**; nothing
  enforces mutual consistency. The `Species`/`Maxwellian` `T`, `n`, `dlnn_dr`,
  `dlnT_dr` fields exist but are **inert** — no Level-0 coefficient reads them (the
  gradients that matter are `inv_Ln0`/`eta_i` on `Level0Physics`).

## York replication — demonstrated?

**No.** `benchmarks/islands/benchmark_B5_york_thresholds.jl` is the intended
replication (the 8.73→1.46 ρ_bi threshold, T2 `:original`/`:improved` differential).
Its **physics config is actually sound** (ε=0.1, m/n=2/1, τ=1, ν_★=0.01, ρ̂_θi=0.05,
ŵ=0.05 ⇒ ŵ/ρ̂_θi=1, the threshold regime; η_i=1). But it is **gated**
(`const UNGATED=false`; `threshold_width` throws), its **domain is unphysical**
(`halfwidth_x=8`, `y_max=1.2`), and it depends on a converged physical `Δ_neo` we do
not yet have. The only islands regression case is `islands_l0_structural.toml` —
**structural, not a physics result**. So there is no demonstrated York number.

## Findings & recommendations

1. **The physics knobs are fine; the domains are not, and there's no physical
   grounding.** The immediate corrections: (a) every *physics* run/benchmark uses a
   physically-bounded local domain `|x| ≲ 0.2–0.3` (a few island widths, matched to
   the outer region), never `halfwidth_x=6–8`; (b) fix the B5 benchmark domain and
   un-gate it.
2. **Add a physical-input layer (or at least a pinned physical scenario).** A real
   2/1 NTM case (R_0, a, B_0, T_i(r), n_i(r), q(r)) → the normalized vector, so
   `ρ̂_θi`, `ν_★`, `inv_Ln0`, `η_i`, `inv_Lq`, `ψ̃` are **derived and mutually
   consistent**, and documented (docs/09 input manifest is the intended home). This
   is what makes "scan from vanishingly small to large islands" meaningful — you scan
   `w` at *fixed physical equilibrium*, with the far-field matching radius fixed
   (~fraction of r_s), not scaled with w.
3. **Reconcile the ŵ-normalization statements in docs/01** — §5 line 400 says
   `ŵ = w/r_s`, the input-vector line 427 says `ŵ = w/ρ_θi`; the code uses `w_psi`
   directly in `Ω = 2x²/w_psi²` with `x=(ψ−ψ_s)/ψ_s`. Pin one and write the map
   (a normalization-clarity item, kin to the ψ̃ typo history).
4. **Demonstrate exact York replication (B5) as a gate** before trusting any
   `Δ_neo(w)`: same ε/ν_★/ρ̂_θi/η_i, a physical domain, the T2 toggle differential
   (~×6) and the T3 trends. Until that passes, the code is not validated against the
   lineage it implements.
