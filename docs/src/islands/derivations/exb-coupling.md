# Derivation — the E×B coupling coefficient `c_E`

**Provenance:** `[DERIVED: 2026-07-12]` — independent re-derivation (Decision D7)
of the `E×B` advection channel of the master orbit-averaged drift-kinetic
equation.
**Clears (on sign-off):** the `Operators.ExBDrift` coupling `c_E`
(`[VERIFY: I19 Eq. (32) E×B terms, with the L23 §2.6 amendments]`, QUESTIONS
Q5), building on the already-cleared `ω̂_D` (`magnetic_drift_frequency`) whose
normalization and orbit-average machinery it reuses.
**Status:** ✅ **signed off 2026-07-12** — implemented as
`Coefficients.orbit_average_exb_bracket` + `Configure.exb_coupling_table` and
wired into `Operators.ExBDrift` (now a velocity-dependent array coefficient); the
σ-parity (passing σ-odd, trapped ≡ 0) is verified in
`test/runtests_islands_configure.jl`.

## 1. The channel in the master equation

The Level-0 master equation is I19 Eq. (32) (docs/01 §2), the orbit-averaged DKE
for `Ḡ₀(p̂, ξ, y; v̂, σ)`. Its two `E×B` terms are the last brace of each
advection coefficient:

```math
-m\Big[\;\cdots\;-\;\tfrac{\hat\rho_{\theta i}}{2}\big\langle\tfrac1{\hat v_\parallel}\tfrac{\partial\hat\Phi}{\partial x}\big\rangle_\theta\Big]\partial_\xi\bar G_0
\;+\;m\Big[\;\cdots\;-\;\tfrac{\hat\rho_{\theta i}}{2}\big\langle\tfrac1{\hat v_\parallel}\tfrac{\partial\hat\Phi}{\partial\xi}\big\rangle_\theta\Big]\partial_{p̂}\bar G_0
\;=\;\cdots
```

\noindent
This is the advection of `Ḡ₀` by the island's own `E×B` flow `v_E = B×∇Φ̂/B²`:
the potential `Φ̂(x,ξ)` (closed by quasineutrality, `01 §3`) drives a
`(x,ξ)`-plane flow that shears the distribution.
It is the one Level-0 kinetic term nonlinear in the state — it couples `g` and
`Φ̂` (docs/03 §2).
The normalizations are I19's (docs/01 §5, matched to the `ω̂_D` derivation §1):
`x=(ψ−ψ_s)/ψ_s`, `y=λB_max`, `v̂=v/v_thi`, `b(θ)=B/B_max=(1−ε cosθ)/(1+ε)`,
`σ=sgn(v_∥)`, `v̂_∥=σv̂√(1−λB)=σv̂√(1−yb)`, and the orbit average
`⟨·⟩_θ = (1/2π)∮dθ` (passing) or `(1/2π)Σ_σ∫_{−θ_b}^{θ_b}dθ` (trapped), at fixed
`p_φ` (I19 Eq. 31 — identical to the `ω̂_D` bracket).

## 2. Pull `∂Φ̂` out of the bracket

`Φ̂ = Φ̂(x,ξ)` is a field on the solve plane and carries **no** `θ`-dependence,
so it factors out of the poloidal orbit average:

```math
\big\langle\tfrac1{\hat v_\parallel}\tfrac{\partial\hat\Phi}{\partial x}\big\rangle_\theta
 = \frac{\partial\hat\Phi}{\partial x}\,\big\langle\tfrac1{\hat v_\parallel}\big\rangle_\theta,
\qquad
\big\langle\tfrac1{\hat v_\parallel}\tfrac{\partial\hat\Phi}{\partial\xi}\big\rangle_\theta
 = \frac{\partial\hat\Phi}{\partial\xi}\,\big\langle\tfrac1{\hat v_\parallel}\big\rangle_\theta .
```

\noindent
The entire velocity/orbit content of the channel is the single orbit bracket
`⟨1/v̂_∥⟩_θ`; everything else is the `(x,ξ)`-gradient of `Φ̂`.

## 3. Normalization — matched to the cleared `ω̂_D`, and the operator match

As for streaming (`parallel-streaming.md` §2), divide the whole master equation
by `−m ρ̂_θi` so the cleared drift `c_D = ω̂_D` is unchanged.
Carrying the two signs (`−m` on the `∂_ξ` brace, `+m` on the `∂_{p̂}` brace) and
using `p̂ → x`, `Ḡ₀ → g` (the `O(ρ̂_θi)` orbit-width correction, `01 §2`), the
`E×B` contribution to the **normalized** residual is

```math
\frac{-m\big[-\tfrac{\hat\rho_{\theta i}}{2}\langle\tfrac1{\hat v_\parallel}\rangle_\theta\,\partial_x\hat\Phi\big]\partial_\xi g
      + m\big[-\tfrac{\hat\rho_{\theta i}}{2}\langle\tfrac1{\hat v_\parallel}\rangle_\theta\,\partial_\xi\hat\Phi\big]\partial_x g}{-m\,\hat\rho_{\theta i}}
= \tfrac12\big\langle\tfrac1{\hat v_\parallel}\big\rangle_\theta
  \Big[\frac{\partial\hat\Phi}{\partial\xi}\,\partial_x g
       - \frac{\partial\hat\Phi}{\partial x}\,\partial_\xi g\Big].
```

\noindent
`m` and `ρ̂_θi` cancel completely — unlike streaming, the `E×B` coefficient does
**not** retain a bare `ρ̂_θi` (the two `E×B` terms both carry `ρ̂_θi/2`, which the
`−m ρ̂_θi` normalization removes).
The bracketed differential operator is **exactly** the Poisson bracket that
`Operators.ExBDrift` accumulates,
`c_E\,[(\partial_\xi\hat\Phi)(\partial_x g) - (\partial_x\hat\Phi)(\partial_\xi g)]`
(`Operators.jl`, `apply!(::ExBDrift)`), so the coefficient is fixed with no sign
freedom:

```math
\boxed{\;
c_E = \tfrac12\big\langle\tfrac1{\hat v_\parallel}\big\rangle_\theta,
\qquad \frac1{\hat v_\parallel} = \frac{\sigma}{\hat v\sqrt{1-yb}}\ \ (\sigma^{-1}=\sigma)
\;}
```

## 4. The σ-parity — the crux (passing σ-odd, trapped identically zero)

`1/v̂_∥ = σ/(v̂√(1−yb))` is **σ-odd** (odd under `v_∥ → −v_∥`).
Its orbit average splits sharply by orbit class, and this is the whole physics of
the coefficient:

**Passing** (`y < y_c = 1`).
`σ` is constant along the full poloidal circuit (a passing particle never
reverses `v_∥`), so it pulls out of the plain-angle average:

```math
\big\langle\tfrac1{\hat v_\parallel}\big\rangle_\theta^{\rm pass}
 = \frac{\sigma}{\hat v}\,\frac1{2\pi}\!\int_0^{2\pi}\!\frac{d\theta}{\sqrt{1-yb}}
 = \frac{\sigma}{\hat v}\,B_1(y),
\qquad
B_1(y)\equiv\Big\langle\frac1{\sqrt{1-yb}}\Big\rangle_\theta^{\rm pass}.
```

\noindent
So `c_E^{\rm pass} = (σ/2v̂)\,B_1(y)` — **σ-odd, nonzero**, equal and opposite for
the `v_∥ ≷ 0` populations.
This is the same σ-parity as the magnetic drift `ω̂_D ∝ σv̂` and the passing
drift-island shift `x_D ∝ σ` (docs/01 §2.2): the `E×B` shift of the drift island
is `−½⟨(ρ̂_θi/v̂_∥)Φ̂⟩_θ ∝ σ`, equal and opposite for the two σ, exactly like
`x_D`.

**Trapped** (`1 < y < 1/b_min`).
The bracket is `(1/2π)Σ_σ∫_{−θ_b}^{θ_b}dθ` — a sum over the two banana legs,
whose signs of `v_∥` are opposite.
Because the rest of the integrand `1/(v̂√(1−yb))` is σ-**even** (identical on both
legs), the σ-odd factor makes the two legs cancel:

```math
\big\langle\tfrac1{\hat v_\parallel}\big\rangle_\theta^{\rm trap}
 = \frac1{2\pi}\sum_{\sigma=\pm1}\int_{-\theta_b}^{\theta_b}\frac{\sigma\,d\theta}{\hat v\sqrt{1-yb}}
 = \frac1{2\pi\hat v}\Big[\!\int\frac{d\theta}{\sqrt{1-yb}}-\int\frac{d\theta}{\sqrt{1-yb}}\Big]=0.
\qquad\Rightarrow\qquad c_E^{\rm trap}=0 .
```

\noindent
This is the identical mechanism the code already encodes for the σ-**even** drift
brackets `A`, `G` — there `Σ_σ` gives `×2` (the `A/π`, `G/π` normalization,
`Coefficients.orbit_average_drift_brackets`); here, for the σ-**odd** `1/v̂_∥`,
`Σ_σ` gives `0`.

**The decisive internal check.** The drift-island label `S` (docs/01 §2.2, I19
Eq. 33) reads, for trapped particles,
`S = −p̂ ρ̂_θi ω̂_D Θ(y−y_c) − ½⟨(ρ̂_θi/v̂_∥)Φ̂⟩_θ`.
The sources require **`S ∝ p̂` for trapped — "no island structure"** (docs/01
§2.2: *"Trapped particles: S ∝ p̂ … response tied to the magnetic island"*).
The `E×B` piece `−½⟨(ρ̂_θi/v̂_∥)Φ̂⟩_θ` carries the only ξ-dependence in the trapped
`S` (through `Φ̂(x,ξ)`), so it would **break** `S ∝ p̂` unless it vanishes.
The σ-cancellation makes it vanish — so trapped `c_E = 0` is not a modelling
choice but a **requirement of the published drift-island structure**.
The `E×B` coupling is therefore a **passing-particle** effect, joining
island-streaming (`Θ(y_c−y)`, passing-only) as the passing-particle island
physics (D21 §7: *"passing-particle physics, not banana-orbit physics"*,
docs/01 §2.2).
Trapped particles precess (`ω̂_D ≠ 0`) but neither stream along nor `E×B`-shear
against the island.

## 5. The new orbit bracket `B₁(y)`

The coefficient needs one new poloidal-orbit integral, structurally a sibling of
the cleared `A(y)`, `G(y)`:

```math
B_1(y) = \Big\langle\frac1{\sqrt{1-yb}}\Big\rangle_\theta
       = \frac1{2\pi}\int_0^{2\pi}\frac{d\theta}{\sqrt{1-yb(\theta)}},
\qquad b(\theta)=\frac{1-\varepsilon\cos\theta}{1+\varepsilon},
```

evaluated on the **passing** interval `0 ≤ y < 1` only (trapped `c_E ≡ 0`, §4).
`B₁` shares the `y_c`-layer singularity of `G`: as `y → 1⁻`, near `θ = π` the
field `b → 1` and `1−yb → 1−y + O((π−θ)²)`, so the integrand `∝ 1/|π−θ|` and
`B₁(y) → ∞` **logarithmically** at the trapped–passing boundary `y_c = 1` (the
same integrable near-separatrix layer the `ω̂_D` `G`-bracket develops).
Interior passing nodes (`y` comfortably `< 1`) are finite and computed by the
existing adaptive quadrature; the `y_c` layer is handled exactly as for the drift
(a documented gated placeholder `0` at the node the quadrature cannot resolve,
part of the `y_c`-matching treatment, `04 §3`/ladder A8/QUESTIONS Q5), never a
guessed value.

## 6. Cross-check table

| Source / check | Statement | Agrees with `[DERIVED]`? |
|---|---|---|
| I19 Eq. (32) `E×B` braces (docs/01 §2) | `−(ρ̂_θi/2)⟨(1/v̂_∥)∂_xΦ̂⟩_θ ∂_ξ`, `−(ρ̂_θi/2)⟨(1/v̂_∥)∂_ξΦ̂⟩_θ ∂_{p̂}` | ✅ structure (§1) |
| cleared `c_D = ω̂_D` normalization | divide by `−m ρ̂_θi` ⇒ `c_D` unchanged, `ρ̂_θi` cancels in `c_E` | ✅ (§3) |
| `Operators.ExBDrift` Poisson bracket | `c_E[(∂_ξΦ̂)(∂_x g)−(∂_xΦ̂)(∂_ξ g)]` | ✅ **exact** sign/magnitude (§3) |
| drift-island label `S` for trapped (I19 Eq. 33; docs/01 §2.2) | `S ∝ p̂`, no island structure | ✅ forces trapped `c_E = 0` (§4) |
| `ω̂_D` σ-parity (`x_D ∝ σ`, equal-and-opposite; docs/01 §2.2) | passing `E×B` shift `∝ σ` | ✅ same σ-odd parity (§4) |
| `orbit_average_drift_brackets` `Σ_σ` handling | σ-even ⇒ ×2 (`/π`); σ-odd ⇒ 0 | ✅ reuses machinery (§4, §5) |

**Triage:** no discrepancy.
The operator-bracket match (§3) fixes the sign and magnitude with no freedom;
the drift-island `S ∝ p̂` requirement (§4) fixes the trapped value to exactly `0`
independently.
The one supplied ingredient is the physical parameter set already in
`Level0Physics` (`ε`) — `c_E` introduces **no new physics parameter** (contrast
streaming's `ρ̂_θi`), only the new orbit bracket `B₁(y)`.

## 7. What sign-off authorizes

On sign-off (recorded in docs/01 §2):

1. A new cleared orbit bracket `Coefficients.orbit_average_exb_bracket(; y, ε)`
   returning `B₁(y)` of §5 (passing `y < 1` only; reuses the `orbit_average_drift_brackets`
   quadrature idiom).
2. A `Configure.exb_coupling_table(grid, phys)` building
   `c_E[ix,iξ,iy,iE,iσ] = (σ/2√E)·B₁(y)·Θ(y_c−y)` (passing-only; `v̂=√E`;
   forbidden region and the `y_c` layer → `0`, as `drift_coefficient_table`),
   broadcast over `(x,ξ)`.
3. `Operators.ExBDrift` generalized to accept an **array** `c_E` (velocity-
   dependent `(y,E,σ)`) in addition to the current scalar (backward-compatible;
   the scalar path stays for manufactured tests).
4. `configure_level0` wires `ExBDrift(exb_coupling_table(grid, phys))` and moves
   `:exb` from `gated` to `cleared`; `c_E` is removed from `GatedLevel0Inputs`.

This un-gates the `E×B` family (QUESTIONS Q5).
The collision magnitude `⟨ν̂_ii⟩_u`, the orbit-averaged pitch measure, and the
Hirshman–Sigmar `k` remain gated.
`c_E` introduces no new physics parameter beyond the already-cleared `ε`.
