# Derivation — the full orbit-averaged collision operator (Level-0, all six terms)

**Provenance:** `[DERIVED: 2026-07-12]` — independent re-derivation (Decision D7)
of the complete orbit-averaged, `(1/v̂_∥)`-weighted collision operator of the
Level-0 master equation, in the code normalization (÷ −m ρ̂_θi).
**Clears (on sign-off):** the six collision terms of I19 Eq. (32) / L23 Eq. (2.3.47)
— the orbit-averaged pitch diffusion, its σ-odd prefactor, the `∂_p` drag, the
neoclassical `∂²_p` diffusion, the `∂²_{yp}` cross term, and the
momentum-restoring integral (`[CHECKED: I19 Eqs. 9–13, 32; L23 Eqs. 2.3.34–2.3.35,
2.3.47–2.3.49; appendix 8.3.2]`, QUESTIONS Q5).
**Supersedes/corrects:** the placeholder `B_profile` (local single-`B` diffusivity)
**and** the σ-even `Configure.collision_coefficient` — the pitch diffusion is
**σ-odd** (§4).
**Status:** ✅ **signed off 2026-07-12**. The five **differential** terms (D+E
pitch diffusion [σ-odd, orbit-averaged `P_oa`, flat measure], A drag, B
neoclassical, C cross) are **implemented** — `Coefficients.orbit_average_pitch_brackets`,
`Configure.{pitch_diffusivity_profile, pitch_collision_coefficient,
collisional_drag_coefficient, neoclassical_diffusion_coefficient,
collisional_cross_coefficient}`, `Operators.{PitchAngleDiffusion (rebuilt),
CollisionalDrag, NeoclassicalDiffusion, CollisionalCross}`, plus the forbidden-pitch
`g=0` domain BC and the new `Level0Physics.m`. The sixth term — the
**momentum-restoring** nonlocal integral (F) — is **now also implemented**
(sign-off 2026-07-14, once the physical `∫d³v` measure landed,
`velocity-moment-measure.md`): `Operators.MomentumRestoring` +
`Configure.momentum_restoring_term`. **All six terms complete.**

## 1. Scope — what "clearing B_profile" actually is

The `B_profile` gate is the visible tip of the **full** collision operator. The
Level-0 master equation's collision side is `⟨(1/v̂_∥)Ĉ_ii(Ḡ₀)⟩_θ` (I19 Eq. 32
RHS). Orbit-averaging the Lorentz operator (`collision-operator.md`, signed off)
and the `(1/v̂_∥)` transit weight produces **six** terms (L23 Eq. 2.3.47, derived
in appendix 8.3.2). Already cleared: the operator *structure* (`collision-operator.md`),
the deflection shape `ν̃(v̂)` and magnitude `ν̂_ii = ε^{3/2}ν_★ ν̃(v̂)`
(`collision-magnitude.md`), and the momentum-restoring average `⟨ν̂_ii⟩_u`
(`collision-magnitude.md`). This derivation clears the **orbit-averaged pitch-space
structure and the six coefficients** on the code grid — including the σ-parity the
current placeholder gets wrong.

## 2. The six terms as printed (L23 appendix 8.3.2, first-hand)

Each L23 term (its Eq. 2.3.38 = 2.3.47 coefficient, appendix p. 158–159), with
`u = v̂`, `b(θ) = (1−ε cosθ)/(1+ε)`, `ρ̂_θ = ρ̂_θi`, orbit average
`⟨·⟩_θ` (passing full circuit / trapped bounce, as for the drift brackets):

| Term | acts on | coefficient (as printed) | source |
|---|---|---|---|
| **A** drag | `∂ḡ/∂p` | `ν̂_ii ρ̂_θ` (passing `Θ_y`) | 8.3.2 l. 8565–8570 |
| **B** neoclassical | `∂²ḡ/∂p²` | `(ν̂_ii σu/2(1+ε)) ρ̂_θ² y⟨1/√(1−yb)⟩_θ` | l. 8571–8582 |
| **C** cross | `∂²ḡ/∂y∂p` | `2ν̂_ii ρ̂_θ y` (passing `Θ_y`) | l. 8583–8588 |
| **D** pitch-drift | `∂ḡ/∂y` | `(ν̂_ii/σu)(1+ε)⟨(2−3yb)/√(1−yb)⟩_θ` | l. 8589–8597 |
| **E** pitch-diffusion | `∂²ḡ/∂y²` | `(2ν̂_ii/σu)(1+ε) y⟨√(1−yb)⟩_θ` | l. 8598–8603 |
| **F** momentum-restoring | source | `2ν̂_ii(1+ε) Ū_∥ᵢ(ĝ+pF̂′)` | l. 8644–8666 |

with the momentum operator (L23 Eq. 8.3.17 / 2.3.49)

```math
\bar U_{\parallel i}(\hat f) = \frac1{\sqrt\pi\,\langle\hat\nu_{ii}\rangle_u}
   \sum_\sigma\sigma\int_0^\infty du\,u^3\hat\nu_{ii}\!\int_0^{b^{-1}}\!dy\,\hat f
   \quad\text{(the }\langle\hat\nu_{ii}\rangle_u\text{ cleared in }\texttt{collision-magnitude.md}).
```

\noindent
**Note the σ-parity:** the `∂_y`/`∂²_y` diffusion (D, E) and the `∂²_p`
neoclassical term (B) carry a `1/(σu)` or `σu` — they are **σ-odd** (the same
`1/v̂_∥` weighting that makes `ω̂_D ∝ σu` and `c_E = ½⟨1/v̂_∥⟩` σ-odd). The `∂_p`
drag (A), the `∂²_{yp}` cross term (C), and the momentum source (F, via its
internal `Σ_σσ`) are **σ-even**. The `ρ̂_θ²` in B is an L23 §2.6 **amendment** to
I19 (l. 7434). The `∫dy` measures (B, F, Eq. 8.3.14/8.3.17) are **flat**.

## 3. Code normalization — the `1/(m ρ̂_θi)` the drift did not have

The cleared advection coefficients divide the master equation by `−m ρ̂_θi` to
keep `c_D = ω̂_D` (`parallel-streaming.md` §2). The drift/streaming/E×B terms
carry an explicit `−m`, so `m` **cancels**. The collision terms A–F **have no `m`**
(L23 Eq. 2.3.47), so dividing by `−m ρ̂_θi` leaves an explicit **`1/(m ρ̂_θi)`** —
a genuine `m`-dependence the advection channels lack.

**The sign-flip step (this fixes the earlier draft).** In L23 Eq. 2.3.47 the
collision terms A–E all appear on the **LHS with a leading `−`** (the `∂_p`-bracket
drag `−ν̂_ii ρ̂_θ Θ`; the four terms `−(…)∂²_p, −(…)∂²_{yp}, −(…)∂_y, −(…)∂²_y`).
The code coefficient is `(L23 LHS coefficient)/(−m ρ̂_θi)`, so the leading `−`
**cancels the `−m`** and every collision code coefficient comes out **positive** —
the *same* division that makes streaming positive (`a_ξ = −m(x/L̂_q)Θ/(−m ρ̂_θi) =
+(x/L̂_q)Θ/ρ̂_θi`, `parallel-streaming.md`). Concretely:

| Term | code coefficient (÷ −m ρ̂_θi) | σ |
|---|---|---|
| A | `+ν̂_ii/m` (·Θ_y), on `∂_x` | even |
| B | `+(ν̂_ii σu ρ̂_θ/2m(1+ε)) y⟨1/√(1−yb)⟩_θ`, on `∂²_x` | odd |
| C | `+2ν̂_ii y/m` (·Θ_y), on `∂²_{xy}` | even |
| D+E | `+(2ν̂_ii(1+ε)/m ρ̂_θ σu)·∂_y(y⟨√(1−yb)⟩_θ ∂_y)` | odd |
| F | `+(2ν̂_ii(1+ε)/m ρ̂_θ) Ū_∥ᵢ` (from the RHS `= F`, ÷ −m ρ̂_θi) | even |

(`p → x` at leading order, `parallel-streaming.md`; `1/σ = σ`; `ν̂_ii = ε^{3/2}
ν_★ ν̃(v̂)`, `v̂ = √E`.) All five differential terms are **positive**, consistent
with the advection block under the shared `÷(−m ρ̂_θi)` convention. So the collision
magnitude relative to the drift scales as `ν̂_ii/(m ρ̂_θi)` — **`m` and `ρ̂_θi` are
required inputs** (`m` is new; `ρ̂_θi` already in `Level0Physics`).

## 4. D+E is an exact divergence — `∂_y(P_oa ∂_y)`, and it is σ-odd

The two pitch terms combine into a single conservative operator. With
`P_oa(y) = y⟨√(1−yb)⟩_θ`, the `∂_y` coefficient (D) is **exactly** `P_oa′`:

```math
\frac{d}{dy}\big[y\langle\sqrt{1-yb}\rangle_\theta\big]
 = \langle\sqrt{1-yb}\rangle_\theta - \tfrac{y}{2}\langle\tfrac{b}{\sqrt{1-yb}}\rangle_\theta
 = \Big\langle\frac{2-3yb}{2\sqrt{1-yb}}\Big\rangle_\theta
```

\noindent
(the integrand identity `(2−3yb)/(2√(1−yb)) = √(1−yb) − yb/(2√(1−yb))` under the
linear `⟨·⟩_θ`; the trapped Leibniz boundary term vanishes because `√(1−yb)=0` at
the turning points). **Verified numerically to ~1e-11** across passing and trapped
(`pitch_verify.jl`). Hence

```math
\boxed{\ \text{D+E} = \frac{2\hat\nu_{ii}(1+\varepsilon)}{m\hat\rho_{\theta i}\,\sigma u}\,
   \partial_y\!\big(y\langle\sqrt{1-yb}\rangle_\theta\,\partial_y g\big)\ }
```

\noindent
(positive: L23's LHS `−(2ν̂_ii/σu)(1+ε)∂_y(P_oa ∂_y)` divided by `−m ρ̂_θi` — the
leading `−` cancels the `−m`, §3.)

a **σ-odd** (`1/σu`), flat-measure divergence with diffusivity
`P_oa = y⟨√(1−yb)⟩_θ`. Consequences:

- **The current placeholder is doubly wrong**: it uses a *local* single-`B`
  diffusivity `λ√(1−λB)` (should be the orbit average `⟨√(1−yb)⟩_θ`) and a
  **σ-even** coefficient (should be **σ-odd**). `⟨√(1−yb)⟩_θ` is a new orbit
  bracket, **smooth everywhere** (no turning-point singularity — the mimetic
  divergence form never forms `P_oa′`), with the physical passing↔trapped jump at
  `y_c` handled by the existing `y_c` matching (L23 Eqs. 2.3.52–54).
- **Flat measure** `wmeas = 1`: L23 confirms the `⟨√(1−yb)⟩` term of `∂²_y`
  vanishes at `y=0` and `y=1/b` → natural (Neumann) zero-flux BC (L23 l. 2740),
  matching `conservative_pitch_operator`'s built-in zero-flux endpoints; the
  operator is a pure `y`-divergence, conserving `∫g dy` (A4 preserved for the new
  `P_oa ≥ 0`).

## 5. Orbit brackets needed

- `⟨√(1−yb)⟩_θ` — **new**, `Coefficients.orbit_average_pitch_bracket`; feeds
  `P_oa` (D+E). Smooth (bounded integrand), passing + trapped.
- `⟨1/√(1−yb)⟩_θ = B₁(y)` — **already cleared** (`orbit_average_exb_bracket`,
  passing) for the neoclassical term B; needs a trapped extension (bounce),
  where the `1/√` turning-point singularity is integrable (as the drift `G`).
- The `∂_p`/cross terms A, C carry **no** orbit bracket (constant in the pitch
  integrand) — only `ν̂_ii ρ̂_θ`, `2ν̂_ii ρ̂_θ y`.

## 6. Operator mapping — three new operator terms

| Term | operator | status |
|---|---|---|
| D+E pitch | `PitchAngleDiffusion(K, c)` | **modify**: `P_oa = y⟨√(1−yb)⟩`, `wmeas=1`; `c` now σ-**odd** `−2ν̂_ii(1+ε)/(m ρ̂_θ σ√E)` |
| A drag | a `∂_x` advection on `g` | **new** small term (coeff `−ν̂_ii/m` Θ_y); the `F̂′` part is the drive (§7) |
| B neoclassical | `∂²_x` (like `PerpTransport`) | **modify** `PerpTransport` to an array coeff `−(ν̂_ii σu ρ̂_θ/2m(1+ε)) y B₁(y)` |
| C cross | `∂²_{xy}` mixed derivative | **new** operator (no cross term exists) |
| F momentum | nonlocal velocity integral of `g` | **new** operator (couples all `E`, `σ` at fixed `(x,ξ,y)` via `Ū_∥ᵢ`) |

## 7. Subtleties for implementation (flagged, not guessed)

- **`ĝ` vs `ĝ₁ = ĝ + pF̂′`.** L23 writes the operator on `ĝ₁`; the code solves for
  `g = ĝ` with the `pF̂′` (Maxwellian-gradient) piece carried by the far-field/drive
  (`gradient-drive.md`). Terms A and F act on `ĝ+pF̂′`, so each contributes an
  operator-on-`g` **plus** a known `F̂′` source that folds into the existing drive
  structure — the split must be made explicitly at assembly (not a new physics
  coefficient).
- **Passing/trapped `Θ_y`.** A and C carry `Θ_y` (passing-only), like streaming;
  B, D, E act on both (with the trapped orbit-average branch).
- **Residual signs — resolved (§3).** All five differential terms are **positive**
  in the code: L23's LHS leading `−` cancels the `−m` of the `÷(−m ρ̂_θi)`
  normalization, exactly as for streaming (`a_ξ` positive). An earlier draft
  asserted negative signs (dropping the flip step); the physics-verifier caught it
  and §3 now shows the flip explicitly.
- **`Ū_∥ᵢ` normalization** uses the cleared `⟨ν̂_ii⟩_u` (`collision-magnitude.md`);
  its `Σ_σσ` makes F σ-even.

## 8. Cross-check table

| Source / check | Statement | Agrees? |
|---|---|---|
| L23 appendix 8.3.2 (first-hand, p. 158–159) | the six coefficients A–F | ✅ transcribed (§2) |
| L23 Eq. 2.3.47 (first-hand, p. 51) | assembled form, `Θ_y`, σ-structure | ✅ (§2–§4) |
| L23 l. 2740 | `⟨√(1−yb)⟩` `∂²_y` term vanishes at `y=0,1/b` → Neumann | ✅ flat-measure zero-flux (§4) |
| divergence identity `P_oa′ = ⟨(2−3yb)/2√(1−yb)⟩` | D = `∂_y`(E) | ✅ **numeric 1e-11** (§4) |
| σ-parity vs `ω̂_D`, `c_E` (`1/v̂_∥` weight) | D,E,B σ-odd; A,C,F σ-even | ✅ consistent (§2, §3) |
| L23 §2.6 amendment (l. 7434) | `ρ̂_θ²` in the neoclassical term B | ✅ retained (§2) |
| cleared `⟨ν̂_ii⟩_u`, `ν̂_ii`, `ν̃` | F normalization, magnitude, shape | ✅ (`collision-magnitude.md`) |

**Triage:** no discrepancy with L23. The one **correction to existing code** is
the σ-parity of the pitch diffusion (σ-even → σ-odd) and its local→orbit-averaged
diffusivity; the rest are additions. Every coefficient traces to L23 appendix
8.3.2 first-hand; the only non-transcription step (the D+E divergence identity) is
proved and numerically confirmed.

## 9. What sign-off authorizes

On sign-off (recorded in docs/01 §2.3):

1. `Coefficients.orbit_average_pitch_bracket(; y, ε)` = `⟨√(1−yb)⟩_θ` (passing +
   trapped), and a trapped extension of `orbit_average_exb_bracket` (`B₁` for B).
2. Add `m` (poloidal mode number) to `Configure.Level0Physics` (scenario input,
   the rational `m/n`).
3. **Modify `PitchAngleDiffusion`**: `P_oa = y⟨√(1−yb)⟩_θ`, `wmeas = 1`, and the
   σ-**odd** coefficient `c[x,ξ,E,σ] = +2ν̂_ii(√E)(1+ε)/(m ρ̂_θi σ√E)` (positive —
   §3 sign-flip) — replacing the σ-even placeholder and the local-`B` diffusivity.
4. Add the four **differential** terms as operators (D+E pitch, A `∂_x` drag, B
   `∂²_x` neoclassical, C `∂²_{xy}` cross), σ-parities per §2, positive per §3.
5. Remove `GatedLevel0Inputs`/`level0_placeholders`; `configure_level0` takes no
   `gated` argument. Add the forbidden-pitch `g=0` domain BC.

**Status of the six terms.** The five **differential** terms (1–4 above) are
**implemented and physics-verifier PASS** (2026-07-12). The sixth — the
**momentum-restoring** nonlocal term F — is **blocked and escalated** (QUESTIONS
Q5): its structure is clear (§6, `Ū_∥ᵢ` moment + `2ν̂_ii(1+ε)Ū F̂_M`
redistribution, positive sign), but `Ū` is the **parallel-flow velocity moment**
whose weight `W(y,E,σ)` is `[VERIFY]`-gated (Q3), and the `g↔F̂_M` energy-measure
convention (the Gauss–Laguerre `e^{−E}` vs `Ū`'s plain `∫du`) must be pinned
first. Not guessed — F waits on Q3. Its magnitude `⟨ν̂_ii⟩_u` is already cleared.

This is a **multi-term milestone**: it clears the last Level-0 operator gate for
the **differential** collision physics (QUESTIONS Q5). The momentum-restoring F
and the Hirshman–Sigmar `k ≃ −1.173` (electron closure, its own
moment problem) then remains deferred.
