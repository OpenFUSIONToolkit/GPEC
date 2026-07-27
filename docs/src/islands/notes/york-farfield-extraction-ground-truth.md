# York ground-truth: far-field BC and Δ_neo extraction vs. our implementation

Purpose: pin exactly how the York codes (Imada/Dudkovskaia/Leigh lineage) impose
the far-field boundary and extract `Δ_neo`, with equation cites, laid side-by-side
with our implementation — so the Q7 sign-off is a **checked comparison**, not a
guess. Read alongside `QUESTIONS.md` Q7 and `docs/01 §3–§4`.

Sources read (in-repo library, docs/08):
- **L23** = Leigh PhD thesis (York 2023), the `kokuchou` code — §2.3.6 (BCs, pp.
  53/56), §2.4 (quasineutrality, pp. 54–56), §2.5–2.6 (current + Δ_loc + amendments,
  pp. 57–60), §7.1 (future work, pp. 140–142).
- **Diss19** = Dudkovskaia PhD dissertation (York 2019), §4.2 (flow moments, pp.
  79–81; the S-coordinate localization).

---

## 1. The Δ extraction — **ours already matches York** (this closes (c))

**York (L23 Eq. 2.5.10 → 2.5.13):** the island-drive parameter is a **volume
moment** of the cos-ξ-projected parallel current, not a matched jump:

```
Δ_loc = −μ₀R ∫_{−1}^{∞} dΩ ∮ dξ ⟨J_∥⟩_θ cos nξ                       (2.5.10)
      = −(1/2)(w²/4 · q_s′/q_s)^{−1} ∫_{−∞}^{+∞} dψ ∮ dξ ⟨J_∥⟩_θ^ψ cos ξ  (2.5.12)
MRE:  (1/r_s²) dw/dt = −(1/τ_R)(1/2)(w²/4L̂_q) ∫_{−∞}^{+∞} dx ∮ dξ ⟨J_∥⟩_θ^ψ cos ξ  (2.5.13)
```

- The **jump** `Δ'` is the *outer* classical-tearing parameter, which York
  **neglects** (`Δ'(w→0) → 0`, L23 §2.5). So the tearing "jump" is NOT `Δ_neo`.
- **L23 footnote 8 (p.59):** "The inclusion of cos ξ will also eliminate the
  ξ-independent component of the current far from the island in ψ, so only the
  ξ-dependent perturbation contributes." → the far-field background is **ξ-independent**,
  so its `∮cos ξ dξ` vanishes by construction. This is the *same cancellation* we
  verified (`Δ − Δpert = 3.6e−14` when subtracting the ξ-independent, σ-even `g_far`):
  a *feature of the correct operator*, not a bug.
- **L23 §2.6 amendment:** "Introduced minus sign to `Δ_loc` dispersion relation
  Eq.(2.5.10)" — sign correction vs I19 (already reflected in our `∓μ₀R/2ψ̃`).

**Ours (`Moments.delta_moments`, `moments/Moments.jl`):**

```
Δ_cos = (∓μ₀R/2ψ̃) · ∫dx ∮dξ J̄_∥ cos ξ            # cubic-spline x-quadrature
```

**Verdict:** structurally identical to L23 Eq. 2.5.13. The prefactor `∓μ₀R/2ψ̃`
matches `(1/2)(w²/4·q_s′/q_s)^{−1}` up to the `ψ̃` `[VERIFY]`/sign items already
tracked in Q4. **⇒ Q7 option (c) (reformulate as a matched-asymptotic Δ′ jump) is
OFF the table** — York's `Δ_neo` *is* our volume moment. The moment operator needs
no change.

Open detail to confirm (not a blocker): York's integrand is the **θ-averaged**
`⟨J_∥⟩_θ^ψ`; verify our `J̄_∥` carries the same θ-average (it should, via the
orbit-average in `weighted_moment!`).

---

## 2. Why York's moment converges and ours doesn't — the far-field BC

Our extraction is right, so the non-localization is upstream: **our perturbed
response does not localize**, whereas York's does *by construction*. Three
ingredients make York's response localize; we are missing all three.

### 2a. The far-field BC is Neumann on the response, **with a null-mode anchor**

**York (L23 Eq. 2.3.51):** "the island perturbation's effect on the ion
distribution must be localised to its vicinity. This requires that `ĝ` must tend to
a constant far from the island (`p → ±∞`), so that `∂f̂_i/∂p → F̂'_{M,i}`. This
results in the … boundary condition:"

```
lim_{p→±∞} ∂ĝ/∂p = 0                                                (2.3.51)
```

plus trapped–passing matching across `y_c = 1` (2.3.52–2.3.54). Here `p ≡ p̂_ϕ` is
the canonical-momentum radial coordinate.

**The winged branch is a known problem in York's own code, and they name the fix.**
L23 §7.1 (p.141): "a more ambitious … improvement would be to implement an
**analytic result for Eq.(2.3.47) in the limit of large `p`, to use in addition to
(or in place of) the `∂ĝ/∂p` boundary condition. This may also help the algorithm
avoid the non-physical 'winged' distributions … if there are **multiple
numerically-valid solutions for `ĝ` satisfying `∂ĝ/∂p = 0`**."

→ Bare `∂ĝ/∂p = 0` is **under-determined** (a constant null mode ⇒ the "winged"
branch). York's remedy is an **analytic large-p asymptotic** that anchors the null
mode.

**Ours (`Operators.FarFieldConditions`, `operators/Operators.jl`):** we already
have both forms —
- `:dirichlet` (default): pins the *value* `g → g_far ∝ x` (the I19 Formulation-A
  drive-in-the-BC form). **Over-constrains** the edge → resolution-sharpening
  boundary layer. (The diagnosed Q7 failure of this mode.)
- `:neumann`: pins the *slope* `∂g/∂x → s_far` on `ĝ = g − g_far` (the York
  `∂ĝ/∂p = 0` form).

**Gap:** our `:neumann` is **bare** — no null-mode anchor, so it admits York's
winged branch. **The missing piece is the analytic large-p asymptotic anchor**
(L23 §7.1), not a new extraction.

### 2b. The radial coordinate must carry the drift-island shift

**York (Diss19 §4.2, p.81):** localization happens in the **drift-shifted**
coordinate

```
p̂_ϕ = x − ρ̂_θi V̂_∥ √(1 − λB(θ))            (the "S-coordinate")
```

At large `ρ̂_θi/w` the shift sustains a non-zero `Σ_σ σ g_i^σ` in the island centre
— "which … provides the basis for an NTM threshold" — and the flow moment **decays
in `p_ϕ`** away from the island (Fig. 4.12b). L23 §7.1 Eq. (7.1.1) proposes a
mapped coordinate `p̃ = ψ − I(ψ)(v_∥/ω_c − v_∥(0)/ω_c(0))` precisely so "the `p`
mesh would no longer need to span a large extent to capture islands of different
radial shift."

**Ours:** we work in **bare `x = (ψ−ψ_s)/ψ_s`**, not the drift-shifted `p̂_ϕ`. This
is the same `x_D` drift-island shift tracked in **Q8**. Without it, the response
does not localize on a bounded box — matching our finding that growing `Lx`
starves the outer region and `Δ_neo` never domain-converges.

### 2c. The nonlinear solve needs globalization (warm start)

**York (L23 §7.1, p.141):** "a possible short-term means of addressing the issue of
`Φ̂` growing unstably is to **reuse `Φ̂` from a 'numerically stable' run as the
initial state of another run** where a stable result is not obtainable when starting
with `Φ̂ = 0`." And Diss19/L23 self-consistent potential `Φ̂ = δn_i/2 + (x −
h(Ω))/(2L̂_n)` (L23 Eq. 2.4.14) localizes only because `h(Ω) → x` cancels far away
(`∂Φ̂/∂x → 0`).

**Ours:** the physical solve "crawls to resmax ~1e-3 from any init" (LOG cont. 2–3).
**This is the same symptom** — York's own `kokuchou` does **not** converge from a
cold `Φ̂ = 0` start at low `ν_★` / high `ŵ/ρ̂_θi`; they warm-start. So our
non-convergence is not necessarily a bug in our equations; it is the known
globalization difficulty of this problem, and the York remedy is continuation /
warm-start, not a tolerance change.

---

## 3. Decision this supports (for human sign-off)

The evidence collapses the Q7 fork to a single, cited path:

- **Extraction (`Δ_neo`): keep our volume moment unchanged.** It is York's
  `Δ_loc` (L23 Eq. 2.5.10–2.5.13). **(c) matched-jump is ruled out** — the jump is
  the outer `Δ'` York neglects.
- **Localization = Q7 option (b), with a concrete recipe:**
  1. Use `:neumann` (`∂ĝ/∂p = 0` on `ĝ = g − g_far`) **plus an analytic large-`p`
     asymptotic anchor** to remove the winged null-mode (L23 §2.3.6 Eq. 2.3.51 +
     §7.1). This is the missing physics — it is a **signed-off item** (the analytic
     large-p form and its normalization) → do not guess; physics-verifier before
     implementing.
  2. Carry the **drift-island shift** in the radial coordinate (`p̂_ϕ`, Diss19 §4.2 /
     L23 Eq. 7.1.1) so the response localizes on a bounded, `w`-independent box —
     this is **Q8** and is a prerequisite, not optional.
  3. **Globalize the solve** (warm-start `Φ̂` / natural continuation), matching L23
     §7.1 — this is a numerics change (allowed without a physics sign-off) and
     directly targets the "crawls from any init" symptom.

Net: Q7 is not "which operator" (settled — ours) but "make the response localize."
Items (1)+(2) are the physics sign-offs; (3) is numerics we can pursue immediately.

Provenance: this note transcribes L23 Eqs. 2.3.51, 2.5.10–2.5.13, footnote 8 (p.59),
§7.1 (pp. 140–142), and Diss19 §4.2 (pp. 79–81) from the in-repo PDFs on 2026-07-23.
Equation numbers are as printed. No coefficient was cleared here — this is a
read-only ground-truth for the Q7/Q8 decisions.
