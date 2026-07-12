# Derivation — the gradient drive (= the far-field boundary condition)

**Provenance:** `[DERIVED: 2026-07-11]` — independent re-derivation (Decision D7)
of the Level-0 gradient drive, from the ion response of I19 (first-hand,
Eqs. 8, 23–32).
**Status:** 🔶 **structural finding recorded — normalized amplitude + frame sign
awaiting completion/sign-off.** Does not clear any coefficient into `src/`.

## 1. The finding: the drive is a boundary condition, not an interior source

I19 §4 splits the ion distribution (Eq. 28) as
``f_i = (1 - Ze\Phi/T_i)F_{Mis} + \bar G_0(p_\phi, \xi, v)``, and the
orbit-averaged non-adiabatic part (Eq. 29) is

```math
\bar G_0 = p_\phi\,\frac{\omega_{si}^T}{\omega_{ci}}\,\frac{n'}{n}\,F_{Mi} + \bar h_0 ,
```

\noindent
where `ω_si = mcT_i n'/(Zeqn)` is the ion diamagnetic frequency (Eq. 25),
`ω_si^T/ω_si = 1 + (v̂² − 3/2)η_i` (the energy-dependent temperature-gradient
correction), `η_i = (T_i'/T_i)/(n'/n)`, and `h̄₀` is the free function.

The master orbit-averaged equation (I19 **Eq. 32**, reproduced in docs/01 §2) is
**homogeneous** in `Ḡ₀` — transport `= ⟨(1/v̂_∥)Ĉ_ii(Ḡ₀)⟩_θ` — with **no
interior source**. The inhomogeneity is imposed entirely through the **far-field
boundary condition**: I19 seeks "a Maxwellian solution in the vicinity of the
island (the equilibrium profile is assumed far away from the island)" (§4, after
Eq. 20), i.e.

```math
\boxed{\;
\bar G_0 \;\to\; g_{\rm drive}
   \;=\; p_\phi\,\frac{\omega_{si}^T}{\omega_{ci}}\,\frac{n'}{n}\,F_{Mi}
   \quad\text{as } |x| \to L_x .
\;}
```

**Consequence for the operator stack.** In I19's formulation the Level-0
`Operators.GradientDrive` source is **zero**; the drive is the neoclassical
far-field state `Operators.FarFieldConditions.g_far`. This merges two of the
Q5 "gated" items (`gradient_drive` and `far_field`) into one physical object —
the diamagnetic far field `g_drive` — and matches the design's far-field spec
(docs/01 §3: "g → neoclassical (no-island) solution … never bare Neumann").

## 2. The form (rigorous) and the normalized amplitude (pending)

Far away `p_φ → ψ_s x` and `n'/n = L̂_{n0}⁻¹/ψ_s`, so the drive is **linear in
`x`**, carries the **temperature-gradient-corrected diamagnetic amplitude**, and
is Maxwellian in energy:

```math
g_{\rm drive}(x,\xi,y;\,\hat v,\sigma)\big|_{\rm far}
 \;=\; D_{\rm dia}\;x\;\big[\,1 + (\hat v^2 - \tfrac32)\eta_i\,\big]\;F_{Mi}(\hat v),
```

\noindent
with `D_dia = (ω_si^T/ω_si)`-stripped diamagnetic amplitude
`∝ L̂_{n0}⁻¹ · (ω_si/ω_ci)`-in-normalized-units. **Pending (the sub-decision):**
`D_dia` in the code's normalization (docs/01 §5) requires the ion diamagnetic
frequency in normalized form, which is the **frame-convention** quantity
`Frames.FrameConvention.C_dia` — currently `NaN`-gated (QUESTIONS Q3). The frame
identity `ω_dia,e = m T_e n₀′/(−e q_s n₀)` (docs/01 §5, `[CHECKED: Diss19 p. 46]`)
and its ion analogue pin the sign and the `m/q_s` structure; clearing `D_dia`
means clearing `C_dia` in the same step (a bundled sign-off). The `x`-linearity,
the `(v̂²−3/2)η_i` temperature correction, and the Maxwellian factor are **not**
gated — they are the first-hand I19 Eq. 29 structure.

## 3. Cross-check table

| Source | Statement | Agrees with `[DERIVED]`? |
|---|---|---|
| I19 Eq. (28)–(29) (first-hand, p. 5) | `Ḡ₀ = p_φ(ω_si^T/ω_ci)(n'/n)F_Mi + h̄₀` | ✅ structure (§1) |
| I19 Eq. (32) (first-hand, p. 6) | master equation **homogeneous** in `Ḡ₀` | ✅ ⇒ drive is a BC, not a source (§1) |
| I19 §4 after Eq. (20) | "Maxwellian … far away from the island" | ✅ far-field BC = `g_drive` (§1) |
| design docs/01 §3 far-field spec | `g → neoclassical (no-island) solution` | ✅ = `g_drive` |
| frame identity docs/01 §5 | `ω_dia` sign/`m/q_s` structure | ⚠️ `C_dia` `NaN`-gated (Q3) — `D_dia` pending |

**Triage:** no discrepancy in the *structure*; the drive is the far-field
diamagnetic distribution and the interior source is zero (I19 formulation). The
*normalized amplitude* `D_dia` is bundled with the frame convention `C_dia` (Q3)
and is the piece a human must sign off (with the ion `ω_dia` normalization).

## 4. What sign-off would authorize

On sign-off (recorded in docs/01 §2/§5): (i) set the Level-0
`Operators.GradientDrive` source to zero (I19 Formulation A); (ii) build the
far-field `Operators.FarFieldConditions.g_far = D_dia·x·[1+(v̂²−3/2)η_i]·F_Mi`
with `D_dia` from the cleared ion diamagnetic normalization (`Frames`, `C_dia`);
(iii) clear the frame convention `C_dia`/`sign_omega0`/`C_gradient_shift`
(QUESTIONS Q3, the frame identities). This un-gates the `gradient_drive` **and**
`far_field` families together. The `E×B` coupling, collision magnitude, and
orbit-averaged pitch measure remain gated.
