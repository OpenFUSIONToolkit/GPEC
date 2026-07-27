# Derivation — the gradient drive (= the diamagnetic far-field boundary condition)

**Provenance:** `[DERIVED: 2026-07-11]` — independent re-derivation (Decision D7)
of the Level-0 gradient drive, from the ion response of I19 (first-hand,
Eqs. 8, 23–32).
**Status:** ✅ **signed off 2026-07-11** — implemented as
`Configure.gradient_far_field` (the far-field `g_far`) with the Level-0
`GradientDrive` source set to zero; a new `Level0Physics.eta_i`.

> **Correction (2026-07-11).** An earlier draft of this chapter read I19 Eq. (29)
> as `p_φ(ω_si^T/ω_ci)(n'/n)F_Mi` and concluded the amplitude was bundled with the
> frame convention `C_dia`. A first-hand re-read (PDF p. 5) shows the ratio is
> **`ω_si^T/ω_si`** — a *dimensionless temperature correction*, not a frequency
> ratio. The drive is the standard neoclassical `p_φ F'_Mi` and needs **no frame
> convention**. The corrected result is below.

## 1. The finding: the drive is a boundary condition, not an interior source

I19 §4 splits the ion distribution (Eq. 28) as
`f_i = (1 − Ze\Phi/T_i)F_{Mis} + \bar G_0(p_\phi, \xi, v)`, and the orbit-averaged
non-adiabatic part is (Eq. **29**, first-hand p. 5)

```math
\bar G_0 = p_\phi\,\frac{\omega_{si}^T}{\omega_{si}}\,\frac{n'}{n}\,F_{Mi} + \bar h_0 ,
\qquad
\frac{\omega_{si}^T}{\omega_{si}} = 1 + \Big(\frac{v^2}{v_{thi}^2} - \tfrac32\Big)\eta_i ,
```

\noindent
with `η_i = (T_i'/T_i)/(n'/n)`. The master orbit-averaged equation (I19 **Eq. 32**,
docs/01 §2) is **homogeneous** in `Ḡ₀` — transport `= ⟨(1/v̂_∥)Ĉ_ii(Ḡ₀)⟩_θ`,
**no interior source**. The inhomogeneity is imposed through the **far-field
boundary condition**: I19 seeks "a Maxwellian solution in the vicinity of the
island (the equilibrium profile assumed far away)" (§4), so `h̄₀ → 0` and

```math
\boxed{\;
\bar G_0 \;\to\; g_{\rm drive}
   \;=\; p_\phi\,\frac{\omega_{si}^T}{\omega_{si}}\,\frac{n'}{n}\,F_{Mi}
   \;=\; p_\phi\,F'_{Mi}
   \quad\text{as } |x| \to L_x .
\;}
```

The equality `g_drive = p_φ F'_{Mi}` follows from
`F'_{Mi} = (n'/n)(1+(v̂²−3/2)η_i)F_{Mi}` (differentiate the Maxwellian at fixed
`v`): the drive is the **standard neoclassical drive** — canonical momentum times
the equilibrium-gradient Maxwellian — recognizable and coefficient-free.

**Consequence for the operator stack.** In I19's formulation the Level-0
`Operators.GradientDrive` source is **zero**; the drive is the neoclassical
far-field state `Operators.FarFieldConditions.g_far`. This merges the two Q5
"gated" items (`gradient_drive` and `far_field`) into one physical object.

## 2. The normalized far-field (cleared, no frame convention)

Far away `p_φ → ψ_s x` (the orbit-width term `Iv_∥/ω_ci = O(ρ̂_θi)` is a small
shift at `|x| = L_x`), and `n'/n = L̂_{n0}^{-1}/ψ_s`, so `p_φ(n'/n) → x L̂_{n0}^{-1}`.
The Maxwellian `F_{Mi} ∝ e^{-v̂^2} = e^{-E}` is carried by the energy-grid measure
(Gauss–Laguerre, `04 §1`; `velocity_moment!` sums against `e^{-E}`), so the code's
`g` is the distribution with that factor stripped. Hence

```math
\boxed{\;
g_{\rm far}(x{=}\pm L_x,\,\xi,\,y,\,E,\,\sigma)
   \;=\; x\;\hat L_{n0}^{-1}\;\big[\,1 + (E - \tfrac32)\,\eta_i\,\big]
\;}
```

\noindent
— linear in the boundary `x`, the temperature correction through `E = v̂²`,
independent of `ξ, y, σ` at leading order (the finite-orbit-width `σ`-dependence
is an interior effect near `x = 0`, not the far field). `L̂_{n0}^{-1} = inv_Ln0`
(existing input); `η_i = eta_i` is the only new parameter — a standard scenario
ratio (`= L_n/L_T`), **not a gated coefficient**. At Level 0 `ω_E = 0`, the
far-field potential is `Φ̂_far = 0`.

## 3. Cross-check table

| Source | Statement | Agrees with `[DERIVED]`? |
|---|---|---|
| I19 Eq. (29) (first-hand p. 5) | `Ḡ₀ = p_φ(ω_si^T/ω_si)(n'/n)F_Mi + h̄₀` | ✅ (§1) — **ω_si^T/ω_si**, not ω_ci |
| I19 Eq. (32) (first-hand p. 6) | master equation **homogeneous** in `Ḡ₀` | ✅ ⇒ drive is a BC, not a source (§1) |
| Maxwellian derivative | `F'_Mi = (n'/n)(1+(v̂²−3/2)η_i)F_Mi` | ✅ ⇒ `g_drive = p_φ F'_Mi` (§1) |
| design docs/01 §3 far-field spec | `g → neoclassical (no-island) solution` | ✅ = `g_drive` |

**Triage:** no discrepancy. The drive is the neoclassical far field `p_φ F'_Mi`
and the interior source is zero (I19 Formulation A). The only new input is `η_i`;
no frame convention enters (the corrected reading of Eq. 29).

## 4. What sign-off authorizes — now implemented

On sign-off (recorded in docs/01 §2/§4): (i) `Operators.GradientDrive` source set
to zero (I19 Formulation A); (ii) the far-field
`Operators.FarFieldConditions.g_far = x·L̂_{n0}⁻¹·[1+(E−3/2)η_i]` (and `Φ̂_far = 0`)
built by `Configure.gradient_far_field` from `inv_Ln0` and the new `eta_i`. This
un-gates the `gradient_drive` **and** `far_field` families. The `E×B` coupling,
collision magnitude `⟨ν̂_ii⟩_u`, and orbit-averaged pitch measure remain gated.
