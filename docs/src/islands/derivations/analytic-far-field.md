# Derivation — the analytic (drift-orbit-shifted) far-field boundary condition

**Status:** `[DERIVED: 2026-07-17]` — awaiting human sign-off.

**Clears / revisits:** the far-field boundary state built by
`Configure.gradient_far_field` for the `:analytic` mode. It **restores a term the
signed-off `gradient-drive.md` §2 explicitly dropped** (the `O(ρ̂_θi)` drift-orbit
shift), on the evidence that the dropped term is *not* negligible at the boundary —
it is the offset that makes `g_far` the true asymptote, and its absence seeds the
diagnosed resolution-sharpening boundary layer (QUESTIONS Q7). Built entirely from
**already-cleared** quantities — introduces no new coefficient.

## 1. The exact far field is the neoclassical response at the canonical momentum

The ion distribution is a function of the toroidal canonical momentum, not the flux
label (I19 Eq. 2; docs/01 §5, `[CHECKED]`):

```math
p_\phi = (\psi-\psi_s) - \frac{I v_\parallel}{\omega_{cj}} ,
```

and the neoclassical gradient drive enters as the far-field
`Ḡ₀ → p_φ F'_{Mi}` (gradient-drive.md §1, signed off). In the code's normalization
(`x = (\psi-\psi_s)/\psi_s`), this is

```math
g_{\rm far}(x) \;=\; \hat p\, \hat L_{n0}^{-1}\,[1+(E-\tfrac32)\eta_i],
\qquad \hat p \;=\; x - x_D ,
```

\noindent
where the **drift-orbit width** `x_D = x-\hat p` is the *cleared* quantity boxed in
`omega-D-drift-frequency.md` §2 (signed off, part of clearing `ω̂_D`):

```math
x_D(\theta;\,y,\hat v,\sigma) \;=\; \frac{I v_\parallel}{\omega_{cj}\,\psi_s}
   \;=\; \hat\rho_{\theta i}\,\frac{\sigma\hat v}{1+\varepsilon}\,\frac{\sqrt{1-yb}}{b} .
```

## 2. The orbit-averaged shift (what enters the far field)

The far field is the orbit-averaged distribution, so it carries the orbit-averaged
shift `⟨x_D⟩_θ`. The `θ`-average is exactly the **already-cleared** drift bracket
`A(y) = ⟨√(1-yb)/b⟩_θ` (`Coefficients.orbit_average_drift_brackets`, signed off):

```math
\boxed{\;
\langle x_D\rangle_\theta(y,E,\sigma)
   \;=\; \hat\rho_{\theta i}\,\frac{\sigma\sqrt E}{1+\varepsilon}\,A(y),
\qquad A(y)=\Big\langle \tfrac{\sqrt{1-yb}}{b}\Big\rangle_\theta \;}
```

with `\hat v = \sqrt E`. Hence the **analytic far field**

```math
g_{\rm far}^{\rm A}(x_{\rm bnd}; y,E,\sigma)
  \;=\; \big(x_{\rm bnd} - \langle x_D\rangle_\theta(y,E,\sigma)\big)\,
        \hat L_{n0}^{-1}\,[1+(E-\tfrac32)\eta_i].
```

Every factor is cleared: `ρ̂_θi = phys.rho_hat_theta_i`, `ε = phys.epsilon`,
`√E` from the energy grid, `σ` from the grid, `A(y)` from
`orbit_average_drift_brackets`, and the slope `L̂_{n0}^{-1}[1+(E-3/2)η_i]` from the
signed-off `gradient-drive.md`. **No new coefficient, sign, or normalization.**

## 3. Why this is the fix (well-posed *and* accurate)

- **Accurate.** `x·slope` (the `:dirichlet` default) is the leading `p̂≈x` term;
  `(x-⟨x_D⟩)·slope` restores the `O(ρ̂_θi)` orbit shift `gradient-drive.md` §2
  dropped. It is the true asymptote to this order, so the interior solution
  self-matches it and no boundary layer forms.
- **Well-posed.** It is a **Dirichlet** (value) condition, so it pins the far-field
  *level* — unlike the York `:neumann` slope-only form, which leaves an
  additive-constant null mode (σ-even `c(E,σ)` with zero density moment) that
  makes the Jacobian singular (the "winged" branch; QUESTIONS Q7, LOG 2026-07-17).
  The shift `⟨x_D⟩` is precisely the physical value of that otherwise-free constant.

## 4. Contrast with the drift-island shift `ρ_shift`

`⟨x_D⟩` (orbit width, `∝ ρ̂_θi σ\sqrt E A(y)`) is **not** the drift-island shift
`x_D^{\rm island} = ρ̂_θi ω̂_D L̂_q` (docs/01 §2.2 `[CHECKED: I19 Eq. 33; D21 Eq. 21;
Diss19 Eq. 2.37]`; L23 §3.1, the location of the drift-island O-point — the `ω̂_D`
coefficient is `[CLEARED]` §2.1, but the shift *structure* is `[CHECKED]`, awaiting
sign-off, QUESTIONS Q8). They are distinct finite-orbit-width effects; the far
field carries the orbit width `⟨x_D⟩`, the mesh-sizing uses `x_D^{\rm island}`. Note
the two are consistent: the shear part of `ρ̂_θi ω̂_D L̂_q` equals `⟨x_D⟩` exactly (the
`L̂_q` cancels the `1/L̂_q` inside `ω̂_D`). (An earlier draft of this note wrote
`L̂_q^{-1}` here — a typo; the CLEARED docs/01 §2.2 form `L̂_q` is authoritative.)

## 5. Scope / open

- Leading `O(ρ̂_θi)` correction only; higher-order far-field structure (and the full
  `θ`-resolved vs orbit-averaged distinction near `y_c`) is not included — validated
  numerically by the resolution-convergence + localization test, not claimed exact.
- `A(y)` is evaluated per `(y,E,σ)`; the forbidden pitch region (`y ≥ 1/b_min`) is
  pinned `g=0` as before, and the near-`y_c` bracket miss is handled gracefully
  (as in `drift_coefficient_table`).
- **Sign-off needed** (human): the physical decision to restore the term
  `gradient-drive.md` §2 dropped, and the `⟨x_D⟩_θ = ρ̂_θi(σ√E/(1+ε))A(y)` form.
