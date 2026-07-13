# Derivation — the collision magnitude `ν_★` normalization and the momentum-restoring velocity average `⟨ν̂_ii⟩_u`

**Provenance:** `[DERIVED: 2026-07-12]` — independent re-derivation (Decision D7)
of the two remaining pieces of the Level-0 collision magnitude, completing the
signed-off collision-operator derivation (`collision-operator.md` §7, deferred).
**Clears (on sign-off):**
1. the collision-coefficient magnitude `ν̂_jj = ε^{3/2}ν_★ ν̃_jj(v̂)`, i.e. the
   scalar `nu_tilde = ε^{3/2}ν_★` that scales the cleared deflection-frequency
   shape (`[CHECKED: L23 Eq. 2.3.40]`, the ν_★ normalization already signed off in
   `collision-operator.md` §4);
2. the momentum-restoring velocity average
   `⟨ν̂_ii⟩_u = (4ε^{3/2}ν_★/3√π)(√2 − ln(1+√2))`
   (`[CHECKED: L23 Eq. 4.1.6, p. 88]`, QUESTIONS Q3/Q5 — the deferred sub-constant).
**Status:** ✅ **signed off 2026-07-12** — implemented as
`Coefficients.momentum_restoring_average` (the `⟨ν̂_ii⟩_u` constant) and the
`nu_tilde = ε^{3/2}ν_★` wiring in `Configure` (from the new `Level0Physics.nu_star`
scenario field); the L23 `1.267537×10⁻⁴` reproduction is verified in
`test/runtests_islands_configure.jl`.

## 1. What these two quantities are

The Level-0 collision operator (`collision-operator.md`, signed off) has two
pieces (I19 Eq. 9): the **pitch-angle scattering** operator with diffusivity
`P(λ)=λ√(1−λB)` and velocity dependence `ν_{jj}(v̂)=ν̃_jj·[φ−G]/v̂³`, and the
**momentum-restoring** term that returns the parallel momentum scattering
removes.
Two magnitudes remain uncleared:

- The **overall collision magnitude** multiplying the deflection-frequency shape.
  In the dimensionless DKE it is `ν̂_jj = ε^{3/2}ν_★ ν̃_jj(v̂)` (§4 of the
  signed-off collision derivation), so the scalar that the code calls `nu_tilde`
  (scaling `Coefficients.deflection_frequency`) is `nu_tilde = ε^{3/2}ν_★`.
- The **momentum-restoring velocity average** `⟨ν̂_ii⟩_u`. The restoring flow
  `ū_∥i ∝ (1/n⟨ν_ii⟩_v)∫d³v ν_ii v_∥ f` (I19 Eq. 12) carries the speed-averaged
  collision frequency `⟨ν̂_ii⟩_u` in its normalization. Because `ν̃(v̂)` diverges
  as `v̂→0` (`∝ v̂⁻²` Chandrasekhar, `collision-operator.md` §3), this average is
  the constant L23 evaluates analytically rather than by the coarse `~20`-point
  speed quadrature (L23 §4.1).

Part 1 is a direct application of the **already-signed-off** ν_★ normalization
(§2 below); part 2 is the genuinely new derivation needing L23's reduced
integrand (§3–§5).

## 2. The collision magnitude `nu_tilde = ε^{3/2}ν_★`

The banana-regime collisionality (`collision-operator.md` §4, signed off; L23
Eq. 2.3.40) is `ν_★ = ν_{jj}Rq/(ε^{3/2}v_th)`, and the dimensionless DKE carries
the collision coefficient

```math
\hat\nu_{jj}(\hat v) = \varepsilon^{3/2}\,\nu_\star\,\tilde\nu_{jj}(\hat v),
\qquad \tilde\nu_{jj}(\hat v)=\frac{\phi(\hat v)-G(\hat v)}{\hat v^{3}} ,
```

\noindent
so the scalar multiplying the cleared `Coefficients.deflection_frequency` shape
`ν̃_jj(v̂)` is simply

```math
\boxed{\ \texttt{nu\_tilde} = \varepsilon^{3/2}\,\nu_\star\ }
```

with `ε` a cleared `Level0Physics` field and `ν_★` a **scenario scan input**
(Decision D7/Q2: ω_E and ν_★ are scanned from day one, banana regime `ν_★ ≪ 1`).
No new physics is asserted here — this only wires the signed-off §4 normalization,
replacing the supplied placeholder `nu_tilde`.

## 3. The momentum-restoring average — setup

The normalized speed-average operator (I19 Eq. 13 / L23 Eq. 2.2.20 / Eq. 4.1.5)
is the Maxwellian `d³v` speed moment,

```math
\langle\cdots\rangle_u = \frac{8}{3\sqrt\pi}\int_0^\infty u^4 e^{-u^2}\,(\cdots)\,du,
\qquad u=\hat v ,
```

normalized so that `⟨1⟩_u = (8/3√π)∫₀^∞ u⁴e^{−u²}du = (8/3√π)(3√π/8) = 1`
(a genuine average, verified §5).
Applied to `ν̂_ii(u) = ε^{3/2}ν_★ ν̃_jj(u)`,

```math
\langle\hat\nu_{ii}\rangle_u
 = \varepsilon^{3/2}\nu_\star\,\frac{8}{3\sqrt\pi}\int_0^\infty u^4 e^{-u^2}\,\tilde\nu_{jj}(u)\,du .
```

## 4. Reduce `u⁴ν̃_jj` and integrate

With `φ = erf` and `G(u) = [erf(u) − u\,erf'(u)]/(2u²)`, `erf'(u)=(2/√π)e^{−u²}`,
the shape times `u⁴` collapses to three elementary pieces:

```math
u^4\,\tilde\nu_{jj}(u) = u^4\,\frac{\phi-G}{u^3}
 = u\,\mathrm{erf}(u) \;-\; \frac{\mathrm{erf}(u)}{2u} \;+\; \frac{e^{-u^2}}{\sqrt\pi}
```

\noindent
(the last term from `u\,erf'(u)/2 = e^{-u^2}/\sqrt\pi`; verified pointwise, §5).
The speed average therefore needs three standard integrals (L23 Sec. 8.1),

```math
\int_0^\infty u\,e^{-u^2}\mathrm{erf}(u)\,du = \frac1{2\sqrt2},\quad
\int_0^\infty e^{-2u^2}\,du = \frac12\sqrt{\frac\pi2},\quad
\int_0^\infty \frac{e^{-u^2}\mathrm{erf}(u)}{u}\,du = \ln(1+\sqrt2),
```

\noindent
the last being `arcsinh(1) = ln(1+√2)`. Combining (the `e^{−u²}/√π` term picks up
`(1/√π)·½√(π/2) = 1/(2√2)`),

```math
\langle\hat\nu_{ii}\rangle_u
 = \varepsilon^{3/2}\nu_\star\,\frac{8}{3\sqrt\pi}
   \Big[\underbrace{\tfrac1{2\sqrt2}}_{u\,\mathrm{erf}}
        +\underbrace{\tfrac1{2\sqrt2}}_{e^{-u^2}/\sqrt\pi}
        -\underbrace{\tfrac12\ln(1+\sqrt2)}_{\mathrm{erf}/2u}\Big]
 = \varepsilon^{3/2}\nu_\star\,\frac{8}{3\sqrt\pi}\cdot\frac12\Big[\sqrt2-\ln(1+\sqrt2)\Big],
```

\noindent
using `2/\sqrt2 = \sqrt2`, hence

```math
\boxed{\ \langle\hat\nu_{ii}\rangle_u
 = \frac{4\,\varepsilon^{3/2}\nu_\star}{3\sqrt\pi}\big(\sqrt2-\ln(1+\sqrt2)\big)
 \;\approx\; 0.40083\,\varepsilon^{3/2}\nu_\star\ }
```

exactly L23 Eq. 4.1.6. The pure number `(4/3√π)(√2−ln(1+√2)) = 0.400830…`.

## 5. Numerical verification (independent, this repo)

Evaluated with `QuadGK`/`SpecialFunctions` (scratch check, `nu_verify.jl`):

| Quantity | Derived closed form | Numeric | Match |
|---|---|---|---|
| `∫u e^{−u²}erf du` | `1/(2√2) = 0.3535534` | `0.3535534` | ✅ 15 digits |
| `∫e^{−2u²}du` | `½√(π/2) = 0.6266571` | `0.6266571` | ✅ 15 digits |
| `∫u⁻¹e^{−u²}erf du` | `ln(1+√2) = 0.8813736` | `0.8813736` | ✅ 15 digits |
| `⟨1⟩_u` | `1` (normalized) | `1.0000000` | ✅ |
| `u⁴ν̃_jj` reduction | `u erf − erf/2u + e^{−u²}/√π` | agrees at `u=0.3,1,2.5` | ✅ 15 digits |
| `(4/3√π)(√2−ln(1+√2))` | `0.4008304` | `0.4008304` (direct ∫u⁴e^{−u²}ν̃) | ✅ 15 digits |
| `⟨ν̂_ii⟩_u` at `ε=0.1, ν_★=0.01` | `1.267537×10⁻⁴` | `1.2675369×10⁻⁴` | ✅ matches L23's 7-digit `1.267537E-4` |

The direct speed integral of `u⁴e^{−u²}ν̃_jj(u)` (using the cleared `[φ−G]/v̂³`
shape) equals the boxed closed form to 15 digits, and the full value reproduces
L23's own unit-test number to all quoted figures.

## 6. Cross-check table

| Source | Form | Agrees with `[DERIVED]`? |
|---|---|---|
| L23 Eq. 2.3.40 (`collision-operator.md` §4, signed off) | `ν̂_jj = ε^{3/2}ν_★ν̃_jj` ⇒ `nu_tilde = ε^{3/2}ν_★` | ✅ (§2) |
| L23 Eq. 4.1.5 (first-hand, p. 87) | `⟨·⟩_u = (8/3√π)∫u⁴e^{−u²}` | ✅ + normalized `⟨1⟩_u=1` (§3, §5) |
| L23 Eq. 4.1.4 (first-hand, p. 87) | `ν̃_jj = erf/u³ + e^{−u²}/√π u⁴ − erf/2u⁵` | ✅ `u⁴ν̃` reduction (§4, §5) |
| L23 Sec. 8.1 standard integrals | `1/2√2`, `½√(π/2)`, `ln(1+√2)` | ✅ all 15-digit (§5) |
| L23 Eq. 4.1.6 (first-hand, p. 88) | `⟨ν̂_ii⟩_u = (4ε^{3/2}ν_★/3√π)(√2−ln(1+√2))` | ✅ **exact** + reproduces L23's `1.267537e-4` |

**Triage:** no discrepancy. The ν_★ normalization is the already-signed-off §4;
the `⟨ν̂_ii⟩_u` constant is derived here first-hand from L23's reduced integrand
and its three standard integrals, each verified numerically, reproducing L23's
own unit-test value to 7 digits.

## 7. What sign-off authorizes

On sign-off (recorded in docs/01 §2.3):

1. Add `ν_★` (`nu_star`) as a `Configure.Level0Physics` **scenario** field
   (banana-regime collisionality, scanned per D7).
2. `Configure` builds `nu_tilde = ε^{3/2}·nu_star` from cleared physics and wires
   it into `collision_coefficient`, removing `nu_tilde` from `GatedLevel0Inputs`
   (moves `:nu_tilde` gated→cleared). This un-gates the collision operator's
   magnitude.
3. A new cleared `Coefficients.momentum_restoring_average(; epsilon, nu_star)`
   returning `⟨ν̂_ii⟩_u = (4ε^{3/2}ν_★/3√π)(√2−ln(1+√2))`, with a test asserting
   the `1.267537×10⁻⁴` reproduction. This clears the deferred sub-constant
   (`collision-operator.md` §7). The momentum-restoring **operator term** is a
   separate future addition (not in the Level-0 stack yet); this clears its
   magnitude constant so it is ready when the term lands.

This closes the **collision magnitude** family (QUESTIONS Q5). Only the
orbit-averaged pitch measure `B_profile` and the Hirshman–Sigmar `k ≃ −1.173`
(a separate parallel-viscosity moment problem, L23 Eq. 4.1.7) remain.
