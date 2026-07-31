# Reference — Redl (2021) bootstrap / neoclassical coefficients (B1 ground truth)

**Status:** `[CHECKED: Redl et al., PoP 28, 022502 (2021), Eqs. 10–21; transcribed from
OpenFUSIONToolkit / TokaMaker `src/python/OpenFUSIONToolkit/TokaMaker/bootstrap.py`
`redl_bootstrap()` (lines 576–795), fetched 2026-07-31]`. This is the **B1 comparison
ground truth**, not Islands physics — it is the external neoclassical bootstrap model the
Islands no-island limit must reproduce (docs/05 §B1; `notes/b1-bootstrap-plan.md`). Redl
(2021) is the modern, NEO-validated update to Sauter–Angioni (fixes the high-`ν_★`/impurity
inaccuracies). Port these to Julia and diff against TokaMaker outputs for an executable
cross-check.

Inputs (as coded): `Te, Ti` [eV]; `ne, ni` [m⁻³]; `pe, pi` [Pa]; `Zeff`; `R` [m]; `q`;
`eps = ε` (inverse aspect ratio); `fT` (trapped fraction); `I_psi = I(ψ) = R B_φ`;
gradients `dX/dψ`; `lnΛ_e, lnΛ_ii`.

## 1. Collisionalities (⚠ CONVENTION — the B1 alignment risk)
```
ν_e★ = 6.921e-18 · q·R·ne·Zeff·lnΛ_e  / (Te² · ε^{1.5})          # Te in eV, ne in m⁻³
ν_i★ = 4.90e-18  · q·R·ni·Zeff⁴·lnΛ_ii / (Ti² · ε^{1.5})         # 'Zeff' ion model (default)
```
(ion models: `'Zeff'`→`Zeff⁴`, `'Koh'`→`Zeff`, `'unity'`→`1`.) **These are the Redl/Sauter
`ν★` definitions — they are NOT identical to the Islands `ν_★ = ν_jj Rq/(ε^{3/2} v_th)`
(L23 Eq. 2.3.40).** Phase 0.2 MUST pin the exact numeric map (√2 / species / lnΛ factors)
before any `ν★`-axis comparison; a mismatch silently ruins the benchmark. The `6.921e-18`
and `4.90e-18` bake in the physical constants + a specific `lnΛ`/thermal-speed convention.

## 2. L31 — bootstrap response to the density gradient (Redl Eqs. 10–11)
```
X31 = fT / (1 + d1 + d2)
  d1 = 0.67·(1 − 0.7·fT)·√ν_e★ / (0.56 + 0.44·Zeff)
  d2 = (0.52 + 0.086·√ν_e★)·(1 + 0.87·fT)·ν_e★ / (1 + 1.13·√(max(Zeff−1,0)))
F31(X) = (1 + 0.15/dZ)·X − (0.22/dZ)·X² + (0.01/dZ)·X³ + (0.06/dZ)·X⁴,   dZ = Zeff^{1.2} − 0.71
L31 = F31(X31)
```

## 3. L34 (Redl Eq. 19)
`L34 = L31` (default). Legacy form: `L34 = F31(f33teff)` with
`f33teff = fT/(1 + 0.25(1−0.7fT)√ν_e★(1+0.45√(max(Zeff−1,0))) + 0.61(1−0.41fT)ν_e★/√Zeff)`.

## 4. L32 (Redl Eqs. 12–16) = F32_ee + F32_ei
```
X32_ee = fT / (1 + dee2 + dee3)
  dee2 = 0.23·(1 − 0.96·fT)·√ν_e★ / √Zeff
  dee3 = 0.13·(1 − 0.38·fT)·ν_e★/Zeff² · [ √(1 + 2√(max(Zeff−1,0)))
                                          + fT²·√((0.075 + 0.25(Zeff−1)²)·ν_e★) ]
F32_ee = (0.1+0.6Zeff)/(Zeff(0.77+0.63(1+(Zeff−1)^{1.1})))·(X − X⁴)
       + 0.7/(1+0.2Zeff)·(X² − X⁴ − 1.2(X³ − X⁴))
       + 1.3/(1+0.5Zeff)·X⁴                                  # X = X32_ee

X32_ei = fT / (1 + dei2 + dei3)
  dei2 = 0.87·(1 + 0.39·fT)·√ν_e★ / (1 + 2.95·(Zeff−1)²)
  dei3 = 1.53·(1 − 0.37·fT)·ν_e★·(2 + 0.375·(Zeff−1))
F32_ei = −(0.4+1.93Zeff)/(Zeff(0.8+0.6Zeff))·(X − X⁴)
       + 5.5/(1.5+2Zeff)·(X² − X⁴ − 0.8(X³ − X⁴))
       − 1.3/(1+0.5Zeff)·X⁴                                  # X = X32_ei
L32 = F32_ee + F32_ei
```

## 5. α — ion temperature-gradient term (Redl Eqs. 20–21)
```
α0 = −(0.62 + 0.055(Zeff−1))/(0.53 + 0.17(Zeff−1)) · (1−fT)
     / (1 − (0.31 − 0.065(Zeff−1))·fT − 0.25·fT²)
α  = [ (α0 + 0.7·Zeff·√fT·√ν_i★)/(1 + 0.18·√ν_i★) − 0.002·ν_i★²·fT⁶ ]
     / (1 + 0.004·ν_i★²·fT⁶)
```

## 6. Bootstrap current assembly
`R_pe = pe/(pe+pi)`, `p = pe+pi`. Default `'jB'` form (⟨j·B⟩-like):
```
j_bs = −I(ψ) · [ p·L31·(1/ne)·dne/dψ                       (density-gradient term)
               + pe·(L31 + L32)·(1/Te)·dTe/dψ              (electron-T term)
               + pi·(L31 + L34·α)·(1/Ti)·dTi/dψ ]          (ion-T term)
```
Alt `'jboot1'` form: `j_bs = −I·pe·(L31·(dp/dψ)/pe + L32·(dTe/dψ)/Te + L34·α·(1−R_pe)/R_pe·(dTi/dψ)/Ti)`.

## 7. Trapped fraction & lnΛ (needed to close the comparison)
- `fT` from TokaMaker `mygs.sauter_fc` (Sauter trapped-fraction; port or take its output).
  Islands uses `Coefficients.passing_fraction(ε)=1−1.4624√ε` ⇒ `fT = 1.4624√ε` (large-aspect
  circular). **Confirm these agree in the circular limit** (Phase 1.1); Sauter `fc` is the
  general-geometry version.
- `lnΛ` models in `calculate_ln_lambda` (`'sauter'`, `'NRL'`) — pin which, to match `ν★`.

## 8. How B1 uses this
- **Phase 2 (analytic electron `L31`, no solve):** the Islands no-island electron flow
  should reproduce `L31` (the dominant, density-gradient bootstrap piece) as a function of
  `(fT, ν_e★, Zeff)`. Diff the Islands analytic `L31` against `F31(X31)` here.
- **Phase 4 (full `J_bs(ν★)`):** assemble `j_bs` (§6) from the Islands-solved flows and diff
  vs this reference across `ν★` and `ε` (→ `fT`).
- **Executable cross-check:** port §§1–6 to a small Julia module
  (`benchmarks/islands/redl_reference.jl`, not committed to `src/`) and diff vs the
  TokaMaker Python on shared `(fT, ν★, Zeff)` grids to confirm the transcription.

Provenance: transcribed verbatim from TokaMaker `bootstrap.py:576–795` (Redl 2021) on
2026-07-31. No Islands physics here; this is the B1 comparison target.
