# Derivation — physical equilibrium → normalized Level-0 vector

**Status:** `[DERIVED: 2026-07-23]` — awaiting human sign-off. Implements
`Configure.physical_scenario` (design 10). Turns SI physical quantities at the
rational surface into the self-consistent normalized `Level0Physics` vector, so the
inputs are **derived from one `T_i(ψ)`, `n_i(ψ)`, `B`, geometry** instead of hand-set
(the parameter-audit gap, `notes/physical-parameter-audit.md`).

All formulas below are **standard plasma-physics textbook relations** (Larmor radius,
banana collisionality) — *not* the disputed drift-kinetic island coefficients the
`[VERIFY]` policy guards. Provenance is cited; the two convention-sensitive items
(the `v_th` factor and the `ν_ii` prefactor matching L23's `ν_jj`) are the sign-off
targets.

## Inputs (SI, at the rational surface `ψ_s` where `q = m/n`)

`R₀` (major radius, m), `r_s` (minor radius of the surface, m), `B` (field magnitude
at the surface, T), `q_s = q(ψ_s)`, `dq/dψ|_s`, `T_i` (ion temperature, J, `= e·T_i`
with `T_i` in eV), `n_i` (density, m⁻³), `dln T_i/dψ|_s`, `dln n_i/dψ|_s`, `ψ_s`
(signed flux from axis, Wb/rad), `Z`, ion mass `m_i` (kg). From `setup_equilibrium`
(`ro`, `bt0`, `amean`, `q_spline`) and `read_kinetic_file` (`T_i`, `n_i`).

## 1. Geometry and gradient normalizations (unambiguous, ψ-based)

Directly from `docs/01 §5` (`x=(ψ−ψ_s)/ψ_s`, `L̂_q⁻¹=(ψ_s/q)dq/dψ`, etc.) — pure
ratios of equilibrium/profile splines, no physical constants:

```math
\varepsilon = r_s/R_0,\quad
\hat L_q^{-1} = \frac{\psi_s}{q_s}\frac{dq}{d\psi}\Big|_s,\quad
\hat L_{n0}^{-1} = \frac{\psi_s}{n_i}\frac{dn_i}{d\psi}\Big|_s = \psi_s\,\frac{d\ln n_i}{d\psi}\Big|_s,
```
```math
\eta_i = \frac{L_{n}}{L_{T_i}} = \frac{d\ln T_i/d\psi}{d\ln n_i/d\psi}\Big|_s,\quad
q_s,\ \frac{dq}{d\psi}\Big|_s\ \text{as given}.
```

\noindent
`ψ̃` and the `Δ`-prefactor use the already-cleared `Coefficients.delta_moment_prefactors`
with `mu0_R = μ₀ R₀` and the scan `w_ψ`.

## 2. Ion gyroradius (standard)

Thermal speed (matched to `docs/01 §5` `v̂ = v/v_thi` with `F_M = e^{-E}`, `E = v̂²`,
so `m_i v²/2 = E T_i` ⇒ `v_th² = 2T_i/m_i` — **the factor-2 convention is a sign-off
item**):

```math
v_{th,i} = \sqrt{2T_i/m_i}.
```

Ion Larmor radius and **poloidal** Larmor radius (large-aspect circular,
`B_θ/B ≈ ε/q` from `q = r B_φ/(R B_θ)`, `B_φ ≈ B`):

```math
\rho_i = \frac{m_i v_{th,i}}{Z e B},\qquad
\rho_{\theta i} = \rho_i\,\frac{B}{B_\theta} = \rho_i\,\frac{q_s}{\varepsilon},\qquad
\boxed{\ \hat\rho_{\theta i} = \rho_{\theta i}/r_s\ } .
```

## 3. Collisionality (banana `ν_★`, `docs/01 §2.3`)

`docs/01 §2.3` `[CHECKED: L23 Eq. (2.3.40)]`: `ν_★ = ν_{jj} R q/(ε^{3/2} v_th)`.
The ion self-collision frequency `ν_{ii}` from the NRL Plasma Formulary
(Braginskii); `T_i` in eV, `n_i` in m⁻³, SI:

```math
\ln\Lambda_{ii} = 23 - \ln\!\Big(\frac{Z^3 \sqrt{2 n_i[\mathrm{cm^{-3}}]}}{T_i[\mathrm{eV}]^{3/2}}\Big),\qquad
\nu_{ii} = \frac{n_i Z^4 e^4 \ln\Lambda_{ii}}{12\,\pi^{3/2}\,\varepsilon_0^2\,\sqrt{m_i}\,T_i^{3/2}}\ \ [\mathrm{s^{-1}}],
```
```math
\boxed{\ \nu_\star = \frac{\nu_{ii}\,R_0\,q_s}{\varepsilon^{3/2}\,v_{th,i}}\ } .
```

\noindent
**Sign-off items**: (i) the `ν_{ii}` prefactor/`lnΛ` convention must match L23's
`ν_{jj}` in Eq. (2.3.40) (Braginskii vs. NRL differ by O(1) factors — the physics
must be the *same* `ν_{jj}` the `ν_★` normalization assumes); (ii) the `v_th` factor
of §2. Both are checked against L23 by the physics-verifier + human sign-off; the
`[DERIVED]` tag gates use until then.

## 4. Sanity targets (the a10 case)

`examples/a10_kinetic_example` (large aspect, `q=2`, `n_i = 2×10^{18}` m⁻³,
`T_i ≈ 0.48` keV, `B₀` and `R₀` from its g-file): expect `ε ≈ 0.1`,
`ν_★ ~ 10^{-2}` (banana, York's regime), `ρ̂_θi ~ 10^{-2}`, `η_i` from the profile.
(`ν_★ ∝ R₀`, so the `R₀ ≈ 1` a10 g-file gives `~10^{-2}`; the unit test forces
`ε=0.1` with a synthetic `R₀=10`, which scales `ν_★` up to `~0.3` — same formula,
same banana regime, just a larger `R₀`.)
These are the audit's "physical, self-consistent" replacements for the hand-set
`_phys`/`_b5_phys` literals; the scenario records them in a docs/09 manifest.

## 5. Scope

Level-0 needs the scalar vector at `ψ_s`; the full `ψ`-profile scan is a *fixed
equilibrium* with `w` (island half-width) the scan variable (design 10). The far-field
matching radius is `Configure.physical_domain` (a fraction of `r_s`, independent of
`w`), never a plasma-scale box.
