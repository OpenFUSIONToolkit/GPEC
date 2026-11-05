# physics.md — orbit-averaged kinetic energy and NTV torque

**Status: v0, unsigned.** Written at hand-off (2026-11-05) without Zotero or
the PDFs; every item marked `[VERIFY]` is a statement Nik confirms, corrects
or strikes in the M0 sign-off card. Nothing after M0 starts while a
`[VERIFY]` is open. Once signed, this file changes only through a Decisions
card.

Equations are transcribed in the code's conventions where the code has one
(see `code_map.md`): ψ is the normalized poloidal flux ψ_N ∈ [0, 1] with
`chi1 = 2π·psio` converting to physical flux per radian; θ is the
straight-field-line angle on [0, 1); `bo` is the axis toroidal field
|F(0)|/(2π R₀); λ = μ B₀/E with B₀ = `bo`; x = E/T.

## 1. What the code computes now (zero orbit width)

The kinetic energy of species s for toroidal mode number n is, in the
action-angle form of Park 2011 as implemented in Logan 2013 Eq. 8 and 19
[VERIFY: equation numbers],

T = T_φ + 2in δW_K,
T = tnorm ∫dψ Σ_ℓ ∫dλ |δJ_ℓ(ψ, λ)|² ∫₀^∞ dx N(x) e^{−x} / [iΩ(x) − ν(x)],

with

- `tnorm = (−2n²/√π)(R₀/B₀) n_s T_s (chi1/2π)` — `Torque.jl:266`,
- Ω(x) = ℓ_eff ω_b √x + n(ω_E + ω_D x), ℓ_eff = ℓ + σ n q (σ = 1 passing, 0 trapped) — `EnergyIntegration.jl:122`, `PitchIntegration.jl`,
- N(x) = (ω_E + ω_*n + ω_*T (x − 3/2)) x^{5/2} for a Maxwellian — `_energy_numerator`,
- ω_*n = −2π T_s n_s′/(Z e chi1 n_s), ω_*T = −2π T_s′/(Z e chi1) — `Torque.jl:452`,
- ν(x) the Krook-harmonic collision model, ν_k(1 + ℓ²/4)/x^{3/2}, trapped divided by 2ε — `EnergyIntegration.jl:51`,
- |δJ_ℓ|² = `dJdJ` = ω̄_b |∮ dθ J B (∇·ξ_⊥ √(v_∥/v) + (δB/B)(1 − 3λB/2B₀)/√(v_∥/v)) e^{−i2πnq(θ−θ₀)} e^{−i2πℓ_eff s(θ)}|² / (2R₀²), with s the fraction of the bounce period elapsed — `BounceAveraging.jl:1054, 1108, 1119`,
- ω_b = 2π/[(2−σ) ∮ J B dθ/√(v_∥/v)] · v_th/R₀, ω_D the bounce average of the ∇B and curvature drift at thermal energy — `BounceAveraging.jl:355–361`.

Three properties of this form matter for what follows.

(a) The orbit is the flux surface ψ: δB/B and ∇·ξ_⊥ are read at (ψ, θ) only. The energy dependence of the frequencies is analytic (ω_b ∝ √x, ω_D ∝ x), which is what allows the energy integral to be done once per (ψ, λ, ℓ) with the resonance pole located in closed form (`find_resonance_energies`).

(b) The drive is a ψ-derivative on the surface: N(x) comes from ∂f_M/∂ψ at fixed E, with ω_E entering both the drive and the precession.

(c) The torque identity T_φ = Re T, δW_K = Im T/(2n) is a property of the action-angle form and does not depend on the orbit being surface-local [VERIFY: Park 2009 states this for general orbits, or only for the surface-local form].

The code also suppresses the kinetic terms inside ψ_c, where a thermal orbit width reaches ⟨r⟩ (`axis_validity_suppression`). The new path has no such limit; the envelope stays on for the old path and off for the new one.

## 2. The orbit in constants of motion

Guiding-center motion in the axisymmetric equilibrium conserves E (kinetic energy; the electrostatic potential is handled in §4), μ = m v_⊥²/(2B), and the canonical toroidal momentum

P_φ = m R v_∥ B_φ/B − Z e ψ_p,

where ψ_p = `psio`·ψ is the poloidal flux per radian [VERIFY: sign of the Ze ψ term with the code's ψ increasing outward and `ip_sign`, `bt_sign`; White & Chance 1984 Eq. for P_ζ in flux coordinates; the form m R v_∥ B_φ/B = m v_∥ F/B with F = R B_φ = `F_spline/2π`].

For given (E, μ, P_φ) and sign of v_∥, the orbit is the curve ψ(θ) solving

P_φ = m v_∥(ψ, θ) F(ψ)/B(ψ, θ) − Z e psio ψ,   v_∥ = ± √(2(E − μ B(ψ, θ))/m),

a scalar root in ψ at each θ. Bounce points are where E = μ B on the orbit (v_∥ = 0). The orbit is

- **trapped** if v_∥ changes sign along it (two bounce points, both legs),
- **passing** if v_∥ keeps its sign over θ ∈ [0, 1),
- **lost** if ψ(θ) reaches `psilim` (the last closed surface) before closing.

Potato and stagnation orbits near the axis are classified by the same rule; their cost is a Decisions card if they dominate the grid.

Time along the orbit: dt = dθ / θ̇ with θ̇ = v_∥ B^θ/B + v_d·∇θ, where B^θ = chi1/(2π J) [VERIFY: the code's J = `rzphi_jac` is the Jacobian of (ψ_N, θ, ζ) with θ on [0,1); check the 2π factors] and v_d the ∇B + curvature drift

v_d = (m/(Z e B)) (v_∥² + v_⊥²/2) (b × ∇B)/B  (low β; the curvature term equals the ∇B term in a force-free sense [VERIFY: keep the ∇p correction or not — the existing `g_wd` integrand uses dB/dψ and dJ/dψ, which is the low-β form]).

Toroidal angle along the orbit: dζ/dt = v_∥ B^ζ/B + v_d·∇ζ, with B^ζ = q B^θ in SFL coordinates.

Definitions on the closed orbit (period τ, the full bounce for trapped, one transit for passing):

- ω_b = 2π/τ,
- ⟨ζ̇⟩ = Δζ/τ, the secular toroidal advance per period (this includes the parallel advance 2πq per transit for passing particles and the drift precession for both),
- ζ_osc(t) = ζ(t) − ζ(0) − ⟨ζ̇⟩ t, the oscillatory part,
- ⟨ψ⟩ = (1/τ) ∮ ψ dt, the orbit-averaged flux label,
- the lost flag.

Zero-orbit-width limit: ψ(θ) → ψ, ω_b → the code's `wb`, ⟨ζ̇⟩ → σ q ω_b + ω_D (so that n⟨ζ̇⟩ + ℓ ω_b = ℓ_eff ω_b + n ω_D), ζ_osc → 2πq(θ − θ₀) − 2π σ q s. These are the M1 and M2 checks.

Analytic reference for the frequencies: Antonenas, Anastassiou, Kominis, J. Plasma Phys. 87, 855870101 (2021), zero-drift-width bounce and transit frequencies in terms of elliptic integrals of the trapping parameter [VERIFY: which of their equations; §3 of the paper].

## 3. The orbit-averaged action variation

The perturbed field is first order along the unperturbed orbit (δB from `PerturbedEquilibrium`, including the plasma response). The action variation for bounce harmonic ℓ is the time integral along the orbit,

δJ_ℓ = ∮ dt  A(t) exp(−i n ζ_osc(t)) exp(−i ℓ ω_b t),
A(t) = (v_∥ ∇·ξ_⊥ + (δB/B)(2E − 3μB)/(2 m v_∥)) evaluated at (ψ(t), θ(t)) [VERIFY: this is the time-domain form of the existing `jvtheta` integrand; the ZOW integrand J B (divx √vpar + dbob (1 − 3λB/2B₀)/√vpar) dθ equals A dt up to the constant v and the Jacobian; confirm the factor and the sign of the ∇·ξ_⊥ term against Park 2011 Eq. for δJ].

The modes δB/B and ∇·ξ_⊥ are the code's `dbob_m(ψ)` and `divx_m(ψ)` summed over poloidal mode numbers at θ(t), with ψ(t) varying along the orbit; this is the only place the new path reads a field, and it reads exactly what the old path reads, at a different ψ.

The resonance operator is unchanged in form with the orbit's own frequencies:

iΩ − ν,   Ω = ℓ ω_b(E, μ, P_φ) + n ⟨ζ̇⟩(E, μ, P_φ).

Consequence for the energy integral: Ω no longer has the analytic √x, x dependence, so the pole-location-and-residue machinery in `EnergyIntegration.jl` cannot be reused as is. The plan (M2) is to tabulate ω_b, ⟨ζ̇⟩ and |δJ_ℓ|² on a grid in √x at fixed (μ/E, P_φ) [VERIFY: the natural fixed labels at varying E — the code's λ = μB₀/E is energy-independent for a given pitch, so a grid in (λ, P_φ, √x) keeps the ZOW structure and makes the limit test trivial], interpolate with cubic splines in √x, and reuse the Sokhotski–Plemelj add-back with the pole found by a root solve on the interpolated Ω(x). The collisionless limit then has the same analytic regular part.

Torque: T_φ = Re T and δW_K = Im T/(2n) with the same `tnorm`, the λ integral replaced by the (λ, P_φ) integral with its Jacobian [VERIFY: the phase-space measure in (E, μ, P_φ) per unit ψ; in ZOW the P_φ integral at fixed (E, μ) collapses to the ψ integral with dP_φ = −Z e psio dψ, which fixes the normalization: the new path's ∫dP_φ must reproduce `tnorm`∫dψ in the limit].

## 4. The drive and the electric potential

For a static perturbation (ω = 0) the drive in the perturbed distribution is n ∂f₀/∂P_φ at fixed (E, μ) [VERIFY: Park 2011 / Gorelenkov 1999 form: δf ∝ (ω ∂f₀/∂E − n ∂f₀/∂P_φ)/(ω − ℓω_b − n⟨ζ̇⟩ + iν); with ω → 0 only the P_φ term survives].

Equilibrium distribution: f₀ = f_M(E; n(ψ̄), T(ψ̄)) with ψ̄ a flux label of the orbit. The choice of ψ̄ is a modeling decision:

- ψ̄ = ⟨ψ⟩, the orbit-averaged flux (recommended; reduces to ψ in ZOW; makes ∂f₀/∂P_φ = (∂f_M/∂ψ̄)(∂⟨ψ⟩/∂P_φ) with ∂⟨ψ⟩/∂P_φ computed numerically on the grid and equal to −1/(Z e psio) in ZOW);
- ψ̄ = ψ at the outboard midplane crossing (what orbit codes often use).

[VERIFY: which; the first is the default in PLAN.md M3.]

The electric potential Φ(ψ) with ω_E = −(2π/chi1) Φ′/... [VERIFY: the code's `omegaE_spline` is the E×B toroidal angular frequency ω_E = −Φ′(ψ)/(ψ′ ...) — fix the exact definition from `KineticProfiles.jl`]. In the ZOW code ω_E enters as n ω_E in Ω and in N(x). In the orbit path it enters through the orbit average: the E×B toroidal rotation is part of ζ̇, so ⟨ζ̇⟩ contains ⟨ω_E(ψ(t))⟩ along the orbit rather than ω_E(ψ) on one surface. This is the mechanism by which the edge peak where ω_E → 0 is bounded: an orbit that straddles the zero sees a non-zero average. Whether E also includes Z e Φ in the orbit solve (electrostatic drift modifying the orbit) is second order in Φ/T for thermal ions and negligible for alphas; it is not included [VERIFY: agree].

Reduction: with ψ̄ = ⟨ψ⟩ and ZOW, n ∂f₀/∂P_φ reproduces the Maxwellian numerator (ω_E + ω_*n + ω_*T(x − 3/2)) x^{5/2} e^{−x} up to the constant absorbed in `tnorm` [VERIFY: by M3 check 2, numerically; and analytically once in the sign-off].

## 5. Lost orbits

Orbits that reach `psilim` are excluded from δW_K (they carry no steady perturbed distribution) and counted: the lost fraction L(ψ̄) is written to HDF5 and is the bound on the edge contribution in KR2. The alternative — a loss-cone correction to f₀ — is out of scope. [VERIFY: agree that exclusion is the right first treatment; the ZOW code has no equivalent and instead relies on the axis envelope and `psilims`.]

## 6. The slowing-down alpha distribution

Isotropic, from a fusion source S(ψ) (m⁻³ s⁻¹) at birth speed v_α (E_α = 3.5 MeV):

f_α(v, ψ) = S(ψ) τ_s(ψ) / (4π (v³ + v_c(ψ)³)),  v < v_α;  0 otherwise,

with the Spitzer slowing-down time τ_s = 3 (2π)^{3/2} ε₀² m_α T_e^{3/2} / (Z_α² e⁴ m_e^{1/2} n_e ln Λ) and the critical speed v_c = (3√π m_e / (4 m_α) Σ_i n_i Z_i² m_α/(n_e m_i))^{1/3} v_te with v_te = √(2T_e/m_e) [VERIFY: constants and the Σ_i Z_i² n_i/(n_e) · (m_α/m_i) form; these are first-principles, not fitted; source: standard (Stix 1972; Helander & Sigmar §3.6) — cite the one Nik prefers].

Derivatives (analytic): ∂f_α/∂E = (1/(m_α v)) ∂f_α/∂v with ∂f_α/∂v = −3 v² f_α/(v³ + v_c³); ∂f_α/∂ψ̄ = f_α [S′/S + τ_s′/τ_s − 3 v_c² v_c′/(v³ + v_c³)], primes being d/dψ̄. The P_φ drive follows §4 with ψ̄ = ⟨ψ⟩. The birth-energy edge is a step in f_α; the energy integral is cut at x_α = E_α/T_ref with T_ref a reference temperature chosen so the √x grid resolves v_c and v_α [VERIFY: fine with cutting at E_α rather than including the birth-edge delta-function contribution to ∂f/∂E, which does not enter a static drive].

Source profile S(ψ): from n_D n_T ⟨σv⟩_DT(T_i) on the kinetic profiles, with ⟨σv⟩ from the Bosch–Hale parameterization (fitted constants, Bosch & Hale, Nucl. Fusion 32, 611 (1992)) [VERIFY: acceptable; alternative is a user-supplied S(ψ) column in the kinetic file]. For the OFT ARC-class and SPARC cases, n_D = n_T = n_e/2 unless profiles say otherwise.

Validity: isotropic (no NBI-like anisotropy), no finite-Larmor-radius correction (ρ_α/Δ_b ~ √ε/q ≈ 0.2, so the orbit width dominates), collisional slowing on electrons and ions in the standard form, no energy diffusion. These bound what the paper may claim.

## 7. Symbols → code names

| Symbol | Meaning | Existing code | New path (proposed names) |
|---|---|---|---|
| ψ, ψ_N | normalized poloidal flux | `psi` | `psi` |
| psio, chi1 | flux per radian at edge; 2π·psio | `equil.psio`, `intr.chi1` | same |
| θ | SFL angle on [0,1) | `theta` | same |
| B(ψ,θ) | |B| | `equil.eqfun_B((psi,theta))` | same, evaluated along the orbit |
| F = R B_φ | 2πF stored | `equil.profiles.F_spline` | `F(psi) = F_spline(psi)/2π` |
| J | Jacobian of (ψ,θ,ζ) | `equil.rzphi_jac` | same |
| R(ψ,θ) | major radius | `flux_surface_metric(equil, psi, theta)` → r | same |
| B₀ | axis toroidal field | `intr.bo` | same |
| λ | μB₀/E | `lambda` | kept as the pitch label |
| σ | 1 passing, 0 trapped | `sigma` | `OrbitClass` enum: trapped / passing / lost |
| E, x | energy, E/T | `x` | `x` grid (√x-spaced), `Tref` |
| P_φ | canonical momentum | — | `pphi` |
| ψ(θ) | orbit | — | `OrbitGeometry.psi_of_theta` |
| τ, ω_b | period, bounce frequency | `total_wb`, `wb` | `OrbitGeometry.wb` |
| ⟨ζ̇⟩ | secular toroidal rate | σ q ω_b + ω_D (implicit) | `OrbitGeometry.wzeta` |
| ζ_osc(t) | oscillatory toroidal phase | `phase` | `OrbitGeometry.zeta_osc` samples |
| ⟨ψ⟩ | orbit-averaged flux | — | `OrbitGeometry.psibar` |
| lost flag | reaches psilim | — | `OrbitGeometry.lost` |
| δB/B, ∇·ξ_⊥ | perturbation modes | `intr.dbob_m`, `intr.divx_m` | same, read at ψ(t) |
| δJ_ℓ | action variation | `bj_integral` → `dJdJ` | `orbit_dJdJ` |
| Ω, ν | resonance, collisions | `_energy_integrand_real` | same operator with tabulated Ω(x) |
| f₀, ∂f₀/∂P_φ | distribution, drive | `f0type`, `_energy_numerator` | `f0type = "slowingdown"` added; `drive_pphi` |
| T_φ, δW_K | torque, energy | `MethodResult.total_torque` | same struct, method name `ogar` (orbit GAR) in `METHOD_REGISTRY` |
| L(ψ̄) | lost fraction | — | HDF5 `KineticForces/ogar/lost_fraction` |

Method name `ogar` and the HDF5 group follow the registry convention in `KineticForcesStructs.jl:13` and `docs/development/hdf5-conventions.md`; the deck key is `ogar_flag`, default `false`. [VERIFY: name.]

## 8. Validity limits of the new path

- Perturbation first order along the unperturbed orbit: no orbit modification by the 3D field, no stochastic ripple loss (Goldston–White–Boozer 1981 is a separate mechanism, named in the paper).
- Guiding-center (no FLR): ρ/Δ_b ~ √ε/q.
- Collisions: the same Krook-harmonic model as the existing path, applied with the orbit's ω_b; no collisional orbit-width broadening beyond that.
- Orbits computed in the axisymmetric equilibrium of the GPEC run; the TokaMaker field scan must keep that equilibrium's q and β (M4 check 1).
- Near the axis, potato orbits are resolved only as far as the (λ, P_φ) grid allows; the paper does not claim axis behavior.

## 9. Where sources disagree

- Kim 2013/2014 (POCA) report the torque exceeding theory by factors of 2–3 at resonances and attribute part of it to finite orbit width and passing particles; Shaing (IAEA 2016, abstract only) reports finite banana width smoothing short-scale radial variation. These are not contradictory (amplitude at a resonance versus structure across one), but the paper must not cite either as a quantitative prediction for KR2. The project's own ZOW-versus-orbit comparison is the evidence.
- Liu 2014 (first-order orbit-width expansion) finds the precession resonance unchanged at first order. KR3's exponent check tests whether the leading correction is second order in Δ_b for this path.
