# physics.md — orbit-averaged kinetic energy and NTV torque

**Status: v1, signed (Q4 answer B, 2026-11-05).** v0 was written at hand-off;
its [VERIFY] items were resolved in the M0 card Q4 and Nik signed with two
exceptions, both resolved here: the finite-β (∇p) physics is kept (§2, §3) and
the electrostatic potential is in the orbit (§2, §4). The beam distribution of
card Q7 (answer B) is in §6. From here this file changes only through a
Decisions card.

Equations are transcribed in the code's conventions where the code has one
(see `code_map.md`): ψ is the normalized poloidal flux ψ_N ∈ [0, 1] with
`chi1 = 2π·psio` converting to physical flux per radian; θ is the
straight-field-line angle on [0, 1); `bo` is the axis toroidal field
|F(0)|/(2π R₀); λ = μ B₀/E with B₀ = `bo`; x = E/T.

## 1. What the code computes now (zero orbit width)

The kinetic energy of species s for toroidal mode number n is, in the
action-angle form of Park 2011 Eqs. 8–15 (δW_K Eq. 8/14, T_φ Eq. 9/15, δJ Eq. 10, bounce-averaged δf Eq. 12) as implemented in Logan 2013: phase-space integrand Eq. 5, bounce-averaged torque Eq. 16, normalized form Eqs. 19–20 (verified from the papers, 2026-11-05; Logan 2013 Eq. 8 is the form of δf, not the integrand),

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

(c) The torque identity T_φ = Re T, δW_K = Im T/(2n) is a property of the action-angle form for a single toroidal harmonic n: δW_K and T_φ are the same quadratic form in δJ with the real and imaginary parts of one resonance operator (Park 2011 Eqs. 8–9, 14–15), and the orbit enters only through δJ and the frequencies. The literature does not state it for finite-width orbits (Park 2009 has no statement of it); the paper says so.

The code also suppresses the kinetic terms inside ψ_c, where a thermal orbit width reaches ⟨r⟩ (`axis_validity_suppression`). The new path has no such limit; the envelope stays on for the old path and off for the new one.

## 2. The orbit in constants of motion

The orbit is the exact guiding-center motion (Littlejohn's equations) in the
axisymmetric equilibrium, with the electrostatic potential in the Hamiltonian:

H = ½ m v∥² + μ B(ψ, θ) + Z e Φ(ψ),
dX/dt = [v∥ B* + b × ∇H / (Z e)] / B∥*,   m dv∥/dt = −B*·∇H / B∥*,
B* = B + (m v∥/Z e) ∇×b,   B∥* = b·B*,

with ∇H taken at fixed v∥. The constants of motion are E = H (total energy),
μ, and

P_φ = m v∥ b_φ − Z e s psio ψ,   b_φ = F/B,  F = R B_φ = `F_spline`/2π,

with s = ±1 the sign that gives a positive q, i.e.
q = (1/2π) ∮ B^φ/B^θ dθ (this fixes the relative sign of the two terms; M1
check 3 confirms it physically: co-current passing ions sit outward in R).
White & Chance 1984 Eq. 16 and Antonenas 2021 Eq. 2.3 have the same form.

Coordinates are (ψ_N, θ, φ) with θ the equilibrium's straight-field-line
angle on [0, 1) and φ the geometric toroidal angle (so g_ψφ = g_θφ = 0 and the
ν offset never enters). The metric, the Jacobian 𝒥 = R(R_θ Z_ψ − R_ψ Z_θ) and
the covariant b_i = B_i/B come from R(ψ, θ), Z(ψ, θ) (`rzphi_rsquared`,
`rzphi_offset`); B^θ = s psio/𝒥, B^φ = F/R², B_ψ = g_ψθ B^θ, B_θ = g_θθ B^θ.
∇×b is taken numerically from the covariant b_i, so the curvature, the ∇p
part of it and the parallel current all enter as the equilibrium has them.
There is no low-β drift formula anywhere in the orbit.

Note on the existing path (correcting the M0 card): the ZOW precession
integrand `g_wd` is ∂/∂ψ at fixed (E, μ) of the bounce action
∮ J B √(v∥/v) dθ, the action-derivative form of the precession. That form
does not depend on β, so the old path is not a low-β approximation either,
and the ZOW limit of the exact orbit should reproduce it.

**Integration.** The orbit is integrated in time (OrdinaryDiffEq, already a
GPEC dependency), in units of R₀/v, with E and P_φ conservation monitored as
the accuracy diagnostic. This replaces the scalar root solve ψ(θ) of v0: the
root solve is singular at the turning points of a finite-width orbit (where
the two v∥ legs meet away from v∥ = 0) and fails for potato orbits, while time
integration treats trapped, passing, potato and lost orbits alike, and M2's δJ
is a time integral anyway.

**Labels.** An orbit is started at its crossing of θ₀ = 0 (outboard midplane)
with (ψ₀, λ, x, σ): kinetic energy E_k = x T at ψ₀, μ = λ E_k/B₀, v∥ = σ v
√(1 − λB/B₀). This fixes (E, μ, P_φ) and makes the ZOW comparison at the
same (ψ, λ) direct.

**Classification.**
- trapped: v∥ changes sign along the orbit; the period is the return to θ₀
  in the starting direction,
- passing: θ advances by ±1 with no sign change of v∥,
- lost: ψ reaches `psilim`,
- axis: ψ reaches the innermost grid surface (flux coordinates are singular
  there); counted, not used; their share is reported.

**Per orbit**: τ, ω_b = 2π/τ, Δφ over one period and ⟨φ̇⟩ = Δφ/τ,
⟨ψ⟩ = (1/τ)∫ψ dt, ψ_min and ψ_max, the class, and the relative drift of E and
P_φ.

Zero-orbit-width limit (measured in M1): ω_b → the code's `wb`·√x; for
trapped, ⟨φ̇⟩ → s·(ω_D x + ω_E), with s the sign in B^θ = s psio/𝒥 (code_map.md).
For passing, ⟨φ̇⟩ − q ω_b depends on which flux label carries q at first order
in orbit width (the q-advance is ~10³ times the precession), so its ZOW limit
is tested on the ψ-integrated torque, not surface by surface. In the toroidal
phase e^{−in·2πζ}, ζ = (φ − ν)/2π, the resonance is Ω = ℓω_b − n⟨φ̇⟩ (M2).

Analytic reference for the frequencies: Antonenas 2021 Eqs. 4.6–4.7 (pendulum
ω_b, ω_t with the complete K) on a large-aspect-ratio circular equilibrium,
where they hold to O(ε).

## 3. The orbit-averaged action variation

The perturbed field is first order along the unperturbed orbit (δB from
`PerturbedEquilibrium`, including the plasma response). The action variation
for bounce harmonic ℓ is the time integral along the orbit,

δJ_ℓ = ∮ dt A(t) exp(−i n φ_osc(t)) exp(−i ℓ ω_b t),
A(t) = [(2E_k − 2μB) ∇·ξ⊥ + (2E_k − 3μB) δB/B] evaluated at (ψ(t), θ(t)),

up to a constant fixed by the ZOW limit (M2 check 1), with E_k = E − ZeΦ(ψ(t))
the local kinetic energy. This is −(μ δB_L + m v∥² κ·ξ) with the Lagrangian
identity δB_L/B = −(∇·ξ⊥ + κ·ξ), which holds at any β; the code builds δB/B
from exactly that identity with p′ inside κ (`KineticForcesStructs.jl`
`jbb_dbob`, `Fourfit.jl` `build_kinetic_metric_matrices`). It is Park 2011
Eq. 13 and Logan 2013 Eq. 16. (v0 had the ∇·ξ⊥ term a factor 2 too large
relative to δB/B; the dt-versus-dl Jacobian multiplies both terms alike and
cannot cause that.)

The modes δB/B and ∇·ξ⊥ are the code's `dbob_m(ψ)` and `divx_m(ψ)` summed
over m at θ(t), read at ψ(t).

Resonance operator: iΩ − ν with Ω = ℓ ω_b + n⟨φ̇⟩ from the orbit. Ω no longer
has the analytic √x, x form, so (M2) ω_b, ⟨φ̇⟩ and |δJ_ℓ|² are tabulated on a
√x grid at fixed (λ, ψ₀) (λ is energy-independent, so at Δ_b → 0 the grid is
today's (ψ, λ, x) grid), splined in √x, and the Sokhotski–Plemelj add-back is
reused with the pole from a root solve on the interpolated Ω(x).

Phase-space measure: Gorelenkov 1999 Eq. 22 (check the denominator against
the PDF in M2), normalized so the new path's integral reproduces
`tnorm`∫dψ in the ZOW limit.

## 4. The drive and the electric potential

Static perturbation (ω = 0): the drive is n ∂f₀/∂P_φ at fixed (E, μ)
(Gorelenkov 1999 Eqs. 25–28 with ω → 0).

Equilibrium distribution: f₀ is a function of the constants of motion through
E and ψ̄ = ⟨ψ⟩. For a Maxwellian

f₀ = n(ψ̄) (m/2πT(ψ̄))^{3/2} exp(−(E − ZeΦ(ψ̄))/T(ψ̄)),

so ∂f₀/∂ψ̄ at fixed total energy carries −ZeΦ′/T next to n′/n and T′/T, which
gives the ω_E + ω_*n + ω_*T(x − 3/2) numerator in the ZOW limit (Logan 2013
Eq. 18, "taken at constant total energy"). ∂⟨ψ⟩/∂P_φ is computed on the grid
and is −1/(Z e s psio) in ZOW.

Φ(ψ) is built from the input E×B frequency: dΦ/dψ_N = −psio ω_E (from
φ̇ = ∂H/∂P_φ ⊃ ZeΦ′ ∂ψ/∂P_φ), integrated from the axis. Φ is in the orbit
(the Hamiltonian above), so the E×B drift is part of ⟨φ̇⟩ along the orbit and
an orbit that straddles an ω_E zero sees its average, which is the mechanism
behind the edge part of KR2. The ordering is the usual drift-kinetic one:
E×B speed small against the thermal speed (no centrifugal or Coriolis
terms). Φ is a switch (`orbit_potential`, default on; Q8 A: KR2 shows Φ on, with Φ off as a
comparison panel); off reproduces the v0 orbit.

## 5. Lost orbits

Orbits that reach `psilim` are excluded from δW_K and counted: the lost
fraction L(ψ̄) goes to HDF5 and bounds the edge contribution in KR2. A
loss-cone correction to f₀ is out of scope.

## 6. Slowing-down distributions (alphas and beams)

One family covers fusion alphas and neutral-beam ions: the MARS-K model of
Liu 2014 Eqs. 6–7,

f₀ = C(ψ̄)/(E^{3/2} + E_c^{3/2}) · (2√π δζ)⁻¹ Σ_i C_i exp[−(ζ − ζ_i)²/δζ²],  E < E_b,
δζ² = δζ₀² − ⅓ ln[E^{3/2}(1 + E_c^{3/2}/E_b^{3/2}) / (E^{3/2} + E_c^{3/2})],

with E_b the birth (injection) energy, E_c the critical energy, δζ = δζ₀ at
birth and widening as E falls. The pitch label must be a constant of motion,
so ζ is defined from λ at the reference field, ζ = σ√(1 − λ) (v∥/v where
B = B₀); passing orbits keep σ, trapped orbits are symmetric in ζ by
construction (Liu's trapped coefficient sets). The image terms (C_i, ζ_i)
follow Liu 2014 (co-tangential: C = {2, 2, −1, −1}, ζ_i = {ζ₀, 2 − ζ₀,
2ζ₊ − ζ₀, 2ζ₊ − 2 + ζ₀} for co-passing, etc.).

- Alphas: δζ₀ → ∞ (isotropic), E_b = 3.5 MeV, C(ψ) from the fusion source
  S τ_s/4π with S = n_D n_T ⟨σv⟩ (Bosch & Hale 1992) or a user S(ψ).
- Beams: ζ₀ and δζ₀ from the injection geometry (Wang 2022 averaged the
  injection pitch for DIII-D), E_b the beam energy, C(ψ) and E_b(ψ) from the
  fast-ion density and pressure profiles (NUBEAM) by Liu's inverse solve, or C
  given directly.

E_c = 14.8 A T_e (Σ_j n_j Z_j²/(n_e A_j))^{2/3} (the textbook form; the v0
expressions for τ_s and v_c give the same E_c = 37 T_e for pure D). Source to
cite: Stix 1972 unless Nik names another.

Derivatives are analytic. With λ = μB₀/E, ∂f₀/∂E at fixed μ is
∂f₀/∂E|_λ − (λ/E) ∂f₀/∂λ; ∂f₀/∂P_φ goes through ψ̄ as in §4. The birth-energy
edge is a cut, with no delta-function term (∂f/∂E does not enter a static
drive).

Validity: the GPEC equilibrium has an isotropic pressure while an
anisotropic beam pressure does not; as in MARS-K (Liu 2014 §II A) the result
holds when the anisotropic fast-ion pressure is a modest fraction of the
total. No FLR, no energy diffusion beyond the δζ(E) form.

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
| f₀, ∂f₀/∂P_φ | distribution, drive | `f0type`, `_energy_numerator` | `f0type = "slowingdown"` (alphas and beams, §6) added; `drive_pphi` |
| T_φ, δW_K | torque, energy | `MethodResult.total_torque` | same struct, method name `ogar` (orbit GAR) in `METHOD_REGISTRY` |
| L(ψ̄) | lost fraction | — | HDF5 `KineticForces/ogar/lost_fraction` |

Method name `ogar` and the HDF5 group follow the registry convention in `KineticForcesStructs.jl:13` and `docs/development/hdf5-conventions.md`; the deck key is `ogar_flag`, default `false`.

## 8. Validity limits of the new path

- Perturbation first order along the unperturbed orbit: no orbit modification by the 3D field, no stochastic ripple loss (Goldston–White–Boozer 1981 is a separate mechanism, named in the paper).
- Guiding-center (no FLR): ρ/Δ_b ~ √ε/q.
- Collisions: the same Krook-harmonic model as the existing path, applied with the orbit's ω_b; no collisional orbit-width broadening beyond that.
- Orbits computed in the axisymmetric equilibrium of the GPEC run; the TokaMaker field scan must keep that equilibrium's q and β (M4 check 1).
- Electrostatic potential in the orbit with the drift-kinetic ordering (E×B speed ≪ thermal speed); no centrifugal or Coriolis terms.
- Near the axis, potato orbits are resolved only as far as the (λ, P_φ) grid allows; the paper does not claim axis behavior.

## 9. Where sources disagree

- Kim 2013/2014 (POCA) report the torque exceeding theory by factors of 2–3 at resonances and attribute part of it to finite orbit width and passing particles; Shaing (IAEA 2016, abstract only) reports finite banana width smoothing short-scale radial variation. These are not contradictory (amplitude at a resonance versus structure across one), but the paper must not cite either as a quantitative prediction for KR2. The project's own ZOW-versus-orbit comparison is the evidence.
- Liu 2014 (first-order orbit-width expansion) finds the precession resonance unchanged at first order. KR3's exponent check tests whether the leading correction is second order in Δ_b for this path.
