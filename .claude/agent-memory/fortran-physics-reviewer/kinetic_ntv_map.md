---
name: KineticForces (NTV) Audit Checklist
description: Curated map and physics-audit checklist for the KineticForces module (NTV, formerly PENTRC); what to read and what to verify
type: reference
---

The KineticForces module computes neoclassical toroidal viscosity (NTV) torque and the
kinetic energy/matrices from trapped-particle nonambipolar transport. Fortran counterpart is
`~/Code/gpec/pentrc/` (file mapping in fortran_correspondence_map.md).

## Governing theory
- Logan & Park (2013) PoP 20, 122507: diamagnetic frequency (Eq. 7), energy integrand (Eq. 8), torque normalization Im(T) = 2n·δW_k (Eq. 19).
- Logan (2015) PhD Thesis, Ch. 7: the six kinetic matrices A_k, B_k, C_k, D_k, E_k, H_k (Eqs 7.30–7.35) as energy-space integrals of perturbed action operators W_X, W_Y, W_Z; resonance splitting/suppression F_h = (Q−P†)F̄(Q−P)+… (Eq. 7.46). App. C: DCON matrix form of the perturbed action. App. D: numerical treatment of integrable singularities in bounce averages.
- Park et al. (2009) PRL 102, 065002: trapped-particle nonambipolar transport foundation.

## Key Julia files and roles
- `Torque.jl` — `tpsi!()`: single-surface torque; poloidal grid sampling; diamagnetic frequencies (Eq. 7).
- `EnergyIntegration.jl` — energy-space quadrature integrand (Eq. 8); Maxwellian vs JKP distribution options.
- `PitchIntegration.jl` — pitch-angle (λ) integration; bounce-averaged operator assembly.
- `BounceAveraging.jl` — bounce-averaging of the perturbed-action matrices (Eqs 7.30–7.35); packs 3 Hermitian + 3 full blocks.
- `Compute.jl` — `compute_torque_all_methods!()` orchestration over all ψ surfaces.
- `CalculatedKineticMatrices.jl` — bridge to ForceFreeStates kinetic stability (`Kinetic.jl`, `FixedKineticMatrices.jl`).
- `KineticForcesStructs.jl` — `KineticForcesControl` (TOML params) and `KineticForcesInternal` (state, hints, mode indexing).
- `Output.jl` — HDF5 writer.

## What to verify in a review
- **Matrix structure**: A symmetric; the Hermitian blocks actually Hermitian; the 3-Hermitian + 3-full packing in BounceAveraging matches Logan 2015 Eqs 7.30–7.35.
- **Resonance handling**: Sokhotski–Plemelj pole shift / singular-denominator gating near rational surfaces (App. D). Check the singular-eps threshold is applied where the denominator vanishes, not globally.
- **Quadrature tolerances**: outer ψ (atol_psi/rtol_psi) and inner pitch/energy (atol_xlmda/rtol_xlmda) are passed through, not hard-coded to lazy defaults.
- **Distribution/collision options**: energy integrand (Eq. 8) honors the configured collision operator (harmonic/Krook/zero) and distribution (Maxwellian/JKP/CGL) — not silently fixed to one.
- **Normalization**: torque Im(T)=2n·δW_k (Eq. 19) and diamagnetic-frequency sign/factor conventions match Fortran `torque.F90`.
- **Mode indexing**: m, n ranges and block packing over n stay consistent with ForceFreeStates.
- **Method variants**: FGAR/TGAR/PGAR/RLAR/CLAR/FCGL/TMM/WMM each present, not stubbed to a single fallback.

## Multi-ion (D-T) NTV — composition/collisionality (2026-07)
Both Fortran PENTRC (`inputs.f90:236-243`, `read_kin`) and Julia (`KineticProfiles.jl:261-278`)
support ONE main ion (zi,mi) + ONE impurity (zimp,mimp) per run. Correct multi-main-ion NTV is
an EXTENSION beyond Fortran, but consistent with Logan-Park 2013's pitch-angle (Lorentz) model.
- Zeff = Σ_s Z_s² n_s / n_e; quasineutrality n_e = Σ_s Z_s n_s (all ions incl. impurity).
- Bug in current D-T split (run D, then T, each with ni=Ni/2): the `ni` column sets BOTH species
  density AND Zeff via z = zimp-(n_i/n_e)zi(zimp-zi). With ni=Ni/2 it treats the missing half as
  high-Z impurity → Zeff≈3.7 instead of true ≈1–1.4. Corrupts zpitch → nue,nui.
- Correct collisionality: ν_a ∝ Z_a² lnΛ · Σ_b n_b Z_b² /(√m_a T_a^{3/2}) = ∝ Z_a² lnΛ·(n_e·Zeff_true).
  The zpitch(Zeff) polynomial is a main-ion+impurity closure (momentum-restoring correction); the
  MINIMAL fix is to feed the TRUE (full-composition) Zeff into the existing zpitch/ν formulas.
- Additivity: τ = Σ_s τ_s. Lorentz operator is additive over field species; species couple only
  through shared δB, shared ω_E, shared Zeff/ν. Additive at this theory's order.
- What changes single→multi: Zeff, zpitch, nue, nui (→nueff). UNCHANGED for equal-shape D-T:
  wdian/wdiat (log-derivative, density factor cancels), wtran/wbhat/wdhat/wgyro (already per-species),
  and the resonant-density prefactor (Ni/2 per species is correct).
- ASIDE (separate fidelity bug, same code block): Fortran `inputs.f90:238` uses natural log for lnΛ;
  Julia `KineticProfiles.jl:270` uses log10 — diverges away from the n=1e20,T=1keV reference point.

## Multi-ion NTV — full-composition species set (2026-07 audit, PASS-with-caveats)
Reviewed `resolve_ntv_species` (KineticProfiles.jl ~214-266) + `compute_calculated_kinetic_matrices`
(CalculatedKineticMatrices.jl ~96-151). Verdict: physics is sound.
- ν_s field-density RECONCILIATION (supersedes the earlier "should be n_e·Zeff" note above):
  code uses `ν_s = (zpitch/3.5e17)·z_s²·n_main·lnΛ/(√m_s·T_i^1.5)`, i.e. field density = zpitch·n_main
  (n_main = Σ MAIN-ion densities, no impurity, unweighted). This is CORRECT and MORE faithful to
  single-ion PENTRC than n_e·Zeff: PENTRC's design is zpitch·n_i, NOT Σ_b n_b Z_b². Numerically
  zpitch·n_main ≈ n_e·Zeff (e.g. Zeff=1.5,C6 → 1.55 vs 1.5); the gap IS the intended momentum-
  restoring correction that zpitch(Zeff) carries. So n_main vs n_e·Zeff is immaterial at D-T (z=1).
- z_s² test-particle factor: CORRECT and correctly placed (deflection freq ∝ test charge²). Single-ion
  had no z² only because zi=1. Reduces EXACTLY to single-ion nui for one z=1,fraction=1 ion (verified).
- Impurity as its own test species: field density zpitch·n_main is NOT undercounting — the impurity's
  z_imp² contribution is already folded into Zeff inside zpitch. CAVEAT: zpitch is strictly a main-ion
  momentum-restoring closure; reusing it for the impurity/electron ν is an approximation beyond the
  single-ion theory. Acceptable (impurity δW ∝ n_imp is small); worth a one-line annotation.
- Electron descriptor passes `ns[1]` (first ion's density) as its `ni_spline` — HARMLESS: the kernel
  `_setup_surface_state` (Torque.jl:659-664) reads `ne_spline` (full shared n_e) when electron=true,
  never ni_spline. Electron n_s = full n_e (correct). Cosmetic smell only.
- Self-consistent δW summation (CalculatedKineticMatrices.jl:112-148): kw_flat/kt_flat are PURELY the
  kinetic matrices (Logan 7.30-7.35), every term ∝ species phase-space density n_s·f0_s — NO species-
  independent baseline. Fluid F,K,G added ONCE downstream in _compute_fkg_matrices!, outside the
  species loop. So `+=` over species is clean additivity; NOTHING in kw is wrongly ×species-count.
  → TC-24 n=3 δW +0.066(D) → −0.10(D+T+C+e) sign flip is PLAUSIBLE physics near marginal stability,
  NOT a double-count. Dominant driver: the newly-ON electron channel (full n_e, opposite precession).
  DECISIVE cheap diagnostic to confirm: run D(½)+T(½) with electron=OFF, impurity absent — should
  return ≈ +0.066 (single-ion D). If D+T alone ≈ +0.066, the whole shift is electron+impurity = physical.

## Single-ion nui vs multi-species z_s² (#339, 2026-07 decision)
Fortran PENTRC `inputs.f90:240-241` single-ion nui = (zpitch/3.5e17)·n_i·lnΛ/(√mi·T_i^1.5) has
NO explicit zi² (implicitly assumes zi=1, main ion hydrogenic). Julia `load_kinetic_profiles`
(~L416) is a faithful exact port. Multi-species `_nu` (~L253) adds explicit test-particle z_s²
(the physically correct pitch-angle/Lorentz form; zpitch is the field-side momentum-restoring
factor of Zeff, independent of test charge — no double count). RECOMMENDATION: option (a) ADD
zi² to single-ion nui. It is numerically identical for the default zi=1 (regression byte-identical),
makes single-ion the true 1-species special case of `_nu`, and only "deviates" from Fortran in the
exotic zi≠1 case where Fortran is physically wrong anyway. Annotate as a documented improvement.
