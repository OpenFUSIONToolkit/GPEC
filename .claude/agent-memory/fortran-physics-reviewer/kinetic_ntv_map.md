---
name: KineticForces (NTV) Audit Checklist
description: Curated map and physics-audit checklist for the KineticForces module (NTV, formerly PENTRC); what to read and what to verify
type: reference
---

The KineticForces module computes neoclassical toroidal viscosity (NTV) torque and the
kinetic energy/matrices from trapped-particle nonambipolar transport. Fortran counterpart is
the Fortran `pentrc/` sources (file mapping in fortran_correspondence_map.md).

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
