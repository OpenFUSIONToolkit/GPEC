---
name: reg_spot vs singfac_min regularization audit
description: Verdict on GPEC-Julia reg_spot (field-reconstruction smoothing) port and its distinction from singfac_min (ODE crossing gate)
metadata:
  type: project
---

## Verdict (audited FieldReconstruction.jl, no local Fortran repo available)
`reg_spot` port judged CORRECT-WITH-CAVEATS (caveat = Fortran verified from GPEC `xm*` convention + physics + known default, not from on-disk source).

## reg_spot — field-reconstruction smoothing (PerturbedEquilibrium)
- Factor form: `reg_factor = singfac²/(singfac²+reg_spot²)`, singfac = m - n·q. Correct GPEC form. → 0 at rational surface, → 1 away. Applied in singfac-space, NOT ψ-space.
- Default 5e-2 = 0.05 (PerturbedEquilibriumStructs.jl:50). Matches GPEC gpec_input namelist default.
- Quantities regularized (matches GPEC `xm*` modified-quantity convention):
  - xmp1 = ξ^ψ' (FieldReconstruction.jl:393-395)
  - xms (clebsch_alpha) computed FROM regularized xmp1 via xms=-A⁻¹(B·xmp1_reg+C·xsp) (line 398-415, gpeq_sol)
  - xmt/xmz = regularized ξ^θ,ξ^ζ (line 584-594, gpeq_contra); xwt/xwz kept unregularized
  - b^θ_reg,b^ζ_reg inherit reg via xmp1/xms (compute_modified_field_modes)
- LEFT UNREGULARIZED: ξ^ψ primitive (xsp / clebsch_psi = copy, line 356; xwp uses raw xsp). CORRECT — ξ^ψ (normal disp) is finite at rational surface; only 1/singfac-divergent tangential/derivative quantities are softened.

## singfac_min — DISTINCT mechanism (ForceFreeStates ODE crossing gate)
- Default 1e-4 (ForceFreeStatesStructs.jl:267), "Matches Fortran STRIDE". Gates singular-surface crossings during EL ODE integration (EulerLagrange.jl:466-519, GeneralizedPerturbedEquilibrium.jl:426).
- SingularCoupling.jl:204 uses `spot = 5e-4` as an offset (5e-4/|n·q1|) to evaluate bwp1 at lpsi/rpsi — labeled "matches Fortran default singfac_min". This is the ψ-offset for one-sided derivative eval, a THIRD distinct use, not reg_spot.
- reg_spot (0.05, singfac-space smoothing of reconstructed fields) and singfac_min (1e-4, ODE crossing gate) are separate; both ported.
