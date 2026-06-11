---
name: InnerLayer (Resistive Matched-Asymptotics) Audit Checklist
description: Curated map and physics-audit checklist for the InnerLayer module (GGJ/SLAYER resistive Delta-prime matching); what to read and what to verify
type: reference
---

The InnerLayer module performs matched-asymptotic analysis of resistive MHD stability at
rational surfaces — solving the resistive inner-layer equations and computing the tearing
stability parameter Δ via parity-projected splitting (Δ_odd, Δ_even). Fortran counterpart is
`~/Code/gpec/rmatch/` (file mapping in fortran_correspondence_map.md).

## Governing theory
- Glasser (2016) PoP 23, 072505: computation of resistive instabilities by matched asymptotic expansions; shooting method, Frobenius exponent splitting.
- Glasser, Wang & Park (2016) PoP 23, 112506: matched-asymptotic matching data (Eqs 34–35, parity projection).
- Glasser (2020) "Asymptotic solutions and convergence studies of the resistive inner region equations": full Wasow asymptotic basis (T, J, P, B, Q, C, D, Y, Z, U matrices); Wang rescaling X₀^(2√(−D_I)).
- Glasser (2018) PoP 25, 032501; Wang et al. (2020) PoP 27, 122509: robust Δ′ matrix / asymptotic-matching resistive response.

## Key Julia files and roles
- `InnerLayer.jl` / `InnerLayerInterface.jl` — pluggable `InnerLayerModel` + `solve_inner()`.
- `GGJ/GGJ.jl` — Glasser–Greene–Johnson solver selector/exports.
- `GGJ/GGJParameters.jl` — physical/dimensionless params (E, F, G, H, K); Mercier indices D_I, D_R; Lundquist ratio S = τ_R/τ_A.
- `GGJ/InnerAsymptotics.jl` — Wasow basis construction (T, J matrices; Lyapunov solve; splitting transform; Y-series Frobenius exponents).
- `GGJ/Shooting.jl` — backward shoot x_max → 0 (Frobenius basis).
- `GGJ/Galerkin.jl` — Hermite-cubic finite-element solver with parity BCs.
- `GGJ/Reference.jl` — benchmark `glasser_wang_2020_eq55`.
- `SLAYER/Slayer.jl` — placeholder (drift-MHD two-fluid, pending).

## What to verify in a review
- **Mercier gating**: code must require D_I < 0 (stable Mercier) before proceeding, or error — confirm the assertion exists.
- **Frobenius origin exponents**: p₁ = √(−D_I); check the stability requirement and the Y-series exponents match Glasser 2020.
- **Wasow basis convergence**: truncation order kmax vs growth-rate scale Q; verify the basis matrices (T,J,P,B,Q,C,D,Y,Z,U) are built per Glasser 2020, not approximated.
- **Parity projection**: (Δ_odd, Δ_even) split follows Glasser–Wang–Park 2016 Eqs 34–35.
- **Rescaling**: physical Δ = matching data × X₀^(2p₁) × v₁^(2p₁) — confirm the rescaling factor is applied.
- **Two solvers agree**: shooting and Galerkin must produce consistent Δ on `glasser_wang_2020_eq55`; a divergence is a red flag.
- **Stiff ODE control**: timestep/tolerance handles the exponential-decay modes in the asymptotic regime.
