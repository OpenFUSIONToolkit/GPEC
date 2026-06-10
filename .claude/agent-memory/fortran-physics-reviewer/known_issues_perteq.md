---
name: Known Physics Issues in PerturbedEquilibrium
description: Documented physics issues found in perturbed equilibrium module during PR #196 review
type: project
---

## xss_mn (toroidal displacement) not implemented
`FieldReconstruction.jl:316` sets `xss = 0`. This makes b_theta_modes and b_zeta_modes incorrect.
b_psi_modes is unaffected. The Fortran computes xss_mn from idcon_matrix A/B/C matrices.
**Why:** Not yet ported from Fortran gpeq_sol (lines 86-96).
**How to apply:** Any downstream use of b_theta or b_zeta (e.g., tangential displacement, parallel field) requires implementing xss_mn first.

## xi_modes.theta and xi_modes.zeta are zero placeholders
`FieldReconstruction.jl:136-137`. Fortran computes these via gpeq_contra Jacobian convolution.

## gpeq_contra Jacobian convolution bypassed for xi_n
Julia skips the Jacobian convolution for xi_psi before computing xi_n. This is mathematically equivalent (J cancels in xwp/(J*delpsi) = xsp/delpsi) but the intermediate quantities differ from Fortran.
