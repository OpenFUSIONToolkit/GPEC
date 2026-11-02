---
name: Known Physics Issues in PerturbedEquilibrium
description: Documented physics issues found in perturbed equilibrium module during PR #196 review
type: project
---

## xi_modes.theta and xi_modes.zeta are zero placeholders
`FieldReconstruction.jl:136-137`. Fortran computes these via gpeq_contra Jacobian convolution.

## gpeq_contra Jacobian convolution bypassed for xi_n
Julia skips the Jacobian convolution for xi_psi before computing xi_n. This is mathematically equivalent (J cancels in xwp/(J*delpsi) = xsp/delpsi) but the intermediate quantities differ from Fortran.
