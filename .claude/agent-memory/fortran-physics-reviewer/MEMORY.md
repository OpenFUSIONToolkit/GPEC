# Agent Memory Index

- [Fortran-Julia Correspondence Map](fortran_correspondence_map.md) — Maps Fortran GPEC source files to Julia JPEC modules
- [Known Physics Issues in PerturbedEquilibrium](known_issues_perteq.md) — xss_mn not implemented, xi_modes placeholders
- [KineticForces (NTV) Audit Checklist](kinetic_ntv_map.md) — pentrc->KineticForces map, Logan 2015 matrices, what to verify
- [InnerLayer (Resistive) Audit Checklist](resistive_layer_map.md) — rmatch->InnerLayer map, GGJ Wasow basis / Δ′, what to verify
- [Galerkin Δ′ Assembly Map](galerkin_assembly_map.md) — gal.f<->GalerkinAssembly.jl; resonant sign chain verified; PASS
- [Torque Scalar vs Profile](torque_scalar_vs_profile.md) — scalar toroidal_torque (ported, faithful) vs gpout_dw/dw_matrix psi-profile (unported); why PE vs fgar mismatch isn't necessarily a bug
