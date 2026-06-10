---
name: Fortran-Julia File Correspondence Map
description: Mapping between Fortran GPEC source files at ~/Code/gpec and Julia JPEC modules for physics review
type: reference
---

## Coil/ForcingTerms
- `~/Code/gpec/coil/coil.F` (coil_read) -> `src/ForcingTerms/CoilGeometry.jl` (apply_transforms)
- `~/Code/gpec/coil/field.F` (field_bs_psi) -> `src/ForcingTerms/BiotSavart.jl` + `src/ForcingTerms/CoilFourier.jl`

## PerturbedEquilibrium
- `~/Code/gpec/gpec/gpeq.f` (gpeq_sol, gpeq_contra, gpeq_surface, gpeq_normal) -> `src/PerturbedEquilibrium/FieldReconstruction.jl` + `src/PerturbedEquilibrium/ResponseMatrices.jl`
- `~/Code/gpec/gpec/gpresp.f` (gpresp_pinduct, gpresp_sinduct, gpresp_permeab) -> `src/PerturbedEquilibrium/ResponseMatrices.jl`
- `~/Code/gpec/gpec/gpout.f` (gpout_singcoup, gpout_xbnormal) -> `src/PerturbedEquilibrium/SingularCoupling.jl` + `src/PerturbedEquilibrium/FieldReconstruction.jl`
- `~/Code/gpec/gpec/gpvacuum.f` (gpvacuum_flxsurf) -> `src/PerturbedEquilibrium/SingularCoupling.jl` (compute_surface_inductance_from_greens)

## KineticForces (NTV, Fortran `pentrc/`) — see kinetic_ntv_map.md for the audit checklist
- `~/Code/gpec/pentrc/torque.F90` -> `src/KineticForces/Torque.jl` (tpsi! single-surface torque)
- `~/Code/gpec/pentrc/pitch.f90` -> `src/KineticForces/PitchIntegration.jl` (pitch-angle lambda integration, bounce-averaged operator assembly)
- `~/Code/gpec/pentrc/energy.f90` -> `src/KineticForces/EnergyIntegration.jl` (energy-space quadrature integrand)
- `~/Code/gpec/pentrc/pentrc.F90` (orchestration) -> `src/KineticForces/{KineticForces.jl,Compute.jl}`
- `~/Code/gpec/pentrc/inputs.f90` + `params.f90` -> `src/KineticForces/KineticForcesStructs.jl`
- `~/Code/gpec/pentrc/dcon_interface.f` -> `src/KineticForces/CalculatedKineticMatrices.jl` (bridge to ForceFreeStates kinetic stability) + `src/ForceFreeStates/{Kinetic.jl,FixedKineticMatrices.jl}`
- bounce averaging of the 6 perturbed-action matrices -> `src/KineticForces/BounceAveraging.jl`

## InnerLayer (resistive matched-asymptotics, Fortran `rmatch/`) — see resistive_layer_map.md for the audit checklist
- `~/Code/gpec/rmatch/deltac.f` (Galerkin solver) -> `src/InnerLayer/GGJ/Galerkin.jl`
- `~/Code/gpec/rmatch/deltar.f` (shooting solver) -> `src/InnerLayer/GGJ/Shooting.jl`
- `~/Code/gpec/rmatch/inps.f` + `inpso.f` (Wasow asymptotic basis: T,J,P,B,Q,C,D,Y,Z,U matrices) -> `src/InnerLayer/GGJ/InnerAsymptotics.jl`
- `~/Code/gpec/rmatch/{inner.f,match.f,msing.f,gamma.f}` -> `src/InnerLayer/GGJ/` (parameters, matching data, special functions)
- SLAYER drift-MHD two-fluid solver: `src/InnerLayer/SLAYER/` is a placeholder pending implementation

## Key Fortran Conventions
- `sq%f(1)` = F (toroidal field function = R*B_tor), `sq%f(4)` = q (safety factor)
- `sq%f1(4)` = dq/dpsi
- `chi1 = twopi*psio` (poloidal flux normalization)
- `singfac = mfac - nn*q` (resonance factor)
- `ifac = (0,1)` (imaginary unit)
- `wegt=0`: no weighting; `wegt=1`: J*|grad psi| weighting (default for bmn)
- Vacuum Green's functions use reversed theta in Fortran (rtheta = mthsurf - itheta)
