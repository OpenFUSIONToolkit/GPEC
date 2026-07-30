# Key References

The papers in `docs/resources/` provide the theoretical foundation for GPEC's algorithms and
should be referenced to understand what the code is doing. **Citing equations from these papers
in code comments and annotations is strongly encouraged** to maintain traceability between theory
and implementation.

## Vacuum Module

The Vacuum module implements the methods described in:

- **Chance et al. (1997)**: "Vacuum calculations in azimuthally symmetric geometry"
  - Location: `docs/resources/1997-Chance-Vacuum_calculations_in_azimuthally_symmetric_geometry.pdf`
  - Published: Physics of Plasmas **4**, 2161 (1997)
  - Link: https://pubs.aip.org/aip/pop/article-abstract/4/6/2161/263192
  - Describes: Fundamental vacuum response calculation method for tokamak geometry

- **Chance et al. (2007)**: "Calculation of the vacuum Green's function valid even for high toroidal mode numbers in tokamaks"
  - Location: `docs/resources/2007-Chance-Calculation of the vacuum Greens function valid even for high toroidal mode numbers in tokamaks.pdf`
  - Published: Physics of Plasmas **14**, 052506 (2007)
  - Describes: Improved Green's function calculation for high-n modes

## ForceFreeStates Module

The ForceFreeStates module (ideal MHD stability analysis) implements methods from:

- **Glasser (2016)**: "The direct criterion of Newcomb for the ideal MHD stability of an axisymmetric toroidal plasma"
  - Location: `docs/resources/2016-Glasser-The_direct_criterion_of_Newcomb_for_the_ideal_MHD_stability_of_an_axisymmetric_toroidal_plasma.pdf`
  - Published: Physics of Plasmas **23**, 112506 (2016)
  - **Describes: FUNDAMENTAL PAPER - Newcomb's criterion for ideal MHD stability (current implementation)**

- **Glasser (2018)**: "A Riccati solution for the ideal MHD plasma response with applications to real-time stability control"
  - Location: `docs/resources/2018-Glasser-A Riccati solution for the ideal MHD plasma response with applications to real-time stability control.pdf`
  - Published: Physics of Plasmas **25**, 032507 (2018)
  - Describes: Riccati method for ideal MHD eigenvalue problem

## PerturbedEquilibrium Module

The PerturbedEquilibrium module implements GPEC-style perturbed equilibrium calculations from:

- **Park et al. (2007a)**: "Computation of three-dimensional tokamak and spherical torus equilibria"
  - Location: `docs/resources/2007-Park-Computation_of_three-dimensional_tokamak_and_spherical_torus_equilibria-compressed.pdf`
  - Published: Physics of Plasmas **14**, 052110 (2007)
  - Describes: 3D equilibrium perturbations in toroidal geometry

- **Park et al. (2007b)**: "Control of Asymmetric Magnetic Perturbations in Tokamaks"
  - Location: `docs/resources/2007-Park-Control_of_Asymmetric_Magnetic_Perturbations_in_Tokamaks.pdf`
  - Published: Physical Review Letters **99**, 195003 (2007)
  - Describes: Plasma response to resonant magnetic perturbations (RMP)

- **Park et al. (2008)**: "Spectral asymmetry due to magnetic coordinates"
  - Location: `docs/resources/2008-Park-Spectral_asymmetry_due_to_magnetic_coordinates.pdf`
  - Published: Physics of Plasmas **15**, 064501 (2008)
  - Describes: Coordinate-dependence of the perturbed-field Fourier spectrum, motivating area-normalization of the resonant harmonic to obtain the coordinate-invariant resonant field

- **Park et al. (2009)**: "Importance of plasma response to nonaxisymmetric perturbations in tokamaks"
  - Location: `docs/resources/2009-Park-Importance_of_plasma_response_to_nonaxisymmetric_perturbations_in_tokamaks-compressed.pdf`
  - Published: Physics of Plasmas **16**, 056115 (2009)
  - Describes: Self-consistent plasma response calculation

- **Park et al. (2011)**: "Kinetic energy principle and neoclassical toroidal torque in tokamaks"
  - Location: `docs/resources/2011-Park-Physics_of_Plasmas_Kinetic_energy_principle_and_neoclassical_toroidal_torque_in_tokamaks.pdf`
  - Published: Physics of Plasmas **18**, 110702 (2011)
  - Describes: Energy principle for perturbed equilibria

- **Park et al. (2017)**: "Self-consistent perturbed equilibrium with neoclassical toroidal torque in tokamaks"
  - Location: `docs/resources/2017-Park-Self_consistent_perturbed_equilibrium_with_neoclassical_toroidal_torque_in_toka.pdf`
  - Published: Physics of Plasmas **24**, 032505 (2017)
  - Describes: Self-consistent coupling with neoclassical effects

## Resistive MHD Stability Analysis (Future Work)

GPEC will eventually implement resistive MHD stability analysis based on:

- **Glasser (2016)**: "Computation of resistive instabilities by matched asymptotic expansions"
  - Location: `docs/resources/2016-Glasser-Computation_of_resistive_instabilities_by_matched_asymptotic_expansions-compressed.pdf`
  - Published: Physics of Plasmas **23**, 072505 (2016)
  - Describes: Resistive stability analysis and Δ' calculation via matched asymptotic expansions

- **Glasser (2018)**: "A robust solution for the resistive MHD toroidal Δ′ matrix in near real-time"
  - Location: `docs/resources/2018-Glasser-A robust solution for the resistive MHD toroidal Delta-prime matrix in near real-time.pdf`
  - Published: Physics of Plasmas **25**, 032501 (2018)
  - Describes: Fast computation of Δ' matrix for resistive stability

- **Wang et al. (2020)**: "Modeling of resistive plasma response in toroidal geometry using an asymptotic matching approach"
  - Location: `docs/resources/2020-Wang-Modeling of resistive plasma response in toroidal geometry using an asymptotic matching approach.pdf`
  - Published: Physics of Plasmas **27**, 122509 (2020)
  - Describes: Asymptotic matching for resistive plasma response

## KineticForces Module (NTV)

The KineticForces module (formerly PENTRC) implements neoclassical toroidal viscosity calculations. Based on:

- **Logan & Park (2013)**: "Neoclassical toroidal viscosity in perturbed equilibria with general tokamak geometry"
  - Location: `docs/resources/2013-Logan-Neoclassical_toroidal_viscosity_in_perturbed_equilibria_with_general_tokamak_geometry.pdf`
  - Published: Physics of Plasmas **20**, 122507 (2013)
  - Describes: Neoclassical toroidal viscosity (NTV) in perturbed equilibria

- **Logan (2015)**: "Electromagnetic Torque in Tokamaks with Toroidal Asymmetries"
  - Location: `docs/resources/2015-Logan-Electromagnetic_Torque_in_Tokamaks_with_Toroidal_Asymmetries-compressed.pdf`
  - Published: PhD Thesis, Princeton University (2015)
  - Describes: Complete NTV theory and implementation. **Chapter 7** details the hybrid drift-kinetic MHD eigenfunction calculation: 6 kinetic matrices Ak,Bk,Ck,Dk,Ek,Hk (Eqs 7.30-7.35) as energy-space integrals of perturbed action operators WX,WY,WZ; hybrid Euler-Lagrange equations; resonance splitting/suppression where Fh=(Q-P†)F̄(Q-P)+... shifts singularities away from rational surfaces (Eq 7.46); convergence to ideal limit. **Appendix C** derives the DCON matrix form of the perturbed action (Eqs C.1-C.11) used to compute the kinetic coefficient matrices. **Appendix D** details numerical treatment of integrable singularities in bounce averages.

## Additional References

- **Park et al. (2009)**: "Nonambipolar Transport by Trapped Particles in Tokamaks"
  - Location: `docs/resources/2009-Park-Nonambipolar_Transport_by_Trapped_Particles_in_Tokamaks.pdf`
  - Published: Physical Review Letters **102**, 065002 (2009)
  - Link: https://doi.org/10.1103/PhysRevLett.102.065002
  - Describes: Trapped-particle nonambipolar transport theory underpinning the NTV calculation
