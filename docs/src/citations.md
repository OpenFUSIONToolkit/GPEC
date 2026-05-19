# Citations

The papers below provide the theoretical foundations for GPEC's algorithms. When working with specific modules, refer to the relevant papers for derivations, equation numbering, and physical context. PDFs are available locally in `docs/resources/`.

---

## Vacuum Module

### Chance et al. (1997) {#chance-1997}

> M. S. Chance, "Vacuum calculations in azimuthally symmetric geometry,"
> *Physics of Plasmas* **4**, 2161 (1997).
> DOI: [10.1063/1.872380](https://doi.org/10.1063/1.872380)

Describes the fundamental vacuum response calculation for tokamak geometry. Derives the vacuum Green's function matrices and their relation to the plasma boundary, including the singular factor (m - nq)(m' - nq) scaling of the response matrix. This is the primary reference for the `Vacuum` module's core algorithm.

Local PDF: `docs/resources/1997-Chance-Vacuum_calculations_in_azimuthally_symmetric_geometry.pdf`

---

### Chance et al. (2007) {#chance-2007}

> M. S. Chance et al., "Calculation of the vacuum Green's function valid even for high toroidal mode numbers in tokamaks,"
> *Physics of Plasmas* **14**, 052506 (2007).

Extends the 1997 method to high toroidal mode numbers by using 32-point Gaussian quadrature for Legendre function evaluation when nρ̂ ≥ 0.1. Replaces the polynomial approximations that break down at large n.

Local PDF: `docs/resources/2007-Chance-Calculation of the vacuum Greens function valid even for high toroidal mode numbers in tokamaks.pdf`

---

## ForceFreeStates Module

### Glasser (2016) — Newcomb's Criterion {#glasser-2016-newcomb}

> A. H. Glasser, "The direct criterion of Newcomb for the ideal MHD stability of an axisymmetric toroidal plasma,"
> *Physics of Plasmas* **23**, 112506 (2016).

The primary reference for the `ForceFreeStates` module. Derives the Euler-Lagrange equations for the DCON ideal MHD stability problem, establishes Newcomb's direct criterion for identifying instability via sign changes, and specifies the jump conditions for crossing singular surfaces (where q = m/n). Equations in this paper are cited directly in `src/ForceFreeStates/`.

Local PDF: `docs/resources/2016-Glasser-The_direct_criterion_of_Newcomb_for_the_ideal_MHD_stability_of_an_axisymmetric_toroidal_plasma.pdf`

---

### Glasser (2018) — Riccati Solution {#glasser-2018-riccati}

> A. H. Glasser, "A Riccati solution for the ideal MHD plasma response with applications to real-time stability control,"
> *Physics of Plasmas* **25**, 032507 (2018).

Reformulates the DCON eigenvalue problem as a Riccati matrix ODE, enabling parallel integration across singular surfaces and faster computation. Implemented in `src/ForceFreeStates/Riccati.jl` and enabled via `use_riccati = true` in `[ForceFreeStates]`.

Local PDF: `docs/resources/2018-Glasser-A Riccati solution for the ideal MHD plasma response with applications to real-time stability control.pdf`

---

## PerturbedEquilibrium Module

### Park et al. (2007a) {#park-2007a}

> J.-K. Park et al., "Computation of three-dimensional tokamak and spherical torus equilibria,"
> *Physics of Plasmas* **14**, 052110 (2007).

Describes the 3D equilibrium perturbation formalism in toroidal geometry that forms the basis of GPEC's perturbed equilibrium approach. Establishes the mode-space representation of displacement and field perturbations.

Local PDF: `docs/resources/2007-Park-Computation_of_three-dimensional_tokamak_and_spherical_torus_equilibria-compressed.pdf`

---

### Park et al. (2007b) {#park-2007b}

> J.-K. Park et al., "Control of Asymmetric Magnetic Perturbations in Tokamaks,"
> *Physical Review Letters* **99**, 195003 (2007).

Demonstrates the GPEC framework for computing plasma response to resonant magnetic perturbations (RMPs) and its application to error field correction and ELM suppression. Introduces the decomposition of the response into resonant and non-resonant components.

Local PDF: `docs/resources/2007-Park-Control_of_Asymmetric_Magnetic_Perturbations_in_Tokamaks.pdf`

---

## InnerLayer Module

### Glasser et al. (2016) — Resistive Instabilities {#glasser-2016-resistive}

> A. H. Glasser, "Computation of resistive instabilities by matched asymptotic expansions,"
> *Physics of Plasmas* **23**, 072505 (2016).

Derives the resistive MHD stability analysis and Δ' calculation using matched asymptotic expansions in the singular layer.

Local PDF: `docs/resources/2016-Glasser-Computation_of_resistive_instabilities_by_matched_asymptotic_expansions-compressed.pdf`

---

### Glasser (2018) — Resistive Δ' Matrix {#glasser-2018-delta-prime}

> A. H. Glasser, "A robust solution for the resistive MHD toroidal Δ′ matrix in near real-time,"
> *Physics of Plasmas* **25**, 032501 (2018).

Provides an efficient algorithm for computing the full Δ' matrix (coupling between all singular surfaces) suitable for near-real-time control applications.

Local PDF: `docs/resources/2018-Glasser-A robust solution for the resistive MHD toroidal Delta-prime matrix in near real-time.pdf`

---

### Glasser and Wang (2020) - Asymptotic solutions and convergence studies of the resistive inner region equations {#glasser-2020-inps}

> A. H. Glasser and Z. Wang, "Asymptotic solutions and convergence studies of the resistive inner region equations,"
> *Physics of Plasmas* **27**, 012506 (2020).

Showcases a different basis that significantly aids convergence of the Galerkin solver used in the GGJ InnerLayer module.

Local PDF: `docs/resources/2020 - Glasser - Asymptotic solutions and convergence studies of the resistive inner region equations.pdf`

---

### Wang et al. (2020)

> X. Wang et al., "Modeling of resistive plasma response in toroidal geometry using an asymptotic matching approach,"
> *Physics of Plasmas* **27**, 122509 (2020).

Demonstrates asymptotic matching for resistive plasma response in realistic toroidal geometry, bridging the outer ideal MHD solution to the inner resistive layer solution.

Local PDF: `docs/resources/2020-Wang-Modeling of resistive plasma response in toroidal geometry using an asymptotic matching approach.pdf`

---

### Park (2022) — Parametric dependencies of resonant layer responses {#park-2022-parametric-dependencies}

> J.-K. Park, "Parametric dependencies of resonant layer responses across linear, two-fluid, drift-MHD regimes,"
> *Physics of Plasmas* **29**, 072506 (2022).

The basis for the presently implemented in fortran SLAYER code which computes the inner layer response in a two-fluid slab layer model.

Local PDF: `docs/resources/2022-Park-Parametric dependencies of resonant layer responses across linear, two-fluid, drift-MHD regimes.pdf`

---

### Burgess et al. (2026) — Tearing Stability Prediction Combining Toroidal Calculations With a Two-Fluid Slab Layer {#burgess-2026-tearing-stability-prediction}

> A. Burgess et al., "Tearing Stability Prediction Combining Toroidal Calculations With a Two-Fluid Slab Layer,"
> Preprint (2026).

Combines the toroidal outer-region calculation with the two-fluid slab layer model to predict tearing stability, extending the matched-asymptotic approach used in the `InnerLayer` module.

Local PDF: `docs/resources/2026-Burgess-Tearing Stability Prediction Combining Toroidal Calculations With a Two-Fluid Slab Layer.pdf`

---

## Kinetic Forces

The following papers develop the kinetic-force and neoclassical toroidal viscosity (NTV) theory underpinning GPEC's kinetic analysis path — the energy principle with kinetic effects, the self-consistent coupling of the perturbed equilibrium to NTV, and the PENTRC (Perturbed Equilibrium Neoclassical Toroidal viscosity in Realistic geometry Code) formalism.

### Park et al. (2009) {#park-2009}

> J.-K. Park et al., "Importance of plasma response to nonaxisymmetric perturbations in tokamaks,"
> *Physics of Plasmas* **16**, 056115 (2009).

Establishes the self-consistent plasma response calculation. Derives the permeability matrix formalism linking external fields to the internal displacement field, and connects the response to island half-widths and singular coupling diagnostics.

Local PDF: `docs/resources/2009-Park-Importance_of_plasma_response_to_nonaxisymmetric_perturbations_in_tokamaks-compressed.pdf`

---

### Park et al. (2011) {#park-2011}

> J.-K. Park et al., "Kinetic energy principle and neoclassical toroidal torque in tokamaks,"
> *Physics of Plasmas* **18**, 110702 (2011).

Extends the energy principle to account for kinetic effects and derives the relationship between the perturbed equilibrium response and neoclassical toroidal torque.

Local PDF: `docs/resources/2011-Park-Physics_of_Plasmas_Kinetic_energy_principle_and_neoclassical_toroidal_torque_in_tokamaks.pdf`

---

### Logan & Park (2013)

> N. C. Logan and J.-K. Park, "Neoclassical toroidal viscosity in perturbed equilibria with general tokamak geometry,"
> *Physics of Plasmas* **20**, 122507 (2013).

Derives the neoclassical toroidal viscosity (NTV) torque in perturbed equilibria for general tokamak geometry, providing the theoretical basis for PENTRC.

Local PDF: `docs/resources/2013-Logan-Neoclassical_toroidal_viscosity_in_perturbed_equilibria_with_general_tokamak_geometry.pdf`

---

### Logan (2015)

> N. C. Logan, "Electromagnetic Torque in Tokamaks with Toroidal Asymmetries,"
> PhD Thesis, Princeton University (2015).

Provides the complete PENTRC theory and implementation details, including NTV in the presence of finite collisionality and general magnetic geometry.

Local PDF: `docs/resources/2015-Logan-Electromagnetic_Torque_in_Tokamaks_with_Toroidal_Asymmetries-compressed.pdf`

---

### Park et al. (2017) {#park-2017}

> J.-K. Park et al., "Self-consistent perturbed equilibrium with neoclassical toroidal torque in tokamaks,"
> *Physics of Plasmas* **24**, 032505 (2017).

Describes the fully self-consistent coupling between the perturbed equilibrium and neoclassical toroidal viscosity (NTV), providing the theoretical foundation for the PENTRC functionality.

Local PDF: `docs/resources/2017-Park-Self_consistent_perturbed_equilibrium_with_neoclassical_toroidal_torque_in_toka.pdf`

---

### Shaing et al. (2009)

> K. C. Shaing et al., "Nonambipolar Transport by Trapped Particles in Tokamaks,"
> (2009).

Describes the neoclassical transport theory for trapped particles in tokamaks that underpins the NTV calculation.

Local PDF: `docs/resources/2009-Nonambipolar_Transport_by_Trapped_Particles_in_Tokamaks.pdf`
