# Citations

The papers below provide the theoretical foundations for GPEC's algorithms. When working with specific modules, refer to the relevant papers for derivations, equation numbering, and physical context. PDFs are available locally in `docs/resources/`.

---

## Vacuum Module

> M. S. Chance, "Vacuum calculations in azimuthally symmetric geometry,"
> *Physics of Plasmas* **4**, 2161 (1997).
> DOI: [10.1063/1.872380](https://doi.org/10.1063/1.872380)

Describes the fundamental vacuum response calculation for tokamak geometry. Derives the vacuum Green's function matrices and their relation to the plasma boundary, including the singular factor (m - nq)(m' - nq) scaling of the response matrix. This is the primary reference for the `Vacuum` module's core algorithm.

---

> M. S. Chance, A. D. Turnbull, and P. B. Snyder, "Calculation of the vacuum Green's function valid even for high toroidal mode numbers in tokamaks,"
> *Journal of Computational Physics* **221**, 330–348 (2007).
> DOI: [10.1016/j.jcp.2006.06.025](https://doi.org/10.1016/j.jcp.2006.06.025)

Extends the 1997 method to high toroidal mode numbers by using 32-point Gaussian quadrature for Legendre function evaluation when nρ̂ ≥ 0.1. Replaces the polynomial approximations that break down at large n.

---

## ForceFreeStates Module

> A. H. Glasser, "The direct criterion of Newcomb for the ideal MHD stability of an axisymmetric toroidal plasma,"
> *Physics of Plasmas* **23**, 072505 (2016).
> DOI: [10.1063/1.4958328](https://doi.org/10.1063/1.4958328)

The primary reference for the `ForceFreeStates` module. Derives the Euler-Lagrange equations for the DCON ideal MHD stability problem, establishes Newcomb's direct criterion for identifying instability via sign changes, and specifies the jump conditions for crossing singular surfaces (where q = m/n). Equations in this paper are cited directly in `src/ForceFreeStates/`.

---

> A. H. Glasser, S. A. Sabbagh, J. M. Bialek, and Y.-S. Park, "A Riccati solution for the ideal MHD plasma response with applications to real-time stability control,"
> *Physics of Plasmas* **25**, 032507 (2018).
> DOI: [10.1063/1.5007042](https://doi.org/10.1063/1.5007042)

Reformulates the DCON eigenvalue problem as a Riccati matrix ODE, enabling parallel integration across singular surfaces and faster computation. Implemented in `src/ForceFreeStates/Riccati.jl` and enabled via `use_riccati = true` in `[ForceFreeStates]`.

---

## PerturbedEquilibrium Module

> J.-K. Park, A. H. Boozer, and A. H. Glasser, "Computation of three-dimensional tokamak and spherical torus equilibria,"
> *Physics of Plasmas* **14**, 052110 (2007).
> DOI: [10.1063/1.2732170](https://doi.org/10.1063/1.2732170)

Describes the 3D equilibrium perturbation formalism in toroidal geometry that forms the basis of GPEC's perturbed equilibrium approach. Establishes the mode-space representation of displacement and field perturbations.

---

> J.-K. Park, A. H. Boozer, and A. H. Glasser, "Control of Asymmetric Magnetic Perturbations in Tokamaks,"
> *Physical Review Letters* **99**, 195003 (2007).
> DOI: [10.1103/PhysRevLett.99.195003](https://doi.org/10.1103/PhysRevLett.99.195003)

Demonstrates the GPEC framework for computing plasma response to resonant magnetic perturbations (RMPs) and its application to error field correction and ELM suppression. Introduces the decomposition of the response into resonant and non-resonant components.

---

> J.-K. Park, M. J. Schaffer, J. E. Menard, and A. H. Boozer, "Importance of plasma response to nonaxisymmetric perturbations in tokamaks,"
> *Physics of Plasmas* **16**, 056115 (2009).
> DOI: [10.1063/1.3122862](https://doi.org/10.1063/1.3122862)

Establishes the self-consistent plasma response calculation. Derives the permeability matrix formalism linking external fields to the internal displacement field, and connects the response to island half-widths and singular coupling diagnostics — the theoretical foundation for the `PerturbedEquilibrium` module.

---

## InnerLayer Module

> A. H. Glasser, Z. R. Wang, and J.-K. Park, "Computation of resistive instabilities by matched asymptotic expansions,"
> *Physics of Plasmas* **23**, 112506 (2016).
> DOI: [10.1063/1.4967862](https://doi.org/10.1063/1.4967862)

Derives the resistive MHD stability analysis and Δ' calculation using matched asymptotic expansions in the singular layer.

---

> A. H. Glasser, J.-K. Park, and Z. R. Wang, "A robust solution for the resistive MHD toroidal Δ′ matrix in near real-time,"
> *Physics of Plasmas* **25**, 082502 (2018).
> DOI: [10.1063/1.5029477](https://doi.org/10.1063/1.5029477)

Provides an efficient algorithm for computing the full Δ' matrix (coupling between all singular surfaces) suitable for near-real-time control applications.

---

> A. H. Glasser and Z. R. Wang, "Asymptotic solutions and convergence studies of the resistive inner region equations,"
> *Physics of Plasmas* **27**, 012506 (2020).
> DOI: [10.1063/1.5134999](https://doi.org/10.1063/1.5134999)

Showcases a different basis that significantly aids convergence of the Galerkin solver used in the GGJ InnerLayer module.

---

> X. Wang, A. H. Glasser, J.-K. Park, Z. R. Wang, and N. C. Logan, "Modeling of resistive plasma response in toroidal geometry using an asymptotic matching approach,"
> *Physics of Plasmas* **27**, 122503 (2020).
> DOI: [10.1063/5.0020010](https://doi.org/10.1063/5.0020010)

Demonstrates asymptotic matching for resistive plasma response in realistic toroidal geometry, bridging the outer ideal MHD solution to the inner resistive layer solution.

---

> J.-K. Park, "Parametric dependencies of resonant layer responses across linear, two-fluid, drift-MHD regimes,"
> *Physics of Plasmas* **29**, 072506 (2022).
> DOI: [10.1063/5.0093079](https://doi.org/10.1063/5.0093079)

The basis for the presently implemented in Fortran SLAYER code which computes the inner layer response in a two-fluid slab layer model.

---

> A. Burgess et al., "Tearing Stability Prediction Combining Toroidal Calculations With a Two-Fluid Slab Layer,"
> Preprint (2026).

Combines the toroidal outer-region calculation with the two-fluid slab layer model to predict tearing stability, extending the matched-asymptotic approach used in the `InnerLayer` module.

---

## Kinetic Forces

The following papers develop the kinetic-force and neoclassical toroidal viscosity (NTV) theory underpinning GPEC's kinetic analysis path — the energy principle with kinetic effects, the self-consistent coupling of the perturbed equilibrium to NTV, and the PENTRC (Perturbed Equilibrium Neoclassical TRansport Code) formalism.

> J.-K. Park, A. H. Boozer, and J. E. Menard, "Nonambipolar Transport by Trapped Particles in Tokamaks,"
> *Physical Review Letters* **102**, 065002 (2009).
> DOI: [10.1103/PhysRevLett.102.065002](https://doi.org/10.1103/PhysRevLett.102.065002)

Grounds the NTV calculation in the trapped-particle nonambipolar transport theory used downstream by Logan & Park (2013) and Logan (2015) — establishes the kinetic transport channel through which the perturbed equilibrium drives a toroidal torque.

---

> J.-K. Park, "Kinetic energy principle and neoclassical toroidal torque in tokamaks,"
> *Physics of Plasmas* **18**, 110702 (2011).
> DOI: [10.1063/1.3662039](https://doi.org/10.1063/1.3662039)

Extends the energy principle to account for kinetic effects and derives the relationship between the perturbed equilibrium response and neoclassical toroidal torque.

---

> N. C. Logan, J.-K. Park, K. Kim, Z. Wang, and J. W. Berkery, "Neoclassical toroidal viscosity in perturbed equilibria with general tokamak geometry,"
> *Physics of Plasmas* **20**, 122507 (2013).
> DOI: [10.1063/1.4849395](https://doi.org/10.1063/1.4849395)

Derives the neoclassical toroidal viscosity (NTV) torque in perturbed equilibria for general tokamak geometry, providing the theoretical basis for the use of KineticForces in ForceFreeStates.

---

> N. C. Logan, "Electromagnetic Torque in Tokamaks with Toroidal Asymmetries,"
> PhD Thesis, Princeton University (2015).
> Link: [www.proquest.com/docview/1658203153](https://www.proquest.com/docview/1658203153)

Provides the complete theory and implementation details, including NTV in the presence of finite collisionality and general magnetic geometry.

---

> J.-K. Park and N. C. Logan, "Self-consistent perturbed equilibrium with neoclassical toroidal torque in tokamaks,"
> *Physics of Plasmas* **24**, 032505 (2017).
> DOI: [10.1063/1.4977898](https://doi.org/10.1063/1.4977898)

Describes the fully self-consistent coupling between the perturbed equilibrium and neoclassical toroidal viscosity (NTV), providing the theoretical foundation for the the KineticForces functionality.
