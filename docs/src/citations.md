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

### 3D vacuum singular correction

The 3D vacuum singular correction (`src/Vacuum/Kernel3D.jl`) is translated from BIEST and licensed under Apache-2.0 (see `NOTICE`). If you use the 3D vacuum calculation, please cite the three references below.

> D. Malhotra, "BIEST: Boundary Integral Equation Solver for Taylor states,"
> [https://github.com/dmalhotra/BIEST](https://github.com/dmalhotra/BIEST).
> DOI: [10.5281/zenodo.23511944](https://doi.org/10.5281/zenodo.23511944)

The C++ library from which the 3D singular-correction quadrature was translated.

---

> D. Malhotra, A. Cerfon, L.-M. Imbert-Gérard, and M. O'Neil, "Taylor states in stellarators: A fast high-order boundary integral solver,"
> *Journal of Computational Physics* **397**, 108791 (2019).
> DOI: [10.1016/j.jcp.2019.06.067](https://doi.org/10.1016/j.jcp.2019.06.067)

Sec. 3.2 defines the polar singular quadrature: partition-of-unity split, polar Gauss–Legendre/trapezoidal rule, and Lagrange interpolation from the collocation grid to the polar nodes.

---

> D. Malhotra, A. J. Cerfon, M. O'Neil, and E. Toler, "Efficient high-order singular quadrature schemes in magnetic fusion,"
> *Plasma Physics and Controlled Fusion* **62**, 024004 (2020).
> DOI: [10.1088/1361-6587/ab57f4](https://doi.org/10.1088/1361-6587/ab57f4)

Applies the scheme to the singular surface integrals of magnetic fusion (exterior vacuum fields, virtual casing) on non-axisymmetric toroidal surfaces, and shows its high-order convergence where singularity subtraction is low order.

---

## ForceFreeStates Module

> A. H. Glasser, "The direct criterion of Newcomb for the ideal MHD stability of an axisymmetric toroidal plasma,"
> *Physics of Plasmas* **23**, 072505 (2016).
> DOI: [10.1063/1.4958328](https://doi.org/10.1063/1.4958328)

The primary reference for the `ForceFreeStates` module. Derives the Euler-Lagrange equations for the DCON ideal MHD stability problem, establishes Newcomb's direct criterion for identifying instability via sign changes, and specifies the jump conditions for crossing singular surfaces (where q = m/n). Equations in this paper are cited directly in `src/ForceFreeStates/`.

---

> A. S. Glasser, E. Kolemen, and A. H. Glasser, "A Riccati solution for the ideal MHD plasma response with applications to real-time stability control,"
> *Physics of Plasmas* **25**, 032507 (2018).
> DOI: [10.1063/1.5007042](https://doi.org/10.1063/1.5007042)

Reformulates the DCON eigenvalue problem as a Riccati matrix ODE, enabling parallel integration across singular surfaces and faster computation. Implemented in `src/ForceFreeStates/Riccati/` and enabled via `integrator = "riccati"` in `[ForceFreeStates]`.

---

## PerturbedEquilibrium Module

> J.-K. Park, A. H. Boozer, and A. H. Glasser, "Computation of three-dimensional tokamak and spherical torus equilibria,"
> *Physics of Plasmas* **14**, 052110 (2007).
> DOI: [10.1063/1.2732170](https://doi.org/10.1063/1.2732170)

Describes the 3D equilibrium perturbation formalism in toroidal geometry that forms the basis of GPEC's perturbed equilibrium approach. Establishes the mode-space representation of displacement and field perturbations.

---

> J.-K. Park, M. J. Schaffer, J. E. Menard, and A. H. Boozer, "Control of Asymmetric Magnetic Perturbations in Tokamaks,"
> *Physical Review Letters* **99**, 195003 (2007).
> DOI: [10.1103/PhysRevLett.99.195003](https://doi.org/10.1103/PhysRevLett.99.195003)

Demonstrates the GPEC framework for computing plasma response to resonant magnetic perturbations (RMPs) and its application to error field correction and ELM suppression. Introduces the decomposition of the response into resonant and non-resonant components.

---

> J.-K. Park, A. H. Boozer, J. E. Menard, A. M. Garofalo, M. J. Schaffer, R. J. Hawryluk, S. M. Kaye, S. P. Gerhardt, S. A. Sabbagh, and the NSTX Team, "Importance of plasma response to nonaxisymmetric perturbations in tokamaks,"
> *Physics of Plasmas* **16**, 056115 (2009).
> DOI: [10.1063/1.3122862](https://doi.org/10.1063/1.3122862)

Establishes the self-consistent plasma response calculation. Derives the permeability matrix formalism linking external fields to the internal displacement field, and connects the response to island half-widths and singular coupling diagnostics — the theoretical foundation for the `PerturbedEquilibrium` module.

---

> J.-K. Park, A. H. Boozer, and J. E. Menard, "Spectral asymmetry due to magnetic coordinates,"
> *Physics of Plasmas* **15**, 064501 (2008).
> DOI: [10.1063/1.2932110](https://doi.org/10.1063/1.2932110)

Shows that the Fourier spectrum of a perturbed field on a flux surface is asymmetric and depends on the magnetic working coordinate, so the raw spectrum is not by itself a physical measure of the field. Motivates normalizing the resonant harmonic by the scalar surface area to obtain the coordinate-independent resonant field — see the [Conventions Reference](conventions.md).

---

> M. Pharr, E. Bursch, N. C. Logan, P. Lunia, J.-K. Park, and C. Paz-Soldan, "Coordinate-invariant flux-surface Fourier analysis in tokamaks,"
> Preprint (2026).
> [arXiv:2606.02901](https://arxiv.org/abs/2606.02901)

Establishes the area weightings that make flux-surface Fourier spectra of perturbed fields coordinate-invariant. Shows that normalizing the resonant flux by the scalar surface area (the resonant field) and the square-root-area-weighted normal field by the scalar square-root area (the power-normalized field) leaves both quantities in tesla and independent of the magnetic working coordinate. The reference for the field-amplitude conventions reported by the `PerturbedEquilibrium` module — see the [Conventions Reference](conventions.md).

---

## InnerLayer Module

> A. H. Glasser, Z. R. Wang, and J.-K. Park, "Computation of resistive instabilities by matched asymptotic expansions,"
> *Physics of Plasmas* **23**, 112506 (2016).
> DOI: [10.1063/1.4967862](https://doi.org/10.1063/1.4967862)

Derives the resistive MHD stability analysis and Δ' calculation using matched asymptotic expansions in the singular layer.

---

> A. S. Glasser and E. Kolemen, "A robust solution for the resistive MHD toroidal Δ′ matrix in near real-time,"
> *Physics of Plasmas* **25**, 082502 (2018).
> DOI: [10.1063/1.5029477](https://doi.org/10.1063/1.5029477)

Provides an efficient algorithm for computing the full Δ' matrix (coupling between all singular surfaces) suitable for near-real-time control applications.

---

> A. H. Glasser and Z. R. Wang, "Asymptotic solutions and convergence studies of the resistive inner region equations,"
> *Physics of Plasmas* **27**, 012506 (2020).
> DOI: [10.1063/1.5134999](https://doi.org/10.1063/1.5134999)

Constructs the Wasow large-``x`` asymptotic basis (the `inps` kernel) that supplies the far-field boundary condition for every GGJ InnerLayer backend — the traditional Galerkin solver and the rotated-contour (`:ray`) solver alike.

---

> Z. Wang, A. H. Glasser, D. Brennan, Y. Liu, and J.-K. Park, "Modeling of resistive plasma response in toroidal geometry using an asymptotic matching approach,"
> *Physics of Plasmas* **27**, 122503 (2020).
> DOI: [10.1063/5.0020010](https://doi.org/10.1063/5.0020010)

Demonstrates asymptotic matching for resistive plasma response in realistic toroidal geometry, bridging the outer ideal MHD solution to the inner resistive layer solution.

---

> J.-K. Park, "Parametric dependencies of resonant layer responses across linear, two-fluid, drift-MHD regimes,"
> *Physics of Plasmas* **29**, 072506 (2022).
> DOI: [10.1063/5.0093079](https://doi.org/10.1063/5.0093079)

The basis for the presently implemented in Fortran SLAYER code which computes the inner layer response in a two-fluid slab layer model.

---

> A. Cole and R. Fitzpatrick, "Drift-magnetohydrodynamical model of error-field penetration in tokamak plasmas,"
> *Physics of Plasmas* **13**, 032503 (2006).
> DOI: [10.1063/1.2178167](https://doi.org/10.1063/1.2178167)

The torque-balance error-field penetration threshold (Eqs. 61–62) behind the critical resonant field computed in `Tearing.CriticalResonantField`.

---

> D. A. Burgess, N. C. Logan, J.-K. Park, C. J. Hansen, and C. Paz-Soldan, "Tearing stability prediction combining toroidal calculations with a two-fluid slab layer,"
> *Plasma Physics and Controlled Fusion* **68**, 065017 (2026).
> DOI: [10.1088/1361-6587/ae7156](https://doi.org/10.1088/1361-6587/ae7156)

Combines the toroidal outer-region calculation with the two-fluid slab layer model to predict tearing stability, extending the matched-asymptotic approach used in the `InnerLayer` module.

---

> R. Fitzpatrick, "Response of a magnetically diverted tokamak plasma to a resonant magnetic perturbation,"
> *Nuclear Fusion* (2025).
> DOI: [10.1088/1741-4326/ae4fdd](https://doi.org/10.1088/1741-4326/ae4fdd) (open access: arXiv:2511.07666)

Gives the diffusive-resistive layer width (Eq. 100) and, in Sect. 5.9, restricts the response calculation to the region inside the point where adjacent resonant layers first overlap. Both are implemented by the layer-overlap scan in `src/Tearing/LayerOverlap.jl`, which feeds the `psilim_from_layer_overlap` cap described in [Integration domain truncation](@ref).

---

> R. Fitzpatrick, "Investigation of tearing mode stability near ideal stability boundaries via asymptotic matching techniques,"
> *Physics of Plasmas* **32**, 062509 (2025).
> DOI: [10.1063/5.0273568](https://doi.org/10.1063/5.0273568) (arXiv:2503.24184)

The critical island width `W_d` and its short/long-mean-free-path parallel conductivity blend, used for the critical-Δ offset in `src/InnerLayer/SLAYER/LayerParameters.jl`.

---

> J. W. Connor, C. J. Ham, R. J. Hastie, and Y. Q. Liu, "The role of thermal conduction in tearing mode theory,"
> *Plasma Physics and Controlled Fusion* **57**, 065001 (2015). arXiv:1410.7240

The toroidal (Eq. 59) and large-aspect-ratio (Eq. 61) critical-Δ offsets from finite parallel thermal conduction, building on H. Lütjens, J.-F. Luciani, and X. Garbet, *Physics of Plasmas* **8**, 4267 (2001).

---

## ErrorFields Module

> N. C. Logan et al., "Robustness of the tokamak error field correction tolerance scaling,"
> *Plasma Physics and Controlled Fusion* **62**, 084001 (2020).

The n = 1 ITPA error-field penetration-threshold fits on the ohmic and L-mode (`"O,L"`) and
ohmic, L- and H-mode (`"O,L,H"`) databases, each by OLS, downsampled OLS and weighted least
squares. Used by `ErrorFields.Risk` (`ITPA_THRESHOLD_SCALINGS`).

---

> N. C. Logan et al., "Empirical scaling of the n = 2 error field penetration threshold in tokamaks,"
> *Nuclear Fusion* **60**, 086010 (2020).

The n = 2 penetration-threshold fits used by `ErrorFields.Risk`.

---

> E. M. Bursch, J.-K. Park, N. C. Logan, F. Mao, N. Wang, C. F. B. Zimmermann, R. J. Buttery, C. Paz-Soldan, M. Pharr, L. Piron, G. Szepesi, H. Wang, S. M. Yang, JET Contributors, and the EUROfusion Tokamak Exploitation Team,
> "Improved n=1 empirical error field penetration threshold scaling with Ohmic and L-mode conventional tokamak plasma discharges,"
> *Plasma Physics and Controlled Fusion* (2026).
> DOI: [10.1088/1361-6587/aea7d6](https://doi.org/10.1088/1361-6587/aea7d6)

The 2026 n = 1 penetration-threshold scalings on the core dominant-mode overlap δ, fitted to
ohmic and L-mode discharges of conventional tokamaks only (C-Mod, DIII-D, EAST, JET, J-TEXT,
KSTAR) and adding a plasma-current term: Eq. 7 (OLS) and Eq. 8 (KDE-weighted WLS),
`δ = 10^α_c (β_n/l_i)^α_β |I_p|^α_I R_0^α_R n_e^α_n |B_T|^α_B`. These are the `year = 2026`
fits of `ErrorFields.Risk`. The paper's overlap is the same dominant-mode metric the `ErrorFields`
module computes (Eqs. 1–5).

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

Describes the fully self-consistent coupling between the perturbed equilibrium and neoclassical toroidal viscosity (NTV), providing the theoretical foundation for the KineticForces functionality.
