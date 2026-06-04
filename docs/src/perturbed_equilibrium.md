# Perturbed Equilibrium

The `PerturbedEquilibrium` module computes the plasma response to external magnetic perturbations.

## Types

```@docs
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.PerturbedEquilibriumControl
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.PerturbedEquilibriumInternal
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.PerturbedEquilibriumState
```

## Functions

```@docs
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.compute_perturbed_equilibrium
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.write_outputs_to_HDF5
```

## Unit conventions

This section is an authoritative SI unit reference for the quantities flowing through the
perturbed-equilibrium pipeline. The units below were derived by **tracing the literal
calculation** — what the code actually multiplies together, starting from raw SI inputs
(`B` in tesla, lengths in metres, currents in amperes) — and then cross-checked against the
Fortran GPEC `gpout.f` annotations and the Park et al. papers in `docs/resources/`. It
resolves the audit requested in issue #239.

Scope: this is a **dimensional** audit (it establishes the SI unit of each quantity). It is not
a numerical benchmark — confirming that the Julia magnitudes match Fortran GPEC belongs in the
regression harness, not here.

### Pipeline overview

```
coil current [A]
  └─ Biot–Savart (MU0_OVER_4PI = 1e-7 T·m/A)            → B_R, B_Z [T]
       └─ project_normal_flux!  (× 2πR·∂(R,Z)/∂θ)        → bn [T·m²]
            └─ fourier_decompose_bn                      → ForcingMode.amplitude = Φ_x [Wb]
                 └─ map to eigenmode basis (forcing_vec) → Φ_x [Wb]
                      └─ response matrices (Λ, L, P, ϱ)  → Φ_tot, energies, singular coupling
```

Throughout, note the identity **T·m² ≡ Wb** (weber) and **Wb²/H ≡ J** (joule), since several
quantities are most naturally written one way but reduce to the other. Factors of `2π` and
`(2π)²` are dimensionless angle-convention rescalings and never change SI units.

### Forcing chain

| Quantity | Source (file · symbol) | SI unit | Authority |
|----------|------------------------|---------|-----------|
| `B_R`, `B_Z` (coil) | `ForcingTerms/BiotSavart.jl` · Biot–Savart kernel | **T** | `MU0_OVER_4PI [T·m/A] × I [A] × (dl×r̂)/r² [1/m]` |
| `bn` (normal flux element) | `ForcingTerms/CoilFourier.jl` · `project_normal_flux!` | **T·m² (= Wb)** | `2π·R[m]·(B_R·∂Z/∂θ − B_Z·∂R/∂θ)[m]` |
| `ForcingMode.amplitude` (Φ_x) | `ForcingTerms/CoilFourier.jl` · `fourier_decompose_bn` | **Wb** | Park 2007a Eq. (3): weight `w(θ)=1/(J\|∇ψ\|)` has units 1/area, so `Φ` is a flux representing `b·n̂` |
| `"normal_field_T"` file input | `ForcingTerms/ForcingTerms.jl` · `convert_forcing_normalization!` | **Wb** (after × 2πR·\|dr/dθ\|) | `B·n̂ [T] × area [m²]` |
| `"sfl_flux_Wb"` file input | `ForcingTerms/ForcingTerms.jl` · `convert_forcing_normalization!` | **Wb** (after × (2π)²) | flux input; (2π)² dimensionless |
| `Phi_x` / `forcing_vec` (eigenmode basis) | `PerturbedEquilibrium/ResponseMatrices.jl` · `map_forcing_to_eigenmodes` | **Wb** | copies amplitudes into basis slots |
| `Phi_xe`, `ftop` | — | — | **Needs review:** named in #239 but absent from the Julia code (Fortran-GPEC artefacts). Confirm whether an energy-norm flux variant (`Φ_xe`) is still required. |

### Response and inductance matrices

| Quantity | Source (file · symbol) | SI unit | Authority |
|----------|------------------------|---------|-----------|
| `plasma_inductance` (Λ) | `PerturbedEquilibrium/ResponseMatrices.jl` · `calc_plasma_inductance` | **H** | Park 2007a Eq. (6) `Φ = Λ·Iˣ`, Fig. 3 (eigenvalues in H ~3×10⁻⁴); `psio²/(μ₀·2)` carries `wt0` into henries |
| `surface_inductance` (L) | `PerturbedEquilibrium/ResponseMatrices.jl` · `compute_surface_inductance_from_greens` | **H** | Park 2007a Eq. (7) `Φˣ = L·Iˣ` (θ-reversal question #242 is orthogonal to units) |
| `permeability` (P = Λ·L⁻¹) | `PerturbedEquilibrium/Response.jl` | **dimensionless** | Park 2007a Eq. (8) `P = Λ·L⁻¹` = H/H |
| `reluctance` (ϱ = L⁻¹·(Λ−L)·L⁻¹) | `PerturbedEquilibrium/Response.jl` | **1/H** | (1/H)·(H)·(1/H) |

### Singular-coupling diagnostics

| Quantity | Source (file · symbol) | SI unit | Authority |
|----------|------------------------|---------|-----------|
| `Chirikov` parameter | `PerturbedEquilibrium/SingularCoupling.jl` · `chirikov_parameter` | **dimensionless** | island half-width / half-separation; numerator and denominator share the flux coordinate |
| `island_half_width` | `PerturbedEquilibrium/SingularCoupling.jl` · `island_half_width` | **√(normalized poloidal flux)** (ψ_N^{1/2}; not metres, not radians) | Fortran `gpout.f` annotates `singcoup(3)` "square of the penetrated island half-width in normalized poloidal flux"; Julia formula is identical |
| `delta_prime` (Δ′) | `PerturbedEquilibrium/SingularCoupling.jl` · `C_delta_prime` | **dimensionless** (by construction) | Park 2007a Eq. (1) defines the "dimensionless quantity Δ_mn"; this is GPEC's normalized-flux Δ′ and is **distinct** from the classical resistive-layer Δ′ (`1/m`, Glasser 2016) |
| `resonant_flux` (Φ_res) | `PerturbedEquilibrium/SingularCoupling.jl` · `C_resonant_flux` | **T** (area-normalized resonant flux Φ_r/A) | Fortran `gpout.f`: `singbnoflxs = singflx_mn/area ! units Tesla`; NetCDF `Phi_res` attr "resonant flux normalized by the surface area"; Park 2009 Eq. (4) weighted resonant field |
| `vacuum_energy` | `PerturbedEquilibrium/Response.jl` · ¼ Re⟨Φ_x, L⁻¹·Φ_x⟩ | **J** | Wb²/H = J; Park 2007a Eq. (31) `δW = ½Φ†Λ⁻¹Φ` (¼ from the symmetrized Eq. (29)); Fortran prints "Required energy to perturb vacuum" |
| `plasma_energy` | `PerturbedEquilibrium/Response.jl` · ¼ Re⟨Φ_tot, Λ⁻¹·Φ_tot⟩ | **J** | as `vacuum_energy`; Fortran's "total energy" ("Required energy to perturb plasma") |

### Open questions for the authors

The dimensional audit is otherwise settled by the citations above. Two items remain for author
confirmation (issue #239):

1. **`Phi_xe` / `ftop`** named in #239 are absent from the Julia code. Confirm whether the
   energy-norm flux variant (`Φ_xe`) is still required, or whether it was a Fortran-only
   intermediate now superseded by the unit-norm `Φ_x` convention used throughout.
2. **Numerical (not dimensional) agreement.** The units above are confirmed; whether the Julia
   output *magnitudes* for `resonant_flux`, `delta_prime`, `island_half_width`, and the energies
   match Fortran GPEC should be tracked in the regression harness, independent of this table.

!!! note "Convention reminder"
    `delta_prime` and `island_half_width` here use GPEC's **normalized-poloidal-flux** convention
    (Park 2007a), which is deliberately different from the classical real-space tearing
    quantities (Δ′ in 1/m, island width in m). Both `vacuum_energy` and `plasma_energy` are
    physical joules once `Φ_x` (Wb) and the inductances (H) are recognised.
