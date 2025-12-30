# GPEC Response Implementation Plan

## Overview

Implementing plasma response calculations from `gpresp.f`, focusing on `resp_index=0` (energy-based inductance) and excluding Galerkin logic.

## Key Subroutines from gpresp.f

### Execution Order:
1. **gpresp_eigen** - Build matrices from DCON eigenmodes
2. **gpresp_pinduct** - Plasma inductance matrices
3. **gpresp_sinduct** - Self/surface inductance
4. **gpresp_permeab** - Permeability matrices
5. **gpresp_reluct** - Reluctance (resp_index specific)
6. **gpresp_minduct** - Mutual inductance

## resp_index=0 Specifics

`resp_index=0` uses **energy-based inductance**:
- Lines 214-226 in gpresp.f
- Formula: `L(i,j) = Σ_k flux(i,k) * conj(flux(j,k)) / (E_k * 2)`
- Uses total energy `et(k)` from DCON for normalization

Other resp_index values (1-4) use different magnetic scalar potentials - we skip these.

## Implementation Strategy

### Phase 1: Core Matrix Construction (resp_index=0 only)

**Input from DCON:**
- Eigenmode solutions: `odet.u_store` (displacement)
- Energy matrix: From vacuum calculation (wt matrices)
- Mode numbers: `m`, `n` arrays
- Singular surface data: `intr.sing`

**Build Matrices:**

1. **Flux Matrix** (`flxmats[mpert, mpert]`):
   - Vacuum poloidal flux from each eigenmode
   - Needs vacuum response from DCON results

2. **Plasma Inductance** (`plas_indmats[mpert, mpert]`):
   ```julia
   for i in 1:mpert, j in 1:mpert
       for k in 1:mpert
           L[i,j] += flxmats[i,k] * conj(flxmats[j,k]) / (et[k] * 2.0)
       end
   end
   ```

3. **Surface/Self Inductance** (`surf_indmats[mpert, mpert]`):
   - From vacuum Green's function
   - Already have `grri` from vacuum calculation

4. **Permeability Matrix** (`permeabmats[mpert, mpert]`):
   ```julia
   μ = inv(L_plasma) * L_surf
   ```

### Phase 2: Apply Forcing and Compute Response

**Input:**
- Forcing modes: `n`, `m`, `amplitude` from `forcing.dat`

**Calculation:**
```julia
# Match forcing modes to eigenmode basis
forcing_vector = map_forcing_to_modes(forcing_modes, m_array, n_array)

# Compute response
response = permeab_matrix * forcing_vector
```

### Phase 3: Extract Physical Quantities

- Perturbed displacement fields
- Perturbed magnetic fields
- Coupling to singular surfaces (key GPEC output)

## Missing Components / Questions

### What we have from DCON:
- ✅ Eigenmode solutions (`u_store`, `ud_store`)
- ✅ Energy matrix (`wt`, `wt0` from vacuum)
- ✅ Singular surface locations
- ✅ Mode numbers (m, n)

### What we need to extract/calculate:
- ❓ Vacuum poloidal flux from eigenmodes → Need to calculate from `u_store`
- ❓ Surface quantities at plasma boundary → May need additional equilibrium evaluation
- ❓ Green's function matrices → Have `grri` from vacuum, need to use correctly

### Simplifications for MWE:
- Skip Galerkin projection (gal_flag = false)
- Skip alternate resp_index values (only use 0)
- Skip error estimation initially
- Focus on single toroidal mode (n=1)

## Proposed Julia Implementation Structure

### New file: `src/PerturbedEquilibrium/ResponseMatrices.jl`

```julia
"""Build flux matrix from DCON eigenmode solutions"""
function build_flux_matrix(
    equil::Equilibrium.PlasmaEquilibrium,
    dcon_results::OdeState,
    intr::DconInternal
)::Matrix{ComplexF64}
    # Extract vacuum flux from eigenmodes
    # TODO: Implement
end

"""Calculate plasma inductance matrix (resp_index=0)"""
function calc_plasma_inductance(
    flux_matrix::Matrix{ComplexF64},
    energy_vector::Vector{ComplexF64}
)::Matrix{ComplexF64}
    mpert = size(flux_matrix, 1)
    L = zeros(ComplexF64, mpert, mpert)
    for i in 1:mpert, j in 1:mpert
        for k in 1:mpert
            L[i,j] += flux_matrix[i,k] * conj(flux_matrix[j,k]) / (energy_vector[k] * 2.0)
        end
    end
    return L
end

"""Calculate surface/vacuum inductance matrix"""
function calc_surface_inductance(
    grri::Matrix{Float64},
    m_array::Vector{Int},
    n::Int
)::Matrix{ComplexF64}
    # Use Green's function from vacuum calculation
    # TODO: Implement
end

"""Calculate permeability matrix"""
function calc_permeability(
    plasma_inductance::Matrix{ComplexF64},
    surface_inductance::Matrix{ComplexF64}
)::Matrix{ComplexF64}
    return inv(plasma_inductance) * surface_inductance
end

"""Map forcing modes to eigenmode basis"""
function map_forcing_to_eigenmodes(
    forcing_modes::Vector{ForcingMode},
    m_array::Vector{Int},
    n_array::Vector{Int}
)::Vector{ComplexF64}
    # TODO: Implement mode matching
end
```

## Questions for Clarification

1. Do we have all necessary vacuum quantities from the DCON vacuum calculation?
2. Should we extract additional surface quantities at the plasma boundary?
3. For the flux matrix, do we evaluate eigenmodes at the boundary or use existing vacuum response?
4. Is the Green's function `grri` from DCON's vacuum calculation sufficient?

## Next Steps

1. Identify what DCON outputs we can reuse vs. what needs new calculation
2. Implement `build_flux_matrix()` to extract vacuum flux from eigenmodes
3. Implement inductance and permeability calculations
4. Test with DIIID-like example (n=1, m=6 forcing)
5. Verify coupling to singular surfaces

---

**Note**: This is complex physics. We should implement incrementally and validate against Fortran GPEC outputs at each step.

## Current Implementation Status (December 2024)

### What's Implemented

1. **Response calculation framework** - Complete workflow in `Response.jl` and `ResponseMatrices.jl`
2. **Forcing data input** - ASCII and HDF5 formats with comment support
3. **Plasma inductance** - Energy-based formula (resp_index=0) fully implemented
4. **Flux matrix** - Improved approximation using DCON eigenvectors
5. **Surface inductance** - Green's function-based approximation
6. **Permeability calculation** - Matrix inversion and multiplication
7. **Mode mapping** - Forcing modes to eigenmode basis
8. **HDF5 output** - Results written to euler.h5

### Current Approximations and Limitations

#### 1. Flux Matrix (`build_flux_matrix`)
**Current implementation:**
```julia
flxmats[:, k] = vac_data.wv * vac_data.wt[:, k]
```

**Physical meaning:** Projects eigenmode k through vacuum energy coupling matrix to get vacuum flux response.

**Limitations:**
- Does not extract actual boundary displacement from `u_store`
- Does not compute normal magnetic field: B_ψ = i*(dΨ/dρ)*(m - n*q)*ξ_ψ
- Missing singular factor (m - n*q) evaluation at boundary

**Full GPEC algorithm would:**
1. Extract boundary displacement: `ξ_boundary = u_store[:, :, :, end]`
2. Evaluate flux surface spacing: `dΨ/dρ` at boundary
3. Compute singular factors for each mode: `singfac[i] = m[i] - n*q_boundary`
4. Calculate normal field: `bwp_mn = i * (dΨ/dρ) * singfac * ξ_ψ`

#### 2. Surface Inductance (`calc_surface_inductance`)
**Current implementation:**
- Uses cross-correlation of Green's function over poloidal angle
- Computes: `correlation = Σ_θ G_i(θ) * G_j*(θ) / n_theta`
- Builds hermitianized inductance: `L_surf = hermitianize(μ₀ * correlation)`

**Limitations:**
- DCON only computes ONE Green's function (grri with `kernelsign=1.0`)
- Full GPEC needs TWO Green's functions:
  - `grri` with `kernelsign=-1` for interior potential χ
  - `grre` with `kernelsign=+1` for exterior potential χ_e
- Missing surface current calculation: `kax = (χ - χ_e) / μ₀`

**Full GPEC algorithm would:**
1. Run vacuum solver twice with different kernel signs
2. Compute interior and exterior potentials for each eigenmode
3. Calculate surface current matrix: `kaxmats = (chimats - chemats) / μ₀`
4. Compute surface inductance: `surf_indmats = hermitianize(flxmats / kaxmats)`

#### 3. Green's Function Structure
**DCON provides:**
- `vac_data.grri`: Green's function matrix [2*(mthvac+5), 2*mpert]
- Computed with `kernelsign = 1.0` (default in VacuumInput)
- Complex numbers stored as adjacent real/imaginary pairs

**What's missing:**
- Green's function with `kernelsign = -1`
- This requires modifying DCON vacuum calculation to compute both

#### 4. Eigenmode Boundary Conditions
**DCON provides:**
- `odet.u_store[:, :, :, end]`: Eigenmode displacements at boundary
- `vac_data.wt`: Eigenvectors of total energy matrix
- `vac_data.wv`: Vacuum energy matrix (already scaled by singfac²)

**What needs extraction:**
- Actual boundary displacement ξ_ψ (normal component)
- Conversion to magnetic field perturbation
- Proper normalization and phase conventions

### Performance Characteristics

**Current implementation:**
- Successfully runs on DIIID-like example (n=1, m=6)
- Response amplitude: ~6.2e-11 (with improved matrices)
- Computation time: ~0.16 seconds (perturbed equilibrium portion)
- Matrix sizes: 34×34 for mpert=34, npert=1

**Scaling:**
- Computational cost: O(numpert_total³) for matrix operations
- Memory: O(numpert_total²) for storing matrices
- For multi-n cases, numpert_total = mpert × npert grows linearly with npert

### Path to Full Implementation

To achieve full GPEC-equivalent accuracy, the following enhancements are needed:

1. **Modify vacuum calculation** (in `DCON/Free.jl`):
   - Compute vacuum response with both `kernelsign = -1` and `kernelsign = +1`
   - Store both `grri` and `grre` in `VacuumData` structure
   - This requires two calls to `Vacuum.compute_vacuum_response()`

2. **Extract boundary displacements** (in `ResponseMatrices.jl`):
   - Read `u_store[:, :, :, end]` to get boundary values
   - Extract normal component (ξ_ψ) for each eigenmode
   - Evaluate equilibrium quantities at boundary (dΨ/dρ, q)

3. **Compute normal magnetic field** (in `ResponseMatrices.jl`):
   - Calculate singular factors: `singfac[i] = m[i] - n*q_boundary`
   - Apply formula: `bwp_mn = i * (dΨ/dρ) * singfac * ξ_ψ`
   - Handle resonant modes (singfac ≈ 0) carefully

4. **Build proper surface inductance** (in `ResponseMatrices.jl`):
   - Compute interior potential: `chi_mn = apply_green(grri, bwp_mn)`
   - Compute exterior potential: `che_mn = apply_green(grre, bwp_mn)`
   - Calculate surface current: `kax_mn = (chi_mn - che_mn) / μ₀`
   - Solve: `surf_indmats = hermitianize(flxmats / kaxmats)`

5. **Add Fourier transform utilities** (new file):
   - Implement discrete Fourier transform (equivalent to GPEC's `iscdftf`)
   - Handle toroidal phase corrections: `exp(-i*n*ϕ(θ))`
   - Reverse theta convention between plasma and vacuum codes

### Validation Strategy

To validate the full implementation:

1. **Energy conservation**: Check `surfei + surfee = vacuum energy from DCON`
2. **Hermiticity**: Verify all inductance matrices are Hermitian
3. **Comparison with Fortran GPEC**: Run identical cases and compare:
   - Flux matrix elements
   - Plasma and surface inductance matrices
   - Permeability matrix
   - Final response amplitudes
4. **Physical limits**: Test edge cases (no wall, far wall, resistive wall)

### References

- **Fortran GPEC source**: `/Users/nlogan/Code/gpec/gpec/gpresp.f`
- **Key subroutines**: `gpresp_eigen`, `gpresp_pinduct`, `gpresp_sinduct`, `gpresp_permeab`
- **Vacuum calculation**: `/Users/nlogan/Code/gpec/gpec/gpeq.f` (lines 506-640)
- **Green's function setup**: `/Users/nlogan/Code/gpec/gpec/idcon.f` (lines 890-932)
