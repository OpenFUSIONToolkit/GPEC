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
