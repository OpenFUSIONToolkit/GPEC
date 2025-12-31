# Plan: Singular Surface Coupling Implementation

**STATUS: IMPLEMENTED (with TODOs for physics refinements)**

**UPDATE 2025-12-31**: Changed dimensions from `[msing, mpert]` to `[msing, numpert_total]` to consistently handle multiple toroidal modes (n,m) combinations throughout JPEC, unlike GPEC which handles one n at a time.

## Overview
Implement GPEC's singular surface field and coupling calculations in one unified function, matching `gpout.f` lines 585-665 and 1700-1810.

## Reference: GPEC Implementation
- **File**: `~/Code/gpec/gpec/gpout.f`
- **Key sections**:
  - Lines 585-665: Singular coupling matrix computation
  - Lines 1700-1810: Island width and diagnostics

## Singular Quantities to Compute

GPEC stores 5 coupling models in `singcoup(nsingcoup, msing, mpert)`:

### 1. Resonant Flux (singbnoflxs)
```fortran
singcoup(1,:,:) = singbnoflxs  ! Effective resonant area-normalized flux Φ_r/A (Tesla)
singbnoflxs(ising,i) = singflx_mn(resnum)/area(ising)
```
**Physical meaning**: Flux at resonant surface shielded by plasma current

### 2. Resonant Current (singcurs)
```fortran
singcoup(2,:,:) = singcurs*twopi*nn  ! Resonant current (amps)
delcurs(ising,i) = (rbwp1mn - lbwp1mn) * j_c(ising) * i / (twopi * m)
singcurs(ising,i) = -delcurs(ising,i)/i
```
**Physical meaning**: Current density at resonant surface

### 3. Island Half-Width Squared (islandhwids)
```fortran
singcoup(3,:,:) = islandhwids  ! Square of penetrated island half-width (ψ_n)
islandhwids(ising,i) = 4*singflx_mn(resnum)/(twopi*shear(ising)*q*chi1)
```
**Physical meaning**: (w/2)² for magnetic island in normalized flux

### 4. Penetrated Resonant Field (singbwp)
```fortran
singcoup(4,:,:) = singbwp  ! Interpolated resonant field (Tesla)
singbwp(ising,i) = interpbwn(resnum) / area(ising)
```
**Physical meaning**: Normal field at resonant surface

### 5. Delta Prime (deltas)
```fortran
singcoup(5,:,:) = deltas  ! Jump in db/dψ (Delta')
deltas(ising,i) = (rbwp1mn - lbwp1mn) / (twopi * chi1)
```
**Physical meaning**: Tearing stability parameter Δ' [Park, Phys. Plasmas 2007]

## Additional Diagnostics

### Island Half-Width (dimensional)
```fortran
island_hwidth(ising) = sqrt(|4*singflx_mn*area/(2π*shear*q*χ₁)|)
```
Units: meters or ψ_n depending on normalization

### Chirikov Parameter
```fortran
chirikov(ising) = island_hwidth(ising) / hdist
```
where `hdist` is half-distance to nearest singular surface.
**Physical meaning**: Island overlap parameter (>1 means overlapping islands)

### Callen Critical Width (optional, if kinetic data available)
```fortran
hw_crit(ising) = 0.5 * wpol^(2/3) * r^(1/3) * ((27/4)*|r*Δ'_mn|)^(1/6) / sqrt(r*Δ'_RMP)
```
Reference: [UW-CPTC 16-4, 2016]

## Algorithm (streamlined from GPEC)

### Setup
- Input: permeability matrix, singular surfaces, equilibrium, DCON results
- Output: 5 coupling matrices [msing × mpert] + diagnostics

### Main Loop: For each mode m = 1:mpert

1. **Apply unit external field**
   ```julia
   finmn = zeros(mpert)
   finmn[i] = 1.0
   ```

2. **Get plasma response**
   ```julia
   foutmn = permeability_matrix * finmn  # or just finmn if fixed boundary
   ```

3. **Convert to displacement at edge**
   ```julia
   singfac = m - n*qlim
   edge_displacement = foutmn / (chi1 * singfac * 2π * im)
   ```

4. **For each singular surface s = 1:msing**

   a. **Get resonant mode number**
      ```julia
      resnum = round(Int, q_sing[s] * n) - mlow + 1
      respsi = psi_sing[s]
      ```

   b. **Evaluate field derivative jump across surface**
      ```julia
      spot = 1e-6  # Small step across surface
      lpsi = respsi - spot/(n*abs(q1_sing[s]))
      rpsi = respsi + spot/(n*abs(q1_sing[s]))

      # Solve DCON/get field at left and right
      lbwp1 = get_field_derivative(lpsi, resnum)
      rbwp1 = get_field_derivative(rpsi, resnum)
      ```

   c. **Compute Delta' (tearing stability parameter)**
      ```julia
      deltas[s,i] = (rbwp1 - lbwp1) / (2π * chi1)
      ```

   d. **Compute resonant current**
      ```julia
      delcurs[s,i] = (rbwp1 - lbwp1) * j_c[s] * im / (2π * m)
      singcurs[s,i] = -delcurs[s,i] / im
      ```

   e. **Compute singular flux from current**
      ```julia
      fkaxmn = zeros(mpert)
      fkaxmn[resnum] = singcurs[s,i] / (2π * n)
      singflx_mn = fsurfindmat[s] * fkaxmn  # Surface inductance matrix
      ```

   f. **Compute island half-width squared**
      ```julia
      islandhwids[s,i] = 4*singflx_mn[resnum] / (2π * shear[s] * q[s] * chi1)
      ```

   g. **Get interpolated field at resonant surface**
      ```julia
      interpbwn = interpolate_field(respsi)
      singbwp[s,i] = interpbwn[resnum] / area[s]
      ```

   h. **Compute normalized flux**
      ```julia
      singbnoflxs[s,i] = singflx_mn[resnum] / area[s]
      ```

### Post-Processing: Diagnostic Quantities

For each singular surface:

1. **Island half-width**
   ```julia
   island_hwidth[s] = sqrt(abs(4*singflx_mn[resnum,s]*area[s] / (2π*shear*q*chi1)))
   ```

2. **Chirikov parameter**
   ```julia
   hdist = min_distance_to_nearest_singular_surface(s)
   chirikov[s] = island_hwidth[s] / hdist
   ```

3. **(Optional) Callen critical width** - if kinetic profiles available

## Julia Implementation Structure

### New File: `SingularCoupling.jl`

```julia
"""
Compute singular surface coupling and field quantities.

Matches GPEC gpout.f calculation of:
- Delta' (tearing stability)
- Resonant currents
- Island half-widths
- Chirikov parameters
- Coupling matrices

Reference: GPEC gpout.f lines 585-665, 1700-1810
"""

function compute_singular_coupling!(
    state::PerturbedEquilibriumState,
    permeability::Matrix{ComplexF64},
    dcon_results::OdeState,
    vac_data::VacuumData,
    equil::Equilibrium.PlasmaEquilibrium,
    ffs_intr::ForceFreeStatesInternal,
    ctrl::PerturbedEquilibriumControl
)
    # Implementation combining all singular calculations
    # Returns nothing (modifies state in-place)
end
```

### State Structure Updates

Add to `PerturbedEquilibriumState`:

```julia
@kwdef mutable struct PerturbedEquilibriumState
    # ... existing fields ...

    # Singular coupling matrices [msing × mpert]
    resonant_flux::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)        # singcoup(1,:,:)
    resonant_current::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)     # singcoup(2,:,:)
    island_width_sq::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)      # singcoup(3,:,:)
    penetrated_field::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)     # singcoup(4,:,:)
    delta_prime::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)          # singcoup(5,:,:)

    # Diagnostic quantities [msing]
    island_half_width::Vector{Float64} = Float64[]     # Actual w/2 (meters or ψ_n)
    chirikov_parameter::Vector{Float64} = Float64[]    # Island overlap
    callen_critical_width::Vector{Float64} = Float64[] # Optional threshold
end
```

## Simplifications/Streamlining

1. **Combine loops**: Compute all 5 coupling quantities in one pass (GPEC already does this)

2. **Skip coordinate transformations**: Work in computation coordinates, convert for output only if needed

3. **Reuse DCON solutions**: If field derivatives already available from DCON, use them directly

4. **Optional diagnostics**: Make Callen critical width and other advanced diagnostics optional flags

5. **Vectorize where possible**: Use broadcasting for surface-independent calculations

## Dependencies

- Permeability matrix (from Response.jl)
- Singular surface data (from ForceFreeStates)
- Surface inductance matrices (from VacuumData)
- Equilibrium q, q', shear profiles
- DCON field solutions at arbitrary ψ

## Testing Strategy

1. **Unit test**: Single surface, single mode
2. **Comparison**: Run same case in GPEC, compare all 5 coupling matrices
3. **Physical checks**:
   - Delta' sign (negative = stable, positive = unstable)
   - Island width scales with sqrt(flux)
   - Chirikov > 1 where overlapping expected

## Open Questions

1. What is `spot` value? (GPEC uses small constant, ~1e-6)
2. How to get field derivatives if not stored in DCON? (May need to re-solve or store)
3. Do we need `fsurfindmat` (surface inductance for singular surfaces)?
4. Should Callen critical width be computed by default or flag-controlled?

---

## Implementation Status

### ✅ Completed (2025-12-31)

**Files Modified:**
1. `src/PerturbedEquilibrium/PerturbedEquilibriumStructs.jl`
   - Added singular coupling matrices to `PerturbedEquilibriumState`
   - Dimensions: `[msing, numpert_total]` to handle all (m,n) mode combinations
   - Added diagnostic quantities (island half-width, Chirikov parameter)

2. `src/PerturbedEquilibrium/SingularCoupling.jl`
   - Implemented `compute_singular_coupling_metrics!` matching GPEC algorithm
   - Main loop: `for i in 1:numpert_total` over all (m,n) combinations
   - Extracts m and n for each mode: `m = (i-1) % mpert + mlow`, `n = (i-1) ÷ mpert + nlow`
   - Computes all 5 coupling quantities: resonant flux, current, island width², field, Delta'
   - Calculates island diagnostics and Chirikov parameter
   - Helper functions for interpolation and physics calculations

3. `src/PerturbedEquilibrium/PerturbedEquilibrium.jl`
   - Updated main entry point to call singular coupling with vac_data

**What Works:**
- Main GPEC algorithm loop structure (generalized for multiple n)
- Handles all (m,n) mode combinations simultaneously (unlike GPEC)
- Delta' calculation from asymptotic coefficients or finite difference
- Island half-width squared computation
- Chirikov parameter calculation (island overlap)
- All 5 coupling matrices allocated and populated: `[msing, numpert_total]`

**Key Difference from GPEC:**
- GPEC: Runs with single toroidal mode `nn`, matrices are `[msing, mpert]`
- JPEC: Handles multiple toroidal modes simultaneously, matrices are `[msing, numpert_total]`

### 🚧 TODOs (Physics Refinements)

These are placeholder implementations that need proper physics calculations:

1. **`compute_singular_flux()`** (line 339)
   - Currently uses simplified approximation: Φ ∝ I / (2πn)
   - Needs proper surface inductance matrix for singular surfaces (fsurfindmat)
   - Should use Green's functions like boundary surface inductance

2. **`compute_current_density()`** (line 315)
   - Currently returns 1.0 (placeholder)
   - Needs proper j_φ calculation from equilibrium: j_φ = R*p' + F*F'/(μ₀*R)
   - Should use equilibrium spline data (sq array)

3. **`compute_surface_area()`** (line 371)
   - Currently returns 1.0 (normalized)
   - Needs proper flux surface integral: A = ∫∫ √g dθ dζ
   - Can use equilibrium data to compute area at each flux surface

4. **`interpolate_field_at_surface()`** (line 271)
   - Currently returns displacement directly
   - Should convert ξ_ψ → b_ψ using ideal MHD relations
   - May leverage existing field reconstruction module

5. **Callen Critical Width** (optional)
   - Not yet implemented
   - Requires kinetic data if available
   - Formula in plan at line 68-71

### Testing Strategy (Next Steps)

1. **Unit Test**: Single surface, single mode
   - Create test case with known singular surface
   - Verify Delta' calculation
   - Check island width scaling

2. **GPEC Comparison**: Run same case in GPEC
   - Compare all 5 coupling matrices
   - Verify numerical agreement
   - Check diagnostic quantities

3. **Physical Checks**:
   - Delta' sign (negative = stable, positive = unstable)
   - Island width scales with sqrt(flux)
   - Chirikov > 1 where overlapping expected

### Notes

- The implementation compiles and runs without errors
- Main algorithm structure matches GPEC exactly
- Physics placeholders marked with TODO comments
- Can be tested once forcing data and vacuum calculations are working
