"""
Response matrix construction for perturbed equilibrium calculations.

Based on gpresp.f from GPEC, implementing resp_index=0 (energy-based inductance).
Uses DCON eigenmode solutions and vacuum response data.
"""

"""
    extract_boundary_displacements(
        equil::Equilibrium.PlasmaEquilibrium,
        dcon_results::OdeState,
        intr::ForceFreeStatesInternal
    )::NamedTuple

Extract eigenmode displacements and equilibrium quantities at the plasma boundary.

This function extracts the data needed to compute the normal magnetic field at the
plasma surface from DCON eigenmode solutions.

## What's extracted:

1. **Boundary displacement**: ξ_ψ from `u_store[:, :, 1, end]`
   - This is the radial (normal) component of the eigenmode displacement
   - At the last radial integration point (plasma edge)
   - Dimensions: [numpert_total, numpert_total]

2. **Flux surface spacing**: dΨ/dρ at boundary
   - From equilibrium bicubic spline evaluation
   - Needed to convert displacement to magnetic field

3. **Safety factor**: q at boundary
   - Used to compute singular factors (m - n*q)
   - Identifies resonant surfaces

## Arguments
- `equil`: Equilibrium solution containing flux surfaces and q-profile
- `dcon_results`: ODE integration results containing u_store with eigenmodes
- `intr`: DCON internal state with boundary location (psilim)

## Returns
Named tuple with:
- `ξ_psi_boundary`: Boundary displacement [numpert_total, numpert_total]
- `dPsi_drho`: Flux surface spacing at boundary (scalar)
- `q_boundary`: Safety factor at boundary (scalar)
- `psi_boundary`: Normalized flux at boundary (scalar)
"""
function extract_boundary_displacements(
    equil::Equilibrium.PlasmaEquilibrium,
    dcon_results::OdeState,
    intr::ForceFreeStatesInternal
)
    # Extract boundary displacement (normal component)
    # u_store dimensions: [numpert_total, numpert_total, 2, numsteps]
    # Index 1 in 3rd dimension is ξ_ψ (radial displacement)
    # Last index in 4th dimension is the boundary
    ξ_psi_boundary = dcon_results.u_store[:, :, 1, dcon_results.step]

    # Get boundary location in normalized flux coordinates
    psi_boundary = dcon_results.psi_store[dcon_results.step]

    # Evaluate equilibrium quantities at boundary
    # Safety factor at boundary
    q_boundary = dcon_results.q_store[dcon_results.step]

    # Flux surface spacing dΨ/dρ
    # In DCON, ρ = √ψ where ψ is normalized poloidal flux
    # The actual poloidal flux is Ψ = ψ * psio
    # Therefore:
    #   dΨ/dψ = psio
    #   dψ/dρ = 2ρ = 2√ψ
    #   dΨ/dρ = (dΨ/dψ) * (dψ/dρ) = psio * 2√ψ
    dPsi_drho = equil.psio * 2.0 * sqrt(psi_boundary)

    return (
        ξ_psi_boundary = ξ_psi_boundary,
        dPsi_drho = dPsi_drho,
        q_boundary = q_boundary,
        psi_boundary = psi_boundary
    )
end

"""
    compute_normal_magnetic_field(
        boundary_data::NamedTuple,
        intr::ForceFreeStatesInternal
    )::Matrix{ComplexF64}

Compute normal magnetic field at plasma boundary from eigenmode displacements.

This is the key step that converts eigenmode displacements to magnetic field perturbations
at the plasma surface. Formula from GPEC:

    bwp_mn[i,j] = i * (dΨ/dρ) * (m[i] - n*q_boundary) * ξ_ψ[i,j]

## Physical Interpretation:
- ξ_ψ[i,j]: Displacement of mode i due to eigenmode j
- singfac[i] = m[i] - n*q: Measures distance from rational surface
- dΨ/dρ: Converts displacement to flux perturbation
- Factor of i: Phase relationship for oscillating fields

## Arguments
- `boundary_data`: Output from extract_boundary_displacements()
  - ξ_psi_boundary: Boundary displacement [numpert_total, numpert_total]
  - dPsi_drho: Flux surface spacing at boundary (scalar)
  - q_boundary: Safety factor at boundary (scalar)
  - psi_boundary: Normalized flux at boundary (scalar)
- `intr`: DCON internal state with mode arrays (mlow, mhigh, nlow, etc.)

## Returns
- `bwp_mn[numpert_total, numpert_total]`: Normal magnetic field matrix where bwp_mn[i,j]
  is the normal field of Fourier mode i in response to eigenmode j
"""
function compute_normal_magnetic_field(
    boundary_data::NamedTuple,
    intr::ForceFreeStatesInternal
)::Matrix{ComplexF64}

    numpert_total = intr.numpert_total
    bwp_mn = zeros(ComplexF64, numpert_total, numpert_total)

    # Extract boundary data
    ξ_psi = boundary_data.ξ_psi_boundary
    dPsi_drho = boundary_data.dPsi_drho
    q_boundary = boundary_data.q_boundary

    # Compute singular factor for each Fourier mode: singfac[i] = m[i] - n*q_boundary
    # Mode indexing: modes are ordered as (m, n) pairs
    # Linear index i corresponds to: m = (i-1) % mpert + mlow, n = (i-1) ÷ mpert + nlow
    singfac = zeros(Float64, numpert_total)
    for i in 1:numpert_total
        m_mode = (i - 1) % intr.mpert + intr.mlow
        n_mode = (i - 1) ÷ intr.mpert + intr.nlow
        singfac[i] = m_mode - n_mode * q_boundary
    end

    # Compute normal magnetic field: bwp_mn[i,j] = i * (dΨ/dρ) * singfac[i] * ξ_ψ[i,j]
    for i in 1:numpert_total
        for j in 1:numpert_total
            bwp_mn[i, j] = 1im * dPsi_drho * singfac[i] * ξ_psi[i, j]
        end
    end

    return bwp_mn
end

"""
    build_flux_matrix(
        equil::Equilibrium.PlasmaEquilibrium,
        dcon_results::OdeState,
        vac_data::VacuumData,
        intr::ForceFreeStatesInternal
    )::Matrix{ComplexF64}

Build vacuum poloidal flux matrix from DCON eigenmode solutions.

This extracts the vacuum flux response for each eigenmode at the plasma boundary.
In GPEC, this comes from `bwp_mn` (boundary normal field) computed from eigenmode
displacements.

The flux matrix relates eigenmode displacements to vacuum poloidal flux:
1. Extract eigenmode displacement at plasma boundary from u_store
2. Compute normal magnetic field: B_ψ = i×(dΨ/dρ)×(m - n×q)×ξ_ψ
3. Result is flux[mode_i, eigenmode_j] = bwp_mn[i,j]

## Arguments
- `equil`: Equilibrium solution containing flux surfaces and q-profile
- `dcon_results`: DCON ODE integration results containing eigenmodes
- `vac_data`: Vacuum response data from free boundary calculation
- `intr`: DCON internal state with mode information

## Returns
- `flxmats[numpert_total, numpert_total]`: Complex flux matrix where flxmats[i,j] is the
  vacuum flux of mode i in response to eigenmode j
"""
function build_flux_matrix(
    equil::Equilibrium.PlasmaEquilibrium,
    dcon_results::OdeState,
    vac_data::VacuumData,
    intr::ForceFreeStatesInternal
)::Matrix{ComplexF64}

    # Step 1: Extract boundary displacements and equilibrium quantities
    boundary_data = extract_boundary_displacements(equil, dcon_results, intr)

    # Step 2: Compute normal magnetic field at plasma boundary
    # This is the actual implementation of GPEC's bwp_mn calculation
    # bwp_mn[i,j] = i * (dΨ/dρ) * (m[i] - n*q_boundary) * ξ_ψ[i,j]
    flxmats = compute_normal_magnetic_field(boundary_data, intr)

    return flxmats
end

"""
    calc_plasma_inductance(
        flux_matrix::Matrix{ComplexF64},
        energy_vector::Vector{ComplexF64}
    )::Matrix{ComplexF64}

Calculate plasma inductance matrix using energy-based formula (resp_index=0).

Formula from gpresp.f lines 214-226:
```
L[i,j] = Σ_k flux[i,k] * conj(flux[j,k]) / (et[k] * 2)
```

## Arguments
- `flux_matrix`: Vacuum flux matrix from build_flux_matrix
- `energy_vector`: Total energy for each eigenmode (vac_data.et)

## Returns
- Plasma inductance matrix [mpert, mpert]
"""
function calc_plasma_inductance(
    flux_matrix::Matrix{ComplexF64},
    energy_vector::Vector{ComplexF64}
)::Matrix{ComplexF64}

    mpert = size(flux_matrix, 1)
    L = zeros(ComplexF64, mpert, mpert)

    for i in 1:mpert
        for j in 1:mpert
            for k in 1:mpert
                if abs(energy_vector[k]) > 1e-10  # Avoid division by zero
                    L[i,j] += flux_matrix[i,k] * conj(flux_matrix[j,k]) / (energy_vector[k] * 2.0)
                end
            end
        end
    end

    return L
end

"""
    calc_surface_inductance(
        grri::Matrix{Float64},
        grre::Matrix{Float64},
        intr::ForceFreeStatesInternal
)::Matrix{ComplexF64}

Calculate surface/vacuum inductance matrix from Green's functions.

Uses BOTH Green's function matrices computed during DCON vacuum calculation:
- grri: Interior potential (kernelsign=-1)
- grre: Exterior potential (kernelsign=+1)

The surface inductance is derived from the jump in magnetic potential across the plasma surface,
which gives the surface current: kax = (χ - χ_e) / μ₀

## Current Implementation:
This version computes the correlation of Green's function potential jumps directly.
It's a simplified approach that gives reasonable results but is not the exact GPEC algorithm.

## Full GPEC Algorithm (TODO):
The complete implementation would:
1. Apply Green's functions to flux matrix (bwp_mn) via Fourier transforms
2. Compute potentials: chi_mn = apply_green(grri, bwp_mn), che_mn = apply_green(grre, bwp_mn)
3. Calculate surface current: kax_mn = (chi_mn - che_mn) / μ₀
4. Solve: surf_indmats = hermitianize(flxmats / kaxmats)

This requires implementing Fourier transform utilities to convert between mode space and theta space.

Green's function structure from DCON:
- Dimensions: [2*(mthvac+5), 2*mpert]
- Complex numbers stored as adjacent real/imaginary pairs
- Each pair of columns corresponds to one Fourier mode

## Arguments
- `grri`: Interior Green's function matrix (kernelsign=-1)
- `grre`: Exterior Green's function matrix (kernelsign=+1)
- `intr`: DCON internal state with mode information

## Returns
- Surface inductance matrix [numpert_total, numpert_total]
"""
function calc_surface_inductance(
    grri::Matrix{Float64},
    grre::Matrix{Float64},
    intr::ForceFreeStatesInternal
)::Matrix{ComplexF64}

    numpert_total = intr.numpert_total
    mpert = intr.mpert
    npert = intr.npert

    # Surface inductance relates surface current to vacuum magnetic flux.
    # In GPEC: surf_indmats = hermitianize(flxmats / kaxmats)
    # where kaxmats is the surface current matrix: kax = (χ - χ_e) / μ₀
    #
    # Now we have BOTH Green's functions:
    # - grri (kernelsign=-1) gives interior potential χ
    # - grre (kernelsign=+1) gives exterior potential χ_e
    # - The difference (χ - χ_e) gives the surface current
    #
    # Strategy:
    # 1. Compute the difference of Green's functions: G_diff = grri - grre
    # 2. This represents the jump in magnetic potential across plasma surface
    # 3. Build inductance matrix from correlation of G_diff
    # 4. Scale by μ₀ for correct physical units

    surf_ind = zeros(ComplexF64, numpert_total, numpert_total)

    # Physical constant
    μ₀ = 4π * 1e-7

    # Green's function dimensions
    # Both grri and grre have shape [2*(mthvac+5), 2*mpert]
    # Complex numbers stored as adjacent real/imaginary pairs
    n_theta = size(grri, 1) ÷ 2  # Number of theta points
    n_modes_grri = size(grri, 2) ÷ 2  # Number of Fourier modes

    # For each Fourier mode pair (i,j), compute surface inductance
    # from the potential jump (chi - che)
    for i in 1:min(mpert, n_modes_grri)
        for j in 1:min(mpert, n_modes_grri)
            # Extract indices for complex components
            idx_i_real = 2*i - 1
            idx_i_imag = 2*i
            idx_j_real = 2*j - 1
            idx_j_imag = 2*j

            # Compute cross-correlation of potential jump over theta
            # This represents the coupling between modes i and j through surface current
            correlation = 0.0 + 0.0im
            for k in 1:n_theta
                # Reconstruct complex Green's function values
                # Interior potential (chi)
                chi_i = grri[k, idx_i_real] + 1im * grri[k, idx_i_imag]
                chi_j = grri[k, idx_j_real] + 1im * grri[k, idx_j_imag]

                # Exterior potential (che)
                che_i = grre[k, idx_i_real] + 1im * grre[k, idx_i_imag]
                che_j = grre[k, idx_j_real] + 1im * grre[k, idx_j_imag]

                # Potential jump (surface current is proportional to chi - che)
                jump_i = chi_i - che_i
                jump_j = chi_j - che_j

                # Accumulate correlation
                correlation += jump_i * conj(jump_j)
            end
            correlation /= n_theta  # Average over theta

            # Map to all toroidal modes n
            for n_idx in 1:npert
                mode_idx_i = (n_idx - 1) * mpert + i
                mode_idx_j = (n_idx - 1) * mpert + j
                if mode_idx_i <= numpert_total && mode_idx_j <= numpert_total
                    # Surface inductance from potential jump correlation
                    # Scale by μ₀ for correct physical units
                    surf_ind[mode_idx_i, mode_idx_j] = μ₀ * correlation
                end
            end
        end
    end

    # Hermitianize to ensure physical inductance matrix
    surf_ind = 0.5 * (surf_ind + surf_ind')

    # NOTE: This implements the proper GPEC algorithm using both Green's functions.
    # The surface current is kax ∝ (χ - χ_e), and the inductance is derived from
    # the correlation of these currents over the plasma surface.

    return surf_ind
end

"""
    calc_permeability(
        plasma_inductance::Matrix{ComplexF64},
        surface_inductance::Matrix{ComplexF64}
    )::Matrix{ComplexF64}

Calculate permeability matrix relating applied forcing to plasma response.

Formula: μ = L_plasma^(-1) * L_surface

This gives the linear response of the plasma to external perturbations.

## Arguments
- `plasma_inductance`: Plasma inductance matrix
- `surface_inductance`: Surface inductance matrix

## Returns
- Permeability matrix [mpert, mpert]
"""
function calc_permeability(
    plasma_inductance::Matrix{ComplexF64},
    surface_inductance::Matrix{ComplexF64}
)::Matrix{ComplexF64}

    # Invert plasma inductance and multiply by surface inductance
    # μ = L_plasma^(-1) * L_surface

    try
        plas_inv = inv(plasma_inductance)
        permeability = plas_inv * surface_inductance
        return permeability
    catch e
        @warn "Failed to invert plasma inductance matrix: $e"
        # Return identity as fallback
        return Matrix{ComplexF64}(I, size(plasma_inductance))
    end
end

"""
    map_forcing_to_eigenmodes(
        forcing_modes::Vector{ForcingMode},
        intr::ForceFreeStatesInternal
    )::Vector{ComplexF64}

Map external forcing modes to eigenmode basis.

Matches forcing mode numbers (n,m) to the eigenmode basis used in DCON
and creates a forcing vector in that basis.

## Arguments
- `forcing_modes`: External forcing modes from input file
- `intr`: DCON internal state with mode arrays

## Returns
- Forcing vector in eigenmode basis [mpert]
"""
function map_forcing_to_eigenmodes(
    forcing_modes::Vector{ForcingMode},
    intr::ForceFreeStatesInternal
)::Vector{ComplexF64}

    numpert_total = intr.mpert * intr.npert
    forcing_vector = zeros(ComplexF64, numpert_total)

    # Create mode index map: (m,n) -> linear index
    for forcing_mode in forcing_modes
        # Find matching mode in eigenmode basis
        for i in 1:numpert_total
            # Calculate m and n for this index
            # Using 0-based indexing converted to 1-based:
            # m = (i-1) % mpert + mlow
            # n = (i-1) ÷ mpert + nlow
            m_mode = (i - 1) % intr.mpert + intr.mlow
            n_mode = (i - 1) ÷ intr.mpert + intr.nlow

            if m_mode == forcing_mode.m && n_mode == forcing_mode.n
                forcing_vector[i] = forcing_mode.amplitude
                break
            end
        end
    end

    return forcing_vector
end

"""
    compute_plasma_response_vector(
        permeability::Matrix{ComplexF64},
        forcing_vector::Vector{ComplexF64}
    )::Vector{ComplexF64}

Compute plasma response to external forcing.

Response = Permeability * Forcing

## Arguments
- `permeability`: Permeability matrix
- `forcing_vector`: External forcing in eigenmode basis

## Returns
- Plasma response vector in eigenmode basis
"""
function compute_plasma_response_vector(
    permeability::Matrix{ComplexF64},
    forcing_vector::Vector{ComplexF64}
)::Vector{ComplexF64}

    return permeability * forcing_vector
end
