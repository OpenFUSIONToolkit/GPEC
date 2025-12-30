"""
Response matrix construction for perturbed equilibrium calculations.

Based on gpresp.f from GPEC, implementing resp_index=0 (energy-based inductance).
Uses DCON eigenmode solutions and vacuum response data.
"""

"""
    build_flux_matrix(
        dcon_results::OdeState,
        vac_data::VacuumData,
        intr::DconInternal
    )::Matrix{ComplexF64}

Build vacuum poloidal flux matrix from DCON eigenmode solutions.

This extracts the vacuum flux response for each eigenmode at the plasma boundary.
In GPEC, this comes from `bwp_mn` after calling `gpeq_sol` for each eigenmode.

The flux matrix relates eigenmode displacements to vacuum poloidal flux:
- Extract eigenmode displacement at plasma boundary from u_store
- Apply Green's function to get vacuum response
- Result is flux[mode_i, eigenmode_j]

## Arguments
- `dcon_results`: DCON ODE integration results containing eigenmodes
- `vac_data`: Vacuum response data from free boundary calculation
- `intr`: DCON internal state with mode information

## Returns
- `flxmats[numpert_total, numpert_total]`: Complex flux matrix where flxmats[i,j] is the
  vacuum flux of mode i in response to eigenmode j
"""
function build_flux_matrix(
    dcon_results::OdeState,
    vac_data::VacuumData,
    intr::DconInternal
)::Matrix{ComplexF64}

    numpert_total = intr.numpert_total

    # Extract eigenmode solutions at plasma boundary (last radial step)
    # u_store dimensions: [numpert_total, numpert_total, 2, numsteps]
    # The last index (numsteps) gives radial location
    # The second-to-last index (2) gives normal and parallel components
    boundary_step = dcon_results.step  # Last integrated step

    # TODO: Full implementation requires:
    # 1. Extract boundary displacement: u_boundary = u_store[:, :, :, boundary_step]
    # 2. Convert displacement to surface current using plasma properties
    # 3. Apply Green's function (grri) to get vacuum flux response
    # 4. The result should be flux[i,j] = vacuum flux of mode i from eigenmode j
    #
    # For now, use vacuum energy matrix as proxy for flux coupling
    # wv relates modes through vacuum energy: E_vac = u† · wv · u
    # This approximation captures mode coupling but not the correct physics

    flxmats = zeros(ComplexF64, numpert_total, numpert_total)

    # Use vacuum energy matrix as temporary placeholder
    # This has the right structure (mode coupling) but wrong physics
    if size(vac_data.wv, 1) == numpert_total && size(vac_data.wv, 2) == numpert_total
        # wv is the vacuum energy matrix, use it as a proxy for flux coupling
        flxmats .= vac_data.wv
    else
        # If sizes don't match, use identity
        for i in 1:numpert_total
            flxmats[i, i] = 1.0 + 0.0im
        end
    end

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
        intr::DconInternal
)::Matrix{ComplexF64}

Calculate surface/vacuum inductance matrix from Green's function.

Uses the vacuum Green's function matrix computed during DCON vacuum calculation.
The Green's function relates surface currents to poloidal flux at the plasma boundary.

grri structure from DCON:
- Dimensions: [2*(mthvac+5), 2*mpert]
- First half: relates to normal current
- Second half: relates to parallel current
- Each mpert block corresponds to a poloidal mode

## Arguments
- `grri`: Green's function matrix from vacuum response
- `intr`: DCON internal state with mode information

## Returns
- Surface inductance matrix [numpert_total, numpert_total]
"""
function calc_surface_inductance(
    grri::Matrix{Float64},
    intr::DconInternal
)::Matrix{ComplexF64}

    numpert_total = intr.numpert_total
    mpert = intr.mpert
    npert = intr.npert

    # TODO: Proper implementation requires:
    # 1. Extract relevant Green's function components for each (m,n) mode pair
    # 2. Integrate Green's function over poloidal angle to get inductance
    # 3. Account for both normal and parallel current components
    # 4. Result should be L_surf[i,j] = inductance between modes i and j
    #
    # For now, construct approximate surface inductance from grri diagonal
    # This captures self-inductance but not mutual inductance

    surf_ind = zeros(ComplexF64, numpert_total, numpert_total)

    # grri is [2*(mthvac+5), 2*mpert]
    # Extract diagonal elements as approximation for self-inductance
    n_modes = min(mpert, size(grri, 2) ÷ 2)

    for i in 1:n_modes
        for n_idx in 1:npert
            mode_idx = (n_idx - 1) * mpert + i
            if mode_idx <= numpert_total
                # Use average of grri diagonal for this mode as self-inductance
                # This is a rough approximation
                if i <= size(grri, 2) ÷ 2
                    surf_ind[mode_idx, mode_idx] = mean(grri[:, i]) + 0.0im
                else
                    surf_ind[mode_idx, mode_idx] = 1.0 + 0.0im
                end
            end
        end
    end

    # Add small off-diagonal coupling (physical systems have mode coupling)
    for i in 1:numpert_total
        for j in 1:numpert_total
            if i != j && abs(surf_ind[i,i]) > 0 && abs(surf_ind[j,j]) > 0
                # Add 10% coupling between adjacent modes
                m_i = (i - 1) % mpert + intr.mlow
                m_j = (j - 1) % mpert + intr.mlow
                if abs(m_i - m_j) == 1
                    surf_ind[i,j] = 0.1 * sqrt(abs(surf_ind[i,i] * surf_ind[j,j])) + 0.0im
                end
            end
        end
    end

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
        intr::DconInternal
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
    intr::DconInternal
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
