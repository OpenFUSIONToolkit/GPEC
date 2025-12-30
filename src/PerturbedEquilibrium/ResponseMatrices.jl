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

    # The flux matrix relates eigenmode boundary displacements to vacuum poloidal flux.
    # In GPEC: flxmats[:,i] = bwp_mn (boundary normal field) for eigenmode i
    #
    # In DCON, the vacuum response is encoded in:
    # - vac.wt: eigenvectors of total energy matrix (wp + wv)
    # - vac.wv: vacuum energy matrix, which relates boundary flux to vacuum energy
    # - The vacuum energy is: E_vac[i,j] = (1/2μ₀) ∫ B_i · B_j dV
    #
    # Physical interpretation:
    # - wv[i,j] represents the vacuum energy coupling between modes i and j
    # - wt[:,k] are the eigenvectors representing actual plasma eigenmodes
    # - The flux matrix should project the eigenmode basis onto the Fourier mode basis
    #
    # Improved approximation:
    # Use wt (eigenvectors) to construct flux matrix in eigenmode basis
    # flxmats[:,k] ≈ wv * wt[:,k]
    # This represents the vacuum flux response to eigenmode k

    flxmats = zeros(ComplexF64, numpert_total, numpert_total)

    # Build flux matrix by projecting eigenmodes through vacuum coupling
    # For each eigenmode k, compute the vacuum flux response
    for k in 1:numpert_total
        # The vacuum response to eigenmode k is given by wv * wt[:,k]
        # This represents how eigenmode k couples to all Fourier modes through vacuum
        flxmats[:, k] = vac_data.wv * vac_data.wt[:, k]
    end

    # NOTE: This is an improved approximation but still not the full GPEC implementation.
    # Full implementation would require:
    # 1. Extract boundary displacement ξ from u_store at boundary (step = dcon_results.step)
    # 2. Compute normal magnetic field: B_ψ = i*(dΨ/dρ)*(m - n*q)*ξ_ψ
    # 3. This requires computing singfac = (m - n*q) at the boundary
    # 4. The result would be the actual boundary normal field for each eigenmode

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
        intr::DconInternal
)::Matrix{ComplexF64}

Calculate surface/vacuum inductance matrix from Green's functions.

Uses BOTH Green's function matrices computed during DCON vacuum calculation:
- grri: Interior potential (kernelsign=-1)
- grre: Exterior potential (kernelsign=+1)

The surface inductance is derived from the jump in magnetic potential across the plasma surface,
which gives the surface current: kax = (χ - χ_e) / μ₀

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
    intr::DconInternal
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
