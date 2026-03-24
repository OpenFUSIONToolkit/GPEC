"""
Response matrix construction for perturbed equilibrium calculations.

Based on gpresp.f from GPEC, implementing resp_index=0 (energy-based inductance).
Uses ForceFreeStates eigenmode solutions and vacuum response data.

Reference: [Park Phys. Plasmas 2009 056115]
"""

# Use FourierTransform utility instead of FFTW for theta ↔ mode transforms
using ..Utilities.FourierTransforms

"""
    extract_boundary_displacements(
        equil::Equilibrium.PlasmaEquilibrium,
        ForceFreeStates_results::OdeState,
        intr::ForceFreeStatesInternal
    )::NamedTuple

Extract eigenmode displacements and equilibrium quantities at the plasma boundary.

This function extracts the data needed to compute the normal magnetic field at the
plasma surface from ForceFreeStates eigenmode solutions.

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
- `ForceFreeStates_results`: ODE integration results containing u_store with eigenmodes
- `intr`: ForceFreeStates internal state with boundary location (psilim)

## Returns
Named tuple with:
- `ξ_psi_boundary`: Boundary displacement [numpert_total, numpert_total]
- `dPsi_drho`: Flux surface spacing at boundary (scalar)
- `q_boundary`: Safety factor at boundary (scalar)
- `psi_boundary`: Normalized flux at boundary (scalar)
"""
function extract_boundary_displacements(
    equil::Equilibrium.PlasmaEquilibrium,
    ForceFreeStates_results::OdeState,
    intr::ForceFreeStatesInternal
)
    # Extract boundary displacement (normal component)
    # u_store dimensions: [numpert_total, numpert_total, 2, numsteps]
    # Index 1 in 3rd dimension is ξ_ψ (radial displacement)
    # Last index in 4th dimension is the boundary
    ξ_psi_boundary = ForceFreeStates_results.u_store[:, :, 1, ForceFreeStates_results.step]

    # Get boundary location in normalized flux coordinates
    psi_boundary = ForceFreeStates_results.psi_store[ForceFreeStates_results.step]

    # Evaluate equilibrium quantities at boundary
    # Safety factor at boundary
    q_boundary = ForceFreeStates_results.q_store[ForceFreeStates_results.step]

    # Flux surface spacing dΨ/dρ
    # In ForceFreeStates, ρ = √ψ where ψ is normalized poloidal flux
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
at the plasma surface. From the ideal MHD constraint [Park Phys. Plasmas 2009 056115 eq. 4]:

    B_n = i * (dΨ/dρ) * (m - n*q) * ξ_ψ

where ξ_ψ is the radial displacement eigenfunction.

## Physical Interpretation [Park Phys. Plasmas 2007 052110 Section II]:
- ξ_ψ[i,j]: Displacement of mode i due to eigenmode j
- singfac[i] = m[i] - n*q: Singular factor measuring distance from rational surface
- dΨ/dρ: Converts displacement to flux perturbation (poloidal flux gradient)
- Factor of i: Phase relationship for oscillating fields in complex representation

## Arguments
- `boundary_data`: Output from extract_boundary_displacements()
  - ξ_psi_boundary: Boundary displacement [numpert_total, numpert_total]
  - dPsi_drho: Flux surface spacing at boundary (scalar)
  - q_boundary: Safety factor at boundary (scalar)
  - psi_boundary: Normalized flux at boundary (scalar)
- `intr`: ForceFreeStates internal state with mode arrays (mlow, mhigh, nlow, etc.)

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

    # Compute singular factor for each Fourier mode [Park Phys. Plasmas 2009 056115 eq. 4]
    # singfac[i] = m[i] - n*q_boundary measures distance from rational surface
    # Mode indexing: modes are ordered as (m, n) pairs
    # Linear index i corresponds to: m = (i-1) % mpert + mlow, n = (i-1) ÷ mpert + nlow
    singfac = zeros(Float64, numpert_total)
    for i in 1:numpert_total
        m_mode = (i - 1) % intr.mpert + intr.mlow
        n_mode = (i - 1) ÷ intr.mpert + intr.nlow
        singfac[i] = m_mode - n_mode * q_boundary
    end

    # Compute normal magnetic field [Park Phys. Plasmas 2009 056115 eq. 4]
    # bwp_mn[i,j] = i * (dΨ/dρ) * singfac[i] * ξ_ψ[i,j]
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
        ForceFreeStates_results::OdeState,
        vac_data::VacuumData,
        intr::ForceFreeStatesInternal
    )::Matrix{ComplexF64}

Build vacuum poloidal flux matrix from ForceFreeStates eigenmode solutions.

This extracts the vacuum flux response for each eigenmode at the plasma boundary.
In GPEC, this comes from `bwp_mn` (boundary normal field) computed from eigenmode
displacements.

The flux matrix relates eigenmode displacements to vacuum poloidal flux:
1. Extract eigenmode displacement at plasma boundary from u_store
2. Compute normal magnetic field: B_ψ = i×(dΨ/dρ)×(m - n×q)×ξ_ψ
3. Result is flux[mode_i, eigenmode_j] = bwp_mn[i,j]

## Arguments
- `equil`: Equilibrium solution containing flux surfaces and q-profile
- `ForceFreeStates_results`: ForceFreeStates ODE integration results containing eigenmodes
- `vac_data`: Vacuum response data from free boundary calculation
- `intr`: ForceFreeStates internal state with mode information

## Returns
- `flxmats[numpert_total, numpert_total]`: Complex flux matrix where flxmats[i,j] is the
  vacuum flux of mode i in response to eigenmode j
"""
function build_flux_matrix(
    equil::Equilibrium.PlasmaEquilibrium,
    ForceFreeStates_results::OdeState,
    vac_data::VacuumData,
    intr::ForceFreeStatesInternal
)::Matrix{ComplexF64}

    # Step 1: Extract boundary displacements and equilibrium quantities
    boundary_data = extract_boundary_displacements(equil, ForceFreeStates_results, intr)

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
    pack_complex_to_realimag(modes::AbstractVector{ComplexF64})::AbstractVector{Float64}

Pack complex mode coefficients into real/imaginary pairs for Green's function application.

Converts [a+bi, c+di, ...] to [a, b, c, d, ...]

## Arguments
- `modes`: Complex Fourier mode coefficients [mpert]

## Returns
- Packed real/imaginary array [2*mpert]
"""
function pack_complex_to_realimag(modes::AbstractVector{ComplexF64})::AbstractVector{Float64}
    mpert = length(modes)
    packed = zeros(Float64, 2 * mpert)
    return pack_complex_to_realimag!(packed, modes)
end

function pack_complex_to_realimag!(packed::AbstractVector{Float64}, modes::AbstractVector{ComplexF64})::AbstractVector{Float64}
    mpert = length(modes)
    for i in 1:mpert
        packed[2*i - 1] = real(modes[i])
        packed[2*i] = imag(modes[i])
    end
    return packed
end

"""
    unpack_realimag_to_complex(packed::AbstractVector{Float64})::Vector{ComplexF64}

Unpack real/imaginary pairs back to complex coefficients.

Converts [a, b, c, d, ...] to [a+bi, c+di, ...]

## Arguments
- `packed`: Real/imaginary pairs [2*mpert]

## Returns
- Complex mode coefficients [mpert]
"""
function unpack_realimag_to_complex(packed::AbstractVector{Float64})::Vector{ComplexF64}
    mpert = length(packed) ÷ 2
    modes = zeros(ComplexF64, mpert)
    for i in 1:mpert
        modes[i] = packed[2*i - 1] + 1im * packed[2*i]
    end
    return modes
end

"""
    apply_green_function(
        green::Matrix{Float64},
        mode_coeffs::Vector{ComplexF64}
    )::Vector{Float64}

Apply Green's function to Fourier mode coefficients to get potential in theta space.

The Green's function matrix maps Fourier coefficients to theta-space values:
    χ(θ) = G * b_fourier

## Green's Function Structure

From ForceFreeStates vacuum calculation, `green` is a real matrix [2*mtheta, 2*mpert] where:
- Rows 1:mtheta correspond to plasma surface theta points
- Rows mtheta+1:2*mtheta correspond to wall surface theta points (if wall present)
- Columns are packed as [Re(mode_1), Im(mode_1), Re(mode_2), Im(mode_2), ...]

## Arguments
- `green`: Green's function matrix [2*mtheta, 2*mpert]
- `mode_coeffs`: Complex Fourier mode coefficients [mpert]

## Returns
- Potential values in theta space [mtheta] at plasma surface (real-valued)
"""
function apply_green_function(
    green::Matrix{Float64},
    mode_coeffs::Vector{ComplexF64}
)::Vector{Float64}
    mtheta = size(green, 1) ÷ 2
    chi_theta = Vector{Float64}(undef, mtheta)
    return apply_green_function!(chi_theta, green, mode_coeffs)
end

@with_pool pool function apply_green_function!(
    chi_theta::AbstractVector{Float64},
    green::Matrix{Float64},
    mode_coeffs::AbstractVector{ComplexF64}
)
    # Pack complex coefficients to real/imag format for Green's function
    # Format: [Re(mode_1), Im(mode_1), Re(mode_2), Im(mode_2), ...]
    mpert = length(mode_coeffs)
    packed_coeffs = zeros!(pool, Float64, 2 * mpert) 
    pack_complex_to_realimag!(packed_coeffs, mode_coeffs)

    # Apply Green's function: chi_theta = green * b_fourier
    # Extract only plasma surface rows (first mtheta rows)
    # Result is real-valued potential at theta points
    mtheta = length(chi_theta)
    mul!(chi_theta, @view(green[1:mtheta, :]), packed_coeffs)
    return chi_theta
end


"""
    calc_surface_inductance(
        grri::Matrix{Float64},
        grre::Matrix{Float64},
        flux_matrix::Matrix{ComplexF64},
        intr::ForceFreeStatesInternal
)::Matrix{ComplexF64}

Calculate surface/vacuum inductance matrix from Green's functions using GPEC algorithm.

Uses BOTH Green's function matrices computed during ForceFreeStates vacuum calculation:
- grri: Interior potential (kernelsign=-1)
- grre: Exterior potential (kernelsign=+1)

## GPEC Algorithm:
1. For each eigenmode, apply Green's functions to flux matrix (bwp_mn)
2. Compute interior potential: chi_mn = grri * bwp_mn
3. Compute exterior potential: che_mn = grre * bwp_mn
4. Calculate surface current: kax_mn = (chi_mn - che_mn) / μ₀
5. Transform back to mode space using Fourier transform
6. Solve: surf_indmats = hermitianize(flxmats * inv(kaxmats))

Green's function structure from ForceFreeStates:
- Dimensions: [2*mtheta, 2*mpert]
- Complex numbers stored as adjacent real/imaginary pairs
- Maps Fourier coefficients to theta-space potential values

## Arguments
- `grri`: Interior Green's function matrix (kernelsign=-1)
- `grre`: Exterior Green's function matrix (kernelsign=+1)
- `flux_matrix`: Normal magnetic field matrix (bwp_mn) [numpert_total, numpert_total]
- `intr`: ForceFreeStates internal state with mode information

## Returns
- Surface inductance matrix [numpert_total, numpert_total]
"""
function calc_surface_inductance(
    grri::Matrix{Float64},
    grre::Matrix{Float64},
    flux_matrix::Matrix{ComplexF64},
    intr::ForceFreeStatesInternal
)::Matrix{ComplexF64}

    numpert_total = intr.numpert_total
    mpert = intr.mpert
    npert = intr.npert
    mtheta = size(grri, 1) ÷ 2

    # Physical constant
    μ₀ = 4π * 1e-7

    # Create Fourier transform object for theta ↔ mode conversions
    # This pre-computes basis functions cos(m*θ), sin(m*θ) for efficient reuse
    # No phase shift needed since we're working in magnetic coordinates
    ft = FourierTransform(mtheta, mpert, intr.mlow)

    # Initialize matrices for surface current in mode space
    kax_matrix = zeros(ComplexF64, numpert_total, numpert_total)

    # Initialize surface inductance matrix (declare outside try-catch for scoping)
    surf_ind = zeros(ComplexF64, numpert_total, numpert_total)

    # For each eigenmode j, compute surface current in all modes
    for j in 1:numpert_total
        # Extract normal field for eigenmode j (only poloidal modes for this toroidal n)
        # Assuming single toroidal mode for now (npert == 1)
        if npert > 1
            @warn "calc_surface_inductance: Multiple toroidal modes not yet supported, using n=1 only" maxlog=1
        end

        # Extract poloidal mode coefficients for this eigenmode
        # For n=1: indices 1:mpert correspond to the poloidal modes
        mode_start = 1
        mode_end = min(mpert, numpert_total)
        bwp_modes = flux_matrix[mode_start:mode_end, j]

        # Apply interior Green's function to get χ(θ) at plasma surface
        chi_theta = apply_green_function(grri, bwp_modes)

        # Apply exterior Green's function to get χ_e(θ) at plasma surface
        che_theta = apply_green_function(grre, bwp_modes)

        # Compute surface current in theta space: kax(θ) = (χ - χ_e) / μ₀
        # This is the jump in potential across the plasma surface
        kax_theta = (chi_theta .- che_theta) ./ μ₀

        # Transform back to Fourier mode space using pre-computed FourierTransform
        # ft(data) performs forward transform: theta → modes
        kax_modes = ft(kax_theta)

        # Store in kax_matrix for this eigenmode
        kax_matrix[mode_start:mode_end, j] = kax_modes
    end

    # Compute surface inductance: L_surf = flxmats * inv(kaxmats)
    # Use pseudo-inverse for numerical stability
    try
        # Regularization: add small diagonal term to improve conditioning
        regularization = 1e-10 * maximum(abs.(kax_matrix))
        kax_reg = kax_matrix + regularization * I

        surf_ind = flux_matrix * inv(kax_reg)

        # Hermitianize to ensure physical inductance matrix
        surf_ind = 0.5 * (surf_ind + surf_ind')
    catch e
        @warn "Surface inductance inversion failed: $e. Using fallback method."
        # Fallback: use correlation method
        surf_ind = zeros(ComplexF64, numpert_total, numpert_total)
        for i in 1:mpert
            for j in 1:mpert
                correlation = 0.0 + 0.0im
                for k in 1:mtheta
                    # Simple correlation of Green's function differences
                    chi_i = grri[k, 2*i-1] + 1im * grri[k, 2*i]
                    che_i = grre[k, 2*i-1] + 1im * grre[k, 2*i]
                    chi_j = grri[k, 2*j-1] + 1im * grri[k, 2*j]
                    che_j = grre[k, 2*j-1] + 1im * grre[k, 2*j]
                    jump_i = chi_i - che_i
                    jump_j = chi_j - che_j
                    correlation += jump_i * conj(jump_j)
                end
                surf_ind[i, j] = μ₀ * correlation / mtheta
            end
        end
        surf_ind = 0.5 * (surf_ind + surf_ind')
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
        intr::ForceFreeStatesInternal
    )::Vector{ComplexF64}

Map external forcing modes to eigenmode basis.

Matches forcing mode numbers (n,m) to the eigenmode basis used in ForceFreeStates
and creates a forcing vector in that basis.

**Unit convention**: `ForcingMode.amplitude` must be in `sfl_flux_Wb` units
(T·m², Julia 2π-angle convention). Files loaded in `normal_field_T` convention are
automatically converted by `convert_forcing_normalization!` before this function is called.

## Arguments
- `forcing_modes`: External forcing modes, amplitudes in `sfl_flux_Wb` convention
- `intr`: ForceFreeStates internal state with mode arrays

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
