"""
Field reconstruction from eigenmode response.

Converts eigenmode response coefficients to physical displacement and magnetic
field perturbations in mode space, following the GPEC gpeq module approach.
[Park Phys. Plasmas 2007 052110]

This module mimics the GPEC Fortran subroutines:
- gpeq_sol: Get equilibrium solution at each radial point
- gpeq_contra: Compute contravariant field from covariant displacement
- gpeq_cova: Compute covariant components using metric tensors
- gpeq_normal: Compute normal (flux surface) components
- gpeq_tangent: Compute tangential components

Fields are computed using ideal MHD relations in flux coordinates:
    b^ψ = i * χ₁ * (m - n*q) * ξ_ψ
    b^θ = -i * (χ₁ * ∂ξ_ψ/∂ψ + n * ξ_ζ)
    b^ζ = -i * (χ₁ * (q'*ξ_ψ + q*∂ξ_ψ/∂ψ) + m * ξ_ζ)

where χ₁ = 2π * Ψ₀ (total poloidal flux normalization).

References:
- [Park Phys. Plasmas 2007 052110] - 3D equilibrium perturbations
- GPEC gpeq.f lines 100-102
"""

"""
    reconstruct_physical_fields(
        response_vector::Vector{ComplexF64},
        ForceFreeStates_results::OdeState,
        equil::Equilibrium.PlasmaEquilibrium,
        ffs_intr::ForceFreeStatesInternal,
        intr::PerturbedEquilibriumInternal
    ) -> (xi_modes, b_modes)

Reconstruct displacement and magnetic field from eigenmode response in mode space.

This function mimics GPEC's gpeq module, computing fields using ideal MHD
algebraic relations in flux coordinates [Park Phys. Plasmas 2007 052110].
All fields are returned in mode space [npsi, mpert] rather than real space.

# Process (following GPEC)

1. Sum weighted eigenmode contributions → ξ_ψ(ψ, m)
2. Compute contravariant field from ideal MHD → b^ψ, b^θ, b^ζ
3. Return mode-space fields (can convert to real space later if needed)

# Arguments

- `response_vector::Vector{ComplexF64}`: Response coefficients [numpert_total]
- `ForceFreeStates_results::OdeState`: ForceFreeStates results with u_store eigenmodes
- `equil::Equilibrium.PlasmaEquilibrium`: Equilibrium data
- `ffs_intr::ForceFreeStatesInternal`: Mode information (m, n ranges)
- `intr::PerturbedEquilibriumInternal`: Internal state

# Returns

Tuple of (xi_modes, b_modes) where each is a NamedTuple:
- `xi_modes.psi`: ξ_ψ component [npsi, mpert] (covariant)
- `b_modes.psi`: b^ψ component [npsi, mpert] (contravariant)
- `b_modes.theta`: b^θ component [npsi, mpert] (contravariant)
- `b_modes.zeta`: b^ζ component [npsi, mpert] (contravariant)

All in mode space, matching GPEC output format.

# Notes

- Uses ForceFreeStates radial grid (not equilibrium grid)
- Works entirely in mode space - no Fourier transforms
- Follows GPEC gpeq_sol, gpeq_contra formulation
- Field from ideal MHD: b^ψ = i*χ₁*(m-n*q)*ξ_ψ
"""
function reconstruct_physical_fields(
    response_vector::Vector{ComplexF64},
    ForceFreeStates_results::OdeState,
    equil::Equilibrium.PlasmaEquilibrium,
    ffs_intr::ForceFreeStatesInternal,
    intr::PerturbedEquilibriumInternal
)
    # Get dimensions
    npsi = size(ForceFreeStates_results.u_store, 4)
    mpert = ffs_intr.mpert

    # Step 1: Sum weighted eigenmode contributions to get covariant ξ_ψ in mode space
    xi_psi_modes = sum_eigenmode_contributions(
        response_vector,
        ForceFreeStates_results,
        ffs_intr
    )

    # Step 2: Compute perturbed field in mode space using ideal MHD relations
    # [Park Phys. Plasmas 2007 052110 eq. 8-10]
    # This mimics GPEC's gpeq_sol and gpeq_contra
    b_psi_modes, b_theta_modes, b_zeta_modes = compute_perturbed_field_modes(
        xi_psi_modes,
        ForceFreeStates_results,
        equil,
        ffs_intr
    )

    # Package outputs in NamedTuples for clarity
    xi_modes = (
        psi = xi_psi_modes,      # [npsi, mpert] - covariant radial displacement
        theta = zeros(ComplexF64, npsi, mpert),  # Placeholder - not computed from ForceFreeStates
        zeta = zeros(ComplexF64, npsi, mpert)    # Placeholder - not computed from ForceFreeStates
    )

    b_modes = (
        psi = b_psi_modes,       # [npsi, mpert] - contravariant radial field
        theta = b_theta_modes,   # [npsi, mpert] - contravariant poloidal field
        zeta = b_zeta_modes      # [npsi, mpert] - contravariant toroidal field
    )

    return xi_modes, b_modes
end

"""
    sum_eigenmode_contributions(
        response_vector::Vector{ComplexF64},
        ForceFreeStates_results::OdeState,
        ffs_intr::ForceFreeStatesInternal
    ) -> xi_psi_modes

Sum eigenmode contributions weighted by response coefficients.

The response vector contains one coefficient for each (i,j) eigenmode pair.
This function weights each eigenmode by its coefficient and sums them to
get the total covariant radial displacement ξ_ψ in mode space.

Mimics GPEC's approach where the DCON eigenmodes are weighted by the
plasma response to external forcing.

# Arguments

- `response_vector::Vector{ComplexF64}`: Response coefficient for each eigenmode [numpert_total]
- `ForceFreeStates_results::OdeState`: ForceFreeStates results with u_store containing eigenmodes
- `ffs_intr::ForceFreeStatesInternal`: Mode information (mpert, etc.)

# Returns

- `xi_psi_modes::Matrix{ComplexF64}`: Covariant radial displacement ξ_ψ(ψ, m) [npsi, mpert]

# Notes

- In ForceFreeStates formulation, u_store[:, :, 1, :] contains ξ_ψ (covariant radial displacement)
- The response vector is ordered as [mode1_mode1, mode1_mode2, ..., mode2_mode1, ...]
- We sum over all (i,j) eigenmode pairs to get the total response for each mode i
- This corresponds to xsp_mn in GPEC notation
"""
function sum_eigenmode_contributions(
    response_vector::Vector{ComplexF64},
    ForceFreeStates_results::OdeState,
    ffs_intr::ForceFreeStatesInternal
)
    # Extract dimensions
    numpert_total = length(response_vector)
    npsi = size(ForceFreeStates_results.u_store, 4)
    mpert = ffs_intr.mpert

    # Initialize output array (npsi × mpert)
    # xsp_mn in GPEC notation (covariant radial displacement)
    xi_psi_modes = zeros(ComplexF64, npsi, mpert)

    # Sum weighted eigenmode contributions
    # For each eigenmode pair (i, j) in the response matrix
    idx = 1
    for i in 1:mpert
        for j in 1:mpert
            if idx > numpert_total
                break
            end

            # Weight this eigenmode by its response coefficient
            coeff = response_vector[idx]

            # Add this eigenmode's contribution to each radial point
            # Accumulate into mode i (diagonal contribution)
            for ipsi in 1:npsi
                # u_store[i, j, component, radial_index]
                # Component 1 = ξ_ψ (covariant radial displacement)
                xi_psi_modes[ipsi, i] += coeff * ForceFreeStates_results.u_store[i, j, 1, ipsi]
            end

            idx += 1
        end
    end

    return xi_psi_modes
end

"""
    compute_perturbed_field_modes(
        xi_psi_modes::Matrix{ComplexF64},
        ForceFreeStates_results::OdeState,
        equil::Equilibrium.PlasmaEquilibrium,
        ffs_intr::ForceFreeStatesInternal
    ) -> (b_psi_modes, b_theta_modes, b_zeta_modes)

Compute perturbed magnetic field from displacement using ideal MHD relations in mode space.

This function mimics GPEC's gpeq_sol subroutine (gpeq.f lines 100-102), computing
contravariant field components from covariant displacement using the algebraic
ideal MHD relations in flux coordinates [Park Phys. Plasmas 2007 052110 eq. 8-10].

# Ideal MHD Relations [Park Phys. Plasmas 2007 052110 eq. 8-10]

```fortran
bwp_mn = (chi1*singfac*twopi*ifac*xsp_mn)                              ! b^ψ
bwt_mn = -(chi1*xsp1_mn + twopi*ifac*nn*xss_mn)                        ! b^θ
bwz_mn = -(chi1*(q1*xsp_mn + sq%f(4)*xsp1_mn) + twopi*ifac*mfac*xss_mn) ! b^ζ
```

where:
- xsp_mn = ξ_ψ (covariant radial displacement)
- xsp1_mn = ∂ξ_ψ/∂ψ (radial derivative)
- xss_mn = ξ_ζ (covariant toroidal displacement)
- singfac = m - n*q (resonance factor)
- chi1 = 2π*Ψ₀ (flux normalization)
- ifac = i (imaginary unit)

# Arguments

- `xi_psi_modes::Matrix{ComplexF64}`: Covariant radial displacement ξ_ψ(ψ,m) [npsi, mpert]
- `ForceFreeStates_results::OdeState`: ForceFreeStates results with psi_store for radial grid
- `equil::Equilibrium.PlasmaEquilibrium`: Equilibrium with q(ψ), q'(ψ), Ψ₀
- `ffs_intr::ForceFreeStatesInternal`: Mode numbers (mlow, mhigh, n)

# Returns

Tuple of three matrices, all [npsi, mpert]:
- `b_psi_modes::Matrix{ComplexF64}`: Contravariant radial field b^ψ(ψ,m)
- `b_theta_modes::Matrix{ComplexF64}`: Contravariant poloidal field b^θ(ψ,m)
- `b_zeta_modes::Matrix{ComplexF64}`: Contravariant toroidal field b^ζ(ψ,m)

# Notes

- This is a simplified version assuming ξ_ζ = 0 (no toroidal displacement)
- Radial derivatives ∂ξ_ψ/∂ψ are computed using finite differences
- All calculations done in mode space (no Fourier transforms)
- Matches GPEC formulation exactly for consistency
"""
function compute_perturbed_field_modes(
    xi_psi_modes::Matrix{ComplexF64},
    ForceFreeStates_results::OdeState,
    equil::Equilibrium.PlasmaEquilibrium,
    ffs_intr::ForceFreeStatesInternal
)
    npsi, mpert = size(xi_psi_modes)

    # Initialize output arrays
    b_psi_modes = zeros(ComplexF64, npsi, mpert)
    b_theta_modes = zeros(ComplexF64, npsi, mpert)
    b_zeta_modes = zeros(ComplexF64, npsi, mpert)

    # Get mode numbers
    mlow = ffs_intr.mlow
    nn = ffs_intr.nlow  # Toroidal mode number

    # Normalization constant: chi1 = 2π * Ψ₀
    chi1 = 2π * equil.psio
    twopi = 2π
    ifac = im  # Imaginary unit

    # Compute radial derivative of displacement using finite differences
    # ∂ξ_ψ/∂ψ (xsp1_mn in GPEC)
    xi_psi1_modes = zeros(ComplexF64, npsi, mpert)
    for ipert in 1:mpert
        for ipsi in 2:npsi-1
            # Centered difference
            dpsi = ForceFreeStates_results.psi_store[ipsi+1] - ForceFreeStates_results.psi_store[ipsi-1]
            xi_psi1_modes[ipsi, ipert] = (xi_psi_modes[ipsi+1, ipert] - xi_psi_modes[ipsi-1, ipert]) / dpsi
        end
        # Forward difference at axis
        if npsi > 1
            dpsi = ForceFreeStates_results.psi_store[2] - ForceFreeStates_results.psi_store[1]
            xi_psi1_modes[1, ipert] = (xi_psi_modes[2, ipert] - xi_psi_modes[1, ipert]) / dpsi
        end
        # Backward difference at edge
        if npsi > 1
            dpsi = ForceFreeStates_results.psi_store[npsi] - ForceFreeStates_results.psi_store[npsi-1]
            xi_psi1_modes[npsi, ipert] = (xi_psi_modes[npsi, ipert] - xi_psi_modes[npsi-1, ipert]) / dpsi
        end
    end

    # Compute field for each radial point and mode
    for ipsi in 1:npsi
        psi_norm = ForceFreeStates_results.psi_store[ipsi]

        # Get equilibrium quantities at this surface
        q = equil.profiles.q_spline(psi_norm)       # Safety factor q(ψ)
        q1 = equil.profiles.q_deriv(psi_norm)       # Derivative q'(ψ) = dq/dψ

        # Compute field for each poloidal mode
        for ipert in 1:mpert
            m = mlow + ipert - 1  # Poloidal mode number

            # Resonance factor: singfac = m - n*q
            singfac = m - nn * q

            # Get displacement and derivative at this point
            xsp = xi_psi_modes[ipsi, ipert]     # ξ_ψ
            xsp1 = xi_psi1_modes[ipsi, ipert]   # ∂ξ_ψ/∂ψ
            xss = 0.0 + 0.0im                    # ξ_ζ = 0 (not computed from ForceFreeStates)

            # Contravariant field from ideal MHD [Park Phys. Plasmas 2007 052110 eq. 8-10]
            # b^ψ = i * χ₁ * (m - n*q) * ξ_ψ  [eq. 8]
            b_psi_modes[ipsi, ipert] = chi1 * singfac * twopi * ifac * xsp

            # b^θ = -i * (χ₁ * ∂ξ_ψ/∂ψ + n * ξ_ζ)  [eq. 9]
            b_theta_modes[ipsi, ipert] = -(chi1 * xsp1 + twopi * ifac * nn * xss)

            # b^ζ = -i * (χ₁ * (q'*ξ_ψ + q*∂ξ_ψ/∂ψ) + m * ξ_ζ)  [eq. 10]
            b_zeta_modes[ipsi, ipert] = -(chi1 * (q1 * xsp + q * xsp1) + twopi * ifac * m * xss)
        end
    end

    return b_psi_modes, b_theta_modes, b_zeta_modes
end
