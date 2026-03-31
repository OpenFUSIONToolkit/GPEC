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
    flux_matrix::Matrix{ComplexF64},
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
        flux_matrix,
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
        flux_matrix::Matrix{ComplexF64},
        ForceFreeStates_results::OdeState,
        ffs_intr::ForceFreeStatesInternal
    ) -> xi_psi_modes

Sum eigenmode contributions weighted by response coefficients.

`response_vector` (= P * Phi_x) is in mode (m,n) basis. To reconstruct physical
fields from eigenmode solutions in u_store, first project to eigenmode amplitudes:
    alpha = flux_matrix \\ response_vector

Then sum eigenmode contributions at each radial point:
    xi_psi[ipsi, :] = u_store[:, :, 1, ipsi] * alpha

Matches Fortran's idcon_build + gpeq_sol workflow (gpresp.f / gpeq.f).

# Arguments

- `response_vector`: Mode-basis response Phi_tot [numpert_total]
- `flux_matrix`: Flux matrix [mode × eigenmode] from build_flux_matrix
- `ForceFreeStates_results`: OdeState with u_store (eigenmode-transformed)
- `ffs_intr`: Mode information (mpert, etc.)

# Returns

- `xi_psi_modes`: Covariant radial displacement ξ_ψ(ψ, m) [npsi, mpert]
"""
function sum_eigenmode_contributions(
    response_vector::Vector{ComplexF64},
    flux_matrix::Matrix{ComplexF64},
    ForceFreeStates_results::OdeState,
    ffs_intr::ForceFreeStatesInternal
)
    mpert = ffs_intr.mpert
    npsi  = size(ForceFreeStates_results.u_store, 4)

    # Convert mode-basis response (Phi_tot) to eigenmode amplitudes alpha
    # flux_matrix[mode, eigenmode], so: flux_matrix * alpha = response_vector
    alpha = flux_matrix \ response_vector   # [mpert]

    # Sum: xi_psi[ipsi, :] = u_store[:, :, 1, ipsi] * alpha
    # u_store[mode_i, eigenmode_j, 1, ipsi] * alpha[j] summed over j
    xi_psi_modes = zeros(ComplexF64, npsi, mpert)
    for ipsi in 1:npsi
        mul!(view(xi_psi_modes, ipsi, :),
             ForceFreeStates_results.u_store[:, :, 1, ipsi],
             alpha)
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

"""
    compute_b_n_xi_n_modes(
        xi_psi_modes::Matrix{ComplexF64},
        b_psi_modes::Matrix{ComplexF64},
        ForceFreeStates_results::OdeState,
        equil::Equilibrium.PlasmaEquilibrium,
        ffs_intr::ForceFreeStatesInternal,
        mthvac::Int
    ) -> (b_n_modes, xi_n_modes)

Compute physical normal field b_n and displacement xi_n in mode space.

Mimics Fortran gpout_xbnormal (gpout.f lines 3353–3358):
1. IDFT: reconstruct theta-space function from mode amplitudes (b^ψ or ξ_ψ)
2. Divide by |∇ψ|(θ) at each theta point to get the physical normal component
3. Forward DFT back to mode space

Note: Julia's b_psi_modes = b^ψ and xi_psi_modes = ξ_ψ, with no Jacobian factor.
Fortran's xwp_mn includes a Jacobian convolution (gpeq_contra), making xwp_fun(θ) = J(θ)·ξ_ψ(θ),
so Fortran divides by J·|∇ψ|. Here we skip the Jacobian convolution and divide by |∇ψ| only,
which gives the same result: ξ_n = ξ_ψ/|∇ψ|, b_n = b^ψ/|∇ψ| [Park Phys. Plasmas 2007 052110].

`mthvac` sets the DFT theta resolution; Fortran uses mthsurf = max(mthvac, mtheta).  The
bicubic splines support arbitrary evaluation points, so any mthvac ≥ length(equil.rzphi_ys)-1
gives correct results; higher values reduce aliasing from the 1/|∇ψ| division.

# Returns

Tuple (b_n_modes, xi_n_modes), each [npsi, mpert] ComplexF64.
"""
function compute_b_n_xi_n_modes(
    xi_psi_modes::Matrix{ComplexF64},
    b_psi_modes::Matrix{ComplexF64},
    ForceFreeStates_results::OdeState,
    equil::Equilibrium.PlasmaEquilibrium,
    ffs_intr::ForceFreeStatesInternal,
    mthvac::Int
)
    npsi, mpert = size(b_psi_modes)
    mlow  = ffs_intr.mlow
    # Match Fortran: mthsurf = max(mthvac, mtheta); bicubic splines support any resolution
    mthsurf = max(mthvac, length(equil.rzphi_ys) - 1)
    ro = equil.ro
    twopi = 2π

    b_n_modes  = zeros(ComplexF64, npsi, mpert)
    xi_n_modes = zeros(ComplexF64, npsi, mpert)

    # Pre-compute mode indices and DFT phase table
    m_vals = [mlow + ipert - 1 for ipert in 1:mpert]
    thetas = [(k - 1) / mthsurf for k in 1:mthsurf]   # [0, 1/mthsurf, ..., (mthsurf-1)/mthsurf]
    # exp(+2πi*m*θ) for IDFT; exp(-2πi*m*θ) for forward DFT
    phase_fwd  = [exp( twopi * im * m_vals[ipert] * thetas[k]) for k in 1:mthsurf, ipert in 1:mpert]
    phase_back = [exp(-twopi * im * m_vals[ipert] * thetas[k]) for k in 1:mthsurf, ipert in 1:mpert]

    for ipsi in 1:npsi
        psi = ForceFreeStates_results.psi_store[ipsi]
        hint2d_psi = (Ref(1), Ref(1))

        # IDFT: mode space → theta space (Jbgradpsi and J×ξ_ψ)
        bwp_fun  = phase_fwd * b_psi_modes[ipsi, :]   # [mthsurf]
        xwp_fun  = phase_fwd * xi_psi_modes[ipsi, :]  # [mthsurf]

        # J(θ)×delpsi(θ) at this radial surface
        jd = zeros(Float64, mthsurf)
        for k in 1:mthsurf
            theta = thetas[k]
            r2    = equil.rzphi_rsquared((psi, theta); hint=hint2d_psi)
            deta  = equil.rzphi_offset((psi, theta); hint=hint2d_psi)
            jac   = equil.rzphi_jac((psi, theta); hint=hint2d_psi)
            r2_y  = equil.rzphi_rsquared((psi, theta); deriv=Val((0, 1)), hint=hint2d_psi)
            deta_y = equil.rzphi_offset((psi, theta); deriv=Val((0, 1)), hint=hint2d_psi)
            rfac  = sqrt(abs(r2))
            eta   = twopi * (theta + deta)
            r     = ro + rfac * cos(eta)
            w11   = (1.0 + deta_y) * twopi^2 * rfac * r / jac
            w12   = -r2_y * π * r / (rfac * jac)
            delpsi = sqrt(w11^2 + w12^2)
            jd[k]  = delpsi
        end

        # Divide by J×delpsi → physical normal components in theta space
        bno_fun  = bwp_fun ./ jd
        xno_fun  = xwp_fun ./ jd

        # Forward DFT: theta space → mode space (1/mthsurf normalization matches Fortran iscdftf)
        b_n_modes[ipsi, :]  = (phase_back' * bno_fun) ./ mthsurf
        xi_n_modes[ipsi, :] = (phase_back' * xno_fun) ./ mthsurf
    end

    return b_n_modes, xi_n_modes
end
