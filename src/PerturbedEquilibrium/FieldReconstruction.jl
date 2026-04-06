"""
Field reconstruction from eigenmode response.

Converts eigenmode response coefficients to physical displacement and magnetic
field perturbations in mode space, following the GPEC gpeq module approach.

Displacement components from ODE integration (u_store/ud_store):
- ξ_ψ: radial displacement (u_store[:,:,1,:])
- dξ_ψ/dψ: radial derivative (ud_store[:,:,1,:])
- ξ_s: toroidal displacement (ud_store[:,:,2,:], Glasser 2016 eq. 18)

Contravariant perturbed field from ideal MHD (matches Fortran gpeq_sol):
    b^ψ =  χ₁·(m - n·q)·2πi·ξ_ψ
    b^θ = -(χ₁·dξ_ψ/dψ + 2πi·n·ξ_s)
    b^ζ = -(χ₁·(q'·ξ_ψ + q·dξ_ψ/dψ) + 2πi·m·ξ_s)

where χ₁ = 2π·Ψ₀ [Park Phys. Plasmas 14, 052110 (2007) eq. 8-10].
"""

"""
    reconstruct_physical_fields(
        response_vector::Vector{ComplexF64},
        flux_matrix::Matrix{ComplexF64},
        ForceFreeStates_results::OdeState,
        equil::Equilibrium.PlasmaEquilibrium,
        ffs_intr::ForceFreeStatesInternal,
        intr::PerturbedEquilibriumInternal
    ) -> (xi_modes, b_modes)

Reconstruct displacement and perturbed magnetic field from eigenmode response.

Extracts ξ_ψ, dξ_ψ/dψ, and ξ_s from u_store/ud_store eigenmodes, then computes
contravariant field components via ideal MHD relations (matches Fortran gpeq_sol).
All fields returned in mode space [npsi, mpert].

# Returns

Tuple of (xi_modes, b_modes) NamedTuples:
- `xi_modes.psi`: ξ_ψ [npsi, mpert]
- `xi_modes.theta/zeta`: not yet implemented (requires gpeq_contra Jacobian convolution)
- `b_modes.psi`: b^ψ [npsi, mpert]
- `b_modes.Jbgradpsi`: b^ψ / <J·|∇ψ|>_θ (matches Fortran gpout_xbnormal)
- `b_modes.theta`: b^θ [npsi, mpert]
- `b_modes.zeta`: b^ζ [npsi, mpert]
"""
function reconstruct_physical_fields(
    response_vector::Vector{ComplexF64},
    flux_matrix::Matrix{ComplexF64},
    ForceFreeStates_results::OdeState,
    equil::Equilibrium.PlasmaEquilibrium,
    ffs_intr::ForceFreeStatesInternal,
    intr::PerturbedEquilibriumInternal
)
    npsi = size(ForceFreeStates_results.u_store, 4)
    mpert = ffs_intr.mpert
    psi_grid = ForceFreeStates_results.psi_store[1:npsi]

    # Sum weighted eigenmode contributions to get ξ_ψ, dξ_ψ/dψ, and ξ_s in mode space
    xi_psi_modes, xi_psi1_modes, xi_s_modes = sum_eigenmode_contributions(
        response_vector,
        flux_matrix,
        ForceFreeStates_results,
        ffs_intr
    )

    # Compute perturbed field in mode space using ideal MHD relations
    # [Park Phys. Plasmas 14, 052110 (2007) eq. 8-10]
    b_psi_modes, b_theta_modes, b_zeta_modes = compute_perturbed_field_modes(
        xi_psi_modes, xi_psi1_modes, xi_s_modes,
        psi_grid, equil, ffs_intr
    )

    # Compute b^ψ / area for HDF5 output — matches Fortran gpout_xbnormal fast path
    # area = <J·|∇ψ|>_θ (Hamada: J is const over θ)
    ro = equil.ro
    mthsurf = length(equil.rzphi_ys) - 1
    thetas_for_avg = [(k - 1) / mthsurf for k in 1:mthsurf]
    Jb_psi_modes = similar(b_psi_modes)
    for ipsi in 1:npsi
        psi = psi_grid[ipsi]
        hint2d = (Ref(1), Ref(1))
        area = 0.0
        for k in 1:mthsurf
            theta = thetas_for_avg[k]
            r2     = equil.rzphi_rsquared((psi, theta); hint=hint2d)
            deta   = equil.rzphi_offset((psi, theta);   hint=hint2d)
            jac    = equil.rzphi_jac((psi, theta);      hint=hint2d)
            r2_y   = equil.rzphi_rsquared((psi, theta); deriv=Val((0,1)), hint=hint2d)
            deta_y = equil.rzphi_offset((psi, theta);   deriv=Val((0,1)), hint=hint2d)
            rfac   = sqrt(abs(r2))
            eta    = 2π * (theta + deta)
            r      = ro + rfac * cos(eta)
            w11    = (1.0 + deta_y) * 4π^2 * rfac * r / jac
            w12    = -r2_y * π * r / (rfac * jac)
            delpsi = sqrt(w11^2 + w12^2)
            area  += jac * delpsi
        end
        area /= mthsurf
        Jb_psi_modes[ipsi, :] = b_psi_modes[ipsi, :] ./ area
    end

    # TODO: xi_theta and xi_zeta require gpeq_contra Jacobian convolution (jmat/jmat1
    # mode coupling) — not yet implemented. Only xi_psi is currently output.
    xi_modes = (
        psi   = xi_psi_modes,
        theta = zeros(ComplexF64, npsi, mpert),
        zeta  = zeros(ComplexF64, npsi, mpert)
    )
    b_modes = (
        psi       = b_psi_modes,    # b^ψ (no Jacobian) — used for b_n normal projection
        Jbgradpsi = Jb_psi_modes,   # b^ψ / <J·|∇ψ|>_θ — matches Fortran gpout_xbnormal convention
        theta     = b_theta_modes,
        zeta      = b_zeta_modes
    )

    return xi_modes, b_modes
end

"""
    sum_eigenmode_contributions(
        response_vector::Vector{ComplexF64},
        flux_matrix::Matrix{ComplexF64},
        ForceFreeStates_results::OdeState,
        ffs_intr::ForceFreeStatesInternal
    ) -> (xi_psi_modes, xi_psi1_modes, xi_s_modes)

Sum eigenmode contributions weighted by response coefficients.

`response_vector` (= P * Phi_x) is in mode (m,n) basis. To reconstruct physical
fields from eigenmode solutions in u_store/ud_store, first project to eigenmode amplitudes:
    alpha = flux_matrix \\ response_vector

Then sum eigenmode contributions at each radial point (matches Fortran gpeq_sol):
    xi_psi[ipsi, :]  = u_store[:, :, 1, ipsi]  * alpha   # Ξ_ψ
    xi_psi1[ipsi, :] = ud_store[:, :, 1, ipsi] * alpha   # dΞ_ψ/dψ
    xi_s[ipsi, :]    = ud_store[:, :, 2, ipsi] * alpha   # Ξ_s (toroidal, Glasser 2016 eq. 18)

# Arguments

- `response_vector`: Mode-basis response Phi_tot [numpert_total]
- `flux_matrix`: Flux matrix [mode × eigenmode] from build_flux_matrix
- `ForceFreeStates_results`: OdeState with u_store and ud_store (eigenmode-transformed)
- `ffs_intr`: Mode information (mpert, etc.)

# Returns

- `xi_psi_modes`: Covariant radial displacement ξ_ψ(ψ, m) [npsi, mpert]
- `xi_psi1_modes`: Radial derivative dξ_ψ/dψ(ψ, m) [npsi, mpert]
- `xi_s_modes`: Toroidal displacement ξ_s(ψ, m) = -A⁻¹(B·dξ_ψ/dψ + C·ξ_ψ) [npsi, mpert]
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

    xi_psi_modes  = zeros(ComplexF64, npsi, mpert)
    xi_psi1_modes = zeros(ComplexF64, npsi, mpert)
    xi_s_modes    = zeros(ComplexF64, npsi, mpert)
    for ipsi in 1:npsi
        # u_store[:,:,1] = Ξ_ψ (radial displacement)
        mul!(view(xi_psi_modes, ipsi, :),
             ForceFreeStates_results.u_store[:, :, 1, ipsi],
             alpha)
        # ud_store[:,:,1] = dΞ_ψ/dψ (radial derivative from ODE)
        mul!(view(xi_psi1_modes, ipsi, :),
             ForceFreeStates_results.ud_store[:, :, 1, ipsi],
             alpha)
        # ud_store[:,:,2] = Ξ_s = -A⁻¹(B·Ξ'_ψ + C·Ξ_ψ) (toroidal displacement, Glasser 2016 eq. 18)
        mul!(view(xi_s_modes, ipsi, :),
             ForceFreeStates_results.ud_store[:, :, 2, ipsi],
             alpha)
    end

    return xi_psi_modes, xi_psi1_modes, xi_s_modes
end

"""
    compute_perturbed_field_modes(
        xi_psi_modes::Matrix{ComplexF64},
        xi_psi1_modes::Matrix{ComplexF64},
        xi_s_modes::Matrix{ComplexF64},
        equil::Equilibrium.PlasmaEquilibrium,
        ffs_intr::ForceFreeStatesInternal
    ) -> (b_psi_modes, b_theta_modes, b_zeta_modes)

Compute perturbed magnetic field from displacement using ideal MHD relations in mode space.

Matches Fortran `gpeq_sol` subroutine, computing contravariant field components from
covariant displacement [Park Phys. Plasmas 14, 052110 (2007) eq. 8-10]:

    b^ψ =  χ₁·(m - n·q)·2πi·ξ_ψ
    b^θ = -(χ₁·∂ξ_ψ/∂ψ + 2πi·n·ξ_s)
    b^ζ = -(χ₁·(q'·ξ_ψ + q·∂ξ_ψ/∂ψ) + 2πi·m·ξ_s)

where χ₁ = 2π·Ψ₀, and ξ_s is the toroidal displacement from Glasser 2016 eq. 18.

# Arguments

- `xi_psi_modes`: Covariant radial displacement ξ_ψ(ψ,m) [npsi, mpert]
- `xi_psi1_modes`: Radial derivative dξ_ψ/dψ(ψ,m) from ODE integration [npsi, mpert]
- `xi_s_modes`: Toroidal displacement ξ_s(ψ,m) from ODE integration [npsi, mpert]
- `psi_grid`: Radial grid points [npsi]
- `equil`: Equilibrium with q(ψ), q'(ψ), Ψ₀
- `ffs_intr`: Mode numbers (mlow, mhigh, n)

# Returns

Tuple of three matrices, all [npsi, mpert]:
- `b_psi_modes`: Contravariant radial field b^ψ(ψ,m)
- `b_theta_modes`: Contravariant poloidal field b^θ(ψ,m)
- `b_zeta_modes`: Contravariant toroidal field b^ζ(ψ,m)
"""
function compute_perturbed_field_modes(
    xi_psi_modes::Matrix{ComplexF64},
    xi_psi1_modes::Matrix{ComplexF64},
    xi_s_modes::Matrix{ComplexF64},
    psi_grid::Vector{Float64},
    equil::Equilibrium.PlasmaEquilibrium,
    ffs_intr::ForceFreeStatesInternal
)
    npsi, mpert = size(xi_psi_modes)

    b_psi_modes   = zeros(ComplexF64, npsi, mpert)
    b_theta_modes = zeros(ComplexF64, npsi, mpert)
    b_zeta_modes  = zeros(ComplexF64, npsi, mpert)

    mlow = ffs_intr.mlow
    nn = ffs_intr.nlow
    chi1 = 2π * equil.psio

    for ipsi in 1:npsi
        psi_norm = psi_grid[ipsi]
        q = equil.profiles.q_spline(psi_norm)
        q1 = equil.profiles.q_deriv(psi_norm)

        for ipert in 1:mpert
            m = mlow + ipert - 1
            singfac = m - nn * q

            xsp  = xi_psi_modes[ipsi, ipert]
            xsp1 = xi_psi1_modes[ipsi, ipert]
            xss  = xi_s_modes[ipsi, ipert]

            # Matches Fortran gpeq_sol [Park 2007 eq. 8-10]
            b_psi_modes[ipsi, ipert]  = chi1 * singfac * 2π * im * xsp
            b_theta_modes[ipsi, ipert] = -(chi1 * xsp1 + 2π * im * nn * xss)
            b_zeta_modes[ipsi, ipert]  = -(chi1 * (q1 * xsp + q * xsp1) + 2π * im * m * xss)
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
    ) -> (b_n_modes, xi_n_modes)

Compute physical normal field b_n and displacement xi_n in mode space.

Matches Fortran `gpout_xbnormal`:
1. IDFT: reconstruct theta-space function from mode amplitudes (b^ψ or ξ_ψ)
2. Divide by |∇ψ|(θ) at each theta point to get the physical normal component
3. Forward DFT back to mode space

Note: Julia's b_psi_modes = b^ψ and xi_psi_modes = ξ_ψ, with no Jacobian factor.
Fortran's xwp_mn includes a Jacobian convolution (gpeq_contra), making xwp_fun(θ) = J(θ)·ξ_ψ(θ),
so Fortran divides by J·|∇ψ|. Here we skip the Jacobian convolution and divide by |∇ψ| only,
which gives the same result: ξ_n = ξ_ψ/|∇ψ|, b_n = b^ψ/|∇ψ| [Park Phys. Plasmas 2007 052110].

DFT resolution: `mthsurf = mtheta = length(equil.rzphi_ys) - 1`. This is sufficient since
Nyquist (mtheta/2) >> 2·max|m|, so no aliasing from the 1/|∇ψ| division.

# Returns

Tuple (b_n_modes, xi_n_modes), each [npsi, mpert] ComplexF64.
"""
function compute_b_n_xi_n_modes(
    xi_psi_modes::Matrix{ComplexF64},
    b_psi_modes::Matrix{ComplexF64},
    ForceFreeStates_results::OdeState,
    equil::Equilibrium.PlasmaEquilibrium,
    ffs_intr::ForceFreeStatesInternal,
)
    npsi, mpert = size(b_psi_modes)
    mlow  = ffs_intr.mlow
    mthsurf = length(equil.rzphi_ys) - 1
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

        # IDFT: mode space → theta space
        bwp_fun  = phase_fwd * b_psi_modes[ipsi, :]   # [mthsurf]
        xwp_fun  = phase_fwd * xi_psi_modes[ipsi, :]  # [mthsurf]

        # delpsi(θ) = |∇ψ|(θ) at this radial surface; jd_b = J·|∇ψ| for b_n divisor.
        # xi_n: xwp_fun ≈ J·ξ_ψ (J-weighted via gpeq_contra metric), divide by delpsi; J cancels.
        # b_n:  bwp_fun ≈ b^ψ (not J-weighted), Fortran divides by J·delpsi → match with jd_b.
        delpsis = zeros(Float64, mthsurf)
        jacs  = zeros(Float64, mthsurf)
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
            delpsis[k] = delpsi
            jacs[k]  = jac
        end

        # Divide by |∇ψ| (xi_n) or J·|∇ψ| (b_n) → physical normal components in theta space
        bno_fun  = bwp_fun ./ (jacs .* delpsis)
        xno_fun  = xwp_fun ./ delpsis

        # Forward DFT: theta space → mode space (1/mthsurf normalization matches Fortran iscdftf).
        # Must use transpose (not adjoint) so the phase is exp(-2πi·m·θ), not exp(+2πi·m·θ).
        b_n_modes[ipsi, :]  = (transpose(phase_back) * bno_fun) ./ mthsurf
        xi_n_modes[ipsi, :] = (transpose(phase_back) * xno_fun) ./ mthsurf
    end

    return b_n_modes, xi_n_modes
end
