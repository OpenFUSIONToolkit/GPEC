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

Clebsch displacement components for PENTRC (matches Fortran gpout_xclebsch):
    ξ^ψ         = xsp_mn    (unregularized)
    ∂ξ^ψ/∂ψ    = xmp1_mn   (regularized: xsp1 * singfac²/(singfac² + reg_spot²))
    ξ^α         = xms_mn    (regularized: -A⁻¹(B·xmp1 + C·xsp), divided by χ₁ in output)

Contravariant displacement from Jacobian convolution (matches Fortran gpeq_contra):
    ξ^ψ·J(m) = Σ_dm jmat(dm) · xsp(m+dm)
    ξ^θ(m)   = -Σ_dm [jmat(dm)·xsp1(m+dm) + jmat1(dm)·xsp(m+dm)
                + 2πi·n/χ₁·jmat(dm)·xms(m+dm)] / (2πi·(m - n·q))
    ξ^ζ(m)   = -Σ_dm [q·jmat(dm)·xsp1(m+dm) + q·jmat1(dm)·xsp(m+dm)
                + 2πi·m/χ₁·jmat(dm)·xms(m+dm)] / (2πi·(m - n·q))

Covariant components from metric tensor contraction (matches Fortran gpeq_cova):
    ξ_ψ(m) = Σ_dm g¹¹J(dm)·ξ^ψJ(m+dm) + g¹²J(dm)·ξ^θ_m(m+dm) + g³¹J(dm)·ξ^ζ_m(m+dm)
    ξ_θ(m) = Σ_dm g¹²J(dm)·ξ^ψJ(m+dm) + g²²J(dm)·ξ^θ_m(m+dm) + g²³J(dm)·ξ^ζ_m(m+dm)
    ξ_ζ(m) = Σ_dm g³¹J(dm)·ξ^ψJ(m+dm) + g²³J(dm)·ξ^θ_m(m+dm) + g³³J(dm)·ξ^ζ_m(m+dm)
"""

"""
    reconstruct_physical_fields(
        response_vector, flux_matrix, ForceFreeStates_results,
        equil, ffs_intr, intr, metric, ffit, ctrl
    ) -> (xi_modes, b_modes)

Reconstruct displacement and perturbed magnetic field from eigenmode response.

Implements the full Fortran gpeq pipeline: gpeq_sol → gpeq_contra → gpeq_cova.
All fields returned in mode space [npsi, mpert].

# Returns

Tuple of (xi_modes, b_modes) NamedTuples:
- `xi_modes.psi`: ξ_ψ [npsi, mpert]
- `xi_modes.psi_J`: J·ξ^ψ (Jacobian-weighted, from gpeq_contra; used by gpeq_normal for b_n/xi_n)
- `xi_modes.theta`: ξ^θ contravariant (from gpeq_contra Jacobian convolution)
- `xi_modes.zeta`: ξ^ζ contravariant (from gpeq_contra Jacobian convolution)
- `xi_modes.clebsch_psi`: ξ^ψ for PENTRC (= xi_psi, unregularized)
- `xi_modes.clebsch_psi1`: ∂ξ^ψ/∂ψ regularized for PENTRC
- `xi_modes.clebsch_alpha`: ξ^α/χ₁ regularized for PENTRC (divided by χ₁ per gpout_xclebsch)
- `xi_modes.theta_reg`: ξ^θ regularized (= xmt, from gpeq_contra with reg_spot smoothing)
- `xi_modes.zeta_reg`: ξ^ζ regularized (= xmz, from gpeq_contra with reg_spot smoothing)
- `xi_modes.cova_psi/theta/zeta`: covariant displacement (from gpeq_cova)
- `b_modes.psi`: b^ψ [npsi, mpert]
- `b_modes.psi_area`: b^ψ / ⟨J·|∇ψ|⟩_θ (area-normalized, for b_n computation)
- `b_modes.theta`: b^θ [npsi, mpert]
- `b_modes.zeta`: b^ζ [npsi, mpert]
- `b_modes.theta_reg/zeta_reg`: regularized b^θ, b^ζ (from gpeq_sol with reg_spot smoothing)
- `b_modes.cova_psi/theta/zeta`: covariant field (from gpeq_cova)
"""
function reconstruct_physical_fields(
    response_vector::Vector{ComplexF64},
    flux_matrix::Matrix{ComplexF64},
    ForceFreeStates_results::OdeState,
    equil::Equilibrium.PlasmaEquilibrium,
    ffs_intr::ForceFreeStatesInternal,
    intr::PerturbedEquilibriumInternal,
    metric::MetricData,
    ffit::FourFitVars,
    ctrl::PerturbedEquilibriumControl
)
    npsi = size(ForceFreeStates_results.u_store, 4)
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

    # Compute Clebsch displacements with regularization (matches Fortran gpeq_sol + gpout_xclebsch)
    clebsch_psi, clebsch_psi1, clebsch_alpha = compute_clebsch_displacements(
        xi_psi_modes, xi_psi1_modes, xi_s_modes,
        psi_grid, equil, ffs_intr, ffit, ctrl
    )

    # Compute regularized (modified) b-field components (matches Fortran gpeq_sol bmt/bmz)
    b_theta_reg, b_zeta_reg = compute_modified_field_modes(
        xi_psi_modes, clebsch_psi1, clebsch_alpha,
        psi_grid, equil, ffs_intr
    )

    # Compute contravariant displacement via Jacobian convolution (matches Fortran gpeq_contra)
    xwp_modes, xwt_modes, xwz_modes, xmt_modes, xmz_modes = compute_contra_displacements(
        xi_psi_modes, clebsch_psi1, clebsch_alpha,
        psi_grid, equil, ffs_intr, metric, ctrl
    )

    # Compute covariant components via metric tensor contraction (matches Fortran gpeq_cova)
    xvp_modes, xvt_modes, xvz_modes,
    bvp_modes, bvt_modes, bvz_modes = compute_cova_components(
        xwp_modes, xmt_modes, xmz_modes,
        b_psi_modes, b_theta_reg, b_zeta_reg,
        psi_grid, ffs_intr, metric
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
            r2_y   = equil.rzphi_rsquared((psi, theta); deriv=DerivOp(0, 1), hint=hint2d)
            deta_y = equil.rzphi_offset((psi, theta);   deriv=DerivOp(0, 1), hint=hint2d)
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

    # R,Z,φ cylindrical components (Fortran gpeq_rzphi)
    # ξ: uses J-weighted psi (xwp from gpeq_contra) and regularized theta (xmt)
    # b: uses raw psi (bwp from gpeq_sol, no Jacobian convolution) and regularized theta (bmt)
    @warn "R,Z,φ cylindrical field components are in BETA: expect up to ~20% " *
          "discrepancies with Fortran GPEC (gpeq_rzphi). The ξ^ψ, ξ^θ, ξ^ζ and " *
          "corresponding b components in SFL coordinates remain well-benchmarked." maxlog=1
    mlow = ffs_intr.mlow
    xi_R, xi_Z, xi_phi = _compute_rzphi_modes(equil, psi_grid, mlow, size(xi_psi_modes, 2),
        xwp_modes, xmt_modes, xvz_modes)
    b_R, b_Z, b_phi = _compute_rzphi_modes(equil, psi_grid, mlow, size(b_psi_modes, 2),
        b_psi_modes, b_theta_reg, bvz_modes)

    xi_modes = (
        psi   = xi_psi_modes,
        psi_J = xwp_modes,     # J·ξ^ψ (Jacobian-weighted, from gpeq_contra)
        theta = xwt_modes,     # ξ^θ contravariant (from gpeq_contra)
        zeta  = xwz_modes,     # ξ^ζ contravariant (from gpeq_contra)
        theta_reg = xmt_modes, # ξ^θ regularized (from gpeq_contra, smoothed by reg_spot)
        zeta_reg  = xmz_modes, # ξ^ζ regularized (from gpeq_contra, smoothed by reg_spot)
        clebsch_psi   = clebsch_psi,    # ξ^ψ for PENTRC
        clebsch_psi1  = clebsch_psi1,   # ∂ξ^ψ/∂ψ regularized for PENTRC
        clebsch_alpha = clebsch_alpha,  # ξ^α/χ₁ regularized for PENTRC
        cova_psi   = xvp_modes,   # covariant ξ_ψ (from gpeq_cova)
        cova_theta = xvt_modes,   # covariant ξ_θ (from gpeq_cova)
        cova_zeta  = xvz_modes,   # covariant ξ_ζ (from gpeq_cova)
        R = xi_R, Z = xi_Z, phi = xi_phi,  # cylindrical (from gpeq_rzphi)
    )
    b_modes = (
        psi       = b_psi_modes,      # b^ψ (no Jacobian) — used for b_n normal projection
        psi_area  = Jb_psi_modes,     # b^ψ / ⟨J·|∇ψ|⟩_θ (area-normalized, for b_n)
        theta     = b_theta_modes,    # b^θ unregularized
        zeta      = b_zeta_modes,     # b^ζ unregularized
        theta_reg = b_theta_reg,      # b^θ regularized (from gpeq_sol with reg_spot)
        zeta_reg  = b_zeta_reg,       # b^ζ regularized
        cova_psi   = bvp_modes,       # covariant b_ψ (from gpeq_cova)
        cova_theta = bvt_modes,       # covariant b_θ (from gpeq_cova)
        cova_zeta  = bvz_modes,       # covariant b_ζ (from gpeq_cova)
        R = b_R, Z = b_Z, phi = b_phi,  # cylindrical (from gpeq_rzphi)
    )

    return xi_modes, b_modes
end

"""
    sum_eigenmode_contributions(
        response_vector, flux_matrix, ForceFreeStates_results, ffs_intr
    ) -> (xi_psi_modes, xi_psi1_modes, xi_s_modes)

Sum eigenmode contributions weighted by response coefficients.

`response_vector` (= P * Phi_x) is in mode (m,n) basis. To reconstruct physical
fields from eigenmode solutions in u_store/ud_store, first project to eigenmode amplitudes:
    alpha = flux_matrix \\ response_vector

Then sum eigenmode contributions at each radial point (matches Fortran gpeq_sol):
    xi_psi[ipsi, :]  = u_store[:, :, 1, ipsi]  * alpha   # Ξ_ψ
    xi_psi1[ipsi, :] = finite_diff(xi_psi, psi)           # dΞ_ψ/dψ (centered FD)
    xi_s[ipsi, :]    = ud_store[:, :, 2, ipsi] * alpha   # Ξ_s (toroidal, Glasser 2016 eq. 18)

# Returns

- `xi_psi_modes`: Radial displacement ξ_ψ(ψ, m) [npsi, mpert]
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
    xi_s_modes    = zeros(ComplexF64, npsi, mpert)
    for ipsi in 1:npsi
        # u_store[:,:,1] = Ξ_ψ (radial displacement)
        mul!(view(xi_psi_modes, ipsi, :),
             ForceFreeStates_results.u_store[:, :, 1, ipsi],
             alpha)
        # ud_store[:,:,2] = Ξ_s = -A⁻¹(B·Ξ'_ψ + C·Ξ_ψ) (toroidal displacement, Glasser 2016 eq. 18)
        mul!(view(xi_s_modes, ipsi, :),
             ForceFreeStates_results.ud_store[:, :, 2, ipsi],
             alpha)
    end

    # Compute dΞ_ψ/dψ via finite differences of xi_psi_modes instead of using
    # ud_store[:,:,1] (BS5 cached derivative), which has isolated single-point
    # spikes from stale FSAL derivative caching.
    psi = ForceFreeStates_results.psi_store
    xi_psi1_modes = zeros(ComplexF64, npsi, mpert)
    # Forward difference at first point
    xi_psi1_modes[1, :] .= (xi_psi_modes[2, :] .- xi_psi_modes[1, :]) ./ (psi[2] - psi[1])
    # Centered differences for interior points
    for i in 2:npsi-1
        dp = psi[i+1] - psi[i-1]
        xi_psi1_modes[i, :] .= (xi_psi_modes[i+1, :] .- xi_psi_modes[i-1, :]) ./ dp
    end
    # Backward difference at last point
    xi_psi1_modes[npsi, :] .= (xi_psi_modes[npsi, :] .- xi_psi_modes[npsi-1, :]) ./ (psi[npsi] - psi[npsi-1])

    return xi_psi_modes, xi_psi1_modes, xi_s_modes
end

"""
    compute_perturbed_field_modes(
        xi_psi_modes, xi_psi1_modes, xi_s_modes, psi_grid, equil, ffs_intr
    ) -> (b_psi_modes, b_theta_modes, b_zeta_modes)

Compute contravariant perturbed B-field from displacement using ideal MHD relations.

Matches Fortran `gpeq_sol` bwp/bwt/bwz (unregularized) [Park 2007 eq. 8-10]:

    b^ψ =  χ₁·(m - n·q)·2πi·ξ_ψ
    b^θ = -(χ₁·∂ξ_ψ/∂ψ + 2πi·n·ξ_s)
    b^ζ = -(χ₁·(q'·ξ_ψ + q·∂ξ_ψ/∂ψ) + 2πi·m·ξ_s)
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
    compute_clebsch_displacements(
        xi_psi_modes, xi_psi1_modes, xi_s_modes,
        psi_grid, equil, ffs_intr, ffit, ctrl
    ) -> (clebsch_psi, clebsch_psi1, clebsch_alpha)

Compute Clebsch displacement components for PENTRC output.

Matches Fortran gpeq_sol regularization + gpout_xclebsch output convention:
- `clebsch_psi` = ξ^ψ (unregularized, same as xi_psi_modes)
- `clebsch_psi1` = xmp1 = ∂ξ^ψ/∂ψ × singfac²/(singfac² + reg_spot²)
- `clebsch_alpha` = xms/χ₁ (regularized ξ^α divided by χ₁ per gpout_xclebsch convention)

When reg_spot=0, clebsch_psi1 = xi_psi1 and clebsch_alpha = xi_s/χ₁ (no regularization).

The regularized xms is computed as -A⁻¹(B·xmp1 + C·xsp) matching Fortran gpeq_sol,
where A, B, C are the stability matrices evaluated at each ψ via ffit interpolants.
"""
function compute_clebsch_displacements(
    xi_psi_modes::Matrix{ComplexF64},
    xi_psi1_modes::Matrix{ComplexF64},
    xi_s_modes::Matrix{ComplexF64},
    psi_grid::Vector{Float64},
    equil::Equilibrium.PlasmaEquilibrium,
    ffs_intr::ForceFreeStatesInternal,
    ffit::FourFitVars,
    ctrl::PerturbedEquilibriumControl
)
    npsi, mpert = size(xi_psi_modes)
    nn = ffs_intr.nlow
    mlow = ffs_intr.mlow
    chi1 = 2π * equil.psio
    numpert_total = ffs_intr.numpert_total

    clebsch_psi   = copy(xi_psi_modes)        # ξ^ψ (unregularized)
    clebsch_psi1  = copy(xi_psi1_modes)        # will be regularized below
    clebsch_alpha = xi_s_modes ./ chi1         # ξ^α/χ₁ (will be regularized below)

    reg_spot = ctrl.reg_spot
    @assert reg_spot >= 0 "reg_spot must be non-negative (got $reg_spot)"

    if reg_spot == 0
        return clebsch_psi, clebsch_psi1, clebsch_alpha
    end

    # Pre-allocate buffers for matrix operations
    amat = Matrix{ComplexF64}(undef, numpert_total, numpert_total)
    bmat = Matrix{ComplexF64}(undef, numpert_total, numpert_total)
    cmat_buf = Matrix{ComplexF64}(undef, numpert_total, numpert_total)
    xmp1_vec = Vector{ComplexF64}(undef, mpert)
    xms_vec  = Vector{ComplexF64}(undef, mpert)

    hint = Ref(1)

    for ipsi in 1:npsi
        psi_norm = psi_grid[ipsi]
        q = equil.profiles.q_spline(psi_norm)

        # Apply diagonal regularization to xsp1 → xmp1
        for ipert in 1:mpert
            m = mlow + ipert - 1
            singfac = m - nn * q
            reg_factor = singfac^2 / (singfac^2 + reg_spot^2)
            clebsch_psi1[ipsi, ipert] = xi_psi1_modes[ipsi, ipert] * reg_factor
            xmp1_vec[ipert] = clebsch_psi1[ipsi, ipert]
        end

        # Compute regularized xms = -A⁻¹(B·xmp1 + C·xsp) (matches Fortran gpeq_sol)
        # Evaluate stability matrices at this psi
        ffit.amats(view(amat, :), psi_norm; hint=hint)
        ffit.bmats(view(bmat, :), psi_norm; hint=hint)
        ffit.cmats(view(cmat_buf, :), psi_norm; hint=hint)

        # xms = -(A\B)*xmp1 - (A\C)*xsp
        xsp_vec = view(xi_psi_modes, ipsi, :)
        mul!(xms_vec, bmat, xmp1_vec)                     # xms = B*xmp1
        mul!(xms_vec, cmat_buf, xsp_vec, 1.0+0.0im, 1.0+0.0im)  # xms += C*xsp
        # Solve A*result = xms for result, then negate
        amat_fact = cholesky(Hermitian(amat, :L))
        ldiv!(amat_fact, xms_vec)                          # xms = A\(B*xmp1 + C*xsp)
        xms_vec .*= -1                                     # xms = -A\(B*xmp1 + C*xsp)

        clebsch_alpha[ipsi, :] .= xms_vec ./ chi1         # ξ^α/χ₁
    end

    return clebsch_psi, clebsch_psi1, clebsch_alpha
end

"""
    compute_modified_field_modes(
        xi_psi_modes, clebsch_psi1, clebsch_alpha, psi_grid, equil, ffs_intr
    ) -> (b_theta_reg, b_zeta_reg)

Compute regularized (modified) contravariant B-field components.

Matches Fortran gpeq_sol bmt/bmz using regularized xmp1 and xms:
    b^θ_m = -(χ₁·xmp1 + 2πi·n·xms)
    b^ζ_m = -(χ₁·(q'·xsp + q·xmp1) + 2πi·m·xms)

Note: clebsch_alpha = xms/χ₁, so xms = clebsch_alpha * χ₁.
"""
function compute_modified_field_modes(
    xi_psi_modes::Matrix{ComplexF64},
    clebsch_psi1::Matrix{ComplexF64},
    clebsch_alpha::Matrix{ComplexF64},
    psi_grid::Vector{Float64},
    equil::Equilibrium.PlasmaEquilibrium,
    ffs_intr::ForceFreeStatesInternal
)
    npsi, mpert = size(xi_psi_modes)
    mlow = ffs_intr.mlow
    nn = ffs_intr.nlow
    chi1 = 2π * equil.psio

    b_theta_reg = zeros(ComplexF64, npsi, mpert)
    b_zeta_reg  = zeros(ComplexF64, npsi, mpert)

    for ipsi in 1:npsi
        psi_norm = psi_grid[ipsi]
        q = equil.profiles.q_spline(psi_norm)
        q1 = equil.profiles.q_deriv(psi_norm)

        for ipert in 1:mpert
            m = mlow + ipert - 1
            xsp  = xi_psi_modes[ipsi, ipert]
            xmp1 = clebsch_psi1[ipsi, ipert]
            xms  = clebsch_alpha[ipsi, ipert] * chi1  # undo χ₁ division

            b_theta_reg[ipsi, ipert] = -(chi1 * xmp1 + 2π * im * nn * xms)
            b_zeta_reg[ipsi, ipert]  = -(chi1 * (q1 * xsp + q * xmp1) + 2π * im * m * xms)
        end
    end

    return b_theta_reg, b_zeta_reg
end

"""
    compute_contra_displacements(
        xi_psi_modes, clebsch_psi1, clebsch_alpha,
        psi_grid, equil, ffs_intr, metric, ctrl
    ) -> (xwp_modes, xwt_modes, xwz_modes, xmt_modes, xmz_modes)

Compute contravariant displacement via Jacobian mode coupling convolution.

Matches Fortran `gpeq_contra`:
    ξ^ψ·J(m) = Σ_dm jmat(dm) · xsp(m+dm)
    ξ^θ(m)   = -Σ_dm [...] / (2πi·singfac)
    ξ^ζ(m)   = -Σ_dm [...] / (2πi·singfac)

Uses regularized xmp1 and xms for the theta/zeta components, then optionally applies
additional regularization to get xmt/xmz.

Returns both unregularized (xwt, xwz) and regularized (xmt, xmz) versions.
"""
function compute_contra_displacements(
    xi_psi_modes::Matrix{ComplexF64},
    clebsch_psi1::Matrix{ComplexF64},
    clebsch_alpha::Matrix{ComplexF64},
    psi_grid::Vector{Float64},
    equil::Equilibrium.PlasmaEquilibrium,
    ffs_intr::ForceFreeStatesInternal,
    metric::MetricData,
    ctrl::PerturbedEquilibriumControl
)
    npsi, mpert = size(xi_psi_modes)
    mlow = ffs_intr.mlow
    mband = ffs_intr.mband
    nn = ffs_intr.nlow
    chi1 = 2π * equil.psio
    reg_spot = ctrl.reg_spot
    fc = metric.fourier_coeffs
    mid = mband + 1

    xwp_modes = zeros(ComplexF64, npsi, mpert)
    xwt_modes = zeros(ComplexF64, npsi, mpert)
    xwz_modes = zeros(ComplexF64, npsi, mpert)

    # Pre-allocate Fourier coefficient vectors
    jmat  = Vector{ComplexF64}(undef, 2 * mband + 1)
    jmat1 = Vector{ComplexF64}(undef, 2 * mband + 1)

    # Find closest metric psi grid index for Fourier coefficient lookup
    metric_xs = metric.xs

    for ipsi in 1:npsi
        psi_norm = psi_grid[ipsi]
        q = equil.profiles.q_spline(psi_norm)

        # Find closest metric grid point for Fourier coefficient extraction
        ipsi_metric = _find_closest_index(metric_xs, psi_norm)

        # Extract Jacobian Fourier coefficients at this psi
        for m in 0:mband
            jmat[mid-m]  = Utilities.get_complex_coeff(fc, ipsi_metric, m, 7)
            jmat1[mid-m] = Utilities.get_complex_coeff(fc, ipsi_metric, m, 8)
        end
        for k in 1:mband
            jmat[mid+k]  = conj(jmat[mid-k])
            jmat1[mid+k] = conj(jmat1[mid-k])
        end

        # Matches Fortran gpeq_contra: uses regularized xmp1/xms for theta/zeta components
        for ipert in 1:mpert
            m = mlow + ipert - 1
            singfac = m - nn * q
            twopi_i_singfac = 2π * im * singfac

            xwp_acc = zero(ComplexF64)
            xwt_acc = zero(ComplexF64)
            xwz_acc = zero(ComplexF64)

            for dm in max(1-ipert, -mband):min(mpert-ipert, mband)
                jpert = ipert + dm
                dmidx = dm + mid

                xsp_j  = xi_psi_modes[ipsi, jpert]
                xmp1_j = clebsch_psi1[ipsi, jpert]
                xms_j  = clebsch_alpha[ipsi, jpert] * chi1  # undo χ₁ division

                # ξ^ψ·J: Jacobian-weighted psi displacement
                xwp_acc += jmat[dmidx] * xsp_j

                # theta and zeta numerators (matches Fortran gpeq_contra)
                xwt_acc += jmat[dmidx] * xmp1_j + jmat1[dmidx] * xsp_j +
                           2π * im * nn / chi1 * jmat[dmidx] * xms_j
                xwz_acc += q * jmat[dmidx] * xmp1_j + q * jmat1[dmidx] * xsp_j +
                           2π * im * m / chi1 * jmat[dmidx] * xms_j
            end

            xwp_modes[ipsi, ipert] = xwp_acc

            # Divide by 2πi·singfac (avoid division by zero near rational surfaces)
            if abs(twopi_i_singfac) > 1e-30
                xwt_modes[ipsi, ipert] = -xwt_acc / twopi_i_singfac
                xwz_modes[ipsi, ipert] = -xwz_acc / twopi_i_singfac
            end
        end
    end

    # Regularize xwt/xwz → xmt/xmz (matches Fortran gpeq_contra)
    xmt_modes = copy(xwt_modes)
    xmz_modes = copy(xwz_modes)
    if reg_spot > 0
        for ipsi in 1:npsi
            psi_norm = psi_grid[ipsi]
            q = equil.profiles.q_spline(psi_norm)
            for ipert in 1:mpert
                m = mlow + ipert - 1
                singfac = m - nn * q
                reg_factor = singfac^2 / (singfac^2 + reg_spot^2)
                xmt_modes[ipsi, ipert] = xwt_modes[ipsi, ipert] * reg_factor
                xmz_modes[ipsi, ipert] = xwz_modes[ipsi, ipert] * reg_factor
            end
        end
    end

    return xwp_modes, xwt_modes, xwz_modes, xmt_modes, xmz_modes
end

"""
    compute_cova_components(
        xwp_modes, xmt_modes, xmz_modes,
        bwp_modes, bmt_modes, bmz_modes,
        psi_grid, ffs_intr, metric
    ) -> (xvp, xvt, xvz, bvp, bvt, bvz)

Compute covariant displacement and B-field via metric tensor contraction.

Matches Fortran `gpeq_cova`: contracts contravariant components with metric tensor
Fourier coefficients (g^ij·J, quantities 1-6 in metric.fourier_coeffs) via poloidal
mode coupling convolution.

Uses regularized (modified) xmt/xmz and bmt/bmz for the theta/zeta contravariant inputs.
"""
function compute_cova_components(
    xwp_modes::Matrix{ComplexF64},
    xmt_modes::Matrix{ComplexF64},
    xmz_modes::Matrix{ComplexF64},
    bwp_modes::Matrix{ComplexF64},
    bmt_modes::Matrix{ComplexF64},
    bmz_modes::Matrix{ComplexF64},
    psi_grid::Vector{Float64},
    ffs_intr::ForceFreeStatesInternal,
    metric::MetricData
)
    npsi, mpert = size(xwp_modes)
    mlow = ffs_intr.mlow
    mband = ffs_intr.mband
    mid = mband + 1
    fc = metric.fourier_coeffs
    metric_xs = metric.xs

    xvp_modes = zeros(ComplexF64, npsi, mpert)
    xvt_modes = zeros(ComplexF64, npsi, mpert)
    xvz_modes = zeros(ComplexF64, npsi, mpert)
    bvp_modes = zeros(ComplexF64, npsi, mpert)
    bvt_modes = zeros(ComplexF64, npsi, mpert)
    bvz_modes = zeros(ComplexF64, npsi, mpert)

    # Pre-allocate metric Fourier coefficient vectors
    g11 = Vector{ComplexF64}(undef, 2 * mband + 1)
    g22 = Vector{ComplexF64}(undef, 2 * mband + 1)
    g33 = Vector{ComplexF64}(undef, 2 * mband + 1)
    g23 = Vector{ComplexF64}(undef, 2 * mband + 1)
    g31 = Vector{ComplexF64}(undef, 2 * mband + 1)
    g12 = Vector{ComplexF64}(undef, 2 * mband + 1)

    for ipsi in 1:npsi
        psi_norm = psi_grid[ipsi]
        ipsi_metric = _find_closest_index(metric_xs, psi_norm)

        # Extract metric tensor Fourier coefficients (g^ij·J)
        for m in 0:mband
            g11[mid-m] = Utilities.get_complex_coeff(fc, ipsi_metric, m, 1)
            g22[mid-m] = Utilities.get_complex_coeff(fc, ipsi_metric, m, 2)
            g33[mid-m] = Utilities.get_complex_coeff(fc, ipsi_metric, m, 3)
            g23[mid-m] = Utilities.get_complex_coeff(fc, ipsi_metric, m, 4)
            g31[mid-m] = Utilities.get_complex_coeff(fc, ipsi_metric, m, 5)
            g12[mid-m] = Utilities.get_complex_coeff(fc, ipsi_metric, m, 6)
        end
        for k in 1:mband
            g11[mid+k] = conj(g11[mid-k])
            g22[mid+k] = conj(g22[mid-k])
            g33[mid+k] = conj(g33[mid-k])
            g23[mid+k] = conj(g23[mid-k])
            g31[mid+k] = conj(g31[mid-k])
            g12[mid+k] = conj(g12[mid-k])
        end

        # Tensor contraction with mode coupling (matches Fortran gpeq_cova)
        for ipert in 1:mpert
            xvp_acc = zero(ComplexF64); xvt_acc = zero(ComplexF64); xvz_acc = zero(ComplexF64)
            bvp_acc = zero(ComplexF64); bvt_acc = zero(ComplexF64); bvz_acc = zero(ComplexF64)

            for dm in max(1-ipert, -mband):min(mpert-ipert, mband)
                jpert = ipert + dm
                dmidx = dm + mid

                # Displacement covariant: ξ_i = g_ij · ξ^j (tensor contraction)
                xvp_acc += g11[dmidx] * xwp_modes[ipsi, jpert] + g12[dmidx] * xmt_modes[ipsi, jpert] + g31[dmidx] * xmz_modes[ipsi, jpert]
                xvt_acc += g12[dmidx] * xwp_modes[ipsi, jpert] + g22[dmidx] * xmt_modes[ipsi, jpert] + g23[dmidx] * xmz_modes[ipsi, jpert]
                xvz_acc += g31[dmidx] * xwp_modes[ipsi, jpert] + g23[dmidx] * xmt_modes[ipsi, jpert] + g33[dmidx] * xmz_modes[ipsi, jpert]

                # B-field covariant: b_i = g_ij · b^j
                bvp_acc += g11[dmidx] * bwp_modes[ipsi, jpert] + g12[dmidx] * bmt_modes[ipsi, jpert] + g31[dmidx] * bmz_modes[ipsi, jpert]
                bvt_acc += g12[dmidx] * bwp_modes[ipsi, jpert] + g22[dmidx] * bmt_modes[ipsi, jpert] + g23[dmidx] * bmz_modes[ipsi, jpert]
                bvz_acc += g31[dmidx] * bwp_modes[ipsi, jpert] + g23[dmidx] * bmt_modes[ipsi, jpert] + g33[dmidx] * bmz_modes[ipsi, jpert]
            end

            xvp_modes[ipsi, ipert] = xvp_acc
            xvt_modes[ipsi, ipert] = xvt_acc
            xvz_modes[ipsi, ipert] = xvz_acc
            bvp_modes[ipsi, ipert] = bvp_acc
            bvt_modes[ipsi, ipert] = bvt_acc
            bvz_modes[ipsi, ipert] = bvz_acc
        end
    end

    return xvp_modes, xvt_modes, xvz_modes, bvp_modes, bvt_modes, bvz_modes
end

"""
    _find_closest_index(xs, x) -> Int

Find index of closest value in sorted array `xs` to target `x`.
"""
function _find_closest_index(xs::AbstractVector{Float64}, x::Float64)
    idx = searchsortedfirst(xs, x)
    if idx > length(xs)
        return length(xs)
    elseif idx == 1
        return 1
    else
        return abs(xs[idx] - x) < abs(xs[idx-1] - x) ? idx : idx - 1
    end
end

"""
    compute_b_n_xi_n_modes(
        xwp_modes, b_psi_modes, ForceFreeStates_results, equil, ffs_intr
    ) -> (b_n_modes, xi_n_modes)

Compute physical normal field b_n and displacement xi_n in mode space.

Matches Fortran `gpeq_normal`:
- IDFT: reconstruct theta-space functions from mode amplitudes
- Divide by J·|∇ψ|(θ) at each theta point to get physical normal components
- Forward DFT back to mode space

`xwp_modes` = J·ξ^ψ (Jacobian-weighted from gpeq_contra convolution).
`b_psi_modes` = b^ψ (raw, not J-weighted).
Both are divided by J·|∇ψ| in theta space, matching Fortran gpeq_normal.
For xi_n: IDFT(J·ξ^ψ) / (J·|∇ψ|) = ξ^ψ/|∇ψ| = ξ_n.
For b_n:  IDFT(b^ψ) / (J·|∇ψ|) [Park Phys. Plasmas 14, 052110 (2007)].

DFT resolution: `mthsurf = mtheta = length(equil.rzphi_ys) - 1`. This is sufficient since
Nyquist (mtheta/2) >> 2·max|m|, so no aliasing from the 1/(J·|∇ψ|) division.

# Returns

Tuple (b_n_modes, xi_n_modes), each [npsi, mpert] ComplexF64.
"""
function compute_b_n_xi_n_modes(
    xwp_modes::Matrix{ComplexF64},
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
        bwp_fun  = phase_fwd * b_psi_modes[ipsi, :]   # [mthsurf] — b^ψ (not J-weighted)
        xwp_fun  = phase_fwd * xwp_modes[ipsi, :]     # [mthsurf] — J·ξ^ψ (J-weighted from gpeq_contra)
        delpsis = zeros(Float64, mthsurf)
        jacs  = zeros(Float64, mthsurf)
        for k in 1:mthsurf
            theta = thetas[k]
            r2    = equil.rzphi_rsquared((psi, theta); hint=hint2d_psi)
            deta  = equil.rzphi_offset((psi, theta); hint=hint2d_psi)
            jac   = equil.rzphi_jac((psi, theta); hint=hint2d_psi)
            r2_y  = equil.rzphi_rsquared((psi, theta); deriv=DerivOp(0, 1), hint=hint2d_psi)
            deta_y = equil.rzphi_offset((psi, theta); deriv=DerivOp(0, 1), hint=hint2d_psi)
            rfac  = sqrt(abs(r2))
            eta   = twopi * (theta + deta)
            r     = ro + rfac * cos(eta)
            w11   = (1.0 + deta_y) * twopi^2 * rfac * r / jac
            w12   = -r2_y * π * r / (rfac * jac)
            delpsi = sqrt(w11^2 + w12^2)
            delpsis[k] = delpsi
            jacs[k]  = jac
        end

        # Divide by J·|∇ψ| → physical normal components in theta space (matches Fortran gpeq_normal)
        jd = jacs .* delpsis
        bno_fun  = bwp_fun ./ jd
        xno_fun  = xwp_fun ./ jd

        # Forward DFT: theta space → mode space (1/mthsurf normalization matches Fortran iscdftf).
        # Must use transpose (not adjoint) so the phase is exp(-2πi·m·θ), not exp(+2πi·m·θ).
        b_n_modes[ipsi, :]  = (transpose(phase_back) * bno_fun) ./ mthsurf
        xi_n_modes[ipsi, :] = (transpose(phase_back) * xno_fun) ./ mthsurf
    end

    return b_n_modes, xi_n_modes
end

"""
    _compute_rzphi_modes(equil, psi_grid, mlow, mpert,
        psi_input, theta_input, cova_zeta_input) -> (R_modes, Z_modes, phi_modes)

Convert mode-space perturbation from flux coordinates (ψ,θ,ζ) to cylindrical (R,Z,φ).

Implements Fortran `gpeq_rzphi` (gpeq.f:439-501):
    R(m)   = DFT[ (t11·psi_input + t12·theta_input) / J ]
    Z(m)   = DFT[ (t21·psi_input + t22·theta_input) / J ]
    φ(m)   = DFT[ t33 · cova_zeta_input ]

The caller passes different inputs for ξ vs b (see `reconstruct_physical_fields`).

!!! warning "Beta"
    The R,Z,φ cylindrical components are in beta and currently show up to ~20%
    discrepancies vs Fortran GPEC. The underlying SFL-coordinate quantities
    (ξ^ψ, ξ^θ, ξ^ζ and the corresponding b components) remain well-benchmarked.
"""
function _compute_rzphi_modes(
    equil::Equilibrium.PlasmaEquilibrium,
    psi_grid::Vector{Float64},
    mlow::Int, mpert::Int,
    psi_input::Matrix{ComplexF64},
    theta_input::Matrix{ComplexF64},
    cova_zeta_input::Matrix{ComplexF64}
)
    npsi = length(psi_grid)
    R0 = equil.ro

    # Theta grid for DFT
    mtheta = max(2 * (abs(mlow) + mpert), 512)
    ft = Utilities.FourierTransforms.FourierTransform(mtheta, mpert, mlow)

    R_modes   = zeros(ComplexF64, npsi, mpert)
    Z_modes   = zeros(ComplexF64, npsi, mpert)
    phi_modes = zeros(ComplexF64, npsi, mpert)

    # Pre-allocate theta-space buffers
    t11 = Vector{Float64}(undef, mtheta)
    t12 = Vector{Float64}(undef, mtheta)
    t21 = Vector{Float64}(undef, mtheta)
    t22 = Vector{Float64}(undef, mtheta)
    t33 = Vector{Float64}(undef, mtheta)
    J_theta = Vector{Float64}(undef, mtheta)

    hint = (Ref(1), Ref(1))

    for ipsi in 1:npsi
        psi = psi_grid[ipsi]

        # Compute transformation matrix at each theta point (Fortran gpeq.f:458-475)
        for itheta in 1:mtheta
            theta = (itheta - 1) / mtheta  # SFL theta ∈ [0, 1)
            pt = (psi, theta)

            r2   = equil.rzphi_rsquared(pt; hint=hint)
            deta = equil.rzphi_offset(pt; hint=hint)
            jac  = equil.rzphi_jac(pt; hint=hint)

            dr2_dpsi    = equil.rzphi_rsquared(pt; deriv=DerivOp(1, 0), hint=hint)
            dr2_dtheta  = equil.rzphi_rsquared(pt; deriv=DerivOp(0, 1), hint=hint)
            doff_dpsi   = equil.rzphi_offset(pt; deriv=DerivOp(1, 0), hint=hint)
            doff_dtheta = equil.rzphi_offset(pt; deriv=DerivOp(0, 1), hint=hint)

            rfac = sqrt(max(0.0, r2))
            eta  = 2π * (theta + deta)
            s, c = sincos(eta)
            R_here = R0 + rfac * c

            if rfac > 1e-30
                v11 = dr2_dpsi / (2 * rfac)
                v12 = doff_dpsi * 2π * rfac
                v21 = dr2_dtheta / (2 * rfac)
                v22 = (1 + doff_dtheta) * 2π * rfac
            else
                v11 = 0.0; v12 = 0.0; v21 = 0.0; v22 = 0.0
            end

            t11[itheta] = c * v11 - s * v12
            t12[itheta] = c * v21 - s * v22
            t21[itheta] = s * v11 + c * v12
            t22[itheta] = s * v21 + c * v22
            t33[itheta] = -1.0 / (2π * R_here)
            J_theta[itheta] = jac
        end

        # Inverse DFT: modes → theta-space
        psi_fun  = Utilities.FourierTransforms.inverse(ft, view(psi_input, ipsi, :))
        theta_fn = Utilities.FourierTransforms.inverse(ft, view(theta_input, ipsi, :))
        zeta_fn  = Utilities.FourierTransforms.inverse(ft, view(cova_zeta_input, ipsi, :))

        # Pointwise transformation (Fortran gpeq_rzphi, gpeq.f:484-489)
        R_fun   = Vector{ComplexF64}(undef, mtheta)
        Z_fun   = Vector{ComplexF64}(undef, mtheta)
        phi_fun = Vector{ComplexF64}(undef, mtheta)

        for itheta in 1:mtheta
            J = J_theta[itheta]
            xwp = psi_fun[itheta]
            xwt = theta_fn[itheta]
            xvz = zeta_fn[itheta]

            if abs(J) > 1e-30
                R_fun[itheta] = (t11[itheta] * xwp + t12[itheta] * xwt) / J
                Z_fun[itheta] = (t21[itheta] * xwp + t22[itheta] * xwt) / J
            else
                R_fun[itheta] = zero(ComplexF64)
                Z_fun[itheta] = zero(ComplexF64)
            end
            phi_fun[itheta] = t33[itheta] * xvz
        end

        # Forward DFT: theta-space → modes (round-trip = 1 with 1/N forward normalization)
        R_modes[ipsi, :]   .= ft(R_fun)
        Z_modes[ipsi, :]   .= ft(Z_fun)
        phi_modes[ipsi, :] .= ft(phi_fun)
    end

    return R_modes, Z_modes, phi_modes
end
