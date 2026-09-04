"""
Field reconstruction from eigenmode response.

Converts eigenmode response coefficients to physical displacement and magnetic
field perturbations in mode space, following the GPEC gpeq module approach.

Displacement components from ODE integration (u_store/du_store/xi_s_store):
- ξ_ψ: radial displacement (u_store[:,:,1,:])
- dξ_ψ/dψ: radial derivative (du_store)
- ξ_s: toroidal displacement (xi_s_store, Glasser 2016 eq. 18)

Contravariant perturbed field from ideal MHD (matches Fortran gpeq_sol):
    b^ψ =  χ₁·(m - n·q)·2πi·ξ_ψ
    b^θ = -(χ₁·dξ_ψ/dψ + 2πi·n·ξ_s)
    b^ζ = -(χ₁·(q'·ξ_ψ + q·dξ_ψ/dψ) + 2πi·m·ξ_s)

where χ₁ = 2π·Ψ₀ [Park Phys. Plasmas 14, 052110 (2007) eq. 8-10].

Clebsch displacement components for PENTRC (matches Fortran gpout_xclebsch):
    ξ^ψ         = xsp_mn    (unregularized)
    ∂ξ^ψ/∂ψ    = xmp1_mn   (regularized: xsp1 * singfac²/(singfac² + reg_spot²))
    ξ^α         = xms_mn    (regularized: -A⁻¹(B·xmp1 + C·xsp) ideal, or ξ_s scaled by the same
                             singfac factor when kinetic; divided by χ₁ in output)

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
        response_vector, flux_matrix, solution,
        equil, ffs, intr, metric, mats, ctrl
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
  - `b_modes.b_psi_area_weighted`: b^ψ / ⟨J·|∇ψ|⟩_θ (area-normalized, for b_n computation)
  - `b_modes.theta`: b^θ [npsi, mpert]
  - `b_modes.zeta`: b^ζ [npsi, mpert]
  - `b_modes.theta_reg/zeta_reg`: regularized b^θ, b^ζ (from gpeq_sol with reg_spot smoothing)
  - `b_modes.cova_psi/theta/zeta`: covariant field (from gpeq_cova)
"""
function reconstruct_physical_fields(
    response_vector::Vector{ComplexF64},
    flux_matrix::Matrix{ComplexF64},
    solution::SolutionProfiles,
    equil::Equilibrium.PlasmaEquilibrium,
    ffs::ForceFreeStatesResult,
    intr::PerturbedEquilibriumInternal,
    metric::MetricData,
    mats::MatrixSplines,
    ctrl::PerturbedEquilibriumControl
)
    npsi = size(solution.u_store, 4)
    psi_grid = solution.psi_store[1:npsi]

    # Pin BLAS to a single thread for the per-surface reconstruction below. Each threaded
    # loop over ψ calls only small BLAS kernels (per-surface mpert×mpert solves and mode
    # DFTs), so leaving BLAS multi-threaded oversubscribes the cores against the Julia
    # `@threads` over ψ and destroys thread scaling. Restored in a `finally` so an exception
    # in any of the threaded loops cannot leak the pinned count into the rest of the session.
    _blas_nthreads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        # Sum weighted eigenmode contributions to get ξ_ψ, dξ_ψ/dψ, and ξ_s in mode space
        xi_psi_modes, xi_psi1_modes, xi_s_modes = sum_eigenmode_contributions(
            response_vector,
            flux_matrix,
            solution,
            ffs
        )

        # Compute perturbed field in mode space using ideal MHD relations
        # [Park Phys. Plasmas 14, 052110 (2007) eq. 8-10]
        b_psi_modes, b_theta_modes, b_zeta_modes = compute_perturbed_field_modes(
            xi_psi_modes, xi_psi1_modes, xi_s_modes,
            psi_grid, equil, ffs
        )

        # Compute Clebsch displacements with regularization (matches Fortran gpeq_sol + gpout_xclebsch)
        clebsch_psi, clebsch_psi1, clebsch_alpha = compute_clebsch_displacements(
            xi_psi_modes, xi_psi1_modes, xi_s_modes,
            psi_grid, equil, ffs, mats, ctrl
        )

        # Compute regularized (modified) b-field components (matches Fortran gpeq_sol bmt/bmz)
        b_theta_reg, b_zeta_reg = compute_modified_field_modes(
            xi_psi_modes, clebsch_psi1, clebsch_alpha,
            psi_grid, equil, ffs
        )

        # Compute contravariant displacement via Jacobian convolution (matches Fortran gpeq_contra)
        xwp_modes, xwt_modes, xwz_modes, xmt_modes, xmz_modes = compute_contra_displacements(
            xi_psi_modes, clebsch_psi1, clebsch_alpha,
            psi_grid, equil, ffs, metric, ctrl
        )

        # Compute covariant components via metric tensor contraction (matches Fortran gpeq_cova)
        xvp_modes, xvt_modes, xvz_modes,
        bvp_modes, bvt_modes, bvz_modes = compute_cova_components(
            xwp_modes, xmt_modes, xmz_modes,
            b_psi_modes, b_theta_reg, b_zeta_reg,
            psi_grid, ffs, metric
        )

        # Compute b^ψ / area for HDF5 output — matches Fortran gpout_xbnormal fast path
        # area = <J·|∇ψ|>_θ (Hamada: J is const over θ)
        ro = equil.ro
        mthsurf = length(equil.rzphi_ys) - 1
        thetas_for_avg = [(k - 1) / mthsurf for k in 1:mthsurf]
        Jb_psi_modes = similar(b_psi_modes)
        # Surfaces are independent; :static keeps threadid() stable for any per-thread state.
        Threads.@threads :static for ipsi in 1:npsi
            psi = psi_grid[ipsi]
            hint2d = (Ref(1), Ref(1))
            area = 0.0
            for k in 1:mthsurf
                theta = thetas_for_avg[k]
                r2 = equil.rzphi_rsquared((psi, theta); hint=hint2d)
                deta = equil.rzphi_offset((psi, theta); hint=hint2d)
                jac = equil.rzphi_jac((psi, theta); hint=hint2d)
                r2_y = equil.rzphi_rsquared((psi, theta); deriv=DerivOp(0, 1), hint=hint2d)
                deta_y = equil.rzphi_offset((psi, theta); deriv=DerivOp(0, 1), hint=hint2d)
                rfac = sqrt(abs(r2))
                eta = 2π * (theta + deta)
                r = ro + rfac * cos(eta)
                w11 = (1.0 + deta_y) * 4π^2 * rfac * r / jac
                w12 = -r2_y * π * r / (rfac * jac)
                delpsi = sqrt(w11^2 + w12^2)
                area += jac * delpsi
            end
            area /= mthsurf
            Jb_psi_modes[ipsi, :] = b_psi_modes[ipsi, :] ./ area
        end

        # R,Z,φ cylindrical components (Fortran gpeq_rzphi)
        # ξ: uses J-weighted psi (xwp from gpeq_contra) and regularized theta (xmt)
        # b: uses raw psi (bwp from gpeq_sol, no Jacobian convolution) and regularized theta (bmt)
        # Build the (ψ,θ) flux→cylindrical geometry once and reuse it for both ξ and b: the
        # transform matrices depend only on equilibrium geometry, not on the perturbed field.
        mlow = ffs.mlow
        mpert_rz = size(xi_psi_modes, 2)
        mtheta_rz = max(2 * (abs(mlow) + mpert_rz), 512)
        ft_rz = Utilities.FourierTransforms.FourierTransform(mtheta_rz, mpert_rz, mlow)
        rzphi_geom = _build_rzphi_geometry(equil, psi_grid, mtheta_rz)
        xi_R, xi_Z, xi_phi = _apply_rzphi_transform(rzphi_geom, ft_rz, mtheta_rz, xwp_modes, xmt_modes, xvz_modes)
        b_R, b_Z, b_phi = _apply_rzphi_transform(rzphi_geom, ft_rz, mtheta_rz, b_psi_modes, b_theta_reg, bvz_modes)

        xi_modes = (
            psi=xi_psi_modes,
            psi_J=xwp_modes,     # J·ξ^ψ (Jacobian-weighted, from gpeq_contra)
            theta=xwt_modes,     # ξ^θ contravariant (from gpeq_contra)
            zeta=xwz_modes,     # ξ^ζ contravariant (from gpeq_contra)
            theta_reg=xmt_modes, # ξ^θ regularized (from gpeq_contra, smoothed by reg_spot)
            zeta_reg=xmz_modes, # ξ^ζ regularized (from gpeq_contra, smoothed by reg_spot)
            clebsch_psi=clebsch_psi,    # ξ^ψ for PENTRC
            clebsch_psi1=clebsch_psi1,   # ∂ξ^ψ/∂ψ regularized for PENTRC
            clebsch_alpha=clebsch_alpha,  # ξ^α/χ₁ regularized for PENTRC
            cova_psi=xvp_modes,   # covariant ξ_ψ (from gpeq_cova)
            cova_theta=xvt_modes,   # covariant ξ_θ (from gpeq_cova)
            cova_zeta=xvz_modes,   # covariant ξ_ζ (from gpeq_cova)
            R=xi_R, Z=xi_Z, phi=xi_phi
        )
        b_modes = (
            psi=b_psi_modes,      # b^ψ (no Jacobian) — used for b_n normal projection
            b_psi_area_weighted=Jb_psi_modes,     # b^ψ / ⟨J·|∇ψ|⟩_θ (area-normalized, for b_n)
            theta=b_theta_modes,    # b^θ unregularized
            zeta=b_zeta_modes,     # b^ζ unregularized
            theta_reg=b_theta_reg,      # b^θ regularized (from gpeq_sol with reg_spot)
            zeta_reg=b_zeta_reg,       # b^ζ regularized
            cova_psi=bvp_modes,       # covariant b_ψ (from gpeq_cova)
            cova_theta=bvt_modes,       # covariant b_θ (from gpeq_cova)
            cova_zeta=bvz_modes,       # covariant b_ζ (from gpeq_cova)
            R=b_R, Z=b_Z, phi=b_phi
        )

        return xi_modes, b_modes
    finally
        BLAS.set_num_threads(_blas_nthreads)
    end
end

"""
    sum_eigenmode_contributions(
        response_vector, flux_matrix, solution, ffs
    ) -> (xi_psi_modes, xi_psi1_modes, xi_s_modes)

Sum eigenmode contributions weighted by response coefficients.

`response_vector` (= P * Phi_x) is in mode (m,n) basis. To reconstruct physical
fields from eigenmode solutions in u_store/du_store/xi_s_store, first project to eigenmode amplitudes:
alpha = flux_matrix \\ response_vector

Then sum eigenmode contributions at each radial point (matches Fortran gpeq_sol):
xi_psi[ipsi, :]  = u_store[:, :, 1, ipsi]  * alpha   # Ξ_ψ
xi_psi1[ipsi, :] = du_store[:, :, ipsi] * alpha   # dΞ_ψ/dψ
xi_s[ipsi, :]    = xi_s_store[:, :, ipsi] * alpha    # Ξ_s (toroidal, Glasser 2016 eq. 18)

# Returns

  - `xi_psi_modes`: Radial displacement ξ_ψ(ψ, m) [npsi, mpert]
  - `xi_psi1_modes`: Radial derivative dξ_ψ/dψ(ψ, m) [npsi, mpert]
  - `xi_s_modes`: Toroidal displacement ξ_s(ψ, m) = -A⁻¹(B·dξ_ψ/dψ + C·ξ_ψ) [npsi, mpert]
"""
function sum_eigenmode_contributions(
    response_vector::Vector{ComplexF64},
    flux_matrix::Matrix{ComplexF64},
    solution::SolutionProfiles,
    ffs::ForceFreeStatesResult
)
    mpert = ffs.mpert
    npsi = size(solution.u_store, 4)

    # Convert mode-basis response (Phi_tot) to eigenmode amplitudes alpha
    # flux_matrix[mode, eigenmode], so: flux_matrix * alpha = response_vector
    alpha = flux_matrix \ response_vector   # [mpert]

    xi_psi_modes = zeros(ComplexF64, npsi, mpert)
    xi_psi1_modes = zeros(ComplexF64, npsi, mpert)
    xi_s_modes = zeros(ComplexF64, npsi, mpert)
    # Surfaces are independent; threaded over ψ (run with `julia -t N` or JULIA_NUM_THREADS).
    Threads.@threads :static for ipsi in 1:npsi
        # u_store[:,:,1] = Ξ_ψ (radial displacement). @view avoids copying the mpert×mpert
        # eigenmode-matrix slice on every surface (mul! takes the view directly).
        mul!(view(xi_psi_modes, ipsi, :),
            @view(solution.u_store[:, :, 1, ipsi]),
            alpha)
        # du_store = dΞ_ψ/dψ (radial derivative)
        mul!(view(xi_psi1_modes, ipsi, :),
            @view(solution.du_store[:, :, ipsi]),
            alpha)
        # xi_s_store = Ξ_s = -A⁻¹(B·Ξ'_ψ + C·Ξ_ψ) (toroidal displacement, Glasser 2016 eq. 18)
        mul!(view(xi_s_modes, ipsi, :),
            @view(solution.xi_s_store[:, :, ipsi]),
            alpha)
    end

    return xi_psi_modes, xi_psi1_modes, xi_s_modes
end

"""
    compute_perturbed_field_modes(
        xi_psi_modes, xi_psi1_modes, xi_s_modes, psi_grid, equil, ffs
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
    ffs::ForceFreeStatesResult
)
    npsi, mpert = size(xi_psi_modes)

    b_psi_modes = zeros(ComplexF64, npsi, mpert)
    b_theta_modes = zeros(ComplexF64, npsi, mpert)
    b_zeta_modes = zeros(ComplexF64, npsi, mpert)

    mlow = ffs.mlow
    nn = ffs.nlow
    chi1 = 2π * equil.psio

    Threads.@threads :static for ipsi in 1:npsi
        psi_norm = psi_grid[ipsi]
        q = equil.profiles.q_spline(psi_norm)
        q1 = equil.profiles.q_deriv(psi_norm)

        for ipert in 1:mpert
            m = mlow + ipert - 1
            singfac = m - nn * q

            xsp = xi_psi_modes[ipsi, ipert]
            xsp1 = xi_psi1_modes[ipsi, ipert]
            xss = xi_s_modes[ipsi, ipert]

            # Matches Fortran gpeq_sol [Park 2007 eq. 8-10]
            b_psi_modes[ipsi, ipert] = chi1 * singfac * 2π * im * xsp
            b_theta_modes[ipsi, ipert] = -(chi1 * xsp1 + 2π * im * nn * xss)
            b_zeta_modes[ipsi, ipert] = -(chi1 * (q1 * xsp + q * xsp1) + 2π * im * m * xss)
        end
    end

    return b_psi_modes, b_theta_modes, b_zeta_modes
end

"""
    compute_clebsch_displacements(
        xi_psi_modes, xi_psi1_modes, xi_s_modes,
        psi_grid, equil, ffs, mats, ctrl
    ) -> (clebsch_psi, clebsch_psi1, clebsch_alpha)

Compute Clebsch displacement components for PENTRC output.

Matches Fortran gpeq_sol regularization + gpout_xclebsch output convention:

  - `clebsch_psi` = ξ^ψ (unregularized, same as xi_psi_modes)
  - `clebsch_psi1` = xmp1 = ∂ξ^ψ/∂ψ × singfac²/(singfac² + reg_spot²)
  - `clebsch_alpha` = xms/χ₁ (regularized ξ^α divided by χ₁ per gpout_xclebsch convention)

When reg_spot=0, clebsch_psi1 = xi_psi1 and clebsch_alpha = xi_s/χ₁ (no regularization).

Ideal runs re-solve the regularized xms = -A⁻¹(B·xmp1 + C·xsp) from `mats.ideal`. Kinetic runs
instead scale the stored ξ_s by the same singfac factor and invert nothing — the kinetic A is
non-Hermitian and never re-inverted here.
"""
function compute_clebsch_displacements(
    xi_psi_modes::Matrix{ComplexF64},
    xi_psi1_modes::Matrix{ComplexF64},
    xi_s_modes::Matrix{ComplexF64},
    psi_grid::Vector{Float64},
    equil::Equilibrium.PlasmaEquilibrium,
    ffs::ForceFreeStatesResult,
    mats::MatrixSplines,
    ctrl::PerturbedEquilibriumControl
)
    npsi, mpert = size(xi_psi_modes)
    nn = ffs.nlow
    mlow = ffs.mlow
    chi1 = 2π * equil.psio
    numpert_total = ffs.numpert_total

    clebsch_psi = copy(xi_psi_modes)        # ξ^ψ (unregularized)
    clebsch_psi1 = copy(xi_psi1_modes)        # will be regularized below
    clebsch_alpha = xi_s_modes ./ chi1         # ξ^α/χ₁ (will be regularized below)

    reg_spot = ctrl.reg_spot
    @assert reg_spot >= 0 "reg_spot must be non-negative (got $reg_spot)"

    if reg_spot == 0
        return clebsch_psi, clebsch_psi1, clebsch_alpha
    end

    # Kinetic runs regularize the stored ξ_s directly
    if mats.kinetic !== nothing
        hint = Ref(1)
        for ipsi in 1:npsi
            q = equil.profiles.q_spline(psi_grid[ipsi]; hint=hint)
            for ipert in 1:mpert
                singfac = (mlow + ipert - 1) - nn * q
                reg_factor = singfac^2 / (singfac^2 + reg_spot^2)
                clebsch_psi1[ipsi, ipert] = xi_psi1_modes[ipsi, ipert] * reg_factor
                clebsch_alpha[ipsi, ipert] = xi_s_modes[ipsi, ipert] * reg_factor / chi1
            end
        end
        return clebsch_psi, clebsch_psi1, clebsch_alpha
    end

    # Per-thread workspaces: matrix ops and spline hints are not safe to share across threads.
    # Size by maxthreadid() and index by threadid() under :static scheduling (GPEC convention).
    nt = Threads.maxthreadid()
    amat_bufs = [Matrix{ComplexF64}(undef, numpert_total, numpert_total) for _ in 1:nt]
    bmat_bufs = [Matrix{ComplexF64}(undef, numpert_total, numpert_total) for _ in 1:nt]
    cmat_bufs = [Matrix{ComplexF64}(undef, numpert_total, numpert_total) for _ in 1:nt]
    xmp1_vecs = [Vector{ComplexF64}(undef, mpert) for _ in 1:nt]
    xms_vecs = [Vector{ComplexF64}(undef, mpert) for _ in 1:nt]
    hints = [Ref(1) for _ in 1:nt]

    Threads.@threads :static for ipsi in 1:npsi
        tid = Threads.threadid()
        amat = amat_bufs[tid]
        bmat = bmat_bufs[tid]
        cmat_buf = cmat_bufs[tid]
        xmp1_vec = xmp1_vecs[tid]
        xms_vec = xms_vecs[tid]
        hint = hints[tid]

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

        # Compute regularized xms = -A⁻¹(B·xmp1 + C·xsp)
        mats.ideal.A_spline(view(amat, :), psi_norm; hint=hint)
        mats.ideal.B_spline(view(bmat, :), psi_norm; hint=hint)
        mats.ideal.C_spline(view(cmat_buf, :), psi_norm; hint=hint)

        # xms = -(A\B)*xmp1 - (A\C)*xsp
        xsp_vec = view(xi_psi_modes, ipsi, :)
        mul!(xms_vec, bmat, xmp1_vec)                     # xms = B*xmp1
        mul!(xms_vec, cmat_buf, xsp_vec, 1.0+0.0im, 1.0+0.0im)  # xms += C*xsp
        # ideal A only; the kinetic A is non-Hermitian and is treated separately above
        amat_fact = cholesky!(Hermitian(amat, :L))
        ldiv!(amat_fact, xms_vec)                          # xms = A\(B*xmp1 + C*xsp)
        xms_vec .*= -1                                     # xms = -A\(B*xmp1 + C*xsp)

        clebsch_alpha[ipsi, :] .= xms_vec ./ chi1         # ξ^α/χ₁
    end

    return clebsch_psi, clebsch_psi1, clebsch_alpha
end

"""
    compute_modified_field_modes(
        xi_psi_modes, clebsch_psi1, clebsch_alpha, psi_grid, equil, ffs
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
    ffs::ForceFreeStatesResult
)
    npsi, mpert = size(xi_psi_modes)
    mlow = ffs.mlow
    nn = ffs.nlow
    chi1 = 2π * equil.psio

    b_theta_reg = zeros(ComplexF64, npsi, mpert)
    b_zeta_reg = zeros(ComplexF64, npsi, mpert)

    Threads.@threads :static for ipsi in 1:npsi
        psi_norm = psi_grid[ipsi]
        q = equil.profiles.q_spline(psi_norm)
        q1 = equil.profiles.q_deriv(psi_norm)

        for ipert in 1:mpert
            m = mlow + ipert - 1
            xsp = xi_psi_modes[ipsi, ipert]
            xmp1 = clebsch_psi1[ipsi, ipert]
            xms = clebsch_alpha[ipsi, ipert] * chi1  # undo χ₁ division

            b_theta_reg[ipsi, ipert] = -(chi1 * xmp1 + 2π * im * nn * xms)
            b_zeta_reg[ipsi, ipert] = -(chi1 * (q1 * xsp + q * xmp1) + 2π * im * m * xms)
        end
    end

    return b_theta_reg, b_zeta_reg
end

"""
    compute_contra_displacements(
        xi_psi_modes, clebsch_psi1, clebsch_alpha,
        psi_grid, equil, ffs, metric, ctrl
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
    ffs::ForceFreeStatesResult,
    metric::MetricData,
    ctrl::PerturbedEquilibriumControl
)
    npsi, mpert = size(xi_psi_modes)
    mlow = ffs.mlow
    nn = ffs.nlow
    chi1 = 2π * equil.psio
    reg_spot = ctrl.reg_spot
    fc = metric.fourier_coeffs

    xwp_modes = zeros(ComplexF64, npsi, mpert)
    xwt_modes = zeros(ComplexF64, npsi, mpert)
    xwz_modes = zeros(ComplexF64, npsi, mpert)

    # Per-thread Fourier coefficient vectors and spline hints (not safe to share across threads).
    vlen = 2 * mpert - 1
    nt = Threads.maxthreadid()
    jmat_bufs = [Vector{ComplexF64}(undef, vlen) for _ in 1:nt]
    jmat1_bufs = [Vector{ComplexF64}(undef, vlen) for _ in 1:nt]
    jmat_hints = [Ref(1) for _ in 1:nt]
    jmat1_hints = [Ref(1) for _ in 1:nt]

    # Build cubic spline interpolants for Jacobian Fourier coefficients (quantities 7, 8).
    # Matches Fortran cspline_eval(metric%cs, psi, 0) which interpolates smoothly.
    jmat_interp = _build_metric_interp(fc, 7)
    jmat1_interp = _build_metric_interp(fc, 8)

    Threads.@threads :static for ipsi in 1:npsi
        tid = Threads.threadid()
        jmat = jmat_bufs[tid]
        jmat1 = jmat1_bufs[tid]
        jmat_hint = jmat_hints[tid]
        jmat1_hint = jmat1_hints[tid]

        psi_norm = psi_grid[ipsi]
        q = equil.profiles.q_spline(psi_norm)

        # Interpolate Jacobian Fourier coefficients at this psi
        jmat_vals = jmat_interp(psi_norm; hint=jmat_hint)
        jmat1_vals = jmat1_interp(psi_norm; hint=jmat1_hint)
        for m in 0:(mpert-1)
            jmat[mpert-m] = jmat_vals[m+1]
            jmat1[mpert-m] = jmat1_vals[m+1]
        end
        for k in 1:(mpert-1)
            jmat[k+mpert] = conj(jmat[mpert-k])
            jmat1[k+mpert] = conj(jmat1[mpert-k])
        end

        # Matches Fortran gpeq_contra: uses regularized xmp1/xms for theta/zeta components
        for ipert in 1:mpert
            m = mlow + ipert - 1
            singfac = m - nn * q
            twopi_i_singfac = 2π * im * singfac

            xwp_acc = zero(ComplexF64)
            xwt_acc = zero(ComplexF64)
            xwz_acc = zero(ComplexF64)

            for dm in (1-ipert):(mpert-ipert)
                jpert = ipert + dm
                dmidx = dm + mpert

                xsp_j = xi_psi_modes[ipsi, jpert]
                xmp1_j = clebsch_psi1[ipsi, jpert]
                xms_j = clebsch_alpha[ipsi, jpert] * chi1  # undo χ₁ division

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
        Threads.@threads :static for ipsi in 1:npsi
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
        psi_grid, ffs, metric
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
    ffs::ForceFreeStatesResult,
    metric::MetricData
)
    npsi, mpert = size(xwp_modes)
    mlow = ffs.mlow
    fc = metric.fourier_coeffs

    xvp_modes = zeros(ComplexF64, npsi, mpert)
    xvt_modes = zeros(ComplexF64, npsi, mpert)
    xvz_modes = zeros(ComplexF64, npsi, mpert)
    bvp_modes = zeros(ComplexF64, npsi, mpert)
    bvt_modes = zeros(ComplexF64, npsi, mpert)
    bvz_modes = zeros(ComplexF64, npsi, mpert)

    # Per-thread metric Fourier coefficient vectors and spline hints (not safe to share).
    vlen = 2 * mpert - 1
    nt = Threads.maxthreadid()
    g11_bufs = [Vector{ComplexF64}(undef, vlen) for _ in 1:nt]
    g22_bufs = [Vector{ComplexF64}(undef, vlen) for _ in 1:nt]
    g33_bufs = [Vector{ComplexF64}(undef, vlen) for _ in 1:nt]
    g23_bufs = [Vector{ComplexF64}(undef, vlen) for _ in 1:nt]
    g31_bufs = [Vector{ComplexF64}(undef, vlen) for _ in 1:nt]
    g12_bufs = [Vector{ComplexF64}(undef, vlen) for _ in 1:nt]
    g_hints_bufs = [[Ref(1) for _ in 1:6] for _ in 1:nt]

    # Build cubic spline interpolants for metric tensor Fourier coefficients (quantities 1-6).
    # Matches Fortran cspline_eval(metric%cs, psi, 0) which interpolates smoothly.
    g_interps = [_build_metric_interp(fc, qty) for qty in 1:6]

    Threads.@threads :static for ipsi in 1:npsi
        tid = Threads.threadid()
        g11 = g11_bufs[tid]
        g22 = g22_bufs[tid]
        g33 = g33_bufs[tid]
        g23 = g23_bufs[tid]
        g31 = g31_bufs[tid]
        g12 = g12_bufs[tid]
        g_hints = g_hints_bufs[tid]

        psi_norm = psi_grid[ipsi]

        # Interpolate metric tensor Fourier coefficients at this psi
        g_vals = [g_interps[qty](psi_norm; hint=g_hints[qty]) for qty in 1:6]
        for m in 0:(mpert-1)
            idx = mpert - m
            g11[idx] = g_vals[1][m+1]
            g22[idx] = g_vals[2][m+1]
            g33[idx] = g_vals[3][m+1]
            g23[idx] = g_vals[4][m+1]
            g31[idx] = g_vals[5][m+1]
            g12[idx] = g_vals[6][m+1]
        end
        for k in 1:(mpert-1)
            g11[k+mpert] = conj(g11[mpert-k])
            g22[k+mpert] = conj(g22[mpert-k])
            g33[k+mpert] = conj(g33[mpert-k])
            g23[k+mpert] = conj(g23[mpert-k])
            g31[k+mpert] = conj(g31[mpert-k])
            g12[k+mpert] = conj(g12[mpert-k])
        end

        # Tensor contraction with mode coupling (matches Fortran gpeq_cova)
        for ipert in 1:mpert
            xvp_acc = zero(ComplexF64)
            xvt_acc = zero(ComplexF64)
            xvz_acc = zero(ComplexF64)
            bvp_acc = zero(ComplexF64)
            bvt_acc = zero(ComplexF64)
            bvz_acc = zero(ComplexF64)

            for dm in (1-ipert):(mpert-ipert)
                jpert = ipert + dm
                dmidx = dm + mpert

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
    _build_metric_interp(fc, qty) -> CubicSeriesInterpolant

Build a cubic spline interpolant for all Fourier modes of a metric quantity as
a function of ψ. Returns a callable that evaluates all mpert complex
coefficients at arbitrary ψ: `interp(psi) -> Vector{ComplexF64}`.

Matches Fortran behaviour where `cspline_eval(metric%cs, psi, 0)` interpolates
the coefficients smoothly, rather than snapping to the nearest grid point.
"""
function _build_metric_interp(fc::Utilities.FourierCoefficients, qty::Int)
    npsi = length(fc.xs)
    nmodes = fc.mmax + 1
    coeffs = Matrix{ComplexF64}(undef, npsi, nmodes)
    for ipsi in 1:npsi
        for m in 0:fc.mmax
            coeffs[ipsi, m+1] = Utilities.get_complex_coeff(fc, ipsi, m, qty)
        end
    end
    return cubic_interp(collect(fc.xs), Series(coeffs); bc=CubicFit(), extrap=ExtendExtrap())
end

"""
    compute_b_n_xi_n_modes(
        xwp_modes, b_psi_modes, solution, equil, ffs
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
    solution::SolutionProfiles,
    equil::Equilibrium.PlasmaEquilibrium,
    ffs::ForceFreeStatesResult
)
    npsi, mpert = size(b_psi_modes)
    mlow = ffs.mlow
    mthsurf = length(equil.rzphi_ys) - 1
    ro = equil.ro
    twopi = 2π

    b_n_modes = zeros(ComplexF64, npsi, mpert)
    xi_n_modes = zeros(ComplexF64, npsi, mpert)

    # Pre-compute mode indices and DFT phase table
    m_vals = [mlow + ipert - 1 for ipert in 1:mpert]
    thetas = [(k - 1) / mthsurf for k in 1:mthsurf]   # [0, 1/mthsurf, ..., (mthsurf-1)/mthsurf]
    # exp(+2πi*m*θ) for IDFT; exp(-2πi*m*θ) for forward DFT
    phase_fwd = [exp(twopi * im * m_vals[ipert] * thetas[k]) for k in 1:mthsurf, ipert in 1:mpert]
    phase_back = [exp(-twopi * im * m_vals[ipert] * thetas[k]) for k in 1:mthsurf, ipert in 1:mpert]

    Threads.@threads :static for ipsi in 1:npsi
        psi = solution.psi_store[ipsi]
        hint2d_psi = (Ref(1), Ref(1))

        # IDFT: mode space → theta space
        bwp_fun = phase_fwd * b_psi_modes[ipsi, :]   # [mthsurf] — b^ψ (not J-weighted)
        xwp_fun = phase_fwd * xwp_modes[ipsi, :]     # [mthsurf] — J·ξ^ψ (J-weighted from gpeq_contra)
        delpsis = zeros(Float64, mthsurf)
        jacs = zeros(Float64, mthsurf)
        for k in 1:mthsurf
            theta = thetas[k]
            r2 = equil.rzphi_rsquared((psi, theta); hint=hint2d_psi)
            deta = equil.rzphi_offset((psi, theta); hint=hint2d_psi)
            jac = equil.rzphi_jac((psi, theta); hint=hint2d_psi)
            r2_y = equil.rzphi_rsquared((psi, theta); deriv=DerivOp(0, 1), hint=hint2d_psi)
            deta_y = equil.rzphi_offset((psi, theta); deriv=DerivOp(0, 1), hint=hint2d_psi)
            rfac = sqrt(abs(r2))
            eta = twopi * (theta + deta)
            r = ro + rfac * cos(eta)
            w11 = (1.0 + deta_y) * twopi^2 * rfac * r / jac
            w12 = -r2_y * π * r / (rfac * jac)
            delpsi = sqrt(w11^2 + w12^2)
            delpsis[k] = delpsi
            jacs[k] = jac
        end

        # Divide by J·|∇ψ| → physical normal components in theta space (matches Fortran gpeq_normal)
        jd = jacs .* delpsis
        bno_fun = bwp_fun ./ jd
        xno_fun = xwp_fun ./ jd

        # Forward DFT: theta space → mode space (1/mthsurf normalization matches Fortran iscdftf).
        # Must use transpose (not adjoint) so the phase is exp(-2πi·m·θ), not exp(+2πi·m·θ).
        b_n_modes[ipsi, :] = (transpose(phase_back) * bno_fun) ./ mthsurf
        xi_n_modes[ipsi, :] = (transpose(phase_back) * xno_fun) ./ mthsurf
    end

    return b_n_modes, xi_n_modes
end

"""
    _build_rzphi_geometry(equil, psi_grid, mtheta) -> NamedTuple of [mtheta, npsi] arrays

Build the (ψ,θ) flux→cylindrical transformation geometry on the DFT θ grid: the
transformation matrix entries t11/t12/t21/t22/t33 and the Jacobian J_theta, as
`[mtheta, npsi]` arrays (one column per flux surface).

These depend only on equilibrium geometry, not on the perturbed field, so they are computed
once and reused for both the ξ and b reconstructions (Fortran gpeq.f:458-475).

The terms are smooth in ψ, so rather than evaluating the (expensive, 2-D) `rzphi` bicubic at
every fine u_store ψ (npsi ≈ 1158), the *smooth* rzphi primitives (r², η-offset, J and their
θ-derivatives) are sampled at the coarse equilibrium ψ-knots (`equil.rzphi_xs`, ≈ mpsi+1) and
cubic-resampled onto the fine grid — replacing ~6×-redundant 2-D bicubic evaluations with
cheap 1-D cubic evaluations. θ stays at the DFT resolution.

The ψ-derivatives (dr²/dψ, dη/dψ) are obtained by differentiating the ψ-resample spline, NOT
by resampling the bicubic's ψ-derivatives directly: the latter are only C¹ in ψ and a cubic
fit through their knot samples rings badly near the axis/edge. The resampled function values
reproduce the bicubic exactly on its native knots, so the spline derivative recovers the true
ψ-derivative to machine precision. The trig / t-matrix assembly is done at the fine grid.
"""
function _build_rzphi_geometry(
    equil::Equilibrium.PlasmaEquilibrium,
    psi_grid::Vector{Float64},
    mtheta::Int
)
    npsi = length(psi_grid)
    R0 = equil.ro

    knot_psi = collect(equil.rzphi_xs)   # coarse equilibrium ψ-knots
    nknot = length(knot_psi)

    # Sample the smooth rzphi primitives at the coarse ψ-knots × DFT θ grid, packed into a
    # single [nknot, 5·mtheta] matrix so one cubic Series interpolant (over ψ) carries every
    # (primitive, θ) channel. Block o·mtheta .+ (1:mtheta) holds primitive o at all θ.
    # Channels: r2(0), deta(1), jac(2), dr2_dtheta(3), doff_dtheta(4). The ψ-derivatives of
    # r2/deta come from differentiating this spline (see below).
    nch = 5
    knot_vals = Matrix{Float64}(undef, nknot, nch * mtheta)

    hints = [(Ref(1), Ref(1)) for _ in 1:Threads.maxthreadid()]

    Threads.@threads :static for ik in 1:nknot
        hint = hints[Threads.threadid()]
        psi = knot_psi[ik]

        for itheta in 1:mtheta
            theta = (itheta - 1) / mtheta  # SFL theta ∈ [0, 1)
            pt = (psi, theta)

            knot_vals[ik, 0*mtheta+itheta] = equil.rzphi_rsquared(pt; hint=hint)
            knot_vals[ik, 1*mtheta+itheta] = equil.rzphi_offset(pt; hint=hint)
            knot_vals[ik, 2*mtheta+itheta] = equil.rzphi_jac(pt; hint=hint)
            knot_vals[ik, 3*mtheta+itheta] = equil.rzphi_rsquared(pt; deriv=DerivOp(0, 1), hint=hint)
            knot_vals[ik, 4*mtheta+itheta] = equil.rzphi_offset(pt; deriv=DerivOp(0, 1), hint=hint)
        end
    end

    # Cubic-in-ψ interpolant of every (primitive, θ) channel; matches the metric-coeff
    # resampling convention used in `_build_metric_interp`.
    geom_interp = cubic_interp(knot_psi, Series(knot_vals); bc=CubicFit(), extrap=ExtendExtrap())

    t11 = Matrix{Float64}(undef, mtheta, npsi)
    t12 = Matrix{Float64}(undef, mtheta, npsi)
    t21 = Matrix{Float64}(undef, mtheta, npsi)
    t22 = Matrix{Float64}(undef, mtheta, npsi)
    t33 = Matrix{Float64}(undef, mtheta, npsi)
    J_theta = Matrix{Float64}(undef, mtheta, npsi)

    hints1d = [Ref(1) for _ in 1:Threads.maxthreadid()]

    Threads.@threads :static for ipsi in 1:npsi
        h = hints1d[Threads.threadid()]
        psi = psi_grid[ipsi]
        raw = geom_interp(psi; hint=h)                    # values: r2, deta, jac, dr2_dθ, doff_dθ
        draw = geom_interp(psi; deriv=DerivOp(1), hint=h) # ψ-derivatives of those channels

        # Assemble the transform matrices at the fine grid (Fortran gpeq.f:458-475). The trig
        # is evaluated here, not resampled, because deta(ψ) makes c/s oscillate in ψ.
        for itheta in 1:mtheta
            theta = (itheta - 1) / mtheta
            r2 = raw[0*mtheta+itheta]
            deta = raw[1*mtheta+itheta]
            jac = raw[2*mtheta+itheta]
            dr2_dtheta = raw[3*mtheta+itheta]
            doff_dtheta = raw[4*mtheta+itheta]
            dr2_dpsi = draw[0*mtheta+itheta]   # d(r2)/dψ
            doff_dpsi = draw[1*mtheta+itheta]  # d(deta)/dψ

            rfac = sqrt(max(0.0, r2))
            eta = 2π * (theta + deta)
            s, c = sincos(eta)
            R_here = R0 + rfac * c

            if rfac > 1e-30
                v11 = dr2_dpsi / (2 * rfac)
                v12 = doff_dpsi * 2π * rfac
                v21 = dr2_dtheta / (2 * rfac)
                v22 = (1 + doff_dtheta) * 2π * rfac
            else
                v11 = 0.0
                v12 = 0.0
                v21 = 0.0
                v22 = 0.0
            end

            t11[itheta, ipsi] = c * v11 - s * v12
            t12[itheta, ipsi] = c * v21 - s * v22
            t21[itheta, ipsi] = s * v11 + c * v12
            t22[itheta, ipsi] = s * v21 + c * v22
            t33[itheta, ipsi] = -1.0 / (2π * R_here)
            J_theta[itheta, ipsi] = jac
        end
    end

    return (; t11, t12, t21, t22, t33, J_theta)
end

"""
    _apply_rzphi_transform(geom, ft, mtheta, psi_input, theta_input, cova_zeta_input)
        -> (R_modes, Z_modes, phi_modes)

Apply a prebuilt `_build_rzphi_geometry` cache to one perturbed field, converting it from
flux coordinates (ψ,θ,ζ) to cylindrical (R,Z,φ) in mode space (Fortran `gpeq_rzphi`):

    R(θ) = (t11·ξ^ψ + t12·ξ^θ) / J,  Z(θ) = (t21·ξ^ψ + t22·ξ^θ) / J,  φ(θ) = t33·ξ_ζ

Returns mode-space arrays (npsi × mpert) in SFL coordinates. The pointwise product with
geometry generates harmonics beyond mpert; users needing fuller resolution should increase
the mode count or use `Analysis.PerturbedEquilibriumModes.modes_to_theta` post-hoc.

The caller passes different inputs for ξ vs b (see `reconstruct_physical_fields`).
"""
function _apply_rzphi_transform(
    geom,
    ft::Utilities.FourierTransforms.FourierTransform,
    mtheta::Int,
    psi_input::Matrix{ComplexF64},
    theta_input::Matrix{ComplexF64},
    cova_zeta_input::Matrix{ComplexF64}
)
    npsi, mpert = size(psi_input)

    R_modes = zeros(ComplexF64, npsi, mpert)
    Z_modes = zeros(ComplexF64, npsi, mpert)
    phi_modes = zeros(ComplexF64, npsi, mpert)

    # Per-thread scratch (the immutable `ft` functor and `geom` are shared read-only): θ-space
    # transform inputs/outputs (length mtheta) and mode-space forward-DFT outputs (length mpert),
    # so the DFTs run in place with no per-surface allocation.
    bufs = [
        (R=zeros(ComplexF64, mtheta), Z=zeros(ComplexF64, mtheta), P=zeros(ComplexF64, mtheta),
            psi=zeros(ComplexF64, mtheta), th=zeros(ComplexF64, mtheta), ze=zeros(ComplexF64, mtheta),
            Ro=zeros(ComplexF64, mpert), Zo=zeros(ComplexF64, mpert), Po=zeros(ComplexF64, mpert))
        for _ in 1:Threads.maxthreadid()
    ]

    Threads.@threads :static for ipsi in 1:npsi
        buf = bufs[Threads.threadid()]
        R_fun = buf.R
        Z_fun = buf.Z
        phi_fun = buf.P

        # Inverse DFT: modes → theta-space (in place)
        psi_fun = buf.psi
        theta_fn = buf.th
        zeta_fn = buf.ze
        Utilities.FourierTransforms.inverse_transform!(psi_fun, ft, view(psi_input, ipsi, :))
        Utilities.FourierTransforms.inverse_transform!(theta_fn, ft, view(theta_input, ipsi, :))
        Utilities.FourierTransforms.inverse_transform!(zeta_fn, ft, view(cova_zeta_input, ipsi, :))

        # Pointwise transformation (Fortran gpeq_rzphi, gpeq.f:484-489)
        for itheta in 1:mtheta
            J = geom.J_theta[itheta, ipsi]
            xwp = psi_fun[itheta]
            xwt = theta_fn[itheta]
            xvz = zeta_fn[itheta]

            if abs(J) > 1e-30
                R_fun[itheta] = (geom.t11[itheta, ipsi] * xwp + geom.t12[itheta, ipsi] * xwt) / J
                Z_fun[itheta] = (geom.t21[itheta, ipsi] * xwp + geom.t22[itheta, ipsi] * xwt) / J
            else
                R_fun[itheta] = zero(ComplexF64)
                Z_fun[itheta] = zero(ComplexF64)
            end
            phi_fun[itheta] = geom.t33[itheta, ipsi] * xvz
        end

        # Forward DFT: theta-space → modes (in place)
        Utilities.FourierTransforms.transform!(buf.Ro, ft, R_fun)
        Utilities.FourierTransforms.transform!(buf.Zo, ft, Z_fun)
        Utilities.FourierTransforms.transform!(buf.Po, ft, phi_fun)
        R_modes[ipsi, :] .= buf.Ro
        Z_modes[ipsi, :] .= buf.Zo
        phi_modes[ipsi, :] .= buf.Po
    end

    return R_modes, Z_modes, phi_modes
end
