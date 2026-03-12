"""
    free_run!(odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal) -> VacuumData

Compute the free boundary energies using the Julia port of the VACUUM code. Performs the same function as `free_run`
in the Fortran code, except now all data is passed in memory instead of via files. This
modifies `odet` in place to normalize the eigenfunctions stored in `u_store` and `ud_store`,
and returns a `VacuumData` struct containing the data needed for perturbed equilibrium calculations
and data dumping.
"""
@with_pool pool function free_run!(odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal)

    # Initializations and allocations
    (; mpert, mlow, mhigh, mband, numpert_total, psilim, qlim, npert, nlow, nhigh, wall_settings) = intr
    vac_data = VacuumData(ctrl.mthvac * ctrl.nzvac, intr.numpert_total, ctrl.mthvac)
    etemp = zeros!(pool, ComplexF64, numpert_total)
    wp = zeros!(pool, ComplexF64, numpert_total, numpert_total)
    wpt = zeros!(pool, ComplexF64, numpert_total, numpert_total)
    wvt = zeros!(pool, ComplexF64, numpert_total, numpert_total)
    tmp_mat = zeros!(pool, ComplexF64, numpert_total, numpert_total)

    # Evaluate dV/dpsi at the plasma edge. For diverted plasmas psilim can exceed psihigh,
    # so route to the edge inverse spline when available and psilim is beyond the core grid.
    profiles = equil.profiles
    dV_dpsi = intr.psilim > profiles.xs[end] ?
              profiles.dVdpsi_spline_inv(intr.psilim) :
              profiles.dVdpsi_spline(psilim)

    # Compute plasma response matrix W = U₂ * U₁⁻¹
    if ctrl.ode_flag
        @views wp .= (odet.u[:, :, 2] / odet.u[:, :, 1]) ./ equil.psio^2
    end

    # Compute vacuum response matrix in-place (handles 2D single-n, 2D multi-n block-diagonal, and 3D)
    vac_inputs = Vacuum.VacuumInput(equil, psilim, ctrl.mthvac, ctrl.nzvac, mpert, mlow, npert, nlow; force_wv_symmetry=ctrl.force_wv_symmetry)
    Vacuum.compute_vacuum_response!(vac_data, vac_inputs, wall_settings)

    # Scale by (m - n*q)(m' - n'*q) [Chance Phys. Plasmas 1997 2161 eq. 126]
    singfac = vec((mlow:mhigh) .- qlim .* (nlow:nhigh)')
    @inbounds for ipert in 1:numpert_total
        @views vac_data.wv[ipert, :] .*= singfac[ipert]
        @views vac_data.wv[:, ipert] .*= singfac[ipert]
    end

    # Compute complex energy eigenvalues and vectors
    vac_data.wt .= wp .+ vac_data.wv
    vac_data.wt0 .= vac_data.wt
    Ev = eigen(vac_data.wt)
    vac_data.et .= Ev.values
    eindex = sortperm(real.(vac_data.et); rev=true)

    etemp .= vac_data.et
    # Rearrange wt columns for descending real eigenvalues
    for ipert in 1:numpert_total
        vac_data.wt[:, ipert] .= Ev.vectors[:, eindex[numpert_total+1-ipert]]
        vac_data.et[ipert] = etemp[eindex[numpert_total+1-ipert]]
    end

    # Normalize eigenfunction and energy using J(ψ) evaluated at the plasma boundary psilim.
    jmat_at_psilim = ffit.jmat_spline(psilim)
    for isol in 1:numpert_total
        norm = 0.0 + 0.0im
        for ipert_n in 1:npert, ipert_m in 1:mpert, jpert_m in 1:mpert
            ipert = (ipert_n - 1) * mpert + ipert_m
            jpert = (ipert_n - 1) * mpert + jpert_m
            norm += jmat_at_psilim[jpert_m-ipert_m+mband+1] * vac_data.wt[ipert, isol] * conj(vac_data.wt[jpert, isol])
        end
        norm /= dV_dpsi
        vac_data.wt[:, isol] ./= sqrt(norm)
        vac_data.et[isol] /= norm
    end

    # Normalize phase
    imax = 0
    for isol in 1:numpert_total
        imax = argmax(abs.(vac_data.wt[:, isol]))
        phase = abs(vac_data.wt[imax, isol]) / vac_data.wt[imax, isol]
        vac_data.wt[:, isol] .*= phase
    end

    # Compute plasma and vacuum contributions.
    # wpt = wt' * wp * wt  ; wvt = wt' * wv * wt
    mul!(tmp_mat, wp, vac_data.wt)
    mul!(wpt, adjoint(vac_data.wt), tmp_mat)
    mul!(tmp_mat, vac_data.wv, vac_data.wt)
    mul!(wvt, adjoint(vac_data.wt), tmp_mat)
    for ipert in 1:numpert_total
        vac_data.ep[ipert] = wpt[ipert, ipert]
        vac_data.ev[ipert] = wvt[ipert, ipert]
    end

    # Normalize eigenvectors based on scaled wt
    coeffs = odet.u[:, :, 1, end] \ (vac_data.wt .* (2π * equil.psio * 1e-3))
    @views for istep in 1:odet.step
        mul!(tmp_mat, odet.u_store[:, :, 1, istep], coeffs)
        odet.u_store[:, :, 1, istep] .= tmp_mat
        mul!(tmp_mat, odet.u_store[:, :, 2, istep], coeffs)
        odet.u_store[:, :, 2, istep] .= tmp_mat
        mul!(tmp_mat, odet.ud_store[:, :, 1, istep], coeffs)
        odet.ud_store[:, :, 1, istep] .= tmp_mat
        mul!(tmp_mat, odet.ud_store[:, :, 2, istep], coeffs)
        odet.ud_store[:, :, 2, istep] .= tmp_mat
    end

    # Write energies to screen
    if ctrl.verbose
        @info "Least Stable Eigenmode Energies:\n" *
              "  Plasma = $((@sprintf "%+.3e %+.3ei" real(vac_data.ep[1]) imag(vac_data.ep[1])))\n" *
              "  Vacuum = $((@sprintf "%+.3e %+.3ei" real(vac_data.ev[1]) imag(vac_data.ev[1])))\n" *
              "  Total  = $((@sprintf "%+.3e %+.3ei" real(vac_data.et[1]) imag(vac_data.et[1])))"
    end

    return vac_data
end

"""
    free_compute_wv_spline(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal)

Compute a spline of vacuum response matrices over [psiedge, psilim], evenly spaced in q.
Used for fast evaluation of wt during `findmax_dW_edge!`. Performs the same function as
`free_wvmats` in the Fortran code. Defaults to 9 spline points per rational window.

Grid design: points are spaced evenly in q from qedge+ε to qlim+ε (i/npsi offset reproduces
the develop-branch qi logic). This concentrates nodes near rational surfaces where the vacuum
response varies most rapidly, giving accurate wv interpolation across the full scan range.
For diverted plasmas where psilim > psihigh, the iota inverse spline is used to invert q(psi)
above psihigh.

The spline stores RAW vacuum response without singfac scaling. Singfac = (m - n*q) is
applied analytically in free_compute_total at each scan psi, keeping the spline
well-conditioned despite q divergence near the separatrix.
"""
function free_compute_wv_spline(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal, _psi_scan_end::Float64)

    profiles = equil.profiles
    psihigh  = profiles.xs[end]

    # Limit the wv spline to the core grid [psiedge, psihigh] only.
    # For psi > psihigh, the rzphi geometry uses the X-point asymptotic approximation which
    # gives degenerate vacuum responses near the separatrix (plasma boundary shrinks to a
    # point). The ExtendExtrap BC returns the psihigh value for all psi > psihigh, which is
    # the correct physics: the vacuum response at the plasma-vacuum interface psihigh is
    # well-defined and represents the physical plasma boundary.
    qedge = profiles.q_spline_direct(ctrl.psiedge)
    q_at_psihigh = profiles.q_spline_direct(psihigh)

    # Number of grid points: 9 per rational window over [qedge, q_at_psihigh]
    npsi = max(9, ceil(Int, (q_at_psihigh - qedge) * intr.nhigh * 9))
    psi_array = zeros(Float64, npsi + 1)
    wv_array  = zeros(ComplexF64, npsi + 1, intr.numpert_total, intr.numpert_total)

    for i in 1:(npsi + 1)
        # qi runs from qedge to q_at_psihigh (core only)
        qi   = qedge + (q_at_psihigh - qedge) * ((i-1) / npsi)
        psii = ctrl.psiedge + (psihigh - ctrl.psiedge) * ((i - 1) / npsi)  # linear psi initial guess

        # Invert q(psi) = qi using the direct spline (all points in [psiedge, psihigh])
        jpsi = max(1, searchsortedlast(profiles.q_spline_direct.y, qi))
        hint = Ref(min(jpsi, profiles.npts_minus_1))
        psi_array[i] = find_zero(
            (psi -> profiles.q_spline_direct(psi; hint=hint) - qi,
             psi -> profiles.q_deriv(psi; hint=hint)),
            psii, Roots.Newton()
        )

        # Compute vacuum response matrix at this psi (2D single-n, 2D multi-n block-diagonal, or 3D)
        vac_inputs = Vacuum.VacuumInput(equil, psi_array[i], ctrl.mthvac, ctrl.nzvac, intr.mpert, intr.mlow, intr.npert, intr.nlow; force_wv_symmetry=ctrl.force_wv_symmetry)
        wv, _, _, _, _ = Vacuum.compute_vacuum_response(vac_inputs, intr.wall_settings)

        # Apply singular factor scaling: (m - n*q)(m' - n'*q) [Chance Phys. Plasmas 1997 2161 eq. 126]
        singfac = vec((intr.mlow:intr.mhigh) .- qi .* (intr.nlow:intr.nhigh)')
        @inbounds for ipert in 1:intr.numpert_total
            @views wv[ipert, :] .*= singfac[ipert]
            @views wv[:, ipert] .*= singfac[ipert]
        end

        @views wv_array[i, :, :] .= wv
    end

    wv_flat = reshape(wv_array, npsi + 1, intr.numpert_total^2)
    wvmat   = cubic_interp(psi_array, wv_flat; bc=CubicFit(), extrap=ExtendExtrap(), search=LinearBinary())

    return wvmat
end

"""
    free_compute_total(equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal, odet::OdeState) -> ComplexF64

Compute total complex energy eigenvalue (total1). This is a trimmed down version of `free_run`
that only computes the total energy eigenvalue for the mode unstable mode, used in `findmax_dW_edge!`
which calls this function at each step in the psiedge -> psilim region of integration. This performs
the same function as `free_test` in the Fortran code, except we have moved the creation of the
wv matrix spline to `free_compute_wv_spline` and pass it in `odet.wvmat` (a complex-valued spline).
"""
function free_compute_total(equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal, odet::OdeState)

    # Allocations
    wp = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)
    tot_eigvals = zeros(ComplexF64, intr.numpert_total)
    wt = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)

    profiles = equil.profiles

    # Compute plasma response matrix W = U₂·U₁⁻¹ / psio².
    # Direct matrix division without column normalization, matching the develop-branch approach.
    # Column normalization was found to over-correct well-conditioned steps near core rational
    # surfaces (e.g. just after a q=3 fixup), giving them artificially large et values that
    # dominate the peak search. Without column normalization, those steps give garbage or
    # negative eigenvalues (naturally excluded from findmax), and only physically meaningful
    # steps (well away from fixup boundaries near the edge) give large positive et.
    wp = (@view(odet.u[:, :, 2]) / @view(odet.u[:, :, 1])) ./ equil.psio^2

    # Retrieve raw vacuum matrix from the spline (singfac NOT pre-applied; see free_compute_wv_spline).
    # Apply singfac = (m - n*q(psifac)) analytically: this is the physically correct choice for
    # the scan, since at each scan step we are evaluating stability as if the plasma boundary
    # were at psifac, where q(psifac) is the actual safety factor at that boundary.
    # For psi > psihigh, use the iota inverse spline (if available) to get q.
    odet.wvmat(vec(odet._wv_out), odet.psifac; hint=odet.wv_hint)
    wv = copy(odet._wv_out)
    q_at_psifac = (odet.psifac > profiles.xs[end] && !isnothing(profiles.q_spline_iota_inverse)) ?
                  profiles.q_spline_iota_inverse(odet.psifac) :
                  profiles.q_spline_direct(odet.psifac)
    for ipert_n in 1:intr.npert
        n = ipert_n - 1 + intr.nlow
        singfac = collect(intr.mlow:intr.mhigh) .- (n * q_at_psifac)
        block = ((ipert_n-1)*intr.mpert+1):(ipert_n*intr.mpert)
        # wv[i,j] = wv_raw[i,j] * singfac[i] * singfac[j] (outer-product row×column scaling)
        @views wv[block, block] .= singfac .* wv[block, block] .* singfac'
    end

    # Compute total energy matrix and eigen-decomposition.
    # wt = wp + wv should be Hermitian by construction, but ODE roundoff accumulates over many
    # integration steps. Enforcing the Hermitian structure (as in compute_smallest_eigenvalue)
    # guarantees real eigenvalues and orthonormal eigenvectors.
    wt .= wp .+ wv
    hermitianpart!(wt)
    Ev = eigen(Hermitian(wt))

    # Sort eigenvalues and reorder columns of wt (Ev.values are real for Hermitian)
    eindex = sortperm(Ev.values; rev=true)
    for ipert in 1:intr.numpert_total
        wt[:, ipert] .= Ev.vectors[:, eindex[intr.numpert_total+1-ipert]]
        tot_eigvals[ipert] = Ev.values[eindex[intr.numpert_total+1-ipert]]
    end

    # Compute plasma and vacuum energy components for the dominant (smallest eigenvalue) eigenvector.
    # wt[:,1] now holds the eigenvector corresponding to tot_eigvals[1].
    v = @view wt[:, 1]
    ep1 = ComplexF64(dot(v, wp * v))
    ev1 = ComplexF64(dot(v, wv * v))

    # Least stable eigenvalue of the vacuum matrix alone (EL-solution-independent diagnostic).
    evonly1 = minimum(real(eigvals(Hermitian((wv + wv') / 2))))

    # Normalize eigenvalue by ξ†J(psifac)ξ / dV_dψ(psifac), where J is evaluated at the local
    # scan psi so the normalization is physically consistent at each point of the edge scan.
    dV_dpsi = (!isnothing(profiles.dVdpsi_spline_inv) && odet.psifac > profiles.xs[end]) ?
              profiles.dVdpsi_spline_inv(odet.psifac) :
              profiles.dVdpsi_spline(odet.psifac)
    jmat_local = ffit.jmat_spline(odet.psifac)
    norm_kin = zero(ComplexF64)
    for ipert_n in 1:intr.npert, ipert_m in 1:intr.mpert, jpert_m in 1:intr.mpert
        ipert = (ipert_n - 1) * intr.mpert + ipert_m
        jpert = (ipert_n - 1) * intr.mpert + jpert_m
        jidx  = jpert_m - ipert_m + intr.mband + 1
        norm_kin += jmat_local[jidx] * v[ipert] * conj(v[jpert])
    end
    norm_kin /= dV_dpsi
    real(norm_kin) > 0 || throw(LinearAlgebra.SingularException(0))
    et_normalized = tot_eigvals[1] / norm_kin

    return (et=et_normalized, ep=ep1, ev=ev1, evonly=evonly1)
end
