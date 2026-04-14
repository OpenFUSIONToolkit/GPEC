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

    # Evaluate dV/dpsi at the plasma edge
    dV_dpsi = equil.profiles.dVdpsi_spline(psilim)

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

    # Preserve wp in vac_data so it can be written to HDF5 as W_plasma
    vac_data.wp .= wp

    # Compute complex energy eigenvalues and vectors
    vac_data.wt .= wp .+ vac_data.wv
    vac_data.wt0 .= vac_data.wt
    Ev = eigen(vac_data.wt)
    vac_data.et .= Ev.values
    eindex = sortperm(real.(vac_data.et); rev=true)

    etemp .= vac_data.et
    # Rearrange wt columns for ascending real eigenvalues (most unstable first)
    for ipert in 1:numpert_total
        vac_data.wt[:, ipert] .= Ev.vectors[:, eindex[numpert_total+1-ipert]]
        vac_data.et[ipert] = etemp[eindex[numpert_total+1-ipert]]
    end

    # Normalize eigenfunction and energy.
    for isol in 1:numpert_total
        norm = 0.0 + 0.0im
        for ipert_n in 1:npert, ipert_m in 1:mpert, jpert_m in 1:mpert
            ipert = (ipert_n - 1) * mpert + ipert_m
            jpert = (ipert_n - 1) * mpert + jpert_m
            norm += ffit.jmat[jpert_m-ipert_m+mband+1] * vac_data.wt[ipert, isol] * conj(vac_data.wt[jpert, isol])
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

    # Eigenspectrum of W_Φ at psilim — Jacobian-invariant energy values; see PowerNorm.jl.
    # Computed directly here (no spline); spline is used by the edge scan.
    mtheta_eq = length(equil.rzphi_ys)
    ft_pn = Utilities.FourierTransforms.FourierTransform(mtheta_eq, mpert, mlow)
    sqrtamat_pn = compute_sqrtamat(equil, psilim, ft_pn)
    jarea_pn = compute_surface_area_local(equil, psilim, mtheta_eq)
    pn_result = compute_power_norm_eigenvalues(vac_data.wt0, wp, vac_data.wv, sqrtamat_pn, jarea_pn, equil, psilim, intr; all_eigenvalues=true)
    vac_data.pn_et .= pn_result.pn_et_all
    vac_data.pn_ep .= pn_result.pn_ep_all
    vac_data.pn_ev .= pn_result.pn_ev_all

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

Compute a spline of vacuum response matrices over the range of psi from 'ctrl.psi_edge' to
`intr.qlim`. This is used for fast evaluation of wt during `ode_record_edge`. Performs the
same function as `free_wvmats` in the Fortran code. Currently defaults to 4 spline points per
q-window minimum.
"""
function free_compute_wv_spline(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal)

    profiles = equil.profiles

    # Number of psi grid points for the spline: 4 per q-window minimum
    # TODO: 4 spline points is arbitrary - is there a better way?
    qedge = profiles.q_spline(ctrl.psiedge)
    npsi = max(4, ceil(Int, (intr.qlim - qedge) * intr.nhigh * 4))
    psi_array = zeros(Float64, npsi + 1)
    wv_array = zeros(ComplexF64, npsi + 1, intr.numpert_total, intr.numpert_total)

    for i in 1:(npsi+1)
        # Space points evenly in q over [qedge, qlim] (i=1 → qedge, i=npsi+1 → qlim)
        qi = qedge + (intr.qlim - qedge) * ((i - 1) / npsi)

        psii = ctrl.psiedge + (intr.psilim - ctrl.psiedge) * ((i - 1) / npsi)
        psi_array[i] = find_zero(
            (psi -> profiles.q_spline(psi) - qi,
                psi -> profiles.q_deriv(psi)),
            psii, Roots.Newton()
        )

        # Compute raw vacuum matrix at the actual scan psi (singfac NOT applied; free_compute_total applies it analytically)
        vac_inputs = Vacuum.VacuumInput(equil, psi_array[i], ctrl.mthvac, ctrl.nzvac, intr.mpert, intr.mlow, intr.npert, intr.nlow; force_wv_symmetry=ctrl.force_wv_symmetry)
        wv, _, _, _, _ = Vacuum.compute_vacuum_response(vac_inputs, intr.wall_settings)
        @views wv_array[i, :, :] .= wv
    end

    # Flatten 3D array to (npsi+1 × numpert_total^2) for series interpolant
    wv_flat = reshape(wv_array, npsi + 1, intr.numpert_total^2)

    # FastInterpolations now natively supports complex values - create complex series interpolant directly
    # Use CubicFit() (default) for native endpoint handling
    wvmat = cubic_interp(psi_array, Series(wv_flat); extrap=ExtendExtrap())

    return wvmat
end

"""
    free_compute_total(equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal, odet::OdeState) -> ComplexF64

Compute total complex energy eigenvalue (total1). This is a trimmed down version of `free_run`
that only computes the total energy eigenvalue for the mode unstable mode, used in `findmax_dW_edge!`
which calls this function at each step in the psiedge -> psilim region of integration. This performs
the same function as `free_test` in the Fortran code, except we have moved the creation of the
wv matrix spline to `free_compute_wv_spline` and pass it in `odet.edge_scan.wvmat` (a complex-valued spline).
"""
@with_pool pool function free_compute_total(equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal, odet::OdeState)

    Npert = intr.numpert_total
    wp = zeros!(pool, ComplexF64, Npert, Npert)
    eigenvalues = zeros!(pool, ComplexF64, Npert)
    wt = zeros!(pool, ComplexF64, Npert, Npert)
    wv = zeros!(pool, ComplexF64, Npert, Npert)
    eindex = zeros!(pool, Int, Npert)
    evals_real = zeros!(pool, Float64, Npert)
    tmp_v = zeros!(pool, ComplexF64, Npert)

    dV_dpsi = equil.profiles.dVdpsi_spline(odet.psifac)

    # Compute plasma response matrix
    @views wp .= (odet.u[:, :, 2] / odet.u[:, :, 1]) ./ equil.psio^2

    # Retrieve raw vacuum matrix from spline and apply singfac analytically at the local q.
    # Singfac is not pre-applied in the spline (see free_compute_wv_spline) to avoid interpolating
    # a zero-crossing function near rational surfaces, which would distort the peaks.
    es = odet.edge_scan
    es.wvmat(vec(wv), odet.psifac; hint=es.wv_hint)
    q_at_psifac = equil.profiles.q_spline(odet.psifac)
    # Scale by (m - n*q)(m' - n'*q) [Chance Phys. Plasmas 1997 2161 eq. 126]
    singfac = vec((intr.mlow:intr.mhigh) .- q_at_psifac .* (intr.nlow:intr.nhigh)')
    @inbounds for ipert in 1:Npert
        @views wv[ipert, :] .*= singfac[ipert]
        @views wv[:, ipert] .*= singfac[ipert]
    end

    # Compute total energy matrix and eigen-decomposition
    wt .= wp .+ wv

    # Save wt before eigen (which overwrites the input) for power-norm computation
    wt_saved = copy(wt)

    Ev = eigen(wt)

    # Sort eigenvalues by descending real part
    @inbounds for i in 1:Npert
        evals_real[i] = real(Ev.values[i])
    end
    sortperm!(eindex, evals_real; rev=true)
    for ipert in 1:Npert
        wt[:, ipert] .= Ev.vectors[:, eindex[Npert+1-ipert]]
        eigenvalues[ipert] = Ev.values[eindex[Npert+1-ipert]]
    end

    # Compute kinetic norm xi'*J(psi)*xi / dV_dpsi for the leading eigenvector.
    # This normalizes total_eigenvalue, plasma_energy, vacuum_energy to be dimensionally consistent with free_run!.
    isol = 1
    v = @view wt[:, isol]
    norm = 0.0 + 0.0im
    for ipert_n in 1:intr.npert, ipert_m in 1:intr.mpert, jpert_m in 1:intr.mpert
        ipert = (ipert_n - 1) * intr.mpert + ipert_m
        jpert = (ipert_n - 1) * intr.mpert + jpert_m
        norm += ffit.jmat[jpert_m-ipert_m+intr.mband+1] * v[ipert] * conj(v[jpert])
    end
    norm /= dV_dpsi
    eigenvalues[isol] /= norm

    # Plasma and vacuum energy components for the leading eigenvector, normalized by the same norm.
    # plasma_energy + vacuum_energy = total_eigenvalue by construction (wt = wp + wv; eigenvalue = v'*wt*v / norm).
    mul!(tmp_v, wp, v)
    plasma_energy = ComplexF64(dot(v, tmp_v)) / norm
    mul!(tmp_v, wv, v)
    vacuum_energy = ComplexF64(dot(v, tmp_v)) / norm

    # Smallest eigenvalue of the vacuum matrix alone, normalized by the same kinetic norm as the other energies
    # so all four outputs are directly comparable. The singfac-scaled wv should be PSD by
    # construction (congruence of PSD wv_raw), but numerical noise can make eigenvalues slightly
    # negative. Clamp to zero to enforce the physical constraint.
    vacuum_eigenvalue = real(max(0.0, minimum(real.(eigvals(Hermitian(wv))))) / norm)

    # Eigenspectrum of W_Φ — Jacobian-invariant energy values; see PowerNorm.jl.
    # Evaluate sqrtamat + jarea from the pre-computed spline.
    mpert = intr.mpert
    sqrtamat_flat = Vector{ComplexF64}(undef, mpert^2 + 1)
    es.sqrtamat_spline(sqrtamat_flat, odet.psifac; hint=es.sqrtamat_hint)
    sqrtamat_local = reshape(@view(sqrtamat_flat[1:mpert^2]), mpert, mpert)
    jarea_local = real(sqrtamat_flat[end])

    pn_result = compute_power_norm_eigenvalues(wt_saved, wp, wv, sqrtamat_local, jarea_local, equil, odet.psifac, intr)

    return (total_eigenvalue=eigenvalues[1], plasma_energy=plasma_energy, vacuum_energy=vacuum_energy, vacuum_eigenvalue=vacuum_eigenvalue,
        pn_total_eigenvalue=pn_result.pn_total_eigenvalue, pn_plasma_energy=pn_result.pn_plasma_energy, pn_vacuum_energy=pn_result.pn_vacuum_energy,
        pn_vacuum_eigenvalue=pn_result.pn_vacuum_eigenvalue)
end
