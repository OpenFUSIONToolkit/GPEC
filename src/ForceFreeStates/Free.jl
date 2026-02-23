"""
    free_run!(odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal) -> VacuumData

Compute the free boundary energies using the Julia port of the VACUUM code. Performs the same function as `free_run`
in the Fortran code, except now all data is passed in memory instead of via files. This
modifies `odet` in place to normalize the eigenfunctions stored in `u_store` and `ud_store`,
and returns a `VacuumData` struct containing the data needed for perturbed equilibrium calculations
and data dumping.
"""
function free_run!(odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal)

    # Initializations and allocations
    (; mpert, mlow, mhigh, mband, numpert_total, psilim, qlim, npert, nlow, nhigh, wall_settings) = intr
    vac_data = VacuumData(ctrl.mthvac * ctrl.nzvac, numpert_total)
    etemp = zeros(ComplexF64, numpert_total)
    wp = zeros(ComplexF64, numpert_total, numpert_total)
    wpt = zeros(ComplexF64, numpert_total, numpert_total)
    wvt = zeros(ComplexF64, numpert_total, numpert_total)

    # Evaluate dV/dpsi at the plasma edge
    dV_dpsi = equil.profiles.dVdpsi_spline(psilim)

    # Compute plasma response matrix W = U₂ * U₁⁻¹
    if ctrl.ode_flag
        @views wp = (odet.u[:, :, 2] / odet.u[:, :, 1]) ./ equil.psio^2
    end

    # Compute vacuum response matrix
    if ctrl.nzvac == 1
        vac_data.wv .= 0.0 # need to zero to avoid multiplication of undef by singfac during scaling
        for idx_n in 1:npert
            # Set VACUUM run parameters and boundary shape
            n = idx_n - 1 + nlow
            vac_inputs = Vacuum.VacuumInput(equil, psilim, ctrl.mthvac, mpert, mlow, n; force_wv_symmetry=ctrl.force_wv_symmetry)

            # Compute vacuum energy matrix and both Green's functions
            wv_block, grri, grre, vac_data.plasma_pts, vac_data.wall_pts = Vacuum.compute_vacuum_response(vac_inputs, wall_settings)

            # Store blocks into full matrices
            block_idx = ((idx_n-1)*mpert+1):(idx_n*mpert)
            # Copy the real terms from grri to grre
            @views vac_data.grri[:, block_idx] .= grri[:, 1:mpert]
            @views vac_data.grre[:, block_idx] .= grre[:, 1:mpert]
            # Copy the imaginary terms from grri to grre - offset by numpert_total
            @views vac_data.grri[:, numpert_total .+ block_idx] .= grri[:, (mpert+1):(2*mpert)]
            @views vac_data.grre[:, numpert_total .+ block_idx] .= grre[:, (mpert+1):(2*mpert)]
            @views vac_data.wv[block_idx, block_idx] .= wv_block
        end
    else
        if ctrl.verbose
            println("Computing 3D vacuum response matrix in addition to 2D matrix with nzvac = $(ctrl.nzvac)")
        end

        # Compute 3D vacuum response matrix
        vac_inputs = Vacuum.VacuumInput(equil, psilim, ctrl.mthvac, mpert, mlow, 1; force_wv_symmetry=ctrl.force_wv_symmetry)
        vac_inputs_3D = Vacuum.VacuumInput3D(vac_inputs, ctrl.nzvac, nlow, npert)
        vac_data.wv, vac_data.grri, vac_data.grre, vac_data.plasma_pts, vac_data.wall_pts = @timev Vacuum.compute_vacuum_response(vac_inputs_3D, wall_settings)
    end

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
    wpt .= adjoint(vac_data.wt) * (wp * vac_data.wt)
    wvt .= adjoint(vac_data.wt) * (vac_data.wv * vac_data.wt)
    for ipert in 1:numpert_total
        vac_data.ep[ipert] = wpt[ipert, ipert]
        vac_data.ev[ipert] = wvt[ipert, ipert]
    end

    # Normalize eigenvectors based on scaled wt
    coeffs = odet.u[:, :, 1, end] \ (vac_data.wt .* (2π * equil.psio * 1e-3))
    @views for istep in 1:odet.step
        mul!(odet.tmp, odet.u_store[:, :, 1, istep], coeffs)
        odet.u_store[:, :, 1, istep] .= odet.tmp
        mul!(odet.tmp, odet.u_store[:, :, 2, istep], coeffs)
        odet.u_store[:, :, 2, istep] .= odet.tmp
        mul!(odet.tmp, odet.ud_store[:, :, 1, istep], coeffs)
        odet.ud_store[:, :, 1, istep] .= odet.tmp
        mul!(odet.tmp, odet.ud_store[:, :, 2, istep], coeffs)
        odet.ud_store[:, :, 2, istep] .= odet.tmp
    end

    # Write energies to screen
    if ctrl.verbose
        println("Least Stable Eigenmode Energies:")
        println("  Plasma = ", (@sprintf "%+.3e %+.3ei" real(vac_data.ep[1]) imag(vac_data.ep[1])))
        println("  Vacuum = ", (@sprintf "%+.3e %+.3ei" real(vac_data.ev[1]) imag(vac_data.ev[1])))
        println("  Total  = ", (@sprintf "%+.3e %+.3ei" real(vac_data.et[1]) imag(vac_data.et[1])))
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
        # Space points evenly in q
        qi = qedge + (intr.qlim - qedge) * (i / npsi)

        # Newton iteration to find psi at qi
        psii = ctrl.psiedge + (intr.psilim - ctrl.psiedge) * ((i - 1) / npsi)
        converged = false
        for _ in 1:itmax
            dpsi = (qi - profiles.q_spline(psii)) / profiles.q_deriv(psii)
            psii += dpsi
            if abs(dpsi) < eps * abs(psii)
                converged = true
                psi_array[i] = psii
                break
            end
        end
        if !converged
            error("Newton iteration for psilim did not converge after $itmax iterations.")
        end

        for ipert_n in 1:intr.npert
            # Compute vacuum matrix
            n = ipert_n - 1 + intr.nlow
            vac_inputs = Vacuum.VacuumInput(equil, intr.psilim, intr.mtheta, intr.mpert, intr.mlow, n; force_wv_symmetry=ctrl.force_wv_symmetry)
            wv_block, _, _ = Vacuum.compute_vacuum_response(vac_inputs, intr.wall_settings)

            # Apply singular factor scaling
            singfac = collect(intr.mlow:intr.mhigh) .- (n * qi)
            @inbounds for ipert in 1:intr.mpert
                @views wv_block[ipert, :] .*= singfac[ipert]
                @views wv_block[:, ipert] .*= singfac[ipert]
            end

            # Store block in full wv matrix
            @views wv_array[i, ((ipert_n-1)*intr.mpert+1):(ipert_n*intr.mpert), ((ipert_n-1)*intr.mpert+1):(ipert_n*intr.mpert)] .= wv_block
        end
    end

    # Flatten 3D array to (npsi+1 × numpert_total^2) for series interpolant
    wv_flat = reshape(wv_array, npsi + 1, intr.numpert_total^2)

    # FastInterpolations now natively supports complex values - create complex series interpolant directly
    # Use CubicFit() for native endpoint handling
    wvmat = cubic_interp(psi_array, wv_flat; bc=CubicFit(), extrap=:extension, search=LinearBinary())

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

    dV_dpsi = equil.profiles.dVdpsi_spline(intr.psilim)

    # Compute plasma response matrix
    @views wp = (odet.u[:, :, 2] / odet.u[:, :, 1]) ./ equil.psio^2

    # Compute vacuum matrix from series interpolant (use separate hint for wv grid)
    # FastInterpolations now natively supports complex values
    odet.wvmat(vec(odet._wv_out), odet.psifac; hint=odet.wv_hint)
    wv = odet._wv_out

    # Compute total energy matrix and eigen-decomposition
    wt .= wp .+ wv
    Ev = eigen(wt)

    # Sort eigenvalues and reorder columns of wt
    eindex = sortperm(real.(Ev.values); rev=true)
    for ipert in 1:intr.numpert_total
        wt[:, ipert] .= Ev.vectors[:, eindex[intr.numpert_total+1-ipert]]
        tot_eigvals[ipert] = Ev.values[eindex[intr.numpert_total+1-ipert]]
    end

    # Normalize eigenfunction and energy (only need the first eigenmode)
    isol = 1
    norm = 0.0 + 0.0im
    for ipert_n in 1:intr.npert, ipert_m in 1:intr.mpert, jpert_m in 1:intr.mpert
        ipert = (ipert_n - 1) * intr.mpert + ipert_m
        jpert = (ipert_n - 1) * intr.mpert + jpert_m
        norm += ffit.jmat[jpert_m-ipert_m+intr.mband+1] * wt[ipert, isol] * conj(wt[jpert, isol])
    end
    norm /= dV_dpsi
    tot_eigvals[isol] /= norm

    return tot_eigvals[1]
end
