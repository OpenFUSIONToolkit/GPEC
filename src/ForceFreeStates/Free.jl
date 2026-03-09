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
    vac = VacuumData(ctrl.mthvac, intr.mpert, intr.numpert_total)

    Npert = intr.numpert_total

    etemp = zeros!(pool, ComplexF64, Npert)
    wp = zeros!(pool, ComplexF64, Npert, Npert)
    wpt = zeros!(pool, ComplexF64, Npert, Npert)
    wvt = zeros!(pool, ComplexF64, Npert, Npert)

    tmp_mat = zeros!(pool, ComplexF64, Npert, Npert)

    # Evaluate dV/dpsi at the plasma edge. For diverted plasmas psilim can exceed psihigh,
    # so route to the edge inverse spline when available and psilim is beyond the core grid.
    profiles = equil.profiles
    dV_dpsi = (!isnothing(profiles.dVdpsi_spline_inv) && intr.psilim > profiles.xs[end]) ?
              profiles.dVdpsi_spline_inv(intr.psilim) :
              profiles.dVdpsi_spline(intr.psilim)

    # Compute plasma response matrix W = U₂ * U₁⁻¹
    if ctrl.ode_flag
        @views wp .= (odet.u[:, :, 2] / odet.u[:, :, 1]) ./ equil.psio^2
    end

    # Compute vacuum response matrix
    for ipert_n in 1:intr.npert
        # Set VACUUM run parameters and boundary shape
        n = ipert_n - 1 + intr.nlow
        vac_inputs = Vacuum.VacuumInput(equil, intr.psilim, ctrl.mthvac, intr.mpert, intr.mlow, n; force_wv_symmetry=ctrl.force_wv_symmetry)
        fill!(vac.grri, 0.0)
        fill!(vac.grre, 0.0)
        fill!(vac.xzpts, 0.0)

        # Compute vacuum energy matrix and both Green's functions
        wv_block, vac.grri, vac.grre, vac.xzpts = Vacuum.compute_vacuum_response(vac_inputs, intr.wall_settings)

        # Scale by (m - n*q)(m' - n*q) [Chance Phys. Plasmas 1997 2161 eq. 126]
        singfac = collect(intr.mlow:intr.mhigh) .- (n * intr.qlim)
        @inbounds for ipert in 1:intr.mpert
            @views wv_block[ipert, :] .*= singfac[ipert]
            @views wv_block[:, ipert] .*= singfac[ipert]
        end

        # Store block in full wv matrix
        @views vac.wv[((ipert_n-1)*intr.mpert+1):(ipert_n*intr.mpert), ((ipert_n-1)*intr.mpert+1):(ipert_n*intr.mpert)] .= wv_block
    end

    # Compute complex energy eigenvalues and vectors
    vac.wt .= wp .+ vac.wv
    vac.wt0 .= vac.wt
    Ev = eigen(vac.wt)
    vac.et .= Ev.values
    eindex = sortperm(real.(vac.et); rev=true)

    etemp .= vac.et
    # Rearrange wt columns for descending real eigenvalues
    for ipert in 1:intr.numpert_total
        vac.wt[:, ipert] .= Ev.vectors[:, eindex[intr.numpert_total+1-ipert]]
        vac.et[ipert] = etemp[eindex[intr.numpert_total+1-ipert]]
    end

    # Normalize eigenfunction and energy.
    for isol in 1:intr.numpert_total
        norm = 0.0 + 0.0im
        for ipert_n in 1:intr.npert, ipert_m in 1:intr.mpert, jpert_m in 1:intr.mpert
            ipert = (ipert_n - 1) * intr.mpert + ipert_m
            jpert = (ipert_n - 1) * intr.mpert + jpert_m
            norm += ffit.jmat[jpert_m-ipert_m+intr.mband+1] * vac.wt[ipert, isol] * conj(vac.wt[jpert, isol])
        end
        norm /= dV_dpsi
        vac.wt[:, isol] ./= sqrt(norm)
        vac.et[isol] /= norm
    end

    # Normalize phase
    imax = 0
    for isol in 1:intr.numpert_total
        imax = argmax(abs.(vac.wt[:, isol]))
        phase = abs(vac.wt[imax, isol]) / vac.wt[imax, isol]
        vac.wt[:, isol] .*= phase
    end

    # Compute plasma and vacuum contributions.
    # wpt = wt' * wp * wt  ; wvt = wt' * wv * wt
    mul!(tmp_mat, wp, vac.wt)
    mul!(wpt, adjoint(vac.wt), tmp_mat)
    mul!(tmp_mat, vac.wv, vac.wt)
    mul!(wvt, adjoint(vac.wt), tmp_mat)
    for ipert in 1:intr.numpert_total
        vac.ep[ipert] = wpt[ipert, ipert]
        vac.ev[ipert] = wvt[ipert, ipert]
    end

    # Normalize eigenvectors based on scaled wt
    coeffs = odet.u[:, :, 1, end] \ (vac.wt .* (2π * equil.psio * 1e-3))
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
              "  Plasma = $((@sprintf "%+.3e %+.3ei" real(vac.ep[1]) imag(vac.ep[1])))\n" *
              "  Vacuum = $((@sprintf "%+.3e %+.3ei" real(vac.ev[1]) imag(vac.ev[1])))\n" *
              "  Total  = $((@sprintf "%+.3e %+.3ei" real(vac.et[1]) imag(vac.et[1])))"
    end

    return vac
end

"""
    free_compute_wv_spline(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal)

Compute a spline of vacuum response matrices over the core scan range [psiedge, psihigh].
Used for fast evaluation of wt during `findmax_dW_edge!`. Performs the same function as
`free_wvmats` in the Fortran code. Currently defaults to 20 spline points in the core range.

Grid design: 20 uniform points over [psiedge, psihigh] only. For above-psihigh queries,
ExtendExtrap returns the value at psihigh. Above-psihigh scan steps in `findmax_dW_edge!`
typically fail with SingularException (degenerate raw U₁ after accumulated Gaussian
reductions between fixups), so the spline need not cover that region.

The spline stores RAW vacuum response without singfac scaling. Singfac = (m - n*q) is
applied analytically in free_compute_total at each scan psi, keeping the spline
well-conditioned despite q divergence near the separatrix.
"""
function free_compute_wv_spline(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal)

    profiles = equil.profiles
    psihigh  = profiles.xs[end]

    npsi_core = 20
    psi_array = range(ctrl.psiedge, psihigh; length=npsi_core) |> collect

    wv_array = zeros(ComplexF64, npsi_core, intr.numpert_total, intr.numpert_total)

    for (i, psi_val) in enumerate(psi_array)
        for ipert_n in 1:intr.npert
            n = ipert_n - 1 + intr.nlow
            vac_inputs = Vacuum.VacuumInput(equil, psi_val, ctrl.mthvac, intr.mpert, intr.mlow, n;
                                            force_wv_symmetry=ctrl.force_wv_symmetry)
            wv_block, _, _, _ = Vacuum.compute_vacuum_response(vac_inputs, intr.wall_settings)

            block = ((ipert_n-1)*intr.mpert+1):(ipert_n*intr.mpert)
            @views wv_array[i, block, block] .= wv_block
        end
    end

    wv_flat = reshape(wv_array, npsi_core, intr.numpert_total^2)
    # ExtendExtrap: above-psihigh queries return the psihigh value (constant extrapolation).
    # Above-psihigh scan steps use the frozen wv_raw(psihigh) combined with singfac(psi) applied
    # analytically; those steps typically fail (SingularException from degenerate raw U₁ above
    # psihigh) and are excluded from the peak search.
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
    # Evaluate dV/dpsi at the current scan psi (odet.psifac), not at the fixed psilim.
    # In findmax_dW_edge!, odet.psifac is updated at each step; using psilim here would
    # give a ~100x wrong normalization near the separatrix where dV/dpsi diverges.
    dV_dpsi = (!isnothing(profiles.dVdpsi_spline_inv) && odet.psifac > profiles.xs[end]) ?
              profiles.dVdpsi_spline_inv(odet.psifac) :
              profiles.dVdpsi_spline(odet.psifac)

    # Compute plasma response matrix
    @views wp = (odet.u[:, :, 2] / odet.u[:, :, 1]) ./ equil.psio^2

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
