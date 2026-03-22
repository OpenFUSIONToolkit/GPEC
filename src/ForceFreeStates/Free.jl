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
    if ctrl.use_equal_arc_vacuum && !isnothing(equil.raw_profile)
        vac_data = _free_run_equal_arc_vacuum!(vac_data, ctrl, equil, psilim, mpert, mlow, npert, nlow, wall_settings)
    else
        vac_inputs = Vacuum.VacuumInput(equil, psilim, ctrl.mthvac, ctrl.nzvac, mpert, mlow, npert, nlow; force_wv_symmetry=ctrl.force_wv_symmetry)
        Vacuum.compute_vacuum_response!(vac_data, vac_inputs, wall_settings)
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
    _free_run_equal_arc_vacuum!(vac_data, ctrl, equil, intr, psilim, mpert, mlow, npert, nlow, wall_settings) -> VacuumData

Compute the vacuum response matrix using the equal-arc boundary integration path, then transform
back to jac_type mode coefficients via the mode transformation matrix T.

This replaces the direct call to `Vacuum.compute_vacuum_response!` when `ctrl.use_equal_arc_vacuum = true`.
Writes the result into `vac_data.wv` (and grri, grre if populated) and returns `vac_data`.
"""
function _free_run_equal_arc_vacuum!(
    vac_data::VacuumData, ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    psilim::Float64, mpert::Int, mlow::Int, npert::Int, nlow::Int,
    wall_settings
)
    mhigh = mlow + mpert - 1

    # Equal-arc grid: at least 2×mhigh×factor points, minimum 2×mhigh×64
    mtheta_eq = ctrl.vac_eq_arc_theta_factor * max(abs(mhigh), abs(mlow), 64)

    # Run arc-length ODE and interpolate to equal-arc grid
    R_eq, Z_eq, nu_eq, theta_jac_eq = Equilibrium.equal_arc_fieldline_output(equil, psilim; mtheta_eq=mtheta_eq)

    # Size the equal-arc mode space: enough modes to resolve exp(im_jac * θ_jac(θ_eq))
    # max_rate = max dθ_jac/dθ_eq * mtheta_eq (maximum mode number needed in equal-arc space)
    theta_jac_wrapped = vcat(theta_jac_eq, [1.0])  # close the loop
    dtheta_jac = diff(theta_jac_wrapped)
    max_rate = maximum(dtheta_jac) * mtheta_eq  # dimensionless, ≥ 1 for uniform jac
    mpert_eq_half = ceil(Int, max_rate) * mpert
    mpert_eq = 2 * mpert_eq_half + 1  # symmetric band around 0
    mlow_eq  = -(mpert_eq ÷ 2)

    # Build VacuumInput with equal-arc surface geometry
    vac_inputs_eq = Vacuum.VacuumInput(;
        x=reverse(R_eq), z=reverse(Z_eq), ν=reverse(nu_eq),
        mtheta_in=mtheta_eq, mtheta=mtheta_eq,
        mlow=mlow_eq, mpert=mpert_eq,
        nlow=nlow, npert=npert, nzeta_in=1, nzeta=ctrl.nzvac,
        force_wv_symmetry=ctrl.force_wv_symmetry
    )

    # Compute vacuum response in equal-arc coordinate mode space
    numpert_eq = mpert_eq * npert
    vac_data_eq = VacuumData(mtheta_eq, numpert_eq, mtheta_eq)
    Vacuum.compute_vacuum_response!(vac_data_eq, vac_inputs_eq, wall_settings)

    # Mode transformation T: maps equal-arc modes → jac_type modes
    # G_jac = T * G_eq * T'
    # Build T for a single n; for multi-n, apply block-diagonally
    T_single = Vacuum.jac_to_eq_mode_transform(theta_jac_eq, mlow, mpert, mlow_eq, mpert_eq)

    if npert == 1
        # Single toroidal mode: wv is mpert × mpert
        vac_data.wv .= T_single * vac_data_eq.wv * T_single'
    else
        # Multi-n block-diagonal: build block-diagonal T and apply
        numpert_total = mpert * npert
        T_full = zeros(ComplexF64, numpert_total, numpert_eq)
        for in in 1:npert
            row_range = (in-1)*mpert+1 : in*mpert
            col_range = (in-1)*mpert_eq+1 : in*mpert_eq
            T_full[row_range, col_range] .= T_single
        end
        vac_data.wv .= T_full * vac_data_eq.wv * T_full'
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

        psii = ctrl.psiedge + (intr.psilim - ctrl.psiedge) * ((i - 1) / npsi)
        psi_array[i] = find_zero(
            (psi -> profiles.q_spline(psi) - qi,
                psi -> profiles.q_deriv(psi)),
            psii, Roots.Newton()
        )

        # Compute vacuum response matrix at this psi (2D single-n, 2D multi-n block-diagonal, or 3D)
        vac_inputs = Vacuum.VacuumInput(equil, psii, ctrl.mthvac, ctrl.nzvac, intr.mpert, intr.mlow, intr.npert, intr.nlow; force_wv_symmetry=ctrl.force_wv_symmetry)
        wv, _, _, _, _ = Vacuum.compute_vacuum_response(vac_inputs, intr.wall_settings)

        # Apply singular factor scaling: (m - n*q)(m' - n'*q) [Chance Phys. Plasmas 1997 2161 eq. 126]
        singfac = vec((intr.mlow:intr.mhigh) .- qi .* (intr.nlow:intr.nhigh)')
        @inbounds for ipert in 1:intr.numpert_total
            @views wv[ipert, :] .*= singfac[ipert]
            @views wv[:, ipert] .*= singfac[ipert]
        end

        @views wv_array[i, :, :] .= wv
    end

    # Flatten 3D array to (npsi+1 × numpert_total^2) for series interpolant
    wv_flat = reshape(wv_array, npsi + 1, intr.numpert_total^2)

    # FastInterpolations now natively supports complex values - create complex series interpolant directly
    # Use CubicFit() for native endpoint handling
    wvmat = cubic_interp(psi_array, wv_flat; bc=CubicFit(), extrap=ExtendExtrap(), search=LinearBinary())

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
