"""
    free_run!(odet::OdeState, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::DconInternal)

Compute the free boundary energies using VACUUM. Performs the same function as `free_run`
in the Fortran code, except now all data is passed in memory instead of via files. This
modifies `odet` in place to normalize the eigenfunctions stored in `u_store` and `ud_store`,
and returns a `VacuumData` struct containing the data needed for perturbed equilibrium calculations
and data dumping.

### TODOs
Check if normalize is ever false, currently always true, and if not, remove related logic
"""
function free_run!(odet::OdeState, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::DconInternal, wall_settings::VacuumMod.WallShapeSettings)

    # TODO: it looks like vac_memory is always true - remove all ahg things and just assume true?
    vac_memory = true
    # TODO: this is always true in fortran - just get rid of it?
    normalize = true
    # Flags used within VACUUM
    complex_flag = true
    wall_flag = false
    ahg_file = "ahg2msc_dcon.out" # Deprecated, but needed for VACUUM function call

    # Initializations and allocations
    vac = VacuumData(ctrl.mthvac, intr.mpert, intr.numpert_total)
    etemp = zeros(ComplexF64, intr.numpert_total)
    wp = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)
    wpt = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)
    wvt = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)
    wv_temp = zeros(ComplexF64, intr.mpert, intr.mpert)

    # Evaluate dV/dpsi at the plasma edge
    v1 = Spl.spline_eval!(equil.sq, intr.psilim)[3]

    # Compute plasma response matrix W = U₂ * U₁⁻¹
    if ctrl.ode_flag
        @views wp = (odet.u[:, :, 2] / odet.u[:, :, 1]) ./ equil.psio^2
    end

    # Compute vacuum response matrix
    wv_block = zeros(ComplexF64, intr.mpert, intr.mpert)
    for ipert_n in 1:intr.npert
        n = ipert_n - 1 + intr.nlow
        
        # Set VACUUM run parameters and boundary shape
        vac_inputs = set_vacuum_inputs(intr.psilim, n, equil, intr)
        fill!(vac.grri, 0.0)
        fill!(vac.xzpts, 0.0)
        vac_inputs.mtheta = ctrl.mthvac
        vac_inputs.force_wv_symmetry = ctrl.force_wv_symmetry

        # repeated calls are needed only when gpec code using mutual inductances is added
        # vac_inputs.farwall_flag = true
        # vac_inputs.kernelsign = -1.0
        # # TODO: make this a ! function, it modifies wv, grri, and xzpts in place (but only wv is used)
        # VacuumMod.mscvac(wv_block, intr.mpert, equil.config.control.mtheta, ctrl.mthvac, complex_flag, vac_inputs.kernelsign,
        #     wall_flag, vac_inputs.farwall_flag, vac.grri, vac.xzpts, ahg_file, intr.dir_path)
        
        # vac_inputs.kernelsign = 1.0
        # VacuumMod.mscvac(wv_block, intr.mpert, equil.config.control.mtheta, ctrl.mthvac, complex_flag, vac_inputs.kernelsign,
        #     wall_flag, vac_inputs.farwall_flag, vac.grri, vac.xzpts, ahg_file, intr.dir_path)

        # if ctrl.wv_farwall_flag
        #     wv_temp .= wv_block
        # end

        vac_inputs.farwall_flag = false
        vac_inputs.kernelsign = -1.0
        VacuumMod.mscvac(wv_block, intr.mpert, equil.config.control.mtheta, ctrl.mthvac, complex_flag, vac_inputs.kernelsign,
            wall_flag, vac_inputs.farwall_flag, vac.grri, vac.xzpts, ahg_file, intr.dir_path)

        vac_inputs.kernelsign = 1.0
        VacuumMod.mscvac(wv_block, intr.mpert, equil.config.control.mtheta, ctrl.mthvac, complex_flag, vac_inputs.kernelsign,
            wall_flag, vac_inputs.farwall_flag, vac.grri, vac.xzpts, ahg_file, intr.dir_path)

        println("WV from Fortran")
        display(wv_block)

        # Placeholder for Julia vacuum code
        wv_block, vac.grri, vac.xzpts = VacuumMod.compute_vacuum_response(wall_settings, vac_inputs, wall_flag, intr.dir_path)
        println("WV from Fortran")
        display(wv_block)
        # error("Debug: Made it through compute_vacuum_response in Free.jl!")

        if ctrl.wv_farwall_flag
            wv_block .= wv_temp
        end

        # Scale vacuum matrix by singfac = (m - n*qlim)
        singfac = collect(intr.mlow:intr.mhigh) .- (n * intr.qlim)
        for ipert in 1:intr.mpert
            wv_block[ipert, :] .*= singfac[ipert]
            wv_block[:, ipert] .*= singfac[ipert]
        end

        # Store block in full wv matrix
        @views vac.wv[(ipert_n-1)*intr.mpert+1 : ipert_n*intr.mpert, (ipert_n-1)*intr.mpert+1 : ipert_n*intr.mpert] .= wv_block

        if vac_memory
            VacuumMod.unset_dcon_params()
        end
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
    if normalize
        for isol in 1:intr.numpert_total
            norm = 0.0 + 0.0im
            for ipert_n in 1:intr.npert, ipert_m in 1:intr.mpert, jpert_m in 1:intr.mpert
                ipert = (ipert_n - 1) * intr.mpert + ipert_m
                jpert = (ipert_n - 1) * intr.mpert + jpert_m
                norm += ffit.jmat[jpert_m-ipert_m+intr.mband+1] * vac.wt[ipert, isol] * conj(vac.wt[jpert, isol])
            end
            norm /= v1
            vac.wt[:, isol] ./= sqrt(norm)
            vac.et[isol] /= norm
        end
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
    wpt .= adjoint(vac.wt) * (wp * vac.wt)
    wvt .= adjoint(vac.wt) * (vac.wv * vac.wt)
    for ipert in 1:intr.numpert_total
        vac.ep[ipert] = wpt[ipert, ipert]
        vac.ev[ipert] = wvt[ipert, ipert]
    end

    # Normalize eigenvectors based on scaled wt
    coeffs = odet.u[:,:,1,end] \ (vac.wt .* (2π * equil.psio * 1e-3))
    for istep in 1:odet.step
        odet.u_store[:, :, 1, istep] .= odet.u_store[:, :, 1, istep] * coeffs
        odet.u_store[:, :, 2, istep] .= odet.u_store[:, :, 2, istep] * coeffs
        odet.ud_store[:, :, 1, istep] .= odet.ud_store[:, :, 1, istep] * coeffs
        odet.ud_store[:, :, 2, istep] .= odet.ud_store[:, :, 2, istep] * coeffs
    end

    # Write energies to screen
    if ctrl.verbose
        println("   Energies: plasma = ", real(vac.ep[1]), ", vacuum = ", real(vac.ev[1]),
            ", real = ", real(vac.et[1]), ", imaginary = ", imag(vac.et[1]))
    end

    return vac
end

"""
    set_vacuum_inputs(psifac::Float64, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal)

Prepare and write the necessary parameters and boundary shape to VACUUM for computing the vacuum response matrix.
Performs the same function as `free_write_msc` in the Fortran code, except we will always use in-memory communication.

### Arguments

  - `psifac`: Flux surface value at the plasma boundary (Float64)
  - `n`: Toroidal mode number (Int)
  - `equil`: Plasma equilibrium data (Equilibrium.PlasmaEquilibrium)
  - `intr`: Internal DCON parameters (DconInternal)

"""
function set_vacuum_inputs(psifac::Float64, n::Int, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal)

    # Allocations
    theta_norm = Vector(equil.rzphi.ys)
    mtheta = equil.config.control.mtheta
    angle = zeros(Float64, mtheta + 1)
    r = zeros(Float64, mtheta + 1)
    z = zeros(Float64, mtheta + 1)
    delta = zeros(Float64, mtheta + 1)
    rfac = zeros(Float64, mtheta + 1)

    # Compute output
    qa = Spl.spline_eval!(equil.sq, psifac)[4]
    for itheta in 1:equil.config.control.mtheta+1
        f = Spl.bicube_eval!(equil.rzphi, psifac, theta_norm[itheta])
        rfac[itheta] = sqrt(f[1])
        angle[itheta] = 2π * (theta_norm[itheta] + f[2])
        delta[itheta] = -f[3] / qa
    end
    r .= equil.ro .+ rfac .* cos.(angle)
    z .= equil.zo .+ rfac .* sin.(angle)

    # Invert values for n < 0
    if n < 0
        qa = -qa
        delta .= -delta
        n = -n
    end

    # Pass all required values to VACUUM
    VacuumMod.set_dcon_params(equil.config.control.mtheta, intr.mlow, intr.mhigh, n, qa,
        reverse(r), reverse(z), reverse(delta))

    # For input to the Julia vacuum code
    return VacuumMod.VacuumInput(;
        r = reverse(r),
        z = reverse(z),
        delta = reverse(delta),
        mhigh = intr.mhigh,
        mlow = intr.mlow,
        mpert = intr.mpert,
        n = n,
        qa = qa,
        mtheta_eq = equil.config.control.mtheta
    )
end

"""
    free_compute_wv_spline(ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal)

Compute a spline of vacuum response matrices over the range of psi from 'ctrl.psi_edge' to
`intr.qlim`. This is used for fast evaluation of wt during `ode_record_edge`. Performs the
same function as `free_wvmats` in the Fortran code.
"""
function free_compute_wv_spline(ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal)

    # Number of psi grid points for the spline: 4 per q-window minimum
    # TODO: 4 spline points is arbitrary - is there a better way?
    qedge = Spl.spline_eval!(equil.sq, ctrl.psiedge)[4]
    npsi = max(4, ceil(Int, (intr.qlim - qedge) * intr.nhigh * 4))
    psii = ctrl.psiedge
    psi_array = zeros(Float64, npsi + 1)
    wv_array = zeros(ComplexF64, npsi + 1, intr.numpert_total, intr.numpert_total)

    for i in 1:npsi+1
        # Space points evenly in q
        qi = qedge + (intr.qlim - qedge) * (i / npsi)

        # Shorthand to evaluate q/q1 inside newton iteration
        qval(ψ) = Spl.spline_eval!(equil.sq, ψ)[4]
        q1val(ψ) = Spl.spline_deriv1!(equil.sq, ψ)[2][4]

        # Newton iteration to find psi at qi
        psii = ctrl.psiedge + (intr.psilim - ctrl.psiedge) * ((i - 1) / npsi)
        converged = false
        for _ in 1:itmax
            dpsi = (qi - qval(psii)) / q1val(psii)
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

        wv_block = zeros(ComplexF64, intr.mpert, intr.mpert)
        for ipert_n in 1:intr.npert
            n = ipert_n - 1 + intr.mlow

            # Prepare vacuum matrices
            set_vacuum_inputs(psii, n, equil, intr)
            grri = Array{Float64}(undef, 2 * (ctrl.mthvac + 5), intr.mpert * 2)
            xzpts = Array{Float64}(undef, ctrl.mthvac + 5, 4)

            complex_flag = true
            kernelsignin = 1.0
            wall_flag = false
            farwall_flag = false
            ahg_file = "ahg2msc_dcon.out" # Deprecated

            # Compute vacuum matrix
            VacuumMod.mscvac(wv_block, intr.mpert, equil.config.control.mtheta, ctrl.mthvac, complex_flag, kernelsignin,
                wall_flag, farwall_flag, grri, xzpts, ahg_file, intr.dir_path)

            # Apply singular factor scaling
            singfac = collect(intr.mlow:intr.mhigh) .- (n * qi)
            for ipert in 1:intr.mpert
                wv_block[ipert, :] .*= singfac[ipert]
                wv_block[:, ipert] .*= singfac[ipert]
            end

            # Store block in full wv matrix
            @views wv_array[i, (ipert_n-1)*intr.mpert+1 : ipert_n*intr.mpert, (ipert_n-1)*intr.mpert+1 : ipert_n*intr.mpert] .= wv_block

            # Free VACUUM memory
            VacuumMod.unset_dcon_params()
        end
    end

    return Spl.CubicSpline(psi_array, reshape(wv_array, npsi+1, :); bctype="extrap")
end

"""
    free_compute_total(equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::DconInternal, odet::OdeState) -> ComplexF64

Compute total complex energy eigenvalue (total1). This is a trimmed down version of `free_run`
that only computes the total energy eigenvalue for the mode unstable mode, used in `ode_record_edge_dW`
which calls this function at each step in the psiedge -> psilim region of integration. This performs
the same function as `free_test` in the Fortran code, except we have moved the creation of the
wv matrix spline to `free_compute_wv_spline` and pass it in `odet`.wvmat_spline.
"""
function free_compute_total(equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::DconInternal, odet::OdeState)

    normalize = true

    wp = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)
    tot_eigvals = zeros(ComplexF64, intr.numpert_total)
    wt = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)

    v1 = Spl.spline_eval!(equil.sq, intr.psilim)[3]

    # Compute plasma response matrix
    @views wp = (odet.u[:, :, 2] / odet.u[:, :, 1]) ./ equil.psio^2

    # Compute vacuum matrix from spline
    wv = reshape(Spl.spline_eval!(odet.wvmat_spline, odet.psifac), intr.numpert_total, intr.numpert_total)

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
    if normalize
        isol = 1
        norm = 0.0 + 0.0im
        for ipert_n in 1:intr.npert, ipert_m in 1:intr.mpert, jpert_m in 1:intr.mpert
            ipert = (ipert_n - 1) * intr.mpert + ipert_m
            jpert = (ipert_n - 1) * intr.mpert + jpert_m
            norm += ffit.jmat[jpert_m-ipert_m+intr.mband+1] * wt[ipert, isol] * conj(wt[jpert, isol])
        end
        norm /= v1
        tot_eigvals[isol] /= norm
    end

    return tot_eigvals[1]
end