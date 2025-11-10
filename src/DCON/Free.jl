"""
    free_run!(odet::OdeState, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::DconInternal, outp::DconOutput)

Compute the free boundary energies using VACUUM. Performs the same function as `free_run`
in the Fortran code, except now all data is passed in memory instead of via files.

### TODOs
Check if normalize is ever false, currently always true, and if not, remove related logic
"""

function free_run!(odet::OdeState, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::DconInternal, outp::DconOutput)

    # TODO: it looks like vac_memory is always true - remove all ahg things and just assume true?
    vac_memory = true
    # TODO: this is always true in fortran - just get rid of it?
    normalize = true
    # Flags used within VACUUM
    complex_flag = true
    wall_flag = false
    ahg_file = "ahg2msc_dcon.out" # Deprecated, but needed for VACUUM function call

    # Allocations
    star = fill(' ', intr.mpert, intr.mpert)
    plas_eigvals = zeros(ComplexF64, intr.mpert)
    vac_eigvals = zeros(ComplexF64, intr.mpert)
    tot_eigvals = zeros(ComplexF64, intr.mpert)
    temp_eigvals = zeros(ComplexF64, intr.mpert)
    wv = zeros(ComplexF64, intr.mpert, intr.mpert)
    wt = zeros(ComplexF64, intr.mpert, intr.mpert)
    wt0 = zeros(ComplexF64, intr.mpert, intr.mpert)
    wp = zeros(ComplexF64, intr.mpert, intr.mpert)
    temp = zeros(ComplexF64, intr.mpert, intr.mpert)
    wpt = zeros(ComplexF64, intr.mpert, intr.mpert)
    wvt = zeros(ComplexF64, intr.mpert, intr.mpert)

    # Evaluate dV/dpsi at the plasma edge
    v1 = Spl.spline_eval!(equil.sq, intr.psilim)[3]

    # Compute plasma response matrix W = U₂ * U₁⁻¹
    if ctrl.ode_flag
        @views wp = (odet.u[:, :, 2] / odet.u[:, :, 1]) ./ equil.psio^2
    end

    # Set VACUUM run parameters and boundary shape
    set_vacuum_inputs(intr.psilim, ctrl, equil, intr)

    # Compute vacuum response matrix.
    grri = Array{Float64}(undef, 2 * (ctrl.mthvac + 5), intr.mpert * 2)
    xzpts = Array{Float64}(undef, ctrl.mthvac + 5, 4)

    farwal_flag = true
    kernelsignin = -1.0
    # TODO: make this a ! function, it modifies wv, grri, and xzpts in place (but only wv is used)
    VacuumMod.mscvac(wv, intr.mpert, equil.config.control.mtheta, ctrl.mthvac, complex_flag, kernelsignin,
        wall_flag, farwal_flag, grri, xzpts, ahg_file, intr.dir_path)


    kernelsignin = 1.0
    VacuumMod.mscvac(wv, intr.mpert, equil.config.control.mtheta, ctrl.mthvac, complex_flag, kernelsignin,
        wall_flag, farwal_flag, grri, xzpts, ahg_file, intr.dir_path)

    if ctrl.wv_farwall_flag
        temp .= wv
    end

    farwal_flag = false
    kernelsignin = -1.0
    VacuumMod.mscvac(wv, intr.mpert, equil.config.control.mtheta, ctrl.mthvac, complex_flag, kernelsignin,
        wall_flag, farwal_flag, grri, xzpts, ahg_file, intr.dir_path)

    kernelsignin = 1.0
    VacuumMod.mscvac(wv, intr.mpert, equil.config.control.mtheta, ctrl.mthvac, complex_flag, kernelsignin,
        wall_flag, farwal_flag, grri, xzpts, ahg_file, intr.dir_path)

    if ctrl.wv_farwall_flag
        wv .= temp
    end

    # Scale vacuum matrix by singfac = (m - nn*qlim)
    singfac = collect(intr.mlow:intr.mhigh) .- ctrl.nn .* intr.qlim
    for ipert in 1:intr.mpert
        wv[ipert, :] .*= singfac[ipert]
        wv[:, ipert] .*= singfac[ipert]
    end

    # Compute complex energy eigenvalues and vectors
    wt .= wp .+ wv
    wt0 .= wt
    Ev = eigen(wt)
    tot_eigvals .= Ev.values
    eindex = sortperm(real.(tot_eigvals); rev=true)
    temp_eigvals .= tot_eigvals
    # Rearrange wt columns to correspond to eigenvector reordering similar to Fortran
    for ipert in 1:intr.mpert
        wt[:, ipert] .= Ev.vectors[:, eindex[intr.mpert+1-ipert]]
        tot_eigvals[ipert] = temp_eigvals[eindex[intr.mpert+1-ipert]]
    end

    # Normalize eigenfunction and energy.
    if normalize
        for isol in 1:intr.mpert
            norm = 0.0 + 0.0im
            for ipert in 1:intr.mpert, jpert in 1:intr.mpert
                norm += ffit.jmat[jpert-ipert+intr.mband+1] * wt[ipert, isol] * conj(wt[jpert, isol])
            end
            norm /= v1
            wt[:, isol] ./= sqrt(norm)
            tot_eigvals[isol] /= norm
        end
    end

    # Normalize phase and label largest component.
    imax = 0
    for isol in 1:intr.mpert
        # get index of largest absolute component (first occurrence)
        imax = argmax(abs.(wt[:, isol]))
        phase = abs(wt[imax, isol]) / wt[imax, isol]
        wt[:, isol] .*= phase
        # Mark largest component with *, all others initalized to ' '
        star[imax, isol] = '*'
    end

    # Compute plasma and vacuum contributions.
    # wpt = wt' * wp * wt  ; wvt = wt' * wv * wt
    wpt .= adjoint(wt) * (wp * wt)
    wvt .= adjoint(wt) * (wv * wt)

    for ipert in 1:intr.mpert
        plas_eigvals[ipert] = wpt[ipert, ipert]
        vac_eigvals[ipert] = wvt[ipert, ipert]
    end

    plasma1 = ComplexF64(real(plas_eigvals[1]), 0.0)
    vacuum1 = ComplexF64(real(vac_eigvals[1]), 0.0)
    total1 = ComplexF64(real(tot_eigvals[1]), 0.0)

    if vac_memory
        VacuumMod.unset_dcon_params()
    end

    # Normalize eigenvectors based on scaled wt
    coeffs = odet.u[:,:,1,end] \ (wt .* (2π * equil.psio * 1e-3))
    for istep in 1:odet.step
        odet.u_store[:, :, 1, istep] .= odet.u_store[:, :, 1, istep] * coeffs
        odet.u_store[:, :, 2, istep] .= odet.u_store[:, :, 2, istep] * coeffs
        odet.ud_store[:, :, 1, istep] .= odet.ud_store[:, :, 1, istep] * coeffs
        odet.ud_store[:, :, 2, istep] .= odet.ud_store[:, :, 2, istep] * coeffs
    end

    # Write to euler.h5
    if outp.write_euler_h5
        # We open in r+ mode to add to the existing file from ode_output_init instead of overwriting it
        h5open(joinpath(intr.dir_path, outp.fname_euler_h5), "r+") do euler_h5
            euler_h5["vacuum/wt"] = wt
            euler_h5["vacuum/wt0"] = wt0
            euler_h5["vacuum/ep"] = plas_eigvals
            euler_h5["vacuum/ev"] = vac_eigvals
            euler_h5["vacuum/et"] = tot_eigvals
            euler_h5["vacuum/wv_farwall_flag"] = ctrl.wv_farwall_flag
            euler_h5["integration/xi_psi"] = odet.u_store[:, :, 1, :]
            euler_h5["integration/u2"] = odet.u_store[:, :, 2, :] # TODO: what to name this? These are the "conjugate momenta" of u1
            euler_h5["integration/dxi_psi"] = odet.ud_store[:, :, 1, :]
            euler_h5["integration/xi_s"] = odet.ud_store[:, :, 2, :]
        end
    end

    # Write to screen and copy to output.
    if ctrl.verbose
        println("Energies: plasma = ", real(plas_eigvals[1]), ", vacuum = ", real(vac_eigvals[1]),
            ", real = ", real(tot_eigvals[1]), ", imaginary = ", imag(tot_eigvals[1]))
    end

    if outp.write_dcon_out
        # Write eigenvalues to file
        write_output(outp, :dcon_out, "\nTotal Energy Eigenvalues:")
        write_output(outp, :dcon_out, "\n   isol   plasma      vacuum   re total   im total\n")
        for isol in 1:intr.mpert
            write_output(outp, :dcon_out, @sprintf("%6d %11.3e %11.3e %11.3e %11.3e",
                isol, real(plas_eigvals[isol]), real(vac_eigvals[isol]), real(tot_eigvals[isol]), imag(tot_eigvals[isol])))
        end
        write_output(outp, :dcon_out, "\n   isol   plasma      vacuum   re total   im total\n")

        # Write eigenvectors to file
        write_output(outp, :dcon_out, "Total Energy Eigenvectors:")
        m = intr.mlow .+ collect(0:intr.mpert-1)
        for isol in 1:intr.mpert
            write_output(outp, :dcon_out, "\n   isol   imax   plasma      vacuum   re total   im total\n")
            write_output(outp, :dcon_out, @sprintf("%6d %6d %11.3e %11.3e %11.3e %11.3e",
                isol, imax, real(plas_eigvals[isol]), real(vac_eigvals[isol]), real(tot_eigvals[isol]), imag(tot_eigvals[isol])))
            write_output(outp, :dcon_out, "\n  ipert     m      re wt      im wt      abs wt\n")
            for ipert in 1:intr.mpert
                write_output(outp, :dcon_out,
                    @sprintf("%6d %6d %11.3e %11.3e %11.3e %s",
                        ipert, m[ipert], real(wt[ipert, isol]), imag(wt[ipert, isol]), abs(wt[ipert, isol]), star[ipert, isol]))
            end
            write_output(outp, :dcon_out, "\n  ipert     m      re wt      im wt      abs wt\n")
        end

        # Write the plasma matrix to file
        write_output(outp, :dcon_out, "Plasma Energy Matrix:\n")
        for isol in 1:intr.mpert
            write_output(outp, :dcon_out, "isol = $(isol), m = $(m[isol])")
            write_output(outp, :dcon_out, "\n  i     re wp        im wp        abs wp\n")
            for ipert in 1:intr.mpert
                write_output(outp, :dcon_out, @sprintf("%3d%13.5e%13.5e%13.5e",
                    ipert, real(wp[ipert, isol]), imag(wp[ipert, isol]), abs(wp[ipert, isol])))
            end
            write_output(outp, :dcon_out, "\n  i     re wp        im wp        abs wp\n")
        end
    end

    return plasma1, vacuum1, total1
end

"""
    set_vacuum_inputs(psifac::Float64, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal)

Prepare and write the necessary parameters and boundary shape to VACUUM for computing the vacuum response matrix.
Performs the same function as `free_write_msc` in the Fortran code, except we will always use in-memory communication.

### Arguments

  - `psifac`: Flux surface value at the plasma boundary (Float64)
  - `ctrl`: DCON control parameters (DconControl)
  - `equil`: Plasma equilibrium data (Equilibrium.PlasmaEquilibrium)
  - `intr`: Internal DCON parameters (DconInternal)

"""
function set_vacuum_inputs(psifac::Float64, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal)

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

    # Invert values for nn < 0
    n = ctrl.nn
    if ctrl.nn < 0
        qa = -qa
        delta .= -delta
        n = -n
    end

    # Pass all required values to VACUUM
    VacuumMod.set_dcon_params(equil.config.control.mtheta, intr.mlow, intr.mhigh, n, qa,
        reverse(r), reverse(z), reverse(delta))
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
    npsi = max(4, ceil(Int, (intr.qlim - qedge) * ctrl.nn * 4))
    psii = ctrl.psiedge
    psi_array = zeros(Float64, npsi + 1)
    wv_array = zeros(ComplexF64, npsi + 1, intr.mpert^2)

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

        # Prepare vacuum matrices
        set_vacuum_inputs(psii, ctrl, equil, intr)
        grri = Array{Float64}(undef, 2 * (ctrl.mthvac + 5), intr.mpert * 2)
        xzpts = Array{Float64}(undef, ctrl.mthvac + 5, 4)
        wv = zeros(ComplexF64, intr.mpert, intr.mpert)
        complex_flag = true
        kernelsignin = 1.0
        wall_flag = false
        farwal_flag = false
        ahg_file = "ahg2msc_dcon.out" # Deprecated

        # Compute vacuum matrix
        VacuumMod.mscvac(wv, intr.mpert, equil.config.control.mtheta, ctrl.mthvac, complex_flag, kernelsignin,
            wall_flag, farwal_flag, grri, xzpts, ahg_file, intr.dir_path)

        # Apply singular factor scaling
        singfac = collect(intr.mlow:intr.mhigh) .- ctrl.nn * qi
        for ipert in 1:intr.mpert
            wv[ipert, :] .*= singfac[ipert]
            wv[:, ipert] .*= singfac[ipert]
        end

        # Store flattened matrix in spline field
        wv_array[i, :] .= reshape(wv, intr.mpert^2)
    end

    # Free VACUUM memory
    VacuumMod.unset_dcon_params()

    return Spl.CubicSpline(psi_array, wv_array; bctype=3)
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

    wp = zeros(ComplexF64, intr.mpert, intr.mpert)
    temp = zeros(ComplexF64, intr.mpert, intr.mpert)
    tot_eigvals = zeros(ComplexF64, intr.mpert)
    wt = zeros(ComplexF64, intr.mpert, intr.mpert)

    v1 = Spl.spline_eval!(equil.sq, intr.psilim)[3]

    # Compute plasma response matrix.
    temp .= adjoint(odet.u[:, :, 1])
    wp .= adjoint(odet.u[:, :, 2])
    wp .= temp \ wp
    wp .= adjoint(wp) / equil.psio^2

    # Compute vacuum matrix from spline
    wv = reshape(Spl.spline_eval!(odet.wvmat_spline, odet.psifac), intr.mpert, intr.mpert)

    # Compute total energy matrix and eigen-decomposition
    wt .= wp .+ wv
    Ev = eigen(wt)

    # Sort eigenvalues and reorder columns of wt
    eindex = sortperm(real.(Ev.values); rev=true)
    for ipert in 1:intr.mpert
        wt[:, ipert] .= Ev.vectors[:, eindex[intr.mpert+1-ipert]]
        tot_eigvals[ipert] = Ev.values[eindex[intr.mpert+1-ipert]]
    end

    # Normalize eigenfunction and energy (only need the first eigenmode)
    if normalize
        isol = 1
        norm = 0.0 + 0.0im
        for ipert in 1:intr.mpert, jpert in 1:intr.mpert
            norm += ffit.jmat[jpert-ipert+intr.mband+1] * wt[ipert, isol] * conj(wt[jpert, isol])
        end
        norm /= v1
        tot_eigvals[isol] /= norm
    end

    return tot_eigvals[1]
end