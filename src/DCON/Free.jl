"""
    free_run!(odet::OdeState, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::DconInternal) -> VacuumData

Compute the free boundary energies using the Julia port of the VACUUM code. Performs the same function as `free_run`
in the Fortran code, except now all data is passed in memory instead of via files. This
modifies `odet` in place to normalize the eigenfunctions stored in `u_store` and `ud_store`,
and returns a `VacuumData` struct containing the data needed for perturbed equilibrium calculations
and data dumping.
"""
function free_run!(odet::OdeState, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::DconInternal)

    # Initializations and allocations
    vac = VacuumData(ctrl.mthvac, ctrl.nzvac, intr.numpert_total)
    etemp = zeros(ComplexF64, intr.numpert_total)
    wp = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)
    wpt = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)
    wvt = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)

    # Evaluate dV/dpsi at the plasma edge
    dV_dpsi = Spl.spline_eval!(equil.sq, intr.psilim)[3]

    # Compute plasma response matrix W = U₂ * U₁⁻¹
    if ctrl.ode_flag
        @views wp = (odet.u[:, :, 2] / odet.u[:, :, 1]) ./ equil.psio^2
    end

    # Compute vacuum response matrix
    for ipert_n in 1:intr.npert
        # Set VACUUM run parameters and boundary shape
        n = ipert_n - 1 + intr.nlow
        vac_inputs = compute_vacuum_inputs(intr.psilim, n, ctrl, equil, intr)
        fill!(vac.green_fourier, 0.0)

        # Compute block of vacuum energy matrix for one toroidal mode number
        wv_block, vac.green_fourier, vac.plasma_coords, vac.wall_coords = Vacuum.compute_vacuum_response(vac_inputs, intr.wall_settings)

        # Equation 126 in Chance 1997 - scale by (m - n*q)(m' - n*q)
        singfac = collect(intr.mlow:intr.mhigh) .- (n * intr.qlim)
        @inbounds for ipert in 1:intr.mpert
            @views wv_block[ipert, :] .*= singfac[ipert]
            @views wv_block[:, ipert] .*= singfac[ipert]
        end

        # Store block in full wv matrix
        @views vac.wv[((ipert_n-1)*intr.mpert+1):(ipert_n*intr.mpert), ((ipert_n-1)*intr.mpert+1):(ipert_n*intr.mpert)] .= wv_block
    end

    if ctrl.nzvac > 1
        if ctrl.verbose
            println("Computing 3D vacuum response matrix in addition to 2D matrix with nzvac = $(ctrl.nzvac)")
        end

        # Compute 3D vacuum response matrix
        vac_inputs = compute_vacuum_inputs(intr.psilim, 1, ctrl, equil, intr) # n doesn't matter here
        vac_inputs_3D = Vacuum.VacuumInput3D(vac_inputs, ctrl.nzvac, intr.nlow, intr.npert)
        wv3D, _, _, _ = Vacuum.compute_vacuum_response_3D(vac_inputs_3D, intr.wall_settings)

        # Scale by (m - n*q)(m' - n'*q)
        singfac = vec((intr.mlow:intr.mhigh) .- intr.qlim .* (intr.nlow:intr.nhigh)')
        @inbounds for ipert in 1:intr.numpert_total
            @views wv3D[ipert, :] .*= singfac[ipert]
            @views wv3D[:, ipert] .*= singfac[ipert]
        end

        # DEBUG
        println("2D Vacuum response matrix wv:")
        display(vac.wv)
        println("3D Vacuum response matrix wv3D:")
        display(wv3D)
        println("Maximum relative difference between 2D and 3D vacuum response matrices:")
        display(maximum(abs.(vac.wv .- wv3D)) / maximum(abs.(vac.wv)))
        println("Maximum difference between 2D and 3D vacuum response matrices:")
        display(maximum(abs.(vac.wv .- wv3D)))
        println("Relative difference in maximum eigenvalue:")
        display((maximum(real.(eigvals(vac.wv))) - maximum(real.(eigvals(wv3D)))) / maximum(real.(eigvals(vac.wv))))
        println("Difference in maximum eigenvalue:")
        display((maximum(real.(eigvals(vac.wv))) - maximum(real.(eigvals(wv3D)))))
        error("Vacuum response matrix computation complete.")
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
    wpt .= adjoint(vac.wt) * (wp * vac.wt)
    wvt .= adjoint(vac.wt) * (vac.wv * vac.wt)
    for ipert in 1:intr.numpert_total
        vac.ep[ipert] = wpt[ipert, ipert]
        vac.ev[ipert] = wvt[ipert, ipert]
    end

    # Normalize eigenvectors based on scaled wt
    coeffs = odet.u[:, :, 1, end] \ (vac.wt .* (2π * equil.psio * 1e-3))
    for istep in 1:odet.step
        odet.u_store[:, :, 1, istep] .= odet.u_store[:, :, 1, istep] * coeffs
        odet.u_store[:, :, 2, istep] .= odet.u_store[:, :, 2, istep] * coeffs
        odet.ud_store[:, :, 1, istep] .= odet.ud_store[:, :, 1, istep] * coeffs
        odet.ud_store[:, :, 2, istep] .= odet.ud_store[:, :, 2, istep] * coeffs
    end

    # Write energies to screen
    if ctrl.verbose
        println("Least Stable Eigenmode Energies:")
        println("  Plasma = ", (@sprintf "%+.3e %+.3ei" real(vac.ep[1]) imag(vac.ep[1])))
        println("  Vacuum = ", (@sprintf "%+.3e %+.3ei" real(vac.ev[1]) imag(vac.ev[1])))
        println("  Total  = ", (@sprintf "%+.3e %+.3ei" real(vac.et[1]) imag(vac.et[1])))
    end

    return vac
end

"""
    compute_vacuum_inputs(psifac::Float64, n::Int, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal)

Compute the necessary input paramters to the VACUUM code for computing the vacuum response matrix.
Performs the same function as `free_write_msc` in the Fortran code, except we create a data struct
to pass in memory to the Julia VACUUM code instead of writing to a file. Computed quantities include
the r, z, and ν values at the plasma boundary, as well as mode numbers and number of poloidal points.

### Arguments

  - `ψ`: Flux surface value at the plasma boundary (Float64)
  - `n`: Toroidal mode number (Int)
  - `ctrl`: DCON control parameters (DconControl)
  - `equil`: Plasma equilibrium data (Equilibrium.PlasmaEquilibrium)
  - `intr`: Internal DCON parameters (DconInternal)
"""
function compute_vacuum_inputs(ψ::Float64, n::Int, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal)

    # Allocations
    mtheta = equil.config.control.mtheta + 1
    θ_SFL = zeros(Float64, mtheta)
    R = zeros(Float64, mtheta)
    Z = zeros(Float64, mtheta)
    ν = zeros(Float64, mtheta)
    r_minor = zeros(Float64, mtheta)

    # Compute geometric quantities on plasma boundary
    for (i, θ) in enumerate(equil.rzphi.ys)
        f = Spl.bicube_eval!(equil.rzphi, ψ, θ)
        r_minor[i] = sqrt(f[1])
        θ_SFL[i] = 2π * (θ + f[2]) # f[2] = λ(ψ, θ) / 2π
        ν[i] = f[3]
    end

    # Compute R and Z on straight-fieldline θ grid
    R .= equil.ro .+ r_minor .* cos.(θ_SFL)
    Z .= equil.zo .+ r_minor .* sin.(θ_SFL)

    # Invert values for n < 0 # TODO: move this to VACUUM?
    # This is here because the only thing that changes in 2D for n<0 is (mθ - nν)
    # since Gⁿ = G⁻ⁿ and Kⁿ = K⁻ⁿ? Confirm later and generalize to 3D
    if n < 0
        ν .= -ν
        n = -n
    end

    return Vacuum.VacuumInput(;
        r=reverse(R),
        z=reverse(Z),
        ν=reverse(ν),
        mlow=intr.mlow,
        mpert=intr.mpert,
        n=n,
        mtheta=ctrl.mthvac,
        force_wv_symmetry=ctrl.force_wv_symmetry
    )
end

"""
    free_compute_wv_spline(ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal)

Compute a spline of vacuum response matrices over the range of psi from 'ctrl.psi_edge' to
`intr.qlim`. This is used for fast evaluation of wt during `ode_record_edge`. Performs the
same function as `free_wvmats` in the Fortran code. Currently defaults to 4 spline points per
q-window minimum.
"""
function free_compute_wv_spline(ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal)

    # Number of psi grid points for the spline: 4 per q-window minimum
    # TODO: 4 spline points is arbitrary - is there a better way?
    qedge = Spl.spline_eval!(equil.sq, ctrl.psiedge)[4]
    npsi = max(4, ceil(Int, (intr.qlim - qedge) * intr.nhigh * 4))
    psi_array = zeros(Float64, npsi + 1)
    wv_array = zeros(ComplexF64, npsi + 1, intr.numpert_total, intr.numpert_total)

    for i in 1:(npsi+1)
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

        for ipert_n in 1:intr.npert
            # Compute vacuum matrix
            n = ipert_n - 1 + intr.nlow
            vac_inputs = compute_vacuum_inputs(intr.psilim, n, ctrl, equil, intr)
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

    return Spl.CubicSpline(psi_array, reshape(wv_array, npsi+1, :); bctype="extrap")
end

"""
    free_compute_total(equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::DconInternal, odet::OdeState) -> ComplexF64

Compute total complex energy eigenvalue (total1). This is a trimmed down version of `free_run`
that only computes the total energy eigenvalue for the mode unstable mode, used in `findmax_dW_edge!`
which calls this function at each step in the psiedge -> psilim region of integration. This performs
the same function as `free_test` in the Fortran code, except we have moved the creation of the
wv matrix spline to `free_compute_wv_spline` and pass it in `odet.wvmat_spline`.
"""
function free_compute_total(equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::DconInternal, odet::OdeState)

    # Allocations
    wp = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)
    tot_eigvals = zeros(ComplexF64, intr.numpert_total)
    wt = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)

    dV_dpsi = Spl.spline_eval!(equil.sq, intr.psilim)[3]

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
