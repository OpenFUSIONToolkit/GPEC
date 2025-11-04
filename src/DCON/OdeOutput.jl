"""
    ode_output_init(ctrl, equil, outp, intr, odet)

Write header info to output files at the start of the integration.
Performs similar output writing as the Fortran `ode_output_open`,
except we no longer need to open binary files.

### TODOs

Remove deprecated outputs
Combine spline unpacking if possible, too many extra lines
Replace `println` statements with logging to appropriate output files
Remove `mthsurf0` if deprecated
Place outputs in a Julia do loop for automatic file closing
"""
function ode_output_init(ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal, odet::OdeState, outp::DconOutput)

    # TODO: mess with this to condense the number of calls? Maybe allow it to pass in dicts

    # Write euler.h5 header info
    if outp.write_euler_h5
        h5open(joinpath(intr.dir_path, outp.fname_euler_h5), "w") do euler_h5
            # Write DCON run parameters
            euler_h5["info/mpert"] = intr.mpert
            euler_h5["info/mband"] = intr.mband
            euler_h5["info/mlow"] = intr.mlow
            euler_h5["info/mhigh"] = intr.mhigh
            euler_h5["info/nn"] = ctrl.nn
            euler_h5["info/singfac_min"] = ctrl.singfac_min
            euler_h5["info/kin_flag"] = ctrl.kin_flag
            euler_h5["info/con_flag"] = ctrl.con_flag
            euler_h5["info/mthvac"] = ctrl.mthvac
            euler_h5["info/mthsurf0"] = outp.mthsurf0 #TODO: mthsurf0 is deprecated

            # Write equilibrium parameters
            euler_h5["equil/nr"] = length(equil.rzphi.xs) # TODO: equil save mpsi as really mpsi - 1, fix this
            euler_h5["equil/nz"] = length(equil.rzphi.ys)
            euler_h5["equil/ro"] = equil.ro
            euler_h5["equil/zo"] = equil.zo
            euler_h5["equil/amean"] = equil.params.amean
            euler_h5["equil/rmean"] = equil.params.rmean
            euler_h5["equil/aratio"] = equil.params.aratio
            euler_h5["equil/kappa"] = equil.params.kappa
            euler_h5["equil/delta1"] = equil.params.delta1
            euler_h5["equil/delta2"] = equil.params.delta2
            euler_h5["equil/li1"] = equil.params.li1
            euler_h5["equil/li2"] = equil.params.li2
            euler_h5["equil/li3"] = equil.params.li3
            euler_h5["equil/betap1"] = equil.params.betap1
            euler_h5["equil/betap2"] = equil.params.betap2
            euler_h5["equil/betap3"] = equil.params.betap3
            euler_h5["equil/betat"] = equil.params.betat
            euler_h5["equil/betan"] = equil.params.betan
            euler_h5["equil/bt0"] = equil.params.bt0
            euler_h5["equil/q0"] = equil.params.q0
            euler_h5["equil/q95"] = equil.params.q95
            euler_h5["equil/qmin"] = equil.params.qmin
            euler_h5["equil/qmax"] = equil.params.qmax
            euler_h5["equil/qa"] = equil.params.qa
            euler_h5["equil/crnt"] = equil.params.crnt
            euler_h5["equil/psio"] = equil.psio
            euler_h5["equil/psilow"] = equil.config.control.psilow
            euler_h5["equil/power_b"] = equil.config.control.power_b
            euler_h5["equil/power_r"] = equil.config.control.power_r
            euler_h5["equil/power_bp"] = equil.config.control.power_bp
            euler_h5["equil/shotnum"] = 0 # TODO: equil.params.shotnum
            euler_h5["equil/shottime"] = 0 # TODO: equil.params.shottime

            # Write spline arrays
            euler_h5["splines/sq/xs"] = Vector(equil.sq.xs)
            # TODO: getting errors when trying to dump just fs, so splitting for now, which adds so many lines
            # This should be fixed if we separate these like Nik mentioned
            euler_h5["splines/sq/fs/2piF"] = equil.sq.fs[:, 1]
            euler_h5["splines/sq/fs/mu0p"] = equil.sq.fs[:, 2]
            euler_h5["splines/sq/fs/dVdpsi"] = equil.sq.fs[:, 3]
            euler_h5["splines/sq/fs/q"] = equil.sq.fs[:, 4]
            euler_h5["splines/sq/fs1/2piF"] = equil.sq.fs1[:, 1]
            euler_h5["splines/sq/fs1/mu0p"] = equil.sq.fs1[:, 2]
            euler_h5["splines/sq/fs1/dVdpsi"] = equil.sq.fs1[:, 3]
            euler_h5["splines/sq/fs1/q"] = equil.sq.fs1[:, 4]
            euler_h5["splines/sq/xpower"] = 0 # TODO: equil.sq.xpower
            euler_h5["splines/rzphi/xs"] = Vector(equil.rzphi.xs)
            euler_h5["splines/rzphi/ys"] = Vector(equil.rzphi.ys)
            euler_h5["splines/rzphi/fs/rcoords"] = equil.rzphi.fs[:, 1]
            euler_h5["splines/rzphi/fs/offset"] = equil.rzphi.fs[:, 2]
            euler_h5["splines/rzphi/fs/nu"] = equil.rzphi.fs[:, 3]
            euler_h5["splines/rzphi/fs/jac"] = equil.rzphi.fs[:, 4]
            euler_h5["splines/rzphi/fsx/rcoords"] = equil.rzphi.fsx[:, 1]
            euler_h5["splines/rzphi/fsx/offset"] = equil.rzphi.fsx[:, 2]
            euler_h5["splines/rzphi/fsx/nu"] = equil.rzphi.fsx[:, 3]
            euler_h5["splines/rzphi/fsx/jac"] = equil.rzphi.fsx[:, 4]
            euler_h5["splines/rzphi/fsy/rcoords"] = equil.rzphi.fsy[:, 1]
            euler_h5["splines/rzphi/fsy/offset"] = equil.rzphi.fsy[:, 2]
            euler_h5["splines/rzphi/fsy/nu"] = equil.rzphi.fsy[:, 3]
            euler_h5["splines/rzphi/fsy/jac"] = equil.rzphi.fsy[:, 4]
            euler_h5["splines/rzphi/fsxy/rcoords"] = equil.rzphi.fsxy[:, 1]
            euler_h5["splines/rzphi/fsxy/offset"] = equil.rzphi.fsxy[:, 2]
            euler_h5["splines/rzphi/fsxy/nu"] = equil.rzphi.fsxy[:, 3]
            euler_h5["splines/rzphi/fsxy/jac"] = equil.rzphi.fsxy[:, 4]
            euler_h5["splines/rzphi/x0"] = 0 # TODO: equil.rzphi.x0
            euler_h5["splines/rzphi/y0"] = 0 # TODO: equil.rzphi.y0
            euler_h5["splines/rzphi/xpower"] = 0 # TODO: equil.rzphi.xpower
            euler_h5["splines/rzphi/fpower"] = 0 # TODO: equil.rzphi.fpower
        end
    end

    # Write crit.out header
    if odet.ising > 0 && odet.ising <= intr.msing
        singp = intr.sing[odet.ising]
        if outp.write_crit_out
            write_output(outp, :crit_out, @sprintf("   ising   psi         q          di      re alpha   im alpha\n"))
            write_output(outp, :crit_out, @sprintf("%6d%11.3e%11.3e%11.3e%11.3e%11.3e\n", odet.ising, singp.psifac, singp.q, singp.di, real(singp.alpha), imag(singp.alpha)))
        end
    end
    if outp.write_crit_out
        write_output(outp, :crit_out, "    psifac      dpsi        q       singfac     eval1\n")
    end
end

"""
    ode_output_step(ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal, odet::OdeState, outp::DconOutput)

Performs output and monitoring tasks at each integration step by calling
`ode_output_monitor`. to track critical eigenvalue behavior. Unlike Fortran,
we don't implement any output file dumping here, as Julia's in-memory storage
structure handles this differently.

### TODOs

Depending on output needs, may need to implement `ode_output_get_evals` and `ode_output_sol`
Determine if this function is even necessary of if we should just call `ode_output_monitor` directly
"""
#TODO: depending on how we restructure our outputs, this function might be uncessary
# (i.e. if we don't need an ode_output_get_evals or ode_output_sol, can just replace calls with ode_output_monitor)
function ode_output_step(ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal, odet::OdeState, outp::DconOutput)

    # Compute and print critical data for each time step
    ode_output_monitor(ctrl, equil, intr, odet, outp)
    # if dout.out_evals || dout.bin_evals
    #     ode_output_get_evals(intr, ctrl, dout, fNames, equil, odet) # this is just outputs
    # end

    # All bin_euler logic performed after integration

    # if outp.bin_sol
    #     ode_output_sol()
    # end
end

# TODO: I don't think this function is essentially for the minimal working version - can convert later if needed
# function ode_output_get_evals(intr::DconInternal, ctrl::DconControl, dout::DconOutput, fNames::DconFileNames, equil::Equilibrium.PlasmaEquilibrium, odet::OdeState)
#     return
# end

"""
    ode_output_monitor(ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal, odet::OdeState, outp::DconOutput)

Monitor the evolution of a critical eigenvalue (`crit`) during integration
using `ode_output_get_crit` and detect zero crossings, which indicate instability.
If a sign change is found, it estimates the crossing point via linear interpolation.
If the crossing satisfies sharpness and consistency conditions, it's logged and
`nzero` is incremented. Performs the same function as `ode_output_monitor` in the
Fortran code, with small differences in output handling.

### TODO

Restore or redirect output to appropriate logging units.
Replace `error(...)` with graceful shutdown if zero crossing is an intended exit condition.
All the _prev variables can probably be removed and the logic can be simplified to just take the odet.step - 1 values when needed
"""
function ode_output_monitor(ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal, odet::OdeState, outp::DconOutput)

    # Compute new crit
    dpsi = odet.psifac - odet.psi_prev
    q, singfac, logpsi1, logsingfac, crit = ode_output_get_crit(odet.psifac, odet.u, intr.mpert, odet.m1, ctrl.nn, equil.sq)

    # Check for zero crossing
    if crit * odet.crit_prev < 0 # if crit changes sign, zero crossing has occurred
        fac = crit / (crit - odet.crit_prev)
        psi_med = odet.psifac - fac * (odet.psifac - odet.psi_prev)
        dpsi = psi_med - odet.psi_prev
        u_med = odet.u .- fac .* (odet.u .- odet.u_prev)
        q_med, singfac_med, logpsi1_med, logsingfac_med, crit_med = ode_output_get_crit(psi_med, u_med, intr.mpert, odet.m1, ctrl.nn, equil.sq)

        if (crit_med - crit) * (crit_med - odet.crit_prev) < 0 &&
           abs(crit_med) < 0.5 * min(abs(crit), abs(odet.crit_prev))
            println("Zero crossing detected at psi = $psi_med, q = $q_med")
            if outp.write_dcon_out
                write_output(outp, :dcon_out, @sprintf("Zero crossing at psi = %10.3e, q = %10.3e", psi_med, q_med))
            end
            if outp.write_crit_out
                write_output(outp, :crit_out, @sprintf("Zero crossing at psi = %10.3e, q = %10.3e", psi_med, q_med))
                write_output(outp, :crit_out, @sprintf("%11.3e%11.3e%11.3e%11.3e%11.3e", psi_med, dpsi, q_med, singfac_med, crit_med))
            end
            odet.nzero += 1
        end
        if ctrl.termbycross_flag
            #TODO: not sure if this is really an error, or just a user specified termination condition. So the program shoudln't error out, just stop integrating
            # In fortran, these were both handled by program_stop()
            error("Terminated by zero crossing.")
        end
    end

    # Write new crit
    if outp.write_crit_out
        write_output(outp, :crit_out, @sprintf("%11.3e%11.3e%11.3e%11.3e%11.3e", odet.psifac, dpsi, odet.q, singfac, crit))
    end

    # Update saved values
    odet.psi_prev = odet.psifac
    odet.crit_prev = crit
    odet.u_prev .= odet.u
end

"""
    ode_output_get_crit(psi, u, mpert, m1, nn, sq) -> (q, singfac, logpsi1, logsingfac, crit)

Compute critical quantities at a given flux surface and using the smallest (in magnitude)
inverse eigenvalue in combination with the equilibrium profiles to form `crit`.
Performs the same function as `ode_output_get_crit` in the Fortran code.

### Arguments

  - `psi::Float64`: The flux surface at which to evaluate.
  - `u::Array{ComplexF64, 3}`: Solution matrix at `psi`.
  - `mpert::Int`: Number of poloidal modes considered.
  - `m1::Int`: Poloidal mode number of the perturbation.
  - `nn::Int`: Toroidal mode number of the perturbation.
  - `sq::Spl.CubicSpline`: Spline object containing equilibrium profiles.

### Returns

A tuple `(q, singfac, logpsi1, logsingfac, crit)` of critical data for the time step.

### TODOs

Decide if we want to just pass in the relevant quantities instead of structs for functions like this
"""
function ode_output_get_crit(psi::Float64, u::Array{ComplexF64,3}, mpert::Int, m1::Int, nn::Int, sq::Spl.CubicSpline)

    # Compute inverse plasma response matrix
    wp = adjoint(u[:, 1:mpert, 1])         # adjoint = conjugate transpose
    temp = adjoint(u[:, 1:mpert, 2])

    # Compute wp using LU decomposition
    temp_fact = lu(temp)
    wp = temp_fact \ wp

    # Symmetrize
    wp .+= adjoint(wp)
    wp .*= 0.5

    # Compute and sort inverse eigenvalues
    evalsi = eigvals!(Hermitian(wp))
    indexi = sortperm(evalsi; by=abs)  # bubble in Fortran sorts in descending order of -|evalsi|, we just do ascending order of |evalsi|

    # Compute critical data for each time step
    profiles = Spl.spline_eval!(sq, psi)
    q = profiles[4]
    singfac = abs(m1 - nn * profiles[4])
    logpsi1 = log10(psi)
    logsingfac = log10(singfac)
    crit = evalsi[indexi[1]] * profiles[3]^2

    return q, singfac, logpsi1, logsingfac, crit
end