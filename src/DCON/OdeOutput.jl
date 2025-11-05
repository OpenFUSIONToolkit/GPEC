"""
    ode_output_init(ctrl, equil, outp, intr, odet)

Write header info to output files at the start of the integration.
Performs similar output writing as the Fortran `ode_output_open`,
except we no longer need to open binary files.

### TODOs

Remove deprecated outputs
Combine spline unpacking if possible, too many extra lines
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
end

function evaluate_stability_criterion(ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal, odet::OdeState, outp::DconOutput)

    # Initialization
    if ctrl.verbose
        println("Evaluating stability criterion over entire integration...")
    end
    nzero = 0
    crit_store = zeros(Float64, odet.step)

    # Loop over integration steps, computing crit/checking for zero crossings
    for istep in 1:odet.step
        zero_cross = check_for_zero_crossings!(crit_store, odet, equil.sq, istep)
        if zero_cross
            nzero += 1
        end
    end

    # Write crit values to HDF5 file
    if outp.write_crit_h5
        if ctrl.verbose
            println("   Writing crit.h5 file...")
        end
        h5open(joinpath(intr.dir_path, outp.fname_crit_h5), "w") do crit_h5
            crit_h5["psi"] = odet.psi_store
            crit_h5["q"] = odet.q_store
            crit_h5["crit"] = crit_store
        end
    end
    return nzero
end

"""
    check_for_zero_crossings!(crit_store, odet, sq, istep) -> zero_cross

Monitor the evolution of a critical eigenvalue (`crit`) during integration
using `ode_output_get_crit` and detect zero crossings, which indicate instability.
If a sign change is found, it estimates the crossing point via linear interpolation.
If the crossing satisfies sharpness and consistency conditions, it's logged and
`nzero` is incremented. Performs the same function as `ode_output_monitor` in the
Fortran code, with small differences in output handling.

# TODO: update this once I'm done

Updates crit_store in place
"""
function check_for_zero_crossings!(crit_store::Vector{Float64}, odet::OdeState, sq::Spl.CubicSpline{Float64}, istep::Int)

    # Compute smallest eigenvalue (crit) at current step
    psi = odet.psi_store[istep]
    u = odet.u_store[:, :, :, istep]
    crit_store[istep] = compute_smallest_eigenvalue(psi, u, sq)

    # Check for zero crossing via change in sign of crit between current and previous step
    zero_cross = false
    if istep > 1 && crit_store[istep] * crit_store[istep - 1] < 0
        crit = crit_store[istep]
        crit_prev = crit_store[istep - 1]
        # Ensure the zero crossing is physical and not just numerical noise
        fac = crit / (crit - crit_prev)
        psi_mid = psi - fac * (psi - odet.psi_store[istep - 1])
        u_mid = u .- fac .* (u .- @view(odet.u_store[:, :, :, istep - 1]))
        crit_mid = compute_smallest_eigenvalue(psi_mid, u_mid, sq)
        if (crit_mid - crit) * (crit_mid - crit_prev) < 0 && abs(crit_mid) < 0.5 * min(abs(crit), abs(crit_prev))
            zero_cross = true
            println("Zero crossing detected at psi = $psi_mid, q = $q_mid")
        end
    end
    return zero_cross
end

"""
    compute_smallest_eigenvalue(psi, u, sq) -> crit

Compute critical quantities at a given flux surface and using the smallest (in magnitude)
inverse eigenvalue in combination with the equilibrium profiles to form `crit`.
Performs the same function as `ode_output_get_crit` in the Fortran code.

    TODO: update this once I'm done

### Arguments

  - `psi::Float64`: The flux surface at which to evaluate
  - `u::Array{ComplexF64, 3}`: Solution matrix at `psi`
  - `sq::Spl.CubicSpline`: Spline object containing equilibrium profiles

### Returns

  - `crit::Float64`: the computed scaled critical eigenvalue

"""
function compute_smallest_eigenvalue(psi::Float64, u::Array{ComplexF64,3}, sq::Spl.CubicSpline{Float64})

    # Compute inverse plasma response matrix
    # TODO: is this actually the inverse?
    wp_inverse = adjoint(u[:, :, 1])
    temp = adjoint(u[:, :, 2])
    wp_inverse = temp \ wp_inverse

    # Symmetrize to be Hermitian
    wp_inverse .+= adjoint(wp_inverse)
    wp_inverse .*= 0.5

    # Compute inverse eigenvalues, sort, and return the smallest (with a scale factor)
    evalsi = eigvals!(Hermitian(wp_inverse))
    indexi = sortperm(evalsi; by=abs)
    dVdpsi = Spl.spline_eval!(sq, psi)[3]
    return evalsi[indexi[1]] * dVdpsi^2
end