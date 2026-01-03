function Main(path::String="./")

    println("DCON START")
    println("----------------------------------")
    start_time = time()

    # Read input data and set up data structures
    intr = DconInternal(; dir_path=path)
    # TODO: leaving DCON_CONTROL as a part of the toml file, eventually can combine equil, gpec, etc. into one input file?
    inputs = TOML.parsefile(joinpath(intr.dir_path, "dcon.toml"))
    ctrl = DconControl(; (Symbol(k) => v for (k, v) in inputs["DCON_CONTROL"])...)
    equil = Equilibrium.setup_equilibrium(joinpath(intr.dir_path, "equil.toml"))
    if "WALL" in keys(inputs)
        wall_settings = Vacuum.WallShapeSettings(; (Symbol(k) => v for (k, v) in inputs["WALL"])...)
    else
        wall_settings = Vacuum.WallShapeSettings()
    end
    if "DEBUG" in keys(inputs)
        debug_settings = DebugSettings(; (Symbol(k) => v for (k, v) in inputs["DEBUG"])...)
    else
        debug_settings = DebugSettings()
    end
    intr.debug_settings = debug_settings
    # Set up variables
    # TODO: dcon_kin_threads logic?
    ctrl.delta_mhigh *= 2 # for consistency with Fortran DCON TODO: why is this present in the Fortran?

    # Determine psilim and qlim (where we will integrate to)
    sing_lim!(intr, ctrl, equil)
    if ctrl.set_psilim_via_dmlim && ctrl.psiedge < intr.psilim
        @warn "Only one of set_psilim_via_dmlim and psiedge < psilim can be used at a time.
            Setting psiedge = 1.0 and determining dW from psilim = $(intr.psilim) determined from dmlim = $(ctrl.dmlim)."
        ctrl.psiedge = 1.0
    end

    # If truncating before psihigh, reform equilibrium if desired
    if intr.psilim != equil.config.control.psihigh && ctrl.reform_eq_with_psilim
        @warn "Reforming equilibrium splines from psihigh to psilim not implemented yet. Proceeding with psihigh = $(equil.config.control.psihigh)."
        # JMH - Nik please put the logic we discussed here
        # something like ?
        # equil.config.control.psihigh = intr.psilim
        # equil = set_up_equilibrium(equil.config)
    end

    # Compute Mercier and Ballooning stability (if desired)
    # This holds di, dr, h (calculated in mercier_scan), ca1, and ca2 (calculated in ballooning scan)
    locstab_fs = zeros(Float64, length(equil.psi_grid), 5)
    if ctrl.mer_flag
        if ctrl.verbose
            println("Evaluating Mercier criterion")
        end
        mercier_scan!(locstab_fs, equil)
    end
    # TODO: ballooning stability
    #IF(bal_flag)THEN
    #   IF(ctrl.verbose) WRITE(*,*)"Evaluating ballooning criterion"
    #   CALL bal_scan
    #ENDIF
    # Fit data to splines
    intr.locstab = Spl.CubicSpline(Vector(equil.psi_grid), locstab_fs; bctype="extrap")

    # Determine toroidal mode numbers
    if ctrl.nn_low == 0 && ctrl.nn_high == 0
        error("Either nn_low or nn_high must be set in DCON_CONTROL (both are 0)")
    elseif ctrl.nn_low == 0
        ctrl.nn_low = ctrl.nn_high
    elseif ctrl.nn_high == 0
        ctrl.nn_high = ctrl.nn_low
    end
    if ctrl.nn_low > ctrl.nn_high
        error("nn_low cannot be greater than nn_high")
    end
    intr.nlow = ctrl.nn_low
    intr.nhigh = ctrl.nn_high
    intr.npert = intr.nhigh - intr.nlow + 1
    nstring = intr.npert == 1 ? "$(intr.nlow)" : "$(intr.nlow):$(intr.nhigh)"

    # Find all singular surfaces in the equilibrium
    sing_find!(intr, equil)

    # Determine poloidal mode numbers
    if ctrl.delta_mlow < 0 || ctrl.delta_mhigh < 0
        error("Negative delta_mlow or delta_mhigh not allowed")
    end
    if ctrl.cyl_flag
        intr.mlow = ctrl.delta_mlow
        intr.mhigh = ctrl.delta_mhigh
    elseif ctrl.sing_start == 0
        intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
        intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
    else
        intr.mmin = Inf # HUGE in Fortran
        for ising in Int(ctrl.sing_start):intr.msing
            intr.mmin = min(intr.mmin, sing[ising].m)
        end
        intr.mlow = intr.mmin - ctrl.delta_mlow
        intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
    end
    intr.mpert = intr.mhigh - intr.mlow + 1
    if ctrl.delta_mband >= intr.mpert
        @warn "Banded matrices not implemented yet, setting delta_mband to 0"
        ctrl.delta_mband = 0
    end
    intr.mband = intr.mpert - 1 - ctrl.delta_mband
    intr.mband = min(max(intr.mband, 0), intr.mpert - 1)
    intr.numpert_total = intr.mpert * intr.npert

    # Fit equilibrium quantities to Fourier-spline functions.
    if ctrl.mat_flag || ctrl.ode_flag
        if ctrl.verbose
            println("Run parameters:")
            println("   q0 = $(@sprintf("%.3f", equil.params.q0)), qmin = $(@sprintf("%.3f", equil.params.qmin)), qmax = $(@sprintf("%.3f", equil.params.qmax)), q95 = $(@sprintf("%.3f", equil.params.q95))")
            println("   qlim = $(@sprintf("%.5f", intr.qlim)), psilim = $(@sprintf("%.9f", intr.psilim))")
            println("   betat = $(@sprintf("%.3f", equil.params.betat)), betan = $(@sprintf("%.3f", equil.params.betan)), betap1 = $(@sprintf("%.3f", equil.params.betap1))")
            println("   mlow = $(@sprintf("%4i", intr.mlow)), mhigh = $(@sprintf("%4i", intr.mhigh)), mpert = $(@sprintf("%4i", intr.mpert)), mband = $(@sprintf("%4i", intr.mband))")
            println("   nlow = $(@sprintf("%4i", intr.nlow)), nhigh = $(@sprintf("%4i", intr.nhigh)), npert = $(@sprintf("%4i", intr.npert))")
        end

        # Compute metric tensor
        metric = make_metric(equil; mband=intr.mband)

        if ctrl.verbose
            println("   Computing F, G, and K Matrices")
        end

        # Compute matrices and populate FourFitVars struct
        ffit = make_matrix(equil, intr, metric)

        if ctrl.kin_flag
            error("kin_flag not implemented yet")
        end
        sing_scan!(intr, ctrl, equil, ffit)
        if ctrl.kin_flag
            # ksing_find()
        end
    end

    # Integrate Euler-Lagrange Equation
    if ctrl.ode_flag
        if ctrl.verbose
            println("Integrating Euler-Lagrange equation")
        end
        odet = ode_run(ctrl, equil, ffit, intr)
        if odet.nzero > 0 && ctrl.verbose
            println("Fixed-boundary mode unstable for n = $nstring.")
        end
    end

    # Compute free boundary energies
    if ctrl.vac_flag && !(ctrl.ksing > 0 && ctrl.ksing <= intr.msing + 1)
        if ctrl.verbose
            println("Computing free boundary energies")
        end
        vac_data = free_run!(odet, ctrl, equil, ffit, intr, wall_settings)
        if real(vac_data.et[1]) < 0
            if ctrl.verbose
                println("Free-boundary mode unstable for n = $nstring.")
            end
        else
            if ctrl.verbose
                println("All free-boundary modes stable for n = $nstring.")
            end
        end
    end

    if ctrl.write_outputs_to_HDF5
        if ctrl.verbose
            println("Writing saved data to $(ctrl.HDF5_filename)")
        end
        write_outputs_to_HDF5(ctrl, equil, intr, odet, ctrl.vac_flag ? vac_data : nothing)
    end

    end_time = time() - start_time
    println("----------------------------------")
    println("Run time: $(@sprintf("%.3e", end_time)) seconds") 
    println("Normal termination.")

    # TODO: Do not allow perturbed equilibrium calculations if zero crossings are found
end

"""
    write_outputs_to_HDF5(ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal, odet::OdeState)

Helper function to write the HDF5 output file with relevant run and equilibrium parameters.
This combines the functionality of several pieces of the Fortran code in `ode_output.f`,
primarily `ode_output_open` and the various `bin_euler` writes that occur throughout the
integration. Some parameters are only dumped in their respective flags are true, e.g.
vacuum data if `vac_flag` is true.

### TODOs

Combine spline unpacking if possible, too many extra lines

"""
function write_outputs_to_HDF5(ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal, odet::OdeState, vac::Union{VacuumData, Nothing})

    h5open(joinpath(intr.dir_path, ctrl.HDF5_filename), "w") do out_h5

        # Store input parameters
        for (key, val) in zip(fieldnames(DconControl), getfield.(Ref(ctrl), fieldnames(DconControl)))
            out_h5["input/DCON_CONTROL/$key"] = val
        end
        for (key, val) in zip(fieldnames(Equilibrium.EquilibriumControl), getfield.(Ref(equil.config.control), fieldnames(Equilibrium.EquilibriumControl)))
            out_h5["input/EQUIL_CONTROL/$key"] = val
        end
        # TODO: assuming EQUIL_OUTPUT is going to be deprecated
        # TODO: should we store the equilibrium? difficult since it could be a gfile, sol.in, etc.
        # TODO: if we do one input file, can just pass that in instead and loop easily since its parsed
        # as a dict already (for (k, v) in inputs["DCON_CONTROL"]...). We have to do this since custom structs
        # don't inherently have an iterator by default

        # Write derived run parameters
        out_h5["info/mpert"] = intr.mpert
        out_h5["info/mband"] = intr.mband
        out_h5["info/mlow"] = intr.mlow
        out_h5["info/mhigh"] = intr.mhigh
        out_h5["info/npert"] = intr.npert
        out_h5["info/nlow"] = intr.nlow
        out_h5["info/nhigh"] = intr.nhigh
        m = [(i - 1) % intr.mpert + intr.mlow for i in 1:(intr.numpert_total)]
        n = [(i - 1) ÷ intr.mpert + intr.nlow for i in 1:(intr.numpert_total)]
        out_h5["info/mn_index"] = hcat(m, n)   # (N, 2) matrix
        out_h5["info/psilim"] = intr.psilim
        out_h5["info/qlim"] = intr.qlim
        out_h5["info/q1lim"] = intr.q1lim

        # Write derived equilibrium parameters
        for (key, val) in zip(fieldnames(Equilibrium.EquilibriumParameters), getfield.(Ref(equil.params), fieldnames(Equilibrium.EquilibriumParameters)))
            if val !== nothing # TODO: looks like ro, zo, psio, and b_norm are not set, so skipping those for now but should fix eventually
                out_h5["equil/$key"] = val
            end
        end
        out_h5["equil/psio"] = equil.psio
        out_h5["equil/ro"] = equil.ro
        out_h5["equil/zo"] = equil.zo

        # Write spline arrays
        out_h5["splines/sq/xs"] = Vector(equil.psi_grid)
        # Write spline function values at grid points
        out_h5["splines/sq/fs/2piF"] = equil.F_values
        out_h5["splines/sq/fs/mu0p"] = equil.P_values
        out_h5["splines/sq/fs/dVdpsi"] = equil.dVdpsi_values
        out_h5["splines/sq/fs/q"] = equil.q_values
        # Write spline first derivatives at grid points
        out_h5["splines/sq/fs1/2piF"] = [ForwardDiff.derivative(equil.F_spline, psi) for psi in equil.psi_grid]
        out_h5["splines/sq/fs1/mu0p"] = [ForwardDiff.derivative(equil.P_spline, psi) for psi in equil.psi_grid]
        out_h5["splines/sq/fs1/dVdpsi"] = [ForwardDiff.derivative(equil.dVdpsi_spline, psi) for psi in equil.psi_grid]
        out_h5["splines/sq/fs1/q"] = [ForwardDiff.derivative(equil.q_spline, psi) for psi in equil.psi_grid]
        out_h5["splines/sq/xpower"] = 0 # TODO: equil.sq.xpower

        # Write rzphi spline data by evaluating new splines on grid
        out_h5["splines/rzphi/xs"] = Vector(equil.psi_grid)
        out_h5["splines/rzphi/ys"] = Vector(equil.theta_grid)

        # Evaluate splines on grid to get fs arrays
        mpsi = length(equil.psi_grid)
        mtheta = length(equil.theta_grid)
        r2_fs = [equil.r2_spline(equil.psi_grid[i], equil.theta_grid[j]) for i in 1:mpsi, j in 1:mtheta]
        eta_fs = [equil.eta_spline(equil.psi_grid[i], equil.theta_grid[j]) for i in 1:mpsi, j in 1:mtheta]
        nu_fs = [equil.nu_spline(equil.psi_grid[i], equil.theta_grid[j]) for i in 1:mpsi, j in 1:mtheta]
        jac_fs = [equil.jac_spline(equil.psi_grid[i], equil.theta_grid[j]) for i in 1:mpsi, j in 1:mtheta]

        out_h5["splines/rzphi/fs/rcoords"] = r2_fs
        out_h5["splines/rzphi/fs/offset"] = eta_fs
        out_h5["splines/rzphi/fs/nu"] = nu_fs
        out_h5["splines/rzphi/fs/jac"] = jac_fs

        # Compute psi derivatives
        r2_fsx = [ForwardDiff.derivative(p -> equil.r2_spline(p, equil.theta_grid[j]), equil.psi_grid[i]) for i in 1:mpsi, j in 1:mtheta]
        eta_fsx = [ForwardDiff.derivative(p -> equil.eta_spline(p, equil.theta_grid[j]), equil.psi_grid[i]) for i in 1:mpsi, j in 1:mtheta]
        nu_fsx = [ForwardDiff.derivative(p -> equil.nu_spline(p, equil.theta_grid[j]), equil.psi_grid[i]) for i in 1:mpsi, j in 1:mtheta]
        jac_fsx = [ForwardDiff.derivative(p -> equil.jac_spline(p, equil.theta_grid[j]), equil.psi_grid[i]) for i in 1:mpsi, j in 1:mtheta]

        out_h5["splines/rzphi/fsx/rcoords"] = r2_fsx
        out_h5["splines/rzphi/fsx/offset"] = eta_fsx
        out_h5["splines/rzphi/fsx/nu"] = nu_fsx
        out_h5["splines/rzphi/fsx/jac"] = jac_fsx

        # Compute theta derivatives
        r2_fsy = [ForwardDiff.derivative(t -> equil.r2_spline(equil.psi_grid[i], t), equil.theta_grid[j]) for i in 1:mpsi, j in 1:mtheta]
        eta_fsy = [ForwardDiff.derivative(t -> equil.eta_spline(equil.psi_grid[i], t), equil.theta_grid[j]) for i in 1:mpsi, j in 1:mtheta]
        nu_fsy = [ForwardDiff.derivative(t -> equil.nu_spline(equil.psi_grid[i], t), equil.theta_grid[j]) for i in 1:mpsi, j in 1:mtheta]
        jac_fsy = [ForwardDiff.derivative(t -> equil.jac_spline(equil.psi_grid[i], t), equil.theta_grid[j]) for i in 1:mpsi, j in 1:mtheta]

        out_h5["splines/rzphi/fsy/rcoords"] = r2_fsy
        out_h5["splines/rzphi/fsy/offset"] = eta_fsy
        out_h5["splines/rzphi/fsy/nu"] = nu_fsy
        out_h5["splines/rzphi/fsy/jac"] = jac_fsy

        # Compute mixed derivatives (not currently used, set to zero)
        out_h5["splines/rzphi/fsxy/rcoords"] = zeros(mpsi, mtheta)
        out_h5["splines/rzphi/fsxy/offset"] = zeros(mpsi, mtheta)
        out_h5["splines/rzphi/fsxy/nu"] = zeros(mpsi, mtheta)
        out_h5["splines/rzphi/fsxy/jac"] = zeros(mpsi, mtheta)

        out_h5["splines/rzphi/x0"] = 0
        out_h5["splines/rzphi/y0"] = 0
        out_h5["splines/rzphi/xpower"] = 0
        out_h5["splines/rzphi/fpower"] = 0

        # Write local stability data
        if ctrl.mer_flag
            out_h5["locstab/di"] = Vector(intr.locstab.fs[:, 1] ./ equil.psi_grid)
            out_h5["locstab/dr"] = Vector(intr.locstab.fs[:, 2] ./ equil.psi_grid)
            out_h5["singular/di0"] = [Spl.spline_eval!(intr.locstab, sing.psifac)[1] / sing.psifac for sing in intr.sing]
        end
        if ctrl.bal_flag
            out_h5["locstab/ca1"] = Vector(locstab_fs[:, 4])
        end

        # Write integration data
        # TODO: technically this should only be written if ode_flag is true, but that's going to get deprecated eventually
        out_h5["integration/nstep"] = odet.step
        out_h5["integration/psi"] = odet.psi_store
        out_h5["integration/q"] = odet.q_store
        out_h5["integration/xi_psi"] = odet.u_store[:, :, 1, :]
        out_h5["integration/u2"] = odet.u_store[:, :, 2, :] # TODO: what to name this? These are the "conjugate momenta" of u1
        out_h5["integration/dxi_psi"] = odet.ud_store[:, :, 1, :]
        out_h5["integration/xi_s"] = odet.ud_store[:, :, 2, :]
        out_h5["integration/crit"] = odet.crit_store

        # Write singular surface data
        out_h5["singular/msing"] = intr.msing
        out_h5["singular/psi"] = [sing.psifac for sing in intr.sing]
        out_h5["singular/q"] = [sing.q for sing in intr.sing]
        out_h5["singular/q1"] = [sing.q1 for sing in intr.sing]
        out_h5["singular/di"] = [sing.di for sing in intr.sing]
        out_h5["singular/ca_left"] = odet.ca_l
        out_h5["singular/ca_right"] = odet.ca_r

        # Write vacuum Data
        if ctrl.vac_flag
            out_h5["vacuum/wt"] = vac.wt
            out_h5["vacuum/wt0"] = vac.wt0
            out_h5["vacuum/ep"] = vac.ep
            out_h5["vacuum/ev"] = vac.ev
            out_h5["vacuum/et"] = vac.et
            out_h5["vacuum/x_plasma"] = vac.xzpts[:, 1]
            out_h5["vacuum/z_plasma"] = vac.xzpts[:, 2]
            out_h5["vacuum/x_wall"] = vac.xzpts[:, 3]
            out_h5["vacuum/z_wall"] = vac.xzpts[:, 4]
        end
    end
end