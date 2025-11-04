function Main(path::String)

    println("DCON START")
    println("----------------------------------")
    start_time = time()

    # Read input data and set up data structures
    intr = DconInternal(; dir_path=path)
    inputs = TOML.parsefile(joinpath(intr.dir_path, "dcon.toml"))
    ctrl = DconControl(; (Symbol(k) => v for (k, v) in inputs["DCON_CONTROL"])...)
    outp = DconOutput(; (Symbol(k) => v for (k, v) in inputs["DCON_OUTPUT"])...)
    equil = Equilibrium.setup_equilibrium(joinpath(intr.dir_path, "equil.toml"))
    init_files(outp, intr.dir_path)

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

    # TODO: add some data dump to dcon.out from equil here or in setup_equilibrium

    # Compute Mercier and Ballooning stability (if desired)
    # This holds di, dr, h (calculated in mercier_scan), ca1, and ca2 (calculated in ballooning scan)
    locstab_fs = zeros(Float64, length(equil.sq.xs), 5)
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

    # Fit stability data to splines and dump to file
    intr.locstab = Spl.CubicSpline(Vector(equil.sq.xs), locstab_fs; bctype=3)

    # Dump equilibrium data to files
    if outp.write_eqdata_h5
        h5open(joinpath(intr.dir_path, outp.fname_eqdata_h5), "w") do eqdata_h5
            eqdata_h5["psi"] = Vector(equil.sq.xs)
            eqdata_h5["f"] = Vector(equil.sq.fs[:, 1] ./ (2π))
            eqdata_h5["mu0p"] = Vector(equil.sq.fs[:, 2])
            eqdata_h5["dVdpsi"] = Vector(equil.sq.fs[:, 3])
            eqdata_h5["q"] = Vector(equil.sq.fs[:, 4])
            eqdata_h5["di"] = Vector(locstab_fs[:, 1] ./ equil.sq.xs)
            eqdata_h5["dr"] = Vector(locstab_fs[:, 2] ./ equil.sq.xs)
            eqdata_h5["ca1"] = Vector(locstab_fs[:, 4])
        end
    end

    if outp.write_dcon_out
        write_output(outp, :dcon_out, @sprintf("%4s %12s %12s %12s %12s %12s %12s %12s %12s", "ipsi", "psifac", "f", "mu0 p", "dvdpsi", "q", "di", "dr", "ca1"))
        for ipsi in 1:length(equil.sq.xs)
            write_output(outp, :dcon_out,
                @sprintf("%4d %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e",
                    ipsi,
                    equil.sq.xs[ipsi],
                    equil.sq.fs[ipsi, 1] / (2π),
                    equil.sq.fs[ipsi, 2],
                    equil.sq.fs[ipsi, 3],
                    equil.sq.fs[ipsi, 4],
                    locstab_fs[ipsi, 1] / equil.sq.xs[ipsi],
                    locstab_fs[ipsi, 2] / equil.sq.xs[ipsi],
                    locstab_fs[ipsi, 4]
                )
            )
        end
        write_output(outp, :dcon_out, @sprintf("%4s %12s %12s %12s %12s %12s %12s %12s %12s", "ipsi", "psifac", "f", "mu0 p", "dvdpsi", "q", "di", "dr", "ca1"))
    end

    # Find all singular surfaces in the equilibrium
    sing_find!(intr, ctrl, equil)

    # Determine poloidal mode numbers
    if ctrl.cyl_flag
        intr.mlow = ctrl.delta_mlow
        intr.mhigh = ctrl.delta_mhigh
    elseif ctrl.sing_start == 0
        intr.mlow = min(ctrl.nn * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
        intr.mhigh = trunc(Int, ctrl.nn * equil.params.qmax) + ctrl.delta_mhigh
    else
        intr.mmin = Inf # HUGE in Fortran
        for ising in Int(ctrl.sing_start):intr.msing
            intr.mmin = min(intr.mmin, sing[ising].m)
        end
        intr.mlow = intr.mmin - ctrl.delta_mlow
        intr.mhigh = trunc(Int, intr.nn * equil.qmax) + ctrl.delta_mhigh
    end
    intr.mpert = intr.mhigh - intr.mlow + 1
    if ctrl.delta_mband >= intr.mpert
        @warn "Banded matrices not implemented yet, setting delta_mband to 0"
        ctrl.delta_mband = 0
    end
    intr.mband = intr.mpert - 1 - ctrl.delta_mband
    intr.mband = min(max(intr.mband, 0), intr.mpert - 1)

    # Fit equilibrium quantities to Fourier-spline functions.
    if ctrl.mat_flag || ctrl.ode_flag
        if ctrl.verbose
            println("     q0 = $(equil.params.q0), qmin = $(equil.params.qmin), qmax = $(equil.params.qmax), q95 = $(equil.params.q95)")
            println("     set_psilim_via_dmlim = $(ctrl.set_psilim_via_dmlim), dmlim = $(ctrl.dmlim), qlim = $(intr.qlim), psilim = $(intr.psilim)")
            println("     betat = $(equil.params.betat), betan = $(equil.params.betan), betap1 = $(equil.params.betap1)")
            println("     nn = $(ctrl.nn), mlow = $(intr.mlow), mhigh = $(intr.mhigh), mpert = $(intr.mpert), mband = $(intr.mband)")
            println(" Fourier analysis of metric tensor components")
        end

        if outp.write_dcon_out
            write_output(outp, :dcon_out, @sprintf("\n   mlow   mhigh   mpert   mband   nn   lim_fl   dmlim      qlim      psilim"))
            write_output(
                outp,
                :dcon_out,
                @sprintf("%6d %6d %6d %6d %6d %6s %11.3e %11.3e %11.3e",
                    intr.mlow, intr.mhigh, intr.mpert, intr.mband, ctrl.nn,
                    string(ctrl.set_psilim_via_dmlim), ctrl.dmlim, intr.qlim, intr.psilim)
            )
        end

        # Compute metric tensor
        metric = make_metric(equil; mband=intr.mband, fft_flag=ctrl.fft_flag)

        if ctrl.verbose
            println("Computing F, G, and K Matrices")
        end

        # Compute matrices and populate FourFitVars struct
        ffit = make_matrix(equil, ctrl, intr, metric)

        if ctrl.kin_flag
            error("kin_flag not implemented yet")
        end
        sing_scan!(intr, ctrl, equil, ffit, outp)
        if ctrl.kin_flag
            # ksing_find()
        end
    end

    # Integrate Euler-Lagrange Equation
    if ctrl.ode_flag
        if ctrl.verbose
            println("Integrating Euler-Lagrange equation")
        end
        odet = ode_run(ctrl, equil, ffit, intr, outp)
    end

    # Compute free boundary energies
    plasma1 = 0
    vacuum1 = 0
    total1 = 0
    if ctrl.vac_flag && !(ctrl.ksing > 0 && ctrl.ksing <= intr.msing + 1)
        if ctrl.verbose
            println("Computing free boundary energies")
        end
        plasma1, vacuum1, total1 = free_run!(odet, ctrl, equil, ffit, intr, outp; op_netcdf_out=false) # outp.netcdf_out)
    end

    # Output results of fixed-boundary stability calculations
    if ctrl.ode_flag && odet.nzero != 0
        if ctrl.verbose
            println("Fixed-boundary mode unstable for nn = $(ctrl.nn).")
        end
    end

    # Output results of free-boundary stability calculations
    if ctrl.vac_flag && !(ctrl.ksing > 0 && ctrl.ksing <= intr.msing + 1 && outp.bin_sol)
        if real(total1) < 0
            if ctrl.verbose
                println("Free-boundary mode unstable for nn = $(ctrl.nn).")
            end
        else
            if ctrl.verbose
                println("All free-boundary modes stable for nn = $(ctrl.nn).")
            end
        end
    end

    end_time = time() - start_time
    close_files(outp)
    println("----------------------------------")
    println("Run time: $end_time seconds")
    println("Normal termination.")
end