# PERTURBED EQUILIBRIUM NONAMBIPOLAR TRANSPORT CODE

"""
Main program. Interfaces with input file, sets corresponding
global variables, and runs requested subprocess.

REVISION HISTORY:
    2014.03.06 -Logan- initial writing.
    2025.11.25 -Fairborn- converted to Julia
    Converted using Claude AI for first pass

AUTHOR: Logan
EMAIL: nikolas.logan@columbia.edu
"""

using Printf

# Include required modules (equivalents to Fortran modules)
include("pentrc_interface.jl")
include("utilities.jl")  # For append_1d, append_2d, progressbar

# Include harvest library (if available in Julia)
include("harvest_lib.jl")

function main()
    # Local variables
    local i, j, k, l, m, nvalid
    local psi_out_valid = Float64[]
    
    # Harvest variables
    local ierr::Int
    local hlog::String = " "^65507
    
    if verbose
        println()
        println("PENTRC START => $version")
        println("_____________________________________________")
    end
    
    initialize_pentrc()
    
    if moment == "heat"
        global qt = true
        if verbose
            println("---------------------------------------------")
            println("Calculating heat transport")
            println("---------------------------------------------")
        end
    elseif moment == "pressure"
        global qt = false
        if verbose
            println("---------------------------------------------")
            println("Calculating particle transport and torque")
            println("---------------------------------------------")
        end
    else
        error("ERROR: Input moment must be 'pressure' or 'heat'")
    end
    
    # Start timer
    timer(mode=0)
    
        # Clear working directory
    if clean
        if verbose
            println("clearing working directory")
        end
        # Use an explicit shell invocation to avoid parse-time errors from
        # unquoted glob characters inside a command literal (e.g. "*").
        run(`bash`, "-c", "rm -f pentrc_*.out")
    end
    
    # Administrative setup/diagnostics/debugging
    if fnml_flag
        set_fymnl()
    end
    if ellip_flag
        set_ellip()
    end
    
    if diag_flag
        diagnose_all()
    else
        # Set the number of threads for multithreading
        if pentrc_threads > 0
            # Julia uses JULIA_NUM_THREADS environment variable
            # or Threads.nthreads() to check available threads
            if pentrc_threads > Threads.nthreads()
                @warn "Requested $pentrc_threads threads but only $(Threads.nthreads()) available"
            end
        else
            pentrc_threads = 1
        end
        
        # Run models
        # Start log with harvest
        ierr = init_harvest("CODEDB_PENT", hlog)
        ierr = set_harvest_verbose(0)
        
        # Standard CODEDB records
        ierr = set_harvest_payload_str(hlog, "CODE", "PENT")
        ierr = set_harvest_payload_str(hlog, "VERSION", version)
        
        # Record PENT input
        ierr = set_harvest_payload_int(hlog, "zi", zi)
        ierr = set_harvest_payload_int(hlog, "zimp", zimp)
        ierr = set_harvest_payload_int(hlog, "mi", mi)
        ierr = set_harvest_payload_int(hlog, "mimp", mimp)
        ierr = set_harvest_payload_bol(hlog, "electron", electron)
        ierr = set_harvest_payload_str(hlog, "nutype", nutype)
        ierr = set_harvest_payload_str(hlog, "f0type", f0type)
        
        # Record dcon equilibrium basics
        idcon_harvest(hlog)
        
        # Explicit matrix calculations
        if wxyz_flag && output_ascii
            if verbose
                println("PENTRC - euler-lagrange matrix calculation")
            end
            
            # HACK - this should have its own flag
            wtw = Array{ComplexF64}(undef, mpert, mpert, 6)
            nstring = @sprintf("%4d", nn)
            
            open("pentrc_tgar_elmat_n$(strip(nstring)).out", "w") do f
                println(f, "PERTURBED EQUILIBRIUM NONAMBIPOLAR TRANSPORT CODE:")
                println(f, "Kinetic additions to the ideal Euler-Lagrange matrices")
                println(f)
                @printf(f, "    n = %4d l = %4d\n\n", nn, 0)
                
                for i in 1:npsi_out
                    if verbose
                        println(" psi = ", psi_out[i])
                    end
                    println(f)
                    @printf(f, " psi = %16.8e\n\n", psi_out[i])
                    @printf(f, " %4s%4s", "m_1", "m_2")
                    for label in ["real(A_k)", "imag(A_k)", "real(B_k)", "imag(B_k)", 
                                  "real(C_k)", "imag(C_k)", "real(D_k)", "imag(D_k)", 
                                  "real(E_k)", "imag(E_k)", "real(H_k)", "imag(H_k)"]
                        @printf(f, " %16s", label)
                    end
                    println(f)
                    
                    for l in 0:0  # Should be all
                        if 0 < psi_out[i] <= 1
                            tpsi(tsurf, psi_out[i], nn, l, zi, mi, wdfac, divxfac, 
                                 electron, "twmm"; op_wmats=wtw)
                            for j in 1:mpert
                                for k in 1:mpert
                                    @printf(f, " %4d%4d", mfac[k], mfac[j])
                                    for val in wtw[k, j, :]
                                        @printf(f, " %16.8e", val)
                                    end
                                    println(f)
                                end
                            end
                        end
                    end
                end
            end
        end
        
        flags = [
            fgar_flag, tgar_flag, pgar_flag, rlar_flag, clar_flag, fcgl_flag,
            fwmm_flag, twmm_flag, pwmm_flag, ftmm_flag, ttmm_flag, ptmm_flag,
            fkmm_flag, tkmm_flag, pkmm_flag, frmm_flag, trmm_flag, prmm_flag
        ]
        
        for m in 1:nmethods
            if flags[m]
                method = methods[m]
                if verbose
                    println("---------------------------------------------")
                    println("$method - $(docs[m])")
                end
                
                if method in ["clar", "rlar"]
                    if isempty(data_dir) || data_dir == "default"
                        data_dir = get(ENV, "GPECHOME", "")
                        if isempty(data_dir)
                            error("ERROR: Use of default data directory requires GPECHOME environment variable")
                        end
                        data_dir = joinpath(data_dir, "pentrc")
                    end
                    read_fnml(joinpath(data_dir, "fkmnl.dat"))
                end
                
                if dynamic_grid
                    if verbose
                        println("$method - Calculating using dynamic integration")
                    end
                    tphi = tintgrl_lsode(psilims, nn, nl, zi, mi, wdfac, divxfac, 
                                         electron, methods[m])
                    if verbose
                        @printf("%-24s%11.3e\n", "Total torque = ", real(tphi))
                        @printf("%-24s%11.3e\n", "Total Kinetic Energy = ", imag(tphi)/(2*nn))
                        @printf("%-24s%11.3e\n", "alpha/s  = ", real(tphi)/(-1*imag(tphi)))
                    end
                    ierr = set_harvest_payload_dbl(hlog, "torque_$method", real(tphi))
                    ierr = set_harvest_payload_dbl(hlog, "deltaW_$method", imag(tphi)/(2*nn))
                end
                
                if equil_grid
                    if verbose
                        println("$method - Calculating on equilibrium grid")
                    end
                    teq = tintgrl_grid("equil", psilims, nn, nl, zi, mi, wdfac, 
                                       divxfac, electron, methods[m])
                    if verbose
                        if dynamic_grid
                            @printf("%-24s%11.3e%-12s%11.3e\n", "Total torque = ", real(teq),
                                    ", % error = ", abs(real(teq)-real(tphi))/real(tphi))
                            @printf("%-24s%11.3e%-12s%11.3e\n", "Total Kinetic Energy = ", 
                                    imag(teq)/(2*nn), ", % error = ", 
                                    abs(imag(teq)-imag(tphi))/imag(tphi))
                            @printf("%-24s%11.3e\n", "alpha/s  = ", real(teq)/(-1*imag(teq)))
                        else
                            @printf("%-24s%11.3e\n", "Total torque = ", real(teq))
                            @printf("%-24s%11.3e\n", "Total Kinetic Energy = ", imag(teq)/(2*nn))
                            @printf("%-24s%11.3e\n", "alpha/s  = ", real(teq)/(-1*imag(teq)))
                        end
                    end
                end
                
                if input_grid
                    if verbose
                        println("$method - Calculating on input displacements' grid")
                    end
                    teq = tintgrl_grid("input", psilims, nn, nl, zi, mi, wdfac, 
                                       divxfac, electron, methods[m])
                    if verbose
                        if dynamic_grid
                            @printf("%-24s%11.3e%-12s%11.3e\n", "Total torque = ", real(teq),
                                    ", % error = ", abs(real(teq)-real(tphi))/real(tphi))
                            @printf("%-24s%11.3e%-12s%11.3e\n", "Total Kinetic Energy = ", 
                                    imag(teq)/(2*nn), ", % error = ", 
                                    abs(imag(teq)-imag(tphi))/imag(tphi))
                            @printf("%-24s%11.3e\n", "alpha/s  = ", real(teq)/(-1*imag(teq)))
                        else
                            @printf("%-24s%11.3e\n", "Total torque = ", real(teq))
                            @printf("%-24s%11.3e\n", "Total Kinetic Energy = ", imag(teq)/(2*nn))
                            @printf("%-24s%11.3e\n", "alpha/s  = ", real(teq)/(-1*imag(teq)))
                        end
                    end
                end
                
                # Run select surfaces with detailed output
                if theta_out || xlmda_out
                    if verbose
                        println("$method - Recalculating on psi_out grid for detailed outputs")
                    end
                    
                    # Only use valid output surfaces
                    psi_out_valid = Float64[]
                    for i in 1:npsi_out
                        if 0 < psi_out[i] < 1
                            push!(psi_out_valid, psi_out[i])
                        end
                    end
                    
                    if !isempty(psi_out_valid)
                        nvalid = length(psi_out_valid)
                        for i in 1:nvalid
                            if nvalid < 10
                                @printf("  psi = %11.3e\n", psi_out_valid[i])
                            end
                            for l in -nl:max(1,nl):nl
                                tpsi(tsurf, psi_out_valid[i], nn, l, zi, mi, wdfac, 
                                     divxfac, electron, methods[m]; 
                                     op_erecord=xlmda_out, op_orecord=theta_out)
                            end
                            if nvalid > 10
                                progressbar(i, 1, nvalid; op_percent=20)
                            end
                        end
                    end
                end
                
                if verbose
                    println("$method - Finished")
                    println("---------------------------------------------")
                end
            end
        end
        
        if output_ascii
            if theta_out
                output_orbit_ascii(nn)
            end
            if xlmda_out
                output_pitch_ascii(nn)
                output_energy_ascii(nn)
            end
        end
        
        if output_netcdf
            output_torque_netcdf(nn, nl, zi, mi, electron, wdfac)
            if theta_out
                output_orbit_netcdf(nn)
            end
            if xlmda_out
                output_pitch_netcdf(nn)
                output_energy_netcdf(nn)
            end
        end
    end
    
    # Send harvest record
    ierr = harvest_send(hlog)
    
    # Display timer and stop
    if verbose
        timer(mode=1)
        println(" PENTRC STOP => normal termination.")
    end
end

# Run main program
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end