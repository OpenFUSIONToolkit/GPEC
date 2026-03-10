"""
    Compute

High-level computation functions for PENTRC.
Orchestrates torque/energy calculations across multiple methods.
Can be called from both Main() and DCON kinetic_flag.
"""

"""
    compute_matrix_calculation!(intr::PentrcInternal, ctrl::PentrcControl)

Compute and output explicit kinetic matrix calculations (wxyz_flag).
Stores results in output files for visualization/debugging.

# Arguments
- `intr::PentrcInternal`: Internal state with method flags
- `ctrl::PentrcControl`: Control parameters
"""
function compute_matrix_calculation!(intr::PentrcInternal, ctrl::PentrcControl)
    
    if !ctrl.wxyz_flag
        return
    end
    
    if !ctrl.output_ascii
        return
    end
    
    if ctrl.verbose
        println("PENTRC - euler-lagrange matrix calculation")
    end
    
    # Allocate matrix storage: [m_1, m_2, 6 matrix components]
    wtw = Array{ComplexF64}(undef, 0, 0, 0)  # Will be resized as needed
    
    nstring = @sprintf("%4d", ctrl.nn)
    fname = joinpath(ctrl.data_dir, "pentrc_tgar_elmat_n$(strip(nstring)).out")
    
    open(fname, "w") do f
        println(f, "PERTURBED EQUILIBRIUM NONAMBIPOLAR TRANSPORT CODE:")
        println(f, "Kinetic additions to the ideal Euler-Lagrange matrices")
        println(f)
        @printf(f, "    n = %4d l = %4d\n\n", ctrl.nn, 0)
        
        for i in 1:length(ctrl.psi_out)
            if ctrl.verbose
                println(" psi = ", ctrl.psi_out[i])
            end
            println(f)
            @printf(f, " psi = %16.8e\n\n", ctrl.psi_out[i])
            @printf(f, " %4s%4s", "m_1", "m_2")
            
            label_strs = ["real(A_k)", "imag(A_k)", "real(B_k)", "imag(B_k)", 
                          "real(C_k)", "imag(C_k)", "real(D_k)", "imag(D_k)", 
                          "real(E_k)", "imag(E_k)", "real(H_k)", "imag(H_k)"]
            for label in label_strs
                @printf(f, " %16s", label)
            end
            println(f)
            
            # TODO: Implement matrix calculation via tpsi!
            # for l in 0:0  
            #     if 0 < ctrl.psi_out[i] <= 1
            #         tpsi!(tsurf, ctrl.psi_out[i], ctrl.nn, l, 
            #              ctrl.zi, ctrl.mi, ctrl.wdfac, ctrl.divxfac, 
            #              ctrl.electron, "twmm"; op_wmats=wtw)
            #         # Write matrix elements to file
            #     end
            # end
        end
    end
end


"""
    compute_torque_all_methods!(intr::PentrcInternal, ctrl::PentrcControl)

Calculate torque/energy for all enabled methods.
Main computational loop performing integrations.

# Arguments
- `intr::PentrcInternal`: Internal state (to be updated with results)
- `ctrl::PentrcControl`: Control parameters specifying which methods to run
"""
function compute_torque_all_methods!(intr::PentrcInternal, ctrl::PentrcControl)
    
    # Get method flags in correct order
    flags = get_method_flags(ctrl)
    
    for m in 1:length(intr.methods)
        
        if !flags[m]
            continue
        end
        
        method = intr.methods[m]
        intr.method = method
        
        if ctrl.verbose
            println("---------------------------------------------")
            println("$method - $(intr.docs[m])")
        end
        
        # Load fnml data if needed for RLAR/CLAR methods
        if method in ["clar", "rlar"]
            if isempty(ctrl.data_dir) || ctrl.data_dir == "default"
                data_dir = get(ENV, "GPECHOME", "")
                if isempty(data_dir)
                    error("ERROR: Use of default data directory requires GPECHOME environment variable")
                end
                data_dir = joinpath(data_dir, "pentrc")
            else
                data_dir = ctrl.data_dir
            end
            read_fnml(joinpath(data_dir, "fkmnl.dat"))
        end
        
        # Dynamic grid integration
        if ctrl.dynamic_grid
            if ctrl.verbose
                println("$method - Calculating using dynamic integration")
            end
            tphi = tintgrl_lsode(ctrl.psilims, ctrl.nn, ctrl.nl, ctrl.zi, ctrl.mi, 
                                 ctrl.wdfac, ctrl.divxfac, ctrl.electron, method)
            intr.tphi = tphi
            
            if ctrl.verbose
                @printf("%-24s%11.3e\n", "Total torque = ", real(tphi))
                @printf("%-24s%11.3e\n", "Total Kinetic Energy = ", imag(tphi)/(2*ctrl.nn))
                @printf("%-24s%11.3e\n", "alpha/s  = ", real(tphi)/(-1*imag(tphi)))
            end
        end
        
        # Equilibrium grid integration
        if ctrl.equil_grid
            if ctrl.verbose
                println("$method - Calculating on equilibrium grid")
            end
            teq = tintgrl_grid("equil", ctrl.psilims, ctrl.nn, ctrl.nl, ctrl.zi, ctrl.mi,
                              ctrl.wdfac, ctrl.divxfac, ctrl.electron, method)
            intr.teq = teq
            
            if ctrl.verbose
                if ctrl.dynamic_grid
                    @printf("%-24s%11.3e%-12s%11.3e\n", "Total torque = ", real(teq),
                            ", % error = ", abs(real(teq)-real(intr.tphi))/real(intr.tphi))
                    @printf("%-24s%11.3e%-12s%11.3e\n", "Total Kinetic Energy = ", 
                            imag(teq)/(2*ctrl.nn), ", % error = ", 
                            abs(imag(teq)-imag(intr.tphi))/imag(intr.tphi))
                else
                    @printf("%-24s%11.3e\n", "Total torque = ", real(teq))
                    @printf("%-24s%11.3e\n", "Total Kinetic Energy = ", imag(teq)/(2*ctrl.nn))
                end
                @printf("%-24s%11.3e\n", "alpha/s  = ", real(teq)/(-1*imag(teq)))
            end
        end
        
        # Input grid integration
        if ctrl.input_grid
            if ctrl.verbose
                println("$method - Calculating on input displacements' grid")
            end
            teq = tintgrl_grid("input", ctrl.psilims, ctrl.nn, ctrl.nl, ctrl.zi, ctrl.mi,
                              ctrl.wdfac, ctrl.divxfac, ctrl.electron, method)
            intr.teq = teq
            
            if ctrl.verbose
                if ctrl.dynamic_grid
                    @printf("%-24s%11.3e%-12s%11.3e\n", "Total torque = ", real(teq),
                            ", % error = ", abs(real(teq)-real(intr.tphi))/real(intr.tphi))
                    @printf("%-24s%11.3e%-12s%11.3e\n", "Total Kinetic Energy = ", 
                            imag(teq)/(2*ctrl.nn), ", % error = ", 
                            abs(imag(teq)-imag(intr.tphi))/imag(intr.tphi))
                else
                    @printf("%-24s%11.3e\n", "Total torque = ", real(teq))
                    @printf("%-24s%11.3e\n", "Total Kinetic Energy = ", imag(teq)/(2*ctrl.nn))
                end
                @printf("%-24s%11.3e\n", "alpha/s  = ", real(teq)/(-1*imag(teq)))
            end
        end
        
        # Detailed output at selected surfaces
        if ctrl.theta_out || ctrl.xlmda_out
            if ctrl.verbose
                println("$method - Recalculating on psi_out grid for detailed outputs")
            end
            
            # Filter to valid output surfaces
            intr.psi_out_valid = Float64[]
            for psi in ctrl.psi_out
                if 0 < psi < 1
                    push!(intr.psi_out_valid, psi)
                end
            end
            intr.nvalid = length(intr.psi_out_valid)
            
            if intr.nvalid > 0
                for i in 1:intr.nvalid
                    if intr.nvalid < 10
                        @printf("  psi = %11.3e\n", intr.psi_out_valid[i])
                    end
                    
                    for l in -ctrl.nl:max(1,ctrl.nl):ctrl.nl
                        # TODO: Call tpsi! with op_erecord, op_orecord flags
                        # tpsi!(tsurf, intr.psi_out_valid[i], ctrl.nn, l,
                        #       ctrl.zi, ctrl.mi, ctrl.wdfac, ctrl.divxfac,
                        #       ctrl.electron, method;
                        #       op_erecord=ctrl.xlmda_out, op_orecord=ctrl.theta_out)
                    end
                    
                    if intr.nvalid > 10
                        progressbar(i, 1, intr.nvalid; op_percent=20)
                    end
                end
            end
        end
        
        if ctrl.verbose
            println("$method - Finished")
            println("---------------------------------------------")
        end
    end
end


"""
    prepare_output(intr::PentrcInternal, ctrl::PentrcControl)

Prepare output files based on control flags.
Writes results to ASCII files, NetCDF, or both.

# Arguments
- `intr::PentrcInternal`: Computed results
- `ctrl::PentrcControl`: Output configuration
"""
function prepare_output(intr::PentrcInternal, ctrl::PentrcControl)
    
    if ctrl.output_ascii
        if ctrl.theta_out
            # TODO: output_orbit_ascii(ctrl.nn)
        end
        if ctrl.xlmda_out
            # TODO: output_pitch_ascii(ctrl.nn)
            # TODO: output_energy_ascii(ctrl.nn)
        end
    end
    
    if ctrl.output_netcdf
        # TODO: output_torque_netcdf(ctrl.nn, ctrl.nl, ctrl.zi, ctrl.mi, 
        #                             ctrl.electron, ctrl.wdfac)
        if ctrl.theta_out
            # TODO: output_orbit_netcdf(ctrl.nn)
        end
        if ctrl.xlmda_out
            # TODO: output_pitch_netcdf(ctrl.nn)
            # TODO: output_energy_netcdf(ctrl.nn)
        end
    end
end


"""
    compute_kinetic_contribution(ctrl::PentrcControl, equil, ffit)::Dict

Compute kinetic contributions for DCON when kinetic_flag=true.
This function is called from DCON's Main.jl for kinetic EL calculations.

# Arguments
- `ctrl::PentrcControl`: PENTRC control parameters
- `equil`: Equilibrium structure from DCON
- `ffit`: Fourier-fitted variables from DCON

# Returns
- `Dict`: Dictionary containing kinetic matrix contributions to add to FourFit
         Keys: :fmat_kin, :gmat_kin, :kmat_kin (matrices to add to DCON)
"""
function compute_kinetic_contribution(ctrl::PentrcControl, equil, ffit)
    
    # Create result dictionary
    kinetic_results = Dict(
        :fmat_kin => zeros(ComplexF64, size(ffit.fmat)),
        :gmat_kin => zeros(ComplexF64, size(ffit.gmat)),
        :kmat_kin => zeros(ComplexF64, size(ffit.kmat))
    )
    
    # TODO: Implement kinetic matrix calculations
    # 1. For each psi in equilibrium grid
    # 2. Call tpsi!() to get torque/energy
    # 3. Convert to DCON matrix format
    # 4. Accumulate in kinetic_results
    
    return kinetic_results
end
