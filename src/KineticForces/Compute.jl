"""
    Compute

High-level computation functions for PENTRC.
Orchestrates torque/energy calculations across multiple methods.
Can be called from both Main() and ForceFreeStates kinetic_flag.
"""

"""
    compute_matrix_calculation!(intr::KineticForcesInternal, ctrl::KineticForcesControl, equil)

Compute and output explicit kinetic matrix calculations.
Stores results in output files for visualization/debugging.

# Arguments
- `intr::KineticForcesInternal`: Internal state with method flags
- `ctrl::KineticForcesControl`: Control parameters
- `equil`: PlasmaEquilibrium
"""
function compute_matrix_calculation!(intr::KineticForcesInternal, ctrl::KineticForcesControl, equil)

    if !ctrl.output_ascii
        return
    end

    if ctrl.verbose
        println("PENTRC - euler-lagrange matrix calculation")
    end

    # TODO: Implement matrix calculation via tpsi! with op_wmats
    # This requires the full GAR method to be implemented
end


"""
    compute_torque_all_methods!(intr::KineticForcesInternal, ctrl::KineticForcesControl, equil)

Calculate torque/energy for all enabled methods.
Main computational loop performing integrations.

# Arguments
- `intr::KineticForcesInternal`: Internal state (to be updated with results)
- `ctrl::KineticForcesControl`: Control parameters specifying which methods to run
- `equil`: PlasmaEquilibrium with 2D interpolants
"""
function compute_torque_all_methods!(intr::KineticForcesInternal, ctrl::KineticForcesControl, equil)

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

        if ctrl.verbose
            println("$method - Finished")
            println("---------------------------------------------")
        end
    end
end


"""
    prepare_output(intr::KineticForcesInternal, ctrl::KineticForcesControl)

Prepare output files based on control flags.
Writes results to ASCII files or HDF5.

# Arguments
- `intr::KineticForcesInternal`: Computed results
- `ctrl::KineticForcesControl`: Output configuration
"""
function prepare_output(intr::KineticForcesInternal, ctrl::KineticForcesControl)
    # TODO: Implement output writing
    # ASCII output via Output.jl functions
    # HDF5 output via ctrl.write_outputs_to_HDF5
end


"""
    compute_kinetic_contribution(ctrl::KineticForcesControl, equil, intr::KineticForcesInternal, ffit)::Dict

Compute kinetic contributions for ForceFreeStates when kinetic_flag=true.
This function is called from ForceFreeStates's Main.jl for kinetic EL calculations.

# Arguments
- `ctrl::KineticForcesControl`: PENTRC control parameters
- `equil`: Equilibrium structure from ForceFreeStates
- `intr::KineticForcesInternal`: Internal state with profile interpolants
- `ffit`: Fourier-fitted variables from ForceFreeStates

# Returns
- `Dict`: Dictionary containing kinetic matrix contributions to add to FourFit
         Keys: :fmat_kin, :gmat_kin, :kmat_kin (matrices to add to ForceFreeStates)
"""
function compute_kinetic_contribution(ctrl::KineticForcesControl, equil, intr::KineticForcesInternal, ffit)

    # Create result dictionary
    kinetic_results = Dict(
        :fmat_kin => zeros(ComplexF64, size(ffit.fmat)),
        :gmat_kin => zeros(ComplexF64, size(ffit.gmat)),
        :kmat_kin => zeros(ComplexF64, size(ffit.kmat))
    )

    # TODO: Implement kinetic matrix calculations
    # 1. For each psi in equilibrium grid
    # 2. Call tpsi!() to get torque/energy
    # 3. Convert to ForceFreeStates matrix format
    # 4. Accumulate in kinetic_results

    return kinetic_results
end
