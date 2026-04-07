"""
    Compute

High-level computation functions for KineticForces.
Orchestrates torque/energy calculations across multiple methods.
"""

"""
    compute_torque_all_methods!(state::KineticForcesState, intr::KineticForcesInternal,
                                ctrl::KineticForcesControl, equil)

Calculate torque/energy for all enabled methods using ODE integration.
Results accumulate in `state`.

# Arguments
- `state::KineticForcesState`: Accumulates results for all methods
- `intr::KineticForcesInternal`: Internal state with equilibrium data
- `ctrl::KineticForcesControl`: Control parameters specifying which methods to run
- `equil`: PlasmaEquilibrium with 2D interpolants
"""
function compute_torque_all_methods!(state::KineticForcesState, intr::KineticForcesInternal,
                                     ctrl::KineticForcesControl, equil)

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
            println("$method - Calculating using dynamic integration")
        end

        tphi = tintgrl_lsode(ctrl.psilims, ctrl.nn, ctrl.nl, ctrl.zi, ctrl.mi,
                              ctrl.wdfac, ctrl.divxfac, ctrl.electron, method)
        intr.tphi = tphi

        # Store result in state
        result = MethodResult(;
            method=method,
            nn=ctrl.nn,
            total_torque=tphi,
            total_energy=complex(0.0, imag(tphi) / (2 * ctrl.nn))
        )
        state.method_results[method] = result

        if ctrl.verbose
            @printf("%-24s%11.3e\n", "Total torque = ", real(tphi))
            @printf("%-24s%11.3e\n", "Total Kinetic Energy = ", imag(tphi) / (2 * ctrl.nn))
            @printf("%-24s%11.3e\n", "alpha/s  = ", real(tphi) / (-1 * imag(tphi)))
            println("$method - Finished")
            println("---------------------------------------------")
        end
    end

    state.completed = true
end


"""
    compute_kinetic_contribution(ctrl::KineticForcesControl, equil,
                                 intr::KineticForcesInternal, ffit)::Dict

Compute kinetic contributions for ForceFreeStates when kinetic_flag=true.
Called from ForceFreeStates for kinetic Euler-Lagrange calculations.

# Arguments
- `ctrl::KineticForcesControl`: Control parameters
- `equil`: Equilibrium structure from ForceFreeStates
- `intr::KineticForcesInternal`: Internal state with profile interpolants
- `ffit`: Fourier-fitted variables from ForceFreeStates

# Returns
- `Dict`: Kinetic matrix contributions (:fmat_kin, :gmat_kin, :kmat_kin)
"""
function compute_kinetic_contribution(ctrl::KineticForcesControl, equil,
                                      intr::KineticForcesInternal, ffit)

    kinetic_results = Dict(
        :fmat_kin => zeros(ComplexF64, size(ffit.fmat)),
        :gmat_kin => zeros(ComplexF64, size(ffit.gmat)),
        :kmat_kin => zeros(ComplexF64, size(ffit.kmat))
    )

    # TODO: Implement kinetic matrix calculations
    return kinetic_results
end
