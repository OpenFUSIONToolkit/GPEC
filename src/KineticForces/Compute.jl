"""
    Compute

High-level computation functions for KineticForces.
Orchestrates torque/energy calculations across multiple methods.
"""

"""
    compute_torque_all_methods!(state::KineticForcesState, intr::KineticForcesInternal,
                                ctrl::KineticForcesControl, equil)

Calculate torque/energy for all enabled methods using ODE integration.
For each method, integrates over flux surfaces in `intr.psi_grid` by calling
`tpsi!()` at each surface.  Results accumulate in `state`.

# Arguments
- `state::KineticForcesState`: Accumulates results for all methods
- `intr::KineticForcesInternal`: Internal state with equilibrium data
- `ctrl::KineticForcesControl`: Control parameters specifying which methods to run
- `equil`: PlasmaEquilibrium with 2D interpolants
"""
function compute_torque_all_methods!(state::KineticForcesState, intr::KineticForcesInternal,
                                     ctrl::KineticForcesControl, equil)

    flags = get_method_flags(ctrl)
    psi_grid = intr.psi_grid

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

        # Integrate over flux surfaces
        npsi = length(psi_grid)
        torque_vs_psi = zeros(ComplexF64, npsi)
        energy_vs_psi = zeros(ComplexF64, npsi)
        tpsi_val = Ref{ComplexF64}(0.0 + 0.0im)

        for i in 1:npsi
            tpsi!(tpsi_val, psi_grid[i], ctrl.nn, ctrl.nl, ctrl.zi, ctrl.mi,
                  ctrl.wdfac, ctrl.divxfac, ctrl.electron, method, equil, intr)
            torque_vs_psi[i] = tpsi_val[]
            energy_vs_psi[i] = complex(0.0, imag(tpsi_val[]) / (2 * ctrl.nn))
        end

        # Trapezoidal integration for total torque
        total_torque = ComplexF64(0.0, 0.0)
        for i in 2:npsi
            dpsi = psi_grid[i] - psi_grid[i-1]
            total_torque += 0.5 * (torque_vs_psi[i-1] + torque_vs_psi[i]) * dpsi
        end

        result = MethodResult(;
            method=method,
            nn=ctrl.nn,
            psi_grid=copy(psi_grid),
            torque_vs_psi=torque_vs_psi,
            energy_vs_psi=energy_vs_psi,
            total_torque=total_torque,
            total_energy=complex(0.0, imag(total_torque) / (2 * ctrl.nn))
        )
        state.method_results[method] = result

        if ctrl.verbose
            @printf("%-24s%11.3e\n", "Total torque = ", real(total_torque))
            @printf("%-24s%11.3e\n", "Total Kinetic Energy = ", imag(total_torque) / (2 * ctrl.nn))
            if imag(total_torque) != 0.0
                @printf("%-24s%11.3e\n", "alpha/s  = ", real(total_torque) / (-1 * imag(total_torque)))
            end
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
