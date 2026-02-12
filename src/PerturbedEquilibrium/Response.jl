"""
    compute_plasma_response!(
        state::PerturbedEquilibriumState,
        equil::Equilibrium.PlasmaEquilibrium,
        ForceFreeStates_results::OdeState,
        vac_data::VacuumData,
        ffs_intr::ForceFreeStatesInternal,
        intr::PerturbedEquilibriumInternal,
        ctrl::PerturbedEquilibriumControl
    )

Compute plasma response to external forcing using ForceFreeStates eigenmode solutions.

Implements resp_index=0 calculation from gpresp.f:
1. Build flux matrix from eigenmodes
2. Calculate plasma inductance (energy-based)
3. Calculate surface inductance from Green's function
4. Compute permeability matrix
5. Apply forcing to get response
"""
function compute_plasma_response!(
    state::PerturbedEquilibriumState,
    equil::Equilibrium.PlasmaEquilibrium,
    ForceFreeStates_results::OdeState,
    vac_data::VacuumData,
    ffs_intr::ForceFreeStatesInternal,
    intr::PerturbedEquilibriumInternal,
    ctrl::PerturbedEquilibriumControl
)
    if ctrl.verbose
        println("Computing plasma response using resp_index=0 (energy-based inductance)")
    end

    # Build flux matrix from ForceFreeStates eigenmodes
    if ctrl.verbose
        println("  Building flux matrix from eigenmodes")
    end
    flux_matrix = build_flux_matrix(equil, ForceFreeStates_results, vac_data, ffs_intr)

    # Calculate plasma inductance matrix
    if ctrl.verbose
        println("  Calculating plasma inductance matrix")
    end
    plasma_inductance = calc_plasma_inductance(flux_matrix, vac_data.et)

    # Calculate surface inductance from Green's functions
    if ctrl.verbose
        println("  Calculating surface inductance from Green's functions")
    end
    surface_inductance = calc_surface_inductance(vac_data.grri, vac_data.grre, flux_matrix, ffs_intr)

    # Calculate permeability matrix
    if ctrl.verbose
        println("  Calculating permeability matrix")
    end
    permeability = calc_permeability(plasma_inductance, surface_inductance)

    # Store permeability in internal state for later use
    intr.plasma_response = permeability

    # Map forcing modes to eigenmode basis
    if ctrl.verbose
        println("  Mapping forcing modes to eigenmode basis")
        println("    Number of forcing modes: $(length(intr.forcing_modes))")
    end
    forcing_vector = map_forcing_to_eigenmodes(intr.forcing_modes, ffs_intr)

    # Compute plasma response
    if ctrl.verbose
        println("  Computing response = permeability * forcing")
    end
    response_vector = compute_plasma_response_vector(permeability, forcing_vector)

    # Reconstruct physical fields from response in mode space
    if ctrl.verbose
        println("  Reconstructing physical fields from response (mode space)")
    end

    xi_modes, b_modes = reconstruct_physical_fields(
        response_vector, ForceFreeStates_results, equil, ffs_intr, intr
    )

    state.xi_modes = xi_modes
    state.b_modes = b_modes

    if ctrl.verbose
        println("  Response calculation complete")
        println("    Response vector size: $(length(response_vector))")
        println("    Max response amplitude: $(maximum(abs.(response_vector)))")
        println("    Mode-space field dimensions: $(size(xi_modes.psi))")
        println("    ξ_ψ max amplitude: $(maximum(abs.(xi_modes.psi)))")
        println("    b^ψ max amplitude: $(maximum(abs.(b_modes.psi)))")
    end
end
