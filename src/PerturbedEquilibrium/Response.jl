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
        @info "Computing plasma response (energy-based inductance)"
    end

    # Build flux matrix from ForceFreeStates eigenmodes
    flux_matrix = build_flux_matrix(equil, ForceFreeStates_results, vac_data, ffs_intr)
    plasma_inductance = calc_plasma_inductance(flux_matrix, vac_data.et)
    surface_inductance = calc_surface_inductance(vac_data.grri, vac_data.grre, flux_matrix, ffs_intr)
    permeability = calc_permeability(plasma_inductance, surface_inductance)

    # Store permeability in internal state for later use
    intr.plasma_response = permeability

    forcing_vector = map_forcing_to_eigenmodes(intr.forcing_modes, ffs_intr)
    response_vector = compute_plasma_response_vector(permeability, forcing_vector)

    xi_modes, b_modes = reconstruct_physical_fields(
        response_vector, ForceFreeStates_results, equil, ffs_intr, intr
    )

    state.xi_modes = xi_modes
    state.b_modes = b_modes

    if ctrl.verbose
        @info "Response complete: $(length(intr.forcing_modes)) forcing modes, max amplitude = $(@sprintf("%.3e", maximum(abs.(response_vector))))"
    end
end
