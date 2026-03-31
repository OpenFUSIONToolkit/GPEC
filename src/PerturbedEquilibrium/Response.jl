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
2. Calculate plasma inductance Lambda (wt0-based, resp_induct_flag=TRUE)
3. Calculate surface inductance L from Green's function
4. Compute permeability P = Lambda * L^{-1}
5. Apply forcing to get response: Phi_tot = P * Phi_x
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
        @info "Computing plasma response (wt0-based inductance)"
    end

    # Build flux matrix from ForceFreeStates eigenmodes [mode × eigenmode]
    flux_matrix = build_flux_matrix(equil, ForceFreeStates_results, vac_data, ffs_intr)

    # Plasma inductance Lambda (wt0 formula, Fortran resp_induct_flag=TRUE default)
    plasma_inductance = calc_plasma_inductance(vac_data, ffs_intr, equil.psio)

    # Surface inductance L from Green's functions at psilim.
    # Requires a 2D (nzvac=1) vacuum response so rows are theta points only,
    # matching Fortran gpeq_surface which uses grri(2*mthsurf, 2*mpert).
    # vac_data.grri has shape [2*mthvac*nzvac, 2*mpert] and cannot be used directly.
    nn = ffs_intr.nlow
    vac_input_2d = Vacuum.VacuumInput(equil, ffs_intr.psilim, vac_data.mthvac, 1,
                                      ffs_intr.mpert, ffs_intr.mlow, 1, nn)
    wall_nowall = Vacuum.WallShapeSettings(; shape="nowall")
    _, grri_2d_raw, grre_2d_raw, _, _ = Vacuum.compute_vacuum_response(
        vac_input_2d, wall_nowall; green_only=true)
    grri_2d = Matrix{Float64}(grri_2d_raw)
    grre_2d = Matrix{Float64}(grre_2d_raw)
    ν_vac = Vacuum.PlasmaGeometry(vac_input_2d).ν
    surface_inductance = compute_surface_inductance_from_greens(grri_2d, grre_2d, ffs_intr, nn, ν_vac)
    permeability = calc_permeability(plasma_inductance, surface_inductance)

    # Reluctance Rho = L^{-1} * (Lambda - L) * L^{-1}  (gpout_resp convention)
    L_inv = inv(surface_inductance)
    reluctance = L_inv * (plasma_inductance - surface_inductance) * L_inv

    # Store control surface matrices in state for HDF5 output
    state.plasma_inductance  = plasma_inductance
    state.surface_inductance = surface_inductance
    state.permeability       = permeability
    state.reluctance         = reluctance

    # Store permeability in internal state for singular coupling use
    intr.plasma_response = permeability

    forcing_vector  = map_forcing_to_eigenmodes(intr.forcing_modes, ffs_intr)
    response_vector = compute_plasma_response_vector(permeability, forcing_vector)

    state.forcing_vec  = forcing_vector
    state.response_vec = response_vector

    xi_modes, b_modes = reconstruct_physical_fields(
        response_vector, flux_matrix, ForceFreeStates_results, equil, ffs_intr, intr
    )

    state.xi_modes = xi_modes
    state.b_modes  = b_modes

    b_n_modes, xi_n_modes = compute_b_n_xi_n_modes(
        xi_modes.psi, b_modes.psi, ForceFreeStates_results, equil, ffs_intr, vac_data.mthvac
    )
    state.b_n_modes  = b_n_modes
    state.xi_n_modes = xi_n_modes

    if ctrl.verbose
        @info "Response complete: $(length(intr.forcing_modes)) forcing modes, max amplitude = $(@sprintf("%.3e", maximum(abs.(response_vector))))"
    end
end
