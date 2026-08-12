"""
    compute_plasma_response!(
        state, equil, ForceFreeStates_results, vac_data, ffs_intr,
        intr, ctrl, metric, ffit
    )

Compute plasma response to external forcing using ForceFreeStates eigenmode solutions.

Implements resp_index=0 calculation from Fortran gpresp:

  - Build flux matrix from eigenmodes
  - Calculate plasma inductance Lambda (wt0-based, resp_induct_flag=TRUE)
  - Calculate surface inductance L from Green's function
  - Compute permeability P = Lambda * L^{-1}
  - Apply forcing to get response: Phi_tot = P * Phi_x
"""
function compute_plasma_response!(
    state::PerturbedEquilibriumState,
    equil::Equilibrium.PlasmaEquilibrium,
    ForceFreeStates_results::OdeState,
    vac_data::VacuumData,
    ffs_intr::ForceFreeStatesInternal,
    intr::PerturbedEquilibriumInternal,
    ctrl::PerturbedEquilibriumControl,
    metric::MetricData,
    ffit::FourFitVars
)
    ctrl.verbose && @info "Computing plasma response"

    # Build flux matrix from ForceFreeStates eigenmodes [mode × eigenmode]
    flux_matrix = build_flux_matrix(equil, ForceFreeStates_results, ffs_intr)

    # Compute plasma inductance
    plasma_inductance = calc_plasma_inductance(ffs_intr, vac_data.wt0, equil.psio)

    # Surface inductance L from vacuum surface-current matrix at psilim
    nn = ffs_intr.nlow
    vac_input_2d = Vacuum.VacuumInput(equil, ffs_intr.psilim, vac_data.mthvac, 1, ffs_intr.mlow:ffs_intr.mhigh, [nn])
    _, I_v, _, _ = Vacuum.compute_vacuum_response(vac_input_2d, wall_nowall; compute_Iv=true)
    surface_inductance = calc_surface_inductance(I_v)

    # Compute permeability P = Λ·L⁻¹ and store in internal state for singular coupling / field reconstruction.
    permeability = plasma_inductance / surface_inductance
    intr.plasma_response = permeability

    # Compute reluctance ϱ = L⁻¹·(Λ† − L)·L⁻¹
    # Λ is not Hermitian — its anti-Hermitian part is the dissipative/torque response — so the adjoint matters.
    L_inv = inv(surface_inductance)
    reluctance = L_inv * (plasma_inductance' - surface_inductance) * L_inv

    # Conform the control-surface matrices to the coordinate-invariant root-area-weighted
    # field (b̃) space for output (Pharr 2026). Store the b̃→b̄ operator S = Σ/√A
    # and the scalar surface area A so users can recover the area-weighted field (b̄ = S·b̃) or
    # flux (Φ = A·b̄) — see Utils.jl output docs.
    rootarea_to_area_weight, surface_area = build_control_surface_rootarea_to_area_weight(equil, ffs_intr)
    field_mats = field_space_response_matrices(plasma_inductance, surface_inductance, permeability, reluctance, rootarea_to_area_weight, surface_area)
    state.plasma_inductance = field_mats.plasma_inductance
    state.surface_inductance = field_mats.surface_inductance
    state.permeability = field_mats.permeability
    state.reluctance = field_mats.reluctance
    state.rootarea_to_area_weight = rootarea_to_area_weight
    state.surface_area = surface_area

    # Compute actual flux from external flux and permiability, Φ = P Φ^x
    forcing_flux = map_forcing_to_eigenmodes(intr.forcing_modes, ffs_intr)
    response_flux = permeability * forcing_flux

    # Output forcing/response in the three Pharr field representations (all tesla):
    #   b̃ (root-area-weighted) = R⁻¹·Φ,  b (bare) = Σ⁻¹·b̃,  b̄ (area-weighted) = S·b̃.
    flux_conform = rootarea_to_area_weight .* surface_area          # R = Σ·√A  (b̃ → flux)
    bare_to_rootarea = rootarea_to_area_weight .* sqrt(surface_area)    # Σ        (bare → b̃)
    forcing_b_rootarea = flux_conform \ forcing_flux
    response_b_rootarea = flux_conform \ response_flux
    state.forcing_b_rootarea = forcing_b_rootarea
    state.response_b_rootarea = response_b_rootarea
    state.forcing_b = bare_to_rootarea \ forcing_b_rootarea
    state.response_b = bare_to_rootarea \ response_b_rootarea
    state.forcing_b_area = rootarea_to_area_weight * forcing_b_rootarea
    state.response_b_area = rootarea_to_area_weight * response_b_rootarea

    # Scalar energies and torque (Joules), ported from Fortran gpout. These are congruence-invariant
    # (energy = Φ†·G⁻¹·Φ = b̃†·G̃⁻¹·b̃), so they are evaluated from the brief internal flux vectors with
    # the well-conditioned flux-space inductances — the b̃ form routes through inv(R⁻¹LR⁻†) and is
    # needlessly ill-conditioned. The result is a physical scalar, not a stored flux quantity.
    L_surf_inv = inv(surface_inductance)
    L_plas_inv = inv(plasma_inductance)
    vy = dot(forcing_flux, L_surf_inv * forcing_flux) / 4
    sy = dot(response_flux, L_surf_inv * response_flux) / 4
    py = dot(response_flux, L_plas_inv * response_flux) / 4
    state.vacuum_energy = real(vy)
    state.surface_energy = real(sy)
    state.plasma_energy = real(py)              # Fortran's "total energy" is this pengy
    state.toroidal_torque = -2 * nn * imag(py)

    xi_modes, b_modes = reconstruct_physical_fields(
        response_flux, flux_matrix, ForceFreeStates_results, equil, ffs_intr, intr,
        metric, ffit, ctrl
    )

    npsi = size(ForceFreeStates_results.u_store, 4)
    state.psi_grid = ForceFreeStates_results.psi_store[1:npsi]
    state.xi_modes = xi_modes
    state.b_modes = b_modes

    b_n_modes, xi_n_modes = compute_b_n_xi_n_modes(
        xi_modes.psi_J, b_modes.psi, ForceFreeStates_results, equil, ffs_intr
    )
    state.b_n_modes = b_n_modes
    state.xi_n_modes = xi_n_modes

    ctrl.verbose && @info "Response complete: $(length(intr.forcing_modes)) forcing modes, max amplitude = $(@sprintf("%.3e", maximum(abs.(response_flux))))"
end
