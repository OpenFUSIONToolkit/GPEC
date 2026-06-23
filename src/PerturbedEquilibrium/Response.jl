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
    _, grri_2d_raw, grre_2d_raw, _, _ = Vacuum.compute_vacuum_response(vac_input_2d, wall_nowall)
    grri_2d = Matrix{Float64}(grri_2d_raw)
    grre_2d = Matrix{Float64}(grre_2d_raw)
    ν_vac = Vacuum.PlasmaGeometry(vac_input_2d).ν
    surface_inductance = compute_surface_inductance_from_greens(grri_2d, grre_2d, ffs_intr, nn, ν_vac)
    permeability = calc_permeability(plasma_inductance, surface_inductance)

    # Reluctance ϱ = L⁻¹·(Λ† − L)·L⁻¹ (Fortran gpresp_reluct: diff_indmats = CONJG(TRANSPOSE(plas_indmats)) − surf_indmats).
    # Λ (plasma inductance) is not Hermitian — its anti-Hermitian part is the dissipative/torque response — so the adjoint matters.
    L_inv = inv(surface_inductance)
    reluctance = L_inv * (plasma_inductance' - surface_inductance) * L_inv

    # Store permeability in internal state for singular coupling / field reconstruction.
    # These consumers operate on the physical control-surface flux Φ_x, so the internal
    # copy stays in flux space; only the stored/output quantities are conformed to fields below.
    intr.plasma_response = permeability

    # Conform the control-surface matrices to the coordinate-invariant root-area-weighted
    # field (b̃) space for output (issue #233 / Pharr 2026). Store the b̃→b̄ operator S = Σ/√A
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

    # Forcing and response on the control surface. Flux Φ appears only as a brief internal bridge:
    # forcing arrives as Φ_x, the field reconstruction below consumes Φ_tot, and the b̃ spectra are
    # formed via the conform operator R = S·A (Φ = R·b̃).
    forcing_flux = map_forcing_to_eigenmodes(intr.forcing_modes, ffs_intr)
    response_flux = compute_plasma_response_vector(permeability, forcing_flux)

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

    if ctrl.verbose
        @info "Response complete: $(length(intr.forcing_modes)) forcing modes, max amplitude = $(@sprintf("%.3e", maximum(abs.(response_flux))))"
    end
end
