"""
    build_response_operators!(resp; verbose=false)

Build the control-surface response operators of `resp.ffs` onto `resp`, following Fortran gpresp's
resp_index=0 energy method:

  - Build flux matrix from eigenmodes
  - Calculate plasma inductance Lambda (wt0-based, resp_induct_flag=TRUE)
  - Calculate surface inductance L from Green's function
  - Compute permeability P = Lambda * L^{-1} and reluctance
  - Conform them to the root-area-weighted field space for output
"""
function build_response_operators!(resp::PlasmaResponse; verbose::Bool=false)
    ffs = resp.ffs
    equil = ffs.equil
    verbose && @info "Computing plasma response (wt0-based inductance)"

    # Build flux matrix from ForceFreeStates eigenmodes [mode × eigenmode]
    resp.flux_matrix = build_flux_matrix(equil, ffs.solution, ffs)

    # Plasma inductance Lambda (wt0 formula, Fortran resp_induct_flag=TRUE default)
    plasma_inductance = calc_plasma_inductance(ffs.free_boundary.wt0, ffs, equil.psio)

    # Surface inductance L from vacuum surface-current matrix at psilim, block-diagonal in n.
    surface_inductance = calc_surface_inductance(equil, ffs.psilim, ffs.control.mtheta_vacuum, ffs.mlow:ffs.mhigh, ffs.nlow:ffs.nhigh)
    permeability = calc_permeability(plasma_inductance, surface_inductance)

    # Reluctance ϱ = L⁻¹·(Λ† − L)·L⁻¹ (Fortran gpresp_reluct: diff_indmats = CONJG(TRANSPOSE(plas_indmats)) − surf_indmats).
    # Λ (plasma inductance) is not Hermitian — its anti-Hermitian part is the dissipative/torque response — so the adjoint matters.
    L_inv = inv(surface_inductance)
    reluctance = L_inv * (plasma_inductance' - surface_inductance) * L_inv

    # Singular coupling, field reconstruction and the energies operate on the physical control-surface
    # flux Φ_x, so the flux-space operators are kept; only the stored/output ones are conformed to fields.
    resp.plasma_inductance_flux = plasma_inductance
    resp.surface_inductance_flux = surface_inductance
    resp.permeability_flux = permeability

    # Conform the control-surface matrices to the coordinate-invariant root-area-weighted
    # field (b̃) space for output (Pharr 2026). Store the b̃→b̄ operator S = Σ/√A
    # and the scalar surface area A so users can recover the area-weighted field (b̄ = S·b̃) or
    # flux (Φ = A·b̄) — see Utils.jl output docs.
    rootarea_to_area_weight, surface_area = build_control_surface_rootarea_to_area_weight(equil, ffs)
    field_mats = field_space_response_matrices(plasma_inductance, surface_inductance, permeability, reluctance, rootarea_to_area_weight, surface_area)
    resp.plasma_inductance = field_mats.plasma_inductance
    resp.surface_inductance = field_mats.surface_inductance
    resp.permeability = field_mats.permeability
    resp.reluctance = field_mats.reluctance
    resp.rootarea_to_area_weight = rootarea_to_area_weight
    resp.surface_area = surface_area
    return resp
end

"""
    apply_response!(state; verbose=false)

Apply the response operators of `state.response` to the forcing on `state`: Phi_tot = P * Phi_x,
the forcing and response spectra in the three field representations, the control-surface energies
and torque, and the reconstructed ξ and b profiles.
"""
function apply_response!(state::PerturbedEquilibriumState; verbose::Bool=false)
    resp = state.response
    ffs = resp.ffs
    equil = ffs.equil
    solution = ffs.solution
    permeability = resp.permeability_flux
    surface_inductance = resp.surface_inductance_flux
    plasma_inductance = resp.plasma_inductance_flux
    rootarea_to_area_weight = resp.rootarea_to_area_weight
    surface_area = resp.surface_area

    # Forcing and response on the control surface. Flux Φ appears only as a brief internal bridge:
    # forcing arrives as Φ_x, the field reconstruction below consumes Φ_tot, and the b̃ spectra are
    # formed via the conform operator R = S·A (Φ = R·b̃).
    forcing_flux = map_forcing_to_eigenmodes(state.forcing_modes, ffs)
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
    vy = dot(forcing_flux, L_surf_inv * forcing_flux) / 4
    sy = dot(response_flux, L_surf_inv * response_flux) / 4
    state.vacuum_energy = real(vy)
    state.surface_energy = real(sy)
    # Plasma energy py = Σₙ Φₙ†·Λₙₙ⁻¹·Φₙ/4 and torque T = Σₙ −2n·Im(py_n), from T = −2n·Im(δW)
    # [Park 2011 PoP 18 110702, eq. 1]. Only the diagonal n blocks of Λ enter
    py = zero(ComplexF64)
    state.toroidal_torque = 0.0
    for (in, nn) in enumerate(ffs.nlow:ffs.nhigh)
        blk = ((in - 1) * ffs.mpert + 1):(in * ffs.mpert)
        Φ_n = response_flux[blk]
        py_n = dot(Φ_n, inv(plasma_inductance[blk, blk]) * Φ_n) / 4
        py += py_n
        state.toroidal_torque += -2 * nn * imag(py_n)
    end
    state.plasma_energy = real(py)              # Fortran's "total energy" is this pengy

    xi_modes, b_modes = reconstruct_physical_fields(
        response_flux, resp.flux_matrix, solution, equil, ffs,
        ffs.metric, ffs.mats, state.regularization_width
    )

    npsi = size(solution.u_store, 4)
    state.psi_grid = solution.psi_store[1:npsi]
    state.xi_modes = xi_modes
    state.b_modes = b_modes

    b_n_modes, xi_n_modes = compute_b_n_xi_n_modes(
        xi_modes.psi_J, b_modes.psi, solution, equil, ffs
    )
    state.b_n_modes = b_n_modes
    state.xi_n_modes = xi_n_modes

    if verbose
        @info "Response complete: $(length(state.forcing_modes)) forcing modes, max amplitude = $(@sprintf("%.3e", maximum(abs.(response_flux))))"
    end
end
