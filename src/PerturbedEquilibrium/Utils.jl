import ..Utilities: save_to_hdf5
import ..Utilities.HDF5Annotations: ensure_group, replace_group!, write_fields!

"""
    mode_arrays(ffs::ForceFreeStatesResult) -> (m_modes, n_modes)

Mode numbers of each linear index i ∈ [1, numpert_total], in the ForceFreeStates ordering
(m1,n1), (m2,n1), ..., (mpert,n1), (m1,n2), ...:

  - m_modes[i] = (i-1) % mpert + mlow
  - n_modes[i] = (i-1) ÷ mpert + nlow
"""
function mode_arrays(ffs::ForceFreeStatesResult)
    idx = 1:ffs.numpert_total
    return [(i - 1) % ffs.mpert + ffs.mlow for i in idx], [(i - 1) ÷ ffs.mpert + ffs.nlow for i in idx]
end

"""
    save_to_hdf5(file::Union{HDF5.File,HDF5.Group}, state::PerturbedEquilibriumState)

Write perturbed equilibrium results to HDF5 file (appends to existing ForceFreeStates output).

## Output Structure

```
PerturbedEquilibrium/
├── mode_m / mode_n  # (m, n) of each entry along every `mode` axis below [numpert_total]
├── ForcingModes/
│   ├── n              # Toroidal mode numbers
│   ├── m              # Poloidal mode numbers
│   └── amplitude      # ComplexF64 forcing amplitudes
├── forcing_b / forcing_b_root_area / forcing_b_area      # control-surface forcing spectrum (b, b̃, b̄) [numpert_total], tesla
├── response_b / response_b_root_area / response_b_area   # control-surface response spectrum (b, b̃, b̄) [numpert_total], tesla
├── Response/
│   ├── psi             # Radial abscissa ψ_N [npsi] shared by every response profile below
│   ├── xi_psi         # Radial displacement ξ^ψ = ξ·∇ψ (ComplexF64 [npsi, numpert_total])
│   ├── Jxi_psi        # J·ξ^ψ Jacobian-weighted (from gpeq_contra)
│   ├── b_psi_area_weighted       # b^ψ / ⟨J·|∇ψ|⟩_θ area-normalized (ComplexF64 [npsi, numpert_total])
│   ├── b_n            # Physical normal field b_n (ComplexF64 [npsi, numpert_total])
│   ├── xi_n           # Physical normal displacement xi_n (ComplexF64 [npsi, numpert_total])
│   ├── Jb_theta
│   └── Jb_zeta
├── ResponseMatrices/         # [numpert_total × numpert_total], root-area-weighted field (b̃) space; R = S·A
│   ├── plasma_inductance     # Λ̃ = R⁻¹·Λ·R⁻†
│   ├── surface_inductance    # L̃ = R⁻¹·L·R⁻†
│   ├── permeability          # P̃ = R⁻¹·P·R  (P = Λ·L⁻¹)
│   ├── reluctance            # ϱ̃ = R†·ϱ·R
│   ├── rootarea_to_area_weight_operator  # S = Σ/√A at psilim; recover area-weighted field b̄ = S·b̃
│   └── surface_area          # scalar A = ∫J|∇ψ|dθ; recover flux via Φ = A·b̄
├── SingularCoupling/
│   ├── C_resonant_area_weighted_field     # [n_rational × numpert_total] coupling matrix (b̃-space input, resonant area-weighted field b^r=Φ^r/A^r [T])
│   ├── C_resonant_current
│   ├── C_island_width_sq
│   ├── C_penetrated_area_weighted_field
│   ├── C_Delta_prime
│   ├── resonant_area_weighted_field       # [n_rational] applied vector = C̃ · b̃_x (resonant area-weighted field b^r [T])
│   ├── resonant_current
│   ├── island_width_sq
│   ├── penetrated_area_weighted_field
│   ├── Delta_prime
│   ├── island_half_width    # [n_rational] Float64
│   ├── chirikov_parameter
│   ├── rational_psi         # [n_rational] surface metadata
│   ├── rational_q
│   ├── rational_m
│   ├── rational_n
│   └── DominantMode/        # SVD U·diag(σ)·Vᴴ of C_resonant_area_weighted_field over the core-window rational surfaces (ψ_N ≤ 0.9)
│       ├── singular_values          # σ, descending [rank]
│       ├── right_singular_vectors   # V: applied-b̃ spectra ranked by resonant drive [numpert_total × rank]
│       ├── left_singular_vectors    # U: resonant-field patterns over the retained surfaces [n_retained × rank]
│       ├── rational_index           # rows of rational_* retained in the ψ_N window [n_retained]
│       └── forcing_overlap          # Vᴴ·b̃_x, applied forcing's coefficient on each mode [rank]
├── Energies/
│   ├── vacuum_energy
│   ├── surface_energy
│   ├── plasma_energy
│   └── toroidal_torque
└── TorqueResponse/           # kinetic forward solves with compute_torque_response; b̃ basis, cumulative in ψ
    ├── psi                   # ψ_N of each stored node [npsi]
    ├── m, n                  # (m, n) of each row/column of T_xe [numpert_total]
    ├── T_xe                  # torque response on b̃, N·m/T² (ComplexF64 [numpert_total, numpert_total, npsi])
    ├── source_name           # forcing-source names: the coil sets, or the forcing file [nsource]
    ├── T_source              # torque response in forcing-source space (ComplexF64 [nsource, nsource, npsi])
    ├── T_applied             # b̃ₓ†·T_xe(ψ)·b̃ₓ/2: Re = torque inside ψ, Im = 2n·δW inside ψ (ComplexF64 [npsi])
    └── total_torque          # Re T_applied at the control surface [N·m]
```
"""
function save_to_hdf5(file::Union{HDF5.File,HDF5.Group}, state::PerturbedEquilibriumState)
    resp = state.response
    pe_group = replace_group!(file, "PerturbedEquilibrium")

    # Forcing modes
    forcing_group = haskey(pe_group, "ForcingModes") ? pe_group["ForcingModes"] : create_group(pe_group, "ForcingModes")
    forcing_group["n"] = [mode.n for mode in state.forcing_modes]
    forcing_group["m"] = [mode.m for mode in state.forcing_modes]
    forcing_group["amplitude"] = [mode.amplitude for mode in state.forcing_modes]

    # Labels of the mode axis shared by the spectra, matrices and profiles below
    !isempty(resp.m_modes) && (pe_group["mode_m"] = resp.m_modes)
    !isempty(resp.n_modes) && (pe_group["mode_n"] = resp.n_modes)

    # Control-surface forcing/response spectra in the three Pharr field representations
    # (all tesla; flux/weber is never stored). b̃ = root-area-weighted (coordinate-invariant).
    !isempty(state.forcing_b) && (pe_group["forcing_b"] = state.forcing_b)
    !isempty(state.forcing_b_rootarea) && (pe_group["forcing_b_root_area"] = state.forcing_b_rootarea)
    !isempty(state.forcing_b_area) && (pe_group["forcing_b_area"] = state.forcing_b_area)
    !isempty(state.response_b) && (pe_group["response_b"] = state.response_b)
    !isempty(state.response_b_rootarea) && (pe_group["response_b_root_area"] = state.response_b_rootarea)
    !isempty(state.response_b_area) && (pe_group["response_b_area"] = state.response_b_area)

    # Control surface matrices [numpert_total × numpert_total], in coordinate-invariant
    # root-area-weighted field (b̃) space. Recover the area-weighted field b̄ with the stored
    # operator S ≡ rootarea_to_area_weight (b̄ = S·b̃): e.g. L_b̄ = S·L̃·S†; recover flux with the
    # scalar surface_area A: Φ = A·b̄  (internally R = S·A, Φ = R·b̃). [Pharr 2026]
    mat_group = haskey(pe_group, "ResponseMatrices") ? pe_group["ResponseMatrices"] : create_group(pe_group, "ResponseMatrices")
    !isempty(resp.plasma_inductance) && (mat_group["plasma_inductance"] = resp.plasma_inductance)
    !isempty(resp.surface_inductance) && (mat_group["surface_inductance"] = resp.surface_inductance)
    !isempty(resp.permeability) && (mat_group["permeability"] = resp.permeability)
    !isempty(resp.reluctance) && (mat_group["reluctance"] = resp.reluctance)
    !isempty(resp.rootarea_to_area_weight) && (mat_group["rootarea_to_area_weight_operator"] = resp.rootarea_to_area_weight)
    (resp.surface_area != 0.0) && (mat_group["surface_area"] = resp.surface_area)

    # Response fields (ComplexF64 directly)
    response_group = haskey(pe_group, "Response") ? pe_group["Response"] : create_group(pe_group, "Response")
    !isempty(state.psi_grid) && (response_group["psi"] = state.psi_grid)
    have_xi = !isnothing(state.xi_modes)
    have_b = have_xi && !isnothing(state.b_modes)
    response_group["xi_psi"] = have_xi ? state.xi_modes.psi : ComplexF64[]
    response_group["b_psi_area_weighted"] = have_b ? state.b_modes.b_psi_area_weighted : ComplexF64[]
    response_group["Jb_theta"] = have_b ? state.b_modes.theta : ComplexF64[]
    response_group["Jb_zeta"] = have_b ? state.b_modes.zeta : ComplexF64[]
    response_group["b_n"] = !isnothing(state.b_n_modes) ? state.b_n_modes : ComplexF64[]
    response_group["xi_n"] = !isnothing(state.xi_n_modes) ? state.xi_n_modes : ComplexF64[]

    # Clebsch displacements for PENTRC (matches Fortran gpout_xclebsch)
    if have_xi
        response_group["xi_clebsch_psi"] = state.xi_modes.clebsch_psi
        response_group["dxi_clebsch_psidpsi"] = state.xi_modes.clebsch_psi1
        response_group["xi_clebsch_alpha"] = state.xi_modes.clebsch_alpha
    end

    # Contravariant displacement (from gpeq_contra, all J-weighted)
    if have_xi
        response_group["Jxi_psi"] = state.xi_modes.psi_J
        response_group["Jxi_theta"] = state.xi_modes.theta
        response_group["Jxi_zeta"] = state.xi_modes.zeta
    end

    # Covariant components (from gpeq_cova)
    if have_xi
        response_group["xi_cov_psi"] = state.xi_modes.cova_psi
        response_group["xi_cov_theta"] = state.xi_modes.cova_theta
        response_group["xi_cov_zeta"] = state.xi_modes.cova_zeta
    end
    if have_xi
        response_group["Jxi_theta_reg"] = state.xi_modes.theta_reg
        response_group["Jxi_zeta_reg"] = state.xi_modes.zeta_reg
    end
    if have_b
        response_group["Jb_theta_reg"] = state.b_modes.theta_reg
        response_group["Jb_zeta_reg"] = state.b_modes.zeta_reg
        response_group["b_cov_psi"] = state.b_modes.cova_psi
        response_group["b_cov_theta"] = state.b_modes.cova_theta
        response_group["b_cov_zeta"] = state.b_modes.cova_zeta
    end

    # R,Z,φ cylindrical components in mode-space (from gpeq_rzphi)
    if have_xi
        response_group["xi_R"] = state.xi_modes.R
        response_group["xi_Z"] = state.xi_modes.Z
        response_group["xi_phi"] = state.xi_modes.phi
    end
    if have_b
        response_group["b_R"] = state.b_modes.R
        response_group["b_Z"] = state.b_modes.Z
        response_group["b_phi"] = state.b_modes.phi
    end

    # Singular coupling
    coupling_group = haskey(pe_group, "SingularCoupling") ? pe_group["SingularCoupling"] : create_group(pe_group, "SingularCoupling")

    # Coupling matrices [n_rational × numpert_total]
    !isempty(resp.C_resonant_area_weighted_field) && (coupling_group["C_resonant_area_weighted_field"] = resp.C_resonant_area_weighted_field)
    !isempty(resp.C_resonant_current) && (coupling_group["C_resonant_current"] = resp.C_resonant_current)
    !isempty(resp.C_island_width_sq) && (coupling_group["C_island_width_sq"] = resp.C_island_width_sq)
    !isempty(resp.C_penetrated_area_weighted_field) && (coupling_group["C_penetrated_area_weighted_field"] = resp.C_penetrated_area_weighted_field)
    !isempty(resp.C_delta_prime) && (coupling_group["C_Delta_prime"] = resp.C_delta_prime)

    # Applied resonant vectors [n_rational]
    !isempty(state.resonant_area_weighted_field) && (coupling_group["resonant_area_weighted_field"] = state.resonant_area_weighted_field)
    !isempty(state.resonant_current) && (coupling_group["resonant_current"] = state.resonant_current)
    !isempty(state.island_width_sq) && (coupling_group["island_width_sq"] = state.island_width_sq)
    !isempty(state.penetrated_area_weighted_field) && (coupling_group["penetrated_area_weighted_field"] = state.penetrated_area_weighted_field)
    !isempty(state.delta_prime) && (coupling_group["Delta_prime"] = state.delta_prime)
    !isempty(state.forcing_solution_weights) && (coupling_group["forcing_solution_weights"] = state.forcing_solution_weights)
    !isempty(resp.rational_area) && (coupling_group["rational_area"] = resp.rational_area)
    !isempty(state.island_half_width) && (coupling_group["island_half_width"] = state.island_half_width)
    !isempty(state.chirikov_parameter) && (coupling_group["chirikov_parameter"] = state.chirikov_parameter)

    # Metadata [n_rational]
    !isempty(resp.rational_psi) && (coupling_group["rational_psi"] = resp.rational_psi)
    !isempty(resp.rational_q) && (coupling_group["rational_q"] = resp.rational_q)
    !isempty(resp.rational_m_res) && (coupling_group["rational_m"] = resp.rational_m_res)
    !isempty(resp.rational_n) && (coupling_group["rational_n"] = resp.rational_n)

    # Dominant resonant-coupling modes (SVD of the b̃-space coupling matrix over the retained rows)
    if !isempty(resp.dominant_singular_values)
        dominant_group = haskey(coupling_group, "DominantMode") ? coupling_group["DominantMode"] : create_group(coupling_group, "DominantMode")
        dominant_group["singular_values"] = resp.dominant_singular_values
        dominant_group["right_singular_vectors"] = resp.dominant_right_singular_vectors
        dominant_group["left_singular_vectors"] = resp.dominant_left_singular_vectors
        dominant_group["rational_index"] = resp.dominant_rational_index
        !isempty(state.dominant_forcing_overlap) && (dominant_group["forcing_overlap"] = state.dominant_forcing_overlap)
    end

    # Energies
    energy_group = haskey(pe_group, "Energies") ? pe_group["Energies"] : create_group(pe_group, "Energies")
    energy_group["vacuum_energy"] = state.vacuum_energy
    energy_group["surface_energy"] = state.surface_energy
    energy_group["plasma_energy"] = state.plasma_energy
    energy_group["toroidal_torque"] = state.toroidal_torque

    # Torque response matrices (kinetic forward solves only)
    tr = state.torque_response
    if tr !== nothing
        tr_group = haskey(pe_group, "TorqueResponse") ? pe_group["TorqueResponse"] : create_group(pe_group, "TorqueResponse")
        tr_group["psi"] = tr.psi
        tr_group["m"] = tr.m_modes
        tr_group["n"] = tr.n_modes
        tr_group["T_xe"] = tr.T_xe
        if !isempty(tr.source_names)
            tr_group["source_name"] = tr.source_names
            tr_group["T_source"] = tr.T_source
        end
        if !isempty(tr.T_applied)
            tr_group["T_applied"] = tr.T_applied
            tr_group["total_torque"] = real(tr.T_applied[end])
        end
    end

    annotate_pe!(pe_group)
    if !isempty(state.coil_sets)
        coils = replace_group!(file, "Input/RawInputs/Coils")
        ForcingTerms.save_coils_to_h5(state.coil_sets, coils)
    end
    _record_forcing_snapshot!(file, state)
    Utilities.record_inputs!(file, state)
    return nothing
end

# The forcing a replay reads back from Input/RawInputs/ForcingTerms: a file source's raw contents with their
# normalization, or the materialized control-surface modes of a field no deck can state (a weighted source, a spectrum,
# a sum). Coil forcing is replayed from its recorded geometry instead.
function _record_forcing_snapshot!(file, state::PerturbedEquilibriumState)
    forcing = state.forcing
    forcing === nothing && return nothing
    snapshot = forcing isa ForcingTerms.RMPSource && forcing.scale == 1 ? ForcingTerms.snapshot_forcing_modes(forcing.ctrl) :
               (state.forcing_modes, "unit_flux")
    snapshot === nothing && return nothing
    modes, normalization = snapshot
    ForcingTerms.save_forcing_to_h5(modes, replace_group!(file, "Input/RawInputs/ForcingTerms"); normalization)
    return nothing
end

# Metadata tables for the PerturbedEquilibrium group, applied post-write (paths are
# relative to the PerturbedEquilibrium group). Field-representation naming follows
# docs/src/conventions.md (Pharr 2026): b = bare, b̄ = area-weighted, b̃ =
# root-area-weighted; all in tesla.
const PE_H5_ANNOTATIONS = [
    "ForcingModes/n" => (; long_name="toroidal mode number of each forcing mode"),
    "ForcingModes/m" => (; long_name="poloidal mode number of each forcing mode"),
    "ForcingModes/amplitude" => (; long_name="complex forcing amplitude of each mode", units="T"),
    "mode_m" => (; long_name="poloidal mode number m of each entry along the mode axis (m fastest, one block per n)", units="1", dims=("mode",)),
    "mode_n" => (; long_name="toroidal mode number n of each entry along the mode axis (m fastest, one block per n)", units="1", dims=("mode",)),
    "forcing_b" => (; long_name="control-surface forcing spectrum, bare normal field b", units="T", dims=("mode",)),
    "forcing_b_root_area" => (; long_name="control-surface forcing spectrum, root-area-weighted field b̃ (coordinate-invariant)", units="T", dims=("mode",)),
    "forcing_b_area" => (; long_name="control-surface forcing spectrum, area-weighted field b̄ (Φ = A·b̄)", units="T", dims=("mode",)),
    "response_b" => (; long_name="control-surface response spectrum, bare normal field b", units="T", dims=("mode",)),
    "response_b_root_area" => (; long_name="control-surface response spectrum, root-area-weighted field b̃ (coordinate-invariant)", units="T", dims=("mode",)),
    "response_b_area" => (; long_name="control-surface response spectrum, area-weighted field b̄ (Φ = A·b̄)", units="T", dims=("mode",)),
    "ResponseMatrices/plasma_inductance" => (; long_name="plasma inductance Λ̃ in root-area-weighted field space", dims=("mode_row", "mode_col")),
    "ResponseMatrices/surface_inductance" => (; long_name="surface inductance L̃ in root-area-weighted field space", dims=("mode_row", "mode_col")),
    "ResponseMatrices/permeability" => (; long_name="permeability P̃ = Λ̃·L̃⁻¹ in root-area-weighted field space", dims=("mode_row", "mode_col")),
    "ResponseMatrices/reluctance" => (; long_name="reluctance ϱ̃ in root-area-weighted field space", dims=("mode_row", "mode_col")),
    "ResponseMatrices/rootarea_to_area_weight_operator" => (; long_name="operator S = Σ/√A at ψ_lim; b̄ = S·b̃", dims=("mode_row", "mode_col")),
    "ResponseMatrices/surface_area" => (; long_name="control-surface scalar area A = ∮J|∇ψ|dθ; Φ = A·b̄", units="m^2"),
    "Response/psi" => (; long_name="normalized poloidal flux ψ_N grid shared by the response profiles", scale="psi"),
    "Response/xi_psi" => (; long_name="contravariant radial displacement ξ^ψ = ξ·∇ψ_N", dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/Jxi_psi" => (; long_name="Jacobian-weighted contravariant radial displacement J·ξ^ψ", units="m^3", dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/Jxi_theta" => (; long_name="Jacobian-weighted contravariant poloidal displacement J·ξ^θ", units="m^3", dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/Jxi_zeta" => (; long_name="Jacobian-weighted contravariant toroidal displacement J·ξ^ζ", units="m^3", dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/Jxi_theta_reg" =>
        (; long_name="regularized Jacobian-weighted contravariant poloidal displacement J·ξ^θ", units="m^3", dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/Jxi_zeta_reg" =>
        (; long_name="regularized Jacobian-weighted contravariant toroidal displacement J·ξ^ζ", units="m^3", dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/xi_cov_psi" => (; long_name="covariant radial displacement ξ_ψ = ξ·e_ψ", units="m^2", dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/xi_cov_theta" => (; long_name="covariant poloidal displacement ξ_θ = ξ·e_θ", units="m^2", dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/xi_cov_zeta" => (; long_name="covariant toroidal displacement ξ_ζ = ξ·e_ζ", units="m^2", dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/xi_clebsch_psi" =>
        (; long_name="Clebsch displacement component ξ^ψ (PENTRC input, gpout_xclebsch convention)", dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/dxi_clebsch_psidpsi" =>
        (; long_name="regularized ψ_N derivative of ξ^ψ (× singfac²/(singfac²+regularization_width²))", dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/xi_clebsch_alpha" =>
        (; long_name="Clebsch displacement component ξ^α/χ₁ (PENTRC input, gpout_xclebsch convention)", dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/xi_n" => (; long_name="physical normal displacement ξ_n", units="m", dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/xi_R" => (; long_name="cylindrical displacement component ξ_R (mode space)", units="m", dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/xi_Z" => (; long_name="cylindrical displacement component ξ_Z (mode space)", units="m", dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/xi_phi" => (; long_name="cylindrical displacement component ξ_φ (mode space)", units="m", dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/b_psi_area_weighted" =>
        (; long_name="area-weighted normal field b̄ = (J·b^ψ)_m/A (Φ = A·b̄; same representation as response_b_area)", units="T",
            dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/b_n" => (; long_name="physical normal field b_n", units="T", dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/Jb_theta" =>
        (; long_name="Jacobian-weighted contravariant poloidal field J·b^θ (flux-like density, not a tesla field amplitude)", units="T*m^2",
            dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/Jb_zeta" =>
        (; long_name="Jacobian-weighted contravariant toroidal field J·b^ζ (flux-like density, not a tesla field amplitude)", units="T*m^2",
            dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/Jb_theta_reg" =>
        (; long_name="regularized Jacobian-weighted contravariant poloidal field J·b^θ (flux-like density, not a tesla field amplitude)", units="T*m^2", dims=("psi", "mode"),
            attach=(1 => "Response/psi",)),
    "Response/Jb_zeta_reg" =>
        (; long_name="regularized Jacobian-weighted contravariant toroidal field J·b^ζ (flux-like density, not a tesla field amplitude)", units="T*m^2",
            dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/b_cov_psi" => (; long_name="covariant radial field b_ψ = b·e_ψ", units="T*m", dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/b_cov_theta" => (; long_name="covariant poloidal field b_θ = b·e_θ", units="T*m", dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/b_cov_zeta" => (; long_name="covariant toroidal field b_ζ = b·e_ζ", units="T*m", dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/b_R" => (; long_name="cylindrical field component b_R (mode space)", units="T", dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/b_Z" => (; long_name="cylindrical field component b_Z (mode space)", units="T", dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "Response/b_phi" => (; long_name="cylindrical field component b_φ (mode space)", units="T", dims=("psi", "mode"), attach=(1 => "Response/psi",)),
    "SingularCoupling/C_resonant_area_weighted_field" =>
        (; long_name="coupling matrix: applied b̃ → resonant area-weighted field b̄^r = Φ^r/A^r", dims=("surface", "mode"),
            attach=(1 => "SingularCoupling/rational_psi", 1 => "SingularCoupling/rational_q")),
    "SingularCoupling/C_resonant_current" =>
        (; long_name="coupling matrix: applied b̃ → pitch-resonant current", units="A/T", dims=("surface", "mode"),
            attach=(1 => "SingularCoupling/rational_psi", 1 => "SingularCoupling/rational_q")),
    "SingularCoupling/C_island_width_sq" =>
        (; long_name="coupling matrix: applied b̃ → squared island half-width", units="1/T", dims=("surface", "mode"),
            attach=(1 => "SingularCoupling/rational_psi", 1 => "SingularCoupling/rational_q")),
    "SingularCoupling/C_penetrated_area_weighted_field" =>
        (; long_name="coupling matrix: applied b̃ → penetrated area-weighted field", dims=("surface", "mode"),
            attach=(1 => "SingularCoupling/rational_psi", 1 => "SingularCoupling/rational_q")),
    "SingularCoupling/C_Delta_prime" =>
        (; long_name="coupling matrix: applied b̃ → forcing-driven Δ'", units="1/T", dims=("surface", "mode"),
            attach=(1 => "SingularCoupling/rational_psi", 1 => "SingularCoupling/rational_q")),
    "SingularCoupling/resonant_area_weighted_field" =>
        (; long_name="resonant area-weighted field b̄^r = Φ^r/A^r per rational surface (coordinate-invariant)", units="T", dims=("surface",),
            attach=(1 => "SingularCoupling/rational_psi", 1 => "SingularCoupling/rational_q")),
    "SingularCoupling/resonant_current" =>
        (; long_name="pitch-resonant current per rational surface", units="A", dims=("surface",),
            attach=(1 => "SingularCoupling/rational_psi", 1 => "SingularCoupling/rational_q")),
    "SingularCoupling/island_width_sq" =>
        (; long_name="squared island half-width per rational surface (in ψ_N²)", dims=("surface",),
            attach=(1 => "SingularCoupling/rational_psi", 1 => "SingularCoupling/rational_q")),
    "SingularCoupling/penetrated_area_weighted_field" =>
        (; long_name="penetrated area-weighted field per rational surface", units="T", dims=("surface",),
            attach=(1 => "SingularCoupling/rational_psi", 1 => "SingularCoupling/rational_q")),
    "SingularCoupling/Delta_prime" =>
        (; long_name="forcing-driven tearing Δ' per rational surface (Riccati; response to applied forcing)", dims=("surface",),
            attach=(1 => "SingularCoupling/rational_psi", 1 => "SingularCoupling/rational_q")),
    "SingularCoupling/forcing_solution_weights" =>
        (; long_name="weights of the forcing solutions in the singular-coupling decomposition", dims=("surface",),
            attach=(1 => "SingularCoupling/rational_psi", 1 => "SingularCoupling/rational_q")),
    "SingularCoupling/rational_area" =>
        (; long_name="scalar surface area A^r of each rational surface", units="m^2", dims=("surface",),
            attach=(1 => "SingularCoupling/rational_psi", 1 => "SingularCoupling/rational_q")),
    "SingularCoupling/island_half_width" =>
        (; long_name="island half-width per rational surface (in ψ_N)", dims=("surface",), attach=(1 => "SingularCoupling/rational_psi", 1 => "SingularCoupling/rational_q")),
    "SingularCoupling/chirikov_parameter" =>
        (; long_name="Chirikov overlap parameter: island half-width / half-distance to the neighbouring rational surface", dims=("surface",),
            attach=(1 => "SingularCoupling/rational_psi", 1 => "SingularCoupling/rational_q")),
    "SingularCoupling/rational_psi" => (; long_name="normalized poloidal flux ψ_N of each rational surface", scale="psi_rational"),
    "SingularCoupling/rational_q" =>
        (; long_name="safety factor q = m/n at each rational surface", dims=("surface",), scale="q_rational", attach=(1 => "SingularCoupling/rational_psi",)),
    "SingularCoupling/rational_m" =>
        (; long_name="resonant poloidal mode number m at each rational surface", dims=("surface",),
            attach=(1 => "SingularCoupling/rational_psi", 1 => "SingularCoupling/rational_q")),
    "SingularCoupling/rational_n" =>
        (; long_name="resonant toroidal mode number n at each rational surface",
            dims=("surface",), attach=(1 => "SingularCoupling/rational_psi", 1 => "SingularCoupling/rational_q")),
    "SingularCoupling/DominantMode/singular_values" =>
        (;
            long_name="singular values σ (descending) of the applied-b̃ → resonant-field coupling matrix over the retained rational surfaces (core window ψ_N ≤ 0.9)",
            dims=("rank",)
        ),
    "SingularCoupling/DominantMode/right_singular_vectors" =>
        (; long_name="right singular vectors V: applied root-area-weighted field spectra ranked by resonant drive; overlap of b̃ with mode k is dot(V[:,k], b̃)",
            dims=("mode", "rank")),
    "SingularCoupling/DominantMode/left_singular_vectors" =>
        (; long_name="left singular vectors U: resonant area-weighted field patterns over the retained rational surfaces", dims=("surface_retained", "rank")),
    "SingularCoupling/DominantMode/rational_index" =>
        (; long_name="1-based index into the SingularCoupling rational_* arrays of each rational surface retained in the SVD window", dims=("surface_retained",)),
    "SingularCoupling/DominantMode/forcing_overlap" =>
        (; long_name="coefficient of the applied forcing b̃_x on each singular mode, Vᴴ·b̃_x", units="T", dims=("rank",)),
    "Energies/vacuum_energy" => (; long_name="perturbed vacuum energy", units="J"),
    "Energies/surface_energy" => (; long_name="perturbed surface energy", units="J"),
    "Energies/plasma_energy" => (; long_name="perturbed plasma energy", units="J"),
    "Energies/toroidal_torque" => (;
        long_name="boundary-response toroidal torque −2n·Im⟨Φ_tot,Λ⁻¹Φ_tot⟩/4: equals the volume-integrated Euler-Lagrange kinetic torque for converged self-consistent solutions; distinct construction from the KineticForces NTV torque",
        units="N*m"
    ),
    "TorqueResponse/psi" => (; long_name="normalized poloidal flux ψ_N of each stored node of the solve; the torque response matrices are cumulative inside it", scale="psi"),
    "TorqueResponse/m" => (; long_name="poloidal mode number of each row and column of T_xe", scale="m"),
    "TorqueResponse/n" => (; long_name="toroidal mode number of each row and column of T_xe", dims=("mode",), attach=(1 => "TorqueResponse/m",)),
    "TorqueResponse/T_xe" =>
        (;
            long_name="torque response matrix on the applied root-area-weighted field b̃, cumulative inside ψ_N: b̃†·T_xe·b̃/2 is the complex torque (Re = toroidal torque, Im = 2n·δW ideal+kinetic); the Hermitian part is the torque quadratic form",
            units="N*m/T^2", dims=("mode_row", "mode_col", "psi"), attach=(1 => "TorqueResponse/m", 2 => "TorqueResponse/m", 3 => "TorqueResponse/psi")),
    "TorqueResponse/source_name" => (; long_name="name of each forcing source of T_source (coil set or forcing file)"),
    "TorqueResponse/T_source" =>
        (;
            long_name="torque response matrix in forcing-source space, cumulative inside ψ_N: s†·T_source·s/2 is the complex torque of the sources driven at s times their strength as built",
            units="N*m", dims=("source_row", "source_col", "psi"), attach=(3 => "TorqueResponse/psi",)),
    "TorqueResponse/T_applied" =>
        (;
            long_name="run's own forcing contracted with the torque response, b̃ₓ†·T_xe(ψ)·b̃ₓ/2: Re = toroidal torque on the plasma inside ψ_N, Im = 2n·δW (ideal + kinetic) of that plasma",
            units="N*m", dims=("psi",), attach=(1 => "TorqueResponse/psi",)),
    "TorqueResponse/total_torque" => (;
        long_name="toroidal torque of the run's own forcing from the torque response at the control surface, Re T_applied(ψ_lim); equals Energies/toroidal_torque",
        units="N*m"
    )
]

# Attach long_name/units/dims + dimension scales (declared in-table) to the
# PerturbedEquilibrium group.
function annotate_pe!(pe_group)
    Utilities.HDF5Annotations.annotate!(pe_group, PE_H5_ANNOTATIONS)
    return nothing
end
