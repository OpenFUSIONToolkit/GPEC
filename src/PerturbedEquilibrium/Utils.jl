"""
    initialize_mode_arrays!(
        intr::PerturbedEquilibriumInternal,
        ffs::ForceFreeStatesResult
    )

Initialize mode number arrays for convenient indexing.

Pre-computes m_modes[i] and n_modes[i] for each linear index i in 1:numpert_total,
avoiding repeated index arithmetic throughout the code.

## Mode Indexing Convention

For linear index i ∈ [1, numpert_total]:

  - m_modes[i] = (i-1) % mpert + mlow
  - n_modes[i] = (i-1) ÷ mpert + nlow

This matches the convention used in ForceFreeStates where modes are ordered as:
(m1,n1), (m2,n1), ..., (mpert,n1), (m1,n2), (m2,n2), ..., (mpert,npert)
"""
function initialize_mode_arrays!(
    intr::PerturbedEquilibriumInternal,
    ffs::ForceFreeStatesResult
)
    numpert_total = ffs.numpert_total
    mpert = ffs.mpert
    mlow = ffs.mlow
    nlow = ffs.nlow

    # Allocate arrays
    intr.m_modes = zeros(Int, numpert_total)
    intr.n_modes = zeros(Int, numpert_total)

    # Fill arrays using indexing convention
    for i in 1:numpert_total
        m_idx = (i - 1) % mpert + 1  # 1-based index into mpert range
        n_idx = (i - 1) ÷ mpert + 1  # 1-based index into npert range

        intr.m_modes[i] = m_idx - 1 + mlow
        intr.n_modes[i] = n_idx - 1 + nlow
    end
end

"""
    write_outputs_to_HDF5(
        state::PerturbedEquilibriumState,
        intr::PerturbedEquilibriumInternal,
        filename::String
    )

Write perturbed equilibrium results to HDF5 file (appends to existing ForceFreeStates output).

## Output Structure

```
PerturbedEquilibrium/
├── ForcingModes/
│   ├── n              # Toroidal mode numbers
│   ├── m              # Poloidal mode numbers
│   └── amplitude      # ComplexF64 forcing amplitudes
├── forcing_b / forcing_b_root_area / forcing_b_area      # control-surface forcing spectrum (b, b̃, b̄) [numpert_total], tesla
├── response_b / response_b_root_area / response_b_area   # control-surface response spectrum (b, b̃, b̄) [numpert_total], tesla
├── Response/
│   ├── psi             # Radial abscissa ψ_N [npsi] shared by every response profile below
│   ├── xi_psi         # Radial displacement ξ^ψ = ξ·∇ψ (ComplexF64 [npsi, mpert])
│   ├── Jxi_psi        # J·ξ^ψ Jacobian-weighted (from gpeq_contra)
│   ├── b_psi_area_weighted       # b^ψ / ⟨J·|∇ψ|⟩_θ area-normalized (ComplexF64 [npsi, mpert])
│   ├── b_n            # Physical normal field b_n (ComplexF64 [npsi, mpert])
│   ├── xi_n           # Physical normal displacement xi_n (ComplexF64 [npsi, mpert])
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
│   └── rational_n
└── Energies/
    ├── vacuum_energy
    ├── surface_energy
    ├── plasma_energy
    └── toroidal_torque
```
"""
function write_outputs_to_HDF5(
    state::PerturbedEquilibriumState,
    intr::PerturbedEquilibriumInternal,
    filename::String
)
    h5open(filename, "cw") do file
        pe_group = haskey(file, "PerturbedEquilibrium") ? file["PerturbedEquilibrium"] : create_group(file, "PerturbedEquilibrium")

        # Forcing modes
        forcing_group = haskey(pe_group, "ForcingModes") ? pe_group["ForcingModes"] : create_group(pe_group, "ForcingModes")
        forcing_group["n"] = [mode.n for mode in intr.forcing_modes]
        forcing_group["m"] = [mode.m for mode in intr.forcing_modes]
        forcing_group["amplitude"] = [mode.amplitude for mode in intr.forcing_modes]

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
        !isempty(state.plasma_inductance) && (mat_group["plasma_inductance"] = state.plasma_inductance)
        !isempty(state.surface_inductance) && (mat_group["surface_inductance"] = state.surface_inductance)
        !isempty(state.permeability) && (mat_group["permeability"] = state.permeability)
        !isempty(state.reluctance) && (mat_group["reluctance"] = state.reluctance)
        !isempty(state.rootarea_to_area_weight) && (mat_group["rootarea_to_area_weight_operator"] = state.rootarea_to_area_weight)
        (state.surface_area != 0.0) && (mat_group["surface_area"] = state.surface_area)

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
        !isempty(state.C_resonant_area_weighted_field) && (coupling_group["C_resonant_area_weighted_field"] = state.C_resonant_area_weighted_field)
        !isempty(state.C_resonant_current) && (coupling_group["C_resonant_current"] = state.C_resonant_current)
        !isempty(state.C_island_width_sq) && (coupling_group["C_island_width_sq"] = state.C_island_width_sq)
        !isempty(state.C_penetrated_area_weighted_field) && (coupling_group["C_penetrated_area_weighted_field"] = state.C_penetrated_area_weighted_field)
        !isempty(state.C_delta_prime) && (coupling_group["C_Delta_prime"] = state.C_delta_prime)

        # Applied resonant vectors [n_rational]
        !isempty(state.resonant_area_weighted_field) && (coupling_group["resonant_area_weighted_field"] = state.resonant_area_weighted_field)
        !isempty(state.resonant_current) && (coupling_group["resonant_current"] = state.resonant_current)
        !isempty(state.island_width_sq) && (coupling_group["island_width_sq"] = state.island_width_sq)
        !isempty(state.penetrated_area_weighted_field) && (coupling_group["penetrated_area_weighted_field"] = state.penetrated_area_weighted_field)
        !isempty(state.delta_prime) && (coupling_group["Delta_prime"] = state.delta_prime)
        !isempty(state.forcing_solution_weights) && (coupling_group["forcing_solution_weights"] = state.forcing_solution_weights)
        !isempty(state.rational_area) && (coupling_group["rational_area"] = state.rational_area)
        !isempty(state.island_half_width) && (coupling_group["island_half_width"] = state.island_half_width)
        !isempty(state.chirikov_parameter) && (coupling_group["chirikov_parameter"] = state.chirikov_parameter)

        # Metadata [n_rational]
        !isempty(state.rational_psi) && (coupling_group["rational_psi"] = state.rational_psi)
        !isempty(state.rational_q) && (coupling_group["rational_q"] = state.rational_q)
        !isempty(state.rational_m_res) && (coupling_group["rational_m"] = state.rational_m_res)
        !isempty(state.rational_n) && (coupling_group["rational_n"] = state.rational_n)

        # Energies
        energy_group = haskey(pe_group, "Energies") ? pe_group["Energies"] : create_group(pe_group, "Energies")
        energy_group["vacuum_energy"] = state.vacuum_energy
        energy_group["surface_energy"] = state.surface_energy
        energy_group["plasma_energy"] = state.plasma_energy
        energy_group["toroidal_torque"] = state.toroidal_torque

        annotate_pe!(pe_group)
    end
end

# Metadata tables for the PerturbedEquilibrium group, applied post-write (paths are
# relative to the PerturbedEquilibrium group). Field-representation naming follows
# docs/src/conventions.md (Pharr 2026): b = bare, b̄ = area-weighted, b̃ =
# root-area-weighted; all in tesla.
const PE_H5_ANNOTATIONS = [
    "ForcingModes/n" => (; long_name="toroidal mode number of each forcing mode"),
    "ForcingModes/m" => (; long_name="poloidal mode number of each forcing mode"),
    "ForcingModes/amplitude" => (; long_name="complex forcing amplitude of each mode", units="T"),
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
        (; long_name="regularized ψ_N derivative of ξ^ψ (× singfac²/(singfac²+reg_spot²))", dims=("psi", "mode"), attach=(1 => "Response/psi",)),
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
    "Energies/vacuum_energy" => (; long_name="perturbed vacuum energy", units="J"),
    "Energies/surface_energy" => (; long_name="perturbed surface energy", units="J"),
    "Energies/plasma_energy" => (; long_name="perturbed plasma energy", units="J"),
    "Energies/toroidal_torque" => (; long_name="net toroidal torque on the plasma", units="N*m")
]

# Attach long_name/units/dims + dimension scales (declared in-table) to the
# PerturbedEquilibrium group.
function annotate_pe!(pe_group)
    Utilities.HDF5Annotations.annotate!(pe_group, PE_H5_ANNOTATIONS)
    return nothing
end
