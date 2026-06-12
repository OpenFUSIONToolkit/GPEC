"""
    initialize_mode_arrays!(
        intr::PerturbedEquilibriumInternal,
        ffs_intr::ForceFreeStatesInternal
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
    ffs_intr::ForceFreeStatesInternal
)
    numpert_total = ffs_intr.numpert_total
    mpert = ffs_intr.mpert
    mlow = ffs_intr.mlow
    nlow = ffs_intr.nlow

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
perturbed_equilibrium/
├── forcing_modes/
│   ├── n              # Toroidal mode numbers
│   ├── m              # Poloidal mode numbers
│   └── amplitude      # ComplexF64 forcing amplitudes
├── forcing_b / forcing_b_root_area / forcing_b_area      # control-surface forcing spectrum (b, b̃, b̄) [numpert_total], tesla
├── response_b / response_b_root_area / response_b_area   # control-surface response spectrum (b, b̃, b̄) [numpert_total], tesla
├── response/
│   ├── xi_psi         # Radial displacement ξ^ψ = ξ·∇ψ (ComplexF64 [npsi, mpert])
│   ├── xi_psi_J       # J·ξ^ψ Jacobian-weighted (from gpeq_contra)
│   ├── b_psi_area_weighted       # b^ψ / ⟨J·|∇ψ|⟩_θ area-normalized (ComplexF64 [npsi, mpert])
│   ├── b_n            # Physical normal field b_n (ComplexF64 [npsi, mpert])
│   ├── xi_n           # Physical normal displacement xi_n (ComplexF64 [npsi, mpert])
│   ├── b_theta
│   └── b_zeta
├── response_matrices/        # [numpert_total × numpert_total], root-area-weighted field (b̃) space; R = S·A
│   ├── plasma_inductance     # Λ̃ = R⁻¹·Λ·R⁻†
│   ├── surface_inductance    # L̃ = R⁻¹·L·R⁻†
│   ├── permeability          # P̃ = R⁻¹·P·R  (P = Λ·L⁻¹)
│   ├── reluctance            # ϱ̃ = R†·ϱ·R
│   ├── rootarea_to_area_weight_operator  # S = Σ/√A at psilim; recover area-weighted field b̄ = S·b̃
│   └── surface_area          # scalar A = ∫J|∇ψ|dθ; recover flux via Φ = A·b̄
├── singular_coupling/
│   ├── C_resonant_area_weighted_field     # [n_rational × numpert_total] coupling matrix (b̃-space input, resonant area-weighted field b^r=Φ^r/A^r [T])
│   ├── C_resonant_current
│   ├── C_island_width_sq
│   ├── C_penetrated_area_weighted_field
│   ├── C_delta_prime
│   ├── resonant_area_weighted_field       # [n_rational] applied vector = C̃ · b̃_x (resonant area-weighted field b^r [T])
│   ├── resonant_current
│   ├── island_width_sq
│   ├── penetrated_area_weighted_field
│   ├── delta_prime
│   ├── island_half_width    # [n_rational] Float64
│   ├── chirikov_parameter
│   ├── rational_psi         # [n_rational] surface metadata
│   ├── rational_q
│   ├── rational_m_res
│   └── rational_n
└── energies/
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
        pe_group = haskey(file, "perturbed_equilibrium") ? file["perturbed_equilibrium"] : create_group(file, "perturbed_equilibrium")

        # Forcing modes
        forcing_group = haskey(pe_group, "forcing_modes") ? pe_group["forcing_modes"] : create_group(pe_group, "forcing_modes")
        forcing_group["n"]         = [mode.n for mode in intr.forcing_modes]
        forcing_group["m"]         = [mode.m for mode in intr.forcing_modes]
        forcing_group["amplitude"] = [mode.amplitude for mode in intr.forcing_modes]

        # Control-surface forcing/response spectra in the three Pharr field representations
        # (all tesla; flux/weber is never stored). b̃ = root-area-weighted (coordinate-invariant).
        !isempty(state.forcing_b)           && (pe_group["forcing_b"]            = state.forcing_b)
        !isempty(state.forcing_b_rootarea)  && (pe_group["forcing_b_root_area"]  = state.forcing_b_rootarea)
        !isempty(state.forcing_b_area)      && (pe_group["forcing_b_area"]       = state.forcing_b_area)
        !isempty(state.response_b)          && (pe_group["response_b"]           = state.response_b)
        !isempty(state.response_b_rootarea) && (pe_group["response_b_root_area"] = state.response_b_rootarea)
        !isempty(state.response_b_area)     && (pe_group["response_b_area"]      = state.response_b_area)

        # Control surface matrices [numpert_total × numpert_total], in coordinate-invariant
        # root-area-weighted field (b̃) space. Recover the area-weighted field b̄ with the stored
        # operator S ≡ rootarea_to_area_weight (b̄ = S·b̃): e.g. L_b̄ = S·L̃·S†; recover flux with the
        # scalar surface_area A: Φ = A·b̄  (internally R = S·A, Φ = R·b̃). [Pharr 2026]
        mat_group = haskey(pe_group, "response_matrices") ? pe_group["response_matrices"] : create_group(pe_group, "response_matrices")
        !isempty(state.plasma_inductance)  && (mat_group["plasma_inductance"]  = state.plasma_inductance)
        !isempty(state.surface_inductance) && (mat_group["surface_inductance"] = state.surface_inductance)
        !isempty(state.permeability)       && (mat_group["permeability"]       = state.permeability)
        !isempty(state.reluctance)         && (mat_group["reluctance"]         = state.reluctance)
        !isempty(state.rootarea_to_area_weight) && (mat_group["rootarea_to_area_weight_operator"] = state.rootarea_to_area_weight)
        (state.surface_area != 0.0)        && (mat_group["surface_area"]       = state.surface_area)

        # Response fields (ComplexF64 directly)
        response_group = haskey(pe_group, "response") ? pe_group["response"] : create_group(pe_group, "response")
        have_xi = !isnothing(state.xi_modes)
        have_b  = have_xi && !isnothing(state.b_modes)
        response_group["xi_psi"]    = have_xi ? state.xi_modes.psi      : ComplexF64[]
        response_group["b_psi_area_weighted"]  = have_b  ? state.b_modes.b_psi_area_weighted : ComplexF64[]
        response_group["b_theta"]   = have_b  ? state.b_modes.theta    : ComplexF64[]
        response_group["b_zeta"]    = have_b  ? state.b_modes.zeta     : ComplexF64[]
        response_group["b_n"]       = !isnothing(state.b_n_modes)  ? state.b_n_modes  : ComplexF64[]
        response_group["xi_n"]      = !isnothing(state.xi_n_modes) ? state.xi_n_modes : ComplexF64[]

        # Clebsch displacements for PENTRC (matches Fortran gpout_xclebsch)
        if have_xi
            response_group["clebsch_psi"]   = state.xi_modes.clebsch_psi
            response_group["clebsch_psi1"]  = state.xi_modes.clebsch_psi1
            response_group["clebsch_alpha"] = state.xi_modes.clebsch_alpha
        end

        # Contravariant displacement (from gpeq_contra, all J-weighted)
        if have_xi
            response_group["xi_psi_J"] = state.xi_modes.psi_J
            response_group["xi_theta"] = state.xi_modes.theta
            response_group["xi_zeta"]  = state.xi_modes.zeta
        end

        # Covariant components (from gpeq_cova)
        if have_xi
            response_group["xi_cova_psi"]   = state.xi_modes.cova_psi
            response_group["xi_cova_theta"] = state.xi_modes.cova_theta
            response_group["xi_cova_zeta"]  = state.xi_modes.cova_zeta
        end
        if have_xi
            response_group["xi_theta_reg"] = state.xi_modes.theta_reg
            response_group["xi_zeta_reg"]  = state.xi_modes.zeta_reg
        end
        if have_b
            response_group["b_theta_reg"] = state.b_modes.theta_reg
            response_group["b_zeta_reg"]  = state.b_modes.zeta_reg
            response_group["b_cova_psi"]   = state.b_modes.cova_psi
            response_group["b_cova_theta"] = state.b_modes.cova_theta
            response_group["b_cova_zeta"]  = state.b_modes.cova_zeta
        end

        # R,Z,φ cylindrical components in mode-space (from gpeq_rzphi)
        if have_xi
            response_group["xi_R"]   = state.xi_modes.R
            response_group["xi_Z"]   = state.xi_modes.Z
            response_group["xi_phi"] = state.xi_modes.phi
        end
        if have_b
            response_group["b_R"]   = state.b_modes.R
            response_group["b_Z"]   = state.b_modes.Z
            response_group["b_phi"] = state.b_modes.phi
        end

        # Singular coupling
        coupling_group = haskey(pe_group, "singular_coupling") ? pe_group["singular_coupling"] : create_group(pe_group, "singular_coupling")

        # Coupling matrices [n_rational × numpert_total]
        !isempty(state.C_resonant_area_weighted_field)   && (coupling_group["C_resonant_area_weighted_field"]   = state.C_resonant_area_weighted_field)
        !isempty(state.C_resonant_current) && (coupling_group["C_resonant_current"] = state.C_resonant_current)
        !isempty(state.C_island_width_sq)  && (coupling_group["C_island_width_sq"]  = state.C_island_width_sq)
        !isempty(state.C_penetrated_area_weighted_field) && (coupling_group["C_penetrated_area_weighted_field"] = state.C_penetrated_area_weighted_field)
        !isempty(state.C_delta_prime)      && (coupling_group["C_delta_prime"]      = state.C_delta_prime)

        # Applied resonant vectors [n_rational]
        !isempty(state.resonant_area_weighted_field)     && (coupling_group["resonant_area_weighted_field"]     = state.resonant_area_weighted_field)
        !isempty(state.resonant_current)   && (coupling_group["resonant_current"]   = state.resonant_current)
        !isempty(state.island_width_sq)    && (coupling_group["island_width_sq"]    = state.island_width_sq)
        !isempty(state.penetrated_area_weighted_field)   && (coupling_group["penetrated_area_weighted_field"]   = state.penetrated_area_weighted_field)
        !isempty(state.delta_prime)        && (coupling_group["delta_prime"]        = state.delta_prime)
        !isempty(state.island_half_width)  && (coupling_group["island_half_width"]  = state.island_half_width)
        !isempty(state.chirikov_parameter) && (coupling_group["chirikov_parameter"] = state.chirikov_parameter)

        # Metadata [n_rational]
        !isempty(state.rational_psi)       && (coupling_group["rational_psi"]       = state.rational_psi)
        !isempty(state.rational_q)         && (coupling_group["rational_q"]         = state.rational_q)
        !isempty(state.rational_m_res)     && (coupling_group["rational_m_res"]     = state.rational_m_res)
        !isempty(state.rational_n)         && (coupling_group["rational_n"]         = state.rational_n)

        # Energies
        energy_group = haskey(pe_group, "energies") ? pe_group["energies"] : create_group(pe_group, "energies")
        energy_group["vacuum_energy"]   = state.vacuum_energy
        energy_group["surface_energy"]  = state.surface_energy
        energy_group["plasma_energy"]   = state.plasma_energy
        energy_group["toroidal_torque"] = state.toroidal_torque
    end
end
