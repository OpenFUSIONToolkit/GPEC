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
├── response/
│   ├── xi_psi         # Displacement field (ComplexF64 [npsi, mpert])
│   ├── b_psi          # Magnetic field (ComplexF64 [npsi, mpert])
│   ├── b_theta
│   └── b_zeta
├── singular_coupling/
│   ├── C_resonant_flux      # [n_rational × numpert_total] coupling matrix
│   ├── C_resonant_current
│   ├── C_island_width_sq
│   ├── C_penetrated_field
│   ├── C_delta_prime
│   ├── resonant_flux        # [n_rational] applied vector = C · amp_vec
│   ├── resonant_current
│   ├── island_width_sq
│   ├── penetrated_field
│   ├── delta_prime
│   ├── island_half_width    # [n_rational] Float64
│   ├── chirikov_parameter
│   ├── rational_psi         # [n_rational] surface metadata
│   ├── rational_q
│   ├── rational_m_res
│   └── rational_n
└── energies/
    ├── plasma_energy
    ├── vacuum_energy
    └── total_energy
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

        # Control surface perturbation vectors (Phi_x and Phi_tot = P*Phi_x)
        !isempty(state.forcing_vec)  && (pe_group["forcing_vec"]  = state.forcing_vec)
        !isempty(state.response_vec) && (pe_group["response_vec"] = state.response_vec)

        # Response fields (ComplexF64 directly)
        response_group = haskey(pe_group, "response") ? pe_group["response"] : create_group(pe_group, "response")
        have_xi = !isnothing(state.xi_modes)
        have_b  = have_xi && !isnothing(state.b_modes)
        response_group["xi_psi"]  = have_xi ? state.xi_modes.psi   : ComplexF64[]
        response_group["b_psi"]   = have_b  ? state.b_modes.psi    : ComplexF64[]
        response_group["b_theta"] = have_b  ? state.b_modes.theta  : ComplexF64[]
        response_group["b_zeta"]  = have_b  ? state.b_modes.zeta   : ComplexF64[]

        # Singular coupling
        coupling_group = haskey(pe_group, "singular_coupling") ? pe_group["singular_coupling"] : create_group(pe_group, "singular_coupling")

        # Coupling matrices [n_rational × numpert_total]
        !isempty(state.C_resonant_flux)    && (coupling_group["C_resonant_flux"]    = state.C_resonant_flux)
        !isempty(state.C_resonant_current) && (coupling_group["C_resonant_current"] = state.C_resonant_current)
        !isempty(state.C_island_width_sq)  && (coupling_group["C_island_width_sq"]  = state.C_island_width_sq)
        !isempty(state.C_penetrated_field) && (coupling_group["C_penetrated_field"] = state.C_penetrated_field)
        !isempty(state.C_delta_prime)      && (coupling_group["C_delta_prime"]      = state.C_delta_prime)

        # Applied resonant vectors [n_rational]
        !isempty(state.resonant_flux)      && (coupling_group["resonant_flux"]      = state.resonant_flux)
        !isempty(state.resonant_current)   && (coupling_group["resonant_current"]   = state.resonant_current)
        !isempty(state.island_width_sq)    && (coupling_group["island_width_sq"]    = state.island_width_sq)
        !isempty(state.penetrated_field)   && (coupling_group["penetrated_field"]   = state.penetrated_field)
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
        energy_group["plasma_energy"] = state.plasma_energy
        energy_group["vacuum_energy"] = state.vacuum_energy
        energy_group["total_energy"]  = state.total_energy
    end
end
