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
        ctrl::PerturbedEquilibriumControl,
        filename::String
    )

Write perturbed equilibrium results to HDF5 file (appends to existing DCON output).

## Output Structure

```
perturbed_equilibrium/
├── forcing_modes/
│   ├── n              # Toroidal mode numbers
│   ├── m              # Poloidal mode numbers
│   ├── amplitude_real # Real parts of forcing amplitudes
│   └── amplitude_imag # Imaginary parts of forcing amplitudes
├── response/
│   ├── xi_perturbed   # Displacement field
│   └── b_perturbed    # Magnetic field perturbation
├── singular_coupling/
│   ├── coupling_coefficient_real
│   ├── coupling_coefficient_imag
│   └── resonant_amplitude
└── energies/
    ├── plasma_energy
    ├── vacuum_energy
    └── total_energy
```
"""
function write_outputs_to_HDF5(
    state::PerturbedEquilibriumState,
    intr::PerturbedEquilibriumInternal,
    ctrl::PerturbedEquilibriumControl,
    filename::String
)
    if ctrl.verbose
        println("Writing perturbed equilibrium data to $filename")
    end

    h5open(filename, "cw") do file  # "cw" = create or read/write
        # Create perturbed_equilibrium group
        pe_group = haskey(file, "perturbed_equilibrium") ? file["perturbed_equilibrium"] : create_group(file, "perturbed_equilibrium")

        # Write forcing modes
        forcing_group = haskey(pe_group, "forcing_modes") ? pe_group["forcing_modes"] : create_group(pe_group, "forcing_modes")
        n_modes = [mode.n for mode in intr.forcing_modes]
        m_modes = [mode.m for mode in intr.forcing_modes]
        amp_real = [real(mode.amplitude) for mode in intr.forcing_modes]
        amp_imag = [imag(mode.amplitude) for mode in intr.forcing_modes]

        forcing_group["n"] = n_modes
        forcing_group["m"] = m_modes
        forcing_group["amplitude_real"] = amp_real
        forcing_group["amplitude_imag"] = amp_imag

        # Write response fields (if computed)
        if size(state.xi_perturbed, 1) > 0
            response_group = haskey(pe_group, "response") ? pe_group["response"] : create_group(pe_group, "response")
            response_group["xi_perturbed_real"] = real.(state.xi_perturbed)
            response_group["xi_perturbed_imag"] = imag.(state.xi_perturbed)
            response_group["b_perturbed_real"] = real.(state.b_perturbed)
            response_group["b_perturbed_imag"] = imag.(state.b_perturbed)
        end

        # Write singular coupling metrics
        coupling_group = haskey(pe_group, "singular_coupling") ? pe_group["singular_coupling"] : create_group(pe_group, "singular_coupling")
        coupling_group["coupling_coefficient_real"] = real(state.coupling_coefficient)
        coupling_group["coupling_coefficient_imag"] = imag(state.coupling_coefficient)
        coupling_group["resonant_amplitude"] = state.resonant_amplitude

        # Write additional metrics
        for (key, val) in intr.singular_coupling_metrics
            coupling_group[key] = val
        end

        # Write energies
        energy_group = haskey(pe_group, "energies") ? pe_group["energies"] : create_group(pe_group, "energies")
        energy_group["plasma_energy"] = state.plasma_energy
        energy_group["vacuum_energy"] = state.vacuum_energy
        energy_group["total_energy"] = state.total_energy
    end

    if ctrl.verbose
        println("  Perturbed equilibrium output complete")
    end
end
