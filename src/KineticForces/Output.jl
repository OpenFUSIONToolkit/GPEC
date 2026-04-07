"""
    Output

HDF5 output for KineticForces results.
All results accumulate in KineticForcesState during computation,
then write to gpec.h5 in a single pass.
"""

"""
    write_to_hdf5!(h5file::HDF5.File, state::KineticForcesState)

Write all KineticForces results to the "kinetic_forces" group in gpec.h5.

# Arguments
- `h5file::HDF5.File`: Open HDF5 file handle
- `state::KineticForcesState`: Accumulated computation results
"""
function write_to_hdf5!(h5file::HDF5.File, state::KineticForcesState)
    g = create_group(h5file, "kinetic_forces")

    for (method_name, result) in state.method_results
        mg = create_group(g, method_name)
        mg["nn"] = result.nn
        mg["psi"] = result.psi_grid
        mg["torque_real"] = real.(result.torque_vs_psi)
        mg["torque_imag"] = imag.(result.torque_vs_psi)
        mg["energy_real"] = real.(result.energy_vs_psi)
        mg["energy_imag"] = imag.(result.energy_vs_psi)
        mg["total_torque"] = [real(result.total_torque), imag(result.total_torque)]
        mg["total_energy"] = [real(result.total_energy), imag(result.total_energy)]

        if !isempty(result.records)
            write_integration_records!(mg, result.records)
        end
    end
end


"""
    write_integration_records!(mg::HDF5.Group, records::Vector{EnergyIntegrationResult})

Write variable-length ODE trajectory records using offset-indexed concatenated arrays.
This is the standard HDF5 ragged array pattern for storing variable-length data.

# Arguments
- `mg::HDF5.Group`: HDF5 group for this method
- `records::Vector{EnergyIntegrationResult}`: Integration records to write
"""
function write_integration_records!(mg::HDF5.Group, records::Vector{EnergyIntegrationResult})
    rg = create_group(mg, "records")
    n = length(records)

    # Scalar fields per record
    rg["psi"] = [r.psi for r in records]
    rg["lambda"] = [r.lambda for r in records]
    rg["ell"] = [r.ell for r in records]
    rg["leff"] = [r.leff for r in records]
    rg["torque_real"] = [real(r.torque) for r in records]
    rg["torque_imag"] = [imag(r.torque) for r in records]
    rg["kinetic_energy_real"] = [real(r.kinetic_energy) for r in records]
    rg["kinetic_energy_imag"] = [imag(r.kinetic_energy) for r in records]

    # Variable-length trajectories: concatenate all, store offsets for indexing
    lengths = [length(r.x_trajectory) for r in records]
    offsets = cumsum([0; lengths])
    rg["trajectory_offsets"] = offsets

    if sum(lengths) > 0
        rg["x_all"] = vcat([r.x_trajectory for r in records]...)
        rg["integrand_real_all"] = vcat([real.(r.integrand_trajectory) for r in records]...)
        rg["integrand_imag_all"] = vcat([imag.(r.integrand_trajectory) for r in records]...)
        rg["integral_real_all"] = vcat([real.(r.integral_trajectory) for r in records]...)
        rg["integral_imag_all"] = vcat([imag.(r.integral_trajectory) for r in records]...)
    end
end


"""
    print_summary(state::KineticForcesState; verbose::Bool=false)

Print a summary of KineticForces results to stdout.

# Arguments
- `state::KineticForcesState`: Accumulated computation results
- `verbose::Bool`: Print detailed per-surface results
"""
function print_summary(state::KineticForcesState; verbose::Bool=false)
    for (method_name, result) in state.method_results
        @printf("%-8s  T_phi = %11.3e   2n*dW_k = %11.3e\n",
                method_name, real(result.total_torque), imag(result.total_torque))
        if verbose && !isempty(result.psi_grid)
            for i in eachindex(result.psi_grid)
                @printf("  psi = %8.5f  T = %11.3e + %11.3e i\n",
                        result.psi_grid[i],
                        real(result.torque_vs_psi[i]),
                        imag(result.torque_vs_psi[i]))
            end
        end
    end
end
