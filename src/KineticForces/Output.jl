"""
    Output

HDF5 output for KineticForces results.
All results accumulate in KineticForcesState during computation,
then write to gpec.h5 in a single pass.
"""

"""
    write_to_hdf5!(h5file::HDF5.File, state::KineticForcesState)

Write all KineticForces results to the "KineticForces" group in gpec.h5.

# Arguments
- `h5file::HDF5.File`: Open HDF5 file handle
- `state::KineticForcesState`: Accumulated computation results
"""
function write_to_hdf5!(h5file::HDF5.File, state::KineticForcesState)
    g = create_group(h5file, "KineticForces")

    for (method_name, result) in state.method_results
        mg = create_group(g, method_name)
        mg["nn"] = result.nn
        # Torque and kinetic energy are the two real physical quantities packed into the
        # complex T (Re = T_φ, Im = 2n·δW_k); total_energy stores δW_k = Im(T)/(2n).
        mg["total_torque"] = real(result.total_torque)
        mg["total_energy"] = result.total_energy
        mg["psi_nsteps"] = result.psi_nsteps

        # ψ-quadrature panel boundaries and located kinetic-resonance surfaces (first n)
        if !isempty(result.panel_psis)
            mg["panel_psi"] = result.panel_psis
        end
        if !isempty(result.resonance_psis)
            mg["resonance_psi"] = result.resonance_psis
        end

        # Per-ψ torque profiles from quadrature evaluation points.
        # dT/dψ integrand values and cumulative T(ψ) via trapezoidal integration.
        if !isempty(result.psi_grid)
            mg["psi"] = result.psi_grid
            mg["dTdpsi_real"] = real.(result.dtdpsi)
            mg["dTdpsi_imag"] = imag.(result.dtdpsi)
            mg["T_real"] = real.(result.t_cumulative)
            mg["T_imag"] = imag.(result.t_cumulative)
        end

        if !isempty(result.records)
            write_integration_records!(mg, result.records)
        end
    end

    # Write the six drift-kinetic coefficient matrices if present
    for (method_name, mat) in state.kinetic_matrices
        method_g = haskey(g, method_name) ? g[method_name] : create_group(g, method_name)
        mat_g = create_group(method_g, "KineticMatrices")
        for k in 1:6
            mat_g["matrix_$k"] = mat[:, :, k]
        end
    end

    # Metadata pass: method tokens are data-driven, so annotate each method group.
    for method_name in keys(g)
        annotate_kinetic_forces!(g[method_name])
    end
end

# Metadata table per KineticForces/<method>/ group (paths relative to the method group).
# The NTV torque and kinetic energy follow Logan et al. (2013); the six drift-kinetic
# coefficient matrices are Logan 2015 Eqs 7.30-7.35.
const KF_METHOD_H5_ANNOTATIONS = [
    "nn" => (; long_name="toroidal mode number n of this torque calculation"),
    "total_torque" => (; long_name="total NTV toroidal torque T_φ", units="N*m"),
    "total_energy" => (; long_name="total perturbed kinetic energy δW_k = Im(T)/(2n) (unlike the T_imag profile, the 2n is divided out)", units="J"),
    "psi_nsteps" => (; long_name="number of ψ_N quadrature evaluations"),
    "panel_psi" => (; long_name="ψ_N panel boundaries of the radial quadrature"),
    "resonance_psi" => (; long_name="ψ_N of located kinetic-resonance surfaces"),
    "psi" => (; long_name="normalized poloidal flux ψ_N at quadrature evaluation points"),
    "dTdpsi_real" => (; long_name="Re dT_φ/dψ_N torque density at quadrature points", units="N*m", dims=("psi",)),
    "dTdpsi_imag" => (; long_name="Im dT_φ/dψ_N (2n·dδW_k/dψ_N energy density) at quadrature points", units="J", dims=("psi",)),
    "T_real" => (; long_name="cumulative toroidal torque T_φ(ψ_N) (trapezoidal)", units="N*m", dims=("psi",)),
    "T_imag" => (; long_name="cumulative 2n·δW_k(ψ_N) (trapezoidal; raw Im(T) — not divided by 2n like total_energy)", units="J", dims=("psi",)),
    "EnergyIntegrals/psi" => (; long_name="ψ_N of each energy-integration record"),
    "EnergyIntegrals/lambda" => (; long_name="pitch λ = μB0/E of each record"),
    "EnergyIntegrals/ell" => (; long_name="bounce harmonic ℓ of each record"),
    "EnergyIntegrals/leff" => (; long_name="effective bounce harmonic ℓ_eff of each record"),
    "EnergyIntegrals/torque_real" => (; long_name="Re of the record's torque contribution", units="N*m"),
    "EnergyIntegrals/torque_imag" => (; long_name="Im of the record's torque contribution", units="N*m"),
    "EnergyIntegrals/kinetic_energy_real" => (; long_name="Re of the record's kinetic energy contribution", units="J"),
    "EnergyIntegrals/kinetic_energy_imag" => (; long_name="Im of the record's kinetic energy contribution", units="J"),
    "EnergyIntegrals/trajectory_offsets" => (; long_name="ragged-array offsets: record k spans offsets[k]+1:offsets[k+1] of the *_all arrays"),
    "EnergyIntegrals/x_all" => (; long_name="normalized energy x = E/T abscissae of all integration trajectories (concatenated)"),
    "EnergyIntegrals/integrand_real_all" => (; long_name="Re of the energy-space torque integrand along all trajectories (concatenated)"),
    "EnergyIntegrals/integrand_imag_all" => (; long_name="Im of the energy-space torque integrand along all trajectories (concatenated)"),
    "EnergyIntegrals/integral_real_all" => (; long_name="Re of the cumulative energy-space integral along all trajectories (concatenated)"),
    "EnergyIntegrals/integral_imag_all" => (; long_name="Im of the cumulative energy-space integral along all trajectories (concatenated)"),
    "KineticMatrices/matrix_1" => (; long_name="drift-kinetic coefficient matrix 1 of 6 (Logan 2015 Eqs 7.30-7.35)", dims=("mode", "mode")),
    "KineticMatrices/matrix_2" => (; long_name="drift-kinetic coefficient matrix 2 of 6 (Logan 2015 Eqs 7.30-7.35)", dims=("mode", "mode")),
    "KineticMatrices/matrix_3" => (; long_name="drift-kinetic coefficient matrix 3 of 6 (Logan 2015 Eqs 7.30-7.35)", dims=("mode", "mode")),
    "KineticMatrices/matrix_4" => (; long_name="drift-kinetic coefficient matrix 4 of 6 (Logan 2015 Eqs 7.30-7.35)", dims=("mode", "mode")),
    "KineticMatrices/matrix_5" => (; long_name="drift-kinetic coefficient matrix 5 of 6 (Logan 2015 Eqs 7.30-7.35)", dims=("mode", "mode")),
    "KineticMatrices/matrix_6" => (; long_name="drift-kinetic coefficient matrix 6 of 6 (Logan 2015 Eqs 7.30-7.35)", dims=("mode", "mode"))
]

# Attach long_name/units/dims + the ψ_N quadrature scale to one method group.
function annotate_kinetic_forces!(method_g)
    ann = Utilities.HDF5Annotations
    ann.annotate!(method_g, KF_METHOD_H5_ANNOTATIONS)
    ann.make_scale!(method_g, "psi", "psi")
    for a in ("dTdpsi_real", "dTdpsi_imag", "T_real", "T_imag")
        ann.attach_scale!(method_g, a, 1, "psi", "psi")
    end
    return nothing
end

"""
    write_integration_records!(mg::HDF5.Group, records::Vector{EnergyIntegrationResult})

Write variable-length integration trajectory records using offset-indexed concatenated arrays.
This is the standard HDF5 ragged array pattern for storing variable-length data.

# Arguments
- `mg::HDF5.Group`: HDF5 group for this method
- `records::Vector{EnergyIntegrationResult}`: Integration records to write
"""
function write_integration_records!(mg::HDF5.Group, records::Vector{EnergyIntegrationResult})
    rg = create_group(mg, "EnergyIntegrals")

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
    end
    if verbose
        for (method_name, _) in state.kinetic_matrices
            println("  Kinetic matrices stored for: $method_name")
        end
    end
end
