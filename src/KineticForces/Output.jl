"""
    Output

HDF5 output for KineticForces results.
All results accumulate in KineticForcesState during computation,
then write to gpec.h5 in a single pass.
"""

"""
    write_to_hdf5!(h5file::HDF5.File, state::KineticForcesState; dVdpsi_spline=nothing,
                   species_label=nothing)

Write KineticForces results to the "KineticForces" group in gpec.h5.

# Arguments
- `h5file::HDF5.File`: Open HDF5 file handle
- `state::KineticForcesState`: Accumulated computation results
- `dVdpsi_spline`: Optional dV/dψ_N profile interpolant; when given, dV/dψ_N is
  written at the quadrature points so the torque density dT/dV = (dT/dψ)/(dV/dψ)
  is directly available
- `species_label`: `nothing` writes the run total to `KineticForces/<method>/`; a label
  (e.g. `"ion_z1_m2"`, `"electron"`) writes one species' contribution to
  `KineticForces/PerSpecies/<label>/<method>/`. A multi-species run calls this once per
  species and once for the summed total, so the group is opened-or-created each time.
"""
function write_to_hdf5!(h5file::HDF5.File, state::KineticForcesState; dVdpsi_spline=nothing,
    species_label::Union{Nothing,AbstractString}=nothing)
    root = haskey(h5file, "KineticForces") ? h5file["KineticForces"] : create_group(h5file, "KineticForces")
    g = root
    if species_label !== nothing
        per = if haskey(root, "PerSpecies")
            root["PerSpecies"]
        else
            p = create_group(root, "PerSpecies")
            attributes(p)["long_name"] = "per-species NTV contributions; their sum is the KineticForces total"
            p
        end
        g = create_group(per, species_label)
    end

    # Method groups actually written here, so the metadata pass annotates the method level
    # and never mistakes the PerSpecies container for a method token.
    method_names = String[]

    for (method_name, result) in state.method_results
        mg = create_group(g, method_name)
        push!(method_names, method_name)
        mg["n"] = result.nn
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

        # Per-ψ complex torque profiles from quadrature evaluation points:
        # dT/dψ integrand values and cumulative T(ψ) via trapezoidal integration.
        if !isempty(result.psi_grid)
            mg["psi"] = result.psi_grid
            mg["dTdpsi"] = result.dtdpsi
            mg["T"] = result.t_cumulative
            if dVdpsi_spline !== nothing
                mg["dVdpsi"] = [dVdpsi_spline(p) for p in result.psi_grid]
            end
        end

        if !isempty(result.records)
            write_integration_records!(mg, result.records)
        end
    end

    # Write the six drift-kinetic coefficient matrices if present
    for (method_name, mat) in state.kinetic_matrices
        method_g = if haskey(g, method_name)
            g[method_name]
        else
            push!(method_names, method_name)
            create_group(g, method_name)
        end
        mat_g = create_group(method_g, "KineticMatrices")
        for (letter, k) in _KINETIC_MATRIX_LETTERS
            mat_g[letter] = mat[:, :, k]
        end
    end

    # Metadata pass: method tokens are data-driven, so annotate each method group.
    for method_name in method_names
        Utilities.HDF5Annotations.annotate!(g[method_name], KF_METHOD_H5_ANNOTATIONS)
    end
end

# Storage slice → Logan 2015 matrix letter (Eqs 7.30-7.35); see BounceAveraging.jl
# for the Hermitian/full packing details.
const _KINETIC_MATRIX_LETTERS = (("A", 1), ("B", 2), ("C", 3), ("D", 4), ("E", 5), ("H", 6))

# Metadata table per KineticForces/<method>/ group (paths relative to the method group).
# The NTV torque and kinetic energy follow Logan et al. (2013); the six drift-kinetic
# coefficient matrices are Logan 2015 Eqs 7.30-7.35, stored in the energy (δW)
# normalization (torque integrand divided by 2in).
const KF_METHOD_H5_ANNOTATIONS = [
    "n" => (; long_name="toroidal mode number n of this torque calculation"),
    "total_torque" => (; long_name="total NTV toroidal torque T_φ", units="N*m"),
    "total_energy" => (; long_name="total perturbed kinetic energy δW_k = Im(T)/(2n)", units="J"),
    "psi_nsteps" => (; long_name="number of ψ_N quadrature evaluations"),
    "panel_psi" => (; long_name="ψ_N panel boundaries of the radial quadrature"),
    "resonance_psi" => (; long_name="ψ_N of located kinetic-resonance surfaces"),
    "psi" => (; long_name="normalized poloidal flux ψ_N at quadrature evaluation points", scale="psi"),
    "dTdpsi" => (; long_name="complex torque density dT/dψ_N = dT_φ/dψ_N + 2i·n·dδW_k/dψ_N; divide by dVdpsi for dT/dV", units="N*m", dims=("psi",), attach=(1 => "psi",)),
    "T" => (; long_name="cumulative complex torque T(ψ_N) = T_φ + 2i·n·δW_k (trapezoidal)", units="N*m", dims=("psi",), attach=(1 => "psi",)),
    "dVdpsi" => (; long_name="flux-surface volume derivative dV/dψ_N at quadrature points", units="m^3", dims=("psi",), attach=(1 => "psi",)),
    "EnergyIntegrals/psi" => (; long_name="ψ_N of each energy-integration record"),
    "EnergyIntegrals/lambda" => (; long_name="pitch λ = μB0/E of each record"),
    "EnergyIntegrals/ell" => (; long_name="bounce harmonic ℓ of each record"),
    "EnergyIntegrals/leff" => (; long_name="effective bounce harmonic ℓ_eff = ℓ + n·q (circulating) or ℓ (trapped)"),
    "EnergyIntegrals/torque" => (; long_name="complex torque contribution of the record", units="N*m"),
    "EnergyIntegrals/kinetic_energy" => (; long_name="complex kinetic energy contribution of the record", units="J"),
    "EnergyIntegrals/trajectory_offsets" => (; long_name="ragged-array offsets: record k spans offsets[k]+1:offsets[k+1] of the *_all arrays"),
    "EnergyIntegrals/x_all" => (; long_name="normalized energy x = E/T abscissae of all integration trajectories (concatenated)"),
    "EnergyIntegrals/integrand_all" => (; long_name="complex energy-space torque integrand along all trajectories (concatenated)"),
    "EnergyIntegrals/integral_all" => (; long_name="complex cumulative energy-space integral along all trajectories (concatenated)"),
    "KineticMatrices/A" => (; long_name="Logan 2015 drift-kinetic coefficient matrix A = W_Z†W_Z (energy normalization)", dims=("mode_row", "mode_col")),
    "KineticMatrices/B" => (; long_name="Logan 2015 drift-kinetic coefficient matrix B = W_Z†W_X (energy normalization)", dims=("mode_row", "mode_col")),
    "KineticMatrices/C" => (; long_name="Logan 2015 drift-kinetic coefficient matrix C = W_Z†W_Y (energy normalization)", dims=("mode_row", "mode_col")),
    "KineticMatrices/D" => (; long_name="Logan 2015 drift-kinetic coefficient matrix D = W_X†W_X (energy normalization)", dims=("mode_row", "mode_col")),
    "KineticMatrices/E" => (; long_name="Logan 2015 drift-kinetic coefficient matrix E = W_X†W_Y (energy normalization)", dims=("mode_row", "mode_col")),
    "KineticMatrices/H" => (; long_name="Logan 2015 drift-kinetic coefficient matrix H = W_Y†W_Y (energy normalization)", dims=("mode_row", "mode_col"))
]

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
    rg["torque"] = [r.torque for r in records]
    rg["kinetic_energy"] = [r.kinetic_energy for r in records]

    # Variable-length trajectories: concatenate all, store offsets for indexing
    lengths = [length(r.x_trajectory) for r in records]
    offsets = cumsum([0; lengths])
    rg["trajectory_offsets"] = offsets

    if sum(lengths) > 0
        rg["x_all"] = vcat([r.x_trajectory for r in records]...)
        rg["integrand_all"] = vcat([r.integrand_trajectory for r in records]...)
        rg["integral_all"] = vcat([r.integral_trajectory for r in records]...)
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
