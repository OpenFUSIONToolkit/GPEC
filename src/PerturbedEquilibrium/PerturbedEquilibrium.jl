module PerturbedEquilibrium

# Imports
using HDF5
using Printf
using LinearAlgebra
using Statistics
using AdaptiveArrayPools
using FastInterpolations

# Import parent modules
import ..Equilibrium
import ..ForceFreeStates
import ..ForceFreeStates: SolutionProfiles, ForceFreeStatesResult, MatrixSplines, MetricData
import ..Vacuum
import ..ForcingTerms
import ..ForcingTerms: ForcingMode, CoilSet, load_forcing_data!, convert_forcing_normalization
import ..Utilities
import ..Utilities.FourierTransforms

# Include module files
include("PerturbedEquilibriumStructs.jl")
include("ResponseMatrices.jl")
include("TorqueResponse.jl")
include("FieldReconstruction.jl")
include("Response.jl")
include("SingularCoupling.jl")
include("ResonantCoupling.jl")
include("Utils.jl")

# Export main types
export PlasmaResponse
export PerturbedEquilibriumState
export ForcingMode

# Export main functions
export plasma_response, perturbed_equilibrium
export compute_perturbed_equilibrium
export save_to_hdf5
export ResonantCoupling, DominantCoupling, dominant_coupling, rootarea_field, coupling_overlap, check_mode_basis, CORE_PSI_LOW, CORE_PSI_HIGH
export TorqueResponse, torque_response_matrix, torque_profile

"""
    plasma_response(ffs; compute_singular_coupling=true, compute_torque_response=false, verbose=false) -> PlasmaResponse

Build the source-independent half of a perturbed equilibrium from a force-free-states result: the
control-surface response operators (inductances, permeability, reluctance; Park et al., Phys.
Plasmas 14, 052110 (2007)) and, with `compute_singular_coupling`, the singular-coupling matrices and
their dominant modes (Park, Phys. Plasmas 16, 056115 (2009)), and with `compute_torque_response` the
ψ-resolved torque response `T_xe` of a single-`n` forward kinetic solve. Apply it to an external field
with `perturbed_equilibrium(resp, field)` as many times as needed.

A result without a ξ solution leaves the operators empty: the step warns and is skipped. A result without
the free-boundary δW is an error (it needs `vac_flag`, and for Galerkin also `error_field_response`), and so is
a solve with a wall: the surface inductance is free-space, and the response needs Λ from the same no-wall vacuum.
A `:matched` result gets only the `_asymptotic` singular-coupling metrics and the penetrated field.
"""
function plasma_response(ffs::ForceFreeStatesResult; compute_singular_coupling::Bool=true, compute_torque_response::Bool=false,
    verbose::Bool=false)::PlasmaResponse
    ffs.wall_settings === nothing ||
        error(
            "plasma_response needs a force-free-states solve without a wall: its surface inductance L is free-space, " *
            "so Λ from a wall run would mix two vacuum models. Solve with wall=nothing (no [Wall] section)."
        )
    ffs.free_boundary === nothing &&
        error(
            "plasma_response needs the free-boundary δW, which this $(ffs.integrator) result does not carry: solve with " *
            "vac_flag=true (and Galerkin(; error_field_response=true) for a Galerkin solve)"
        )

    m_modes, n_modes = mode_arrays(ffs)
    # A run has at most one ξ solution, already closed at the rationals and carrying populated
    # Ξ′ / Ξ_s stores. A unit-edge solution's stored Ξ′ is used as is downstream.
    solution = ffs.solution
    resp = PlasmaResponse(; ffs, m_modes, n_modes, compute_singular_coupling, compute_torque_response,
        unit_edge_basis=solution !== nothing && solution.basis === :unit_edge)

    if ForceFreeStates.require_solution(ffs, "plasma response calculation")
        build_response_operators!(resp; verbose)
        if compute_singular_coupling
            build_coupling_operators!(resp; verbose)
            build_dominant_coupling!(resp; verbose)
        end
    end
    compute_torque_response && build_torque_response!(resp; verbose)
    return resp
end

"""
    perturbed_equilibrium(resp, field; regularization_width=0.0, verbose=false, coil_sets=nothing)
        -> PerturbedEquilibriumState

Drive the plasma response `resp` (from [`plasma_response`](@ref)) with the external field `field`, an
[`ForcingField`](@ref GeneralizedPerturbedEquilibrium.ForcingTerms.ForcingField), and return the perturbed-equilibrium result. One response serves any number of
fields, so a scan rebuilds no operator.

`regularization_width` smooths the ideal 1/(m−nq) divergence in the profile reconstruction (Fortran
`reg_spot`); its default is 0, and 0.05 is typical for driving KineticForces with an ideal solve.
Call `save_to_hdf5(target, pe)` to save the result. `coil_sets` injects already-built coil geometry (the gpec.h5 replay path), bypassing the geometry read.

```julia
resp = plasma_response(ffs)
pe = perturbed_equilibrium(resp, ForcingField("forcing.dat"))
```
"""
function perturbed_equilibrium(
    resp::PlasmaResponse,
    field::ForcingTerms.ForcingField;
    regularization_width::Real=0.0,
    verbose::Bool=false,
    coil_sets::Union{Nothing,Vector{CoilSet}}=nothing
)
    # Injected coil geometry (gpec.h5 replay with `--coil-source coils`) lets the coil field
    # be recomputed from stored geometry without the .dat/.h5 files.
    pe_state = compute_perturbed_equilibrium(resp, field; regularization_width, verbose,
        preloaded_coil_sets=coil_sets === nothing ? CoilSet[] : copy(coil_sets))

    return pe_state
end

"""
    compute_perturbed_equilibrium(resp, forcing; regularization_width=0.0, verbose=true,
                                  preloaded_coil_sets=CoilSet[])::PerturbedEquilibriumState

Drive the operators of `resp` with one external field: materialize `forcing` (any
[`ForcingTerms.ForcingField`](@ref)) on the control surface, then apply the response and the
singular coupling to it, reconstruct the perturbed profiles, and contract any torque response with
the forcing sources. Steps whose operators `resp` does not carry are skipped, leaving only the
forcing spectrum.

## Arguments

  - `regularization_width`: width of the ideal 1/(m−nq) smoothing in the profile reconstruction
    (Fortran `reg_spot`); must be ≥ 0
  - `preloaded_coil_sets`: coil geometry to integrate instead of reading it (the gpec.h5 replay path)
"""
function compute_perturbed_equilibrium(
    resp::PlasmaResponse,
    forcing::ForcingTerms.ForcingField;
    regularization_width::Real=0.0,
    verbose::Bool=true,
    preloaded_coil_sets::Vector{CoilSet}=CoilSet[]
)::PerturbedEquilibriumState
    ffs = resp.ffs
    sources, coil_sets = materialize_forcing_modes(ffs, forcing; preloaded_coil_sets, verbose)
    state = PerturbedEquilibriumState(; response=resp, forcing, forcing_modes=_total_forcing(sources),
        source_names=_unique_names(first.(sources)), source_modes=last.(sources),
        coil_sets=isempty(coil_sets) ? preloaded_coil_sets : coil_sets, regularization_width=Float64(regularization_width))

    isempty(resp.permeability_flux) || apply_response!(state; verbose)
    isempty(resp.coupling_flux) || apply_coupling!(state; verbose)
    isempty(resp.T_xe) || apply_torque_response!(state; verbose)
    if !isempty(resp.dominant_right_singular_vectors) && !isempty(state.forcing_b_rootarea)
        state.dominant_forcing_overlap = resp.dominant_right_singular_vectors' * state.forcing_b_rootarea
        state.dominant_source_overlap = resp.dominant_right_singular_vectors' * state.source_b_rootarea
    end
    return state
end

# The forcing of all sources: a lone source's modes as they are, otherwise the (n, m) sums sorted by (n, m).
function _total_forcing(sources::Vector{Pair{String,Vector{ForcingMode}}})
    length(sources) == 1 && return [ForcingMode(; n=m.n, m=m.m, amplitude=m.amplitude) for m in only(sources).second]
    amplitudes = Dict{Tuple{Int,Int},ComplexF64}()
    for (_, modes) in sources, mode in modes
        amplitudes[(mode.n, mode.m)] = get(amplitudes, (mode.n, mode.m), 0.0 + 0.0im) + mode.amplitude
    end
    return [ForcingMode(; n=n, m=m, amplitude=amplitudes[(n, m)]) for (n, m) in sort!(collect(keys(amplitudes)))]
end

# Source names made unique by appending _2, _3, … to repeats.
function _unique_names(names::Vector{String})
    seen = Dict{String,Int}()
    return map(names) do name
        k = seen[name] = get(seen, name, 0) + 1
        k == 1 ? name : "$(name)_$k"
    end
end

# Sum the per-member spectra of each label, in order of first appearance, and apply a leaf's weight.
function _grouped_sources(labels::Vector{String}, member_modes::Vector{Vector{ForcingMode}}, scale::ComplexF64)
    return map(unique(labels)) do label
        modes = _total_forcing([label => member for (l, member) in zip(labels, member_modes) if l == label])
        label => [ForcingMode(; n=m.n, m=m.m, amplitude=scale * m.amplitude) for m in modes]
    end
end

"""
    materialize_forcing_modes(ffs, field; preloaded_coil_sets=CoilSet[], verbose=false)
        -> (sources::Vector{Pair{String,Vector{ForcingMode}}}, coil_sets::Vector{CoilSet})

The control-surface spectrum of each forcing source of `field`, in the unit-norm Φ_x convention the response consumes,
on the control surface `ffs.psilim` for every toroidal mode of the solve, with each leaf's weight applied. A source is a
coil set, a forcing file (named after it), a spectrum literal (`"spectrum"`) or a magnetic material set; sets sharing a
`group` are one source named by it. A sum concatenates its leaves' sources. The second return value is the coil geometry
used, which the writer snapshots for replay (`preloaded_coil_sets` replaces it on the gpec.h5 replay path).
"""
function materialize_forcing_modes(
    ffs::ForceFreeStatesResult,
    field::ForcingTerms.ForcingCoils;
    preloaded_coil_sets::Vector{ForcingTerms.CoilSet}=ForcingTerms.CoilSet[],
    verbose::Bool=false
)
    equil, cfg = ffs.equil, field.config
    coil_sets = isempty(preloaded_coil_sets) ? ForcingTerms.load_coil_sets(cfg, ffs.nlow; equil=equil) : preloaded_coil_sets
    labels = length(cfg.coil_sets) == length(coil_sets) ?
             [isempty(sc.group) ? cs.name : sc.group for (sc, cs) in zip(cfg.coil_sets, coil_sets)] : [cs.name for cs in coil_sets]
    per_set = [ForcingMode[] for _ in coil_sets]
    for n in ffs.nlow:ffs.nhigh
        grid = ForcingTerms.CoilForcingGrid(equil, cfg, n; psi=ffs.psilim)
        verbose && @info "Computing coil forcing modes: mtheta=$(grid.grid.mtheta), nzeta=$(grid.grid.nzeta), n=$n, $(length(coil_sets)) coil sets"
        for (k, cs) in enumerate(coil_sets)
            append!(per_set[k], ForcingTerms.coil_forcing_modes(cs, grid, n, ffs.mlow, ffs.mhigh))
        end
    end
    return _grouped_sources(labels, per_set, field.scale), coil_sets
end

function materialize_forcing_modes(ffs::ForceFreeStatesResult, field::ForcingTerms.ForcingFile; verbose::Bool=false, kwargs...)
    file_modes = ForcingMode[]
    norm_tag = load_forcing_data!(file_modes, field.path, field.format, verbose)
    modes = ForcingMode[]
    for n in ffs.nlow:ffs.nhigh
        modes_n = filter(m -> m.n == n, file_modes)
        isempty(modes_n) && continue
        # Normalize on the same control surface as the coil field: psilim, the integration limit.
        append!(modes, convert_forcing_normalization(modes_n, norm_tag, ffs.equil, n, ffs.mlow:ffs.mhigh; psi=ffs.psilim))
    end
    return _grouped_sources([basename(field.path)], [modes], field.scale), ForcingTerms.CoilSet[]
end

function materialize_forcing_modes(ffs::ForceFreeStatesResult, field::ForcingTerms.ForcingMagneticMaterial; verbose::Bool=false, kwargs...)
    equil = ffs.equil
    labels = [isempty(g) ? m.name : g for (m, g) in zip(field.materials, field.groups)]
    magnetizations = [ForcingTerms.saturated_magnetization(material, equil) for material in field.materials]
    cfg = ForcingTerms.CoilConfig(; mtheta_coil=field.mtheta, nzeta_coil=field.nzeta)
    per_set = [ForcingMode[] for _ in field.materials]
    for n in ffs.nlow:ffs.nhigh
        verbose && @info "Computing magnetic material forcing modes for n = $n"
        for k in eachindex(field.materials)
            append!(per_set[k], ForcingTerms.material_forcing_modes(field.materials[k:k], magnetizations[k:k], equil, cfg, n, ffs.mlow, ffs.mhigh; psi=ffs.psilim))
        end
    end
    return _grouped_sources(labels, per_set, field.scale), ForcingTerms.CoilSet[]
end

function materialize_forcing_modes(::ForceFreeStatesResult, field::ForcingTerms.ForcingSpectrum; kwargs...)
    return _grouped_sources(["spectrum"], [field.modes], field.scale), ForcingTerms.CoilSet[]
end

function materialize_forcing_modes(ffs::ForceFreeStatesResult, field::ForcingTerms.ForcingFieldSum;
    preloaded_coil_sets::Vector{ForcingTerms.CoilSet}=ForcingTerms.CoilSet[],
    verbose::Bool=false)
    sources = Pair{String,Vector{ForcingMode}}[]
    coil_sets = ForcingTerms.CoilSet[]
    for term in field.terms
        term_sources, term_coils = materialize_forcing_modes(ffs, term; verbose=verbose)
        append!(sources, term_sources)
        append!(coil_sets, term_coils)
    end
    return sources, coil_sets
end

end # module PerturbedEquilibrium
