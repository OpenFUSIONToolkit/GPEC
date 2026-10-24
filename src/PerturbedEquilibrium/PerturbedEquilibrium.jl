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
export plasma_response
export compute_perturbed_equilibrium
export write_outputs_to_HDF5
export ResonantCoupling, DominantCoupling, dominant_coupling, rootarea_field, coupling_overlap, check_mode_basis, CORE_PSI_LOW, CORE_PSI_HIGH

"""
    plasma_response(ffs; compute_singular_coupling=true, verbose=true) -> PlasmaResponse

Build the source-independent half of a perturbed equilibrium from a force-free-states result: the
control-surface response operators (inductances, permeability, reluctance; Park et al., Phys.
Plasmas 14, 052110 (2007)) and, with `compute_singular_coupling`, the singular-coupling matrices and
their dominant modes (Park, Phys. Plasmas 16, 056115 (2009)). Apply it to an external field with
`perturbed_equilibrium(resp, rmp)` as many times as needed.

Products the producing integrator could not supply (free-boundary energies, a ξ solution) leave the
corresponding operators empty: the step warns and is skipped. Galerkin and `:matched` results publish
no free-boundary δW yet, so building a response from one is an error.
"""
function plasma_response(ffs::ForceFreeStatesResult; compute_singular_coupling::Bool=true, verbose::Bool=true)::PlasmaResponse
    # Galerkin and matched results publish no plasma edge response wp, so no operator could be built.
    if ffs.free_boundary === nothing && (ffs.integrator === :galerkin || ffs.closure === :matched)
        error("plasma_response needs the free-boundary δW, which a $(ffs.integrator) result with $(ffs.closure) closure " *
              "does not publish yet; use a forward solve")
    end

    m_modes, n_modes = mode_arrays(ffs)
    # A run has at most one ξ solution, already closed at the rationals and carrying populated
    # Ξ′ / Ξ_s stores. The gal-native basis takes the analytic Galerkin Ξ′ downstream.
    solution = ffs.solution
    resp = PlasmaResponse(; ffs, m_modes, n_modes, regularization_width=default_regularization_width(ffs),
        odet_from_gal=solution !== nothing && solution.basis === :gal_native)

    if ForceFreeStates.require(ffs, :free_boundary, "plasma response calculation") &&
       ForceFreeStates.require_solution(ffs, "plasma response calculation")
        build_response_operators!(resp; verbose)
        if compute_singular_coupling
            build_coupling_operators!(resp; verbose)
            build_dominant_coupling!(resp; verbose)
        end
    end
    return resp
end

"""
    compute_perturbed_equilibrium(resp, forcing; regularization_width=resp.regularization_width, verbose=true,
                                  preloaded_coil_sets=CoilSet[], runtimes=nothing)::PerturbedEquilibriumState

Drive the operators of `resp` with one external field: materialize `forcing` (a `ForcingTermsControl`
or any [`ForcingTerms.RMPField`](@ref)) on the control surface, then apply the response and the
singular coupling to it and reconstruct the perturbed profiles. Steps whose operators `resp` does not
carry are skipped, leaving only the forcing spectrum.

## Arguments

  - `regularization_width`: width of the ideal 1/(m−nq) smoothing in the profile reconstruction
    (Fortran `reg_spot`); must be ≥ 0
  - `preloaded_coil_sets`: coil geometry to integrate instead of reading it (the gpec.h5 replay path)
  - `runtimes`: optional collector; receives `"forcing_terms" => dt` for the forcing-mode
    materialization (coil Biot-Savart onto the plasma surface)
"""
function compute_perturbed_equilibrium(
    resp::PlasmaResponse,
    forcing::Union{ForcingTerms.ForcingTermsControl,ForcingTerms.RMPField};
    regularization_width::Real=resp.regularization_width,
    verbose::Bool=true,
    preloaded_coil_sets::Vector{CoilSet}=CoilSet[],
    runtimes::Union{Nothing,Vector{Pair{String,Float64}}}=nothing
)::PerturbedEquilibriumState
    ffs = resp.ffs
    ft_start = time()
    modes, coil_sets = materialize_forcing_modes(ffs, forcing; dir_path=ffs.dir_path, preloaded_coil_sets, verbose)
    runtimes === nothing || push!(runtimes, "forcing_terms" => time() - ft_start)
    state = PerturbedEquilibriumState(; response=resp, forcing_modes=modes,
        coil_sets=isempty(coil_sets) ? preloaded_coil_sets : coil_sets, regularization_width=Float64(regularization_width))

    isempty(resp.permeability_flux) || apply_response!(state; verbose)
    isempty(resp.coupling_flux) || apply_coupling!(state; verbose)
    if !isempty(resp.dominant_right_singular_vectors) && !isempty(state.forcing_b_rootarea)
        state.dominant_forcing_overlap = resp.dominant_right_singular_vectors' * state.forcing_b_rootarea
    end
    return state
end

"""
    materialize_forcing_modes(ffs, forcing; dir_path, preloaded_coil_sets=CoilSet[], verbose=false)
        -> (modes::Vector{ForcingMode}, coil_sets::Vector{CoilSet})

Pure computation of the control-surface forcing spectrum: turn a forcing description into
fresh `ForcingMode`s, for every toroidal mode of the solve and in the unit-norm convention
the response step consumes. Nothing is mutated — the caller owns where the result lands.

Coil formats build the geometry (or reuse `preloaded_coil_sets` from the gpec.h5 replay
path) and integrate the Biot-Savart field on the control surface; file formats read the
modes from `dir_path` and convert their normalization onto the run's `mlow:mhigh`. Both branches evaluate on
`ffs.psilim`, the integration limit. The second return value is the coil geometry actually
used (empty for file formats), which the writer snapshots for replay.
"""
function materialize_forcing_modes(
    ffs::ForceFreeStatesResult,
    forcing::ForcingTerms.ForcingTermsControl;
    dir_path::AbstractString,
    preloaded_coil_sets::Vector{ForcingTerms.CoilSet}=ForcingTerms.CoilSet[],
    verbose::Bool=false
)
    equil = ffs.equil
    modes = ForcingMode[]
    coil_sets = ForcingTerms.CoilSet[]

    if forcing.forcing_data_format == "coil"
        cfg = ForcingTerms.CoilConfig(forcing)
        coil_sets = isempty(preloaded_coil_sets) ?
                    ForcingTerms.load_coil_sets(cfg, ffs.nlow; equil=equil) : preloaded_coil_sets
        for n in ffs.nlow:ffs.nhigh
            modes_n = ForcingMode[]
            ForcingTerms.compute_coil_forcing_modes!(
                modes_n, coil_sets, equil, cfg, n, ffs.mlow, ffs.mhigh;
                psi=ffs.psilim, verbose=verbose
            )
            append!(modes, modes_n)
        end
    else
        file_modes = ForcingMode[]
        norm_tag = load_forcing_data!(file_modes, dir_path, forcing.forcing_data_file, forcing.forcing_data_format, verbose)
        for n in ffs.nlow:ffs.nhigh
            modes_n = filter(m -> m.n == n, file_modes)
            isempty(modes_n) && continue
            # Same control surface as the coil branch above: psilim, the integration
            # limit (Fortran: gpec/gpec.f:431 `field_bs_psi(psilim, ...)`). Without it
            # the normalization was taken on the equilibrium-spline limit, which differs
            # whenever rational_offset_fraction/qhigh/dW_edge_scan_start truncation moves psilim inward.
            append!(modes, convert_forcing_normalization(modes_n, norm_tag, equil, n, ffs.mlow:ffs.mhigh; psi=ffs.psilim))
        end
    end

    return modes, coil_sets
end

"""
    materialize_forcing_modes(ffs, rmp::ForcingTerms.RMPSource; kwargs...) -> (modes, coil_sets)

Materialize a single-source [`ForcingTerms.RMPField`](@ref) leaf and apply its weight. The
returned modes are fresh objects, so weighting never reaches back into any caller state.
"""
function materialize_forcing_modes(ffs::ForceFreeStatesResult, rmp::ForcingTerms.RMPSource; kwargs...)
    modes, coil_sets = materialize_forcing_modes(ffs, rmp.ctrl; kwargs...)
    if rmp.scale != 1
        modes = [ForcingMode(; n=mode.n, m=mode.m, amplitude=rmp.scale * mode.amplitude) for mode in modes]
    end
    return modes, coil_sets
end

"""
    materialize_forcing_modes(ffs, rmp::ForcingTerms.RMPSpectrum; kwargs...) -> (modes, coil_sets)

Materialize a spectrum-literal leaf: fresh copies of its unit-norm modes, weighted. Nothing is read
or integrated, and no coil geometry is produced.
"""
function materialize_forcing_modes(::ForceFreeStatesResult, rmp::ForcingTerms.RMPSpectrum; kwargs...)
    return [ForcingMode(; n=mode.n, m=mode.m, amplitude=rmp.scale * mode.amplitude) for mode in rmp.modes], ForcingTerms.CoilSet[]
end

"""
    materialize_forcing_modes(ffs, rmp::ForcingTerms.RMPFieldSum; kwargs...) -> (modes, coil_sets)

Materialize a lazy linear combination of forcing sources: each weighted leaf is evaluated
against the equilibrium independently, then the mode amplitudes are summed per (n, m) — the
perturbed-equilibrium response is linear in the forcing, so this equals forcing with the
combined field. The merged modes are sorted by (n, m) for a deterministic order. Coil
geometry from every coil-format leaf is concatenated in the second return value; the
preloaded-geometry shortcut does not apply to sums (the replay path injects materialized
modes upstream and never materializes a sum).
"""
function materialize_forcing_modes(ffs::ForceFreeStatesResult, rmp::ForcingTerms.RMPFieldSum;
    dir_path::AbstractString, preloaded_coil_sets::Vector{ForcingTerms.CoilSet}=ForcingTerms.CoilSet[],
    verbose::Bool=false)
    amplitudes = Dict{Tuple{Int,Int},ComplexF64}()
    coil_sets = ForcingTerms.CoilSet[]
    for term in rmp.terms
        term_modes, term_coils = materialize_forcing_modes(ffs, term; dir_path=dir_path, verbose=verbose)
        for mode in term_modes
            key = (mode.n, mode.m)
            amplitudes[key] = get(amplitudes, key, 0.0 + 0.0im) + mode.amplitude
        end
        append!(coil_sets, term_coils)
    end
    modes = [ForcingMode(; n=n, m=m, amplitude=amplitudes[(n, m)]) for (n, m) in sort!(collect(keys(amplitudes)))]
    return modes, coil_sets
end

end # module PerturbedEquilibrium
