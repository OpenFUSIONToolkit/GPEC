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
import ..ForceFreeStates: SolutionProfiles, ForceFreeStatesResult, FourFitVars, MetricData
import ..Vacuum
import ..ForcingTerms
import ..ForcingTerms: ForcingMode, CoilSet, load_forcing_data!, convert_forcing_normalization!
import ..Utilities
import ..Utilities.FourierTransforms
import DelimitedFiles: readdlm

# Include module files
include("PerturbedEquilibriumStructs.jl")
include("ResponseMatrices.jl")
include("FieldReconstruction.jl")
include("Response.jl")
include("SingularCoupling.jl")
include("Utils.jl")

# Export main types
export PerturbedEquilibriumControl
export PerturbedEquilibriumInternal
export PerturbedEquilibriumState
export ForcingMode

# Export main functions
export compute_perturbed_equilibrium
export write_outputs_to_HDF5

"""
    compute_perturbed_equilibrium(ffs, forcing, ctrl, intr)::PerturbedEquilibriumState

Main entry point for perturbed equilibrium calculations.

Computes plasma response to external forcing and calculates singular layer
coupling metrics. Every ForceFreeStates input — the equilibrium, the mode space, the
metric and matrix fits, the free-boundary energies and the ξ solution — is read off `ffs`.
Products the producing integrator could not supply gate the corresponding calculation:
the step warns and is skipped instead of erroring.

## Arguments

  - `ffs`: `ForceFreeStates.ForceFreeStatesResult` from the stability solve
  - `forcing`: the external-field description — a `ForcingTermsControl` (TOML path) or any [`ForcingTerms.RMPField`](@ref)
  - `ctrl`: Control parameters from [PerturbedEquilibrium] section
  - `intr`: Internal state variables

## Returns

  - `PerturbedEquilibriumState`: Calculation results
"""
function compute_perturbed_equilibrium(
    ffs::ForceFreeStatesResult,
    forcing::Union{ForcingTerms.ForcingTermsControl,ForcingTerms.RMPField},
    ctrl::PerturbedEquilibriumControl,
    intr::PerturbedEquilibriumInternal
)::PerturbedEquilibriumState

    state = PerturbedEquilibriumState()
    equil = ffs.equil
    ffit = ffs.ffit
    mthvac = ffs.control.mthvac

    # Step 0: Initialize mode arrays for convenient indexing
    initialize_mode_arrays!(intr, ffs)

    # A run has at most one ξ solution, already closed at the rationals and carrying populated
    # Ξ′ / Ξ_s stores. The gal-native basis takes the analytic Galerkin Ξ′ downstream.
    solution = ffs.solution
    intr.odet_from_gal = solution !== nothing && solution.basis === :gal_native

    # The one place forcing state lands on `intr`: injected (replay) modes short-circuit the
    # materialization entirely, so they are never re-converted or re-weighted.
    if isempty(intr.forcing_modes)
        modes, coil_sets = materialize_forcing_modes(ffs, forcing;
            dir_path=intr.dir_path, preloaded_coil_sets=intr.coil_sets, verbose=ctrl.verbose)
        intr.forcing_modes = modes
        isempty(coil_sets) || (intr.coil_sets = coil_sets)
    end

    # Step 2: Compute plasma response
    if ctrl.compute_response &&
       ForceFreeStates.require(ffs, :free_boundary, "plasma response calculation") &&
       ForceFreeStates.require_solution(ffs, "plasma response calculation")
        compute_plasma_response!(state, equil, solution, ffs.free_boundary.wt0, mthvac, ffs, intr, ctrl, ffs.metric, ffit)
    end

    # Step 3: Compute singular coupling metrics
    if ctrl.compute_singular_coupling &&
       ForceFreeStates.require(ffs, :free_boundary, "singular coupling calculation") &&
       ForceFreeStates.require_solution(ffs, "singular coupling calculation")
        compute_singular_coupling_metrics!(state, equil, solution, mthvac, ffs, intr, ctrl, ffit)
    end

    # Step 4: Output eigenmode fields (integrated into HDF5 output)
    # This is handled by write_outputs_to_HDF5 in main()

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
modes from `dir_path` and convert their normalization. Both branches evaluate on
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
        norm_tag = load_forcing_data!(modes, dir_path, forcing.forcing_data_file, forcing.forcing_data_format, verbose)
        for n in ffs.nlow:ffs.nhigh
            # filter returns a new Vector but still holds references to same ForcingMode objects
            modes_n = filter(m -> m.n == n, modes)
            isempty(modes_n) && continue
            m_vals = [m.m for m in modes_n]
            # Same control surface as the coil branch above: psilim, the integration
            # limit (Fortran: gpec/gpec.f:431 `field_bs_psi(psilim, ...)`). Without it
            # the normalization was taken on the equilibrium-spline limit, which differs
            # whenever dmlim/qhigh/psiedge truncation moves psilim inward.
            convert_forcing_normalization!(modes_n, norm_tag, equil, n,
                minimum(m_vals), maximum(m_vals); psi=ffs.psilim)
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
