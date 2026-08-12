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
import ..ForceFreeStates: OdeState, ForceFreeStatesInternal, FourFitVars, MetricData
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
    compute_perturbed_equilibrium(
        equil, ForceFreeStates_results, wt0, mthvac, ffs_intr,
        ft_ctrl, ctrl, intr, metric, ffit
    )::PerturbedEquilibriumState

Main entry point for perturbed equilibrium calculations.

Computes plasma response to external forcing and calculates singular layer
coupling metrics.

## Arguments

  - `equil`: Equilibrium solution from Equilibrium module
  - `ForceFreeStates_results`: Stability calculation results from ForceFreeStates module
  - `wt0`: Free-boundary total-energy matrix W = wp + wv from `free_run`, or `nothing` when the free-boundary calculation was not run (response and singular coupling are then skipped)
  - `mthvac`: Vacuum poloidal grid resolution to reuse for the Green's-function solves
  - `ffs_intr`: ForceFreeStates internal state with mode information
  - `ft_ctrl`: Forcing terms control parameters from [ForcingTerms] section
  - `ctrl`: Control parameters from [PerturbedEquilibrium] section
  - `intr`: Internal state variables
  - `metric`: Metric tensor data with Fourier coefficients for Jacobian convolution
  - `ffit`: FourFitVars with stability matrix interpolants (A, B, C) for regularization

## Returns

  - `PerturbedEquilibriumState`: Calculation results
"""
function compute_perturbed_equilibrium(
    equil::Equilibrium.PlasmaEquilibrium,
    ForceFreeStates_results::OdeState,
    wt0::Union{Matrix{ComplexF64},Nothing},
    mthvac::Int,
    ffs_intr::ForceFreeStates.ForceFreeStatesInternal,
    ft_ctrl::ForcingTerms.ForcingTermsControl,
    ctrl::PerturbedEquilibriumControl,
    intr::PerturbedEquilibriumInternal,
    metric::MetricData,
    ffit::FourFitVars
)::PerturbedEquilibriumState

    state = PerturbedEquilibriumState()

    # Step 0: Initialize mode arrays for convenient indexing
    initialize_mode_arrays!(intr, ffs_intr)

    # Load forcing data. On the gpec.h5 replay path the caller preloads
    # `intr.forcing_modes` from the snapshot, so skip re-reading the original file.
    if isempty(intr.forcing_modes)
        if ft_ctrl.forcing_data_format == "coil"
            cfg = ForcingTerms.CoilConfig(ft_ctrl)
            # Reuse preloaded coil geometry (gpec.h5 rerun with `--coil-source coils`)
            # when present; otherwise build it from the TOML coil-set config. Either
            # way, retain it on `intr.coil_sets` for the rerun snapshot writer.
            coil_sets = isempty(intr.coil_sets) ?
                        ForcingTerms.load_coil_sets(cfg, ffs_intr.nlow; equil=equil) : intr.coil_sets
            intr.coil_sets = coil_sets
            for n in ffs_intr.nlow:ffs_intr.nhigh
                modes_n = ForcingMode[]
                ForcingTerms.compute_coil_forcing_modes!(
                    modes_n, coil_sets, equil, cfg, n, ffs_intr.mlow, ffs_intr.mhigh;
                    psi=ffs_intr.psilim, verbose=ctrl.verbose
                )
                append!(intr.forcing_modes, modes_n)
            end
        else
            norm_tag = load_forcing_data!(intr.forcing_modes, intr.dir_path, ft_ctrl.forcing_data_file, ft_ctrl.forcing_data_format, ctrl.verbose)
            for n in ffs_intr.nlow:ffs_intr.nhigh
                # filter returns a new Vector but still holds references to same ForcingMode objects
                modes_n = filter(m -> m.n == n, intr.forcing_modes)
                isempty(modes_n) && continue
                m_vals = [m.m for m in modes_n]
                convert_forcing_normalization!(modes_n, norm_tag, equil, n,
                    minimum(m_vals), maximum(m_vals))
            end
        end
    end

    # Step 2: Compute plasma response
    if ctrl.compute_response
        if wt0 === nothing
            @warn "Vacuum data not available. Skipping plasma response calculation. Set vac_flag=true in [ForceFreeStates] section."
        else
            compute_plasma_response!(state, equil, ForceFreeStates_results, wt0, mthvac, ffs_intr, intr, ctrl, metric, ffit)
        end
    end

    # Step 3: Compute singular coupling metrics
    if ctrl.compute_singular_coupling
        if wt0 === nothing
            @warn "Vacuum data not available. Skipping singular coupling calculation. Set vac_flag=true in [ForceFreeStates] section."
        else
            compute_singular_coupling_metrics!(state, equil, ForceFreeStates_results, mthvac, ffs_intr, intr, ctrl, ffit)
        end
    end

    # Step 4: Output eigenmode fields (integrated into HDF5 output)
    # This is handled by write_outputs_to_HDF5 in main()

    return state
end

end # module PerturbedEquilibrium
