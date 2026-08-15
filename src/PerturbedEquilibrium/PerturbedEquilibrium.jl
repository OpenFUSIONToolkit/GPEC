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
    compute_perturbed_equilibrium(ffs, ft_ctrl, ctrl, intr)::PerturbedEquilibriumState

Main entry point for perturbed equilibrium calculations.

Computes plasma response to external forcing and calculates singular layer
coupling metrics. Every ForceFreeStates input — the equilibrium, the mode space, the
metric and matrix fits, the free-boundary energies and the ξ solution — is read off `ffs`.
Products the producing integrator could not supply gate the corresponding calculation:
the step warns and is skipped instead of erroring.

## Arguments

  - `ffs`: `ForceFreeStates.ForceFreeStatesResult` from the stability solve
  - `ft_ctrl`: Forcing terms control parameters from [ForcingTerms] section
  - `ctrl`: Control parameters from [PerturbedEquilibrium] section
  - `intr`: Internal state variables

## Returns

  - `PerturbedEquilibriumState`: Calculation results
"""
function compute_perturbed_equilibrium(
    ffs::ForceFreeStatesResult,
    ft_ctrl::ForcingTerms.ForcingTermsControl,
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

    # Load forcing data. On the gpec.h5 replay path the caller preloads
    # `intr.forcing_modes` from the snapshot, so skip re-reading the original file.
    if isempty(intr.forcing_modes)
        if ft_ctrl.forcing_data_format == "coil"
            cfg = ForcingTerms.CoilConfig(ft_ctrl)
            # Reuse preloaded coil geometry (gpec.h5 rerun with `--coil-source coils`)
            # when present; otherwise build it from the TOML coil-set config. Either
            # way, retain it on `intr.coil_sets` for the rerun snapshot writer.
            coil_sets = isempty(intr.coil_sets) ?
                        ForcingTerms.load_coil_sets(cfg, ffs.nlow; equil=equil) : intr.coil_sets
            intr.coil_sets = coil_sets
            for n in ffs.nlow:ffs.nhigh
                modes_n = ForcingMode[]
                ForcingTerms.compute_coil_forcing_modes!(
                    modes_n, coil_sets, equil, cfg, n, ffs.mlow, ffs.mhigh;
                    psi=ffs.psilim, verbose=ctrl.verbose
                )
                append!(intr.forcing_modes, modes_n)
            end
        else
            norm_tag = load_forcing_data!(intr.forcing_modes, intr.dir_path, ft_ctrl.forcing_data_file, ft_ctrl.forcing_data_format, ctrl.verbose)
            for n in ffs.nlow:ffs.nhigh
                # filter returns a new Vector but still holds references to same ForcingMode objects
                modes_n = filter(m -> m.n == n, intr.forcing_modes)
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

end # module PerturbedEquilibrium
