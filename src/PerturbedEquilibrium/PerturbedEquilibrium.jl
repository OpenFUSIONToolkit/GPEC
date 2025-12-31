module PerturbedEquilibrium

# Imports
using HDF5
using Printf
using LinearAlgebra
using Statistics

# Import parent modules
import ..Equilibrium
import ..ForceFreeStates
import ..ForceFreeStates: OdeState, VacuumData, ForceFreeStatesInternal
import ..Vacuum
import ..ForcingTerms
import ..ForcingTerms: ForcingMode, load_forcing_data!
import DelimitedFiles: readdlm

# Include module files
include("PerturbedEquilibriumStructs.jl")
include("ResponseMatrices.jl")
include("Response.jl")
include("SingularCoupling.jl")
include("ModeOutput.jl")
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
        equil::Equilibrium.PlasmaEquilibrium,
        dcon_results::OdeState,
        vac_data::Union{VacuumData, Nothing},
        ffs_intr::ForceFreeStatesInternal,
        ctrl::PerturbedEquilibriumControl,
        intr::PerturbedEquilibriumInternal
    )::PerturbedEquilibriumState

Main entry point for perturbed equilibrium calculations.

Computes plasma response to external forcing and calculates singular layer
coupling metrics.

## Arguments
  - `equil`: Equilibrium solution from Equilibrium module
  - `dcon_results`: Stability calculation results from ForceFreeStates module
  - `vac_data`: Vacuum response data from ForceFreeStates free boundary calculation
  - `ffs_intr`: ForceFreeStates internal state with mode information
  - `ctrl`: Control parameters from TOML configuration
  - `intr`: Internal state variables

## Returns
  - `PerturbedEquilibriumState`: Calculation results

## Workflow
1. Load forcing data from file
2. Compute plasma response (if enabled)
3. Calculate singular coupling metrics (if enabled)
4. Output eigenmode fields (if enabled)
"""
function compute_perturbed_equilibrium(
    equil::Equilibrium.PlasmaEquilibrium,
    dcon_results::OdeState,
    vac_data::Union{VacuumData, Nothing},
    ffs_intr::ForceFreeStates.ForceFreeStatesInternal,
    ctrl::PerturbedEquilibriumControl,
    intr::PerturbedEquilibriumInternal
)::PerturbedEquilibriumState

    if ctrl.verbose
        println("\nPERTURBED EQUILIBRIUM START")
        println("----------------------------------")
    end
    start_time = time()

    state = PerturbedEquilibriumState()

    # Step 1: Load forcing data
    load_forcing_data!(intr.forcing_modes, intr.dir_path, ctrl.forcing_data_file, ctrl.forcing_data_format, ctrl.verbose)

    # Step 2: Compute plasma response
    if ctrl.compute_response
        if vac_data === nothing
            @warn "Vacuum data not available. Skipping plasma response calculation. Set vac_flag=true in [ForceFreeStates] section."
        else
            compute_plasma_response!(state, equil, dcon_results, vac_data, ffs_intr, intr, ctrl)
        end
    end

    # Step 3: Compute singular coupling metrics
    if ctrl.compute_singular_coupling
        compute_singular_coupling_metrics!(state, equil, dcon_results, intr, ctrl)
    end

    # Step 4: Output eigenmode fields (integrated into HDF5 output)
    # This is handled by write_outputs_to_HDF5 in main()

    end_time = time() - start_time
    if ctrl.verbose
        println("----------------------------------")
        println("Run time: $(@sprintf("%.3e", end_time)) seconds")
        println("PERTURBED EQUILIBRIUM COMPLETE")
    end

    return state
end

end # module PerturbedEquilibrium
