# GeneralizedPerturbedEquilibrium.jl
module GeneralizedPerturbedEquilibrium

# External dependencies used by main and the rerun helpers
using TOML
using Printf
using HDF5
using FastInterpolations
import IMASdd

import CommonSolve: solve

const _BANNER = "="^60
const _SECTION = "-"^40

include("Utilities/Utilities.jl")
import .Utilities as Utilities
export Utilities
import .Utilities: save_to_hdf5
export save_to_hdf5

include("Equilibrium/Equilibrium.jl")
import .Equilibrium as Equilibrium
export Equilibrium

# Local high-n stability (Mercier, resistive interchange, ballooning Δ'); depends only on Equilibrium.
include("LocalStability/LocalStability.jl")
import .LocalStability as LocalStability
export LocalStability

include("Vacuum/Vacuum.jl")
import .Vacuum as Vacuum
export Vacuum

# InnerLayer holds the pure inner-region solvers and must load before
# ForceFreeStates, which calls them for the matched-Δ′ Galerkin solve.
include("InnerLayer/InnerLayer.jl")
import .InnerLayer as InnerLayer
export InnerLayer

include("ForcingTerms/ForcingTerms.jl")
import .ForcingTerms as ForcingTerms
export ForcingTerms

include("ForceFreeStates/ForceFreeStates.jl")
import .ForceFreeStates as ForceFreeStates
export ForceFreeStates

include("Tearing/Tearing.jl")
import .Tearing as Tearing
export Tearing
import .Tearing.CriticalResonantField as CriticalResonantField

include("PerturbedEquilibrium/PerturbedEquilibrium.jl")
import .PerturbedEquilibrium as PerturbedEquilibrium
export PerturbedEquilibrium

include("KineticForces/KineticForces.jl")
import .KineticForces as KineticForces
export KineticForces

include("ErrorFields/ErrorFields.jl")
import .ErrorFields as ErrorFields
export ErrorFields

include("Analysis/Analysis.jl")
import .Analysis as Analysis
export Analysis

include("Rerun.jl")

# Scripting-API surface: the integrator selectors, the published result, the equilibrium
# constructor and the forcing description, re-exported so a user needs one `using`.
using .ForceFreeStates: AbstractIntegrator, Forward, Riccati, Galerkin, TestKineticMatrices, KineticMatrices, ForceFreeStatesResult
using .ForceFreeStates: MatchProblem
using .InnerLayer: GGJModel, SLAYERModel
using .Tearing: TearingProblem, TearingConfig, TearingResult, CriticalResonantFieldControl
using .LocalStability: LocalStabilityProblem, Ballooning, LocalStabilityResult
using .Equilibrium: PlasmaEquilibrium, KineticProfileData, IonSpecies, attach_kinetic_profiles!
using .ForcingTerms: RMPField
using .PerturbedEquilibrium: plasma_response, perturbed_equilibrium, PlasmaResponse
using .KineticForces: kinetic_torque, KineticForcesControl, KineticForcesResult
using .ErrorFields: error_fields, ErrorFieldsResult

include("ForceBalance.jl")
include("Deck.jl")

function main(args::Vector{String}=String[]; dd::Union{IMASdd.dd,Nothing}=nothing)
    # Every input source builds a ready `(inputs, eq_config, additional_input)` and hands it to
    # `main_from_inputs`: a gpec.toml working directory, an IMAS `dd`, or a gpec.h5 snapshot.
    if !isempty(args) && endswith(lowercase(args[1]), ".h5")
        inputs, eq_config, additional_input, path, git_version, preloaded_coils = build_inputs_from_h5(args)
        return main_from_inputs(inputs, eq_config, additional_input, path, git_version; preloaded_coil_sets=preloaded_coils)
    end

    path = length(args) >= 1 ? args[1] : "./"

    # Capture git version for reproducibility
    git_version = try
        String(readchomp(`git -C $(@__DIR__) describe --tags --always`))
    catch
        "unknown"
    end

    @info "\n$_BANNER\n  GPEC - Generalized Perturbed Equilibrium Code  [$git_version]\n$_BANNER"

    inputs, eq_config, additional_input = build_inputs_from_toml(path; dd=dd)
    return main_from_inputs(inputs, eq_config, additional_input, path, git_version)
end

"""
    build_inputs_from_toml(path; dd=nothing) -> (inputs, eq_config, additional_input)

Build the pipeline inputs from a working directory containing `gpec.toml`. Returns the
parsed `inputs` dict, the `EquilibriumConfig`, and the `additional_input` consumed by
`PlasmaEquilibrium` — the analytic `*Config` a `[Solovev]`, `[LargeAspectRatio]` or
`[TJAnalytic]` section states, the `dd` data dictionary for IMAS, or `nothing` for file-based
equilibria (efit, chease) that `PlasmaEquilibrium` reads from disk.
"""
function build_inputs_from_toml(path::String; dd::Union{IMASdd.dd,Nothing}=nothing)
    inputs = TOML.parsefile(joinpath(path, "gpec.toml"))

    _stop_on_deprecated_deck_keys(inputs)
    eq_config = equilibrium_config(inputs, path)

    # An equilibrium is IMAS-fed (via the dd kwarg), analytic (stated by its section), or read from a file (nothing).
    additional_input = eq_config.eq_type == "imas" ? dd : analytic_config(inputs)

    return inputs, eq_config, additional_input
end

"""
    main_from_inputs(inputs, eq_config, additional_input, path, git_version;
                     preloaded_coil_sets=nothing)

Run the pipeline every input source (`build_inputs_from_*`) funnels into. `additional_input` is forwarded
to `PlasmaEquilibrium`: a prebuilt `DirectRunInput`/`InverseRunInput` (rerun), an analytic `*Config` or
IMAS `dd`, or `nothing` for file-based equilibria.

Each section present runs its stage through the public API, exactly as a script would, and saves
the result into the one `[Run] output_file`; a deck without `[ForceFreeStates]` stops after the
equilibrium and writes nothing. The raw forcing file contents are captured in
`Input/RawInputs/ForcingTerms/`, and `preloaded_coil_sets` lets the rerun path inject the coil
geometry stored in `Input/RawInputs/Coils/`.

Returns `(; equil, local_stability, ffs, match, resp, pe, kinetic, error_fields, tearing)`, each
`nothing` when its section is absent. `equil` is the equilibrium the stages used (the solve's, on
the two-pass grid), `ffs` the result the later stages read (the matched one when the deck has
`[Match]`), and `match` its `MatchResult`.
"""
function main_from_inputs(
    inputs::Dict{String,Any},
    eq_config::Equilibrium.EquilibriumConfig,
    additional_input,
    path::String,
    git_version::String;
    preloaded_coil_sets::Union{Nothing,Vector{ForcingTerms.CoilSet}}=nothing
)
    total_start = time()
    # Per-stage wall-clock seconds, written to Info/Runtimes at the end of the run.
    runtimes = Vector{Pair{String,Float64}}()
    deck = parse_deck(inputs, path)
    (; output_file, verbose) = deck.run
    local_stability = ffs = match = resp = pe = kinetic = error_fields = tearing = nothing

    equil = _timed_stage(runtimes, "equilibrium", "Equilibrium") do
        Equilibrium.PlasmaEquilibrium(eq_config, additional_input)
    end

    if deck.force_balance !== nothing
        ffs = _timed_stage(runtimes, "force_free_states", "Force-Free States") do
            solve(ForceBalanceProblem(equil; deck.force_balance.keywords...), deck.force_balance.algorithm; verbose)
        end
        # Every later stage, local stability included, sees the equilibrium the solve formed (re-formed on the two-pass grid).
        equil = ffs.equil
        if deck.local_stability !== nothing
            local_stability = _timed_stage(runtimes, "local_stability", "Local Stability") do
                solve(LocalStabilityProblem(equil; deck.local_stability.problem...), deck.local_stability.algorithm; verbose)
            end
        end
        if deck.match !== nothing
            ffs = _timed_stage(runtimes, "match", "Match") do
                solve(MatchProblem(ffs; deck.match.problem...), deck.match.model)
            end
            match = ffs.match
        end
        h5open(output_file, "w") do target
            save_to_hdf5(target, ffs; git_version, local_stability)
        end
        @info "Results written to $output_file"
    end

    if deck.perturbed_equilibrium !== nothing
        pe = _timed_stage(runtimes, "perturbed_equilibrium", "Perturbed Equilibrium") do
            perturbed_equilibrium(plasma_response(ffs; verbose, deck.perturbed_equilibrium.response...), RMPField(deck.forcing);
                coil_sets=preloaded_coil_sets, verbose, deck.perturbed_equilibrium.drive...)
        end
        resp = pe.response
        save_to_hdf5(output_file, pe)
    end

    # Without [PerturbedEquilibrium], [KineticForces] only sets the kinetic source of the force-free-states solve.
    if deck.kinetic_forces !== nothing && pe !== nothing
        kinetic = _timed_stage(runtimes, "kinetic_forces", "KineticForces") do
            kinetic_torque(pe, deck.kinetic_forces; verbose)
        end
        save_to_hdf5(output_file, kinetic)
    end

    if deck.error_fields !== nothing
        ef = deck.error_fields
        error_fields = _timed_stage(runtimes, "error_fields", "ErrorFields") do
            scenario = ef.scenario === nothing ? nothing : ErrorFields.ScenarioParameters(equil; ef.scenario...)
            ErrorFields.error_fields(resp, pe.coil_sets, ForcingTerms.CoilConfig(deck.forcing); ef.ctrl, ef.tolerances, ef.monte_carlo,
                scenario, ef.risk, ef.ntv, kinetic=something(deck.kinetic_forces, KineticForcesControl()), pe.regularization_width, verbose)
        end
        save_to_hdf5(output_file, error_fields)
    end

    if deck.tearing !== nothing
        tearing = _timed_stage(runtimes, "tearing", "Tearing") do
            solve(TearingProblem(ffs; deck.tearing.problem...), deck.tearing.model; config=deck.tearing.config)
        end
        save_to_hdf5(output_file, tearing)
    end

    total_dt = time() - total_start
    push!(runtimes, "total" => total_dt)
    ffs === nothing || _write_runtimes!(output_file, runtimes)
    @info "\n$_BANNER\n  GPEC completed successfully in $(@sprintf("%.3f", total_dt)) s\n$_BANNER"

    return (; equil, local_stability, ffs, match, resp, pe, kinetic, error_fields, tearing)
end

"""
    _timed_stage(f, runtimes, key, label) -> f()

Run the pipeline stage `f` under a `label` section banner, log its wall-clock time and record
it in `runtimes` as `key => seconds` for the `Info/Runtimes` record.
"""
function _timed_stage(f, runtimes::Vector{Pair{String,Float64}}, key::String, label::String)
    @info "\n  $label\n$_SECTION"
    start = time()
    out = f()
    dt = time() - start
    push!(runtimes, key => dt)
    @info "$label completed in $(@sprintf("%.3f", dt)) s"
    return out
end

"""
    _write_runtimes!(h5_path::String, runtimes)

Write the `stage => seconds` pairs of `runtimes` into `Info/Runtimes/` of an existing gpec.h5. The
timings are informational, never a regression quantity. Call it only when this run produced the file:
the `isfile` guard alone would also stamp a stale gpec.h5.
"""
function _write_runtimes!(h5_path::String, runtimes)
    isfile(h5_path) || return nothing
    h5open(h5_path, "r+") do out_h5
        haskey(out_h5, "Info/Runtimes") && HDF5.delete_object(out_h5, "Info/Runtimes")
        for (stage, dt) in runtimes
            out_h5["Info/Runtimes/$stage"] = dt
        end
        Utilities.HDF5Annotations.annotate!(out_h5, Utilities.RUNTIME_H5_ANNOTATIONS)
    end
    return nothing
end

"""
    write_imas(dd, result)

Write one `dd.mhd_linear` `toroidal_mode` entry per toroidal mode number, with the least-stable
(minimum real part) free-boundary `energy_perturbed` of that n. `result` is the named tuple `main` returns.
"""
function write_imas(dd, result)
    ffs = result.ffs
    if ffs.free_boundary === nothing
        @warn "Skipping IMAS mhd_linear write: the $(ffs.integrator) run produced no free-boundary energies. " *
              "Set vac_flag=true in [ForceFreeStates]."
        return
    end
    free_energies = ffs.free_boundary

    # Top-level metadata
    dd.mhd_linear.code.name = "GPEC"
    dd.mhd_linear.ideal_flag = 1

    # Time slice at the current global_time; wipe=false reuses an existing one or appends.
    ts = resize!(dd.mhd_linear.time_slice; wipe=false)

    # n_tor_idx[i] (0-based) is the n-block of eigenvalue i.
    resize!(ts.toroidal_mode, ffs.npert)
    for j in 0:(ffs.npert-1)
        n_indices = findall(==(j), free_energies.n_tor_idx) # indices of eigenvalues in the j-th n-block
        mode = ts.toroidal_mode[j+1]
        mode.n_tor = ffs.nlow + j
        mode.energy_perturbed = minimum(real.(free_energies.et[n_indices])) # least-stable energy for this n-toroidal mode
    end

    return dd
end

export main, write_imas
export solve, plasma_response, perturbed_equilibrium, kinetic_torque, error_fields
export KineticProfileData, IonSpecies, attach_kinetic_profiles!
export PlasmaEquilibrium, ForceBalanceProblem, Forward, Riccati, Galerkin, TestKineticMatrices, KineticMatrices, kinetic_matrices, ForceFreeStatesResult, RMPField, PlasmaResponse
export MatchProblem, TearingProblem, TearingConfig, TearingResult, GGJModel, SLAYERModel
export LocalStabilityProblem, Ballooning, LocalStabilityResult
export KineticForcesControl, KineticForcesResult, ErrorFieldsResult

include("Precompile.jl")

end # module GeneralizedPerturbedEquilibrium
