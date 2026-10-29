# GeneralizedPerturbedEquilibrium.jl
module GeneralizedPerturbedEquilibrium

# External dependencies used by main and the rerun helpers
using TOML
using Printf
using HDF5
using FastInterpolations
import LinearAlgebra: dot, norm
import IMASdd
import AdaptiveArrayPools: @with_pool

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

# Import ForceFreeStates types and functions needed for main
using .ForceFreeStates: ForceFreeStatesInternal, ForceFreeStatesControl, DebugSettings
using .ForceFreeStates: ForceFreeStatesResult, build_result
using .ForceFreeStates: sing_lim!, sing_min!, sing_find!, remove_singular_surfs!, resist_eval_all!, resist_geometry, ResistGeometry
using .ForceFreeStates: make_metric, build_matrix_splines, build_kinetic_matrix_splines
using .ForceFreeStates: find_kinetic_singular_surfaces!
using .ForceFreeStates: eulerlagrange_integration, free_run, normalize_eigenfunctions!
using .ForceFreeStates: galerkin_solve, write_galerkin!

# Scripting-API surface: the integrator selectors, the published result, the equilibrium
# constructor and the forcing description, re-exported so a user needs one `using`.
using .ForceFreeStates: AbstractIntegrator, Forward, Riccati, Galerkin
using .ForceFreeStates: MatchProblem
using .InnerLayer: GGJModel, SLAYERModel
using .Tearing: TearingProblem, TearingConfig, TearingResult
using .LocalStability: LocalStabilityProblem, Ballooning, LocalStabilityResult
using .Equilibrium: PlasmaEquilibrium, KineticProfileData, IonSpecies, attach_kinetic_profiles!
using .ForcingTerms: RMPField
using .PerturbedEquilibrium: plasma_response, PlasmaResponse

const _DEPRECATED_FFS_KEYS = ("mer_flag", "force_wv_symmetry", "ode_flag", "cyl_flag", "mat_flag", "reform_eq_with_psilim",
    "use_riccati", "use_parallel", "parallel_threads", "populate_dense_xi",
    "gal_flag", "nstep", "diagnose_ca", "diagnose", "ksing", "local_stability_flag", "gal_solver", "gal_sing_order", "force_termination")
const _DEPRECATED_EQUIL_KEYS = ("power_bp", "power_b", "power_r", "power_rc", "use_galgrid", "force_termination")
const _DEPRECATED_KF_KEYS = ("moment", "eq_out", "theta_out", "xlmda_out", "eqpsi_out", "save_records", "gpec_file",
    "fnml_flag", "ellip_flag", "diag_flag", "force_xialpha")
const _DEPRECATED_PE_KEYS = ("fixed_boundary", "output_eigenmodes", "filter_modes", "singular_point_method", "compute_response")
const _DEPRECATED_KEYS = ("Equilibrium" => _DEPRECATED_EQUIL_KEYS, "ForceFreeStates" => _DEPRECATED_FFS_KEYS,
    "KineticForces" => _DEPRECATED_KF_KEYS, "PerturbedEquilibrium" => _DEPRECATED_PE_KEYS)
# Per-section output and verbosity keys, replaced by the run-wide [Run] output_file and verbose.
const _RUN_SECTION_KEYS = ("verbose", "HDF5_filename", "output_filename", "write_outputs_to_HDF5")
const _RUN_KEY_SECTIONS = ("Equilibrium", "LocalStability", "ForceFreeStates", "PerturbedEquilibrium", "KineticForces", "ErrorFields")

# Stop the run when a parsed gpec.toml section sets keys that no longer exist, so the deck is updated.
function _stop_on_deprecated_keys(table, deprecated_keys, section::String; remedy::String="remove from gpec.toml")
    found = [k for k in deprecated_keys if haskey(table, k)]
    isempty(found) && return nothing
    listed = join(("`$k`" for k in found), ", ")
    error(length(found) == 1 ? "[$section] sets $listed, which is no longer a key; $remedy" : "[$section] sets $listed, which are no longer keys; $remedy")
end

# Stop on every retired key of a parsed deck. `remedy` replaces the per-key advice, e.g. for a stored deck that cannot be edited.
function _stop_on_deprecated_deck_keys(inputs::AbstractDict; remedy::Union{Nothing,String}=nothing)
    for (section, keys) in _DEPRECATED_KEYS
        haskey(inputs, section) && _stop_on_deprecated_keys(inputs[section], keys, section; remedy=something(remedy, "remove from gpec.toml"))
    end
    run_remedy = something(remedy, "set [Run] output_file or verbose instead; a deck run always writes its output file")
    for section in _RUN_KEY_SECTIONS
        haskey(inputs, section) && _stop_on_deprecated_keys(inputs[section], _RUN_SECTION_KEYS, section; remedy=run_remedy)
    end
    ntv = get(get(inputs, "ErrorFields", Dict{String,Any}()), "NTV", nothing)
    ntv === nothing || _stop_on_deprecated_keys(ntv, ("verbose",), "ErrorFields.NTV"; remedy=run_remedy)
    return nothing
end

"""
    run_settings(inputs, path) -> (; output_file, verbose)

The run-wide `[Run]` settings of a parsed deck: the one output file every stage saves into,
resolved against the deck directory `path` when relative, and the one verbosity of the run.
Defaults are `output_file = "gpec.h5"` and `verbose = false`. Unknown keys are an error.
"""
function run_settings(inputs::AbstractDict, path::AbstractString)
    table = get(inputs, "Run", Dict{String,Any}())
    allowed = ("output_file", "verbose")
    for k in keys(table)
        k in allowed || throw(ArgumentError("unknown key \"$k\" in [Run] (allowed: $(join(allowed, ", ")))"))
    end
    file = get(table, "output_file", "gpec.h5")
    return (; output_file=isabspath(file) ? file : normpath(joinpath(path, file)), verbose=Bool(get(table, "verbose", false)))
end

"""
    local_stability_algorithm(inputs) -> Union{LocalStability.Ballooning,Nothing}

Read the numerical settings from the current `[LocalStability]` section, or return `nothing`
when absent. Problem settings and verbosity are consumed separately by the driver.
Unknown keys are an error.
"""
function local_stability_algorithm(inputs)
    haskey(inputs, "LocalStability") || return nothing
    table = inputs["LocalStability"]
    algorithm_keys = string.(fieldnames(LocalStability.Ballooning))
    allowed = [algorithm_keys..., "theta_k", "alpha_boundary_scan"]
    for k in keys(table)
        k in allowed || throw(ArgumentError("unknown key \"$k\" in [LocalStability] (allowed: $(join(allowed, ", ")))"))
    end
    kwargs = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in table if k in algorithm_keys)
    haskey(kwargs, :psi_window) && (kwargs[:psi_window] = Tuple{Float64,Float64}(kwargs[:psi_window]))
    return LocalStability.Ballooning(; kwargs...)
end

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
`PlasmaEquilibrium` — an analytic `*Config` for
sol/lar/tj equilibria (parameters from the embedded TOML section), the `dd` data dictionary
for IMAS, or `nothing` for file-based equilibria (efit, chease) that `PlasmaEquilibrium`
reads from disk.
"""
function build_inputs_from_toml(path::String; dd::Union{IMASdd.dd,Nothing}=nothing)
    inputs = TOML.parsefile(joinpath(path, "gpec.toml"))

    haskey(inputs, "Equilibrium") || error("No [Equilibrium] section in gpec.toml")
    _stop_on_deprecated_deck_keys(inputs)
    eq_config = Equilibrium.EquilibriumConfig(inputs["Equilibrium"], path)

    # An equilibrium is analytic (sol/lar/tj, parameters from its embedded section),
    # IMAS-fed (via the dd kwarg), or read from a file (additional_input = nothing).
    additional_input = if haskey(Equilibrium.ANALYTIC_EQ, eq_config.eq_type)
        build_analytic_config(eq_config.eq_type, inputs)
    elseif eq_config.eq_type == "imas"
        dd
    else
        nothing
    end

    return inputs, eq_config, additional_input
end

"""
    main_from_inputs(inputs, eq_config, additional_input, path, git_version;
                     preloaded_coil_sets=nothing)

Shared pipeline body that every input source funnels into. The caller (one of the
`build_inputs_from_*` builders) is responsible for producing a fully merged
`inputs::Dict`, an `EquilibriumConfig`, and a ready `additional_input` — this body
does no source dispatch of its own. `additional_input` is forwarded directly to
`PlasmaEquilibrium`: a prebuilt `DirectRunInput`/`InverseRunInput` (rerun path), an
analytic `*Config` or IMAS `dd` (TOML path), or `nothing` for file-based equilibria.

File forcing always comes from the `[ForcingTerms]` file, read and normalized by the PE stage
exactly as on the scripting path; the raw file contents are also captured in
`Input/RawInputs/ForcingTerms/`. The rerun path replays them by pointing `[ForcingTerms]` at a
forcing file it writes from that snapshot.

`preloaded_coil_sets` lets the rerun path inject coil geometry read from
`Input/RawInputs/Coils/` so a coil run can be replayed (recomputing the field
against the current equilibrium) without the original `.dat`/`.h5` files. The coil
geometry actually used by the run is always written back into `Input/RawInputs/Coils/`.

Returns `(; ffs, pe, slayer, local_stability, ...)`: the `ForceFreeStates.ForceFreeStatesResult`, the
`PerturbedEquilibriumState` (`nothing` when that stage did not run), the SLAYER result
(`nothing` when that stage did not run or failed) and the `LocalStabilityResult` (`nothing`
without a `[LocalStability]` section). Every stage saves into the one `[Run] output_file`. A deck
without a `[ForceFreeStates]` section stops after the equilibrium and returns `nothing`.
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
    _stop_on_deprecated_deck_keys(inputs)
    (; output_file, verbose) = run_settings(inputs, path)

    # ----------------------------------------------------------------
    # Equilibrium
    # ----------------------------------------------------------------
    @info "\n  Equilibrium\n$_SECTION"
    equil_start = time()

    # A deck without [ForceFreeStates] stops after the equilibrium.
    ffs_table = get(inputs, "ForceFreeStates", nothing)
    intr = ForceFreeStatesInternal(; dir_path=path, verbose)
    ctrl = ffs_table === nothing ? nothing : _deck_layer_overlap(ForceFreeStates.control_from_deck(ffs_table, get(inputs, "DEBUG", Dict{String,Any}())), inputs)
    local_stability_alg = local_stability_algorithm(inputs)

    ctrl === nothing || resolve_mode_space!(intr, ctrl)

    equil = Equilibrium.PlasmaEquilibrium(eq_config, additional_input)
    # GPEC differentiates the spline of F, so single-precision FPOL rounding becomes noise in Δ′.
    if ctrl !== nothing && eq_config.eq_type in Equilibrium.EFIT_KINDS && ctrl.algorithm isa Union{Riccati,Galerkin} &&
       Equilibrium.gfile_fpol_single_precision(eq_config.eq_filename)
        @warn "The g-file's FPOL is single precision. Its rounding puts noise of order ±2–3 on Δ′ at 257 radial " *
              "nodes; regenerate the file in double precision for converged Δ′." file = eq_config.eq_filename
    end

    _deck_kinetic_composition!(equil, inputs)
    if ctrl === nothing
        push!(runtimes, "equilibrium" => time() - equil_start)
        @info "\n$_BANNER\n  GPEC completed successfully in $(@sprintf("%.3f", time() - total_start)) s\n$_BANNER"
        return
    end
    kf_ctrl = kinetic_forces_control(inputs, ctrl, equil)
    equil = maybe_reform_equilibrium(equil, intr, ctrl; kf_ctrl=kf_ctrl)

    equil_dt = time() - equil_start
    push!(runtimes, "equilibrium" => equil_dt)
    @info "Equilibrium construction completed in $(@sprintf("%.3f", equil_dt)) s"

    intr.wall_settings = Vacuum.parse_wall(get(inputs, "Wall", nothing))

    if "DEBUG" in keys(inputs)
        intr.debug_settings = DebugSettings(; (Symbol(k) => v for (k, v) in inputs["DEBUG"] if k != "gal_basis_output")...)
    else
        intr.debug_settings = DebugSettings()
    end

    forcing_snapshot = snapshot_forcing_modes(inputs, path; verbose)

    # ----------------------------------------------------------------
    # Local Stability
    # ----------------------------------------------------------------
    local_stability = nothing
    if local_stability_alg !== nothing
        local_stability = _timed_stage(runtimes, "local_stability", "Local Stability") do
            section = inputs["LocalStability"]
            problem = LocalStabilityProblem(equil;
                (Symbol(k) => section[k] for k in ("theta_k", "alpha_boundary_scan") if haskey(section, k))...)
            solve(problem, local_stability_alg; verbose)
        end
    end

    # ----------------------------------------------------------------
    # Force-Free States
    # ----------------------------------------------------------------
    @info "\n  Force-Free States\n$_SECTION"
    ffs_start = time()

    metric, mats, layer_overlap = prepare_force_free_states!(intr, ctrl, equil, kf_ctrl)
    ffs_result = run_deck_match(run_force_free_states(ctrl, equil, mats, intr, metric), ctrl; verbose)
    alg = ctrl.algorithm
    problem_kwargs = ForceFreeStates.problem_keywords(ctrl)
    problem = ForceBalanceProblem(equil; nn=ctrl.nn_low:ctrl.nn_high, wall=intr.wall_settings, dir_path=intr.dir_path,
        debug=intr.debug_settings, problem_kwargs...)
    ffs_result = ForceFreeStates._with(ffs_result; problem=problem, algorithm=alg, layer_overlap=layer_overlap)

    h5open(output_file, "w") do target
        save_to_hdf5(target, ffs_result;
            git_version=git_version,
            inputs=inputs,
            forcing_snapshot=forcing_snapshot,
            local_stability=local_stability
        )
    end
    @info "Results written to $output_file"

    # Validity describes the kinetic model, not the torque contraction, so it is written for every
    # run that reads the kinetic profiles through KineticForces — including FFS-only runs.
    if _needs_kinetic_profiles(inputs, ctrl)
        h5open(output_file, "cw") do h5file
            KineticForces.write_validity!(h5file, kf_ctrl, KineticForces.ntv_species(kf_ctrl, equil), KineticForces.ntv_profiles(kf_ctrl, equil), equil,
                Float64[sng.psifac for sng in intr.sing])
        end
    end

    ffs_dt = time() - ffs_start
    push!(runtimes, "force_free_states" => ffs_dt)
    @info "Force-Free States completed in $(@sprintf("%.3f", ffs_dt)) s"

    pe_state = nothing
    if "PerturbedEquilibrium" in keys(inputs)
        pe_state = _timed_stage(runtimes, "perturbed_equilibrium", "Perturbed Equilibrium") do
            run_perturbed_equilibrium(ffs_result, inputs, preloaded_coil_sets; verbose)
        end
        save_to_hdf5(output_file, pe_state)
    end

    if "KineticForces" in keys(inputs)
        kinetic = _timed_stage(runtimes, "kinetic_forces", "KineticForces") do
            run_kinetic_forces(inputs, ffs_result, pe_state, kf_ctrl; verbose)
        end
        if kinetic !== nothing
            h5open(output_file, "cw") do target
                for (state, label) in zip(kinetic.states, kinetic.labels)
                    save_to_hdf5(target, state; dVdpsi_spline=ffs_result.equil.profiles.dVdpsi_spline, species_label=label)
                end
                save_to_hdf5(target, kinetic.total; dVdpsi_spline=ffs_result.equil.profiles.dVdpsi_spline)
            end
        end
    end

    coil_sensitivities, monte_carlo, locking_risk, efc_couplings = nothing, nothing, nothing, nothing
    if "ErrorFields" in keys(inputs)
        error_fields = _timed_stage(runtimes, "error_fields", "ErrorFields") do
            run_error_fields(inputs, ffs_result, pe_state, preloaded_coil_sets, kf_ctrl; verbose)
        end
        coil_sensitivities, monte_carlo = error_fields.sens, error_fields.monte_carlo
        locking_risk, efc_couplings = error_fields.risk, error_fields.couplings
        h5open(output_file, "cw") do target
            save_to_hdf5(target, error_fields.sens; dom=error_fields.dom)
            error_fields.tolerances === nothing || ErrorFields.write_tolerance_snapshot!(target, error_fields.tolerances)
            monte_carlo === nothing || save_to_hdf5(target, monte_carlo)
            locking_risk === nothing || save_to_hdf5(target, locking_risk; scan=error_fields.scan)
            efc_couplings === nothing || save_to_hdf5(target, efc_couplings)
        end
    end

    tearing_start = time()
    slayer_result = run_slayer_stage(ffs_result, inputs)
    tearing_dt = time() - tearing_start
    # Save a tearing result only when that stage ran.
    if slayer_result !== nothing
        save_to_hdf5(output_file, slayer_result)
        push!(runtimes, "tearing" => tearing_dt)
        @info "SLAYER completed in $(@sprintf("%.3f", tearing_dt)) s"
    end

    # ----------------------------------------------------------------
    # Done
    # ----------------------------------------------------------------
    total_dt = time() - total_start
    push!(runtimes, "total" => total_dt)
    _write_runtimes!(output_file, runtimes)
    @info "\n$_BANNER\n  GPEC completed successfully in $(@sprintf("%.3f", total_dt)) s\n$_BANNER"

    # TODO: Do not allow perturbed equilibrium calculations if zero crossings are found

    return (; ffs=ffs_result, pe=pe_state, slayer=slayer_result, local_stability, coil_sensitivities, monte_carlo, locking_risk, efc_couplings)

end

"""
    _mode_range_label(intr) -> String

Compact label for the resolved toroidal mode range, `"1"` for a single `n` and `"1:3"` for a range.
"""
_mode_range_label(intr::ForceFreeStatesInternal) = intr.npert == 1 ? "$(intr.nlow)" : "$(intr.nlow):$(intr.nhigh)"

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
    resolve_mode_space!(intr, ctrl) -> intr

Resolve the requested toroidal mode range onto `intr`, filling an unspecified bound from the
other one and rejecting ranges that contain no supported `n >= 1` mode.
"""
function resolve_mode_space!(intr::ForceFreeStatesInternal, ctrl::ForceFreeStatesControl)
    # Determine toroidal mode numbers (n >= 1 required; 0 means "not specified")
    intr.nlow, intr.nhigh = ctrl.nn_low, ctrl.nn_high
    if intr.nlow == 0 && intr.nhigh == 0
        error("Either nn_low or nn_high must be set in [ForceFreeStates] (both are 0)")
    elseif intr.nlow == 0
        intr.nlow = intr.nhigh
    elseif intr.nhigh == 0
        intr.nhigh = intr.nlow
    end
    if intr.nlow > intr.nhigh
        error("nn_low=$(intr.nlow) cannot be greater than nn_high=$(intr.nhigh)")
    end
    if intr.nhigh < 1
        error("All requested toroidal modes (n=$(intr.nlow):$(intr.nhigh)) are below 1; " *
              "n < 1 modes are not supported")
    end
    if intr.nlow < 1
        @warn "Clamping nn_low from $(intr.nlow) to 1; n < 1 modes are not supported"
        intr.nlow = 1
    end
    intr.npert = intr.nhigh - intr.nlow + 1
    return intr
end

"""
    kinetic_forces_control(inputs, ctrl, equil) -> kf_ctrl

The `[KineticForces]` control of the run, defaults when the deck has no such section. A run that
needs kinetic profiles (a `[KineticForces]` section, or the calculated kinetic source) errors
here when the equilibrium carries none; every consumer derives its view from `equil.kinetic`.
"""
# The [SLAYER] layer closures a deck's layer-overlap cap scans with: `psilim_from_layer_overlap = true` becomes that SLAYERModel.
const _SLAYER_MODEL_KEYS = ("P_perp_model", "D_c_type", "D_R", "D_geo", "theta_sample")

function _deck_layer_overlap(ctrl::ForceFreeStatesControl, inputs::Dict{String,Any})
    (ctrl.psilim_from_layer_overlap === true && haskey(inputs, "SLAYER")) || return ctrl
    closures = (Symbol(k) => v for (k, v) in inputs["SLAYER"] if k in _SLAYER_MODEL_KEYS)
    return ForceFreeStates._with(ctrl; psilim_from_layer_overlap=SLAYERModel(; closures...))
end

# The deck's species keys, which describe the plasma: they set the equilibrium's kinetic data rather than the NTV control.
const _KF_SPECIES_KEYS = ("ion_charge", "ion_mass", "impurity_charge", "impurity_mass", "ion_species")

# Apply the deck's [KineticForces] species keys to the equilibrium's kinetic data, which every kinetic consumer reads.
function _deck_kinetic_composition!(equil::Equilibrium.PlasmaEquilibrium, inputs::Dict{String,Any})
    kf = get(inputs, "KineticForces", nothing)
    (kf === nothing || equil.kinetic === nothing) && return equil
    settings = Dict{Symbol,Any}()
    if haskey(kf, "ion_species")
        settings[:ion_species] = kf["ion_species"]
    elseif haskey(kf, "ion_charge") || haskey(kf, "ion_mass")
        settings[:ion_species] = [Equilibrium.IonSpecies(; z=get(kf, "ion_charge", 1), m=get(kf, "ion_mass", 2), fraction=1.0)]
    end
    haskey(kf, "impurity_charge") && (settings[:impurity_charge] = kf["impurity_charge"])
    haskey(kf, "impurity_mass") && (settings[:impurity_mass] = kf["impurity_mass"])
    isempty(settings) || Equilibrium.attach_kinetic_profiles!(equil, Equilibrium.KineticProfileData(equil.kinetic; settings...))
    return equil
end

function kinetic_forces_control(inputs::Dict{String,Any}, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium)
    kf_ctrl = haskey(inputs, "KineticForces") ?
              KineticForces.KineticForcesControl(; (Symbol(k) => v for (k, v) in inputs["KineticForces"] if !(k in _KF_SPECIES_KEYS))...) :
              KineticForces.KineticForcesControl()
    _needs_kinetic_profiles(inputs, ctrl) && equil.kinetic === nothing &&
        error("this run needs kinetic profiles ([KineticForces], or kinetic_source = \"calculated\"): set kinetic_file in [Equilibrium]")
    return kf_ctrl
end

# Whether a run reads the kinetic profiles through KineticForces (its stage, or the calculated EL source).
_needs_kinetic_profiles(inputs, ctrl) = haskey(inputs, "KineticForces") || (ctrl.kinetic_factor > 0 && ctrl.kinetic_source == "calculated")

"""
    maybe_reform_equilibrium(equil, intr, ctrl; kf_ctrl=KineticForcesControl()) -> equil

Two-pass auto grid: measure the pass-1 equilibrium's curvature (profiles, geometry, and the NTV
view of its attached kinetic profiles), pin knots on rational surfaces, and re-form on the refined
grid with `Equilibrium.reform` from the source the equilibrium remembers. Returns `equil` untouched
when its configuration wants a single pass.
"""
function maybe_reform_equilibrium(
    equil::Equilibrium.PlasmaEquilibrium,
    intr::ForceFreeStatesInternal,
    ctrl::ForceFreeStatesControl;
    kf_ctrl::KineticForces.KineticForcesControl=KineticForces.KineticForcesControl()
)
    eq_config = equil.config
    Equilibrium.wants_two_pass(eq_config) || return equil
    kinetic_profiles = KineticForces.ntv_profiles(kf_ctrl, equil)

    mandatory = ForceFreeStates.rational_psi_nodes(equil; nlow=intr.nlow, nhigh=intr.nhigh)
    # Kinetic-resonance nodes are located here rather than at the call site so the validity
    # boundary that filters them is cleared against this pass's rationals (intr.sing is empty yet).
    pinned = KineticForces.resonance_grid_nodes(ctrl, kf_ctrl, kinetic_profiles, KineticForces.ntv_species(kf_ctrl, equil), equil, intr;
        rationals=mandatory)
    # Smallest |n| in the run sets the widest matching half-stencil dpsi = singfac_min/(n_min·|q′|),
    # so the rational-surface brackets clear a zone large enough for every mode.
    n_min = minimum(abs(n) for n in intr.nlow:intr.nhigh if n != 0)
    isempty(pinned) ||
        @info "Pinning $(length(pinned)) kinetic-resonance surfaces into the ψ grid: $(round.(sort(pinned); digits=3))"
    psi_nodes = Equilibrium.refined_psi_grid(equil;
        tau=eq_config.psi_accuracy, kin=kinetic_profiles, mandatory=mandatory, pinned=pinned,
        singfac_min=ctrl.singfac_min, n_min=n_min)
    equil = Equilibrium.reform(equil, psi_nodes)
    implied = Equilibrium.implied_knot_count(equil; tau=eq_config.psi_accuracy, kin=kinetic_profiles)
    if implied > 1.5 * (length(psi_nodes) - 1)
        @warn "Two-pass psi grid: refined equilibrium implies $implied knots vs $(length(psi_nodes) - 1) used — " *
              "pass 1 may have under-sampled a feature; consider tightening psi_accuracy"
    end
    @info "Two-pass psi grid: $(length(psi_nodes)) knots, $(length(mandatory)) rational surfaces pinned (n=$(_mode_range_label(intr)))"

    return equil
end

"""
    snapshot_forcing_modes(inputs, path; verbose=false) -> Union{Nothing,Tuple{Vector{ForcingMode},String}}

The raw file-based forcing modes and their normalization tag, read when PerturbedEquilibrium is
enabled so they land in `Input/RawInputs/ForcingTerms/` alongside the TOML blob. They feed only
the writer: the PE stage reads and normalizes the forcing file itself.
"""
function snapshot_forcing_modes(inputs::Dict{String,Any}, path::String; verbose::Bool=false)
    # Coil forcing is recomputed from the `[[ForcingTerms.coil_set]]` TOML blob on replay.
    "PerturbedEquilibrium" in keys(inputs) || return nothing
    ft_ctrl = forcing_terms_control(inputs)
    ft_ctrl.forcing_data_format in ("ascii", "hdf5") || return nothing
    modes = ForcingTerms.ForcingMode[]
    normalization = ForcingTerms.load_forcing_data!(modes, path, ft_ctrl.forcing_data_file, ft_ctrl.forcing_data_format, verbose)
    return modes, normalization
end

"""
    prepare_force_free_states!(intr, ctrl, equil, kf_ctrl) -> (metric, mats, overlap)

Set up the force-free-states solve on `intr`: integration limits, the surviving singular
surfaces and their GGJ coefficients, the poloidal mode range, and the metric plus
Euler-Lagrange (and, when requested, kinetic) matrices.
"""
function prepare_force_free_states!(
    intr::ForceFreeStatesInternal,
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    kf_ctrl::KineticForces.KineticForcesControl
)
    # Resistive-layer overlap: where adjacent rational surfaces' layers run into each other bounds the domain, when asked.
    cap = ctrl.psilim_from_layer_overlap
    overlap = cap === false ? nothing : Tearing.layer_overlap_scan(equil, cap === true ? SLAYERModel() : cap, intr.nlow:intr.nhigh)

    # Determine psilim and qlim (where we will integrate to)
    sing_lim!(intr, ctrl, equil; psilim_cap=overlap === nothing ? nothing : overlap.psihigh)

    # Find all singular surfaces in the equilibrium
    sing_find!(intr, equil)

    # Keep only surfaces inside the integration domain [qlow, qlim], so the Δ' BVP sees only crossable ones.
    remove_singular_surfs!(intr; qmin=max(ctrl.qlow, equil.params.qmin))

    # For the outer-region Galerkin solve, exclude the q < qlow core (incl. any q≤1 sawtooth
    # surfaces) by raising psilow to where q = qlow (RDCON sing_min). Without this, the gal FEM
    # integrates through the unhandled q≤1 ideal singularity and contaminates Δ′ at the innermost
    # kept surface when q0 < qlow. No-op (keeps the axis bound) when qlow ≤ qmin.
    if ctrl.algorithm isa ForceFreeStates.Galerkin
        sing_min!(intr, ctrl, equil)
    end

    # Populate Glasser-Greene-Johnson geometric coefficients (E, F, G, H, K, M) for each
    # surviving singular surface; `ggj_parameters` layers τ_A / τ_R on top.
    if intr.msing > 0
        ForceFreeStates.resist_eval_all!(intr, equil)
    end

    # Determine poloidal mode numbers
    # TODO: delta_mhigh is doubled for consistency with Fortran - why is this present in the Fortran?
    delta_mhigh = 2 * ctrl.delta_mhigh
    if ctrl.delta_mlow < 0 || ctrl.delta_mhigh < 0
        error("Negative delta_mlow or delta_mhigh not allowed")
    end
    if ctrl.sing_start == 0
        intr.mlow = trunc(Int, min(intr.nlow * equil.params.qmin, 0)) - 4 - ctrl.delta_mlow
        intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + delta_mhigh
    else
        intr.mmin = Inf # HUGE in Fortran
        for ising in Int(ctrl.sing_start):intr.msing
            intr.mmin = min(intr.mmin, sing[ising].m)
        end
        intr.mlow = intr.mmin - ctrl.delta_mlow
        intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + delta_mhigh
    end
    intr.mpert = intr.mhigh - intr.mlow + 1
    intr.numpert_total = intr.mpert * intr.npert

    # Fit equilibrium quantities to Fourier-spline functions.
    if intr.verbose
        @info "Run parameters:\n" *
              "   q0 = $(@sprintf("%.3f", equil.params.q0)), qmin = $(@sprintf("%.3f", equil.params.qmin)), qmax = $(@sprintf("%.3f", equil.params.qmax)), q95 = $(@sprintf("%.3f", equil.params.q95))\n" *
              "   qlim = $(@sprintf("%.3f", intr.qlim)), psilim = $(@sprintf("%.3f", intr.psilim))\n" *
              "   betat = $(@sprintf("%.3f", equil.params.betat)), betan = $(@sprintf("%.3f", equil.params.betan)), betap1 = $(@sprintf("%.3f", equil.params.betap1))\n" *
              "   mlow = $(@sprintf("%4i", intr.mlow)), mhigh = $(@sprintf("%4i", intr.mhigh)), mpert = $(@sprintf("%4i", intr.mpert))\n" *
              "   nlow = $(@sprintf("%4i", intr.nlow)), nhigh = $(@sprintf("%4i", intr.nhigh)), npert = $(@sprintf("%4i", intr.npert))"
    end

    # Compute metric tensor
    metric = make_metric(equil, intr.mpert)

    if intr.verbose
        @info "Computing F, G, and K matrices"
    end

    # Compute matrices and build the MatrixSplines container
    mats = build_matrix_splines(equil, intr, metric)

    if ctrl.kinetic_factor > 0
        if intr.verbose
            @info "Computing kinetic matrices (source: $(ctrl.kinetic_source), factor: $(ctrl.kinetic_factor))"
        end
        # Inject the KineticForces callback so the "calculated" source can
        # invoke compute_calculated_kinetic_matrices without ForceFreeStates
        # importing KineticForces (which would invert the load order).
        kinetic_profiles = KineticForces.ntv_profiles(kf_ctrl, equil)
        species = KineticForces.ntv_species(kf_ctrl, equil)
        # One boundary for the whole run: the widest-orbit species, moved clear of any rational
        # surface whose window the transition band would otherwise cut through.
        axis_psi_c = KineticForces.axis_validity_boundary(kf_ctrl, species, kinetic_profiles, equil,
            Float64[sng.psifac for sng in intr.sing])
        calculated_cb =
            (c, e, i, m, f; psis::Vector{Float64}=Float64[]) ->
                KineticForces.compute_calculated_kinetic_matrices(
                    c, e, i, m, f;
                    kf_ctrl=kf_ctrl, kinetic_profiles=kinetic_profiles, species=species, psis=psis,
                    axis_psi_c=axis_psi_c)
        mats = build_kinetic_matrix_splines(ctrl, equil, mats, intr, metric;
            calculated_source=calculated_cb, axis_validity_psi_c=axis_psi_c)

        # Find kinetically-displaced singular surfaces (zeros of det(F̄)) for ODE crossings.
        # Matches Fortran ksing_find (sing.f:1486-1616). singfac_min > 0 gates crossings;
        # singfac_min == 0 preserves single-chunk behavior.
        if ctrl.singfac_min > 0
            find_kinetic_singular_surfaces!(mats, equil, intr)
        end
    end

    return metric, mats, overlap
end

"""
    run_force_free_states(ctrl, equil, mats, intr, metric; cut_solution=false) -> ForceFreeStatesResult

Run the formalism selected by `ctrl.algorithm` — the standalone Galerkin solve, or the
Euler-Lagrange sweep with its free-boundary energies and Δ′ BVP — and publish its products as a
`ForceFreeStatesResult`. `cut_solution` is the scripting-only `Galerkin(; cut_solution)` option.
"""
function run_force_free_states(
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    mats,
    intr::ForceFreeStatesInternal,
    metric;
    cut_solution::Bool=false
)
    nstring = _mode_range_label(intr)

    # The three formalisms are exclusive. Galerkin solves the same Euler-Lagrange system
    # variationally rather than by radial ODE integration, so it replaces both the integration
    # and the free-boundary energies, and supplies its own vacuum response at psilim.
    odet = nothing
    free_energies = nothing
    gal_data = nothing
    gal_dp = nothing
    if ctrl.algorithm isa ForceFreeStates.Galerkin
        ctrl.kinetic_factor == 0 ||
            error("integrator = \"galerkin\" does not support kinetic runs (kinetic_factor > 0); use integrator = \"forward\".")
        gal_start = time()
        wv = ctrl.vac_flag ? first(ForceFreeStates.compute_scaled_wv(ctrl, equil, intr)) : nothing
        gal_data, gal_dp = galerkin_solve(ctrl, equil, mats, intr; wv=wv, cut_solution=cut_solution)
        gal_dt = time() - gal_start
        @info "Galerkin solve completed in $(@sprintf("%.3f", gal_dt)) s"
    else
        # Integrate Euler-Lagrange Equation
        if intr.verbose
            @info "Integrating Euler-Lagrange equation"
        end
        odet, fm_propagators, fm_chunks, fm_S_left = eulerlagrange_integration(ctrl, equil, mats, intr)
        if odet.nzero > 0 && intr.verbose
            @warn "Fixed-boundary mode unstable for n = $nstring"
        end

        # Compute free boundary energies.
        if ctrl.vac_flag
            if intr.verbose
                wall_desc = intr.wall_settings === nothing ? "no wall" : nameof(typeof(intr.wall_settings))
                @info "Computing free boundary energies ($wall_desc)"
            end
            free_energies = free_run(odet, ctrl, equil, mats, intr)
            normalize_eigenfunctions!(odet, free_energies.wt, equil.psio)
            if real(free_energies.et[1]) < 0
                if intr.verbose
                    @warn "Free-boundary mode unstable for n = $nstring"
                end
            else
                if intr.verbose
                    @info "All free-boundary modes stable for n = $nstring"
                end
            end

            # Compute inter-surface Δ' matrix (STRIDE BVP) using vacuum edge BC.
            # Requires propagators from parallel FM path and wv from free_run.
            if ctrl.kinetic_factor == 0 && intr.msing > 0 && fm_propagators !== nothing
                if intr.verbose
                    @info "Computing Δ' matrix (STRIDE BVP with vacuum coupling)"
                end
                ForceFreeStates.compute_delta_prime_matrix!(intr, fm_propagators, fm_chunks;
                    wv=free_energies.wv, psio=equil.psio, debug=intr.verbose,
                    S_at_surface_left=fm_S_left,
                    ctrl=ctrl, equil=equil, mats=mats)
            end
        end
    end

    # Publish the solve: from here on the downstream stages read the result, never `intr`.
    result = build_result(ForceFreeStates._integrator_symbol(ctrl.algorithm), ctrl, equil, intr, metric, mats, odet, free_energies, gal_data, gal_dp)

    return result
end

"""
    run_deck_match(result, ctrl; verbose=false) -> ForceFreeStatesResult

Solve the deck's inner-layer match (`gal_match_flag`) on a finished solve, posing the same
`MatchProblem` a script would from the `gal_*` keys. Returns `result` unchanged when the deck
requests no match.
"""
function run_deck_match(result::ForceFreeStatesResult, ctrl::ForceFreeStatesControl; verbose::Bool=false)
    ctrl.gal_match_flag || return result
    ctrl.gal_inner_solver in ("ray", "galerkin") ||
        error("gal_inner_solver = \"$(ctrl.gal_inner_solver)\" (expected \"ray\" or \"galerkin\")")
    verbose && @info(
        ctrl.gal_ideal_flag ?
        "RPEC matching: IDEAL solution (inner layer skipped, bare coil columns)" :
        "RPEC matching: inner-layer Δ(Q) + outer↔inner solve for the coil-driven ξ"
    )
    # The gal_inner_* knobs override the "galerkin" backend's defaults; unset keys keep them.
    model = if ctrl.gal_inner_solver == "galerkin"
        knobs = (xfac=ctrl.gal_inner_xfac, nx=ctrl.gal_inner_nx, nq=ctrl.gal_inner_nq,
            cutoff=ctrl.gal_inner_cutoff, kmax=ctrl.gal_inner_kmax)
        InnerLayer.GGJModel(; solver=:galerkin, (k => v for (k, v) in pairs(knobs) if v !== nothing)...)
    else
        InnerLayer.GGJModel(; solver=:ray)
    end
    prob = ForceFreeStates.MatchProblem(result; eta=ctrl.gal_eta, rho=ctrl.gal_rho, rotation=ctrl.gal_rotation,
        gamma=ctrl.gal_gamma, ideal=ctrl.gal_ideal_flag)
    matched = solve(prob, model)
    ctrl.gal_ideal_flag || (verbose && @info "RPEC matching: linear-solve residual = $(matched.match.residual)")
    return matched
end

"""
    ForceBalanceProblem(equil; nn, wall=nothing,
                        dir_path=".", debug=DebugSettings(), kwargs...)

The perturbed force-balance problem posed on an equilibrium, whose solutions are the force-free
(or, with `kinetic_factor > 0`, kinetic) perturbed states. With kinetic terms the force operator
is not self-adjoint, so this is not always an energy extremization.

This is the WHAT of a stability solve; the integrator passed to
[`solve`](@ref) is the HOW. A `PlasmaEquilibrium` hosts many possible problems — this type
names this one, so `solve` stays unambiguous as other problem classes appear. Inner-layer
matching is a separate problem posed on the finished solve: see `MatchProblem`.

`nn` is the toroidal mode number or range. `wall` is the vacuum wall shape, `dir_path` the
working directory outputs are written to, and `debug` the diagnostic dump settings of the
DEBUG deck section. Remaining keywords describe physics and workflow controls;
numerical settings belong on the integrator passed to `solve`. `nn_low`/`nn_high` are rejected — they
come from `nn`.

## Fields

  - `equil::Equilibrium.PlasmaEquilibrium` - The equilibrium the problem is posed on.
  - `wall::Union{Nothing,Vacuum.WallSettings}` - Vacuum wall shape for the free-boundary energies.
  - `dir_path::String` - Working directory for outputs.
  - `debug::DebugSettings` - Diagnostic dump settings.
  - `ctrl_kwargs::Dict{Symbol,Any}` - `ForceFreeStatesControl` keywords, `nn` already folded in.
"""
struct ForceBalanceProblem
    equil::Equilibrium.PlasmaEquilibrium
    wall::Union{Nothing,Vacuum.WallSettings}
    dir_path::String
    debug::DebugSettings
    ctrl_kwargs::Dict{Symbol,Any}
end

function ForceBalanceProblem(
    equil::Equilibrium.PlasmaEquilibrium;
    nn::Union{Int,AbstractUnitRange{Int}},
    wall::Union{Nothing,Vacuum.WallSettings}=nothing,
    dir_path::AbstractString=".",
    debug::DebugSettings=DebugSettings(),
    kwargs...
)
    ctrl_kwargs = Dict{Symbol,Any}(kwargs)
    allowed = setdiff(fieldnames(ForceFreeStatesControl), (:algorithm,))
    for name in keys(ctrl_kwargs)
        name in allowed || throw(ArgumentError("`$name` is not a ForceBalanceProblem keyword; numerical settings belong on the integrator."))
    end
    (haskey(ctrl_kwargs, :nn_low) || haskey(ctrl_kwargs, :nn_high)) &&
        error("the toroidal mode range comes from the `nn` keyword; drop nn_low/nn_high")
    ctrl_kwargs[:nn_low] = first(nn)
    ctrl_kwargs[:nn_high] = last(nn)
    return ForceBalanceProblem(equil, wall, String(dir_path), debug, ctrl_kwargs)
end

"""
    solve(prob::ForceBalanceProblem, alg; verbose=false) -> ForceFreeStatesResult
    solve(equil, alg; nn, verbose=false, kwargs...) -> ForceFreeStatesResult

Solve the perturbed-plasma [`ForceBalanceProblem`](@ref) with the formalism `alg` —
[`Forward`](@ref), [`Riccati`](@ref) or [`Galerkin`](@ref) — and return the published
[`ForceFreeStatesResult`](@ref). This is the scripting entry point; it runs the same stages
a `gpec.toml` run of `main` does and produces the same result object. The second form is
sugar building the problem from an equilibrium and the problem keywords in one call.

Knobs owned by `alg` are rejected as `ForceFreeStatesControl` keywords. Kinetic runs
(`kinetic_factor > 0`) take `Forward()`; the `"calculated"` source needs kinetic profiles on the
equilibrium (`kinetic_file`, or `attach_kinetic_profiles!`) and uses the default
`KineticForcesControl` species and collision settings.

```julia
eq = PlasmaEquilibrium("input.geqdsk"; jac_type="hamada")
prob = ForceBalanceProblem(eq; nn=1, delta_mlow=8, delta_mhigh=8, vac_flag=true)
ffs = solve(prob, Riccati())
ffs = solve(eq, Riccati(); nn=1, vac_flag=true)   # equivalent one-line form
```
"""
function solve(prob::ForceBalanceProblem, alg::ForceFreeStates.AbstractIntegrator; verbose::Bool=false)
    total_start = time()

    equil = prob.equil
    ctrl_kwargs = copy(prob.ctrl_kwargs)
    ctrl = ForceFreeStatesControl(; algorithm=alg, ctrl_kwargs...)
    ForceFreeStates.validate_algorithm(ctrl)

    ctrl.kinetic_factor > 0 && ctrl.kinetic_source == "calculated" && equil.kinetic === nothing &&
        error("kinetic_source = \"calculated\" needs kinetic profiles on the equilibrium: set kinetic_file or call attach_kinetic_profiles!")
    ctrl.kinetic_factor > 0 && !(alg isa ForceFreeStates.Forward) &&
        error("$(nameof(typeof(alg)))() does not support kinetic runs (kinetic_factor > 0); use Forward().")

    intr = ForceFreeStatesInternal(; dir_path=prob.dir_path, verbose)
    intr.wall_settings = prob.wall
    intr.debug_settings = prob.debug

    resolve_mode_space!(intr, ctrl)
    ctrl.gal_match_flag &&
        error("gal_match_flag is the deck's match request; match the result instead: solve(MatchProblem(ffs; eta, rho, rotation), GGJModel())")

    # The problem has no [KineticForces] section, so the views of the attached kinetic profiles use the default control.
    kf_ctrl = KineticForces.KineticForcesControl()

    equil = maybe_reform_equilibrium(equil, intr, ctrl)

    metric, mats, layer_overlap = prepare_force_free_states!(intr, ctrl, equil, kf_ctrl)
    result = run_force_free_states(ctrl, equil, mats, intr, metric; cut_solution=alg isa ForceFreeStates.Galerkin && alg.cut_solution)
    result = ForceFreeStates._with(result; problem=prob, algorithm=alg, layer_overlap=layer_overlap)

    @info "Force-Free States completed in $(@sprintf("%.3f", time() - total_start)) s"

    return result
end

"""
    solve(equil::PlasmaEquilibrium, alg; nn, verbose=false, kwargs...) -> ForceFreeStatesResult

Convenience form of [`solve`](@ref): builds the [`ForceBalanceProblem`](@ref) from the
equilibrium and the problem keywords, then solves it with `alg`.
"""
solve(equil::Equilibrium.PlasmaEquilibrium, alg::ForceFreeStates.AbstractIntegrator; verbose::Bool=false, kwargs...) =
    solve(ForceBalanceProblem(equil; kwargs...), alg; verbose)

"""
    run_perturbed_equilibrium(result, inputs, preloaded_coil_sets; verbose=false) -> pe_state

Run the perturbed-equilibrium stage against a published force-free-states `result`. Returns `nothing` when the deck carries no `[PerturbedEquilibrium]` section.
"""
function run_perturbed_equilibrium(
    result::ForceFreeStatesResult,
    inputs::Dict{String,Any},
    preloaded_coil_sets::Union{Nothing,Vector{ForcingTerms.CoilSet}};
    verbose::Bool=false
)
    # Check for PerturbedEquilibrium section and run if present
    pe_state = nothing
    if "PerturbedEquilibrium" in keys(inputs)
        ft_ctrl = forcing_terms_control(inputs)
        section = inputs["PerturbedEquilibrium"]

        # The deck's section feeds the two scripting calls, and its forcing block is an unscaled
        # RMPField, so the TOML path and the scripting API share one stage.
        response_keys = ("compute_singular_coupling", "compute_torque_response")
        resp = plasma_response(result; verbose, (Symbol(k) => v for (k, v) in section if k in response_keys)...)
        pe_state = perturbed_equilibrium(resp, ForcingTerms.RMPField(ft_ctrl);
            coil_sets=preloaded_coil_sets, verbose, (Symbol(k) => v for (k, v) in section if !(k in response_keys))...)
    end

    return pe_state
end

"""
    perturbed_equilibrium(resp, rmp; regularization_width=resp.regularization_width, verbose=false, coil_sets=nothing)
        -> PerturbedEquilibriumState

Drive the plasma response `resp` (from [`plasma_response`](@ref)) with the external field `rmp`, an
[`RMPField`](@ref), and return the perturbed-equilibrium result. One response serves any number of
fields, so a scan rebuilds no operator.

`regularization_width` smooths the ideal 1/(m−nq) divergence in the profile reconstruction (Fortran
`reg_spot`); its default is 0, and 0.05 is typical for driving KineticForces with an ideal solve.
Call `save_to_hdf5(target, pe)` to save the result. `coil_sets` injects already-built coil geometry (the gpec.h5 replay path), bypassing the geometry read.

```julia
resp = plasma_response(ffs)
pe = perturbed_equilibrium(resp, RMPField("forcing.dat"))
```
"""
function perturbed_equilibrium(
    resp::PerturbedEquilibrium.PlasmaResponse,
    rmp::ForcingTerms.RMPField;
    regularization_width::Real=resp.regularization_width,
    verbose::Bool=false,
    coil_sets::Union{Nothing,Vector{ForcingTerms.CoilSet}}=nothing
)
    # Injected coil geometry (gpec.h5 replay with `--coil-source coils`) lets the coil field
    # be recomputed from stored geometry without the .dat/.h5 files.
    pe_state = PerturbedEquilibrium.compute_perturbed_equilibrium(resp, rmp; regularization_width, verbose,
        preloaded_coil_sets=coil_sets === nothing ? ForcingTerms.CoilSet[] : copy(coil_sets))

    return pe_state
end

"""
    run_kinetic_forces(inputs, result, pe_state, kf_ctrl; verbose=false)

Compute the neoclassical toroidal viscosity torque diagnostics when the deck carries a
`[KineticForces]` section, from the kinetic profiles attached to `result.equil`. No-op when the perturbed-equilibrium state the operators contract
against is missing. Returns `(; total, states, labels)` for the summed and per-species results,
or `nothing` when no torque is computed.
"""
function run_kinetic_forces(
    inputs::Dict{String,Any},
    result::ForceFreeStatesResult,
    pe_state,
    kf_ctrl::KineticForces.KineticForcesControl;
    verbose::Bool=false
)
    # ----------------------------------------------------------------
    # KineticForces (Neoclassical Toroidal Viscosity)
    # ----------------------------------------------------------------
    ("KineticForces" in keys(inputs)) || return nothing
    kinetic_profiles = KineticForces.ntv_profiles(kf_ctrl, result.equil)
    species = KineticForces.ntv_species(kf_ctrl, result.equil)

    # One boundary for the whole run, shared with the EL kinetic matrices and the Validity output.
    axis_psi_c = KineticForces.axis_validity_boundary(kf_ctrl, species, kinetic_profiles, result.equil,
        Float64[sng.psifac for sng in result.surfaces])

    # Standalone NTV torque diagnostics need a PE state (they contract kinetic operators
    # against ξ). The self-consistent kinetic_source="calculated" path produces none — skip.
    if pe_state === nothing
        @info "Skipping NTV torque diagnostics: no perturbed-equilibrium data (e.g. kinetic_source=\"calculated\")."
    else
        kf_intr = KineticForces.KineticForcesInternal(result.equil; verbose)
        KineticForces.set_perturbation_data!(kf_intr, pe_state, result, result.equil, result.metric)

        if species === nothing
            kf_state = KineticForces.KineticForcesState()
            KineticForces.compute_torque_all_methods!(kf_state, kf_intr, kf_ctrl, result.equil, kinetic_profiles;
                axis_psi_c=axis_psi_c)
            return (; total=kf_state, states=KineticForces.KineticForcesState[], labels=String[])
        else
            states = KineticForces.KineticForcesState[]
            labels = String[]
            for sp in species
                st = KineticForces.KineticForcesState()
                KineticForces.compute_torque_all_methods!(st, kf_intr, kf_ctrl, result.equil, sp.profiles;
                    axis_psi_c=axis_psi_c, z=sp.z, m=sp.m, electron=sp.electron)
                push!(states, st)
                push!(labels, sp.label)
            end
            kf_state = KineticForces.combine_species_states(states)
            return (; total=kf_state, states, labels)
        end
    end

    return nothing
end

"""
    forcing_terms_control(inputs) -> ForcingTermsControl

The `[ForcingTerms]` section of a parsed deck as a control struct, defaults when absent. The
`[[ForcingTerms.coil_set]]` array of tables is passed through as `coil_sets_raw` rather than
splatted with the scalar keys.
"""
function forcing_terms_control(inputs::Dict{String,Any})
    "ForcingTerms" in keys(inputs) || return ForcingTerms.ForcingTermsControl()
    forcing_raw = inputs["ForcingTerms"]
    coil_sets_raw = Vector{Dict{String,Any}}(get(forcing_raw, "coil_set", Dict{String,Any}[]))
    scalar_forcing = filter(p -> p.first != "coil_set", forcing_raw)
    return ForcingTerms.ForcingTermsControl(; (Symbol(k) => v for (k, v) in scalar_forcing)..., coil_sets_raw=coil_sets_raw)
end

"""
    run_error_fields(inputs, result, pe_state, preloaded_coil_sets, kf_ctrl; verbose=false)

Compute the requested coil sensitivities, tolerance Monte Carlo, locking risk, and NTV
correction-coil couplings. Return `(; sens, monte_carlo, risk, couplings, dom, tolerances, scan)`;
optional computations are `nothing` when omitted. The driver saves each product after this
function returns. Requires coil-format forcing and a perturbed-equilibrium response.
"""
function run_error_fields(
    inputs::Dict{String,Any},
    result::ForceFreeStatesResult,
    pe_state,
    preloaded_coil_sets::Union{Nothing,Vector{ForcingTerms.CoilSet}},
    kf_ctrl::KineticForces.KineticForcesControl=KineticForces.KineticForcesControl();
    verbose::Bool=false
)
    ("ErrorFields" in keys(inputs)) || return nothing

    # [ErrorFields.MonteCarlo], [ErrorFields.Risk] and [ErrorFields.scenario] are nested tables,
    # excluded from the control-struct splat.
    ef_raw = inputs["ErrorFields"]
    nested = ("MonteCarlo", "Risk", "scenario", "NTV")
    ef_ctrl = ErrorFields.ErrorFieldsControl(; (Symbol(k) => v for (k, v) in ef_raw if !(k in nested))...)
    mc_ctrl = ErrorFields.MonteCarloControl(; (Symbol(k) => v for (k, v) in get(ef_raw, "MonteCarlo", Dict{String,Any}()))...)
    risk_ctrl = ErrorFields.RiskControl(; (Symbol(k) => v for (k, v) in get(ef_raw, "Risk", Dict{String,Any}()))...)
    scenario_raw = get(ef_raw, "scenario", nothing)
    ntv_ctrl = ErrorFields.NTVControl(; (Symbol(k) => v for (k, v) in get(ef_raw, "NTV", Dict{String,Any}()))...)
    # Correction arrays and other excluded sets are swept like any coil but are not error-field sources.
    excluded = ErrorFields.excluded_coil_names(ef_raw)
    pe_state === nothing && error("[ErrorFields] needs a [PerturbedEquilibrium] section with compute_singular_coupling = true")
    ft_ctrl = forcing_terms_control(inputs)
    ft_ctrl.forcing_data_format == "coil" ||
        error("[ErrorFields] linearizes coil geometry, so [ForcingTerms] must use forcing_data_format = \"coil\" (got \"$(ft_ctrl.forcing_data_format)\")")

    cfg = ForcingTerms.CoilConfig(ft_ctrl)
    coil_sets = preloaded_coil_sets === nothing ? ForcingTerms.load_coil_sets(cfg, result.nlow; equil=result.equil) : preloaded_coil_sets
    rc = PerturbedEquilibrium.ResonantCoupling(pe_state.response)
    sens = ErrorFields.compute_coil_sensitivities(coil_sets, rc, result.equil, cfg, ef_ctrl;
        psi=result.psilim, b_t0=result.equil.params.bt0, verbose)

    tolerances = nothing
    if !isempty(ef_ctrl.tolerance_file)
        tol_path = joinpath(result.dir_path, ef_ctrl.tolerance_file)
        isfile(tol_path) || error("[ErrorFields] tolerance_file not found: $tol_path")
        tolerances = ErrorFields.validate_tolerances(ErrorFields.read_tolerance_toml(tol_path), sens.coil_names)
        ErrorFields.check_excluded_tolerances(tolerances, excluded)
        @info "Tolerances: $(length(tolerances.coils)) coil sets, $(length(tolerances.groups)) coherent groups from $(ef_ctrl.tolerance_file)"
    end

    # The run's Monte Carlo is the full-window, dominant-mode summary; other windows are re-run
    # post hoc with ErrorFields.run_monte_carlo.
    dom = PerturbedEquilibrium.dominant_coupling(rc)
    table = ErrorFields.without_coils(ErrorFields.sensitivity_table(sens, dom), excluded)
    isempty(excluded) || @info "Error field: $(join(table.coil_names, ", ")); excluded $(join(excluded, ", "))"
    monte_carlo = nothing
    if tolerances !== nothing
        mc_start = time()
        monte_carlo = ErrorFields.run_monte_carlo(table, tolerances, coil_sets, mc_ctrl)
        @info "Monte Carlo: $(mc_ctrl.nbatch) × $(mc_ctrl.nsample) samples in $(@sprintf("%.2f", time() - mc_start)) s; " *
              "⟨|δ|⟩ = $(@sprintf("%.3e", monte_carlo.abs_delta_sampled_mean)) intrinsic, $(@sprintf("%.3e", monte_carlo.abs_delta_efc_sampled_mean)) corrected " *
              "(nominal $(@sprintf("%.3e", monte_carlo.abs_delta_total_as_designed)))"
    end

    # Locking risk needs the operating point: [ErrorFields.scenario] with at least n_e.
    risk = nothing
    scan = nothing
    if monte_carlo !== nothing && scenario_raw !== nothing
        haskey(scenario_raw, "n_e") || error("[ErrorFields.scenario] must give n_e (electron density, 1e19 m^-3)")
        scen = ErrorFields.ScenarioParameters(result.equil; (Symbol(k) => v for (k, v) in scenario_raw)...)
        n_scaling = ErrorFields.single_toroidal_mode(result.nlow, result.nhigh; where="the in-run locking risk of this run")
        sc = ErrorFields.threshold_scaling(; n=n_scaling, year=risk_ctrl.year, dataset=risk_ctrl.dataset, fit=risk_ctrl.fit)
        risk_start = time()
        risk = ErrorFields.locking_risk(monte_carlo, sc, scen; ctrl=risk_ctrl)
        @info "Locking risk ($(ErrorFields.scaling_label(sc))): threshold $(@sprintf("%.3e", risk.threshold_fit)); " *
              "P_lock = $(@sprintf("%.2f", risk.locking_probability_percent)) % intrinsic, $(@sprintf("%.2f", risk.locking_probability_efc_percent)) % corrected, " *
              "$(@sprintf("%.2f", risk.locking_probability_as_designed_percent)) % as designed ($(@sprintf("%.2f", time() - risk_start)) s)"
        if !isempty(risk_ctrl.scan_scales)
            scan_start = time()
            scan = ErrorFields.tolerance_scan(table, tolerances, coil_sets, mc_ctrl, sc, scen;
                scales=risk_ctrl.scan_scales, risk_ctrl)
            @info "Tolerance scan over $(length(scan.tolerance_scale)) scales in $(@sprintf("%.2f", time() - scan_start)) s"
        end
    end

    # Correction-coil couplings: overlap per kAt and the NTV torque of the full and residual fields.
    couplings = nothing
    if !isempty(ntv_ctrl.efc_coils)
        efc_sets = [cs for cs in coil_sets if cs.name in ntv_ctrl.efc_coils]
        missing = setdiff(ntv_ctrl.efc_coils, [cs.name for cs in efc_sets])
        isempty(missing) || error("[ErrorFields.NTV] efc_coils not among the run's coil sets: $(join(missing, ", "))")
        ntv_start = time()
        couplings = efc_couplings(pe_state.response, efc_sets, rc, dom, cfg, kf_ctrl; method=ntv_ctrl.method, verbose,
            regularization_width=pe_state.regularization_width,
            rotation_scan=ntv_ctrl.rotation_scan, scan_points=ntv_ctrl.rotation_scan_points, scan_max_points=ntv_ctrl.rotation_scan_max_points,
            scan_tolerance=ntv_ctrl.rotation_scan_tolerance, span_factor=ntv_ctrl.rotation_span_factor, offset_factor=ntv_ctrl.rotation_offset_factor)
        @info "NTV couplings of $(length(couplings)) correction arrays in $(@sprintf("%.1f", time() - ntv_start)) s"
    end

    return (; sens, monte_carlo, risk, couplings, dom, tolerances, scan)

end

"""
    efc_couplings(resp, coil_sets, rc, dom, cfg, kf_ctrl; mode=1, method="fgar",
                  regularization_width=resp.regularization_width, verbose=false) -> Vector{ErrorFields.EFCCoupling}

The couplings of each correction coil array in `coil_sets` per kilo-ampere-turn: its
dominant-mode overlap and resonant fraction from its spectrum on the control surface, and its
NTV torque for the whole field and for the field with mode `mode` of `dom` projected out. The
torque is a quadratic form of the applied spectrum, so each is one evaluation of the run's plasma
response `resp` on the unit-current spectrum (a spectrum-literal `RMPField`, nothing written, with the
run's `regularization_width`) followed by the kinetic torque of `method`; both scale exactly with the
square of the current. `cfg` is the run's `[ForcingTerms]` coil configuration, so the boundary grid
matches the one the run's sensitivities were evaluated on; `rc` must be the coupling of `resp`; the
torque uses the NTV view of the kinetic profiles attached to `resp.ffs.equil`. The spectrum is normalized by the magnitude of the
array's ampere-turns, `|nw| × max|I|`, keeping its winding sense and current pattern as the
phase reference; the torque is stored with its sign.
"""
function efc_couplings(
    resp::PerturbedEquilibrium.PlasmaResponse,
    coil_sets::Vector{ForcingTerms.CoilSet},
    rc::PerturbedEquilibrium.ResonantCoupling,
    dom::PerturbedEquilibrium.DominantCoupling,
    cfg::ForcingTerms.CoilConfig,
    kf_ctrl::KineticForces.KineticForcesControl;
    mode::Int=1,
    method::AbstractString="fgar",
    rotation_scan::Bool=true,
    scan_points::Int=9,
    scan_max_points::Int=21,
    scan_tolerance::Real=0.05,
    span_factor::Real=1.0,
    offset_factor::Real=2.0,
    regularization_width::Real=resp.regularization_width,
    verbose::Bool=false
)
    ffs = resp.ffs
    ffs.equil.kinetic === nothing && error("efc_couplings needs kinetic profiles: set kinetic_file in [Equilibrium]")
    getfield(kf_ctrl, Symbol(method * "_flag")) || error("efc_couplings: KineticForces method \"$method\" is not enabled in the control")
    equil = ffs.equil
    grids = [(n, ForcingTerms.CoilForcingGrid(equil, cfg, n; psi=ffs.psilim)) for n in sort(unique(rc.n_modes))]
    m_low, m_high = extrema(rc.m_modes)
    v = dom.right_singular_vectors[:, mode]

    # Reference rotation and scan span from the kinetic profiles, with the torque kernel's own
    # diamagnetic frequencies (Logan & Park 2013 Eq. 7): ω_φ = ω_E + ω_*n + ω_*T.
    kp = KineticForces.ntv_profiles(kf_ctrl, equil)
    chi1 = 2π * equil.psio
    chrg = KineticForces.main_ion(equil)[1] * Utilities.E_CHG
    ψk = kp.xs
    ω_E = [kp.omegaE_spline(ψ) for ψ in ψk]
    ω_star_T = [-2π * kp.Ti_deriv(ψ) / (chrg * chi1) for ψ in ψk]
    ω_star_n = [
        (n=kp.ni_spline(ψ); n > 0 ? -2π * kp.Ti_spline(ψ) * kp.ni_deriv(ψ) / (chrg * chi1 * n) : 0.0) for ψ in ψk
    ]
    ω_φ = ω_E .+ ω_star_n .+ ω_star_T
    # Momentum weight n_i·dV/dψ with ⟨R²⟩ taken as R₀², inside the plasma only.
    w = [ψ <= ffs.psilim ? max(kp.ni_spline(ψ), 0.0) * equil.profiles.dVdpsi_spline(clamp(ψ, equil.profiles.xs[1], equil.profiles.xs[end])) : 0.0 for ψ in ψk]
    ω_ref = sum(w .* ω_φ) / sum(w)
    span, ω_offset = ErrorFields.rotation_scan_span(ω_E[1], ω_star_T[1]; span_factor, offset_factor)
    npts = isodd(scan_points) ? scan_points : scan_points + 1
    grid0 = rotation_scan ? collect(range(-span, span; length=max(npts, 3))) : [0.0]
    verbose && @info "NTV couplings: ω_ref = $(@sprintf("%.3e", ω_ref)) rad/s, rough offset $(@sprintf("%.3e", ω_offset)) rad/s, " *
          "scan ±$(@sprintf("%.3e", span)) rad/s on $(length(grid0)) initial points (max $(scan_max_points))"

    function response_of(modes::Vector{ForcingTerms.ForcingMode})
        pe_state = PerturbedEquilibrium.compute_perturbed_equilibrium(resp, ForcingTerms.RMPField(modes); regularization_width, verbose)
        kf_intr = KineticForces.KineticForcesInternal(equil; verbose=verbose)
        KineticForces.set_perturbation_data!(kf_intr, pe_state, ffs, equil, ffs.metric)
        return kf_intr
    end
    # Torque and cumulative torque profile (resampled onto the kinetic ψ grid) of one perturbation at one rotation.
    function torque_of(kf_intr, profiles)
        kf_state = KineticForces.KineticForcesState()
        logger = verbose ? Base.CoreLogging.current_logger() : Base.CoreLogging.SimpleLogger(stderr, Base.CoreLogging.Warn)
        Base.CoreLogging.with_logger(logger) do
            KineticForces.compute_torque_all_methods!(kf_state, kf_intr, kf_ctrl, equil, profiles)
        end
        r = kf_state.method_results[String(method)]
        T = real(r.total_torque)
        prof = zeros(Float64, length(ψk))
        if length(r.psi_grid) >= 2
            # Cubic spline of the cumulative torque on the quadrature's own ψ points (deduplicated), zero
            # inside the first point and the total beyond the last.
            keep = [i == 1 || r.psi_grid[i] > r.psi_grid[i-1] for i in eachindex(r.psi_grid)]
            pg, tc = r.psi_grid[keep], real.(r.t_cumulative[keep])
            spl = length(pg) >= 4 ? cubic_interp(pg, tc) : nothing
            for (k, ψ) in enumerate(ψk)
                prof[k] = ψ <= pg[1] ? 0.0 : ψ >= pg[end] ? tc[end] : spl === nothing ? tc[searchsortedlast(pg, ψ)] : spl(ψ)
            end
        end
        return T, prof
    end

    out = ErrorFields.EFCCoupling[]
    for cs in coil_sets
        kat = abs(cs.nw) * maximum(abs, cs.currents) / 1e3
        kat > 0 || error("efc_couplings: coil set \"$(cs.name)\" carries no current")
        # Unit-current spectrum: per kilo-ampere-turn of the array's current pattern.
        modes = ForcingTerms.ForcingMode[]
        for (n, grid) in grids
            append!(modes, ForcingTerms.coil_forcing_modes(cs, grid, n, m_low, m_high))
        end
        modes = [ForcingTerms.ForcingMode(; n=m.n, m=m.m, amplitude=m.amplitude / kat) for m in modes]
        b̃ = PerturbedEquilibrium.rootarea_field(rc, modes)
        δ = dot(v, b̃) / equil.params.bt0
        overlap = 100 * abs(dot(v, b̃)) / norm(b̃)
        # The residual field as forcing modes: back through the conform operator to Φ_x.
        Φ_res = rc.flux_conform * ErrorFields.residual_spectrum(dom, b̃; mode)
        residual_modes = [ForcingTerms.ForcingMode(; n=rc.n_modes[k], m=rc.m_modes[k], amplitude=Φ_res[k]) for k in eachindex(Φ_res)]
        t_start = time()
        kf_full = response_of(modes)
        kf_res = response_of(residual_modes)
        profiles_full = Dict{Float64,Vector{Float64}}()
        profiles_res = Dict{Float64,Vector{Float64}}()
        function torques(Δ)
            profiles = Δ == 0 ? kp : Equilibrium.shift_exb_rotation(kp, Δ)
            T_f, p_f = torque_of(kf_full, profiles)
            T_r, p_r = torque_of(kf_res, profiles)
            profiles_full[Δ] = p_f
            profiles_res[Δ] = p_r
            verbose && @info "    $(cs.name) Δω = $(@sprintf("%+.3e", Δ)) rad/s: torque $(@sprintf("%.3e", T_f)) N·m per kAt² (residual $(@sprintf("%.3e", T_r)))"
            return [T_f, T_r]
        end
        if rotation_scan
            Δs, Ts = Utilities.adaptive_sample(torques, grid0; max_points=scan_max_points, rtol=scan_tolerance)
        else
            Δs = [0.0]
            Ts = permutedims(torques(0.0))
        end
        i0 = findfirst(==(0.0), Δs)
        T_full, T_res = Ts[i0, 1], Ts[i0, 2]
        c = ErrorFields.EFCCoupling(cs.name, abs(δ), overlap, T_full, T_res, Δs, Ts[:, 1], Ts[:, 2],
            rotation_scan ? ω_ref : NaN, rotation_scan ? ω_offset : NaN, rotation_scan ? collect(ψk) : Float64[],
            rotation_scan ? reduce(hcat, profiles_full[Δ] for Δ in Δs) : zeros(0, 1),
            rotation_scan ? reduce(hcat, profiles_res[Δ] for Δ in Δs) : zeros(0, 1))
        if rotation_scan
            crossings = ErrorFields.torque_zero_crossings(c)
            isempty(crossings) && @warn "NTV couplings: the residual torque of $(cs.name) does not change sign within ±$(@sprintf("%.3e", span)) rad/s; " *
                  "raise rotation_span_factor to reach the neoclassical offset"
        end
        verbose && @info "  $(cs.name): |δ| = $(@sprintf("%.3e", abs(δ))) per kAt, resonant fraction $(@sprintf("%.1f", overlap)) %, " *
              "torque $(@sprintf("%.3e", T_full)) N·m per kAt² (residual $(@sprintf("%.3e", T_res))) at the nominal rotation, " *
              "$(length(Δs)) scan points in $(@sprintf("%.1f", time() - t_start)) s"
        push!(out, c)
    end
    return out
end

"""
    run_slayer_stage(result, inputs) -> slayer_result

Run the SLAYER tearing-mode analysis off the force-free-states `result` and return its result.
"""
function run_slayer_stage(result::ForceFreeStatesResult, inputs::Dict{String,Any})
    ("SLAYER" in keys(inputs)) || return nothing
    _stop_on_deprecated_keys(inputs["SLAYER"], ("gap_kHz_threshold",), "SLAYER")
    slayer_ctrl = Tearing.slayer_control_from_toml(inputs["SLAYER"])
    slayer_ctrl.enabled || return nothing
    @info "\n  SLAYER\n$_SECTION"
    model = Tearing._build_inner_model(slayer_ctrl.inner_model)
    slayer_result = solve(Tearing.TearingProblem(result, slayer_ctrl), model)
    return slayer_result
end

"""
    _write_runtimes!(h5_path::String, runtimes)

Write the per-stage wall-clock seconds collected during a run into `Info/Runtimes/` of an
existing gpec.h5 file. `runtimes` iterates `stage => seconds` pairs; only the stages that ran
are recorded. Metadata comes from `RUNTIME_H5_ANNOTATIONS`, which skips the absent stages.
These timings are informational only — machine- and load-dependent, never a regression quantity.

Call it only when this run produced the file; the `isfile` guard alone would also stamp a
stale gpec.h5 left over from an earlier run.
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

Write GPEC stability results into `dd.mhd_linear`. Creates one `toroidal_mode` entry per
requested toroidal mode number, storing the least-stable (minimum real part) `energy_perturbed`
for that `n_tor`. For multi-n runs the eigenvalue array `et` is sorted by stability across all
n-blocks; `n_tor_idx[i]` identifies which n-block eigenvalue `i` belongs to, so each n_tor
receives the correct least-stable δW regardless of how modes are interleaved in `et`.

The `result` argument is the named tuple returned by `main`; its `ffs` field carries the
force-free-states result the energies are read from.
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

    # Add a time_slice at the current global_time (wipe=false reuses an existing slice
    # at the same time, or appends a new one if none exists yet)
    ts = resize!(dd.mhd_linear.time_slice; wipe=false)

    # Write the least-stable energy for each toroidal mode number
    # n_tor_idx[i] (0-based) identifies which n-block eigenvalue i belongs to.
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
export solve, plasma_response, perturbed_equilibrium
export KineticProfileData, IonSpecies, attach_kinetic_profiles!
export PlasmaEquilibrium, ForceBalanceProblem, Forward, Riccati, Galerkin, ForceFreeStatesResult, RMPField, PlasmaResponse
export MatchProblem, TearingProblem, TearingConfig, TearingResult, GGJModel, SLAYERModel
export LocalStabilityProblem, Ballooning, LocalStabilityResult

include("Precompile.jl")

end # module GeneralizedPerturbedEquilibrium
