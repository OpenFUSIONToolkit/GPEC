# GeneralizedPerturbedEquilibrium.jl
module GeneralizedPerturbedEquilibrium

# External dependencies used by main and the rerun helpers
using TOML
using Printf
using HDF5
using FastInterpolations
import IMASdd
import AdaptiveArrayPools: @with_pool

import CommonSolve: solve

const _BANNER = "="^60
const _SECTION = "-"^40

include("Utilities/Utilities.jl")
import .Utilities as Utilities
export Utilities

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

include("ForceFreeStates/ForceFreeStates.jl")
import .ForceFreeStates as ForceFreeStates
export ForceFreeStates

include("Tearing/Tearing.jl")
import .Tearing as Tearing
export Tearing
# Backward-compat top-level aliases so callers can still reach these
# directly; the canonical nested path is `Tearing.{Dispersion,Runner}`.
import .Tearing.Dispersion as Dispersion
import .Tearing.Runner as Runner
export Dispersion, Runner

include("ForcingTerms/ForcingTerms.jl")
import .ForcingTerms as ForcingTerms
export ForcingTerms

include("PerturbedEquilibrium/PerturbedEquilibrium.jl")
import .PerturbedEquilibrium as PerturbedEquilibrium
export PerturbedEquilibrium

include("KineticForces/KineticForces.jl")
import .KineticForces as KineticForces
export KineticForces

include("Analysis/Analysis.jl")
import .Analysis as Analysis
export Analysis

include("HDF5Schema.jl")
include("Rerun.jl")

# Import ForceFreeStates types and functions needed for main
using .ForceFreeStates: ForceFreeStatesInternal, ForceFreeStatesControl, DebugSettings
using .ForceFreeStates: ForceFreeStatesResult, build_result
using .ForceFreeStates: sing_lim!, sing_min!, sing_find!, resist_eval_all!, resist_geometry, ResistGeometry
using .ForceFreeStates: make_metric, build_matrix_splines, build_kinetic_matrix_splines
using .ForceFreeStates: find_kinetic_singular_surfaces!
using .ForceFreeStates: eulerlagrange_integration, free_run, normalize_eigenfunctions!
using .ForceFreeStates: galerkin_solve, write_galerkin!

# Scripting-API surface: the integrator selectors, the published result, the equilibrium
# constructor and the forcing description, re-exported so a user needs one `using`.
using .ForceFreeStates: AbstractIntegrator, Forward, Riccati, Galerkin, ResistiveMatch
using .Equilibrium: PlasmaEquilibrium
using .ForcingTerms: RMPField

const _DEPRECATED_FFS_KEYS = ("mer_flag", "force_wv_symmetry", "ode_flag", "cyl_flag", "mat_flag", "reform_eq_with_psilim",
                              "use_riccati", "use_parallel", "parallel_threads", "populate_dense_xi",
                              "gal_flag")
const _DEPRECATED_EQUIL_KEYS = ("power_bp", "power_b", "power_r", "power_rc")

# Drop deprecated keys from a parsed gpec.toml section so legacy files keep parsing
# instead of throwing an unknown-keyword error; warn so the removal is not silent.
function _drop_deprecated_keys!(table, deprecated_keys, section::String)
    for k in deprecated_keys
        if haskey(table, k)
            @warn "`$k` in [$section] is deprecated and ignored; please remove it from gpec.toml."
            delete!(table, k)
        end
    end
    return table
end

function main(args::Vector{String}=String[]; dd::Union{IMASdd.dd,Nothing}=nothing)
    # Every input source builds a ready `(inputs, eq_config, additional_input)` and hands it to
    # `main_from_inputs`: a gpec.toml working directory, an IMAS `dd`, or a gpec.h5 snapshot.
    if !isempty(args) && endswith(lowercase(args[1]), ".h5")
        inputs, eq_config, additional_input, path, git_version, preloaded_forcing, preloaded_coils = build_inputs_from_h5(args)
        return main_from_inputs(inputs, eq_config, additional_input, path, git_version;
            preloaded_forcing_modes=preloaded_forcing, preloaded_coil_sets=preloaded_coils)
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
`setup_equilibrium` — an analytic `*Config` for
sol/lar/tj equilibria (parameters from the embedded TOML section), the `dd` data dictionary
for IMAS, or `nothing` for file-based equilibria (efit, chease) that `setup_equilibrium`
reads from disk.
"""
function build_inputs_from_toml(path::String; dd::Union{IMASdd.dd,Nothing}=nothing)
    inputs = TOML.parsefile(joinpath(path, "gpec.toml"))

    haskey(inputs, "Equilibrium") || error("No [Equilibrium] section in gpec.toml")
    _drop_deprecated_keys!(inputs["Equilibrium"], _DEPRECATED_EQUIL_KEYS, "Equilibrium")
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
                     preloaded_forcing_modes=nothing)

Shared pipeline body that every input source funnels into. The caller (one of the
`build_inputs_from_*` builders) is responsible for producing a fully merged
`inputs::Dict`, an `EquilibriumConfig`, and a ready `additional_input` — this body
does no source dispatch of its own. `additional_input` is forwarded directly to
`setup_equilibrium`: a prebuilt `DirectRunInput`/`InverseRunInput` (rerun path), an
analytic `*Config` or IMAS `dd` (TOML path), or `nothing` for file-based equilibria.

`preloaded_forcing_modes` lets the rerun path inject a `Vector{ForcingMode}`
already read from the source HDF5 snapshot, so `compute_perturbed_equilibrium`
does not have to touch the original `forcing.dat` path. When `nothing`, the
ForcingTerms data is loaded from disk at snapshot time (if PerturbedEquilibrium
is enabled) so it still ends up in `Input/RawInputs/ForcingTerms/`.

`preloaded_coil_sets` similarly lets the rerun path inject coil geometry read from
`Input/RawInputs/Coils/` so a coil run can be replayed (recomputing the field
against the current equilibrium) without the original `.dat`/`.h5` files. The coil
geometry actually used by the run is always written back into `Input/RawInputs/Coils/`.

Returns `(; ffs, pe, slayer)`: the `ForceFreeStates.ForceFreeStatesResult`, the
`PerturbedEquilibriumState` (`nothing` when that stage did not run) and the SLAYER result
(`nothing` when that stage did not run or failed). An equilibrium-only run
(`force_termination` in `[Equilibrium]`) returns `nothing` — it never reaches the solve.
"""
function main_from_inputs(
    inputs::Dict{String,Any},
    eq_config::Equilibrium.EquilibriumConfig,
    additional_input,
    path::String,
    git_version::String;
    preloaded_forcing_modes::Union{Nothing,Vector{ForcingTerms.ForcingMode}}=nothing,
    preloaded_coil_sets::Union{Nothing,Vector{ForcingTerms.CoilSet}}=nothing
)
    total_start = time()

    # ----------------------------------------------------------------
    # Equilibrium
    # ----------------------------------------------------------------
    @info "\n  Equilibrium\n$_SECTION"
    equil_start = time()

    # Build data structures from inputs
    intr = ForceFreeStatesInternal(; dir_path=path)
    ffs_table = inputs["ForceFreeStates"]
    _drop_deprecated_keys!(ffs_table, _DEPRECATED_FFS_KEYS, "ForceFreeStates")
    ctrl = ForceFreeStatesControl(; (Symbol(k) => v for (k, v) in ffs_table)...)

    resolve_mode_space!(intr, ctrl)

    equil = Equilibrium.setup_equilibrium(eq_config, additional_input)

    kf_ctrl, kinetic_profiles = load_kinetic_context(inputs, intr, ctrl, equil)
    equil = maybe_reform_equilibrium(equil, eq_config, additional_input, intr, ctrl, kinetic_profiles)

    @info "Equilibrium construction completed in $(@sprintf("%.3f", time() - equil_start)) s"

    # Early exit if user only requested equilibrium setup
    if equil.config.force_termination
        @info "\n$_BANNER\n  GPEC completed successfully in $(@sprintf("%.3f", time() - total_start)) s\n$_BANNER"
        return
    end

    if "Wall" in keys(inputs)
        intr.wall_settings = Vacuum.WallShapeSettings(; (Symbol(k) => v for (k, v) in inputs["Wall"])...)
    else
        intr.wall_settings = Vacuum.WallShapeSettings()
    end

    if "DEBUG" in keys(inputs)
        intr.debug_settings = DebugSettings(; (Symbol(k) => v for (k, v) in inputs["DEBUG"])...)
    else
        intr.debug_settings = DebugSettings()
    end

    forcing_modes_snapshot = snapshot_forcing_modes(inputs, path, ctrl, preloaded_forcing_modes)

    # ----------------------------------------------------------------
    # Force-Free States
    # ----------------------------------------------------------------
    @info "\n  Force-Free States\n$_SECTION"
    ffs_start = time()

    locstab, ballooning_boundary = run_local_stability(ctrl, equil)
    metric, mats = prepare_force_free_states!(intr, ctrl, equil, kf_ctrl, kinetic_profiles)
    ffs_result = run_force_free_states(ctrl, equil, mats, intr, metric)

    if ctrl.write_outputs_to_HDF5
        write_outputs_to_HDF5(
            ffs_result;
            git_version=git_version,
            inputs=inputs,
            forcing_modes=forcing_modes_snapshot,
            locstab=locstab,
            ballooning_boundary=ballooning_boundary
        )
        @info "Results written to $(ctrl.HDF5_filename)"
    end

    @info "Force-Free States completed in $(@sprintf("%.3f", time() - ffs_start)) s"

    # Early exit if user only requested force-free states (SLAYER still runs).
    if ctrl.force_termination
        slayer_result = run_slayer_stage(ffs_result, inputs, nothing)
        @info "\n$_BANNER\n  GPEC completed successfully in $(@sprintf("%.3f", time() - total_start)) s\n$_BANNER"
        return (; ffs=ffs_result, pe=nothing, slayer=slayer_result)
    end

    pe_state = run_perturbed_equilibrium(ffs_result, inputs, forcing_modes_snapshot, preloaded_coil_sets)

    run_kinetic_forces(inputs, ffs_result, pe_state, kf_ctrl, kinetic_profiles)

    # SLAYER runs after PE so it appends to the PE output file; it falls back to the
    # ForceFreeStates file when PE did not run.
    pe_file = if "PerturbedEquilibrium" in keys(inputs)
        pe_out = get(inputs["PerturbedEquilibrium"], "output_filename", "")
        isempty(pe_out) ? ctrl.HDF5_filename : pe_out
    else
        ctrl.HDF5_filename
    end
    slayer_result = run_slayer_stage(ffs_result, inputs, pe_file)

    # ----------------------------------------------------------------
    # Done
    # ----------------------------------------------------------------
    @info "\n$_BANNER\n  GPEC completed successfully in $(@sprintf("%.3f", time() - total_start)) s\n$_BANNER"

    # TODO: Do not allow perturbed equilibrium calculations if zero crossings are found

    return (; ffs=ffs_result, pe=pe_state, slayer=slayer_result)

end

"""
    _mode_range_label(intr) -> String

Compact label for the resolved toroidal mode range, `"1"` for a single `n` and `"1:3"` for a range.
"""
_mode_range_label(intr::ForceFreeStatesInternal) = intr.npert == 1 ? "$(intr.nlow)" : "$(intr.nlow):$(intr.nhigh)"

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
    load_kinetic_context(inputs, intr, ctrl, equil) -> (kf_ctrl, kinetic_profiles)

Build the KineticForces control and load the kinetic profiles once for the whole run.
`kinetic_profiles` is `nothing` when no stage asks for them.
"""
function load_kinetic_context(
    inputs::Dict{String,Any},
    intr::ForceFreeStatesInternal,
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium
)
    # The profiles are reused by the grid refinement, the stability kinetic callback (via
    # `calculated_cb`), and the post-PE torque diagnostics block. The `"fixed"` kinetic source
    # path in stability does not need kinetic_profiles, but the post-PE block always does, so we
    # load whenever a [KineticForces] section is present or the stability path requests the
    # calculated source. psio is invariant across grid re-formation.
    kf_ctrl =
        haskey(inputs, "KineticForces") ?
        KineticForces.KineticForcesControl(;
            (Symbol(k) => v for (k, v) in inputs["KineticForces"])...) :
        KineticForces.KineticForcesControl()

    kinetic_profiles = nothing
    needs_kinetic_profiles = haskey(inputs, "KineticForces") ||
                             (ctrl.kinetic_factor > 0 && ctrl.kinetic_source == "calculated")
    if needs_kinetic_profiles
        kinetic_file = joinpath(intr.dir_path, kf_ctrl.kinetic_file)
        kinetic_profiles = Equilibrium.load_kinetic_profiles(
            kinetic_file;
            zi=kf_ctrl.zi, zimp=kf_ctrl.zimp,
            mi=kf_ctrl.mi, mimp=kf_ctrl.mimp,
            density_factor=kf_ctrl.density_factor, temperature_factor=kf_ctrl.temperature_factor,
            ExB_rotation_factor=kf_ctrl.ExB_rotation_factor, toroidal_rotation_factor=kf_ctrl.toroidal_rotation_factor,
            chi1=2π * equil.psio)
    end

    return kf_ctrl, kinetic_profiles
end

"""
    maybe_reform_equilibrium(equil, eq_config, additional_input, intr, ctrl, kinetic_profiles) -> equil

Two-pass auto grid: measure the pass-1 equilibrium's curvature (profiles, geometry, kinetic
profiles), pin knots on rational surfaces, and re-form on the refined grid from the in-memory
input — no file re-read. Returns `equil` untouched when the configuration wants a single pass.
"""
function maybe_reform_equilibrium(
    equil::Equilibrium.PlasmaEquilibrium,
    eq_config::Equilibrium.EquilibriumConfig,
    additional_input,
    intr::ForceFreeStatesInternal,
    ctrl::ForceFreeStatesControl,
    kinetic_profiles
)
    Equilibrium.wants_two_pass(eq_config) || return equil

    mandatory = ForceFreeStates.rational_psi_nodes(equil; nlow=intr.nlow, nhigh=intr.nhigh)
    # Smallest |n| in the run sets the widest matching half-stencil dpsi = singfac_min/(n_min·|q′|),
    # so the rational-surface brackets clear a zone large enough for every mode.
    n_min = minimum(abs(n) for n in intr.nlow:intr.nhigh if n != 0)
    psi_nodes = Equilibrium.refined_psi_grid(equil;
        tau=eq_config.psi_accuracy, kin=kinetic_profiles, mandatory=mandatory,
        singfac_min=ctrl.singfac_min, n_min=n_min)
    rerun_input = if additional_input !== nothing
        # Analytic *Config, IMAS dd, or prebuilt RunInput — all re-formable. The IMAS
        # path re-runs read_imas, which must resolve the same psihigh both passes;
        # _validate_psi_nodes errors loudly if it does not.
        additional_input
    elseif equil.ingest isa Equilibrium.DirectIngest
        Equilibrium.build_direct_from_ingest(eq_config, equil.ingest)
    elseif equil.ingest isa Equilibrium.InverseIngest
        Equilibrium.build_inverse_from_ingest(eq_config, equil.ingest)
    else
        nothing  # fall back to re-reading the input file
    end
    equil = Equilibrium.setup_equilibrium(eq_config, rerun_input; override_psi_nodes=psi_nodes)
    implied = Equilibrium.implied_knot_count(equil; tau=eq_config.psi_accuracy, kin=kinetic_profiles)
    if implied > 1.5 * (length(psi_nodes) - 1)
        @warn "Two-pass psi grid: refined equilibrium implies $implied knots vs $(length(psi_nodes) - 1) used — " *
              "pass 1 may have under-sampled a feature; consider tightening psi_accuracy"
    end
    @info "Two-pass psi grid: $(length(psi_nodes)) knots, $(length(mandatory)) rational surfaces pinned (n=$(_mode_range_label(intr)))"

    return equil
end

"""
    snapshot_forcing_modes(inputs, path, ctrl, preloaded) -> Union{Nothing,Vector{ForcingMode}}

Capture the file-based forcing modes before the solve, when PerturbedEquilibrium is enabled, so
they land in `Input/RawInputs/ForcingTerms/` alongside the TOML blob. Returns `preloaded`
unchanged when the caller (the rerun path) already supplied the modes.
"""
function snapshot_forcing_modes(
    inputs::Dict{String,Any},
    path::String,
    ctrl::ForceFreeStatesControl,
    preloaded::Union{Nothing,Vector{ForcingTerms.ForcingMode}}
)
    # Coil forcing is recomputed from the `[[ForcingTerms.coil_set]]` TOML blob on replay,
    # so only the file-based formats need their modes captured here.
    forcing_modes_snapshot = preloaded
    if forcing_modes_snapshot === nothing && "PerturbedEquilibrium" in keys(inputs)
        ft_raw = get(inputs, "ForcingTerms", Dict{String,Any}())
        scalar_forcing = filter(p -> p.first != "coil_set", ft_raw)
        ft_ctrl_snapshot = ForcingTerms.ForcingTermsControl(;
            (Symbol(k) => v for (k, v) in scalar_forcing)...
        )
        if ft_ctrl_snapshot.forcing_data_format in ("ascii", "hdf5")
            forcing_modes_snapshot = ForcingTerms.ForcingMode[]
            ForcingTerms.load_forcing_data!(
                forcing_modes_snapshot,
                path,
                ft_ctrl_snapshot.forcing_data_file,
                ft_ctrl_snapshot.forcing_data_format,
                ctrl.verbose
            )
        end
    end
    return forcing_modes_snapshot
end

"""
    run_local_stability(ctrl, equil) -> (locstab, ballooning_boundary)

Run the LocalStability stage when `local_stability_flag` is set. `locstab` holds `D_I` from the
ballooning coefficient system and the local ballooning result, `ballooning_boundary` the first
α-vs-ψ_N stability boundary; both are empty placeholders when the stage is off.
"""
function run_local_stability(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium)
    locstab = nothing
    ballooning_boundary = (psi=Float64[], alpha=Float64[], alpha_critical=Float64[])
    if ctrl.local_stability_flag
        locstab = LocalStability.compute_local_stability(equil; verbose=ctrl.verbose)
        # First ballooning stability boundary (α vs ψ_N) for BALOO-style diagnostics.
        ballooning_boundary = LocalStability.ballooning_alpha_boundary(equil; verbose=ctrl.verbose)
    end
    return locstab, ballooning_boundary
end

"""
    prepare_force_free_states!(intr, ctrl, equil, kf_ctrl, kinetic_profiles) -> (metric, mats)

Set up the force-free-states solve on `intr`: integration limits, the surviving singular
surfaces and their GGJ coefficients, the poloidal mode range, and the metric plus
Euler-Lagrange (and, when requested, kinetic) matrices.
"""
function prepare_force_free_states!(
    intr::ForceFreeStatesInternal,
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    kf_ctrl::KineticForces.KineticForcesControl,
    kinetic_profiles
)
    # Determine psilim and qlim (where we will integrate to)
    sing_lim!(intr, ctrl, equil)

    # Find all singular surfaces in the equilibrium
    sing_find!(intr, equil)

    # Filter out surfaces outside the integration domain [qlow, qlim].
    # Fortran STRIDE excludes these at the integration level; we remove them
    # from intr.sing so the Δ' BVP sees only crossable surfaces.
    if intr.msing > 0
        qmin_integration = max(ctrl.qlow, equil.params.qmin)
        n_before = intr.msing
        keep = [j for j in 1:intr.msing if intr.sing[j].q >= qmin_integration && intr.sing[j].psifac <= intr.psilim]
        if length(keep) < n_before
            excluded = setdiff(1:n_before, keep)
            excluded_mq = [(intr.sing[j].m, intr.sing[j].q) for j in excluded]
            @info "Filtered $(n_before - length(keep)) singular surface(s) outside integration domain: $(excluded_mq)"
            intr.sing = intr.sing[keep]
            intr.msing = length(keep)
        end
    end

    # For the outer-region Galerkin solve, exclude the q < qlow core (incl. any q≤1 sawtooth
    # surfaces) by raising psilow to where q = qlow (RDCON sing_min). Without this, the gal FEM
    # integrates through the unhandled q≤1 ideal singularity and contaminates Δ′ at the innermost
    # kept surface when q0 < qlow. No-op (keeps the axis bound) when qlow ≤ qmin.
    if ctrl.integrator == "galerkin"
        sing_min!(intr, ctrl, equil)
    end

    # Populate Glasser-Greene-Johnson geometric coefficients (E, F, G, H,
    # K, M) for each surviving singular surface. Needed by the Julia GGJ
    # inner-layer analysis; kinetic timescales (τ_A, τ_R) are layered on
    # top by `build_ggj_inputs` using the same kinetic profiles as SLAYER.
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
    if ctrl.verbose
        @info "Run parameters:\n" *
              "   q0 = $(@sprintf("%.3f", equil.params.q0)), qmin = $(@sprintf("%.3f", equil.params.qmin)), qmax = $(@sprintf("%.3f", equil.params.qmax)), q95 = $(@sprintf("%.3f", equil.params.q95))\n" *
              "   qlim = $(@sprintf("%.3f", intr.qlim)), psilim = $(@sprintf("%.3f", intr.psilim))\n" *
              "   betat = $(@sprintf("%.3f", equil.params.betat)), betan = $(@sprintf("%.3f", equil.params.betan)), betap1 = $(@sprintf("%.3f", equil.params.betap1))\n" *
              "   mlow = $(@sprintf("%4i", intr.mlow)), mhigh = $(@sprintf("%4i", intr.mhigh)), mpert = $(@sprintf("%4i", intr.mpert))\n" *
              "   nlow = $(@sprintf("%4i", intr.nlow)), nhigh = $(@sprintf("%4i", intr.nhigh)), npert = $(@sprintf("%4i", intr.npert))"
    end

    # Compute metric tensor
    metric = make_metric(equil, intr.mpert)

    if ctrl.verbose
        @info "Computing F, G, and K matrices"
    end

    # Compute matrices and build the MatrixSplines container
    mats = build_matrix_splines(equil, intr, metric)

    if ctrl.kinetic_factor > 0
        if ctrl.verbose
            @info "Computing kinetic matrices (source: $(ctrl.kinetic_source), factor: $(ctrl.kinetic_factor))"
        end
        # Inject the KineticForces callback so the "calculated" source can
        # invoke compute_calculated_kinetic_matrices without ForceFreeStates
        # importing KineticForces (which would invert the load order).
        calculated_cb = (c, e, i, m, f) ->
            KineticForces.compute_calculated_kinetic_matrices(
                c, e, i, m, f;
                kf_ctrl=kf_ctrl, kinetic_profiles=kinetic_profiles)
        mats = build_kinetic_matrix_splines(ctrl, equil, mats, intr, metric;
            calculated_source=calculated_cb)

        # Find kinetically-displaced singular surfaces (zeros of det(F̄)) for ODE crossings.
        # Matches Fortran ksing_find (sing.f:1486-1616). singfac_min > 0 gates crossings;
        # singfac_min == 0 preserves single-chunk behavior.
        if ctrl.singfac_min > 0
            find_kinetic_singular_surfaces!(mats, equil, intr)
        end
    end

    return metric, mats
end

"""
    run_force_free_states(ctrl, equil, mats, intr, metric) -> ForceFreeStatesResult

Run the formalism selected by `ctrl.integrator` — the standalone Galerkin solve, or the
Euler-Lagrange sweep with its free-boundary energies and Δ′ BVP — and publish its products as a
`ForceFreeStatesResult`.
"""
function run_force_free_states(
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    mats,
    intr::ForceFreeStatesInternal,
    metric
)
    nstring = _mode_range_label(intr)

    # The three formalisms are exclusive. Galerkin solves the same Euler-Lagrange system
    # variationally rather than by radial ODE integration, so it replaces both the integration
    # and the free-boundary energies, and supplies its own vacuum response at psilim.
    odet = nothing
    free_energies = nothing
    gal_data = nothing
    gal_dp = nothing
    if ctrl.integrator == "galerkin"
        ctrl.kinetic_factor == 0 ||
            error("integrator = \"galerkin\" does not support kinetic runs (kinetic_factor > 0); use integrator = \"forward\".")
        gal_start = time()
        wv = ctrl.vac_flag ? first(ForceFreeStates.compute_scaled_wv(ctrl, equil, intr)) : nothing
        gal_data, gal_dp = galerkin_solve(ctrl, equil, mats, intr; wv=wv)
        @info "Galerkin solve completed in $(@sprintf("%.3f", time() - gal_start)) s"
    else
        # Integrate Euler-Lagrange Equation
        if ctrl.verbose
            @info "Integrating Euler-Lagrange equation"
        end
        odet, fm_propagators, fm_chunks, fm_S_left = eulerlagrange_integration(ctrl, equil, mats, intr)
        if odet.nzero > 0 && ctrl.verbose
            @warn "Fixed-boundary mode unstable for n = $nstring"
        end

        # Compute free boundary energies.
        if ctrl.vac_flag && !(ctrl.ksing > 0 && ctrl.ksing <= intr.msing + 1)
            if ctrl.verbose
                wall_desc = intr.wall_settings.shape == "nowall" ? "no wall" : intr.wall_settings.shape
                @info "Computing free boundary energies ($wall_desc)"
            end
            free_energies = free_run(odet, ctrl, equil, mats, intr)
            normalize_eigenfunctions!(odet, free_energies.wt, equil.psio)
            if real(free_energies.et[1]) < 0
                if ctrl.verbose
                    @warn "Free-boundary mode unstable for n = $nstring"
                end
            else
                if ctrl.verbose
                    @info "All free-boundary modes stable for n = $nstring"
                end
            end

            # Compute inter-surface Δ' matrix (STRIDE BVP) using vacuum edge BC.
            # Requires propagators from parallel FM path and wv from free_run.
            if ctrl.kinetic_factor == 0 && intr.msing > 0 && fm_propagators !== nothing
                if ctrl.verbose
                    @info "Computing Δ' matrix (STRIDE BVP with vacuum coupling)"
                end
                ForceFreeStates.compute_delta_prime_matrix!(intr, fm_propagators, fm_chunks;
                    wv=free_energies.wv, psio=equil.psio, debug=ctrl.verbose,
                    S_at_surface_left=fm_S_left,
                    ctrl=ctrl, equil=equil, mats=mats)
            end
        end
    end

    # Publish the solve: from here on the downstream stages read the result, never `intr`.
    return build_result(Symbol(ctrl.integrator), ctrl, equil, intr, metric, mats, odet, free_energies, gal_data, gal_dp)
end

"""
    EulerLagrangeProblem(equil; nn, wall=Vacuum.WallShapeSettings(), match=nothing,
                         dir_path=".", debug=DebugSettings(), kwargs...)

The perturbed-plasma Euler-Lagrange problem posed on an equilibrium: the extremization of
the perturbed potential energy whose solutions are the force-free (and, via the TOML path,
kinetic) perturbed states. This is the WHAT of a stability solve; the integrator passed to
[`solve`](@ref) is the HOW. A `PlasmaEquilibrium` hosts many possible problems — this type
names this one, so `solve` stays unambiguous as other problem classes appear.

`nn` is the toroidal mode number or range. `wall` is the vacuum wall shape, `match` an
optional [`ResistiveMatch`](@ref) closing the basis with an inner-layer solution instead of
the ideal jump, `dir_path` the working directory outputs are written to, and `debug` the
diagnostic dump settings of the DEBUG deck section. Any remaining keyword is a
`ForceFreeStatesControl` field, so the TOML keys and the problem keywords are the same
knobs. `nn_low`/`nn_high` are rejected — they come from `nn`.

## Fields

  - `equil::Equilibrium.PlasmaEquilibrium` - The equilibrium the problem is posed on.
  - `wall::Vacuum.WallShapeSettings` - Vacuum wall shape for the free-boundary energies.
  - `match::Union{Nothing,ForceFreeStates.ResistiveMatch}` - Optional inner-layer closure.
  - `dir_path::String` - Working directory for outputs.
  - `debug::DebugSettings` - Diagnostic dump settings.
  - `ctrl_kwargs::Dict{Symbol,Any}` - `ForceFreeStatesControl` keywords, `nn` already folded in.
"""
struct EulerLagrangeProblem
    equil::Equilibrium.PlasmaEquilibrium
    wall::Vacuum.WallShapeSettings
    match::Union{Nothing,ForceFreeStates.ResistiveMatch}
    dir_path::String
    debug::DebugSettings
    ctrl_kwargs::Dict{Symbol,Any}
end

function EulerLagrangeProblem(
    equil::Equilibrium.PlasmaEquilibrium;
    nn::Union{Int,AbstractUnitRange{Int}},
    wall::Vacuum.WallShapeSettings=Vacuum.WallShapeSettings(),
    match::Union{Nothing,ForceFreeStates.ResistiveMatch}=nothing,
    dir_path::AbstractString=".",
    debug::DebugSettings=DebugSettings(),
    kwargs...
)
    ctrl_kwargs = Dict{Symbol,Any}(kwargs)
    (haskey(ctrl_kwargs, :nn_low) || haskey(ctrl_kwargs, :nn_high)) &&
        error("the toroidal mode range comes from the `nn` keyword; drop nn_low/nn_high")
    ctrl_kwargs[:nn_low] = first(nn)
    ctrl_kwargs[:nn_high] = last(nn)
    return EulerLagrangeProblem(equil, wall, match, String(dir_path), debug, ctrl_kwargs)
end

"""
    solve(prob::EulerLagrangeProblem, alg) -> ForceFreeStatesResult
    solve(equil, alg; nn, kwargs...) -> ForceFreeStatesResult

Solve the perturbed-plasma [`EulerLagrangeProblem`](@ref) with the formalism `alg` —
[`Forward`](@ref), [`Riccati`](@ref) or [`Galerkin`](@ref) — and return the published
[`ForceFreeStatesResult`](@ref). This is the scripting entry point; it runs the same stages
a `gpec.toml` run of `main` does and produces the same result object. The second form is
sugar building the problem from an equilibrium and the problem keywords in one call.

Knobs owned by `alg` or `match` are rejected as `ForceFreeStatesControl` keywords. Kinetic
runs are TOML-driven this cycle: `kinetic_factor > 0` needs the `[KineticForces]` profiles
and errors here.

```julia
eq   = PlasmaEquilibrium("input.geqdsk"; jac_type="hamada")
prob = EulerLagrangeProblem(eq; nn=1, delta_mlow=8, delta_mhigh=8, vac_flag=true)
ffs  = solve(prob, Riccati())
ffs  = solve(eq, Riccati(); nn=1, vac_flag=true)   # equivalent one-line form
```
"""
function solve(prob::EulerLagrangeProblem, alg::ForceFreeStates.AbstractIntegrator)
    total_start = time()

    equil = prob.equil
    ctrl_kwargs = copy(prob.ctrl_kwargs)
    ForceFreeStates._apply_alg!(ctrl_kwargs, alg)
    ForceFreeStates._apply_match!(ctrl_kwargs, prob.match, alg)
    ctrl = ForceFreeStatesControl(; ctrl_kwargs...)

    ctrl.kinetic_factor > 0 &&
        error("kinetic runs (kinetic_factor > 0) need the [KineticForces] profiles and are TOML-driven; run them through `main`")

    intr = ForceFreeStatesInternal(; dir_path=prob.dir_path)
    intr.wall_settings = prob.wall
    intr.debug_settings = prob.debug

    resolve_mode_space!(intr, ctrl)

    # The API path never reads kinetic profiles, so the KineticForces control is only the
    # placeholder `prepare_force_free_states!` threads into its (unused) callback.
    kf_ctrl = KineticForces.KineticForcesControl()

    if Equilibrium.wants_two_pass(equil.config) && equil.ingest === nothing
        @warn "Two-pass auto grid needs the equilibrium's raw ingest, which analytic and IMAS equilibria do not carry; " *
              "solving on the single-pass grid. Set mpsi explicitly to choose the grid."
    else
        equil = maybe_reform_equilibrium(equil, equil.config, nothing, intr, ctrl, nothing)
    end

    locstab, ballooning_boundary = run_local_stability(ctrl, equil)
    metric, mats = prepare_force_free_states!(intr, ctrl, equil, kf_ctrl, nothing)
    result = run_force_free_states(ctrl, equil, mats, intr, metric)

    if ctrl.write_outputs_to_HDF5
        write_outputs_to_HDF5(result; locstab=locstab, ballooning_boundary=ballooning_boundary)
        @info "Results written to $(ctrl.HDF5_filename)"
    end

    @info "Force-Free States completed in $(@sprintf("%.3f", time() - total_start)) s"

    return result
end

"""
    solve(equil::PlasmaEquilibrium, alg; nn, kwargs...) -> ForceFreeStatesResult

Convenience form of [`solve`](@ref): builds the [`EulerLagrangeProblem`](@ref) from the
equilibrium and the problem keywords, then solves it with `alg`.
"""
solve(equil::Equilibrium.PlasmaEquilibrium, alg::ForceFreeStates.AbstractIntegrator; kwargs...) =
    solve(EulerLagrangeProblem(equil; kwargs...), alg)

"""
    run_perturbed_equilibrium(result, inputs, forcing_modes_snapshot, preloaded_coil_sets) -> pe_state

Run the perturbed-equilibrium stage against a published force-free-states `result` and write its
outputs. Returns `nothing` when the deck carries no `[PerturbedEquilibrium]` section.
"""
function run_perturbed_equilibrium(
    result::ForceFreeStatesResult,
    inputs::Dict{String,Any},
    forcing_modes_snapshot::Union{Nothing,Vector{ForcingTerms.ForcingMode}},
    preloaded_coil_sets::Union{Nothing,Vector{ForcingTerms.CoilSet}}
)
    # ----------------------------------------------------------------
    # Perturbed Equilibrium
    # ----------------------------------------------------------------
    @info "\n  Perturbed Equilibrium\n$_SECTION"
    pe_start = time()

    # Check for PerturbedEquilibrium section and run if present
    pe_state = nothing
    if "PerturbedEquilibrium" in keys(inputs)
        # Read ForcingTerms control parameters
        if "ForcingTerms" in keys(inputs)
            forcing_raw = inputs["ForcingTerms"]
            # [[ForcingTerms.coil_set]] becomes a Vector{Dict} — must be excluded from
            # kwarg splatting and passed as the explicit coil_sets_raw keyword
            coil_sets_raw = Vector{Dict{String,Any}}(get(forcing_raw, "coil_set", Dict{String,Any}[]))
            scalar_forcing = filter(p -> p.first != "coil_set", forcing_raw)
            ft_ctrl = ForcingTerms.ForcingTermsControl(;
                (Symbol(k) => v for (k, v) in scalar_forcing)..., coil_sets_raw=coil_sets_raw
            )
        else
            ft_ctrl = ForcingTerms.ForcingTermsControl()  # Use defaults
        end

        # The deck's forcing block is an unscaled RMPField, so the TOML path and the
        # scripting API share one stage.
        pe_state = perturbed_equilibrium(result, ForcingTerms.RMPField(ft_ctrl);
            forcing_modes=forcing_modes_snapshot, coil_sets=preloaded_coil_sets,
            (Symbol(k) => v for (k, v) in inputs["PerturbedEquilibrium"])...)
    end

    @info "Perturbed Equilibrium completed in $(@sprintf("%.3f", time() - pe_start)) s"

    return pe_state
end

"""
    perturbed_equilibrium(ffs, rmp; forcing_modes=nothing, coil_sets=nothing, kwargs...) -> PerturbedEquilibriumState

Compute the plasma response to the external field `rmp` on top of a force-free-states solve
`ffs`, and write the perturbed-equilibrium outputs. `rmp` is an [`RMPField`](@ref); keyword
arguments are `PerturbedEquilibrium.PerturbedEquilibriumControl` fields.

Products the producing integrator could not supply gate the corresponding calculation: the
step warns and is skipped rather than erroring, so a Riccati- or Galerkin-fed result still
flows through.

`forcing_modes` injects already-loaded modes (the gpec.h5 replay path) and `coil_sets`
already-built coil geometry, both bypassing the corresponding read.

```julia
pe = perturbed_equilibrium(ffs, RMPField("forcing.dat"))
```
"""
function perturbed_equilibrium(
    ffs::ForceFreeStatesResult,
    rmp::ForcingTerms.RMPField;
    forcing_modes::Union{Nothing,Vector{ForcingTerms.ForcingMode}}=nothing,
    coil_sets::Union{Nothing,Vector{ForcingTerms.CoilSet}}=nothing,
    kwargs...
)
    ctrl = ffs.control
    pe_ctrl = PerturbedEquilibrium.PerturbedEquilibriumControl(; kwargs...)
    pe_intr = PerturbedEquilibrium.PerturbedEquilibriumInternal(; dir_path=ffs.dir_path)

    # Inner-layer penetrated resonant field; zeros under ideal closure.
    pe_intr.inner_bpen = ffs.bpen

    # Reuse forcing modes loaded at snapshot time (or injected by `build_inputs_from_h5`) so
    # the materialization step never re-reads the original forcing file.
    if forcing_modes !== nothing
        pe_intr.forcing_modes = copy(forcing_modes)
    end

    # Injected coil geometry (gpec.h5 replay with `--coil-source coils`) lets the coil field
    # be recomputed from stored geometry without the .dat/.h5 files.
    if coil_sets !== nothing
        pe_intr.coil_sets = copy(coil_sets)
    end

    pe_state = PerturbedEquilibrium.compute_perturbed_equilibrium(ffs, rmp, pe_ctrl, pe_intr)

    # Write perturbed equilibrium outputs to same HDF5 file
    if pe_ctrl.write_outputs_to_HDF5
        output_file = isempty(pe_ctrl.output_filename) ? ctrl.HDF5_filename : pe_ctrl.output_filename
        PerturbedEquilibrium.write_outputs_to_HDF5(
            pe_state, pe_intr, joinpath(ffs.dir_path, output_file)
        )
        @info "Results written to $output_file"
    end

    # Snapshot the coil geometry actually used into the gpec.h5 output so the run
    # is replayable from the output file alone (see `main_from_h5 --coil-source coils`).
    if ctrl.write_outputs_to_HDF5 && !isempty(pe_intr.coil_sets)
        _write_coil_snapshot!(joinpath(ffs.dir_path, ctrl.HDF5_filename), pe_intr.coil_sets)
    end

    return pe_state
end

"""
    run_kinetic_forces(inputs, result, pe_state, kf_ctrl, kinetic_profiles)

Compute and write the neoclassical toroidal viscosity torque diagnostics when the deck carries a
`[KineticForces]` section. No-op when the perturbed-equilibrium state the operators contract
against is missing.
"""
function run_kinetic_forces(
    inputs::Dict{String,Any},
    result::ForceFreeStatesResult,
    pe_state,
    kf_ctrl::KineticForces.KineticForcesControl,
    kinetic_profiles
)
    # ----------------------------------------------------------------
    # KineticForces (Neoclassical Toroidal Viscosity)
    # ----------------------------------------------------------------
    ("KineticForces" in keys(inputs)) || return nothing

    @info "\n  KineticForces\n$_SECTION"
    kf_start = time()

    # Standalone NTV torque diagnostics need a PE state (they contract kinetic operators
    # against ξ). The self-consistent kinetic_source="calculated" path produces none — skip.
    if pe_state === nothing
        @info "Skipping NTV torque diagnostics: no perturbed-equilibrium data (e.g. kinetic_source=\"calculated\")."
    else
        # kf_ctrl and kinetic_profiles were loaded once before the equilibrium was re-formed.
        kf_intr = KineticForces.KineticForcesInternal(result.equil; verbose=kf_ctrl.verbose)
        KineticForces.set_perturbation_data!(kf_intr, pe_state, result, result.equil, result.metric)

        kf_state = KineticForces.KineticForcesState()
        KineticForces.compute_torque_all_methods!(kf_state, kf_intr, kf_ctrl, result.equil, kinetic_profiles)

        if kf_ctrl.write_outputs_to_HDF5
            h5open(joinpath(result.dir_path, kf_ctrl.HDF5_filename), "cw") do h5file
                KineticForces.write_to_hdf5!(h5file, kf_state; dVdpsi_spline=result.equil.profiles.dVdpsi_spline)
            end
        end
    end

    @info "KineticForces completed in $(@sprintf("%.3f", time() - kf_start)) s"

    return nothing
end

"""
    run_slayer_stage(result, inputs, pe_file) -> slayer_result

Run the SLAYER tearing-mode analysis off the force-free-states `result`, appending its group to
`pe_file` (or the force-free-states output when PE did not run). Needs only the result, so it
runs in both the `force_termination = true` path and the full pipeline.
"""
function run_slayer_stage(result::ForceFreeStatesResult, inputs::Dict{String,Any}, pe_file::Union{String,Nothing})
    ("SLAYER" in keys(inputs)) || return nothing
    # SLAYER is a post-processing diagnostic. A failure here must not
    # discard the equilibrium / stability / PE results already computed,
    # so the whole stage is guarded: on error we log loudly and return
    # `nothing` for the `slayer` field rather than propagating.
    try
        slayer_ctrl = Runner.slayer_control_from_toml(inputs["SLAYER"])
        slayer_ctrl.enabled || return nothing
        @info "\n  SLAYER\n$_SECTION"
        slayer_start = time()
        slayer_result = Runner.run_slayer(result, slayer_ctrl;
            dir_path=result.dir_path)
        @info "SLAYER completed in $(@sprintf("%.3f", time() - slayer_start)) s"
        h5_filename = pe_file === nothing ? result.control.HDF5_filename : pe_file
        h5_path = joinpath(result.dir_path, h5_filename)
        # Append the Tearing/ group; create the file if no prior stage wrote
        # it (e.g. write_outputs_to_HDF5 disabled) rather than failing on "r+".
        HDF5.h5open(h5_path, isfile(h5_path) ? "r+" : "w") do f
            Runner.write_slayer_hdf5!(f, slayer_result)
        end
        @info "SLAYER results written to $h5_filename"
        return slayer_result
    catch err
        @error "SLAYER stage failed; continuing without tearing results. " *
               "Equilibrium / stability / PE outputs are unaffected." exception =
            (err, catch_backtrace())
        return nothing
    end
end

"""
    write_outputs_to_HDF5(result::ForceFreeStatesResult; git_version, inputs, forcing_modes,
                          locstab, ballooning_boundary)

Write the HDF5 output file with the run, equilibrium and stability products carried by a
force-free-states `result`. This combines the functionality of several pieces of the Fortran
code in `ode_output.f`, primarily `ode_output_open` and the various `bin_euler` writes that
occur throughout the integration. Groups fed by an optional result field fall back to empty
datasets when the producing integrator did not supply it, so the schema is the same shape
whatever ran.

The solution-adjacent datasets come from two independent sources: the dense
`Solutions/ForwardIntegration/xi_*` arrays from `result.solution` when it is in the forward
`:el_axis` basis (a matched Galerkin solution is persisted in full under the Galerkin group
instead), and ψ, q, the step counts, `crit`, the asymptotic coefficients and the edge scan
from `result.diagnostics`. Either may be absent, and is then written empty.

`locstab` and `ballooning_boundary` come from the LocalStability stage, which is not part of
the force-free-states solve.

### TODOs

Combine spline unpacking if possible, too many extra lines
"""
function write_outputs_to_HDF5(
    result::ForceFreeStatesResult;
    git_version::String="unknown",
    inputs::Union{Nothing,Dict{String,Any}}=nothing,
    forcing_modes::Union{Nothing,Vector{ForcingTerms.ForcingMode}}=nothing,
    locstab::Union{FastInterpolations.CubicSeriesInterpolant,Nothing}=nothing,
    ballooning_boundary=(psi=Float64[], alpha=Float64[], alpha_critical=Float64[])
)

    ctrl = result.control
    equil = result.equil
    mats = result.mats
    free_energies = result.free_boundary
    gal_data = result.galerkin
    diag = result.diagnostics
    msing = length(result.surfaces)
    # Closed ξ profiles are written into the producing formalism's Solutions group with the
    # same dataset names and (mode, solution, psi) axis order: ForwardIntegration for the
    # axis-basis sweep, GalerkinIntegration for the matched gal solution (its own grid).
    xi_solution = (result.solution !== nothing && result.solution.basis === :el_axis) ? result.solution : nothing
    gal_solution = (result.solution !== nothing && result.solution.basis === :gal_native) ? result.solution : nothing

    h5open(joinpath(result.dir_path, ctrl.HDF5_filename), "w") do out_h5

        # File-level metadata contract (schema_version, Conventions, title, date).
        Utilities.HDF5Annotations.write_root_attrs!(out_h5; title="GPEC output: $(basename(abspath(result.dir_path)))")

        # Store git version for reproducibility
        out_h5["Info/git_version"] = git_version

        # Outer-region Galerkin solver outputs (RDCON), if it ran
        if gal_data !== nothing
            write_galerkin!(out_h5, gal_data; basis_output=result.debug_settings.gal_basis_output)
        end

        if gal_solution !== nothing
            gal = "ForceFreeStates/Solutions/GalerkinIntegration"
            out_h5["$gal/psi"] = gal_solution.psi_store
            out_h5["$gal/q"] = gal_solution.q_store
            out_h5["$gal/xi_psi"] = gal_solution.u_store[:, :, 1, :]
            out_h5["$gal/dxi_psidpsi"] = gal_solution.du_store
            out_h5["$gal/xi_s"] = gal_solution.xi_s_store
        end

        # Self-contained run snapshot: the full merged TOML (so a rerun can reconstruct every
        # ForceFreeStates/Equilibrium/Wall/PE control struct), plus the equilibrium ingest
        # arrays so a file-based rerun never needs the original g-file / CHEASE / IMAS source.
        if inputs !== nothing
            out_h5["Input/gpec_toml_raw"] = sprint(TOML.print, inputs)
        end
        if equil.ingest !== nothing  # analytic equilibria are regenerated from their TOML section
            eq_group = "Input/RawInputs/Equilibrium"  # read back by Rerun.read_equilibrium_ingest
            out_h5["$eq_group/ingest_kind"] = equil.ingest isa Equilibrium.DirectIngest ? "direct" : "inverse"
            for f in fieldnames(typeof(equil.ingest))
                out_h5["$eq_group/$f"] = getfield(equil.ingest, f)
            end
        end
        if forcing_modes !== nothing
            forcing_group = create_group(out_h5, "Input/RawInputs/ForcingTerms")
            ForcingTerms.save_forcing_to_h5(forcing_modes, forcing_group)
        end

        # Write derived run parameters
        out_h5["Info/mpert"] = result.mpert
        out_h5["Info/mlow"] = result.mlow
        out_h5["Info/mhigh"] = result.mhigh
        out_h5["Info/npert"] = result.npert
        out_h5["Info/nlow"] = result.nlow
        out_h5["Info/nhigh"] = result.nhigh
        m = [(i - 1) % result.mpert + result.mlow for i in 1:(result.numpert_total)]
        n = [(i - 1) ÷ result.mpert + result.nlow for i in 1:(result.numpert_total)]
        out_h5["Info/mn_index"] = hcat(m, n)   # (N, 2) matrix
        out_h5["Info/psilim"] = result.psilim
        out_h5["Info/qlim"] = result.qlim
        out_h5["Info/dqdpsi_lim"] = result.q1lim

        # Write derived equilibrium parameters. The struct keeps its legacy field spellings;
        # EQUIL_H5_NAMES maps them to literature dataset names and EQUIL_H5_SKIP drops
        # duplicates and control-flag echoes. Fields left `nothing` are not written.
        for f in fieldnames(Equilibrium.EquilibriumParameters)
            f in EQUIL_H5_SKIP && continue
            val = getfield(equil.params, f)
            val === nothing && continue
            out_h5["Equilibrium/$(get(EQUIL_H5_NAMES, f, String(f)))"] = val
        end
        out_h5["Equilibrium/psi_total"] = equil.psio
        out_h5["Equilibrium/R_axis"] = equil.ro
        out_h5["Equilibrium/Z_axis"] = equil.zo

        # Write equilibrium profile and geometry arrays (from the named splines)
        profiles = equil.profiles
        out_h5["Equilibrium/Profiles/psi"] = profiles.xs
        out_h5["Equilibrium/Profiles/2piF"] = profiles.F_spline.y
        out_h5["Equilibrium/Profiles/mu0p"] = profiles.P_spline.y
        out_h5["Equilibrium/Profiles/dVdpsi"] = profiles.dVdpsi_spline.y
        out_h5["Equilibrium/Profiles/q"] = profiles.q_spline.y
        out_h5["Equilibrium/Geometry/psi"] = equil.rzphi_xs
        out_h5["Equilibrium/Geometry/theta"] = equil.rzphi_ys
        # Extract grid point values from interpolants for HDF5 output
        out_h5["Equilibrium/Geometry/rcoords"] = equil.rzphi_rsquared.nodal_derivs.partials[1, :, :]
        out_h5["Equilibrium/Geometry/offset"] = equil.rzphi_offset.nodal_derivs.partials[1, :, :]
        out_h5["Equilibrium/Geometry/nu"] = equil.rzphi_nu.nodal_derivs.partials[1, :, :]
        out_h5["Equilibrium/Geometry/jac"] = equil.rzphi_jac.nodal_derivs.partials[1, :, :]

        # Write local stability data; always write all entries, using empty arrays when not computed.
        # LocalStability/D_I = Mercier D_I (det(d0bar)); LocalStability/D_R = resistive interchange D_R;
        # LocalStability/ballooning_Delta_prime = high-n ballooning Δ' (distinct from the Riccati
        # tearing Δ' under PerturbedEquilibrium/SingularCoupling/Delta_prime).
        if locstab !== nothing
            locstab_xs = locstab.cache.x
            out_h5["LocalStability/psi"] = collect(locstab_xs)  # cached vector → dense for HDF5
            out_h5["LocalStability/D_I"] = locstab.y[:, 1] ./ locstab_xs
            out_h5["LocalStability/D_R"] = locstab.y[:, 2] ./ locstab_xs
        else
            out_h5["LocalStability/psi"] = Float64[]
            out_h5["LocalStability/D_I"] = Float64[]
            out_h5["LocalStability/D_R"] = Float64[]
        end
        out_h5["SingularSurfaces/D_I"] = (locstab !== nothing && !isempty(result.surfaces)) ?
                                         [locstab(sing.psifac)[1] / sing.psifac for sing in result.surfaces] : Float64[]
        out_h5["LocalStability/ballooning_Delta_prime"] = locstab !== nothing ? locstab.y[:, 4] : Float64[]

        # First ballooning stability boundary: experimental α vs critical α (BALOO-style).
        # Its own scan grid, distinct from the LocalStability/psi profile grid above.
        out_h5["LocalStability/ballooning_psi"] = ballooning_boundary.psi
        out_h5["LocalStability/alpha"] = ballooning_boundary.alpha
        out_h5["LocalStability/alpha_critical"] = ballooning_boundary.alpha_critical

        # Write integration data: the ψ trace and integrator diagnostics from the raw ODE state,
        # the ξ profiles from the solution. Either may be absent (Galerkin has no ODE state;
        # Riccati has no ξ solution), in which case the datasets are written empty.
        fwd = "ForceFreeStates/Solutions/ForwardIntegration"
        out_h5["$fwd/nstep"] = diag !== nothing ? diag.step : 0            # Number of saved solution snapshots
        out_h5["$fwd/nstep_total"] = diag !== nothing ? diag.total_steps : 0  # Total ODE solver steps taken
        out_h5["$fwd/psi"] = diag !== nothing ? diag.psi_store : Float64[]
        out_h5["$fwd/q"] = diag !== nothing ? diag.q_store : Float64[]
        out_h5["$fwd/xi_psi"] = xi_solution !== nothing ? xi_solution.u_store[:, :, 1, :] : ComplexF64[]
        out_h5["$fwd/u2"] = xi_solution !== nothing ? xi_solution.u_store[:, :, 2, :] : ComplexF64[] # TODO: what to name this? These are the "conjugate momenta" of u1
        out_h5["$fwd/dxi_psidpsi"] = xi_solution !== nothing ? xi_solution.du_store : ComplexF64[]
        out_h5["$fwd/xi_s"] = xi_solution !== nothing ? xi_solution.xi_s_store : ComplexF64[]
        out_h5["$fwd/crit"] = diag !== nothing ? diag.crit_store : Float64[]

        # Write edge stability scan data (only present when psiedge < psilim).
        # Generalized (W, N) pencil energies — power-normalized, Jacobian-invariant; these are
        # the values findmax_dW_edge! uses to choose the truncation point.
        if diag !== nothing && !isempty(diag.edge_scan.psi)
            es = diag.edge_scan
            out_h5["ForceFreeStates/EdgeScan/psi"] = es.psi
            out_h5["ForceFreeStates/EdgeScan/q"] = es.q
            out_h5["ForceFreeStates/EdgeScan/total_energy"] = es.total_eigenvalue
            out_h5["ForceFreeStates/EdgeScan/plasma_energy"] = es.plasma_energy
            out_h5["ForceFreeStates/EdgeScan/vacuum_energy"] = es.vacuum_energy
            out_h5["ForceFreeStates/EdgeScan/vacuum_eigenvalue"] = es.vacuum_eigenvalue
        end

        # Write singular surface data
        out_h5["SingularSurfaces/rational_count"] = msing
        out_h5["SingularSurfaces/rational_psi"] = [sing.psifac for sing in result.surfaces]
        out_h5["SingularSurfaces/rational_q"] = [sing.q for sing in result.surfaces]
        out_h5["SingularSurfaces/dqdpsi"] = [sing.q1 for sing in result.surfaces]
        out_h5["SingularSurfaces/ca_left"] = diag !== nothing ? diag.ca_l : ComplexF64[]
        out_h5["SingularSurfaces/ca_right"] = diag !== nothing ? diag.ca_r : ComplexF64[]

        if msing > 0
            # Mode numbers at each surface (jagged — pad with 0 to max_modes width)
            max_modes = maximum(s -> length(s.m), result.surfaces)
            m_matrix = zeros(Int, msing, max_modes)
            n_matrix = zeros(Int, msing, max_modes)
            for (s, sing) in enumerate(result.surfaces)
                for i in 1:length(sing.m)
                    m_matrix[s, i] = sing.m[i]
                    n_matrix[s, i] = sing.n[i]
                end
            end
            out_h5["SingularSurfaces/rational_m"] = m_matrix
            out_h5["SingularSurfaces/rational_n"] = n_matrix

            # Glasser-Greene-Johnson geometric coefficients + surface averages
            # (populated by ForceFreeStates.resist_eval_all! after sing_find!).
            # Both kinetic-free (E, F, G, H, K, M) and geometry-only
            # (avg_bsq_over_dpsisq, avg_bsq) quantities are written so
            # downstream consumers (Tearing.InnerLayer.GGJ.build_ggj_inputs)
            # can reconstruct τ_A / τ_R from any kinetic-profile source.
            if all(s -> s.restype !== nothing, result.surfaces)
                out_h5["SingularSurfaces/E"] = [s.restype.E for s in result.surfaces]
                out_h5["SingularSurfaces/F"] = [s.restype.F for s in result.surfaces]
                out_h5["SingularSurfaces/G"] = [s.restype.G for s in result.surfaces]
                out_h5["SingularSurfaces/H"] = [s.restype.H for s in result.surfaces]
                out_h5["SingularSurfaces/K"] = [s.restype.K for s in result.surfaces]
                out_h5["SingularSurfaces/M"] = [s.restype.M for s in result.surfaces]
                out_h5["SingularSurfaces/avg_bsq_over_dpsisq"] = [s.restype.avg_bsq_over_dpsisq for s in result.surfaces]
                out_h5["SingularSurfaces/avg_bsq"] = [s.restype.avg_bsq for s in result.surfaces]
                out_h5["SingularSurfaces/mu0p"] = [s.restype.p_local for s in result.surfaces]
                out_h5["SingularSurfaces/dmu0pdpsi"] = [s.restype.p1_local for s in result.surfaces]
                out_h5["SingularSurfaces/dVdpsi"] = [s.restype.v1_local for s in result.surfaces]
            end
        end

        # Per-surface ca-based Δ' (`sing.delta_prime`) is a stub; only the BVP matrix is emitted (see SingType.delta_prime docstring).

        # Write the Δ' payload on one set of canonical paths, whichever formalism produced it
        # (Riccati BVP or Galerkin): both compute the same quantities in the same PEST-3
        # convention. Surface indexing follows the producing formalism's surface list, which for
        # Galerkin is the in-domain in-band subset of `SingularSurfaces/`.
        dp = result.delta_prime
        if msing > 0 && dp !== nothing
            # Inter-surface Δ' matrix, shape [msing × msing] — PEST3-convention deltap.
            out_h5["SingularSurfaces/Delta_prime_matrix"] = dp.matrix

            # Edge coil-response matrix, stored (numpert_total × 2msing) = (edge mode, surface-side)
            # so H5Web heatmaps read x = edge mode, y = surface-side. The carried matrix stays
            # (2msing × numpert_total); transpose only at write.
            isempty(dp.coil) || (out_h5["SingularSurfaces/Delta_coil"] = permutedims(dp.coil))

            # Raw 2msing×2msing outer-region D' matrix in side-major ordering
            # [L_s1, R_s1, L_s2, R_s2, …]. Byte-compatible with Fortran
            # rdcon/gal.f::gal_write_delta top 2msing×2msing block of delta_gw.dat.
            # Needed for the full det(D' − D(γ)) = 0 eigenvalue problem via
            # pest3_decompose to recover (A', B', Γ', Δ').
            isempty(dp.raw) || (out_h5["SingularSurfaces/Delta_prime_raw"] = dp.raw)

            # Remaining PEST-3 parity blocks, when the formalism persisted them (Galerkin);
            # Riccati recovers them from Delta_prime_raw via pest3_decompose.
            dp.A === nothing || (out_h5["SingularSurfaces/pest3_A"] = dp.A)
            dp.B === nothing || (out_h5["SingularSurfaces/pest3_B"] = dp.B)
            dp.Gamma === nothing || (out_h5["SingularSurfaces/pest3_Gamma"] = dp.Gamma)
        end

        # Write kinetic singular surface data (det(F̄) near-zeros) and the cond(F̄) scan
        # used to find them. Populated only when kinetic crossings were searched for.
        out_h5["SingularSurfaces/Kinetic/rational_count"] = result.kinetic.kmsing
        out_h5["SingularSurfaces/Kinetic/rational_psi"] = [s.psifac for s in result.kinetic.kinsing]
        out_h5["SingularSurfaces/Kinetic/rational_q"] = [s.q for s in result.kinetic.kinsing]
        out_h5["SingularSurfaces/Kinetic/dqdpsi"] = [s.q1 for s in result.kinetic.kinsing]
        out_h5["SingularSurfaces/Kinetic/scan_psi"] = result.kinetic.scan_psi
        out_h5["SingularSurfaces/Kinetic/scan_cond"] = result.kinetic.scan_cond
        out_h5["SingularSurfaces/Kinetic/scan_threshold"] = result.kinetic.scan_threshold

        # Write free-boundary stability data. The eigenmode energies are the generalized
        # eigenvalues of the pencil (W, N) with N the power-normalization (surface-norm) matrix:
        # power-normalized mode energies (⟨|ξ|²⟩ = 1 metric) that are invariant to the choice of
        # working (Jacobian) coordinate. W_freeboundary_eigenmodes holds the generalized
        # eigenvectors, columns sorted most-unstable first, normalized to unit power norm with
        # the largest-magnitude entry made real-positive.
        fbs = "ForceFreeStates/FreeBoundaryStability"
        out_h5["$fbs/W_freeboundary"] = free_energies !== nothing ? free_energies.wt0 : ComplexF64[]
        out_h5["$fbs/W_plasma"] = free_energies !== nothing ? free_energies.wp : ComplexF64[]
        out_h5["$fbs/W_vacuum"] = free_energies !== nothing ? free_energies.wv : ComplexF64[]
        out_h5["$fbs/W_freeboundary_eigenmodes"] = free_energies !== nothing ? free_energies.wt : ComplexF64[]
        out_h5["$fbs/eigenmode_energies"] = free_energies !== nothing ? free_energies.et : ComplexF64[]
        out_h5["$fbs/eigenmode_plasma_energies"] = free_energies !== nothing ? free_energies.ep : ComplexF64[]
        out_h5["$fbs/eigenmode_vacuum_energies"] = free_energies !== nothing ? free_energies.ev : ComplexF64[]
        out_h5["$fbs/vacuum_eigenvalue"] = free_energies !== nothing ? free_energies.vacuum_eigenvalue : NaN

        # Cartesian surface point clouds used downstream for visualisation and
        # perturbed-equilibrium plotting.
        out_h5["SurfaceGeometries/Plasma/x"] = free_energies !== nothing ? free_energies.plasma_pts[:, 1] : Float64[]
        out_h5["SurfaceGeometries/Plasma/y"] = free_energies !== nothing ? free_energies.plasma_pts[:, 2] : Float64[]
        out_h5["SurfaceGeometries/Plasma/z"] = free_energies !== nothing ? free_energies.plasma_pts[:, 3] : Float64[]
        out_h5["SurfaceGeometries/Wall/x"] = free_energies !== nothing ? free_energies.wall_pts[:, 1] : Float64[]
        out_h5["SurfaceGeometries/Wall/y"] = free_energies !== nothing ? free_energies.wall_pts[:, 2] : Float64[]
        out_h5["SurfaceGeometries/Wall/z"] = free_energies !== nothing ? free_energies.wall_pts[:, 3] : Float64[]

        # Write fundamental matrices on the ψ grid
        xs = equil.rzphi_xs
        npsi = length(xs)
        np = result.numpert_total

        # Helper: evaluate a matrix spline on the psi grid → (npsi, np, np) array
        function _eval_mat_spline(spline)
            arr = zeros(ComplexF64, npsi, np, np)
            hint = Ref(1)
            for i in 1:npsi
                arr[i, :, :] .= reshape(spline(xs[i]; hint=hint), np, np)
            end
            return arr
        end

        elm = "ForceFreeStates/EulerLagrangeMatrices"
        out_h5["$elm/psi"] = xs
        # Ideal primitive matrices (A, B, C, D, E, H)
        out_h5["$elm/Ideal/A"] = _eval_mat_spline(mats.ideal.A_spline)
        out_h5["$elm/Ideal/B"] = _eval_mat_spline(mats.ideal.B_spline)
        out_h5["$elm/Ideal/C"] = _eval_mat_spline(mats.ideal.C_spline)
        out_h5["$elm/Ideal/D"] = _eval_mat_spline(mats.ideal.D_spline_prim)
        out_h5["$elm/Ideal/E"] = _eval_mat_spline(mats.ideal.E_spline_prim)
        out_h5["$elm/Ideal/H"] = _eval_mat_spline(mats.ideal.H_spline)

        # Ideal derived matrices (F, K, G)
        out_h5["$elm/Ideal/F"] = _eval_mat_spline(mats.ideal.F_spline_lower)
        out_h5["$elm/Ideal/K"] = _eval_mat_spline(mats.ideal.K_spline)
        out_h5["$elm/Ideal/G"] = _eval_mat_spline(mats.ideal.G_spline)

        # Kinetic-modified matrices
        kin = mats.kinetic
        if kin !== nothing
            out_h5["$elm/Kinetic/A"] = _eval_mat_spline(kin.A_spline)
            out_h5["$elm/Kinetic/B"] = _eval_mat_spline(kin.B_spline)
            out_h5["$elm/Kinetic/C"] = _eval_mat_spline(kin.C_spline)
            out_h5["$elm/Kinetic/f0"] = _eval_mat_spline(kin.F0_spline)
            out_h5["$elm/Kinetic/K"] = _eval_mat_spline(kin.Kk_spline)
            out_h5["$elm/Kinetic/G"] = _eval_mat_spline(kin.G_spline_adj)
        end

        # Self-describing metadata pass (long_name/units/dims + dimension scales).
        apply_main_h5_metadata!(out_h5, result)
    end
end

"""
    _write_coil_snapshot!(h5_path::String, coil_sets::Vector{CoilSet})

Append the coil geometry used by a run into `Input/RawInputs/Coils/` of an existing
gpec.h5 file (opened in append mode), so the run can be replayed from the output alone.
One subgroup per coil set; see `ForcingTerms.save_coils_to_h5`.
"""
function _write_coil_snapshot!(h5_path::String, coil_sets::Vector{ForcingTerms.CoilSet})
    isfile(h5_path) || return nothing
    h5open(h5_path, "r+") do out_h5
        haskey(out_h5, "Input/RawInputs/Coils") && return nothing
        ForcingTerms.save_coils_to_h5(coil_sets, create_group(out_h5, "Input/RawInputs/Coils"))
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
export solve, perturbed_equilibrium
export PlasmaEquilibrium, EulerLagrangeProblem, Forward, Riccati, Galerkin, ResistiveMatch, ForceFreeStatesResult, RMPField

end # module GeneralizedPerturbedEquilibrium
