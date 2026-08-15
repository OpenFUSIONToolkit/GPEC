# GeneralizedPerturbedEquilibrium.jl
module GeneralizedPerturbedEquilibrium

# External dependencies used by main and the rerun helpers
using TOML
using Printf
using HDF5
using FastInterpolations
import IMASdd
import AdaptiveArrayPools: @with_pool

const _BANNER = "="^60
const _SECTION = "-"^40

include("Utilities/Utilities.jl")
import .Utilities as Utilities
export Utilities

include("Equilibrium/Equilibrium.jl")
import .Equilibrium as Equilibrium
export Equilibrium

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
using .ForceFreeStates: ForceFreeStatesInternal, ForceFreeStatesControl, DebugSettings, FreeBoundaryResult, OdeState, FourFitVars
using .ForceFreeStates: sing_lim!, sing_min!, sing_find!, resist_eval_all!, resist_geometry, ResistGeometry
using .ForceFreeStates: compute_local_stability, compute_ballooning_stability!, ballooning_alpha_boundary, ballooning_alpha_boundaries
using .ForceFreeStates: make_metric, make_matrix, make_kinetic_matrix
using .ForceFreeStates: find_kinetic_singular_surfaces!
using .ForceFreeStates: eulerlagrange_integration, free_run, normalize_eigenfunctions!
using .ForceFreeStates: galerkin_solve, write_galerkin!, GalerkinResult, gal_matched_odestate

const _DEPRECATED_FFS_KEYS = ("mer_flag", "force_wv_symmetry", "ode_flag", "cyl_flag", "mat_flag")
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
    # Per-stage wall-clock seconds, written to Info/Runtimes at the end of the run.
    runtimes = Vector{Pair{String,Float64}}()

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
    nstring = intr.npert == 1 ? "$(intr.nlow)" : "$(intr.nlow):$(intr.nhigh)"

    equil = Equilibrium.setup_equilibrium(eq_config, additional_input)

    # Build KineticForces control and load kinetic profiles once — reused by the grid
    # refinement below, the stability kinetic callback (via `calculated_cb`), and the
    # post-PE torque diagnostics block. The `"fixed"` kinetic source path in stability
    # does not need kinetic_profiles, but the post-PE block always does, so we load
    # whenever a [KineticForces] section is present or the stability path requests the
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

    # Two-pass auto grid: measure the pass-1 equilibrium's curvature (profiles, geometry,
    # kinetic profiles), pin knots on rational surfaces, and re-form on the refined grid
    # from the in-memory input — no file re-read.
    if Equilibrium.wants_two_pass(eq_config)
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
        @info "Two-pass psi grid: $(length(psi_nodes)) knots, $(length(mandatory)) rational surfaces pinned (n=$nstring)"
    end

    equil_dt = time() - equil_start
    push!(runtimes, "equilibrium" => equil_dt)
    @info "Equilibrium construction completed in $(@sprintf("%.3f", equil_dt)) s"

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

    # Forcing-data snapshot: when PerturbedEquilibrium is enabled, load forcing
    # modes early so they can be written into `Input/RawInputs/ForcingTerms/`
    # alongside the TOML blob. On the rerun path the caller passes the modes in
    # directly via `preloaded_forcing_modes`, bypassing the original file. Coil
    # forcing is recomputed from the `[[ForcingTerms.coil_set]]` TOML blob on
    # replay, so only the file-based formats need their modes captured here.
    forcing_modes_snapshot = preloaded_forcing_modes
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

    # ----------------------------------------------------------------
    # Force-Free States
    # ----------------------------------------------------------------
    @info "\n  Force-Free States\n$_SECTION"
    ffs_start = time()

    # Determine psilim and qlim (where we will integrate to)
    sing_lim!(intr, ctrl, equil)

    # If truncating before psihigh, reform equilibrium if desired
    if intr.psilim != equil.config.psihigh && ctrl.reform_eq_with_psilim
        @warn "Reforming equilibrium splines from psihigh to psilim not implemented yet. Proceeding with psihigh = $(equil.config.psihigh)."
        # JMH - Nik please put the logic we discussed here
        # something like ?
        # equil.config.psihigh = intr.psilim
        # equil = set_up_equilibrium(equil.config)
    end

    # Compute local stability (if desired). `locstab` holds `D_I` from the ballooning
    # coefficient system and the local ballooning result; `nothing` when not computed.
    locstab = nothing
    ballooning_boundary = (psi=Float64[], alpha=Float64[], alpha_critical=Float64[])
    if ctrl.local_stability_flag
        locstab = compute_local_stability(ctrl, equil)
        # First ballooning stability boundary (α vs ψ_N) for BALOO-style diagnostics.
        ballooning_boundary = ballooning_alpha_boundary(ctrl, equil)
    end

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
    if ctrl.gal_flag
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

    # Compute matrices and populate FourFitVars struct
    ffit = make_matrix(equil, intr, metric)

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
        make_kinetic_matrix(ctrl, equil, ffit, intr, metric;
            calculated_source=calculated_cb)

        # Find kinetically-displaced singular surfaces (zeros of det(F̄)) for ODE crossings.
        # Matches Fortran ksing_find (sing.f:1486-1616). singfac_min > 0 gates crossings;
        # singfac_min == 0 preserves single-chunk behavior.
        if ctrl.singfac_min > 0
            find_kinetic_singular_surfaces!(ffit, equil, intr)
        end
    end

    # Integrate Euler-Lagrange Equation
    if ctrl.verbose
        @info "Integrating Euler-Lagrange equation"
    end
    odet, fm_propagators, fm_chunks, fm_S_left = eulerlagrange_integration(ctrl, equil, ffit, intr)
    if odet.nzero > 0 && ctrl.verbose
        @warn "Fixed-boundary mode unstable for n = $nstring"
    end

    # Compute free boundary energies.
    free_energies = nothing
    if ctrl.vac_flag && !(ctrl.ksing > 0 && ctrl.ksing <= intr.msing + 1)
        if ctrl.verbose
            wall_desc = intr.wall_settings.shape == "nowall" ? "no wall" : intr.wall_settings.shape
            @info "Computing free boundary energies ($wall_desc)"
        end
        free_energies = free_run(odet, ctrl, equil, ffit, intr)
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
                ctrl=ctrl, equil=equil, ffit=ffit)
        end
    end

    # Outer-region resistive Δ′ matrix via the singular Galerkin method (RDCON gal_solve)
    gal_data = nothing
    if ctrl.gal_flag
        gal_start = time()
        gal_data = galerkin_solve(ctrl, equil, ffit, intr; wv=free_energies !== nothing ? free_energies.wv : nothing)
        gal_dt = time() - gal_start
        push!(runtimes, "galerkin" => gal_dt)
        @info "Galerkin solve completed in $(@sprintf("%.3f", gal_dt)) s"
    end

    if ctrl.write_outputs_to_HDF5
        write_outputs_to_HDF5(
            ctrl,
            equil,
            intr,
            odet,
            free_energies,
            ffit,
            git_version,
            inputs,
            forcing_modes_snapshot,
            gal_data;
            locstab=locstab,
            ballooning_boundary=ballooning_boundary
        )
        @info "Results written to $(ctrl.HDF5_filename)"
    end

    ffs_dt = time() - ffs_start
    push!(runtimes, "force_free_states" => ffs_dt)
    @info "Force-Free States completed in $(@sprintf("%.3f", ffs_dt)) s"

    # SLAYER tearing-mode analysis stage. Needs only equil + intr, so it runs in
    # both the force_termination=true path and the full pipeline. `pe_file` is the
    # HDF5 file PE wrote (to append into), or `nothing` if PE did not run.
    function _run_slayer_stage(pe_file::Union{String,Nothing})
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
            result = Runner.run_slayer(equil, intr, slayer_ctrl;
                dir_path=intr.dir_path)
            slayer_dt = time() - slayer_start
            push!(runtimes, "slayer" => slayer_dt)
            @info "SLAYER completed in $(@sprintf("%.3f", slayer_dt)) s"
            h5_filename = pe_file === nothing ? ctrl.HDF5_filename : pe_file
            h5_path = joinpath(intr.dir_path, h5_filename)
            # Append the Tearing/ group; create the file if no prior stage wrote
            # it (e.g. write_outputs_to_HDF5 disabled) rather than failing on "r+".
            HDF5.h5open(h5_path, isfile(h5_path) ? "r+" : "w") do f
                Runner.write_slayer_hdf5!(f, result)
            end
            @info "SLAYER results written to $h5_filename"
            return result
        catch err
            @error "SLAYER stage failed; continuing without tearing results. " *
                   "Equilibrium / stability / PE outputs are unaffected." exception =
                (err, catch_backtrace())
            return nothing
        end
    end

    # Early exit if user only requested force-free states (SLAYER still runs).
    if ctrl.force_termination
        slayer_result = _run_slayer_stage(nothing)
        total_dt = time() - total_start
        push!(runtimes, "total" => total_dt)
        # A returned SLAYER result means that stage created or appended to the file itself.
        if ctrl.write_outputs_to_HDF5 || slayer_result !== nothing
            _write_runtimes!(joinpath(intr.dir_path, ctrl.HDF5_filename), runtimes)
        end
        @info "\n$_BANNER\n  GPEC completed successfully in $(@sprintf("%.3f", total_dt)) s\n$_BANNER"
        return (ctrl=ctrl, equil=equil, intr=intr, ffit=ffit, odet=odet,
            free_energies=free_energies,
            slayer=slayer_result)
    end

    # ----------------------------------------------------------------
    # Perturbed Equilibrium
    # ----------------------------------------------------------------
    @info "\n  Perturbed Equilibrium\n$_SECTION"
    pe_start = time()

    # Check for PerturbedEquilibrium section and run if present
    if "PerturbedEquilibrium" in keys(inputs)
        # Read ForcingTerms control parameters
        if "ForcingTerms" in keys(inputs)
            forcing_raw = inputs["ForcingTerms"]
            # [[ForcingTerms.coil_set]] becomes a Vector{Dict} — must be excluded from
            # kwarg splatting and handled separately via coil_sets_raw field
            coil_sets_raw = Vector{Dict{String,Any}}(get(forcing_raw, "coil_set", Dict{String,Any}[]))
            scalar_forcing = filter(p -> p.first != "coil_set", forcing_raw)
            ft_ctrl = ForcingTerms.ForcingTermsControl(;
                (Symbol(k) => v for (k, v) in scalar_forcing)...
            )
            ft_ctrl.coil_sets_raw = coil_sets_raw
        else
            ft_ctrl = ForcingTerms.ForcingTermsControl()  # Use defaults
        end

        pe_ctrl = PerturbedEquilibrium.PerturbedEquilibriumControl(;
            (Symbol(k) => v for (k, v) in inputs["PerturbedEquilibrium"])...
        )
        pe_intr = PerturbedEquilibrium.PerturbedEquilibriumInternal(; dir_path=intr.dir_path)

        # DRIVEN (RPEC): feed the coil-matched gal solution to PE instead of the shooting solution.
        # The matched OdeState is in the identity-at-edge basis; build_flux_matrix rederives the edge BC
        # from u_store[:,:,1,step], so PE consumes it unchanged. The shooting odet is left untouched for
        # the Force-Free States HDF5 output.
        pe_odet = odet
        if ctrl.gal_flag && ctrl.gal_match_flag && gal_data !== nothing && gal_data.match !== nothing
            @info "PerturbedEquilibrium: using the RPEC-matched gal solution"
            pe_odet = gal_matched_odestate(gal_data, ffit, intr)
            pe_intr.odet_from_gal = true
            pe_intr.inner_bpen = gal_data.match.bpen
        else
            pe_intr.inner_bpen = zeros(ComplexF64, intr.msing, intr.numpert_total)
        end

        # Reuse the forcing modes loaded at snapshot time (or injected by
        # `build_inputs_from_h5`) so the PE compute step never re-reads the original
        # forcing file. `compute_perturbed_equilibrium` short-circuits
        # `load_forcing_data!` when `pe_intr.forcing_modes` is non-empty.
        if forcing_modes_snapshot !== nothing
            pe_intr.forcing_modes = copy(forcing_modes_snapshot)
        end

        # Inject preloaded coil geometry (gpec.h5 replay with `--coil-source coils`)
        # so the coil field is recomputed from stored geometry without the .dat/.h5 file.
        if preloaded_coil_sets !== nothing
            pe_intr.coil_sets = copy(preloaded_coil_sets)
        end

        # Run perturbed equilibrium calculations
        # Free-boundary wt0 drives the plasma inductance; mthvac sizes the Green's-function solves
        pe_state = PerturbedEquilibrium.compute_perturbed_equilibrium(
            equil, pe_odet, free_energies !== nothing ? free_energies.wt0 : nothing, ctrl.mthvac, intr, ft_ctrl, pe_ctrl, pe_intr,
            metric, ffit
        )

        # Write perturbed equilibrium outputs to same HDF5 file
        if pe_ctrl.write_outputs_to_HDF5
            output_file = isempty(pe_ctrl.output_filename) ? ctrl.HDF5_filename : pe_ctrl.output_filename
            PerturbedEquilibrium.write_outputs_to_HDF5(
                pe_state, pe_intr, joinpath(intr.dir_path, output_file)
            )
            @info "Results written to $output_file"
        end

        # Snapshot the coil geometry actually used into the gpec.h5 output so the run
        # is replayable from the output file alone (see `main_from_h5 --coil-source coils`).
        if ctrl.write_outputs_to_HDF5 && !isempty(pe_intr.coil_sets)
            _write_coil_snapshot!(joinpath(intr.dir_path, ctrl.HDF5_filename), pe_intr.coil_sets)
        end
    end

    pe_dt = time() - pe_start
    push!(runtimes, "perturbed_equilibrium" => pe_dt)
    @info "Perturbed Equilibrium completed in $(@sprintf("%.3f", pe_dt)) s"

    # ----------------------------------------------------------------
    # KineticForces (Neoclassical Toroidal Viscosity)
    # ----------------------------------------------------------------
    if "KineticForces" in keys(inputs)
        @info "\n  KineticForces\n$_SECTION"
        kf_start = time()

        # Standalone NTV torque diagnostics need a PE state (they contract kinetic operators
        # against ξ). The self-consistent kinetic_source="calculated" path produces none — skip.
        if !@isdefined(pe_state)
            @info "Skipping NTV torque diagnostics: no perturbed-equilibrium data (e.g. kinetic_source=\"calculated\")."
        else
            # kf_ctrl and kinetic_profiles were loaded once above the stability block.
            kf_intr = KineticForces.KineticForcesInternal(equil; verbose=kf_ctrl.verbose)
            KineticForces.set_perturbation_data!(kf_intr, pe_state, intr, equil, metric)

            kf_state = KineticForces.KineticForcesState()
            KineticForces.compute_torque_all_methods!(kf_state, kf_intr, kf_ctrl, equil, kinetic_profiles)

            if kf_ctrl.write_outputs_to_HDF5
                h5open(joinpath(intr.dir_path, kf_ctrl.HDF5_filename), "cw") do h5file
                    KineticForces.write_to_hdf5!(h5file, kf_state; dVdpsi_spline=equil.profiles.dVdpsi_spline)
                end
            end
        end

        kf_dt = time() - kf_start
        push!(runtimes, "kinetic_forces" => kf_dt)
        @info "KineticForces completed in $(@sprintf("%.3f", kf_dt)) s"
    end

    # ----------------------------------------------------------------
    # SLAYER tearing-mode analysis (after PE so it appends to the PE output
    # file; falls back to the ForceFreeStates file when PE did not run).
    # ----------------------------------------------------------------
    pe_file = if "PerturbedEquilibrium" in keys(inputs)
        pe_out = get(inputs["PerturbedEquilibrium"], "output_filename", "")
        isempty(pe_out) ? ctrl.HDF5_filename : pe_out
    else
        ctrl.HDF5_filename
    end
    slayer_result = _run_slayer_stage(pe_file)

    # ----------------------------------------------------------------
    # Done
    # ----------------------------------------------------------------
    total_dt = time() - total_start
    push!(runtimes, "total" => total_dt)
    # A returned SLAYER result means that stage created or appended to the file itself.
    if ctrl.write_outputs_to_HDF5 || slayer_result !== nothing
        _write_runtimes!(joinpath(intr.dir_path, ctrl.HDF5_filename), runtimes)
    end
    @info "\n$_BANNER\n  GPEC completed successfully in $(@sprintf("%.3f", total_dt)) s\n$_BANNER"

    # TODO: Do not allow perturbed equilibrium calculations if zero crossings are found

    return (ctrl=ctrl, equil=equil, intr=intr, ffit=ffit, odet=odet,
        free_energies=free_energies,
        slayer=slayer_result)

end

"""
    write_outputs_to_HDF5(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal, odet::OdeState)

Helper function to write the HDF5 output file with relevant run and equilibrium parameters.
This combines the functionality of several pieces of the Fortran code in `ode_output.f`,
primarily `ode_output_open` and the various `bin_euler` writes that occur throughout the
integration. Some parameters are only dumped in their respective flags are true, e.g.
vacuum data if `vac_flag` is true.

### TODOs

Combine spline unpacking if possible, too many extra lines
"""
function write_outputs_to_HDF5(
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    intr::ForceFreeStatesInternal,
    odet::OdeState,
    free_energies::Union{FreeBoundaryResult,Nothing},
    ffit::Union{FourFitVars,Nothing}=nothing,
    git_version::String="unknown",
    inputs::Union{Nothing,Dict{String,Any}}=nothing,
    forcing_modes::Union{Nothing,Vector{ForcingTerms.ForcingMode}}=nothing,
    gal_data::Union{GalerkinResult,Nothing}=nothing;
    locstab::Union{FastInterpolations.CubicSeriesInterpolant,Nothing}=nothing,
    ballooning_boundary=(psi=Float64[], alpha=Float64[], alpha_critical=Float64[])
)

    # Idempotent: already done if a PerturbedEquilibrium stage ran. Leaves the stores empty
    # (and the datasets below empty) on paths whose solution basis cannot supply them.
    ForceFreeStates.materialize_derivative_stores!(odet, equil, ffit, intr)

    h5open(joinpath(intr.dir_path, ctrl.HDF5_filename), "w") do out_h5

        # File-level metadata contract (schema_version, Conventions, title, date).
        Utilities.HDF5Annotations.write_root_attrs!(out_h5; title="GPEC output: $(basename(abspath(intr.dir_path)))")

        # Store git version for reproducibility
        out_h5["Info/git_version"] = git_version

        # Outer-region Galerkin Δ′ matrix (RDCON), if computed
        if gal_data !== nothing
            write_galerkin!(out_h5, gal_data)
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
        out_h5["Info/mpert"] = intr.mpert
        out_h5["Info/mlow"] = intr.mlow
        out_h5["Info/mhigh"] = intr.mhigh
        out_h5["Info/npert"] = intr.npert
        out_h5["Info/nlow"] = intr.nlow
        out_h5["Info/nhigh"] = intr.nhigh
        m = [(i - 1) % intr.mpert + intr.mlow for i in 1:(intr.numpert_total)]
        n = [(i - 1) ÷ intr.mpert + intr.nlow for i in 1:(intr.numpert_total)]
        out_h5["Info/mn_index"] = hcat(m, n)   # (N, 2) matrix
        out_h5["Info/psilim"] = intr.psilim
        out_h5["Info/qlim"] = intr.qlim
        out_h5["Info/dqdpsi_lim"] = intr.q1lim

        # Write derived equilibrium parameters
        for (key, val) in zip(fieldnames(Equilibrium.EquilibriumParameters), getfield.(Ref(equil.params), fieldnames(Equilibrium.EquilibriumParameters)))
            if val !== nothing # TODO: looks like ro, zo, psio, and b_norm are not set, so skipping those for now but should fix eventually
                out_h5["Equilibrium/$key"] = val
            end
        end
        out_h5["Equilibrium/psio"] = equil.psio
        out_h5["Equilibrium/ro"] = equil.ro
        out_h5["Equilibrium/zo"] = equil.zo

        # Write equilibrium profile and geometry arrays (from the named splines)
        profiles = equil.profiles
        out_h5["Equilibrium/Profiles/xs"] = profiles.xs
        out_h5["Equilibrium/Profiles/2piF"] = profiles.F_spline.y
        out_h5["Equilibrium/Profiles/mu0p"] = profiles.P_spline.y
        out_h5["Equilibrium/Profiles/dVdpsi"] = profiles.dVdpsi_spline.y
        out_h5["Equilibrium/Profiles/q"] = profiles.q_spline.y
        out_h5["Equilibrium/Geometry/xs"] = equil.rzphi_xs
        out_h5["Equilibrium/Geometry/ys"] = equil.rzphi_ys
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
            out_h5["LocalStability/D_I"] = locstab.y[:, 1] ./ locstab_xs
            out_h5["LocalStability/D_R"] = locstab.y[:, 2] ./ locstab_xs
        else
            out_h5["LocalStability/D_I"] = Float64[]
            out_h5["LocalStability/D_R"] = Float64[]
        end
        out_h5["SingularSurfaces/D_I"] = (locstab !== nothing && !isempty(intr.sing)) ?
                                         [locstab(sing.psifac)[1] / sing.psifac for sing in intr.sing] : Float64[]
        out_h5["LocalStability/ballooning_Delta_prime"] = locstab !== nothing ? locstab.y[:, 4] : Float64[]

        # First ballooning stability boundary: experimental α vs critical α (BALOO-style).
        out_h5["LocalStability/psi"] = ballooning_boundary.psi
        out_h5["LocalStability/alpha"] = ballooning_boundary.alpha
        out_h5["LocalStability/alpha_critical"] = ballooning_boundary.alpha_critical

        # Write integration data
        fwd = "ForceFreeStates/Solutions/ForwardIntegration"
        out_h5["$fwd/nstep"] = odet.step            # Number of saved solution snapshots
        out_h5["$fwd/nstep_total"] = odet.total_steps  # Total ODE solver steps taken
        out_h5["$fwd/psi"] = odet.psi_store
        out_h5["$fwd/q"] = odet.q_store
        out_h5["$fwd/xi_psi"] = odet.u_store[:, :, 1, :]
        out_h5["$fwd/u2"] = odet.u_store[:, :, 2, :] # TODO: what to name this? These are the "conjugate momenta" of u1
        out_h5["$fwd/dxi_psidpsi"] = odet.du_store
        out_h5["$fwd/xi_s"] = odet.xi_s_store
        out_h5["$fwd/crit"] = odet.crit_store

        # Write edge stability scan data (only present when psiedge < psilim).
        # Generalized (W, N) pencil energies — power-normalized, Jacobian-invariant; these are
        # the values findmax_dW_edge! uses to choose the truncation point.
        if !isempty(odet.edge_scan.psi)
            es = odet.edge_scan
            out_h5["ForceFreeStates/EdgeScan/psi"] = es.psi
            out_h5["ForceFreeStates/EdgeScan/q"] = es.q
            out_h5["ForceFreeStates/EdgeScan/total_energy"] = es.total_eigenvalue
            out_h5["ForceFreeStates/EdgeScan/plasma_energy"] = es.plasma_energy
            out_h5["ForceFreeStates/EdgeScan/vacuum_energy"] = es.vacuum_energy
            out_h5["ForceFreeStates/EdgeScan/vacuum_eigenvalue"] = es.vacuum_eigenvalue
        end

        # Write singular surface data
        out_h5["SingularSurfaces/rational_count"] = intr.msing
        out_h5["SingularSurfaces/rational_psi"] = [sing.psifac for sing in intr.sing]
        out_h5["SingularSurfaces/rational_q"] = [sing.q for sing in intr.sing]
        out_h5["SingularSurfaces/dqdpsi"] = [sing.q1 for sing in intr.sing]
        out_h5["SingularSurfaces/ca_left"] = odet.ca_l
        out_h5["SingularSurfaces/ca_right"] = odet.ca_r

        if intr.msing > 0
            # Mode numbers at each surface (jagged — pad with 0 to max_modes width)
            max_modes = maximum(s -> length(s.m), intr.sing)
            m_matrix = zeros(Int, intr.msing, max_modes)
            n_matrix = zeros(Int, intr.msing, max_modes)
            for (s, sing) in enumerate(intr.sing)
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
            if all(s -> s.restype !== nothing, intr.sing)
                out_h5["SingularSurfaces/E"] = [s.restype.E for s in intr.sing]
                out_h5["SingularSurfaces/F"] = [s.restype.F for s in intr.sing]
                out_h5["SingularSurfaces/G"] = [s.restype.G for s in intr.sing]
                out_h5["SingularSurfaces/H"] = [s.restype.H for s in intr.sing]
                out_h5["SingularSurfaces/K"] = [s.restype.K for s in intr.sing]
                out_h5["SingularSurfaces/M"] = [s.restype.M for s in intr.sing]
                out_h5["SingularSurfaces/avg_bsq_over_dpsisq"] = [s.restype.avg_bsq_over_dpsisq for s in intr.sing]
                out_h5["SingularSurfaces/avg_bsq"] = [s.restype.avg_bsq for s in intr.sing]
                out_h5["SingularSurfaces/mu0p"] = [s.restype.p_local for s in intr.sing]
                out_h5["SingularSurfaces/dmu0pdpsi"] = [s.restype.p1_local for s in intr.sing]
                out_h5["SingularSurfaces/dVdpsi"] = [s.restype.v1_local for s in intr.sing]
            end
        end

        # Per-surface ca-based Δ' (`sing.delta_prime`) is a stub; only the BVP matrix is emitted (see SingType.delta_prime docstring).

        # Write inter-surface Δ' matrix if computed (parallel FM path only).
        # Shape: [msing × msing] — PEST3-convention deltap (STRIDE BVP with vacuum coupling).
        if intr.msing > 0 && !isempty(intr.delta_prime_matrix)
            out_h5["SingularSurfaces/Delta_prime_matrix"] = intr.delta_prime_matrix
        end

        # Edge coil-response matrix, stored (numpert_total × 2msing) = (edge mode, surface-side) to match
        # the SingularSurfaces/GalerkinDeltaPrime/Delta_coil layout so H5Web heatmaps share axes
        # (x = edge mode, y = surface-side).
        # Internal intr.delta_coil stays (2msing × numpert_total); transpose only at write.
        if intr.msing > 0 && !isempty(intr.delta_coil)
            dc = permutedims(intr.delta_coil)
            out_h5["SingularSurfaces/Delta_coil"] = dc
        end

        # Write raw 2msing×2msing outer-region D' matrix in side-major ordering
        # [L_s1, R_s1, L_s2, R_s2, …]. Byte-compatible with Fortran
        # rdcon/gal.f::gal_write_delta top 2msing×2msing block of delta_gw.dat.
        # Needed for the full det(D' − D(γ)) = 0 eigenvalue problem via
        # pest3_decompose to recover (A', B', Γ', Δ').
        if intr.msing > 0 && !isempty(intr.delta_prime_raw)
            out_h5["SingularSurfaces/Delta_prime_raw"] = intr.delta_prime_raw
        end

        # Write kinetic singular surface data (det(F̄) near-zeros) and the cond(F̄) scan
        # used to find them. Populated only when kinetic crossings were searched for.
        out_h5["SingularSurfaces/Kinetic/rational_count"] = intr.kmsing
        out_h5["SingularSurfaces/Kinetic/rational_psi"] = [s.psifac for s in intr.kinsing]
        out_h5["SingularSurfaces/Kinetic/rational_q"] = [s.q for s in intr.kinsing]
        out_h5["SingularSurfaces/Kinetic/dqdpsi"] = [s.q1 for s in intr.kinsing]
        out_h5["SingularSurfaces/Kinetic/scan_psi"] = intr.kinsing_scan_psi
        out_h5["SingularSurfaces/Kinetic/scan_cond"] = intr.kinsing_scan_cond
        out_h5["SingularSurfaces/Kinetic/scan_threshold"] = intr.kinsing_scan_threshold

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
        if ffit !== nothing
            xs = equil.rzphi_xs
            npsi = length(xs)
            np = intr.numpert_total

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
            # When kinetic mode is on, amats/bmats/cmats hold kinetic-modified values,
            # so we write those as the "effective" matrices and save raw kinetic
            # components separately below.
            if ctrl.kinetic_factor > 0
                # Use preserved ideal copies (before kinetic overwrite)
                out_h5["$elm/Ideal/A"] = _eval_mat_spline(ffit.amats_ideal)
                out_h5["$elm/Ideal/B"] = _eval_mat_spline(ffit.bmats_ideal)
                out_h5["$elm/Ideal/C"] = _eval_mat_spline(ffit.cmats_ideal)
            else
                out_h5["$elm/Ideal/A"] = _eval_mat_spline(ffit.amats)
                out_h5["$elm/Ideal/B"] = _eval_mat_spline(ffit.bmats)
                out_h5["$elm/Ideal/C"] = _eval_mat_spline(ffit.cmats)
            end
            out_h5["$elm/Ideal/D"] = _eval_mat_spline(ffit.dmats_prim)
            out_h5["$elm/Ideal/E"] = _eval_mat_spline(ffit.emats_prim)
            out_h5["$elm/Ideal/H"] = _eval_mat_spline(ffit.hmats)

            # Ideal derived matrices (F, K, G)
            out_h5["$elm/Ideal/F"] = _eval_mat_spline(ffit.fmats_lower)
            out_h5["$elm/Ideal/K"] = _eval_mat_spline(ffit.kmats)
            out_h5["$elm/Ideal/G"] = _eval_mat_spline(ffit.gmats)

            # Kinetic-modified matrices
            if ctrl.kinetic_factor > 0
                out_h5["$elm/Kinetic/A"] = _eval_mat_spline(ffit.amats)
                out_h5["$elm/Kinetic/B"] = _eval_mat_spline(ffit.bmats)
                out_h5["$elm/Kinetic/C"] = _eval_mat_spline(ffit.cmats)
                out_h5["$elm/Kinetic/f0"] = _eval_mat_spline(ffit.f0mats)
                out_h5["$elm/Kinetic/K"] = _eval_mat_spline(ffit.kkmats)
                out_h5["$elm/Kinetic/G"] = _eval_mat_spline(ffit.gaats)
            end
        end

        # Self-describing metadata pass (long_name/units/dims + dimension scales).
        apply_main_h5_metadata!(out_h5, intr)
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
        Utilities.HDF5Annotations.annotate!(out_h5, RUNTIME_H5_ANNOTATIONS)
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

The `result` argument is the named tuple returned by `main`.
"""
function write_imas(dd, result)
    result.free_energies === nothing && return

    free_energies = result.free_energies
    intr = result.intr

    # Top-level metadata
    dd.mhd_linear.code.name = "GPEC"
    dd.mhd_linear.ideal_flag = 1

    # Add a time_slice at the current global_time (wipe=false reuses an existing slice
    # at the same time, or appends a new one if none exists yet)
    ts = resize!(dd.mhd_linear.time_slice; wipe=false)

    # Write the least-stable energy for each toroidal mode number
    # n_tor_idx[i] (0-based) identifies which n-block eigenvalue i belongs to.
    resize!(ts.toroidal_mode, intr.npert)
    for j in 0:(intr.npert-1)
        n_indices = findall(==(j), free_energies.n_tor_idx) # indices of eigenvalues in the j-th n-block
        mode = ts.toroidal_mode[j+1]
        mode.n_tor = intr.nlow + j
        mode.energy_perturbed = minimum(real.(free_energies.et[n_indices])) # least-stable energy for this n-toroidal mode
    end

    return dd
end

export main, write_imas

end # module GeneralizedPerturbedEquilibrium
