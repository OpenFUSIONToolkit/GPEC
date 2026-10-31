# Deck.jl: the gpec.toml -> API translation. Every section goes through `section_keywords`, so a deck states exactly
# the keywords of the constructor it feeds and a stale or misspelled key stops the run.
#
# Included directly into the GeneralizedPerturbedEquilibrium module.

import .Utilities: deck_sections

const _DEPRECATED_FFS_KEYS = ("mer_flag", "force_wv_symmetry", "ode_flag", "cyl_flag", "mat_flag", "reform_eq_with_psilim",
    "use_riccati", "use_parallel", "parallel_threads", "populate_dense_xi",
    "gal_flag", "nstep", "diagnose_ca", "diagnose", "ksing", "local_stability_flag", "gal_solver", "gal_sing_order", "force_termination",
    "kinetic_source", "sing_start")
const _DEPRECATED_EQUIL_KEYS = ("power_bp", "power_b", "power_r", "power_rc", "use_galgrid", "force_termination")
const _DEPRECATED_KF_KEYS = ("moment", "eq_out", "theta_out", "xlmda_out", "eqpsi_out", "save_records", "gpec_file",
    "fnml_flag", "ellip_flag", "diag_flag", "force_xialpha")
const _DEPRECATED_PE_KEYS = ("fixed_boundary", "output_eigenmodes", "filter_modes", "singular_point_method", "compute_response")
const _DEPRECATED_KEYS = ("Equilibrium" => _DEPRECATED_EQUIL_KEYS, "ForceFreeStates" => _DEPRECATED_FFS_KEYS,
    "KineticForces" => _DEPRECATED_KF_KEYS, "PerturbedEquilibrium" => _DEPRECATED_PE_KEYS)
# Per-section output and verbosity keys, replaced by the run-wide [Run] output_file and verbose.
const _RUN_SECTION_KEYS = ("verbose", "HDF5_filename", "output_filename", "write_outputs_to_HDF5")
const _RUN_KEY_SECTIONS = ("Equilibrium", "LocalStability", "ForceFreeStates", "PerturbedEquilibrium", "KineticForces", "ErrorFields")

# Keys that moved to another section, each with where it lives now.
const _ODE_DECK_KEYS = ("eulerlagrange_tolerance", "renorm_threshold", "numsteps_init", "numunorms_init", "save_interval", "nchunks")
const _SEARCH_DECK_KEYS = ("scan_mode", "Q_re_range", "Q_im_range", "nre", "nim", "amr_passes", "amr_max_cells", "boxes",
    "multi_box_prescreen_n", "pole_threshold", "pole_threshold_adaptive", "filter_above_poles", "filter_outside_re",
    "gamma_gap_threshold", "polish_roots", "validity_rtol", "root_classification", "store_scan")
const _KINETIC_FILE_HOME = "name the kinetic-profile file once, as kinetic_file (and kinetic_group) in [Equilibrium]"
const _MOVED_KEYS = Dict{String,Dict{String,String}}(
    "ForceFreeStates" => Dict{String,String}(
        "nn_low" => "set nn = n, or nn = [n1, n2, ...] for consecutive toroidal modes",
        "nn_high" => "set nn = n, or nn = [n1, n2, ...] for consecutive toroidal modes",
        "extended_precision_bvp" => "moved to [Riccati]",
        "gal_match_flag" => "add a [Match] section to match the inner layer",
        "gal_ideal_flag" => "moved to [Match] as ideal",
        "gal_inner_solver" => "moved to [GGJ] as solver",
        (k => "moved to the integrator's section, [Forward] or [Riccati]" for k in _ODE_DECK_KEYS)...,
        ("gal_$k" => "moved to [Match] as $k" for k in ("eta", "rho", "rotation", "gamma"))...,
        ("gal_inner_$k" => "moved to [GGJ] as $k" for k in ("xfac", "nx", "nq", "cutoff", "kmax"))...,
        ("gal_$k" => "moved to [Galerkin] as $k" for k in ("nx", "nq", "pfac", "dx0", "dx1", "dx2", "cutoff", "tol", "gnstep",
            "dx1dx2_flag", "rpec_flag", "edge_onesided"))...,
        "gal_sing_order_ceiling" => "moved to [Galerkin] as frobenius_order_ceiling"),
    "LocalStability" => Dict{String,String}(
        k => "moved to [Ballooning]" for k in ("ode_tol", "n_scan", "max_alpha_scale", "alpha_tol", "psi_window")),
    "KineticForces" => Dict{String,String}(
        "ion_species" => "moved to [[Equilibrium.ion_species]]",
        "ion_charge" => "state the main ions as [[Equilibrium.ion_species]] with z and m",
        "ion_mass" => "state the main ions as [[Equilibrium.ion_species]] with z and m",
        "impurity_charge" => "moved to [Equilibrium]",
        "impurity_mass" => "moved to [Equilibrium]",
        "nn" => "removed; KineticForces takes the toroidal mode number from the run",
        "kinetic_file" => _KINETIC_FILE_HOME),
    "SLAYER" => Dict{String,String}(
        "enabled" => "a [Tearing] section with inner_model = \"SLAYER\" runs the tearing analysis",
        "inner_model" => "moved to [Tearing]; the SLAYER variant is [SLAYER] variant",
        "coupling_mode" => "moved to [Tearing]", "msing_max" => "moved to [Tearing]", "omega_E_kHz" => "moved to [Tearing]",
        "scan_grid" => "its keys moved to [Tearing.search]",
        "CriticalResonantField" => "moved to [Tearing.CriticalResonantField]",
        "gap_kHz_threshold" => "set [Tearing.search] gamma_gap_threshold in 1/s",
        (k => "moved to [Tearing.search]" for k in _SEARCH_DECK_KEYS)...,
        "ion_mass" => "the plasma is the equilibrium's kinetic data: state the main ions as [[Equilibrium.ion_species]]",
        "zeff" => "moved to [Equilibrium] as zeff_default",
        "impurity_charge" => "moved to [Equilibrium]",
        "chi_perp_e" => "moved to [Equilibrium] as chi_e_default",
        "chi_perp_i" => "moved to [Equilibrium] as chi_i_default",
        "chi_tor" => "moved to [Equilibrium] as chi_phi_default",
        "tau_E" => "moved to [Equilibrium]",
        "resistivity_model" => "moved to [Equilibrium]",
        "lnLambda_form" => "moved to [Equilibrium]",
        "bt" => "removed; B_T comes from the equilibrium's F",
        "profile_file" => _KINETIC_FILE_HOME, "profile_group" => _KINETIC_FILE_HOME))
const _ANALYTIC_HOME = "; an analytic deck sets no [Equilibrium] eq_type"
const _RETIRED_SECTIONS = Dict("DEBUG" => "[DEBUG] is retired: output_benchmark_data was never read, and gal_basis_output is [Galerkin] basis_output",
    "SOL_INPUT" => "[SOL_INPUT] is now [Solovev]" * _ANALYTIC_HOME, "LAR_INPUT" => "[LAR_INPUT] is now [LargeAspectRatio]" * _ANALYTIC_HOME,
    "TJ_ANALYTIC_INPUT" => "[TJ_ANALYTIC_INPUT] is now [TJAnalytic], whose solver = \"direct\" replaces eq_type = \"tj_analytic_direct\"" * _ANALYTIC_HOME)

# Every section a deck may hold, besides the analytic-equilibrium sections.
const _DECK_SECTIONS = ("Run", "Equilibrium", "Wall", "LocalStability", "Ballooning", "ForceFreeStates", "Forward", "Riccati",
    "Galerkin", "Match", "GGJ", "Tearing", "SLAYER", "ForcingTerms", "PerturbedEquilibrium", "KineticForces", "ErrorFields")
const _INTEGRATORS = ("Forward" => Forward, "Riccati" => Riccati, "Galerkin" => Galerkin)

# Stop the run when a parsed gpec.toml section sets keys that no longer exist, so the deck is updated.
function _stop_on_deprecated_keys(table, deprecated_keys, section::String; remedy::String="remove from gpec.toml")
    found = [k for k in deprecated_keys if haskey(table, k)]
    isempty(found) && return nothing
    listed = join(("`$k`" for k in found), ", ")
    error(length(found) == 1 ? "[$section] sets $listed, which is no longer a key; $remedy" : "[$section] sets $listed, which are no longer keys; $remedy")
end

# Stop on every retired key or section of a parsed deck. `remedy` replaces the per-key advice, e.g. for a stored deck that cannot be edited.
function _stop_on_deprecated_deck_keys(inputs::AbstractDict; remedy::Union{Nothing,String}=nothing)
    for (section, message) in _RETIRED_SECTIONS
        haskey(inputs, section) && error(something(remedy, message))
    end
    for (section, keys) in _DEPRECATED_KEYS
        haskey(inputs, section) && _stop_on_deprecated_keys(inputs[section], keys, section; remedy=something(remedy, "remove from gpec.toml"))
    end
    run_remedy = something(remedy, "set [Run] output_file or verbose instead; a deck run always writes its output file")
    for section in _RUN_KEY_SECTIONS
        haskey(inputs, section) && _stop_on_deprecated_keys(inputs[section], _RUN_SECTION_KEYS, section; remedy=run_remedy)
    end
    ntv = get(get(inputs, "ErrorFields", Dict{String,Any}()), "NTV", nothing)
    ntv === nothing || _stop_on_deprecated_keys(ntv, ("verbose",), "ErrorFields.NTV"; remedy=run_remedy)
    for (section, moved) in _MOVED_KEYS
        table = get(inputs, section, nothing)
        table isa AbstractDict || continue
        found = sort!([k for k in keys(table) if haskey(moved, k)])
        isempty(found) && continue
        error("[$section] sets keys that moved: " * join(("`$k` ($(something(remedy, moved[k])))" for k in found), "; "))
    end
    return nothing
end

# A deck value as the constructor field type `T` takes it: strings become Symbols, arrays become tuples.
_deck_value(::Type, v) = v
_deck_value(::Type{Any}, v) = v
_deck_value(::Type{Symbol}, v::AbstractString) = Symbol(v)
_deck_value(::Type{Float64}, v::Integer) = Float64(v)
_deck_value(::Type{T}, v::AbstractVector) where {T<:Tuple} = T(Tuple(v))
_deck_value(::Type{Union{Nothing,T}}, v) where {T} = _deck_value(T, v)
_deck_value(::Type{Vector{T}}, v::AbstractVector) where {T} = convert(Vector{T}, [_deck_value(T, x) for x in v])

# The keyword fields of struct `T` with their types (a type parameter becomes `Any`), less `exclude`.
function _struct_fields(T::Type, exclude=())
    S = Base.unwrap_unionall(T)
    return Dict{Symbol,Any}(f => (t = fieldtype(S, f); t isa Type ? t : Any) for f in fieldnames(S) if !(f in exclude))
end

"""
    section_keywords(table, section, fields) -> Dict{Symbol,Any}

The constructor keywords a deck section states: Symbol keys, each value coerced to the type
`fields` gives its key (a string to a Symbol, an array to a tuple, an integer to a float). A key
`fields` does not list is an error naming the section and its allowed keys.
"""
function section_keywords(table::AbstractDict, section::AbstractString, fields::AbstractDict{Symbol})
    kwargs = Dict{Symbol,Any}()
    for (k, v) in table
        key = Symbol(k)
        haskey(fields, key) ||
            throw(ArgumentError("unknown key \"$k\" in [$section] (allowed: $(join(sort!(String.(collect(keys(fields)))), ", ")))"))
        kwargs[key] = _deck_value(fields[key], v)
    end
    return kwargs
end

# A section's table without its sub-tables `names`, which are parsed on their own.
_without(table::AbstractDict, names...) = filter(p -> !(p.first in names), table)

# A copy of a deck section with its relative path `keys` resolved against the deck directory `path`; absolute paths are kept.
_deck_paths(::Nothing, path, keys) = nothing
function _deck_paths(section::AbstractDict, path::AbstractString, keys)
    resolved = Dict{String,Any}(section)
    for k in keys
        v = get(resolved, k, "")
        isempty(v) || isabspath(v) || (resolved[k] = normpath(joinpath(path, v)))
    end
    return resolved
end

# The value a selector key names, checked against `choices`.
function _selector(table::AbstractDict, section::String, key::String, choices, default=nothing)
    name = get(table, key, default)
    name === nothing && error("[$section] needs $key = $(join(("\"$c\"" for c in choices), " | "))")
    name in choices || error("[$section] $key = \"$name\" is not one of $(join(("\"$c\"" for c in choices), ", "))")
    return name
end

"""
    run_settings(inputs, path) -> (; output_file, verbose)

The run-wide `[Run]` settings of a parsed deck: the one output file every stage saves into,
resolved against the deck directory `path` when relative, and the one verbosity of the run.
Defaults are `output_file = "gpec.h5"` and `verbose = false`.
"""
function run_settings(inputs::AbstractDict, path::AbstractString)
    kwargs = section_keywords(get(inputs, "Run", Dict{String,Any}()), "Run", Dict{Symbol,Any}(:output_file => String, :verbose => Bool))
    file = get(kwargs, :output_file, "gpec.h5")
    return (; output_file=isabspath(file) ? file : normpath(joinpath(path, file)), verbose=get(kwargs, :verbose, false))
end

# The eq_type => ANALYTIC_EQ row of the one analytic-equilibrium section a deck holds, `nothing` when it holds none.
function _analytic_row(inputs::AbstractDict)
    rows = [row for row in Equilibrium.ANALYTIC_EQ if haskey(inputs, row.second.section)]
    length(rows) > 1 && error("gpec.toml holds more than one analytic equilibrium: " * join(sort!(["[$(row.second.section)]" for row in rows]), ", "))
    return isempty(rows) ? nothing : only(rows)
end

"""
    analytic_config(inputs) -> Union{Nothing,Equilibrium.AnalyticEquilibriumConfig}

The analytic equilibrium a deck's `[Solovev]`, `[LargeAspectRatio]` or `[TJAnalytic]` section
states, `nothing` when the deck holds none.
"""
function analytic_config(inputs::AbstractDict)
    row = _analytic_row(inputs)
    row === nothing && return nothing
    spec = row.second
    return spec.config_type(; section_keywords(inputs[spec.section], spec.section, _struct_fields(spec.config_type))...)
end

"""
    equilibrium_config(inputs, path) -> Equilibrium.EquilibriumConfig

The `[Equilibrium]` section as an `EquilibriumConfig`, its file paths resolved against the deck
directory `path`. An analytic-equilibrium section implies `eq_type`; otherwise it is required.
"""
function equilibrium_config(inputs::AbstractDict, path::AbstractString)
    table = get(inputs, "Equilibrium", nothing)
    table === nothing && error("No [Equilibrium] section in gpec.toml")
    row = _analytic_row(inputs)
    eq_type = get(table, "eq_type", nothing)
    if row !== nothing
        eq_type === nothing || error("[Equilibrium] sets eq_type, but the [$(row.second.section)] section already selects the equilibrium; remove eq_type")
    elseif eq_type === nothing
        error("Missing required key in [Equilibrium]: eq_type")
    elseif haskey(Equilibrium.ANALYTIC_EQ, eq_type)
        error("[Equilibrium] eq_type = \"$eq_type\" is not a deck value: state the analytic equilibrium as a " *
              "[$(Equilibrium.ANALYTIC_EQ[eq_type].section)] section and remove eq_type")
    end
    fields = _struct_fields(Equilibrium.EquilibriumConfig, (:power_bp, :power_b, :power_r, :power_rc))
    kwargs = section_keywords(_deck_paths(table, path, ("eq_filename", "kinetic_file")), "Equilibrium", fields)
    row === nothing || (kwargs[:eq_type] = row.first)
    return Equilibrium.EquilibriumConfig(; kwargs...)
end

"""
    kinetic_forces_control(inputs) -> KineticForces.KineticForcesControl

The `[KineticForces]` control of the run, defaults when the deck has no such section.
"""
function kinetic_forces_control(inputs::AbstractDict)
    table = get(inputs, "KineticForces", Dict{String,Any}())
    return KineticForces.KineticForcesControl(; section_keywords(table, "KineticForces", _struct_fields(KineticForces.KineticForcesControl))...)
end

"""
    slayer_model(inputs) -> SLAYERModel

The SLAYER layer model of the `[SLAYER]` section, the default `SLAYERModel()` when absent.
"""
function slayer_model(inputs::AbstractDict)
    fields = Dict{Symbol,Any}(:variant => Symbol, :P_perp_model => Any, :D_c_type => Any, :D_R => Any, :D_geo => Any, :chi_perp_Dc => Any,
        :theta_sample => Any)
    return SLAYERModel(; section_keywords(get(inputs, "SLAYER", Dict{String,Any}()), "SLAYER", fields)...)
end

"""
    ggj_model(inputs) -> GGJModel

The GGJ layer model of the `[GGJ]` section: `solver` and the backend options, passed to
`GGJModel` as they are. The default `GGJModel()` when absent.
"""
function ggj_model(inputs::AbstractDict)
    kwargs = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in get(inputs, "GGJ", Dict{String,Any}()))
    haskey(kwargs, :solver) && (kwargs[:solver] = Symbol(kwargs[:solver]))
    return GGJModel(; kwargs...)
end

"""
    force_balance_keywords(inputs, path) -> Dict{Symbol,Any}

The `ForceBalanceProblem` keywords of the `[ForceFreeStates]` section, with `wall` from `[Wall]`,
the `kinetic` source it names and the layer-overlap cap's `[SLAYER]` closures.
"""
function force_balance_keywords(inputs::AbstractDict, path::AbstractString)
    fields = _struct_fields(ForceFreeStatesControl, (:algorithm, :nn_low, :nn_high, :kinetic, :psilim_from_layer_overlap))
    merge!(fields, Dict{Symbol,Any}(:nn => Any, :integrator => String, :kinetic => String, :psilim_from_layer_overlap => Bool))
    kwargs = section_keywords(inputs["ForceFreeStates"], "ForceFreeStates", fields)
    delete!(kwargs, :integrator)
    haskey(kwargs, :nn) || error("[ForceFreeStates] needs nn, the toroidal mode number (nn = 1) or consecutive numbers (nn = [1, 2, 3])")
    nn = kwargs[:nn]
    if nn isa AbstractVector
        (!isempty(nn) && all(x -> x isa Integer, nn) && nn == first(nn):last(nn)) ||
            error("[ForceFreeStates] nn = $nn must list consecutive toroidal mode numbers")
        kwargs[:nn] = first(nn):last(nn)
    end
    kinetic = pop!(kwargs, :kinetic, nothing)
    if kinetic == "KineticForces"
        kwargs[:kinetic] = kinetic_forces_control(inputs)
    elseif kinetic == "TestKineticMatrices"
        error("TestKineticMatrices is script-only: pass ForceBalanceProblem(eq; kinetic=TestKineticMatrices(; scale)) to solve")
    elseif kinetic !== nothing
        error("[ForceFreeStates] kinetic = \"$kinetic\" is not a kinetic source (expected \"KineticForces\")")
    end
    get(kwargs, :psilim_from_layer_overlap, false) && haskey(inputs, "SLAYER") && (kwargs[:psilim_from_layer_overlap] = slayer_model(inputs))
    kwargs[:wall] = Vacuum.parse_wall(_deck_paths(get(inputs, "Wall", nothing), path, ("file",)))
    return kwargs
end

"""
    force_balance_control(inputs, path) -> ForceFreeStatesControl

The control a deck's `[ForceFreeStates]` section and integrator section state, for the solve stages
that take one directly; [`ForceBalanceProblem`](@ref) carries the same keywords.
"""
function force_balance_control(inputs::AbstractDict, path::AbstractString)
    kwargs = force_balance_keywords(inputs, path)
    nn = pop!(kwargs, :nn)
    delete!(kwargs, :wall)
    return ForceFreeStates.validate_algorithm(ForceFreeStatesControl(; algorithm=force_balance_algorithm(inputs), nn_low=first(nn), nn_high=last(nn), kwargs...))
end

"""
    force_balance_algorithm(inputs) -> AbstractIntegrator

The integrator `[ForceFreeStates] integrator` names (default `"Riccati"`), with the numerical
settings of its own section, `[Forward]`, `[Riccati]` or `[Galerkin]`.
"""
function force_balance_algorithm(inputs::AbstractDict)
    name = _selector(inputs["ForceFreeStates"], "ForceFreeStates", "integrator", first.(_INTEGRATORS), "Riccati")
    for (other, _) in _INTEGRATORS
        other != name && haskey(inputs, other) && error("[$other] is set but [ForceFreeStates] integrator = \"$name\"")
    end
    T = Dict(_INTEGRATORS)[name]
    return T(; section_keywords(get(inputs, name, Dict{String,Any}()), name, _struct_fields(T))...)
end

"""
    local_stability_from_deck(inputs) -> Union{Nothing,NamedTuple}

`(; problem, algorithm)`: the `LocalStabilityProblem` keywords of `[LocalStability]` and the
`Ballooning` settings of `[Ballooning]` (defaults when absent). `nothing` without `[LocalStability]`.
"""
function local_stability_from_deck(inputs::AbstractDict)
    if !haskey(inputs, "LocalStability")
        haskey(inputs, "Ballooning") && error("[Ballooning] is set without [LocalStability], which runs the stage")
        return nothing
    end
    problem = section_keywords(inputs["LocalStability"], "LocalStability", _struct_fields(LocalStabilityProblem, (:equil,)))
    algorithm = Ballooning(; section_keywords(get(inputs, "Ballooning", Dict{String,Any}()), "Ballooning", _struct_fields(Ballooning))...)
    return (; problem, algorithm)
end

"""
    match_from_deck(inputs) -> Union{Nothing,NamedTuple}

`(; problem, model)`: the `MatchProblem` keywords of `[Match]` and the inner-layer model its
`inner_model` names (`"GGJ"`, from `[GGJ]`). `nothing` without `[Match]`.
"""
function match_from_deck(inputs::AbstractDict)
    haskey(inputs, "Match") || return nothing
    fields = Dict{Symbol,Any}(:eta => Any, :rho => Any, :rotation => Any, :gamma => Any, :ideal => Bool, :inner_model => String)
    problem = section_keywords(inputs["Match"], "Match", fields)
    _selector(inputs["Match"], "Match", "inner_model", ("GGJ",), "GGJ")
    delete!(problem, :inner_model)
    return (; problem, model=ggj_model(inputs))
end

"""
    tearing_from_deck(inputs) -> Union{Nothing,NamedTuple}

`(; problem, model, config)`: the `TearingProblem` keywords of `[Tearing]` (with
`[Tearing.CriticalResonantField]`), the layer model its `inner_model` names (`"SLAYER"` from
`[SLAYER]`, `"GGJ"` from `[GGJ]`) and the `TearingConfig` of `[Tearing.search]`. `nothing` without `[Tearing]`.
"""
function tearing_from_deck(inputs::AbstractDict)
    haskey(inputs, "Tearing") || return nothing
    table = inputs["Tearing"]
    get(table, "omega_E_kHz", Dict{String,Any}()) isa AbstractDict ||
        throw(ArgumentError("[Tearing] omega_E_kHz must be a table keyed by m/n, e.g. omega_E_kHz = { \"2/1\" = 3.0 }"))
    fields = _struct_fields(TearingProblem, (:ffs, :critical_resonant_field))
    fields[:inner_model] = String
    problem = section_keywords(_without(table, "search", "CriticalResonantField"), "Tearing", fields)
    inner_model = _selector(table, "Tearing", "inner_model", ("SLAYER", "GGJ"))
    delete!(problem, :inner_model)
    crf = get(table, "CriticalResonantField", nothing)
    crf === nothing || (problem[:critical_resonant_field] = CriticalResonantFieldControl(;
        section_keywords(crf, "Tearing.CriticalResonantField", _struct_fields(CriticalResonantFieldControl))...))
    config = TearingConfig(; section_keywords(get(table, "search", Dict{String,Any}()), "Tearing.search", _struct_fields(TearingConfig))...)
    return (; problem, model=inner_model == "SLAYER" ? slayer_model(inputs) : ggj_model(inputs), config)
end

"""
    perturbed_equilibrium_from_deck(inputs) -> Union{Nothing,NamedTuple}

`(; response, drive)`: the `plasma_response` and `perturbed_equilibrium` keywords of
`[PerturbedEquilibrium]`. `nothing` without that section.
"""
function perturbed_equilibrium_from_deck(inputs::AbstractDict)
    haskey(inputs, "PerturbedEquilibrium") || return nothing
    fields = Dict{Symbol,Any}(:compute_singular_coupling => Bool, :compute_torque_response => Bool, :regularization_width => Float64)
    kwargs = section_keywords(inputs["PerturbedEquilibrium"], "PerturbedEquilibrium", fields)
    response = filter(p -> p.first != :regularization_width, kwargs)
    return (; response, drive=filter(p -> p.first == :regularization_width, kwargs))
end

"""
    forcing_terms_control(inputs, path) -> ForcingTermsControl

The `[ForcingTerms]` section of a parsed deck as a control struct, defaults when absent, with its
relative file paths resolved against the deck directory `path`. The `[[ForcingTerms.coil_set]]`
array of tables is passed through as `coil_sets_raw`.
"""
function forcing_terms_control(inputs::AbstractDict, path::AbstractString)
    "ForcingTerms" in keys(inputs) || return ForcingTerms.ForcingTermsControl(; forcing_data_file=joinpath(path, "forcing.dat"))
    forcing_raw = _deck_paths(inputs["ForcingTerms"], path, ("forcing_data_file", "dat_dir"))
    haskey(forcing_raw, "forcing_data_file") || (forcing_raw["forcing_data_file"] = joinpath(path, "forcing.dat"))
    coil_sets_raw = Dict{String,Any}[_deck_paths(d, path, ("dat_file", "h5_file")) for d in get(forcing_raw, "coil_set", Dict{String,Any}[])]
    kwargs = section_keywords(_without(forcing_raw, "coil_set"), "ForcingTerms", _struct_fields(ForcingTerms.ForcingTermsControl, (:coil_sets_raw,)))
    return ForcingTerms.ForcingTermsControl(; kwargs..., coil_sets_raw=coil_sets_raw)
end

"""
    error_fields_from_deck(inputs, path) -> Union{Nothing,NamedTuple}

`(; ctrl, tolerance_file, tolerances, monte_carlo, risk, ntv, scenario)`: the `[ErrorFields]` control, the
`tolerance_file` path and the tolerance set it names (`""` and `nothing` when none), the controls of its `MonteCarlo`, `Risk` and `NTV` sub-tables, and the
`[ErrorFields.scenario]` keywords of `ErrorFields.ScenarioParameters` (`nothing` when absent). `nothing` without
`[ErrorFields]`.
"""
function error_fields_from_deck(inputs::AbstractDict, path::AbstractString)
    haskey(inputs, "ErrorFields") || return nothing
    table = _deck_paths(inputs["ErrorFields"], path, ("tolerance_file",))
    sub(name, T) = T(; section_keywords(get(table, name, Dict{String,Any}()), "ErrorFields.$name", _struct_fields(T))...)
    ctrl = ErrorFields.ErrorFieldsControl(; section_keywords(_without(table, "MonteCarlo", "Risk", "scenario", "NTV", "tolerance_file"),
        "ErrorFields", _struct_fields(ErrorFields.ErrorFieldsControl))...)
    file = get(table, "tolerance_file", "")
    isempty(file) || isfile(file) || error("[ErrorFields] tolerance_file not found: $file")
    scenario = haskey(table, "scenario") ? section_keywords(table["scenario"], "ErrorFields.scenario", _struct_fields(ErrorFields.ScenarioParameters)) : nothing
    scenario === nothing || haskey(scenario, :n_e) || error("[ErrorFields.scenario] must give n_e (electron density, 1e19 m^-3)")
    return (; ctrl, tolerance_file=file, tolerances=isempty(file) ? nothing : ErrorFields.read_tolerance_toml(file),
        monte_carlo=sub("MonteCarlo", ErrorFields.MonteCarloControl),
        risk=sub("Risk", ErrorFields.RiskControl), ntv=sub("NTV", ErrorFields.NTVControl), scenario)
end

"""
    parse_deck(inputs, path) -> NamedTuple

Every stage of a parsed deck, translated to API objects before anything runs, so a stale or
misspelled key stops the run at once. A section that no stage would read is an error.
"""
function parse_deck(inputs::AbstractDict, path::AbstractString)
    _stop_on_deprecated_deck_keys(inputs)
    analytic = [spec.section for spec in values(Equilibrium.ANALYTIC_EQ)]
    for section in keys(inputs)
        section in _DECK_SECTIONS || section in analytic ||
            error("unknown section [$section] in gpec.toml (allowed: $(join(sort!([_DECK_SECTIONS..., analytic...]), ", ")))")
    end
    has_ffs = haskey(inputs, "ForceFreeStates")
    if !has_ffs
        for section in ("Wall", "Forward", "Riccati", "Galerkin", "Match", "GGJ", "Tearing", "SLAYER", "LocalStability", "Ballooning",
            "ForcingTerms", "PerturbedEquilibrium", "KineticForces", "ErrorFields")
            haskey(inputs, section) && error("[$section] is set but there is no [ForceFreeStates] section, so the run stops after the equilibrium")
        end
    end
    force_balance = has_ffs ? (; keywords=force_balance_keywords(inputs, path), algorithm=force_balance_algorithm(inputs)) : nothing
    kinetic_source = has_ffs && get(inputs["ForceFreeStates"], "kinetic", nothing) == "KineticForces"
    haskey(inputs, "ForcingTerms") && !haskey(inputs, "PerturbedEquilibrium") &&
        error("[ForcingTerms] is set but there is no [PerturbedEquilibrium] section to apply it")
    haskey(inputs, "KineticForces") && !haskey(inputs, "PerturbedEquilibrium") && !kinetic_source &&
        error("[KineticForces] is set but neither [PerturbedEquilibrium] (the torque) nor [ForceFreeStates] kinetic = \"KineticForces\" uses it")
    (haskey(inputs, "KineticForces") || kinetic_source) && isempty(get(get(inputs, "Equilibrium", Dict{String,Any}()), "kinetic_file", "")) &&
        error("this run needs kinetic profiles ([KineticForces], or kinetic = \"KineticForces\"): set kinetic_file in [Equilibrium]")
    match = match_from_deck(inputs)
    tearing = tearing_from_deck(inputs)
    perturbed_equilibrium = perturbed_equilibrium_from_deck(inputs)
    forcing = forcing_terms_control(inputs, path)
    if haskey(inputs, "ErrorFields")
        perturbed_equilibrium === nothing && error("[ErrorFields] needs a [PerturbedEquilibrium] section")
        forcing.forcing_data_format == "coil" ||
            error("[ErrorFields] linearizes coil geometry, so [ForcingTerms] must use forcing_data_format = \"coil\" (got \"$(forcing.forcing_data_format)\")")
    end
    tearing_model = haskey(inputs, "Tearing") ? get(inputs["Tearing"], "inner_model", nothing) : nothing
    haskey(inputs, "GGJ") && match === nothing && tearing_model != "GGJ" &&
        error("[GGJ] is set but neither [Match] nor [Tearing] inner_model = \"GGJ\" uses it")
    haskey(inputs, "SLAYER") && tearing_model != "SLAYER" && !(has_ffs && get(inputs["ForceFreeStates"], "psilim_from_layer_overlap", false) === true) &&
        error("[SLAYER] is set but neither [Tearing] inner_model = \"SLAYER\" nor [ForceFreeStates] psilim_from_layer_overlap = true uses it")
    return (; run=run_settings(inputs, path), force_balance, local_stability=local_stability_from_deck(inputs), match, tearing,
        kinetic_forces=haskey(inputs, "KineticForces") ? kinetic_forces_control(inputs) : nothing, perturbed_equilibrium, forcing,
        error_fields=error_fields_from_deck(inputs, path))
end

# ---------------------------------------------------------------------------------------------------------------------
# The inverse: the API objects of a parsed deck back to the deck that states them, so every run can record a deck that
# reproduces it.

# A value as a deck states it: Symbols as strings, tuples, ranges and vectors as arrays, structs as tables.
_deck_out(v) = v
_deck_out(v::Symbol) = String(v)
_deck_out(v::Union{Tuple,AbstractVector}) = Any[_deck_out(x) for x in v]
_deck_out(v::AbstractDict) = Dict{String,Any}(String(k) => _deck_out(x) for (k, x) in v)
_deck_out(v::Equilibrium.IonSpecies) = _deck_table(v)

# The fields of `x` as a deck table, less `exclude`; a `nothing` field is left out, as the parser's default.
_deck_table(x, exclude=()) =
    Dict{String,Any}(String(f) => _deck_out(getfield(x, f)) for f in fieldnames(typeof(x)) if !(f in exclude) && getfield(x, f) !== nothing)
_deck_table(kwargs::AbstractDict, exclude=()) = Dict{String,Any}(String(k) => _deck_out(v) for (k, v) in kwargs if v !== nothing && !(k in exclude))

"""
    deck_sections(config::EquilibriumConfig, analytic) -> Dict{String,Any}

The `[Equilibrium]` section of `config` and, for an analytic equilibrium, the section of its config `analytic`
(`nothing` for a file-based one), which then implies `eq_type`.
"""
function deck_sections(config::Equilibrium.EquilibriumConfig, analytic)
    table = _deck_table(config, (:power_bp, :power_b, :power_r, :power_rc))
    analytic === nothing && return Dict{String,Any}("Equilibrium" => table)
    # An analytic equilibrium reads no file: its section implies eq_type.
    delete!(table, "eq_type")
    delete!(table, "eq_filename")
    return Dict{String,Any}("Equilibrium" => table, Equilibrium.ANALYTIC_EQ[config.eq_type].section => _deck_table(analytic))
end

deck_sections(ctrl::KineticForces.KineticForcesControl) = Dict{String,Any}("KineticForces" => _deck_table(ctrl))
deck_sections(model::SLAYERModel) = Dict{String,Any}("SLAYER" => merge(_deck_table(model), Dict{String,Any}("variant" => String(typeof(model).parameters[1]))))
deck_sections(model::GGJModel) = Dict{String,Any}("GGJ" => merge(_deck_table(Dict(pairs(model.options))), Dict{String,Any}("solver" => String(typeof(model).parameters[1]))))
deck_sections(wall::Vacuum.WallSettings) = Dict{String,Any}("Wall" => merge(_deck_table(wall), Dict{String,Any}("shape" => only(k for (k, T) in Vacuum.WALL_SHAPES if wall isa T))))

"""
    deck_sections(keywords, algorithm::AbstractIntegrator) -> Dict{String,Any}

The `[ForceFreeStates]` section, the integrator's own section and the sections its keywords bring with them
(`[Wall]`, `[KineticForces]` for a KineticForces source with non-default settings, `[SLAYER]` for the layer-overlap
cap's closures): the inverse of [`force_balance_keywords`](@ref) and [`force_balance_algorithm`](@ref). `[KineticForces]`
also runs the torque stage when present, so a run with non-default source settings and no torque stage records a
deck that adds that stage.
"""
function deck_sections(keywords::AbstractDict{Symbol}, algorithm::ForceFreeStates.AbstractIntegrator)
    kwargs = copy(keywords)
    wall, kinetic, cap = pop!(kwargs, :wall, nothing), ForceFreeStates.kinetic_source(pop!(kwargs, :kinetic, nothing)), pop!(kwargs, :psilim_from_layer_overlap, false)
    nn = pop!(kwargs, :nn)
    section = _deck_table(kwargs)
    section["nn"] = first(nn) == last(nn) ? first(nn) : collect(nn)
    name = String(nameof(typeof(algorithm)))
    section["integrator"] = name
    sections = Dict{String,Any}("ForceFreeStates" => section, name => _deck_table(algorithm))
    if kinetic isa KineticForces.KineticForcesControl
        section["kinetic"] = "KineticForces"
        # The source reads [KineticForces], whose presence alone also runs the torque stage: state it only when it differs.
        isequal(_deck_table(kinetic), _deck_table(KineticForces.KineticForcesControl())) || merge!(sections, deck_sections(kinetic))
    elseif kinetic !== nothing
        section["kinetic"] = String(nameof(typeof(kinetic)))
    end
    cap === false || (section["psilim_from_layer_overlap"] = true)
    cap isa SLAYERModel && merge!(sections, deck_sections(cap))
    wall === nothing || merge!(sections, deck_sections(wall))
    return sections
end

"""
    deck_inputs(deck, config, analytic) -> Dict{String,Any}

The deck a parsed deck `deck` (see [`parse_deck`](@ref)) states, with the equilibrium `config` and its `analytic`
config: the inverse of `parse_deck`, `equilibrium_config` and `analytic_config`. Every value is explicit and every path
absolute; `[Run] output_file` is left to whoever runs it.
"""
function deck_inputs(deck::NamedTuple, config::Equilibrium.EquilibriumConfig, analytic)
    inputs = deck_sections(config, analytic)
    inputs["Run"] = Dict{String,Any}("verbose" => deck.run.verbose)
    fb = deck.force_balance
    fb === nothing || merge!(inputs, deck_sections(fb.keywords, fb.algorithm))
    ls = deck.local_stability
    ls === nothing || merge!(inputs, Dict{String,Any}("LocalStability" => _deck_table(ls.problem), "Ballooning" => _deck_table(ls.algorithm)))
    if deck.match !== nothing
        merge!(inputs, deck_sections(deck.match.model))
        inputs["Match"] = merge(_deck_table(deck.match.problem), Dict{String,Any}("inner_model" => "GGJ"))
    end
    if deck.tearing !== nothing
        (; problem, model, config) = deck.tearing
        merge!(inputs, _tearing_sections(filter(p -> p.first != :critical_resonant_field, problem), get(problem, :critical_resonant_field, nothing),
            model, config))
    end
    deck.kinetic_forces === nothing || merge!(inputs, deck_sections(deck.kinetic_forces))
    pe = deck.perturbed_equilibrium
    pe === nothing || (inputs["PerturbedEquilibrium"] = _deck_table(merge(pe.response, pe.drive)))
    (pe !== nothing || deck.error_fields !== nothing) && (inputs["ForcingTerms"] = _forcing_table(deck.forcing))
    ef = deck.error_fields
    ef === nothing || merge!(inputs, _error_fields_sections(ef, ef.tolerance_file))
    return inputs
end

# [Tearing] with its search and critical-resonant-field sub-tables, and the section of its layer model.
function _tearing_sections(problem, crf, model, config)
    table = _deck_table(problem, (:ffs, :critical_resonant_field))
    table["inner_model"] = model isa SLAYERModel ? "SLAYER" : "GGJ"
    table["search"] = _deck_table(config)
    crf === nothing || (table["CriticalResonantField"] = _deck_table(crf))
    return merge(deck_sections(model), Dict{String,Any}("Tearing" => table))
end

# [ForcingTerms] of a forcing control, its coil sets as [[ForcingTerms.coil_set]].
function _forcing_table(ctrl::ForcingTerms.ForcingTermsControl)
    table = _deck_table(ctrl, (:coil_sets_raw,))
    isempty(ctrl.coil_sets_raw) || (table["coil_set"] = Any[_deck_out(d) for d in ctrl.coil_sets_raw])
    return table
end

# [ErrorFields] with its sub-tables, from the settings an error-field assessment takes.
function _error_fields_sections(settings, tolerance_file::AbstractString)
    table = _deck_table(settings.ctrl)
    isempty(tolerance_file) || (table["tolerance_file"] = tolerance_file)
    table["MonteCarlo"], table["Risk"], table["NTV"] = _deck_table(settings.monte_carlo), _deck_table(settings.risk), _deck_table(settings.ntv)
    settings.scenario === nothing || (table["scenario"] = _deck_table(settings.scenario))
    return Dict{String,Any}("ErrorFields" => table)
end

# ---------------------------------------------------------------------------------------------------------------------
# Every saved result records the deck sections stating its inputs (`Utilities.record_inputs!`), so a script run and a
# deck run of the same workflow record the same deck, and either replays from its gpec.h5.

# Where a replay finds what a deck cannot state by path: the recorded forcing modes and tolerance file.
const _RECORDED_FORCING = "Input/RawInputs/ForcingTerms"
const _RECORDED_TOLERANCES = "Input/RawInputs/ErrorFields/tolerance_toml_raw"

deck_sections(equil::Equilibrium.PlasmaEquilibrium) = deck_sections(equil.config, equil.analytic)
deck_sections(ls::LocalStabilityResult) = Dict{String,Any}("LocalStability" => _deck_table(ls.problem, (:equil,)), "Ballooning" => _deck_table(ls.algorithm))
deck_sections(m::ForceFreeStates.MatchResult) =
    merge(deck_sections(m.model), Dict{String,Any}("Match" => merge(_deck_table(m.problem, (:ffs,)), Dict{String,Any}("inner_model" => "GGJ"))))
deck_sections(kf::KineticForcesResult) = deck_sections(kf.control)
deck_sections(t::TearingResult) = t.model === nothing ? Dict{String,Any}() : _tearing_sections(t.problem, t.problem.critical_resonant_field, t.model, t.config)
deck_sections(ef::ErrorFieldsResult) = _error_fields_sections(ef.settings, ef.tolerances === nothing ? "" : _RECORDED_TOLERANCES)

function deck_sections(ffs::ForceFreeStatesResult)
    ctrl = ffs.control
    keywords = Dict{Symbol,Any}(f => getfield(ctrl, f) for f in fieldnames(typeof(ctrl)) if !(f in (:algorithm, :nn_low, :nn_high)))
    keywords[:nn] = ctrl.nn_low:ctrl.nn_high
    keywords[:wall] = ffs.wall_settings
    sections = merge(deck_sections(ffs.equil), deck_sections(keywords, ctrl.algorithm))
    ffs.match === nothing || merge!(sections, deck_sections(ffs.match))
    return sections
end

# A single unweighted source is its own [ForcingTerms]; any other field replays from the modes its save records.
function deck_sections(pe::PerturbedEquilibrium.PerturbedEquilibriumState)
    resp, forcing = pe.response, pe.forcing
    table = Dict{String,Any}("compute_singular_coupling" => resp.compute_singular_coupling,
        "compute_torque_response" => resp.compute_torque_response, "regularization_width" => pe.regularization_width)
    sections = Dict{String,Any}("PerturbedEquilibrium" => table)
    forcing === nothing && return sections
    sections["ForcingTerms"] = forcing isa ForcingTerms.RMPSource && forcing.scale == 1 ? _forcing_table(forcing.ctrl) :
                               Dict{String,Any}("forcing_data_format" => "hdf5", "forcing_data_file" => _RECORDED_FORCING)
    return sections
end
