function save_input_snapshot!(out_h5, result::ForceFreeStatesResult; inputs=nothing, forcing_snapshot=nothing)
    equil = result.equil
    # Self-contained run snapshot: the full merged TOML (so a rerun can reconstruct every
    # ForceFreeStates/Equilibrium/Wall/PE control struct), plus the equilibrium ingest
    # arrays so a file-based rerun never needs the original g-file / CHEASE / IMAS source.
    out_h5["Input/gpec_toml_raw"] = sprint(TOML.print, result_inputs(result, inputs))
    if equil.ingest !== nothing  # analytic equilibria are regenerated from their TOML section
        eq_group = "Input/RawInputs/Equilibrium"  # read back by Rerun.read_equilibrium_ingest
        out_h5["$eq_group/ingest_kind"] = equil.ingest isa Equilibrium.DirectIngest ? "direct" : "inverse"
        for f in fieldnames(typeof(equil.ingest))
            out_h5["$eq_group/$f"] = getfield(equil.ingest, f)
        end
    end
    if forcing_snapshot !== nothing  # read back by Rerun.build_inputs_from_h5
        modes, normalization = forcing_snapshot
        ForcingTerms.save_forcing_to_h5(modes, create_group(out_h5, "Input/RawInputs/ForcingTerms"); normalization)
    end

    return nothing
end

# A value as TOML can state it: Symbols as strings, tuples and species as arrays and tables.
_toml_value(v) = v
_toml_value(v::Symbol) = String(v)
_toml_value(v::Union{Tuple,AbstractVector}) = Any[_toml_value(x) for x in v]
_toml_value(v::Equilibrium.IonSpecies) = _input_fields(v)

_input_fields(x) = Dict{String,Any}(String(f) => _toml_value(getfield(x, f)) for f in fieldnames(typeof(x)) if getfield(x, f) !== nothing)

"""
    control_to_deck(ctrl) -> (section, name, algorithm)

The deck form of a force-balance control: its `[ForceFreeStates]` section, the name of its
integrator (which is also the name of that integrator's section) and that section.
"""
function control_to_deck(ctrl::ForceFreeStatesControl)
    section = Dict{String,Any}(String(f) => _toml_value(getfield(ctrl, f)) for f in fieldnames(typeof(ctrl))
                               if !(f in (:algorithm, :kinetic, :nn_low, :nn_high, :psilim_from_layer_overlap)))
    section["nn"] = ctrl.nn_low == ctrl.nn_high ? ctrl.nn_low : collect(ctrl.nn_low:ctrl.nn_high)
    section["psilim_from_layer_overlap"] = ctrl.psilim_from_layer_overlap !== false
    name = String(nameof(typeof(ctrl.algorithm)))
    section["integrator"] = name
    return section, name, _input_fields(ctrl.algorithm)
end

function result_inputs(result::ForceFreeStatesResult, inputs)
    snapshot = inputs === nothing ? Dict{String,Any}() : deepcopy(inputs)
    snapshot["Equilibrium"] = _input_fields(result.equil.config)
    for key in ("power_bp", "power_b", "power_r", "power_rc")
        delete!(snapshot["Equilibrium"], key)
    end
    section, name, algorithm = control_to_deck(result.control)
    snapshot["ForceFreeStates"] = section
    foreach(other -> delete!(snapshot, other), ("Forward", "Riccati", "Galerkin"))
    snapshot[name] = algorithm
    # A deck names the KineticForces model and states it in [KineticForces]; test matrices are recorded by name only.
    kinetic = result.control.kinetic
    if kinetic isa TestKineticMatrices
        section["kinetic"] = "TestKineticMatrices"
    elseif kinetic !== nothing
        section["kinetic"] = "KineticForces"
        haskey(snapshot, "KineticForces") || (snapshot["KineticForces"] = _input_fields(kinetic))
    end
    # A deck states the overlap cap's layer closures in [SLAYER]; a rerun rebuilds the same SLAYERModel from them.
    cap = result.control.psilim_from_layer_overlap
    if cap isa InnerLayer.SLAYERModel
        slayer = get!(snapshot, "SLAYER", Dict{String,Any}())
        merge!(slayer, _input_fields(cap))
        slayer["variant"] = String(typeof(cap).parameters[1])
    end
    if result.equil.analytic !== nothing
        snapshot[Equilibrium.ANALYTIC_EQ[result.equil.config.eq_type].section] = _input_fields(result.equil.analytic)
    end
    wall = result.wall_settings
    if wall === nothing
        pop!(snapshot, "Wall", nothing)
    else
        snapshot["Wall"] = _input_fields(wall)
        snapshot["Wall"]["shape"] = only(k for (k, T) in Vacuum.WALL_SHAPES if wall isa T)
    end
    return snapshot
end
