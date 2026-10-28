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

_input_fields(x) = Dict{String,Any}(String(f) => getfield(x, f) for f in fieldnames(typeof(x)) if getfield(x, f) !== nothing)

function result_inputs(result::ForceFreeStatesResult, inputs)
    snapshot = inputs === nothing ? Dict{String,Any}() : deepcopy(inputs)
    snapshot["Equilibrium"] = _input_fields(result.equil.config)
    for key in ("power_bp", "power_b", "power_r", "power_rc")
        delete!(snapshot["Equilibrium"], key)
    end
    snapshot["ForceFreeStates"] = _input_fields(result.control)
    snapshot["DEBUG"] = _input_fields(result.debug_settings)
    if result.equil.analytic !== nothing
        section = Equilibrium.ANALYTIC_EQ[result.equil.config.eq_type].section
        snapshot[section] = _input_fields(result.equil.analytic)
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
