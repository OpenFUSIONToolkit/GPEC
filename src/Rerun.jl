# Rerun support: save a run's inputs into gpec.h5 and replay them. Turns a gpec.h5 output into
# a self-contained snapshot rerunnable from any directory, with optional overrides, without the
# original g-file / CHEASE / wall / auxiliary TOML inputs.
#
# Included directly into the GeneralizedPerturbedEquilibrium module, so these functions share
# its imports (TOML, HDF5, Equilibrium, ForcingTerms, etc.).

"""
    merge_auxiliary_eq_toml!(inputs::Dict{String,Any}, eq_config)

For analytic equilibria (`lar`, `sol`), the equilibrium parameters live in a
separate auxiliary TOML referenced by `eq_config.eq_filename` (e.g. `sol.toml`
with a `[SOL_INPUT]` section). To make the `gpec.h5` snapshot self-contained,
we merge the relevant auxiliary section into the in-memory `inputs` dict under
the same key (`LAR_INPUT` / `SOL_INPUT`). On replay, `build_inputs_from_h5` reads these
sections back into a `LargeAspectRatioConfig` / `SolovevConfig` and passes
them to `setup_equilibrium` as `additional_input`, bypassing the auxiliary
file read entirely.

No-op for EFIT/CHEASE equilibria — those are snapshotted via `raw_inputs`.
"""
function merge_auxiliary_eq_toml!(inputs::Dict{String,Any}, eq_config::Equilibrium.EquilibriumConfig)
    eq_type = eq_config.eq_type
    if eq_type != "lar" && eq_type != "sol"
        return inputs
    end

    filepath = eq_config.eq_filename
    if !isfile(filepath)
        @warn "Auxiliary equilibrium TOML not found at $filepath; snapshot will not contain analytic parameters"
        return inputs
    end

    aux = TOML.parsefile(filepath)
    section = eq_type == "lar" ? "LAR_INPUT" : "SOL_INPUT"
    if haskey(aux, section)
        inputs[section] = aux[section]
    else
        @warn "Auxiliary TOML $filepath has no [$section] section; snapshot will not contain analytic parameters"
    end
    return inputs
end

"""
    write_equilibrium_raw_inputs!(out_h5, equil)

Write the raw-data dict captured on `PlasmaEquilibrium.raw_inputs` into
`input/raw_inputs/equilibrium/` inside an open HDF5 file. The layout follows
the `kind` field: `direct`, `inverse`, or `analytic`. On replay, the reader
looks at `kind` to decide whether to rebuild a `DirectRunInput` /
`InverseRunInput` or to reconstruct an analytic config from the TOML snapshot.
"""
function write_equilibrium_raw_inputs!(out_h5, equil::Equilibrium.PlasmaEquilibrium)
    raw = equil.raw_inputs
    if isempty(raw)
        @warn "Equilibrium has no raw_inputs captured; gpec.h5 will not be rerunnable"
        return nothing
    end

    group_path = "input/raw_inputs/equilibrium"
    for (key, val) in raw
        out_h5["$group_path/$key"] = val
    end
    return nothing
end

"""
    read_equilibrium_raw_inputs(in_h5) -> Dict{String,Any}

Inverse of `write_equilibrium_raw_inputs!`: read every dataset under
`input/raw_inputs/equilibrium/` into a plain dict. Used by `build_inputs_from_h5`
to hydrate a `DirectRunInput` / `InverseRunInput` without any filesystem
access.
"""
function read_equilibrium_raw_inputs(in_h5)
    group_path = "input/raw_inputs/equilibrium"
    if !haskey(in_h5, group_path)
        error("gpec.h5 has no $group_path group — the file was produced before rerun snapshot support was added")
    end
    group = in_h5[group_path]
    raw = Dict{String,Any}()
    for name in keys(group)
        raw[name] = read(group, name)
    end
    return raw
end

"""
    apply_toml_overrides!(inputs, overrides)

In-place deep-merge of a nested override dict onto `inputs`. Later values
replace earlier ones at the leaf level; intermediate tables are merged
recursively. Used by the rerun path so users can tweak individual
ForceFreeStates / Equilibrium / PerturbedEquilibrium fields without
re-editing the whole TOML blob.
"""
function apply_toml_overrides!(inputs::Dict{String,Any}, overrides::Dict{String,Any})
    for (k, v) in overrides
        if v isa Dict && haskey(inputs, k) && inputs[k] isa Dict
            apply_toml_overrides!(inputs[k], v)
        else
            inputs[k] = v
        end
    end
    return inputs
end

"""
    parse_override_flag(expr::AbstractString) -> Dict{String,Any}

Parse a `--override section.key=value` expression into a nested override dict.
Values are parsed as Julia literals via `TOML.parse("_k=_v")` so users can pass
numbers, booleans, and quoted strings without worrying about shell quoting.
"""
function parse_override_flag(expr::AbstractString)
    eqidx = findfirst(==('='), expr)
    eqidx === nothing && error("--override expects key=value, got: $expr")
    lhs = String(strip(expr[1:eqidx-1]))
    rhs = String(strip(expr[eqidx+1:end]))
    isempty(lhs) && error("--override key cannot be empty: $expr")

    parts = split(lhs, '.')
    length(parts) < 2 && error("--override key must be dotted (section.key[.subkey]): $expr")

    # Parse RHS as a TOML value by wrapping it in a throwaway key=value line.
    parsed_val = try
        parsed = TOML.parse("_rerun_override_ = $rhs")
        parsed["_rerun_override_"]
    catch e
        # Warn instead of silently stringifying a bare word, which would otherwise only fail
        # much later where the field expects a number/bool.
        @warn "Could not parse --override value as a TOML literal; storing it as a string. " *
              "Quote it explicitly if a string was intended." key=lhs value=rhs error=e
        rhs
    end

    # Build nested dict from dotted path.
    root = Dict{String,Any}()
    cursor = root
    for p in parts[1:end-1]
        cursor[String(p)] = Dict{String,Any}()
        cursor = cursor[String(p)]
    end
    cursor[String(parts[end])] = parsed_val
    return root
end

"""
    parse_rerun_cli(args) -> NamedTuple

Parse the rerun CLI. Returns `(source_h5, output_dir, output_name,
override_file, overrides)`. Unknown flags error out early.

Supported flags:

  - `--output-dir <path>`       — directory to write the replay output into (default: pwd())
  - `--output-name <filename>`  — output HDF5 filename (default: `<basename>_rerun.h5`)
  - `--override-file <path>`    — TOML file merged onto the stored TOML
  - `--override key=value`      — single-field override (repeatable)
"""
function parse_rerun_cli(args::Vector{String})
    isempty(args) && error("build_inputs_from_h5 requires a positional source .h5 path")
    source_h5 = args[1]

    output_dir = pwd()
    output_name = nothing
    override_file = nothing
    overrides = Dict{String,Any}()

    i = 2
    while i <= length(args)
        flag = args[i]
        if flag == "--output-dir"
            i + 1 > length(args) && error("--output-dir requires a value")
            output_dir = args[i+1]
            i += 2
        elseif flag == "--output-name"
            i + 1 > length(args) && error("--output-name requires a value")
            output_name = args[i+1]
            i += 2
        elseif flag == "--override-file"
            i + 1 > length(args) && error("--override-file requires a value")
            override_file = args[i+1]
            i += 2
        elseif flag == "--override"
            i + 1 > length(args) && error("--override requires a key=value expression")
            merged = parse_override_flag(args[i+1])
            apply_toml_overrides!(overrides, merged)
            i += 2
        else
            error("Unknown rerun flag: $flag")
        end
    end

    return (source_h5=source_h5, output_dir=output_dir, output_name=output_name,
        override_file=override_file, overrides=overrides)
end

"""
    resolve_rerun_output_path(source_h5, output_dir, output_name) -> (dir, name)

Decide where the rerun output should go. Default naming rule: take the source
basename, drop `.h5`, and append `_rerun.h5`. Refuse to overwrite the source
file if the resolved destination collides.
"""
function resolve_rerun_output_path(source_h5::String, output_dir::String, output_name::Union{Nothing,String})
    source_abs = abspath(source_h5)
    if output_name === nothing
        base = basename(source_abs)
        stem = endswith(lowercase(base), ".h5") ? base[1:end-3] : base
        output_name = string(stem, "_rerun.h5")
    end
    out_abs = abspath(joinpath(output_dir, output_name))
    if out_abs == source_abs
        error("Refusing to overwrite source HDF5 at $source_abs. " *
              "Pass `--output-dir <newdir>` or `--output-name <other.h5>`.")
    end
    return (output_dir, output_name)
end

"""
    rebuild_equilibrium_input(eq_config, raw) -> Union{DirectRunInput,InverseRunInput,Nothing}

Dispatch on `raw["kind"]` to rebuild the solver input. Returns `nothing` for
the `analytic` case — the caller should fall back to constructing a
`LargeAspectRatioConfig` / `SolovevConfig` from the restored TOML sections.
"""
function rebuild_equilibrium_input(eq_config::Equilibrium.EquilibriumConfig, raw::Dict{String,Any})
    kind = get(raw, "kind", "")
    if kind == "direct"
        return Equilibrium.build_direct_from_raw(eq_config, raw)
    elseif kind == "inverse"
        return Equilibrium.build_inverse_from_raw(eq_config, raw)
    elseif kind == "analytic"
        return nothing
    else
        error("Unknown raw equilibrium kind in gpec.h5: $kind")
    end
end

"""
    equilibrium_config_to_dict(eq_config) -> Dict{String,Any}

Serialize an `EquilibriumConfig` back into a plain dict keyed by field name. Used to
synthesize a complete `[Equilibrium]` section when a run was configured via the deprecated
`equil.toml`, so the gpec.h5 snapshot is self-contained and rerunnable.
"""
function equilibrium_config_to_dict(eq_config::Equilibrium.EquilibriumConfig)
    return Dict{String,Any}(String(f) => getfield(eq_config, f) for f in fieldnames(typeof(eq_config)))
end

"""
    build_inputs_from_h5(args::Vector{String})
        -> (inputs, eq_config, additional_input, output_dir, current_git, preloaded_forcing)

Rerun input builder. Parses the rerun-specific CLI flags, reads the snapshot out of the
source HDF5, applies any overrides, and reconstructs the pipeline inputs — a prebuilt
`DirectRunInput`/`InverseRunInput` for file-based equilibria, or an analytic `*Config` for
LAR/SOL. Never touches the source file's original ingest paths. The returned tuple is fed
straight into `main_from_inputs`.
"""
function build_inputs_from_h5(args::Vector{String})
    cli = parse_rerun_cli(args)
    source_h5 = cli.source_h5
    isfile(source_h5) || error("Source HDF5 not found: $source_h5")

    # Pull the stored TOML, raw equilibrium arrays, and (if present) forcing
    # data out of the source file. Forcing modes are only written when the
    # original run had a [PerturbedEquilibrium] section, so the group may be
    # missing — we signal that with `nothing` and let main_from_inputs fall
    # back to loading from `ft_ctrl.forcing_data_file`.
    toml_raw, raw_eq, source_git, preloaded_forcing = h5open(source_h5, "r") do in_h5
        haskey(in_h5, "input/gpec_toml_raw") ||
            error("Source HDF5 $source_h5 has no input/gpec_toml_raw — produced by a pre-rerun version of GPEC")
        forcing_modes = if haskey(in_h5, "input/raw_inputs/forcing_terms")
            modes = ForcingTerms.ForcingMode[]
            ForcingTerms.load_forcing_from_h5_group!(modes, in_h5["input/raw_inputs/forcing_terms"])
            modes
        else
            nothing
        end
        (
            read(in_h5, "input/gpec_toml_raw"),
            read_equilibrium_raw_inputs(in_h5),
            haskey(in_h5, "info/git_version") ? read(in_h5, "info/git_version") : "unknown",
            forcing_modes,
        )
    end

    # Parse TOML blob and fold in overrides: file overrides first, then CLI flags.
    inputs = TOML.parse(toml_raw)
    if cli.override_file !== nothing
        isfile(cli.override_file) || error("Override file not found: $(cli.override_file)")
        file_overrides = TOML.parsefile(cli.override_file)
        apply_toml_overrides!(inputs, file_overrides)
    end
    if !isempty(cli.overrides)
        apply_toml_overrides!(inputs, cli.overrides)
    end

    output_dir, output_name = resolve_rerun_output_path(source_h5, cli.output_dir, cli.output_name)
    isdir(output_dir) || mkpath(output_dir)

    # Force the rerun output file name so the source is never overwritten.
    if haskey(inputs, "ForceFreeStates")
        inputs["ForceFreeStates"]["HDF5_filename"] = output_name
    end

    current_git = try
        String(readchomp(`git -C $(@__DIR__) describe --tags --always`))
    catch
        "unknown"
    end

    @info "\n$_BANNER\n  GPEC RERUN  [source git: $source_git → current: $current_git]\n" *
          "  source: $(abspath(source_h5))\n" *
          "  output: $(abspath(joinpath(output_dir, output_name)))\n$_BANNER"

    # Rebuild the equilibrium solver input from the HDF5-stored arrays (or from
    # the merged LAR/SOL sections for analytic kinds), and run the standard
    # pipeline. Everything downstream is unchanged.
    eq_config = Equilibrium.EquilibriumConfig(inputs["Equilibrium"], output_dir)
    # eq_filename won't be used on replay — clear it so any leftover absolute
    # path from the source machine can't trip up later code paths.
    eq_config.eq_filename = ""

    additional_input = if raw_eq["kind"] == "analytic"
        if eq_config.eq_type == "lar"
            lar_data = get(inputs, "LAR_INPUT", Dict{String,Any}())
            Equilibrium.LargeAspectRatioConfig(; (Symbol(k) => v for (k, v) in lar_data)...)
        else
            sol_data = get(inputs, "SOL_INPUT", Dict{String,Any}())
            Equilibrium.SolovevConfig(; (Symbol(k) => v for (k, v) in sol_data)...)
        end
    else
        rebuild_equilibrium_input(eq_config, raw_eq)
    end

    return inputs, eq_config, additional_input, output_dir, current_git, preloaded_forcing
end
