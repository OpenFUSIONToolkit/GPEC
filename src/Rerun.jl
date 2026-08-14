# Rerun support: save a run's inputs into gpec.h5 and replay them. Turns a gpec.h5 output into
# a self-contained snapshot rerunnable from any directory, with optional overrides, without the
# original g-file / CHEASE / wall inputs.
#
# Included directly into the GeneralizedPerturbedEquilibrium module, so these functions share
# its imports (TOML, HDF5, Equilibrium, ForcingTerms, etc.).

# Analytic equilibria carry their parameters in an embedded TOML section rather than an ingest
# array; replay regenerates them from that section. The eq_type -> section mapping (and which
# kinds are analytic at all) lives in the `Equilibrium.ANALYTIC_EQ` registry, the single source
# of truth shared with `setup_equilibrium`.

"""
    build_analytic_config(eq_type, inputs)

Construct the analytic-equilibrium `*Config` (`SolovevConfig` / `LargeAspectRatioConfig` /
`TJAnalyticConfig`) from the TOML section named in the `Equilibrium.ANALYTIC_EQ` registry.
Shared by the TOML and rerun input builders so both resolve analytic parameters the same way.
"""
function build_analytic_config(eq_type::AbstractString, inputs::Dict)
    spec = get(Equilibrium.ANALYTIC_EQ, eq_type, nothing)
    spec === nothing && error("build_analytic_config: $eq_type is not an analytic equilibrium type")
    haskey(inputs, spec.section) || error("Analytic equilibrium $eq_type requires a [$(spec.section)] section in gpec.toml")
    return spec.config_type(inputs[spec.section])
end

"""
    read_equilibrium_ingest(in_h5) -> EquilibriumIngest

Reconstruct the [`DirectIngest`](@ref)/[`InverseIngest`](@ref) stored under
`input/raw_inputs/equilibrium/` (the inverse of the field-by-field write in
`write_outputs_to_HDF5`). Returns `nothing` when the group is absent, which marks an
analytic equilibrium — replayed from its TOML section rather than stored arrays.
"""
function read_equilibrium_ingest(in_h5)
    group_path = "input/raw_inputs/equilibrium"
    haskey(in_h5, group_path) || return nothing
    group = in_h5[group_path]
    kind = read(group, "ingest_kind")
    T = kind == "direct" ? Equilibrium.DirectIngest :
        kind == "inverse" ? Equilibrium.InverseIngest :
        error("Unknown equilibrium ingest_kind in gpec.h5: $kind (expected \"direct\" or \"inverse\")")
    # Positional reconstruction: relies on the default constructor, so `fieldnames(T)` order
    # must match the struct definition and the field-by-field write in write_outputs_to_HDF5.
    return T((read(group, String(f)) for f in fieldnames(T))...)
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
    sentinel = "_rerun_override_"
    parsed_val = try
        TOML.parse("$sentinel = $rhs")[sentinel]
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

Parse the rerun CLI. Returns `(source_h5, output_dir, output_name, override_file, overrides, coil_source)`. Unknown flags error out early.

Supported flags:

  - `--output-dir <path>`       — directory to write the replay output into (default: pwd())
  - `--output-name <filename>`  — output HDF5 filename (default: `<basename>_rerun.h5`)
  - `--override-file <path>`    — TOML file merged onto the stored TOML
  - `--override key=value`      — single-field override (repeatable)
  - `--coil-source <which>`     — for coil runs, which snapshot drives the replay:
    `forcing-modes` (default) reuses the frozen forcing modes exactly; `coils`
    recomputes the field from the stored coil geometry (lets the equilibrium change).
"""
function parse_rerun_cli(args::Vector{String})
    isempty(args) && error("build_inputs_from_h5 requires a positional source .h5 path")
    source_h5 = args[1]

    output_dir = pwd()
    output_name = nothing
    override_file = nothing
    overrides = Dict{String,Any}()
    coil_source = "forcing-modes"

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
        elseif flag == "--coil-source"
            i + 1 > length(args) && error("--coil-source requires a value (forcing-modes|coils)")
            coil_source = args[i+1]
            coil_source in ("forcing-modes", "coils") ||
                error("--coil-source must be 'forcing-modes' or 'coils', got: $coil_source")
            i += 2
        else
            error("Unknown rerun flag: $flag")
        end
    end

    return (source_h5=source_h5, output_dir=output_dir, output_name=output_name,
        override_file=override_file, overrides=overrides, coil_source=coil_source)
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
    if abspath(joinpath(output_dir, output_name)) == source_abs
        error("Refusing to overwrite source HDF5 at $source_abs. " *
              "Pass `--output-dir <newdir>` or `--output-name <other.h5>`.")
    end
    return (output_dir, output_name)
end

"""
    build_inputs_from_h5(args::Vector{String})
        -> (inputs, eq_config, additional_input, output_dir, current_git, preloaded_forcing, preloaded_coils)

Rerun input builder. Parses the rerun-specific CLI flags, reads the snapshot out of the
source HDF5, applies any overrides, and reconstructs the pipeline inputs — a prebuilt
`DirectRunInput`/`InverseRunInput` rebuilt from the stored ingest for file-based equilibria,
or an analytic `*Config` for sol/lar/tj. Never touches the source file's original ingest
paths. The returned tuple is fed straight into `main_from_inputs`.
"""
function build_inputs_from_h5(args::Vector{String})
    cli = parse_rerun_cli(args)
    source_h5 = cli.source_h5
    isfile(source_h5) || error("Source HDF5 not found: $source_h5")

    # Pull the stored TOML, equilibrium ingest, and (if present) forcing data out of the source
    # file. Forcing modes are only written when the original run had a [PerturbedEquilibrium]
    # section, so the group may be missing — we signal that with `nothing` and let
    # main_from_inputs fall back to loading from `ft_ctrl.forcing_data_file`.
    # `--coil-source coils` recomputes the coil field from stored geometry, so it deliberately
    # ignores the frozen forcing-mode snapshot.
    use_coils = cli.coil_source == "coils"
    toml_raw, ingest, source_git, preloaded_forcing, preloaded_coils = h5open(source_h5, "r") do in_h5
        haskey(in_h5, "input/gpec_toml_raw") ||
            error("Source HDF5 $source_h5 has no input/gpec_toml_raw — produced by a pre-rerun version of GPEC")
        forcing_modes = if use_coils || !haskey(in_h5, "input/raw_inputs/forcing_terms")
            nothing
        else
            modes = ForcingTerms.ForcingMode[]
            ForcingTerms.load_forcing_from_h5_group!(modes, in_h5["input/raw_inputs/forcing_terms"])
            modes
        end
        coil_sets = if use_coils
            haskey(in_h5, "input/raw_inputs/coils") ||
                error("--coil-source coils requested but $source_h5 has no input/raw_inputs/coils " *
                    "(the source run did not use coils, or predates coil-snapshot support)")
            sets = ForcingTerms.CoilSet[]
            ForcingTerms.load_coils_from_h5_group!(sets, in_h5["input/raw_inputs/coils"])
            sets
        else
            nothing
        end
        (
            read(in_h5, "input/gpec_toml_raw"),
            read_equilibrium_ingest(in_h5),
            haskey(in_h5, "info/git_version") ? read(in_h5, "info/git_version") : "unknown",
            forcing_modes,
            coil_sets
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

    _drop_deprecated_keys!(inputs["Equilibrium"], _DEPRECATED_EQUIL_KEYS, "Equilibrium")
    # Clear eq_filename on a copy: unused on replay, a stale absolute path could mislead
    # downstream code, and `inputs` itself is re-serialized into the rerun's gpec_toml_raw.
    equil_dict = merge(inputs["Equilibrium"], Dict{String,Any}("eq_filename" => ""))
    eq_config = Equilibrium.EquilibriumConfig(equil_dict, output_dir)

    # Analytic kinds regenerate from their TOML section; file-based kinds rebuild splines from
    # the stored ingest. A file-based run with no ingest can only come from a pre-ingest gpec.h5.
    additional_input = if haskey(Equilibrium.ANALYTIC_EQ, eq_config.eq_type)
        build_analytic_config(eq_config.eq_type, inputs)
    elseif ingest isa Equilibrium.DirectIngest
        Equilibrium.build_direct_from_ingest(eq_config, ingest)
    elseif ingest isa Equilibrium.InverseIngest
        Equilibrium.build_inverse_from_ingest(eq_config, ingest)
    else
        error("gpec.h5 has no equilibrium ingest and eq_type=$(eq_config.eq_type) is not analytic — cannot replay. " *
              "A file-based eq_type needs a stored ingest (pre-ingest snapshots lack one); a new analytic kind must be " *
              "registered in Equilibrium.ANALYTIC_EQ.")
    end

    return inputs, eq_config, additional_input, output_dir, current_git, preloaded_forcing, preloaded_coils
end
