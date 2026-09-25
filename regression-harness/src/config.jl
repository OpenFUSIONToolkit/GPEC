"""
Case definition loading from TOML files.
"""

"""
    resolve_case_base(data, filepath)

Merge a variant case onto the case named by its `[case] base` key, read from the sibling file
`<base>.toml`. The variant must set its own `name`; its `[case]` keys, `[quantities.*]` blocks and
`[overrides]` entries replace the base's same-named ones, and everything else is inherited.
A missing base, or a base that itself has a base, is an error.
"""
function resolve_case_base(data::Dict{String,Any}, filepath::String)
    base = get(data["case"], "base", nothing)
    base === nothing && return data
    file = basename(filepath)
    haskey(data["case"], "name") || error("Case $file: a case with a base must set its own [case] name")
    base_path = joinpath(dirname(filepath), "$base.toml")
    isfile(base_path) || error("Case $file: base case \"$base\" not found (expected $base_path)")
    base_data = TOML.parsefile(base_path)
    haskey(base_data["case"], "base") && error("Case $file: base case \"$base\" itself has a base; chained bases are not supported")
    merged = Dict{String,Any}()
    for section in ("case", "quantities", "overrides")
        merged[section] = merge(get(base_data, section, Dict{String,Any}()), get(data, section, Dict{String,Any}()))
    end
    delete!(merged["case"], "base")
    return merged
end

function load_case(filepath::String)::CaseSpec
    data = resolve_case_base(TOML.parsefile(filepath), filepath)

    case_section = data["case"]
    name = case_section["name"]
    description = get(case_section, "description", "")
    kind = get(case_section, "kind", "gpec_run")
    example_dir = get(case_section, "example_dir", "")
    precompile_workload = get(case_section, "precompile_workload", false)

    quantities = QuantitySpec[]
    if haskey(data, "quantities")
        for (qty_name, qty_data) in data["quantities"]
            push!(
                quantities,
                QuantitySpec(
                    qty_name,
                    get(qty_data, "h5path", ""),
                    qty_data["type"],
                    qty_data["extract"],
                    get(qty_data, "label", qty_name),
                    get(qty_data, "noise_threshold", 1e-10),
                    get(qty_data, "order", 1000)
                )
            )
        end
        sort!(quantities; by=q -> q.order)
    end

    overrides = Dict{String,Any}()
    if haskey(data, "overrides")
        for (k, v) in data["overrides"]
            overrides[k] = v
        end
    end

    return CaseSpec(name, description, example_dir, quantities, kind, overrides, precompile_workload)
end

function load_all_cases(cases_dir::String)::Dict{String,CaseSpec}
    cases = Dict{String,CaseSpec}()
    if !isdir(cases_dir)
        @warn "Cases directory not found: $cases_dir"
        return cases
    end
    for filename in readdir(cases_dir)
        if endswith(filename, ".toml")
            filepath = joinpath(cases_dir, filename)
            case = load_case(filepath)
            cases[case.name] = case
        end
    end
    return cases
end

function print_cases(cases::Dict{String,CaseSpec})
    if isempty(cases)
        println("No cases found.")
        return
    end
    println("Available regression cases:")
    println("-"^64)
    for name in sort(collect(keys(cases)))
        c = cases[name]
        nqty = length(c.quantities)
        println("  $(rpad(c.name, 24)) $(rpad(c.description, 40))")
        loc = c.kind == "computed" ? "kind: computed" : "dir: $(c.example_dir)"
        println("  $(rpad("", 24)) $loc  ($nqty quantities)")
    end
end
