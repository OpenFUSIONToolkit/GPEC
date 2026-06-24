"""
Case definition loading from TOML files.
"""

function load_case(filepath::String)::CaseSpec
    data = TOML.parsefile(filepath)

    case_section = data["case"]
    name = case_section["name"]
    description = get(case_section, "description", "")
    kind = get(case_section, "kind", "gpec_run")
    example_dir = get(case_section, "example_dir", "")

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

    return CaseSpec(name, description, example_dir, quantities, kind, overrides)
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
