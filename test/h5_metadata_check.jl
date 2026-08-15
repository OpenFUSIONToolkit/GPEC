# Shared metadata-contract walker (docs/development/hdf5-conventions.md): every
# dataset carries long_name + units, and rank ≥ 2 datasets carry a dims axis-name
# attribute. Exempt: the Input/ raw snapshot and the debug-only GalerkinIntegration
# Match/ group. Included by runtests_h5_schema.jl (full-run walk) and
# runtests_slayer_runner.jl (Tearing/ writer walk, which no full-run test deck covers).

_metadata_exempt(path) = startswith(path, "Input/") || occursin("/Match/", path)

function _collect_metadata_violations(h5)
    bad = String[]
    function walk(node, prefix)
        for k in keys(node)
            child = node[k]
            full = isempty(prefix) ? k : prefix * "/" * k
            if child isa HDF5.Group
                walk(child, full)
            elseif !_metadata_exempt(full)
                a = attrs(child)
                haskey(a, "long_name") || push!(bad, "$full: missing long_name")
                haskey(a, "units") || push!(bad, "$full: missing units")
                ndims(child) >= 2 && !haskey(a, "dims") && push!(bad, "$full: missing dims")
            end
        end
    end
    walk(h5, "")
    return bad
end
