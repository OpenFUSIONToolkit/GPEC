"""
HDF5 quantity extraction engine.
"""

"""
Extract all quantities from a gpec.h5 file according to case spec.
Returns a Vector{ExtractedQuantity}.
"""
function extract_quantities(h5path::String, qty_specs::Vector{QuantitySpec}, runtime_s::Float64)::Vector{ExtractedQuantity}
    results = ExtractedQuantity[]

    h5open(h5path, "r") do fid
        for spec in qty_specs
            if spec.type == "runtime"
                push!(results, ExtractedQuantity(
                    spec.name, spec.label,
                    runtime_s, nothing, nothing,
                    "real", spec.noise_threshold))
                continue
            end

            # Check if the H5 path exists
            if !haskey(fid, spec.h5path)
                push!(results, ExtractedQuantity(
                    spec.name, spec.label,
                    nothing, nothing, nothing,
                    "missing", spec.noise_threshold))
                continue
            end

            raw = read(fid[spec.h5path])
            eq = apply_extraction(spec, raw)
            push!(results, eq)
        end
    end

    return results
end

"""
Apply an extraction rule to raw HDF5 data, producing an ExtractedQuantity.
"""
function apply_extraction(spec::QuantitySpec, raw)::ExtractedQuantity
    name = spec.name
    label = spec.label
    threshold = spec.noise_threshold

    if spec.extract == "value"
        if raw isa Integer
            return ExtractedQuantity(name, label, nothing, Int(raw), nothing, "integer", threshold)
        else
            return ExtractedQuantity(name, label, Float64(raw), nothing, nothing, "real", threshold)
        end

    elseif spec.extract == "real_first"
        val = real(raw[1])
        return ExtractedQuantity(name, label, Float64(val), nothing, nothing, "real", threshold)

    elseif spec.extract == "imag_first"
        val = imag(raw[1])
        return ExtractedQuantity(name, label, Float64(val), nothing, nothing, "real", threshold)

    elseif spec.extract == "abs_first"
        val = abs(raw[1])
        return ExtractedQuantity(name, label, Float64(val), nothing, nothing, "real", threshold)

    elseif spec.extract == "norm"
        val = sqrt(sum(abs2, raw))
        return ExtractedQuantity(name, label, Float64(val), nothing, nothing, "real", threshold)

    elseif spec.extract == "all_real"
        arr = Float64.(real.(raw))
        json_str = JSON.json(arr)
        return ExtractedQuantity(name, label, nothing, nothing, json_str, "json_array", threshold)

    elseif spec.extract == "all_complex"
        pairs = [[real(x), imag(x)] for x in raw]
        json_str = JSON.json(pairs)
        return ExtractedQuantity(name, label, nothing, nothing, json_str, "json_array", threshold)

    elseif spec.extract == "diagonal_complex"
        # Extract the diagonal of a square matrix as a complex array.
        # Use for tracking per-surface BVP Δ' from singular/delta_prime_matrix.
        ndims(raw) == 2 && size(raw, 1) == size(raw, 2) ||
            error("diagonal_complex requires a square 2-D matrix; got size $(size(raw))")
        diag_vec = [raw[i, i] for i in 1:size(raw, 1)]
        pairs = [[real(x), imag(x)] for x in diag_vec]
        json_str = JSON.json(pairs)
        return ExtractedQuantity(name, label, nothing, nothing, json_str, "json_array", threshold)

    elseif spec.extract == "checksum"
        bytes = reinterpret(UInt8, vec(collect(raw)))
        hash = bytes2hex(sha256(bytes))
        return ExtractedQuantity(name, label, nothing, nothing, hash, "checksum", threshold)

    else
        error("Unknown extraction mode: $(spec.extract)")
    end
end

"""Compute absolute difference between two JSON-parsed elements (scalars, pairs, or nested arrays)."""
function _json_element_diff(a, b)::Float64
    if a isa Number && b isa Number
        return abs(Float64(a) - Float64(b))
    elseif a isa Vector && b isa Vector
        return sqrt(sum(_json_element_diff(ai, bi)^2 for (ai, bi) in zip(a, b)))
    else
        return Inf
    end
end

"""Compute absolute value/magnitude of a JSON-parsed element."""
function _json_element_abs(x)::Float64
    if x isa Number
        return abs(Float64(x))
    elseif x isa Vector
        return sqrt(sum(_json_element_abs(xi)^2 for xi in x))
    else
        return 0.0
    end
end

"""
Compare two extracted values. Returns (abs_diff, rel_diff, status).
For arrays, returns max element-wise absolute diff.
"""
function compare_values(q1::NamedTuple, q2::NamedTuple)
    vtype = q1.value_type

    # Handle missing values
    if vtype == "missing" || q2.value_type == "missing"
        return (NaN, NaN, "N/A")
    end

    # Storage type differs between refs (e.g. a quantity whose schema changed from a
    # vector to a scalar across the two commits); the values are not directly comparable.
    if vtype != q2.value_type
        return (NaN, NaN, "N/A (type changed)")
    end

    threshold = q1.noise_threshold

    if vtype == "real"
        v1 = q1.value_real
        v2 = q2.value_real
        if v1 === nothing || v2 === nothing
            return (NaN, NaN, "N/A")
        end
        abs_diff = abs(v2 - v1)
        rel_diff = v1 == 0.0 ? (v2 == 0.0 ? 0.0 : Inf) : abs_diff / abs(v1)
        status = abs_diff <= threshold ? "OK" : "CHANGED"
        return (abs_diff, rel_diff, status)

    elseif vtype == "integer"
        v1 = q1.value_int
        v2 = q2.value_int
        if v1 === nothing || v2 === nothing
            return (NaN, NaN, "N/A")
        end
        abs_diff = Float64(abs(v2 - v1))
        rel_diff = v1 == 0 ? (v2 == 0 ? 0.0 : Inf) : abs_diff / abs(v1)
        status = abs_diff <= threshold ? "OK" : "CHANGED"
        return (abs_diff, rel_diff, status)

    elseif vtype == "json_array"
        t1 = q1.value_text
        t2 = q2.value_text
        if t1 === nothing || t2 === nothing
            return (NaN, NaN, "N/A")
        end
        arr1 = JSON.parse(t1)
        arr2 = JSON.parse(t2)
        if length(arr1) != length(arr2)
            return (NaN, NaN, "CHANGED (length)")
        end
        max_diff = 0.0
        for (a, b) in zip(arr1, arr2)
            d = _json_element_diff(a, b)
            max_diff = max(max_diff, d)
        end
        # For relative diff, use max absolute value as reference
        max_val = maximum(x -> _json_element_abs(x), arr1; init=0.0)
        rel_diff = max_val == 0.0 ? (max_diff == 0.0 ? 0.0 : Inf) : max_diff / max_val
        status = max_diff <= threshold ? "OK" : "CHANGED"
        return (max_diff, rel_diff, status)

    elseif vtype == "checksum"
        t1 = q1.value_text
        t2 = q2.value_text
        if t1 === nothing || t2 === nothing
            return (NaN, NaN, "N/A")
        end
        status = t1 == t2 ? "OK" : "CHANGED"
        return (0.0, 0.0, status)

    else
        return (NaN, NaN, "N/A")
    end
end
