"""
Reporter: formats comparison tables for stdout output.
"""

"""
Format a value for display. Returns an unpadded string.
"""
function format_value(q::NamedTuple)::String
    if q.value_type == "missing"
        return "N/A"
    elseif q.value_type == "real"
        v = q.value_real
        v === nothing && return "N/A"
        return @sprintf("%.6e", v)
    elseif q.value_type == "integer"
        v = q.value_int
        v === nothing && return "N/A"
        return string(v)
    elseif q.value_type == "json_array"
        t = q.value_text
        t === nothing && return "N/A"
        arr = JSON.parse(t)
        return "[$(length(arr)) elem]"
    elseif q.value_type == "checksum"
        t = q.value_text
        t === nothing && return "N/A"
        return t[1:min(12, length(t))] * "..."
    else
        return "?"
    end
end

"""
Format a diff value for display.
"""
function format_diff(abs_diff::Float64, rel_diff::Float64, status::String, value_type::String)::String
    if status == "N/A" || isnan(abs_diff)
        return "N/A"
    end
    if status == "OK"
        if value_type == "checksum"
            return "identical"
        end
        return @sprintf("%.1e", abs_diff)
    end
    if contains(status, "length")
        return status
    end
    return @sprintf("%.3e (%.2f%%)", abs_diff, rel_diff * 100)
end

"""Helper: print a row of padded columns."""
function _print_row(cols::Vector{String}, widths::Vector{Int})
    parts = [rpad(cols[i], widths[i]) for i in eachindex(cols)]
    println(join(parts, "  "))
end

"""
Print a two-ref comparison report for a single case.
"""
function report_two_ref_comparison(db::SQLite.DB, case_spec::CaseSpec,
                                   ref1::ResolvedRef, ref2::ResolvedRef)
    info1 = get_run_info(db, ref1.commit_hash, case_spec.name)
    info2 = get_run_info(db, ref2.commit_hash, case_spec.name)

    println()
    println("Regression Report: $(case_spec.name)")

    if info1 === nothing
        println("ERROR: No results for ref 1 ($(ref1.name))")
        return
    end
    if info2 === nothing
        println("ERROR: No results for ref 2 ($(ref2.name))")
        return
    end
    if !info1.success
        println("FAILED at ref 1 ($(ref1.name) @ $(info1.commit_short)): $(info1.error_msg)")
        return
    end
    if !info2.success
        println("FAILED at ref 2 ($(ref2.name) @ $(info2.commit_short)): $(info2.error_msg)")
        return
    end

    q1_all = get_quantities(db, ref1.commit_hash, case_spec.name)
    q2_all = get_quantities(db, ref2.commit_hash, case_spec.name)

    # Pre-compute all rows: (label, v1, v2, diff, status)
    header = ["Quantity", ref1.name, ref2.name, "Diff", "Status"]
    rows = Vector{Vector{String}}()
    n_ok = 0; n_changed = 0; n_missing = 0

    for spec in case_spec.quantities
        qname = spec.name

        if !haskey(q1_all, qname) && !haskey(q2_all, qname)
            n_missing += 1
            continue
        end

        if spec.type == "runtime"
            v1 = (haskey(q1_all, qname) && q1_all[qname].value_real !== nothing) ?
                 @sprintf("%.1fs", q1_all[qname].value_real) : "N/A"
            v2 = (haskey(q2_all, qname) && q2_all[qname].value_real !== nothing) ?
                 @sprintf("%.1fs", q2_all[qname].value_real) : "N/A"
            push!(rows, [spec.label, v1, v2, "", "--"])
            continue
        end

        has1 = haskey(q1_all, qname)
        has2 = haskey(q2_all, qname)
        if !has1 || !has2
            v1 = has1 ? format_value(q1_all[qname]) : "N/A"
            v2 = has2 ? format_value(q2_all[qname]) : "N/A"
            push!(rows, [spec.label, v1, v2, "", "N/A"])
            n_missing += 1
            continue
        end

        q1 = q1_all[qname]; q2 = q2_all[qname]
        abs_diff, rel_diff, status = compare_values(q1, q2)
        st = status == "CHANGED" ? "** CHANGED **" : status
        push!(rows, [spec.label, format_value(q1), format_value(q2),
                     format_diff(abs_diff, rel_diff, status, q1.value_type), st])

        if status == "OK"; n_ok += 1
        elseif contains(status, "CHANGED"); n_changed += 1
        else; n_missing += 1; end
    end

    # Derive column widths from header + all row content
    ncols = length(header)
    widths = [length(header[i]) for i in 1:ncols]
    for row in rows
        for i in 1:ncols
            widths[i] = max(widths[i], length(row[i]))
        end
    end

    total_w = sum(widths) + 2 * (ncols - 1)

    println("="^total_w)
    date1 = length(info1.commit_date) >= 10 ? info1.commit_date[1:10] : info1.commit_date
    date2 = length(info2.commit_date) >= 10 ? info2.commit_date[1:10] : info2.commit_date
    println("Ref 1: $(ref1.name)  @ $(info1.commit_short) ($date1)")
    println("Ref 2: $(ref2.name)  @ $(info2.commit_short) ($date2)")
    println("-"^total_w)

    _print_row(header, widths)
    println("-"^total_w)
    for row in rows
        _print_row(row, widths)
    end

    println("="^total_w)
    parts = String[]
    n_changed > 0 && push!(parts, "$n_changed changed")
    n_ok > 0 && push!(parts, "$n_ok unchanged")
    n_missing > 0 && push!(parts, "$n_missing missing/N/A")
    println("Summary: ", join(parts, ", "))
    println()
end

"""
Print a multi-ref tracking report for a single case.
One row per quantity, one column per ref.
"""
function report_multi_ref(db::SQLite.DB, case_spec::CaseSpec,
                          refs::Vector{ResolvedRef})
    run_infos = [get_run_info(db, ref.commit_hash, case_spec.name) for ref in refs]
    all_qs = [begin
        info = run_infos[i]
        (info !== nothing && info.success) ? get_quantities(db, refs[i].commit_hash, case_spec.name) : Dict{String,NamedTuple}()
    end for i in eachindex(refs)]

    show_diff = length(refs) >= 2
    ref_names = [ref.name for ref in refs]

    # Build header
    header = vcat(["Quantity"], ref_names)
    if show_diff
        push!(header, "Δ prev")
        push!(header, "Status")
    end
    ncols = length(header)

    # Pre-compute rows
    rows = Vector{Vector{String}}()
    n_ok = 0; n_changed = 0; n_missing = 0

    for spec in case_spec.quantities
        qname = spec.name
        any_present = any(haskey(qs, qname) for qs in all_qs)
        if !any_present
            n_missing += 1
            continue
        end

        row = [spec.label]

        if spec.type == "runtime"
            for qs in all_qs
                val = (haskey(qs, qname) && qs[qname].value_real !== nothing) ?
                      @sprintf("%.1fs", qs[qname].value_real) : "N/A"
                push!(row, val)
            end
            if show_diff
                push!(row, "")
                push!(row, "--")
            end
            push!(rows, row)
            continue
        end

        for qs in all_qs
            push!(row, haskey(qs, qname) ? format_value(qs[qname]) : "N/A")
        end

        if show_diff
            q_last = haskey(all_qs[end], qname) ? all_qs[end][qname] : nothing
            q_prev = haskey(all_qs[end-1], qname) ? all_qs[end-1][qname] : nothing
            if q_last !== nothing && q_prev !== nothing
                abs_diff, rel_diff, status = compare_values(q_prev, q_last)
                st = status == "CHANGED" ? "** CHANGED **" : status
                push!(row, format_diff(abs_diff, rel_diff, status, q_prev.value_type))
                push!(row, st)
                if status == "OK"; n_ok += 1
                elseif contains(status, "CHANGED"); n_changed += 1
                else; n_missing += 1; end
            else
                push!(row, "N/A")
                push!(row, "N/A")
                n_missing += 1
            end
        end

        push!(rows, row)
    end

    # Derive column widths
    widths = [length(header[i]) for i in 1:ncols]
    for row in rows
        for i in 1:ncols
            widths[i] = max(widths[i], length(row[i]))
        end
    end
    total_w = sum(widths) + 2 * (ncols - 1)

    println()
    println("Regression Tracking: $(case_spec.name)")
    println("="^total_w)

    for (i, ref) in enumerate(refs)
        info = run_infos[i]
        if info !== nothing
            date_str = length(info.commit_date) >= 10 ? info.commit_date[1:10] : info.commit_date
            status_str = info.success ? "" : " (FAILED)"
            println("Ref $(i): $(ref.name)  @ $(info.commit_short) ($date_str)$status_str")
        else
            println("Ref $(i): $(ref.name)  (no data)")
        end
    end
    println("-"^total_w)

    _print_row(header, widths)
    println("-"^total_w)
    for row in rows
        _print_row(row, widths)
    end

    println("="^total_w)
    if show_diff
        parts = String[]
        n_changed > 0 && push!(parts, "$n_changed changed")
        n_ok > 0 && push!(parts, "$n_ok unchanged")
        n_missing > 0 && push!(parts, "$n_missing missing/N/A")
        println("Summary (last vs prev): ", join(parts, ", "))
    end
    println()
end

"""
Show the history of a single quantity for a case (--show mode).
"""
function show_quantity_history(db::SQLite.DB, case_name::String, qty_name::String)
    runs = get_all_runs_for_case(db, case_name)
    if isempty(runs)
        println("No cached runs found for case '$case_name'")
        return
    end

    header = ["Commit", "Date", "Value", "Δ from prev", "Status"]
    ncols = length(header)
    rows = Vector{Vector{String}}()

    prev_q = nothing
    for row in runs
        if coalesce(row.success, 0) != 1
            push!(rows, [something(row.commit_short, ""), "", "", "", "FAILED"])
            prev_q = nothing
            continue
        end

        qs = get_quantities(db, row.commit_hash, case_name)
        date_str = let d = something(row.commit_date, "")
            length(d) >= 10 ? d[1:10] : d
        end
        short = something(row.commit_short, "")

        if !haskey(qs, qty_name)
            push!(rows, [short, date_str, "N/A", "", "N/A"])
            prev_q = nothing
            continue
        end

        q = qs[qty_name]
        val_str = format_value(q)

        if prev_q === nothing
            push!(rows, [short, date_str, val_str, "--", "--"])
        else
            abs_diff, rel_diff, status = compare_values(prev_q, q)
            diff_str = format_diff(abs_diff, rel_diff, status, q.value_type)
            st = status == "CHANGED" ? "** CHANGED **" : status
            push!(rows, [short, date_str, val_str, diff_str, st])
        end
        prev_q = q
    end

    widths = [length(header[i]) for i in 1:ncols]
    for r in rows
        for i in 1:ncols
            widths[i] = max(widths[i], length(r[i]))
        end
    end
    total_w = sum(widths) + 2 * (ncols - 1)

    println()
    println("History: $qty_name — $case_name")
    println("="^total_w)
    _print_row(header, widths)
    println("-"^total_w)
    for r in rows
        _print_row(r, widths)
    end
    println("="^total_w)
end
