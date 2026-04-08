"""
Reporter: formats comparison tables for stdout output.
"""

"""
Format a value for display in a table column.
"""
function format_value(q::NamedTuple, max_width::Int=14)::String
    if q.value_type == "missing"
        return rpad("N/A", max_width)
    elseif q.value_type == "real"
        v = q.value_real
        v === nothing && return rpad("N/A", max_width)
        return rpad(@sprintf("%.6e", v), max_width)
    elseif q.value_type == "integer"
        v = q.value_int
        v === nothing && return rpad("N/A", max_width)
        return rpad(string(v), max_width)
    elseif q.value_type == "json_array"
        t = q.value_text
        t === nothing && return rpad("N/A", max_width)
        arr = JSON.parse(t)
        return rpad("[$(length(arr)) elements]", max_width)
    elseif q.value_type == "checksum"
        t = q.value_text
        t === nothing && return rpad("N/A", max_width)
        return rpad(t[1:min(12, length(t))] * "...", max_width)
    else
        return rpad("?", max_width)
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

"""
Print a two-ref comparison report for a single case.
"""
function report_two_ref_comparison(db::SQLite.DB, case_spec::CaseSpec,
                                   ref1::ResolvedRef, ref2::ResolvedRef)
    info1 = get_run_info(db, ref1.commit_hash, case_spec.name)
    info2 = get_run_info(db, ref2.commit_hash, case_spec.name)

    println()
    println("Regression Report: $(case_spec.name)")
    println("="^80)

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

    # Header
    date1 = length(info1.commit_date) >= 10 ? info1.commit_date[1:10] : info1.commit_date
    date2 = length(info2.commit_date) >= 10 ? info2.commit_date[1:10] : info2.commit_date
    println("Ref 1: $(rpad(ref1.name, 16)) @ $(info1.commit_short) ($date1)")
    println("Ref 2: $(rpad(ref2.name, 16)) @ $(info2.commit_short) ($date2)")
    println("-"^80)

    # Column headers
    label_w = 24
    val_w = 14
    @printf("%-*s  %-*s  %-*s  %-20s  %s\n",
            label_w, "Quantity", val_w, ref1.name, val_w, ref2.name, "Diff", "Status")
    println("-"^80)

    # Get quantities
    q1_all = get_quantities(db, ref1.commit_hash, case_spec.name)
    q2_all = get_quantities(db, ref2.commit_hash, case_spec.name)

    n_ok = 0
    n_changed = 0
    n_missing = 0

    for spec in case_spec.quantities
        qname = spec.name
        label = length(spec.label) > label_w ? spec.label[1:label_w-1] * "~" : spec.label

        if !haskey(q1_all, qname) && !haskey(q2_all, qname)
            n_missing += 1
            continue
        end

        # Handle runtime separately (always show, never flag)
        if spec.type == "runtime"
            v1_str = if haskey(q1_all, qname) && q1_all[qname].value_real !== nothing
                @sprintf("%.1fs", q1_all[qname].value_real)
            else
                "N/A"
            end
            v2_str = if haskey(q2_all, qname) && q2_all[qname].value_real !== nothing
                @sprintf("%.1fs", q2_all[qname].value_real)
            else
                "N/A"
            end
            @printf("%-*s  %-*s  %-*s  %-20s  %s\n",
                    label_w, label, val_w, v1_str, val_w, v2_str, "", "--")
            continue
        end

        if !haskey(q1_all, qname)
            @printf("%-*s  %-*s  %-*s  %-20s  %s\n",
                    label_w, label, val_w, "N/A", val_w,
                    haskey(q2_all, qname) ? format_value(q2_all[qname], val_w) : "N/A",
                    "", "N/A")
            n_missing += 1
            continue
        end
        if !haskey(q2_all, qname)
            @printf("%-*s  %-*s  %-*s  %-20s  %s\n",
                    label_w, label, val_w, format_value(q1_all[qname], val_w), val_w, "N/A",
                    "", "N/A")
            n_missing += 1
            continue
        end

        q1 = q1_all[qname]
        q2 = q2_all[qname]
        abs_diff, rel_diff, status = compare_values(q1, q2)

        v1_str = format_value(q1, val_w)
        v2_str = format_value(q2, val_w)
        diff_str = format_diff(abs_diff, rel_diff, status, q1.value_type)

        # Status indicator with emphasis for CHANGED
        status_str = status == "CHANGED" ? "** CHANGED **" : status

        @printf("%-*s  %-*s  %-*s  %-20s  %s\n",
                label_w, label, val_w, v1_str, val_w, v2_str, diff_str, status_str)

        if status == "OK"
            n_ok += 1
        elseif status == "CHANGED" || contains(status, "CHANGED")
            n_changed += 1
        else
            n_missing += 1
        end
    end

    println("="^80)
    parts = String[]
    n_changed > 0 && push!(parts, "$n_changed changed")
    n_ok > 0 && push!(parts, "$n_ok unchanged")
    n_missing > 0 && push!(parts, "$n_missing missing/N/A")
    println("Summary: ", join(parts, ", "))
    println()
end

"""
Print a multi-ref tracking report for a single case.
Shows each quantity's value across all refs.
"""
function report_multi_ref(db::SQLite.DB, case_spec::CaseSpec,
                          refs::Vector{ResolvedRef})
    println()
    println("Regression Tracking: $(case_spec.name)")
    println("="^80)

    # Collect all run info
    run_infos = []
    for ref in refs
        info = get_run_info(db, ref.commit_hash, case_spec.name)
        push!(run_infos, info)
    end

    # Print per-quantity tables
    for spec in case_spec.quantities
        spec.type == "runtime" && continue  # skip runtime in multi-ref

        println()
        println("Quantity: $(spec.label)")
        println("-"^80)
        @printf("%-10s  %-12s  %-20s  %-20s  %s\n",
                "Ref", "Date", "Value", "Δ from prev", "Status")
        println("-"^80)

        prev_q = nothing
        for (i, ref) in enumerate(refs)
            info = run_infos[i]
            is_success = info !== nothing && info.success
            if !is_success
                status_str = info === nothing ? "NO DATA" : "FAILED"
                @printf("%-10s  %-12s  %-20s  %-20s  %s\n",
                        ref.name[1:min(10, length(ref.name))], "", "", "", status_str)
                prev_q = nothing
                continue
            end

            qs = get_quantities(db, ref.commit_hash, case_spec.name)
            if !haskey(qs, spec.name)
                @printf("%-10s  %-12s  %-20s  %-20s  %s\n",
                        ref.name[1:min(10, length(ref.name))],
                        info.commit_date[1:min(10, length(info.commit_date))],
                        "N/A", "", "N/A")
                prev_q = nothing
                continue
            end

            q = qs[spec.name]
            val_str = format_value(q, 20)
            date_str = length(info.commit_date) >= 10 ? info.commit_date[1:10] : info.commit_date
            ref_str = ref.name[1:min(10, length(ref.name))]

            if prev_q === nothing
                @printf("%-10s  %-12s  %-20s  %-20s  %s\n",
                        ref_str, date_str, val_str, "--", "--")
            else
                abs_diff, rel_diff, status = compare_values(prev_q, q)
                diff_str = format_diff(abs_diff, rel_diff, status, q.value_type)
                status_str = status == "CHANGED" ? "** CHANGED **" : status
                @printf("%-10s  %-12s  %-20s  %-20s  %s\n",
                        ref_str, date_str, val_str, diff_str, status_str)
            end
            prev_q = q
        end
    end

    println()
    println("="^80)
end

"""
Show the history of a single quantity for a case (--show mode).
Queries all cached runs for the case.
"""
function show_quantity_history(db::SQLite.DB, case_name::String, qty_name::String)
    runs = get_all_runs_for_case(db, case_name)
    if isempty(runs)
        println("No cached runs found for case '$case_name'")
        return
    end

    println()
    println("History: $qty_name — $case_name")
    println("="^80)
    @printf("%-10s  %-12s  %-20s  %-20s  %s\n",
            "Commit", "Date", "Value", "Δ from prev", "Status")
    println("-"^80)

    prev_q = nothing
    for row in runs
        if coalesce(row.success, 0) != 1
            @printf("%-10s  %-12s  %-20s  %-20s  %s\n",
                    row.commit_short, "", "", "", "FAILED")
            prev_q = nothing
            continue
        end

        qs = get_quantities(db, row.commit_hash, case_name)
        if !haskey(qs, qty_name)
            @printf("%-10s  %-12s  %-20s  %-20s  %s\n",
                    row.commit_short,
                    length(row.commit_date) >= 10 ? row.commit_date[1:10] : row.commit_date,
                    "N/A", "", "N/A")
            prev_q = nothing
            continue
        end

        q = qs[qty_name]
        val_str = format_value(q, 20)
        date_str = length(row.commit_date) >= 10 ? row.commit_date[1:10] : row.commit_date

        if prev_q === nothing
            @printf("%-10s  %-12s  %-20s  %-20s  %s\n",
                    row.commit_short, date_str, val_str, "--", "--")
        else
            abs_diff, rel_diff, status = compare_values(prev_q, q)
            diff_str = format_diff(abs_diff, rel_diff, status, q.value_type)
            status_str = status == "CHANGED" ? "** CHANGED **" : status
            @printf("%-10s  %-12s  %-20s  %-20s  %s\n",
                    row.commit_short, date_str, val_str, diff_str, status_str)
        end
        prev_q = q
    end

    println("="^80)
end
