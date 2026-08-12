"""
Reporter: formats comparison tables for stdout output.
"""

"""
Compress a stored error message into a single-line snippet for the report
banner. Picks the most informative line (a Julia ERROR/UndefVarError if
present) and truncates to keep the report compact.
"""
function _short_err(msg)::String
    (msg === nothing || msg === missing) && return "(no message)"
    s = String(msg)
    isempty(strip(s)) && return "(empty)"
    lines = filter(!isempty, strip.(split(s, '\n')))
    isempty(lines) && return "(empty)"
    pick = something(findfirst(l -> startswith(l, "ERROR:") || occursin("Error", l), lines),
        length(lines))
    line = lines[pick]
    return length(line) > 200 ? line[1:200] * "..." : line
end

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
        arr = JSON.parse(t; allownan=true)
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
Print a banner when two compared runs did not share an environment.

Source code is only the sole variable when Julia, host, package set and thread counts all match.
Anything else here means part of the reported difference may be the environment, not the code.
"""
function _warn_env_difference(fp1::EnvFingerprint, fp2::EnvFingerprint)
    (fp1 === UNKNOWN_ENV || fp2 === UNKNOWN_ENV) && return
    (isempty(fp1.julia_version) || isempty(fp2.julia_version)) && return
    differences = String[]
    fp1.julia_version != fp2.julia_version && push!(differences, "julia $(fp1.julia_version) vs $(fp2.julia_version)")
    fp1.os_arch != fp2.os_arch && push!(differences, "host $(fp1.os_arch) vs $(fp2.os_arch)")
    fp1.manifest_sha != fp2.manifest_sha && push!(differences, "different package sets (Manifest hashes differ)")
    fp1.nthreads != fp2.nthreads && push!(differences, "$(fp1.nthreads) vs $(fp2.nthreads) Julia threads")
    fp1.blas_threads != fp2.blas_threads && push!(differences, "$(fp1.blas_threads) vs $(fp2.blas_threads) BLAS threads")
    isempty(differences) && return
    println()
    println("!! ENVIRONMENTS DIFFER — source code is not the only variable in this comparison:")
    for d in differences
        println("     - $d")
    end
    println("   Differences below may be environment artifacts. Re-run with --force to rebuild")
    println("   both refs in the current environment.")
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
Helper: print a row of padded columns.
"""
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

    failed1 = !info1.success
    failed2 = !info2.success

    q1_all = failed1 ? Dict{String,NamedTuple}() :
             get_quantities(db, ref1.commit_hash, case_spec.name)
    q2_all = failed2 ? Dict{String,NamedTuple}() :
             get_quantities(db, ref2.commit_hash, case_spec.name)

    # Pre-compute all rows: (label, v1, v2, diff, status)
    header = ["Quantity", ref1.name, ref2.name, "Diff", "Status"]
    rows = Vector{Vector{String}}()
    n_ok = 0
    n_changed = 0
    n_missing = 0

    # Helper: pick the value-cell text for one ref/quantity, accounting for
    # whole-ref failure (which should render as "FAILED" rather than "N/A").
    cell = (failed::Bool, qs::Dict, qname::String) -> begin
        failed && return "FAILED"
        return haskey(qs, qname) ? format_value(qs[qname]) : "N/A"
    end
    runtime_cell = (failed::Bool, qs::Dict, qname::String) -> begin
        failed && return "FAILED"
        (haskey(qs, qname) && qs[qname].value_real !== nothing) ?
        @sprintf("%.1fs", qs[qname].value_real) : "N/A"
    end

    for spec in case_spec.quantities
        qname = spec.name

        # Skip silently only when neither ref has the quantity AND neither failed
        # (a failed ref still gets a row with FAILED markers).
        if !failed1 && !failed2 && !haskey(q1_all, qname) && !haskey(q2_all, qname)
            n_missing += 1
            continue
        end

        if spec.type == "runtime"
            v1 = runtime_cell(failed1, q1_all, qname)
            v2 = runtime_cell(failed2, q2_all, qname)
            push!(rows, [spec.label, v1, v2, "", "--"])
            continue
        end

        has1 = haskey(q1_all, qname)
        has2 = haskey(q2_all, qname)
        if failed1 || failed2 || !has1 || !has2
            v1 = cell(failed1, q1_all, qname)
            v2 = cell(failed2, q2_all, qname)
            status_label = (failed1 || failed2) ? "FAILED" : "N/A"
            push!(rows, [spec.label, v1, v2, "", status_label])
            n_missing += 1
            continue
        end

        q1 = q1_all[qname]
        q2 = q2_all[qname]
        abs_diff, rel_diff, status = compare_values(q1, q2)
        st = status == "CHANGED" ? "** CHANGED **" : status
        push!(rows, [spec.label, format_value(q1), format_value(q2),
            format_diff(abs_diff, rel_diff, status, q1.value_type), st])

        if status == "OK"
            n_ok += 1
        elseif contains(status, "CHANGED")
            n_changed += 1
        else
            n_missing += 1
        end
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
    tag1 = failed1 ? " (FAILED)" : ""
    tag2 = failed2 ? " (FAILED)" : ""
    println("Ref 1: $(ref1.name)  @ $(info1.commit_short) ($date1)$tag1")
    println("       env: $(describe_env(info1.fingerprint))")
    println("Ref 2: $(ref2.name)  @ $(info2.commit_short) ($date2)$tag2")
    println("       env: $(describe_env(info2.fingerprint))")
    _warn_env_difference(info1.fingerprint, info2.fingerprint)
    if failed1
        println("  Ref 1 error: $(_short_err(info1.error_msg))")
    end
    if failed2
        println("  Ref 2 error: $(_short_err(info2.error_msg))")
    end
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
    return (n_ok=n_ok, n_changed=n_changed, n_missing=n_missing,
            n_failed=count(identity, (failed1, failed2)))
end

"""
Print a multi-ref tracking report for a single case.
One row per quantity, one column per ref.
"""
function report_multi_ref(db::SQLite.DB, case_spec::CaseSpec,
    refs::Vector{ResolvedRef})
    run_infos = [get_run_info(db, ref.commit_hash, case_spec.name) for ref in refs]
    failed_mask = [info === nothing || !info.success for info in run_infos]
    all_qs = [
        begin
            info = run_infos[i]
            (info !== nothing && info.success) ? get_quantities(db, refs[i].commit_hash, case_spec.name) : Dict{String,NamedTuple}()
        end for i in eachindex(refs)
    ]

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
    n_ok = 0
    n_changed = 0
    n_missing = 0

    for spec in case_spec.quantities
        qname = spec.name
        any_failed = any(failed_mask)
        any_present = any(haskey(qs, qname) for qs in all_qs)
        # Skip silently only when nothing is present and nothing failed.
        if !any_failed && !any_present
            n_missing += 1
            continue
        end

        row = [spec.label]

        if spec.type == "runtime"
            for (i, qs) in enumerate(all_qs)
                val = if failed_mask[i]
                    "FAILED"
                elseif haskey(qs, qname) && qs[qname].value_real !== nothing
                    @sprintf("%.1fs", qs[qname].value_real)
                else
                    "N/A"
                end
                push!(row, val)
            end
            if show_diff
                push!(row, "")
                push!(row, "--")
            end
            push!(rows, row)
            continue
        end

        for (i, qs) in enumerate(all_qs)
            if failed_mask[i]
                push!(row, "FAILED")
            else
                push!(row, haskey(qs, qname) ? format_value(qs[qname]) : "N/A")
            end
        end

        if show_diff
            last_failed = failed_mask[end]
            prev_failed = failed_mask[end-1]
            q_last = haskey(all_qs[end], qname) ? all_qs[end][qname] : nothing
            q_prev = haskey(all_qs[end-1], qname) ? all_qs[end-1][qname] : nothing
            if q_last !== nothing && q_prev !== nothing
                abs_diff, rel_diff, status = compare_values(q_prev, q_last)
                st = status == "CHANGED" ? "** CHANGED **" : status
                push!(row, format_diff(abs_diff, rel_diff, status, q_prev.value_type))
                push!(row, st)
                if status == "OK"
                    n_ok += 1
                elseif contains(status, "CHANGED")
                    n_changed += 1
                else
                    n_missing += 1
                end
            elseif last_failed || prev_failed
                push!(row, "")
                push!(row, "FAILED")
                n_missing += 1
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
            println("       env: $(describe_env(info.fingerprint))")
            if !info.success
                println("  Ref $(i) error: $(_short_err(info.error_msg))")
            end
        else
            println("Ref $(i): $(ref.name)  (no data)")
        end
    end
    if length(refs) >= 2
        last_two = filter(!isnothing, run_infos[(end - 1):end])
        length(last_two) == 2 && _warn_env_difference(last_two[1].fingerprint, last_two[2].fingerprint)
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
    return (n_ok=n_ok, n_changed=n_changed, n_missing=n_missing,
            n_failed=count(identity, failed_mask))
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
