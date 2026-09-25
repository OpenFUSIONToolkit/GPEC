"""
Golden values: committed, reviewable reference numbers for a case.

The `--refs` comparison answers "did this change?"; it cannot answer "is this right?", because
both sides come from the same working copy of the code and nothing is recorded in git. Golden
values close that gap: one TOML file per case, tracked alongside the source, holding the value
of every quantity plus the tolerance it must be reproduced within and the evidence behind that
tolerance.

The tolerance policy is the load-bearing part. A tolerance is a claim about how much a quantity
may legitimately move, so it must come from measurement — the residual drift at a convergence
plateau and the spread across platforms — never be chosen to make a check pass. Until measured, an
entry carries a provisional class default and is stamped as such. `save_golden` and `load_golden`
refuse a tolerance below either recorded measurement, and a failing check is fixed by explaining
the physics or fixing the regression, not by widening the bound.
"""

"""
How strictly a quantity must reproduce, and why.

  - `topological` — integer counts, mode numbers, and exact inputs such as echoed scan grids;
    exact equality
  - `equilibrium_scalar` — spline/quadrature outputs with no adaptive branching; very tight
  - `physics_converged` — the quantities golden values exist for (δW, Δ′, torque, growth
    rates); tolerance comes from the measured convergence plateau and platform spread
  - `diagnostic` — ODE step counts, runtimes; recorded and reported, never gating, because they
    describe the numerics rather than the physics
  - `unconverged` — measured and found to have no plateau; tracked differentially, never pinned,
    so that a known-unconverged quantity is visibly excluded instead of quietly given a wide bound

A quantity's class comes from its case file (an explicit `class` key, else `infer_class`), never
from the golden file, so demoting a gate to a non-gating class is a reviewed case-file change.
"""
const TOLERANCE_CLASSES = ("topological", "equilibrium_scalar", "physics_converged", "diagnostic", "unconverged")

"""
Classes whose failure fails the run. `diagnostic` and `unconverged` are reported only.
"""
const GATING_CLASSES = ("topological", "equilibrium_scalar", "physics_converged")

"""
Provisional per-class tolerances, used only when no measurement has been supplied.

These are starting points, not results: a golden file written with them is stamped
`tolerance_basis = "class-default (provisional)"` so that an unmeasured pin is never mistaken
for a converged one.
"""
const CLASS_DEFAULT_TOLERANCE = Dict(
    "topological" => (rtol=0.0, atol=0.0),
    "equilibrium_scalar" => (rtol=1e-9, atol=0.0),
    "physics_converged" => (rtol=1e-6, atol=0.0),
    "diagnostic" => (rtol=Inf, atol=Inf),
    "unconverged" => (rtol=Inf, atol=Inf)
)

"""
One quantity's pinned value and the terms it is judged by.

## Fields

  - `name` — quantity key, matching the case's `[quantities.*]` table
  - `value_type` — "real", "integer", "json_array" (mirrors `ExtractedQuantity`)
  - `value_real` / `value_int` / `value_text` — the pinned value in its stored form
  - `rtol` / `atol` — pass when `|x - gold| <= atol + rtol*|gold|`
  - `class` — one of `TOLERANCE_CLASSES`
  - `tolerance_basis` — how the tolerance was arrived at: "measured" (requires a recorded
    `plateau_drift` or `platform_spread`), "class-default (provisional)", or "provisional" (no
    measurement, tolerance kept tighter than the class default after a re-pin dropped the evidence)
  - `plateau_drift` — relative movement over the final refinement of the convergence scan; NaN
    when not measured
  - `platform_spread` — relative disagreement between reference platforms; NaN when not measured
  - `converged_at` — the discretization the value was taken at, e.g.
    "grid_type=ldp, mpsi=1024, mtheta=512"
"""
struct GoldenValue
    name::String
    value_type::String
    value_real::Union{Float64,Nothing}
    value_int::Union{Int,Nothing}
    value_text::Union{String,Nothing}
    rtol::Float64
    atol::Float64
    class::String
    tolerance_basis::String
    plateau_drift::Float64
    platform_spread::Float64
    converged_at::String
end

is_gating(g::GoldenValue) = g.class in GATING_CLASSES

"""
True when a tolerance is not backed by a recorded measurement, i.e. any basis other than "measured".
"""
is_provisional(g::GoldenValue) = g.tolerance_basis != "measured"

"""
Runtimes are wall-clock and checksums have no notion of "close", so neither is ever pinned.
"""
is_pinnable(spec::QuantitySpec) = spec.type != "runtime" && spec.extract != "checksum"

"""
Refuse an entry that would gate more quietly than its evidence allows. Run on every load as well as
every save, so a hand edit or a merge taking the wrong side is caught as surely as a bad write.
"""
function validate_golden_value(g::GoldenValue, context::AbstractString)
    g.class in TOLERANCE_CLASSES || error("$context: quantity '$(g.name)' has unknown class '$(g.class)'")
    if is_gating(g)
        (isfinite(g.rtol) && g.rtol >= 0 && isfinite(g.atol) && g.atol >= 0) ||
            error("$context: gating quantity '$(g.name)' needs a finite, non-negative rtol and atol (got rtol=$(g.rtol), atol=$(g.atol))")
        for (label, m) in (("platform_spread", g.platform_spread), ("plateau_drift", g.plateau_drift))
            isfinite(m) && g.rtol < m &&
                error(
                    "$context: quantity '$(g.name)' has rtol $(g.rtol) below its measured $label $m. " *
                    "Widen it deliberately, with the measurement recorded, or reduce the $label before pinning."
                )
        end
    end
    g.tolerance_basis == "measured" && !isfinite(g.plateau_drift) && !isfinite(g.platform_spread) &&
        error("$context: quantity '$(g.name)' claims a measured tolerance but records neither plateau_drift nor platform_spread")
    return nothing
end

"""
Provenance for a whole golden file: what produced these numbers and why they last changed.
"""
struct GoldenMeta
    case::String
    golden_version::Int
    generated_at::String
    commit::String
    reason::String
    julia_version::String
    os_arch::String
    manifest_sha::String
    nthreads::Int
    blas_threads::Int
end

golden_path(case_name::AbstractString) = joinpath(GOLDEN_DIR, "$(case_name).toml")

has_golden(case_name::AbstractString) = isfile(golden_path(case_name))

"""
Read a golden file. Returns `(meta, Dict{name => GoldenValue})`, or `nothing` when absent.
"""
function load_golden(case_name::AbstractString)
    path = golden_path(case_name)
    isfile(path) || return nothing
    data = TOML.parsefile(path)
    m = get(data, "meta", Dict{String,Any}())
    meta = GoldenMeta(
        get(m, "case", String(case_name)),
        Int(get(m, "golden_version", 0)),
        get(m, "generated_at", ""),
        get(m, "commit", ""),
        get(m, "reason", ""),
        get(m, "julia_version", ""),
        get(m, "os_arch", ""),
        get(m, "manifest_sha", ""),
        Int(get(m, "nthreads", -1)),
        Int(get(m, "blas_threads", -1))
    )
    values = Dict{String,GoldenValue}()
    for (name, v) in get(data, "values", Dict{String,Any}())
        g = GoldenValue(
            name,
            get(v, "value_type", "real"),
            haskey(v, "value") && v["value"] isa Real ? Float64(v["value"]) : nothing,
            haskey(v, "value_int") ? Int(v["value_int"]) : nothing,
            haskey(v, "value_text") ? String(v["value_text"]) : nothing,
            Float64(get(v, "rtol", NaN)),
            Float64(get(v, "atol", 0.0)),
            get(v, "class", "physics_converged"),
            get(v, "tolerance_basis", "unspecified"),
            Float64(get(v, "plateau_drift", NaN)),
            Float64(get(v, "platform_spread", NaN)),
            get(v, "converged_at", "")
        )
        validate_golden_value(g, "Golden file $path")
        values[name] = g
    end
    return (meta=meta, values=values)
end

"""
Write a golden file.

Refuses any entry `validate_golden_value` rejects, and any entry with no value. Raising a
tolerance to cover a measured spread is a deliberate act the caller performs, never something this
function does behind the caller's back.
"""
function save_golden(meta::GoldenMeta, values::Dict{String,GoldenValue})
    for g in Base.values(values)
        if g.value_real === nothing && g.value_int === nothing && g.value_text === nothing
            error(
                "Quantity '$(g.name)': refusing to write a golden entry with no value (the run " *
                "produced NaN or nothing) — a valueless gating entry fails forever and regenerating " *
                "reproduces it. Exclude the quantity or fix the extraction."
            )
        end
        validate_golden_value(g, "Refusing to write golden for $(meta.case)")
    end
    mkpath(GOLDEN_DIR)
    out = Dict{String,Any}(
        "meta" => Dict{String,Any}(
            "case" => meta.case,
            "golden_version" => meta.golden_version,
            "generated_at" => meta.generated_at,
            "commit" => meta.commit,
            "reason" => meta.reason,
            "julia_version" => meta.julia_version,
            "os_arch" => meta.os_arch,
            "manifest_sha" => meta.manifest_sha,
            "nthreads" => meta.nthreads,
            "blas_threads" => meta.blas_threads
        )
    )
    vals = Dict{String,Any}()
    for (name, g) in values
        entry = Dict{String,Any}(
            "value_type" => g.value_type,
            "class" => g.class,
            "tolerance_basis" => g.tolerance_basis,
            "rtol" => g.rtol,
            "atol" => g.atol
        )
        g.value_real !== nothing && (entry["value"] = g.value_real)
        g.value_int !== nothing && (entry["value_int"] = g.value_int)
        g.value_text !== nothing && (entry["value_text"] = g.value_text)
        isfinite(g.plateau_drift) && (entry["plateau_drift"] = g.plateau_drift)
        isfinite(g.platform_spread) && (entry["platform_spread"] = g.platform_spread)
        isempty(g.converged_at) || (entry["converged_at"] = g.converged_at)
        vals[name] = entry
    end
    out["values"] = vals
    open(golden_path(meta.case), "w") do io
        TOML.print(io, out; sorted=true)
    end
    return golden_path(meta.case)
end

"""
Classify a quantity from its case spec, so a new case gets sensible defaults without every
tolerance having to be written by hand.

An explicit `class` in the case file wins. Otherwise integer counts are topological; runtimes and
step counts describe the numerics rather than the physics and so are diagnostic; everything else
is assumed to be a physics quantity that must be converged, which is the conservative assumption —
it gates.
"""
function infer_class(spec::QuantitySpec)::String
    if !isempty(spec.class)
        spec.class in TOLERANCE_CLASSES || error("Quantity '$(spec.name)': unknown class '$(spec.class)' in its case file")
        return spec.class
    end
    spec.type == "runtime" && return "diagnostic"
    spec.name in ("nstep", "nstep_total") && return "diagnostic"
    spec.type == "int_scalar" && return "topological"
    # A control token has no meaningful tolerance between "riccati" and "galerkin".
    startswith(spec.extract, "toml_key:") && return "topological"
    name = spec.name
    # sing_psi / sing_q are absent on purpose: they come from a root search, not pure quadrature.
    equilibrium_names = ("q0", "q95", "betat", "betan", "betap1", "betap2", "betap3", "betaj",
        "li1", "li2", "li3", "volume", "crnt", "bt0", "bwall", "aratio", "kappa", "psio")
    name in equilibrium_names && return "equilibrium_scalar"
    return "physics_converged"
end

"""
Compare one extracted quantity against its golden value.

Returns `(passed, deviation, detail)` where `deviation` is the relative deviation (absolute when
the golden value is zero) and `detail` is a human-readable reason on failure. Array quantities
pass only when every element is within tolerance; the reported deviation is the worst element.
"""
function compare_to_golden(q::NamedTuple, g::GoldenValue)
    # Non-gating classes are recorded, never judged; gating ones are validated finite on load.
    within = (x, gold) -> !is_gating(g) || abs(x - gold) <= g.atol + g.rtol * abs(gold)

    if q.value_type != g.value_type
        return (false, NaN, "type changed: golden $(g.value_type), got $(q.value_type)")
    end

    if g.value_type == "integer"
        (q.value_int === nothing || g.value_int === nothing) && return (false, NaN, "missing integer value")
        d = Float64(abs(q.value_int - g.value_int))
        passed = g.class == "topological" ? q.value_int == g.value_int : within(Float64(q.value_int), Float64(g.value_int))
        return (passed, g.value_int == 0 ? d : d / abs(g.value_int), passed ? "" : "$(g.value_int) → $(q.value_int)")

    elseif g.value_type == "real"
        (q.value_real === nothing || g.value_real === nothing) && return (false, NaN, "missing value")
        gold = g.value_real
        got = q.value_real
        d = abs(got - gold)
        rel = gold == 0.0 ? d : d / abs(gold)
        return (within(got, gold), rel, within(got, gold) ? "" : @sprintf("%.9g → %.9g", gold, got))

    elseif g.value_type == "json_array"
        (q.value_text === nothing || g.value_text === nothing) && return (false, NaN, "missing array")
        got = JSON.parse(q.value_text; allownan=true)
        gold = JSON.parse(g.value_text; allownan=true)
        length(got) == length(gold) && return _compare_arrays(got, gold, g)
        return (false, NaN, "length $(length(gold)) → $(length(got))")

    elseif g.value_type == "token"
        # Discrete choice, so no tolerance; deviation 1.0 (not NaN) on mismatch sorts as a failure.
        (q.value_text === nothing || g.value_text === nothing) && return (false, NaN, "missing token")
        matched = q.value_text == g.value_text
        return (matched, matched ? 0.0 : 1.0, matched ? "" : "$(g.value_text) → $(q.value_text)")
    end

    return (false, NaN, "unsupported value type $(g.value_type)")
end

"""
Worst-element comparison for array quantities, shared by the real and complex encodings.
"""
function _compare_arrays(got, gold, g::GoldenValue)
    worst_rel = 0.0
    worst_idx = 0
    all_ok = true
    for i in eachindex(gold)
        a = _json_element_abs(gold[i])
        d = _json_element_diff(gold[i], got[i])
        # A NaN on one side counts as the worst possible element, so the detail points at it.
        rel = isnan(d) ? Inf : (a == 0.0 ? d : d / a)
        ok = !is_gating(g) || d <= g.atol + g.rtol * a
        ok || (all_ok = false)
        if rel > worst_rel || (worst_idx == 0 && !ok)
            worst_rel = rel
            worst_idx = i
        end
    end
    detail = all_ok ? "" : "worst element $(worst_idx): rel $(@sprintf("%.3e", worst_rel))"
    return (all_ok, worst_rel, detail)
end

"""
Compare a case's fresh run against its golden file and print the report.

Returns `(n_pass, n_fail, n_untracked, n_informational, n_run_failed)`. Only gating classes
can fail; `diagnostic` and `unconverged` quantities are shown with their deviation and marked
as informational, so a reader sees them move without the run failing on them. A run that
crashed outright is counted in `n_run_failed`, never in `n_fail`, so the caller can exit with
a crash message instead of a tolerance message.
"""
function report_golden_check(db::SQLite.DB, case_spec::CaseSpec, commit_hash::String)
    golden = load_golden(case_spec.name)
    println()
    println("Golden Check: $(case_spec.name)")
    if golden === nothing
        println("  No golden file at $(golden_path(case_spec.name)) — nothing to check against.")
        println("  Generate one with:  regress --update-golden --cases $(case_spec.name) --reason \"...\"")
        return (n_pass=0, n_fail=0, n_untracked=0, n_informational=0, n_run_failed=0)
    end

    quantities = get_quantities(db, commit_hash, case_spec.name)
    info = get_run_info(db, commit_hash, case_spec.name)
    if info !== nothing && !info.success
        println("  RUN FAILED — nothing to compare (this is a crash, not a tolerance failure):")
        println("  $(_short_err(info.error_msg))")
        return (n_pass=0, n_fail=0, n_untracked=0, n_informational=0, n_run_failed=1)
    end

    rows = Vector{Vector{String}}()
    n_pass = n_fail = n_untracked = n_informational = n_provisional = 0

    for spec in case_spec.quantities
        g = get(golden.values, spec.name, nothing)
        if g === nothing
            is_pinnable(spec) || continue
            n_untracked += 1
            continue
        end
        q_raw = get(quantities, spec.name, nothing)
        if q_raw === nothing
            push!(rows, [spec.label, g.class, "MISSING", "—", "—", "—", "FAIL"])
            n_fail += 1
            continue
        end
        # The golden file's class must be the one the case declares, so a demotion cannot hide there.
        declared = infer_class(spec)
        if g.class != declared
            push!(rows, [spec.label, g.class, "CLASS", "—", "—", "—", "** FAIL **  case declares $declared; regenerate goldens"])
            n_fail += 1
            continue
        end
        # SQLite NULLs surface as `missing`; normalize them to nothing as the update path does.
        q = (label=q_raw.label, value_real=_column(q_raw.value_real, nothing),
            value_int=_column(q_raw.value_int, nothing), value_text=_column(q_raw.value_text, nothing),
            value_type=q_raw.value_type, noise_threshold=q_raw.noise_threshold)
        passed, deviation, detail = compare_to_golden(q, g)
        gating = is_gating(g)
        status = if !gating
            "info"
        elseif passed
            "ok"
        else
            "** FAIL **"
        end
        if !gating
            n_informational += 1
        elseif passed
            n_pass += 1
        else
            n_fail += 1
        end
        # A scalar with a zero golden value is judged absolutely, so its deviation is marked as such.
        zero_gold = (g.value_type == "real" && g.value_real == 0.0) || (g.value_type == "integer" && g.value_int == 0)
        dev = isnan(deviation) ? "—" : @sprintf("%.2e", deviation) * (zero_gold ? " (abs)" : "")
        rtol = isfinite(g.rtol) ? @sprintf("%.1e", g.rtol) : "—"
        atol = isfinite(g.atol) ? @sprintf("%.1e", g.atol) : "—"
        basis = !gating ? "—" : is_provisional(g) ? "provisional" : "measured"
        gating && is_provisional(g) && (n_provisional += 1)
        push!(rows, [spec.label, g.class, dev, rtol, atol, basis, status * (isempty(detail) ? "" : "  $detail")])
    end

    # An orphaned entry fails regardless of class: a renamed quantity must not silently drop its gate.
    spec_names = Set(spec.name for spec in case_spec.quantities)
    for name in sort(collect(keys(golden.values)))
        name in spec_names && continue
        push!(rows, [name, golden.values[name].class, "ORPHANED", "—", "—", "—",
            "** FAIL **  no matching quantity in the case — renamed or removed? Regenerate goldens."])
        n_fail += 1
    end

    header = ["Quantity", "Class", "Deviation", "rtol", "atol", "Basis", "Status"]
    widths = [length(h) for h in header]
    for row in rows, i in eachindex(row)
        widths[i] = max(widths[i], length(row[i]))
    end
    total_w = sum(widths) + 2 * (length(header) - 1)

    println("="^total_w)
    println("Golden v$(golden.meta.golden_version), generated $(golden.meta.generated_at) @ $(golden.meta.commit)")
    isempty(golden.meta.reason) || println("Reason: $(golden.meta.reason)")
    println("Golden env: julia $(golden.meta.julia_version), $(golden.meta.os_arch), $(golden.meta.nthreads) thread/$(golden.meta.blas_threads) BLAS")
    if info !== nothing && !isempty(info.fingerprint.julia_version)
        println("This run:   $(describe_env(info.fingerprint))")
        info.fingerprint.os_arch == golden.meta.os_arch ||
            println("  NOTE: different platform from the one the goldens were measured on; deviations at or below the recorded platform_spread are expected.")
    end
    println("-"^total_w)
    _print_row(header, widths)
    println("-"^total_w)
    for row in rows
        _print_row(row, widths)
    end
    println("="^total_w)
    parts = String[]
    n_pass > 0 && push!(parts, "$n_pass pass")
    n_fail > 0 && push!(parts, "$n_fail FAIL")
    n_informational > 0 && push!(parts, "$n_informational informational")
    n_untracked > 0 && push!(parts, "$n_untracked untracked")
    println("Summary: ", join(parts, ", "))
    n_provisional > 0 && println(
        "  $n_provisional gating quantity/quantities are judged against provisional class-default tolerances, " *
        "not a measured plateau drift or platform spread."
    )
    println()
    return (n_pass=n_pass, n_fail=n_fail, n_untracked=n_untracked, n_informational=n_informational, n_run_failed=0)
end

"""
Uncommitted changes to tracked files under `repo_root`, ignoring `exclude_dir` (the goldens themselves).
"""
function uncommitted_changes(repo_root::AbstractString, exclude_dir::AbstractString)
    rel = relpath(exclude_dir, repo_root)
    return String(strip(read(`git -C $repo_root status --porcelain --untracked-files=no -- . ":(exclude)$rel"`, String)))
end

"""
Regenerate a case's golden file from a completed run, reporting what moved.

Prints an old→new delta for every quantity whose value or class changed, flagging any move that
the old tolerance would have failed, so the diff a reviewer sees in git comes with the size and
significance of each move. Refuses a working tree with uncommitted changes outside the golden
directory, because the recorded commit is what a reviewer uses to reproduce a disputed number.
"""
function update_golden_from_run(db::SQLite.DB, case_spec::CaseSpec, commit_hash::String,
    reason::String, repo_root::String)
    info = get_run_info(db, commit_hash, case_spec.name)
    (info === nothing || !info.success) && error("Cannot update goldens for '$(case_spec.name)': the run did not succeed")
    if commit_hash == LOCAL_REF
        dirty = uncommitted_changes(repo_root, GOLDEN_DIR)
        isempty(dirty) || error(
            "Cannot update goldens for '$(case_spec.name)': the working tree has uncommitted " *
            "changes, so the recorded commit would not reproduce these values. Commit first.\n$dirty"
        )
    end

    extracted = ExtractedQuantity[]
    for (name, q) in get_quantities(db, commit_hash, case_spec.name)
        push!(
            extracted,
            ExtractedQuantity(name, String(_column(q.label, name)),
                _column(q.value_real, nothing), _column(q.value_int, nothing),
                _column(q.value_text, nothing), q.value_type, q.noise_threshold)
        )
    end

    previous = load_golden(case_spec.name)
    existing = previous === nothing ? nothing : previous.values
    values = build_golden_values(extracted, case_spec.quantities, existing)

    fp = info.fingerprint
    meta = GoldenMeta(case_spec.name,
        previous === nothing ? 1 : previous.meta.golden_version + 1,
        Dates.format(Dates.now(), "yyyy-mm-dd"),
        strip(read(`git -C $repo_root rev-parse --short $(commit_hash == LOCAL_REF ? "HEAD" : commit_hash)`, String)),
        reason, fp.julia_version, fp.os_arch, fp.manifest_sha, fp.nthreads, fp.blas_threads)

    println()
    println("Golden update: $(case_spec.name)  (v$(previous === nothing ? 0 : previous.meta.golden_version) → v$(meta.golden_version))")
    if existing !== nothing
        unpinnable = Set(s.name for s in case_spec.quantities if !is_pinnable(s))
        for name in sort(collect(keys(existing)))
            if name in unpinnable
                println(@sprintf("  %-34s REMOVED — runtimes and checksums are not pinned", name))
            elseif !haskey(values, name)
                println(
                    @sprintf(
                        "  %-34s REMOVED — extraction returned missing (renamed h5 path?) or the quantity left the case. This deletes its gate; confirm it is intentional.",
                        name
                    )
                )
            end
        end
        for (name, g) in sort(collect(values); by=first)
            old = get(existing, name, nothing)
            old === nothing && (println(@sprintf("  %-34s NEW", name)); continue)
            old.class == g.class ||
                println(@sprintf("  %-34s class %s → %s (tolerances reset to provisional)", name, old.class, g.class))
            line = describe_golden_change(old, g)
            line === nothing || println(@sprintf("  %-34s %s", name, line))
        end
    end
    path = save_golden(meta, values)
    println("Wrote $path  ($(length(values)) quantities)")
    provisional = count(g -> is_gating(g) && is_provisional(g), Base.values(values))
    provisional > 0 && println(
        "  $provisional quantity/quantities still carry provisional class-default tolerances — " *
        "replace them with measured plateau drift and platform spread before relying on this as a gate."
    )
    println()
    return path
end

"""
Describe how a regenerated entry's value moved from the old one, or `nothing` if it did not.

The deviation is the one `compare_to_golden` computes against the old entry, and a move the old
tolerance would have failed is flagged so it cannot pass unnoticed as a routine re-pin.
"""
function describe_golden_change(old::GoldenValue, new::GoldenValue)
    old.value_type == new.value_type || return "type $(old.value_type) → $(new.value_type)"
    (old.value_real, old.value_int, old.value_text) == (new.value_real, new.value_int, new.value_text) && return nothing
    q = (value_real=new.value_real, value_int=new.value_int, value_text=new.value_text, value_type=new.value_type)
    passed, dev, _ = compare_to_golden(q, old)
    what = if new.value_type == "real"
        @sprintf("%.9g → %.9g", something(old.value_real, NaN), something(new.value_real, NaN))
    elseif new.value_type == "integer"
        "$(old.value_int) → $(new.value_int)"
    elseif new.value_type == "token"
        "$(old.value_text) → $(new.value_text)"
    else
        "array[$(length(JSON.parse(new.value_text; allownan=true)))] changed"
    end
    rel = isnan(dev) ? "—" : @sprintf("%.2e", dev)
    flag = is_gating(old) && !passed ? "   ** EXCEEDS the old tolerance: a regression unless --reason explains it **" : ""
    return "$what   (rel $rel)$flag"
end

"""
Build golden entries from a set of freshly extracted quantities.

`existing` carries forward the tolerance and evidence already recorded for a quantity, so
regenerating values after a physics change does not silently reset hard-won measurements to
provisional defaults. A change of value type or of the case-declared class invalidates them. So
does a move beyond the old gating tolerance: the evidence was measured on a different value, so it
is dropped and the entry becomes provisional, with a tolerance no looser than before.
"""
function build_golden_values(extracted::Vector{ExtractedQuantity}, specs::Vector{QuantitySpec},
    existing::Union{Dict{String,GoldenValue},Nothing})
    spec_by_name = Dict(s.name => s for s in specs)
    values = Dict{String,GoldenValue}()
    for eq in extracted
        eq.value_type == "missing" && continue
        spec = get(spec_by_name, eq.name, nothing)
        spec === nothing && continue
        is_pinnable(spec) || continue
        class = infer_class(spec)
        prior = existing === nothing ? nothing : get(existing, eq.name, nothing)
        if prior !== nothing && prior.value_type != eq.value_type
            @warn "Golden '$(eq.name)': value_type changed $(prior.value_type) → $(eq.value_type); resetting tolerances to provisional"
            prior = nothing
        elseif prior !== nothing && prior.class != class
            @warn "Golden '$(eq.name)': class changed $(prior.class) → $class; resetting tolerances to provisional"
            prior = nothing
        end
        defaults = CLASS_DEFAULT_TOLERANCE[class]
        run_value = (value_real=eq.value_real, value_int=eq.value_int, value_text=eq.value_text, value_type=eq.value_type)
        if prior === nothing
            rtol, atol, basis = defaults.rtol, defaults.atol, "class-default (provisional)"
            drift, spread, at = NaN, NaN, ""
        elseif is_gating(prior) && !compare_to_golden(run_value, prior)[1]
            @warn "Golden '$(eq.name)': value moved beyond its old tolerance; dropping its recorded evidence and marking it provisional"
            rtol, atol = min(prior.rtol, defaults.rtol), min(prior.atol, defaults.atol)
            basis = (rtol, atol) == (defaults.rtol, defaults.atol) ? "class-default (provisional)" : "provisional"
            drift, spread, at = NaN, NaN, ""
        else
            rtol, atol, basis = prior.rtol, prior.atol, prior.tolerance_basis
            drift, spread, at = prior.plateau_drift, prior.platform_spread, prior.converged_at
        end
        values[eq.name] = GoldenValue(eq.name, eq.value_type, eq.value_real, eq.value_int,
            eq.value_text, rtol, atol, class, basis, drift, spread, at)
    end
    return values
end
