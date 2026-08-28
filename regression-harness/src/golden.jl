"""
Golden values: committed, reviewable reference numbers for a case.

The `--refs` comparison answers "did this change?"; it cannot answer "is this right?", because
both sides come from the same working copy of the code and nothing is recorded in git. Golden
values close that gap: one TOML file per case, tracked alongside the source, holding the value
of every quantity plus the tolerance it must be reproduced within and the evidence behind that
tolerance.

The tolerance policy is the load-bearing part. A tolerance is a claim about how much a quantity
may legitimately move, so it is *derived* — from the residual drift at a convergence plateau and
from the spread measured across platforms — never chosen to make a check pass. `update_golden`
refuses to write a tolerance below the recorded platform spread, and a failing check is fixed by
explaining the physics or fixing the regression, not by widening the bound.
"""

"""
How strictly a quantity must reproduce, and why.

  - `topological` — integer counts and mode numbers; exact equality
  - `equilibrium_scalar` — spline/quadrature outputs with no adaptive branching; very tight
  - `physics_converged` — the quantities golden values exist for (δW, Δ′, torque, growth
    rates); tolerance comes from the measured convergence plateau and platform spread
  - `diagnostic` — ODE step counts, runtimes; recorded and reported, never gating, because they
    describe the numerics rather than the physics
  - `unconverged` — measured and found to have no plateau; tracked differentially, never pinned,
    so that a known-unconverged quantity is visibly excluded instead of quietly given a wide bound
"""
const TOLERANCE_CLASSES = ("topological", "equilibrium_scalar", "physics_converged", "diagnostic", "unconverged")

"""Classes whose failure fails the run. `diagnostic` and `unconverged` are reported only."""
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
  - `tolerance_basis` — how the tolerance was arrived at: "measured" or
    "class-default (provisional)"
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
        class = get(v, "class", "physics_converged")
        class in TOLERANCE_CLASSES || error("Golden file $path: quantity '$name' has unknown class '$class'")
        # Enforced on every load, not only at write time: a hand edit or a merge taking the
        # wrong side must not produce a gate quieter than the one save_golden refused to write.
        if class in GATING_CLASSES
            rt = Float64(get(v, "rtol", NaN))
            (isfinite(rt) && rt >= 0) || error("Golden file $path: gating quantity '$name' has no finite rtol — malformed or hand-edited entry")
            sp = Float64(get(v, "platform_spread", NaN))
            isfinite(sp) && rt < sp && error("Golden file $path: quantity '$name' has rtol $rt below its recorded platform_spread $sp")
        end
        values[name] = GoldenValue(
            name,
            get(v, "value_type", "real"),
            haskey(v, "value") && v["value"] isa Real ? Float64(v["value"]) : nothing,
            haskey(v, "value_int") ? Int(v["value_int"]) : nothing,
            haskey(v, "value_text") ? String(v["value_text"]) : nothing,
            Float64(get(v, "rtol", NaN)),
            Float64(get(v, "atol", 0.0)),
            class,
            get(v, "tolerance_basis", "unspecified"),
            Float64(get(v, "plateau_drift", NaN)),
            Float64(get(v, "platform_spread", NaN)),
            get(v, "converged_at", "")
        )
    end
    return (meta=meta, values=values)
end

"""
Write a golden file.

Refuses to emit a tolerance tighter than a recorded platform spread: a pin that no second
platform can reproduce is a broken gate, and silently loosening it later is exactly the failure
this whole mechanism exists to prevent. Raising the tolerance to cover a measured spread is a
deliberate act the caller performs, not something this function does behind the caller's back.
"""
function save_golden(meta::GoldenMeta, values::Dict{String,GoldenValue})
    for g in Base.values(values)
        if g.value_real === nothing && g.value_int === nothing && g.value_text === nothing
            error("Quantity '$(g.name)': refusing to write a golden entry with no value (the run " *
                  "produced NaN or nothing) — a valueless gating entry fails forever and regenerating " *
                  "reproduces it. Exclude the quantity or fix the extraction.")
        end
        if isfinite(g.platform_spread) && isfinite(g.rtol) && g.rtol < g.platform_spread
            error("Quantity '$(g.name)': rtol $(g.rtol) is tighter than the measured platform " *
                  "spread $(g.platform_spread). Widen it deliberately, with the measurement recorded, or " *
                  "reduce the spread before pinning.")
        end
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

Integer counts are topological; runtimes and step counts describe the numerics rather than the
physics and so are diagnostic; everything else is assumed to be a physics quantity that must be
converged, which is the conservative assumption — it gates.
"""
function infer_class(spec::QuantitySpec)::String
    spec.type == "runtime" && return "diagnostic"
    spec.name in ("nstep", "nstep_total") && return "diagnostic"
    spec.type == "int_scalar" && return "topological"
    # A declared control token is an exact-match gate, like a count: there is no tolerance
    # that means anything between "riccati" and "galerkin".
    startswith(spec.extract, "toml_key:") && return "topological"
    name = spec.name
    # sing_psi / sing_q are deliberately absent: singular-surface locations come from a root
    # search, not pure spline/quadrature, so they take the measured physics_converged path.
    equilibrium_names = ("q0", "q95", "betat", "betan", "betap1", "betap2", "betap3", "betaj",
        "li1", "li2", "li3", "volume", "crnt", "bt0", "bwall", "aratio", "kappa")
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
    # Non-finite tolerances mean "recorded, never judged" (diagnostic/unconverged); without this
    # guard, gold == 0 turns atol + rtol*|gold| into Inf + NaN and the comparison is false.
    within = (x, gold) -> !isfinite(g.atol) || !isfinite(g.rtol) ||
        abs(x - gold) <= g.atol + g.rtol * abs(gold)

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
        # A token names a discrete choice (which integrator produced the Δ′), so there is no
        # tolerance to apply: it matches or the run is answering a different question than the
        # gold does. Deviation is reported as 1.0 on mismatch rather than NaN so it sorts as a
        # real failure in reports.
        (q.value_text === nothing || g.value_text === nothing) && return (false, NaN, "missing token")
        matched = q.value_text == g.value_text
        return (matched, matched ? 0.0 : 1.0, matched ? "" : "$(g.value_text) → $(q.value_text)")
    end

    return (false, NaN, "unsupported value type $(g.value_type)")
end

"""Worst-element comparison for array quantities, shared by the real and complex encodings."""
function _compare_arrays(got, gold, g::GoldenValue)
    worst_rel = 0.0
    worst_idx = 0
    all_ok = true
    for i in eachindex(gold)
        a = _json_element_abs(gold[i])
        d = _json_element_diff(gold[i], got[i])
        rel = a == 0.0 ? d : d / a
        # Same non-finite guard as the scalar path: Inf tolerances mean "recorded, never
        # judged", and Inf*0 = NaN would otherwise mark a zero gold element as failing.
        ok = !isfinite(g.atol) || !isfinite(g.rtol) || d <= g.atol + g.rtol * a
        ok || (all_ok = false)
        if rel > worst_rel
            worst_rel = rel
            worst_idx = i
        end
    end
    detail = all_ok ? "" : "worst element $(worst_idx): rel $(@sprintf("%.3e", worst_rel))"
    return (all_ok, worst_rel, detail)
end

"""
Compare a case's fresh run against its golden file and print the report.

Returns `(n_pass, n_fail, n_untracked, n_informational)`. Only gating classes can fail;
`diagnostic` and `unconverged` quantities are shown with their deviation and marked as
informational, so a reader sees them move without the run failing on them.
"""
function report_golden_check(db::SQLite.DB, case_spec::CaseSpec, commit_hash::String)
    golden = load_golden(case_spec.name)
    println()
    println("Golden Check: $(case_spec.name)")
    if golden === nothing
        println("  No golden file at $(golden_path(case_spec.name)) — nothing to check against.")
        println("  Generate one with:  regress --update-golden --cases $(case_spec.name) --reason \"...\"")
        return (n_pass=0, n_fail=0, n_untracked=0, n_informational=0)
    end

    quantities = get_quantities(db, commit_hash, case_spec.name)
    info = get_run_info(db, commit_hash, case_spec.name)
    if info !== nothing && !info.success
        println("  RUN FAILED — nothing to compare (this is a crash, not a tolerance failure):")
        println("  $(_short_err(info.error_msg))")
        return (n_pass=0, n_fail=1, n_untracked=0, n_informational=0)
    end

    rows = Vector{Vector{String}}()
    n_pass = n_fail = n_untracked = n_informational = 0

    for spec in case_spec.quantities
        g = get(golden.values, spec.name, nothing)
        if g === nothing
            # Runtime and checksums are structurally un-goldenable (no tolerance semantics), so
            # they must not inflate the untracked count that flags genuinely unpinned physics.
            (spec.type == "runtime" || spec.extract == "checksum") && continue
            n_untracked += 1
            continue
        end
        q_raw = get(quantities, spec.name, nothing)
        if q_raw === nothing
            push!(rows, [spec.label, g.class, "MISSING", "—", "FAIL"])
            n_fail += 1
            continue
        end
        # SQLite NULLs surface as `missing`, which the === nothing guards in compare_to_golden
        # never match; normalize here as the update path already does.
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
        dev = isnan(deviation) ? "—" : @sprintf("%.2e", deviation)
        tol = isfinite(g.rtol) ? @sprintf("%.1e", g.rtol) : "—"
        push!(rows, [spec.label, g.class, dev, tol, status * (isempty(detail) ? "" : "  $detail")])
    end

    # Golden entries with no matching case quantity fail regardless of class: a renamed or
    # removed quantity would otherwise silently delete its gate (the new name enters as merely
    # "untracked"), and a check that quietly stops checking something is worse than one that
    # fails. The fix is to regenerate the goldens for the reshaped case, with a reason.
    spec_names = Set(spec.name for spec in case_spec.quantities)
    for name in sort(collect(keys(golden.values)))
        name in spec_names && continue
        push!(rows, [name, golden.values[name].class, "ORPHANED", "—",
            "** FAIL **  no matching quantity in the case — renamed or removed? Regenerate goldens."])
        n_fail += 1
    end

    header = ["Quantity", "Class", "Deviation", "rtol", "Status"]
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
    println()
    return (n_pass=n_pass, n_fail=n_fail, n_untracked=n_untracked, n_informational=n_informational)
end

"""
Regenerate a case's golden file from a completed run, reporting what moved.

Prints an old→new delta for every quantity whose value changed, so the diff a reviewer sees in
git is accompanied by the size of each move.
"""
function update_golden_from_run(db::SQLite.DB, case_spec::CaseSpec, commit_hash::String,
                                reason::String, repo_root::String)
    info = get_run_info(db, commit_hash, case_spec.name)
    (info === nothing || !info.success) && error("Cannot update goldens for '$(case_spec.name)': the run did not succeed")

    # SQLite NULLs come back as `missing`, which the Union{...,Nothing} fields reject; `_column`
    # normalizes both absent forms to nothing.
    extracted = ExtractedQuantity[]
    for (name, q) in get_quantities(db, commit_hash, case_spec.name)
        push!(extracted, ExtractedQuantity(name, String(_column(q.label, name)),
            _column(q.value_real, nothing), _column(q.value_int, nothing),
            _column(q.value_text, nothing), q.value_type, q.noise_threshold))
    end

    previous = load_golden(case_spec.name)
    existing = previous === nothing ? nothing : previous.values
    values = build_golden_values(extracted, case_spec.quantities, existing)

    fp = info.fingerprint
    meta = GoldenMeta(case_spec.name,
        previous === nothing ? 1 : previous.meta.golden_version + 1,
        Dates.format(Dates.now(), "yyyy-mm-dd"),
        strip(read(`git -C $repo_root rev-parse --short HEAD`, String)),
        reason, fp.julia_version, fp.os_arch, fp.manifest_sha, fp.nthreads, fp.blas_threads)
    if commit_hash == LOCAL_REF && !isempty(strip(read(`git -C $repo_root status --porcelain`, String)))
        # The numbers came from uncommitted source; a clean HEAD checkout will not reproduce
        # them, and the commit field is the one a reviewer uses to reproduce a disputed number.
        meta = GoldenMeta(meta.case, meta.golden_version, meta.generated_at,
            meta.commit * "-dirty", meta.reason, meta.julia_version, meta.os_arch,
            meta.manifest_sha, meta.nthreads, meta.blas_threads)
        @warn "Working tree is dirty: golden provenance recorded as $(meta.commit). Commit first if these values are meant to be reproducible."
    end

    println()
    println("Golden update: $(case_spec.name)  (v$(previous === nothing ? 0 : previous.meta.golden_version) → v$(meta.golden_version))")
    if existing !== nothing
        for name in sort(collect(keys(existing)))
            if !haskey(values, name)
                println(@sprintf("  %-34s REMOVED — extraction returned missing (renamed h5 path?) or the quantity left the case. This deletes its gate; confirm it is intentional.", name))
            end
        end
        for (name, g) in sort(collect(values); by=first)
            old = get(existing, name, nothing)
            old === nothing && (println(@sprintf("  %-34s NEW", name)); continue)
            if g.value_real !== nothing && old.value_real !== nothing && g.value_real != old.value_real
                rel = old.value_real == 0 ? Inf : abs(g.value_real - old.value_real) / abs(old.value_real)
                println(@sprintf("  %-34s %.9g → %.9g   (rel %.2e)", name, old.value_real, g.value_real, rel))
            elseif g.value_text !== nothing && old.value_text !== nothing && g.value_text != old.value_text
                println(@sprintf("  %-34s array changed", name))
            end
        end
    end
    path = save_golden(meta, values)
    println("Wrote $path  ($(length(values)) quantities)")
    provisional = count(g -> startswith(g.tolerance_basis, "class-default"), Base.values(values))
    provisional > 0 && println("  $provisional quantity/quantities still carry provisional class-default tolerances — " *
                               "replace them with measured plateau drift and platform spread before relying on this as a gate.")
    println()
    return path
end

"""
Build golden entries from a set of freshly extracted quantities.

`existing` carries forward the tolerance, class, and evidence already recorded for a quantity, so
regenerating values after a physics change does not silently reset hard-won measurements to
provisional defaults.
"""
function build_golden_values(extracted::Vector{ExtractedQuantity}, specs::Vector{QuantitySpec},
                             existing::Union{Dict{String,GoldenValue},Nothing})
    spec_by_name = Dict(s.name => s for s in specs)
    values = Dict{String,GoldenValue}()
    for eq in extracted
        eq.value_type == "missing" && continue
        spec = get(spec_by_name, eq.name, nothing)
        spec === nothing && continue
        prior = existing === nothing ? nothing : get(existing, eq.name, nothing)
        if prior !== nothing && prior.value_type != eq.value_type
            # A type change invalidates the class and every measurement made under the old type;
            # carrying a topological rtol=0 onto a float (or a float rtol onto a count) mis-gates.
            @warn "Golden '$(eq.name)': value_type changed $(prior.value_type) → $(eq.value_type); resetting class and tolerances to provisional"
            prior = nothing
        end
        class = prior === nothing ? infer_class(spec) : prior.class
        # Checksums have no notion of "close", so they cannot carry a tolerance; they stay a
        # same-machine differential tool rather than a golden gate.
        eq.value_type == "checksum" && continue
        if prior === nothing
            defaults = CLASS_DEFAULT_TOLERANCE[class]
            rtol, atol, basis = defaults.rtol, defaults.atol, "class-default (provisional)"
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
