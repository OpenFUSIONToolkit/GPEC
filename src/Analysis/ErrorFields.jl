"""
    ErrorFields

Plots for the error-field assessment stored under `ErrorFields/` of a GPEC HDF5 output:
per-coil sensitivities, the tolerance Monte Carlo distributions, the locking risk and its
dependence on tolerance, the threshold scaling, the dominant coupling mode, and coil-array
phasing maps. Every plot takes a list of `label => h5path` pairs so design revisions can be
overplotted, with a single-path convenience method for the common case.
"""
module ErrorFields

using HDF5
using Plots
using LinearAlgebra
using TOML

import ...ErrorFields as EF
import ...PerturbedEquilibrium as PE
import ...ForcingTerms as FT

"""
A label paired with something to plot: a `gpec.h5` path, or a result already in memory
(`ErrorFields.SensitivityTable`, `CoilSensitivities`, `MonteCarloResult`, `RiskResult`, or a
vector of `CoilOverlap`). Analysing coil geometry that was never part of a run produces the
latter, so the plots accept both.
"""
const Sources = AbstractVector{<:Pair{String,<:Any}}

"""
A single unlabelled source: a `gpec.h5` path, or one in-memory result.
"""
const SingleSource = Union{AbstractString,EF.SensitivityTable,EF.CoilSensitivities,EF.MonteCarloResult,EF.RiskResult,AbstractVector{EF.CoilOverlap}}

_sources(source) = [_default_label(source) => source]
_default_label(h5path::AbstractString) = basename(dirname(abspath(h5path)))
_default_label(_) = "in memory"

# Every field the plots read. One template so the HDF5 and in-memory loaders cannot drift apart.
const _BLANK = (; coil_names=nothing, delta_as_designed=nothing, abs_delta_shift_per_mm=nothing,
    abs_delta_tilt_per_deg=nothing, abs_delta_rim_per_mm=nothing, shift_sensitivity_per_m=nothing, tilt_sensitivity_per_deg=nothing,
    field_as_designed=nothing, b_t0=nothing, resonant_fraction_percent=nothing, major_radius_m=nothing,
    shift_linearity_residual=nothing, tilt_linearity_residual=nothing, fd_step_shift_m=nothing, fd_step_tilt_deg=nothing,
    tolerances=nothing, abs_delta_bin_edges=nothing, abs_delta_pdf=nothing, abs_delta_efc_pdf=nothing, abs_delta_pdf_batches=nothing, abs_delta_total_as_designed=nothing,
    abs_delta_worst_case=nothing, abs_delta_sampled_mean=nothing, clamped_fraction=nothing, threshold_pdf=nothing,
    locking_probability_given_delta=nothing, threshold_fit=nothing, locking_probability_percent=nothing, locking_probability_efc_percent=nothing,
    scan_scale=nothing, scan_locking_probability_percent=nothing, scan_locking_probability_efc_percent=nothing, scan_locking_probability_spread_percent=nothing,
    scan_locking_probability_efc_spread_percent=nothing, dominant_v=nothing, singular_values=nothing, mn_index=nothing)

_load(d::NamedTuple) = merge(_BLANK, d)
_load(h5path::AbstractString) = _load(_load_h5(h5path))

_load(t::EF.SensitivityTable) = _load((; coil_names=t.coil_names, delta_as_designed=t.delta_as_designed,
    abs_delta_shift_per_mm=t.abs_delta_shift_per_mm, abs_delta_tilt_per_deg=t.abs_delta_tilt_per_deg,
    abs_delta_rim_per_mm=t.abs_delta_rim_per_mm, shift_sensitivity_per_m=t.shift_sensitivity_per_m, tilt_sensitivity_per_deg=t.tilt_sensitivity_per_deg))

_load(s::EF.CoilSensitivities) = _load((; coil_names=s.coil_names, field_as_designed=s.field_as_designed, b_t0=s.b_t0,
    shift_sensitivity_per_m=s.shift_sensitivity_per_m, tilt_sensitivity_per_deg=s.tilt_sensitivity_per_deg, major_radius_m=s.major_radius_m,
    shift_linearity_residual=s.shift_linearity_residual, tilt_linearity_residual=s.tilt_linearity_residual))

_load(r::EF.MonteCarloResult) =
    _load((; abs_delta_bin_edges=r.abs_delta_bin_edges, abs_delta_pdf=r.abs_delta_pdf, abs_delta_efc_pdf=r.abs_delta_efc_pdf, abs_delta_pdf_batches=r.abs_delta_pdf_batches,
        abs_delta_total_as_designed=r.abs_delta_total_as_designed, abs_delta_worst_case=r.abs_delta_worst_case, abs_delta_sampled_mean=r.abs_delta_sampled_mean,
        clamped_fraction=r.clamped_fraction))

_load(r::EF.RiskResult) = _load((; abs_delta_bin_edges=r.abs_delta_bin_edges, threshold_pdf=r.threshold_pdf,
    locking_probability_given_delta=r.locking_probability_given_delta, threshold_fit=r.threshold_fit,
    locking_probability_percent=r.locking_probability_percent, locking_probability_efc_percent=r.locking_probability_efc_percent))

# Drop the unset entries of a loaded source so two of them can be combined without one's blanks
# erasing the other's values.
_present(d::NamedTuple) = NamedTuple(k => v for (k, v) in pairs(d) if v !== nothing)

"""
The overlap distribution lives on the Monte Carlo result and the penetration threshold on the risk
result, so a plot that draws both against each other needs the pair. Pass them together as a tuple
rather than separately, which would leave each half unable to find the other.
"""
_load(pair::Tuple{EF.MonteCarloResult,EF.RiskResult}) =
    _load(merge(_present(_load(pair[1])), _present(_load(pair[2]))))

_load(ovs::AbstractVector{EF.CoilOverlap}) = _load((; coil_names=[o.coil_name for o in ovs],
    delta_as_designed=[o.delta for o in ovs], resonant_fraction_percent=[o.resonant_fraction_percent for o in ovs],
    b_t0=isempty(ovs) ? nothing : first(ovs).b_t0))

# The finite-difference steps the run's sensitivities were taken at, from the echoed deck; a
# file without one gets the control defaults, which is what an unset deck used.
function _fd_steps(f)
    defaults = EF.ErrorFieldsControl()
    ef = haskey(f, "Input/gpec_toml_raw") ? get(TOML.parse(read(f["Input/gpec_toml_raw"])), "ErrorFields", Dict{String,Any}()) : Dict{String,Any}()
    return Float64(get(ef, "fd_step_shift_m", defaults.fd_step_shift_m)), Float64(get(ef, "fd_step_tilt_deg", defaults.fd_step_tilt_deg))
end

function _load_h5(h5path::AbstractString)
    tolerances = EF.read_tolerance_snapshot(h5path)
    h5open(h5path, "r") do f
        has(k) = haskey(f, k)
        # Take the group paths from the writer's own constants rather than repeating them here:
        # two spellings of the schema drift apart silently, and a renamed group would surface as an
        # empty plot rather than an error.
        cs, mc, rk = EF._H5_GROUP, EF._MC_GROUP, EF._RISK_GROUP
        h_shift, h_tilt = _fd_steps(f)
        (
            coil_names=has(cs) ? read(f["$cs/coil_name"]) : nothing,
            delta_as_designed=has(cs) ? read(f["$cs/DominantMode/delta_as_designed"]) : nothing,
            abs_delta_shift_per_mm=has(cs) ? read(f["$cs/DominantMode/abs_delta_shift_per_mm"]) : nothing,
            abs_delta_tilt_per_deg=has(cs) ? read(f["$cs/DominantMode/abs_delta_tilt_per_deg"]) : nothing,
            abs_delta_rim_per_mm=has(cs) ? read(f["$cs/DominantMode/abs_delta_rim_per_mm"]) : nothing,
            shift_sensitivity_per_m=has(cs) ? read(f["$cs/DominantMode/shift_sensitivity_per_m"]) : nothing,
            tilt_sensitivity_per_deg=has(cs) ? read(f["$cs/DominantMode/tilt_sensitivity_per_deg"]) : nothing,
            field_as_designed=has(cs) ? read(f["$cs/field_as_designed"]) : nothing,
            b_t0=has("Equilibrium/B_T_axis") ? Float64(read(f["Equilibrium/B_T_axis"])) : nothing,
            major_radius_m=has(cs) ? read(f["$cs/major_radius_m"]) : nothing,
            shift_linearity_residual=has(cs) ? read(f["$cs/shift_linearity_residual"]) : nothing,
            tilt_linearity_residual=has(cs) ? read(f["$cs/tilt_linearity_residual"]) : nothing,
            fd_step_shift_m=h_shift,
            fd_step_tilt_deg=h_tilt,
            tolerances=tolerances,
            abs_delta_bin_edges=has(mc) ? read(f["$mc/abs_delta_bin_edges"]) : nothing,
            abs_delta_pdf=has(mc) ? read(f["$mc/abs_delta_pdf"]) : nothing,
            abs_delta_efc_pdf=has(mc) ? read(f["$mc/abs_delta_efc_pdf"]) : nothing,
            abs_delta_pdf_batches=has(mc) ? read(f["$mc/abs_delta_pdf_batches"]) : nothing,
            abs_delta_total_as_designed=has(mc) ? read(f["$mc/abs_delta_total_as_designed"]) : nothing,
            abs_delta_worst_case=has(mc) ? read(f["$mc/abs_delta_worst_case"]) : nothing,
            abs_delta_sampled_mean=has(mc) ? read(f["$mc/abs_delta_sampled_mean"]) : nothing,
            clamped_fraction=has(mc) ? read(f["$mc/clamped_fraction"]) : nothing,
            threshold_pdf=has(rk) ? read(f["$rk/threshold_pdf"]) : nothing,
            locking_probability_given_delta=has(rk) ? read(f["$rk/locking_probability_given_delta"]) : nothing,
            threshold_fit=has(rk) ? read(f["$rk/threshold_fit"]) : nothing,
            locking_probability_percent=has(rk) ? read(f["$rk/locking_probability_percent"]) : nothing,
            locking_probability_efc_percent=has(rk) ? read(f["$rk/locking_probability_efc_percent"]) : nothing,
            scan_scale=has("$rk/ToleranceScan") ? read(f["$rk/ToleranceScan/tolerance_scale"]) : nothing,
            scan_locking_probability_percent=has("$rk/ToleranceScan") ? read(f["$rk/ToleranceScan/locking_probability_percent"]) : nothing,
            scan_locking_probability_efc_percent=has("$rk/ToleranceScan") ? read(f["$rk/ToleranceScan/locking_probability_efc_percent"]) : nothing,
            scan_locking_probability_spread_percent=has("$rk/ToleranceScan") ? read(f["$rk/ToleranceScan/locking_probability_spread_percent"]) : nothing,
            scan_locking_probability_efc_spread_percent=has("$rk/ToleranceScan") ? read(f["$rk/ToleranceScan/locking_probability_efc_spread_percent"]) : nothing,
            dominant_v=has("PerturbedEquilibrium/SingularCoupling/DominantMode") ? read(f["PerturbedEquilibrium/SingularCoupling/DominantMode/right_singular_vectors"]) : nothing,
            singular_values=has("PerturbedEquilibrium/SingularCoupling/DominantMode") ? read(f["PerturbedEquilibrium/SingularCoupling/DominantMode/singular_values"]) : nothing,
            mn_index=has("Info/mn_index") ? read(f["Info/mn_index"]) : nothing
        )
    end
end

_centers(edges) = (edges[1:end-1] .+ edges[2:end]) ./ 2
_step_series(m, a) = (vcat(m[1] - 1, m, m[end] + 1), vcat(0.0, a, 0.0))
_empty(msg) = plot(; title=msg, legend=false)

function _save(p, save_path)
    if save_path !== nothing
        Plots.savefig(p, save_path)
        println("Saved: ", abspath(save_path))
    end
    return p
end

"""
    plot_coil_sensitivities(sources; quantity=:shift, coils=nothing, yscale=:identity, save_path=nothing)
    plot_coil_sensitivities(source; kwargs...)

Grouped bars of the per-coil-set dominant-mode sensitivity across runs, matched by coil set name.
`quantity` is `:shift` (per millimetre of rigid in-plane shift), `:tilt` (per degree), `:rim` (the
same tilt as rim displacement, the unit mechanical tolerances arrive in), `:as_designed` (`|δ|` as
built), or `:fraction` (the resonant share of the coil's own applied spectrum, `100·|δ|·B_T0/‖b̃‖`
in percent, which separates a coil that couples weakly because it is small from one that couples
weakly because its spectrum lies off the dominant mode). `coils` restricts and orders the coil
sets shown; a name a source does not carry is an error rather than an empty bar.
`yscale = :log10` draws the values as stems with markers instead of bars, since a bar's height on
a logarithmic axis is set by the axis floor rather than by the data; non-positive values are
omitted.

Each source is a `gpec.h5` path, a `SensitivityTable` already in memory, or a vector of
`CoilOverlap` (`:as_designed` and `:fraction` only), so a sweep of coil geometry that was never part of
a run plots the same way a stored run does. `:fraction` needs the applied field as well as the
overlap, so it is not available from a `SensitivityTable`.
"""
function plot_coil_sensitivities(sources::Sources; quantity::Symbol=:shift, coils=nothing, yscale::Symbol=:identity, save_path=nothing)
    quantity in (:shift, :tilt, :rim, :as_designed, :fraction) || throw(ArgumentError("quantity must be :shift, :tilt, :rim, :as_designed, or :fraction"))
    yscale in (:identity, :log10) || throw(ArgumentError("yscale must be :identity or :log10"))
    data = [(lbl, _load(src)) for (lbl, src) in sources]
    any(d -> d[2].coil_names === nothing, data) && return _empty("No ErrorFields/CoilSensitivities data — run with an [ErrorFields] section")
    names = coils === nothing ? data[1][2].coil_names : String.(collect(coils))
    values = [_sensitivity_values(lbl, d, names, quantity) for (lbl, d) in data]
    ylabel =
        quantity === :shift ? "|δ| per mm of shift" :
        quantity === :tilt ? "|δ| per degree of tilt" :
        quantity === :rim ? "|δ| per mm of rim displacement" : quantity === :as_designed ? "|δ_as_designed|" : "resonant fraction of |b̃| [%]"
    per = quantity === :shift ? "shift" : quantity === :tilt ? "tilt" : "rim displacement"
    title =
        quantity === :as_designed ? "As-designed dominant-mode overlap" :
        quantity === :fraction ? "Resonant fraction of each coil set's applied field" :
        "Dominant-mode error field per $per"
    n = length(names)
    k = length(data)
    width = 0.8 / k
    p = plot(; xlabel="coil set", ylabel=ylabel, title=title, xticks=(1:n, names), xrotation=45, legend=:topright,
        left_margin=12Plots.mm, bottom_margin=8Plots.mm)
    if yscale === :log10
        ylo = _log_floor!(p, reduce(vcat, values))
        ylo === nothing && return _empty("No positive $quantity values to draw on a logarithmic axis")
        for (j, (lbl, _)) in enumerate(data)
            _stems!(p, (1:n) .+ (j - (k + 1) / 2) * width, values[j], ylo; c=j, label=lbl)
        end
    else
        for (j, (lbl, _)) in enumerate(data)
            x = (1:n) .+ (j - (k + 1) / 2) * width
            bar!(p, x, values[j]; bar_width=width, label=lbl, alpha=0.8)
        end
    end
    return _save(p, save_path)
end

# A logarithmic axis for stems: the floor a decade under the smallest positive value, or nothing
# when there is none to draw. Bars on a log axis stretch to whatever the floor is; stems do not.
function _log_floor!(p, values)
    positive = filter(v -> isfinite(v) && v > 0, values)
    isempty(positive) && return nothing
    ylo = minimum(positive) / 10
    plot!(p; yscale=:log10, ylims=(ylo, 3 * maximum(positive)))
    return ylo
end

# Stems from the floor to each positive value, with a marker on top; non-positive values are skipped.
function _stems!(p, x, values, ylo; c, label, marker=:circle, ms=5, lw=3)
    keep = [isfinite(v) && v > 0 for v in values]
    xs = vec(vcat(x[keep]', x[keep]', fill(NaN, 1, count(keep))))
    ys = vec(vcat(fill(ylo, 1, count(keep)), values[keep]', fill(NaN, 1, count(keep))))
    plot!(p, xs, ys; lw, c, label="")
    scatter!(p, x[keep], values[keep]; marker, ms, c, label)
    return p
end

# Positions of `names` among a source's coil sets, erroring on a coil the source does not carry
# rather than drawing an empty bar for it.
function _coil_indices(lbl, d, names)
    absent = [nm for nm in names if nm ∉ d.coil_names]
    isempty(absent) || throw(ArgumentError("source \"$lbl\" has no coil set named $(join(repr.(absent), ", ")); it carries $(join(repr.(d.coil_names), ", "))"))
    return [findfirst(==(nm), d.coil_names) for nm in names]
end

# One source's values of `quantity` in `names` order.
function _sensitivity_values(lbl, d, names, quantity::Symbol)
    idx = _coil_indices(lbl, d, names)
    if quantity === :fraction
        d.resonant_fraction_percent === nothing || return d.resonant_fraction_percent[idx]
        (d.delta_as_designed === nothing || d.field_as_designed === nothing || d.b_t0 === nothing) &&
            throw(
                ArgumentError(
                    "source \"$lbl\" cannot give the resonant fraction: it needs the applied field and B_T0 as well as δ (a gpec.h5 path or a vector of CoilOverlap, not a SensitivityTable)"
                )
            )
        norms = [norm(d.field_as_designed[:, i]) for i in axes(d.field_as_designed, 2)]
        nrm_ref = maximum(norms)
        return [EF.resonant_fraction_percent(d.delta_as_designed[i] * d.b_t0, norms[i], nrm_ref) for i in idx]
    end
    field = quantity === :shift ? :abs_delta_shift_per_mm : quantity === :tilt ? :abs_delta_tilt_per_deg : quantity === :rim ? :abs_delta_rim_per_mm : :delta_as_designed
    vals = getfield(d, field)
    vals === nothing && throw(ArgumentError("source \"$lbl\" carries no $field, so quantity=:$quantity cannot be drawn from it"))
    return abs.(vals[idx])
end
plot_coil_sensitivities(source::SingleSource; kwargs...) = plot_coil_sensitivities(_sources(source); kwargs...)

"""
    plot_tolerance_pdf(sources; corrected=true, normalize=false, xscale=:identity, show_batches=false, save_path=nothing)
    plot_tolerance_pdf(h5path; kwargs...)

The Monte Carlo distributions of the dominant-mode overlap `|δ|` of each run, intrinsic and
(when `corrected`) with error-field correction, with the as-designed overlap marked.
`show_batches` draws each batch's intrinsic histogram faintly under the mean, which is the
sampling noise a risk figure inherits, and notes in the legend the fraction of samples that fell
beyond the last bin edge.
"""
function plot_tolerance_pdf(sources::Sources; corrected::Bool=true, normalize::Bool=false, xscale::Symbol=:identity, show_batches::Bool=false,
    save_path=nothing)
    p = plot(; xlabel="dominant-mode overlap |δ|", ylabel=normalize ? "probability density (normalized)" : "probability density",
        title="Tolerance Monte Carlo: |δ| over sampled misalignments", legend=:topright, xscale=xscale,
        left_margin=12Plots.mm, bottom_margin=6Plots.mm)
    any_data = false
    for (j, (lbl, path)) in enumerate(sources)
        d = _load(path)
        d.abs_delta_pdf === nothing && continue
        any_data = true
        c = _centers(d.abs_delta_bin_edges)
        keep = xscale === :log10 ? c .> 0 : trues(length(c))
        scale = normalize ? maximum(d.abs_delta_pdf) : 1.0
        clamped = ""
        if show_batches && d.abs_delta_pdf_batches !== nothing
            for b in axes(d.abs_delta_pdf_batches, 2)
                plot!(p, c[keep], d.abs_delta_pdf_batches[keep, b] ./ scale; lw=1, c=j, alpha=0.35, label=b == 1 ? "$lbl batches ($(size(d.abs_delta_pdf_batches, 2)))" : "")
            end
            d.clamped_fraction === nothing || (clamped = " ($(round(100 * d.clamped_fraction; sigdigits=2)) % beyond last bin)")
        end
        plot!(p, c[keep], d.abs_delta_pdf[keep] ./ scale; lw=2, c=j, label="$lbl intrinsic$clamped")
        corrected && plot!(p, c[keep], d.abs_delta_efc_pdf[keep] ./ (normalize ? maximum(d.abs_delta_efc_pdf) : 1.0); lw=2, ls=:dash, c=j, label="$lbl corrected")
        vline!(p, [d.abs_delta_total_as_designed]; ls=:dot, c=j, label="$lbl as designed")
    end
    any_data || return _empty("No ErrorFields/MonteCarlo data — run with a tolerance_file")
    return _save(p, save_path)
end
plot_tolerance_pdf(source::SingleSource; kwargs...) = plot_tolerance_pdf(_sources(source); kwargs...)

"""
    plot_overlap_phasors(sources; coils=nothing, save_path=nothing)
    plot_overlap_phasors(source; kwargs...)

Each coil set's as-built dominant-mode overlap `δ` as an arrow in the complex plane, laid head to
tail in `coils` order so the walk ends at the run's total, drawn as one bold arrow from the
origin. The picture is rotated so that the total is real: an arrow pointing along it adds to the
error field, one pointing against it cancels part of another coil's, and the single number
`|Σδ|` a Monte Carlo starts from cannot tell those apart. One panel per source, since the mode
phase is arbitrary between runs. Sources are a `gpec.h5` path, a `SensitivityTable`, or a vector
of `CoilOverlap`; on a file the walk's end is checked against the stored Monte Carlo
`delta_as_designed` and a mismatch is warned about.
"""
function plot_overlap_phasors(sources::Sources; coils=nothing, save_path=nothing)
    data = [(lbl, _load(src)) for (lbl, src) in sources]
    any(d -> d[2].delta_as_designed === nothing, data) && return _empty("No per-coil overlaps — run with an [ErrorFields] section")
    panels = Plots.Plot[]
    for (lbl, d) in data
        names = coils === nothing ? d.coil_names : String.(collect(coils))
        idx = _coil_indices(lbl, d, names)
        z = d.delta_as_designed[idx]
        δ = abs.(z)
        total = sum(z)
        if coils === nothing && d.abs_delta_total_as_designed !== nothing && !isapprox(abs(total), d.abs_delta_total_as_designed; rtol=1e-6, atol=eps(Float64))
            @warn "$lbl: the head-to-tail total |Σδ| = $(abs(total)) differs from the stored Monte Carlo abs_delta_total_as_designed $(d.abs_delta_total_as_designed)"
        end
        rot = abs(total) > 0 ? cis(-angle(total)) : one(ComplexF64)
        w = z .* rot
        tip = cumsum(w)
        tail = vcat(zero(ComplexF64), tip[1:end-1])
        # Equal aspect keeps the angles honest; the box is padded so a walk along the real axis
        # (every coil in phase) does not collapse the panel to a sliver.
        xs = vcat(0.0, real.(tip))
        ys = vcat(0.0, imag.(tip))
        span = max(maximum(xs) - minimum(xs), maximum(ys) - minimum(ys), eps())
        pad = 0.1 * span
        half_y = max(0.3 * span, 0.5 * (maximum(ys) - minimum(ys)) + pad)
        y_mid = 0.5 * (maximum(ys) + minimum(ys))
        p = plot(; xlabel="Re δ  (rotated so the total is real)", ylabel="Im δ", title="Overlap phasors: $lbl,  |Σδ| = $(round(abs(total); sigdigits=3))",
            aspect_ratio=:equal, xlims=(minimum(xs) - pad, maximum(xs) + pad), ylims=(y_mid - half_y, y_mid + half_y), legend=:outerright,
            size=(900, 560), left_margin=12Plots.mm, bottom_margin=6Plots.mm, titlefontsize=12)
        for (k, nm) in enumerate(names)
            plot!(p, [real(tail[k]), real(tip[k])], [imag(tail[k]), imag(tip[k])]; arrow=true, lw=2, c=k, label="$nm (|δ| = $(round(δ[k]; sigdigits=2)))")
        end
        plot!(p, [0.0, abs(total)], [0.0, 0.0]; arrow=true, lw=4, c=:black, alpha=0.6, label="total")
        push!(panels, p)
    end
    length(panels) == 1 && return _save(panels[1], save_path)
    return _save(plot(panels...; layout=(1, length(panels)), size=(900 * length(panels), 560)), save_path)
end
plot_overlap_phasors(source::SingleSource; kwargs...) = plot_overlap_phasors(_sources(source); kwargs...)

"""
    plot_tolerance_budget(h5path; psi_low=0.0, psi_high=CORE_PSI_HIGH, mode=1, tolerance_scale=1.0, sort=:total, top=nothing, save_path=nothing)
    plot_tolerance_budget(table, tolerances, coil_sets; monte_carlo=nothing, kwargs...)
    plot_tolerance_budget(terms::NamedTuple; monte_carlo=nothing, sort=:total, top=nothing, save_path=nothing)

The worst-case tolerance budget of a run as stacked horizontal bars, one per coil set, coherent
group and the unattributed budget: the as-designed `|δ_as_designed|`, the shift term and the tilt term
of `ErrorFields.worst_case_terms`, whose sum over every bar is the `abs_delta_worst_case` the
Monte Carlo sizes its histogram by. It shows which coil's tolerance the budget is spent on, and
whether it is spent on the coil's shift, its tilt, or on the field it makes as designed.
`sort` orders the bars by `:total` (largest at the top), `:name`, or `:none` (file order); `top`
keeps only that many largest. With a Monte Carlo result (read from the file, or passed as
`monte_carlo`) the sample mean of `|δ|` and the coherent as-designed total are marked, which
puts the worst-case bars against what the sampling typically realises.
"""
function plot_tolerance_budget(terms::NamedTuple; monte_carlo=nothing, sort::Symbol=:total, top=nothing, save_path=nothing)
    sort in (:total, :name, :none) || throw(ArgumentError("sort must be :total, :name, or :none"))
    ng = length(terms.group_names)
    nc = length(terms.coil_names)
    labels = vcat(terms.coil_names, ["group $g" for g in terms.group_names], ["unattributed"])
    nominal = vcat(terms.abs_delta_as_designed, zeros(ng), 0.0)
    shift = vcat(terms.abs_delta_shift_tolerance, terms.abs_delta_group_shift_tolerance, 0.0)
    tilt = vcat(terms.abs_delta_tilt_tolerance, terms.abs_delta_group_tilt_tolerance, 0.0)
    other = vcat(zeros(nc + ng), terms.abs_delta_unattributed)
    total = nominal .+ shift .+ tilt .+ other
    order = sort === :total ? sortperm(total) : sort === :name ? sortperm(labels; rev=true) : collect(reverse(eachindex(labels)))
    top === nothing || (order = order[max(1, end - top + 1):end])
    m = length(order)
    p = plot(; xlabel="worst-case contribution to |δ|", ylabel="", yticks=(1:m, labels[order]), ylims=(0.4, m + 0.6),
        title="Tolerance budget by term  (Σ over every bar = abs_delta_worst_case = $(round(terms.abs_delta_worst_case; sigdigits=3)))", legend=:bottomright,
        size=(900, max(420, 26 * m + 160)), left_margin=12Plots.mm, bottom_margin=8Plots.mm, titlefontsize=12)
    segments = (("as designed |δ_as_designed|", nominal, :gray40), ("shift tolerance + 3σ", shift, 1), ("tilt tolerance + 3σ", tilt, 2), ("unattributed budget + 3σ", other, 3))
    labelled = Set{String}()
    for (y, i) in enumerate(order)
        x0 = 0.0
        for (name, vals, col) in segments
            vals[i] > 0 || continue
            x1 = x0 + vals[i]
            plot!(p, Shape([x0, x1, x1, x0], [y - 0.4, y - 0.4, y + 0.4, y + 0.4]); c=col, lw=0.5, label=name in labelled ? "" : name)
            push!(labelled, name)
            x0 = x1
        end
    end
    if monte_carlo !== nothing
        d = _load(monte_carlo)
        d.abs_delta_sampled_mean === nothing ||
            vline!(p, [d.abs_delta_sampled_mean]; ls=:dash, c=:black, label="Monte Carlo mean |δ| = $(round(d.abs_delta_sampled_mean; sigdigits=3))")
        d.abs_delta_total_as_designed === nothing ||
            vline!(p, [d.abs_delta_total_as_designed]; ls=:dot, c=:black, label="as designed |Σδ| = $(round(d.abs_delta_total_as_designed; sigdigits=3))")
    end
    return _save(p, save_path)
end
function plot_tolerance_budget(table::EF.SensitivityTable, ts::EF.ToleranceSet, coil_sets::AbstractVector{FT.CoilSet}; tolerance_scale::Real=1.0, kwargs...)
    return plot_tolerance_budget(EF.worst_case_terms(table, ts, collect(coil_sets); tolerance_scale); kwargs...)
end
function plot_tolerance_budget(h5path::AbstractString; psi_low::Real=0.0, psi_high::Real=PE.CORE_PSI_HIGH, mode::Int=1, tolerance_scale::Real=1.0, kwargs...)
    terms = EF.worst_case_terms(h5path; psi_low, psi_high, mode, tolerance_scale)
    d = _load(h5path)
    mc =
        d.abs_delta_pdf === nothing ? nothing :
        (;
            abs_delta_bin_edges=d.abs_delta_bin_edges,
            abs_delta_pdf=d.abs_delta_pdf,
            abs_delta_sampled_mean=d.abs_delta_sampled_mean,
            abs_delta_total_as_designed=d.abs_delta_total_as_designed
        )
    return plot_tolerance_budget(terms; monte_carlo=mc, kwargs...)
end

"""
    plot_linearity_residuals(sources; at=:tolerance, coils=nothing, tolerances=nothing, fd_step_shift_m=nothing, fd_step_tilt_deg=nothing, yscale=:identity, save_path=nothing)
    plot_linearity_residuals(source; kwargs...)

How linear each coil set's field is in its rigid motions: for the six taps (shift and tilt about
`x`, `y`, `z`) the ratio of the central second difference `‖b̃(+h) + b̃(−h) − 2b̃(0)‖` to the
largest first difference among the set's taps, as stored in
`ErrorFields/CoilSensitivities/*_linearity_residual`. The normalization is against the set's
*largest* linear response, so a tap whose own first difference vanishes by symmetry is judged
against the terms the model keeps, and a weak tap's own curvature is understated. `at = :step`
shows the ratio at the finite-difference steps the run used; `at = :tolerance` rescales each tap
by `tolerance / step` (the second difference grows as `h²`, the first as `h`), which is the
curvature the linear model neglects over the coil's own tolerance range. Coherent-group
amplitudes are not added. Tolerances come from the file's snapshot or `tolerances`; a coil
without a tolerance block draws at zero. The steps come from the file's echoed deck, or from
`fd_step_shift_m` and `fd_step_tilt_deg` for in-memory `CoilSensitivities`. The dashed line is
the 1 % level at which the sensitivity calculation itself warns; on a linear axis it sets the
scale and hides sub-percent residuals, so `yscale = :log10` draws them as stems and markers
(never bars) over the decades they span. One panel per source. For the model's error at finite
displacements rather than at the step, see [`plot_linearity_check`](@ref).
"""
function plot_linearity_residuals(sources::Sources; at::Symbol=:tolerance, coils=nothing, tolerances=nothing, fd_step_shift_m=nothing,
    fd_step_tilt_deg=nothing, yscale::Symbol=:identity, save_path=nothing)
    at in (:step, :tolerance) || throw(ArgumentError("at must be :step or :tolerance"))
    yscale in (:identity, :log10) || throw(ArgumentError("yscale must be :identity or :log10"))
    data = [(lbl, _load(src)) for (lbl, src) in sources]
    any(d -> d[2].shift_linearity_residual === nothing, data) && return _empty("No ErrorFields/CoilSensitivities linearity residuals — run with an [ErrorFields] section")
    taps = ("shift x", "shift y", "shift z", "tilt x", "tilt y", "tilt z")
    panels = Plots.Plot[]
    for (lbl, d) in data
        names = coils === nothing ? d.coil_names : String.(collect(coils))
        r = _linearity_matrix(lbl, d, names, at; tolerances, fd_step_shift_m, fd_step_tilt_deg)
        n = length(names)
        width = 0.8 / 6
        p = plot(; xlabel="coil set", ylabel=at === :step ? "‖2nd diff‖ / max ‖1st diff‖  at the FD step" : "‖2nd diff‖ / max ‖1st diff‖  at the tolerance",
            title="Linearity of the rigid-motion response: $lbl", xticks=(1:n, names), xrotation=45, legend=:topright, size=(900, 520),
            left_margin=12Plots.mm, bottom_margin=8Plots.mm, titlefontsize=12)
        if yscale === :log10
            ylo = _log_floor!(p, vcat(vec(r), 1e-2))
            for (t, tap) in enumerate(taps)
                _stems!(p, (1:n) .+ (t - 3.5) * width, r[t, :], ylo; c=t <= 3 ? t : t + 2, label=tap, lw=2, ms=4)
            end
        else
            for (t, tap) in enumerate(taps)
                x = (1:n) .+ (t - 3.5) * width
                bar!(p, x, r[t, :]; bar_width=width, label=tap, alpha=0.85, c=t <= 3 ? t : t + 2)
            end
        end
        hline!(p, [1e-2]; ls=:dash, c=:black, label="1 % (sensitivity warning level)")
        push!(panels, p)
    end
    length(panels) == 1 && return _save(panels[1], save_path)
    return _save(plot(panels...; layout=(1, length(panels)), size=(900 * length(panels), 520)), save_path)
end
plot_linearity_residuals(source::SingleSource; kwargs...) = plot_linearity_residuals(_sources(source); kwargs...)

"""
    plot_linearity_check(check::EF.LinearityCheck; save_path=nothing)
    plot_linearity_check(h5path; scales=(1.0, 2.0), save_path=nothing, kwargs...)

The relative error of the linear sensitivity model at finite displacements, from
`ErrorFields.linearity_check`: one stem per row on a logarithmic axis, grouped by coil set
or coherent group, with the shift and tilt rows told apart by colour, the scale by marker, and the
sign by fill. The dashed line is the 1 % level the sensitivity calculation warns at for its own step
residuals. A row whose linear change vanishes has no relative error and is not drawn.
"""
function plot_linearity_check(check::EF.LinearityCheck; save_path=nothing)
    isempty(check.name) && return _empty("No linearity rows: the tolerance set names no coil or group")
    names = unique(check.name)
    scales = sort(unique(check.scale))
    markers = (:circle, :diamond, :utriangle, :square, :star5, :hexagon)
    p = plot(; xlabel="coil set or coherent group", ylabel="|δ_actual − δ_linear| / |δ_linear − δ_as_designed|",
        title="Linearity of the overlap at finite displacements", xticks=(1:length(names), names), xrotation=45, legend=:outerright,
        size=(1000, 560), left_margin=12Plots.mm, bottom_margin=8Plots.mm, titlefontsize=12)
    ylo = _log_floor!(p, vcat(check.relative_error, 1e-2))
    ylo === nothing && return _empty("No finite relative errors to draw")
    width = 0.8 / (4 * length(scales))
    slot = 0
    for (k, kind) in enumerate((:shift, :tilt)), a in 1:2, (si, s) in enumerate(scales)
        slot += 1
        for (sgn, fill) in ((1, true), (-1, false))
            rows = findall(i -> check.kind[i] === kind && check.axis[i] == a && check.scale[i] == s && sign(check.displacement[i]) == sgn, eachindex(check.name))
            isempty(rows) && continue
            x = [findfirst(==(check.name[i]), names) + (slot - (4 * length(scales) + 1) / 2) * width for i in rows]
            lbl = sgn > 0 ? "$kind $(a == 1 ? "x" : "y") × $s" : ""
            _stems!(p, x, check.relative_error[rows], ylo; c=k, label=lbl, marker=markers[mod1(si, length(markers))], ms=fill ? 5 : 4, lw=1.5)
        end
    end
    hline!(p, [1e-2]; ls=:dash, c=:black, label="1 %")
    return _save(p, save_path)
end
plot_linearity_check(h5path::AbstractString; save_path=nothing, kwargs...) = plot_linearity_check(EF.linearity_check(h5path; kwargs...); save_path)

# The six taps' residual ratios per coil set (6 × n), at the finite-difference step or rescaled
# by tolerance / step to the coil's own tolerance.
function _linearity_matrix(lbl, d, names, at::Symbol; tolerances=nothing, fd_step_shift_m=nothing, fd_step_tilt_deg=nothing)
    idx = _coil_indices(lbl, d, names)
    r = vcat(d.shift_linearity_residual[:, idx], d.tilt_linearity_residual[:, idx])
    at === :step && return r
    ts = tolerances === nothing ? d.tolerances : tolerances
    ts === nothing && throw(ArgumentError("source \"$lbl\" carries no tolerance snapshot; pass tolerances=ToleranceSet or use at=:step"))
    h_shift = something(fd_step_shift_m, d.fd_step_shift_m, EF.ErrorFieldsControl().fd_step_shift_m)
    h_tilt = something(fd_step_tilt_deg, d.fd_step_tilt_deg, EF.ErrorFieldsControl().fd_step_tilt_deg)
    tol_of = Dict(t.name => t for t in ts.coils)
    for (k, i) in enumerate(idx)
        t = get(tol_of, d.coil_names[i], nothing)
        shift_tol = t === nothing ? 0.0 : t.shift_tol_m
        tilt_tol =
            t === nothing ? 0.0 :
            d.major_radius_m === nothing ? throw(ArgumentError("source \"$lbl\" carries no major radii, needed to convert a tilt tolerance in metres")) :
            EF.tilt_tolerance_deg(t.tilt_tol, t.tilt_units, d.major_radius_m[i]; name="coil set $(d.coil_names[i])")
        r[1:3, k] .*= shift_tol / h_shift
        r[4:6, k] .*= tilt_tol / h_tilt
    end
    return r
end

"""
    plot_locking_risk(sources; corrected=true, target_percent=nothing, save_path=nothing)
    plot_locking_risk(h5path; kwargs...)

Locking probability against tolerance scale from each run's `ErrorFields/Risk/ToleranceScan/`,
intrinsic and (when `corrected`) with error-field correction, batch spread as error bars, and
the allowable scale at `target_percent` marked where the scan reaches it.
"""
function plot_locking_risk(sources::Sources; corrected::Bool=true, target_percent=nothing, save_path=nothing)
    p = plot(; xlabel="tolerance scale", ylabel="locking probability [%]", xscale=:log10, yscale=:log10, legend=:topleft,
        title="Locking risk vs tolerance scale", left_margin=12Plots.mm, bottom_margin=6Plots.mm)
    any_data = false
    risk_floor = 1e-4   # named to avoid shadowing Base.floor inside this function
    for (j, (lbl, path)) in enumerate(sources)
        d = _load(path)
        d.scan_scale === nothing && continue
        any_data = true
        plot!(
            p,
            d.scan_scale,
            max.(d.scan_locking_probability_percent, risk_floor);
            yerror=d.scan_locking_probability_spread_percent ./ 2,
            marker=:circle,
            lw=2,
            c=j,
            label="$lbl intrinsic"
        )
        corrected && plot!(
            p,
            d.scan_scale,
            max.(d.scan_locking_probability_efc_percent, risk_floor);
            yerror=d.scan_locking_probability_efc_spread_percent ./ 2,
            marker=:square,
            lw=2,
            ls=:dash,
            c=j,
            label="$lbl corrected"
        )
        if target_percent !== nothing
            scan = EF.ToleranceScan(
                d.scan_scale,
                d.scan_locking_probability_percent,
                d.scan_locking_probability_efc_percent,
                d.scan_locking_probability_spread_percent,
                d.scan_locking_probability_efc_spread_percent,
                0.0
            )
            for (corr, mk) in ((false, :diamond), (true, :star5))
                (corr && !corrected) && continue
                s = EF.allowable_tolerance(scan, target_percent; corrected=corr)
                isnan(s) || scatter!(p, [s], [target_percent]; marker=mk, ms=8, c=j, label="$lbl allowable ($(corr ? "corrected" : "intrinsic"))")
            end
        end
    end
    any_data || return _empty("No ErrorFields/Risk/ToleranceScan data — set scan_scales in [ErrorFields.Risk]")
    target_percent === nothing || hline!(p, [target_percent]; ls=:dash, c=:gray, label="target $(target_percent) %")
    return _save(p, save_path)
end
plot_locking_risk(source::SingleSource; kwargs...) = plot_locking_risk(_sources(source); kwargs...)

"""
    plot_threshold_scaling(sources; save_path=nothing)
    plot_threshold_scaling(h5path; kwargs...)

The sampled penetration-threshold density and `P(lock|δ)` of each run against its overlap
distributions, on a logarithmic `|δ|` axis, with the fitted threshold marked.
"""
function plot_threshold_scaling(sources::Sources; save_path=nothing)
    p = plot(; xlabel="dominant-mode overlap |δ|", ylabel="normalized density  /  P(lock | δ)", xscale=:log10, legend=:topleft,
        title="Overlap distribution vs penetration threshold", left_margin=12Plots.mm, bottom_margin=6Plots.mm)
    any_data = false
    for (j, (lbl, path)) in enumerate(sources)
        d = _load(path)
        # Needs the overlap distribution and the threshold together; an in-memory source carrying
        # only one of the two is skipped rather than indexed into a nothing.
        (d.threshold_pdf === nothing || d.abs_delta_pdf === nothing || d.abs_delta_bin_edges === nothing) && continue
        any_data = true
        c = _centers(d.abs_delta_bin_edges)
        keep = c .> 0
        plot!(p, c[keep], d.abs_delta_pdf[keep] ./ maximum(d.abs_delta_pdf); lw=2, c=j, label="$lbl intrinsic |δ|")
        plot!(p, c[keep], d.abs_delta_efc_pdf[keep] ./ maximum(d.abs_delta_efc_pdf); lw=2, ls=:dash, c=j, label="$lbl corrected |δ|")
        plot!(p, c[keep], d.threshold_pdf[keep] ./ maximum(d.threshold_pdf); lw=2, ls=:dot, c=j, label="$lbl threshold")
        plot!(p, d.abs_delta_bin_edges[2:end], d.locking_probability_given_delta[2:end]; lw=1.5, ls=:dashdot, c=j, label="$lbl P(lock | δ)")
        vline!(p, [d.threshold_fit]; ls=:dot, c=:black, label=j == 1 ? "fitted threshold" : "")
    end
    any_data || return _empty("No ErrorFields/Risk data — run with an [ErrorFields.scenario] table")
    return _save(p, save_path)
end
plot_threshold_scaling(source::SingleSource; kwargs...) = plot_threshold_scaling(_sources(source); kwargs...)

"""
    plot_dominant_mode_spectrum(sources; mode=1, save_path=nothing)
    plot_dominant_mode_spectrum(h5path; kwargs...)

The right singular vector of the full-window dominant coupling mode of each run, `|V[m, mode]|`
against poloidal mode number (one series per toroidal mode), from
`PerturbedEquilibrium/SingularCoupling/DominantMode/`.
"""
function plot_dominant_mode_spectrum(sources::Sources; mode::Int=1, save_path=nothing)
    p = plot(; xlabel="poloidal mode m", ylabel="|V[m, $mode]|", title="Dominant resonant-coupling mode spectrum", legend=:topright,
        left_margin=12Plots.mm, bottom_margin=6Plots.mm)
    any_data = false
    for (j, (lbl, path)) in enumerate(sources)
        d = _load(path)
        (d.dominant_v === nothing || d.mn_index === nothing) && continue
        any_data = true
        mode <= size(d.dominant_v, 2) || continue
        for n in unique(d.mn_index[:, 2])
            rows = findall(==(n), d.mn_index[:, 2])
            me, ae = _step_series(d.mn_index[rows, 1], abs.(d.dominant_v[rows, mode]))
            plot!(p, me, ae; seriestype=:steppre, lw=2, c=j, label="$lbl n=$n (σ = $(round(d.singular_values[mode]; sigdigits=3)))")
        end
    end
    any_data || return _empty("No PerturbedEquilibrium/SingularCoupling/DominantMode data")
    return _save(p, save_path)
end
plot_dominant_mode_spectrum(source::SingleSource; kwargs...) = plot_dominant_mode_spectrum(_sources(source); kwargs...)

"""
    plot_applied_spectra(ctx, overlaps; normalize=true, save_path=nothing)
    plot_applied_spectra(h5path, coil_sets; normalize=true, save_path=nothing, kwargs...)

Each coil set's applied spectrum against the dominant resonant mode.

Both views answer different questions and the plot shows whichever is asked for: `normalize = true`
scales every curve to its own peak, which says whether a coil drives a *different* part of the
spectrum; `false` leaves the amplitudes, which says whether it simply drives *less*. A coil that
couples weakly for the first reason needs a geometry change; one that couples weakly for the second
may just be further away.
"""
function plot_applied_spectra(ctx::EF.ResonantDriveContext, overlaps::AbstractVector{EF.CoilOverlap};
    normalize::Bool=true, save_path=nothing)
    isempty(overlaps) && return _empty("No coil overlaps to plot")
    m = ctx.rc.m_modes
    mode = first(overlaps).mode
    p = plot(; xlabel="poloidal mode m", ylabel=normalize ? "amplitude / own peak" : "|b̃| (T)",
        title="Applied spectra against the dominant resonant mode", legend=:topright,
        left_margin=13Plots.mm, bottom_margin=7Plots.mm)
    v = abs.(ctx.dom.right_singular_vectors[:, mode])
    mv, av = _step_series(m, v ./ maximum(v))
    normalize && plot!(p, mv, av; seriestype=:steppre, lw=3, c=:black, fillrange=0, fillalpha=0.12, label="dominant mode |V|")
    for (j, o) in enumerate(overlaps)
        a = abs.(o.spectrum)
        scale = normalize ? maximum(a) : 1.0
        ms, as = _step_series(m, scale > 0 ? a ./ scale : a)
        plot!(p, ms, as; seriestype=:steppre, lw=2, c=j, label=o.coil_name)
    end
    return _save(p, save_path)
end

"""
    plot_overlap_contributions(ctx, overlaps; save_path=nothing)
    plot_overlap_contributions(h5path, coil_sets; save_path=nothing, kwargs...)

Which poloidal harmonics produced each coil set's overlap, as `conj(V[m])·b̃[m]` rotated so that
the coil's own total is real. The bars therefore sum to its resonant fraction, printed in the
legend.

This is what separates a coil that drives the resonant harmonics weakly from one that drives them
strongly and cancels against itself. The single complex overlap cannot tell those apart.
"""
function plot_overlap_contributions(ctx::EF.ResonantDriveContext, overlaps::AbstractVector{EF.CoilOverlap}; save_path=nothing)
    isempty(overlaps) && return _empty("No coil overlaps to plot")
    m = ctx.rc.m_modes
    v = ctx.dom.right_singular_vectors[:, first(overlaps).mode]
    p = plot(; xlabel="poloidal mode m", ylabel="contribution to overlap",
        title="Per-harmonic contribution, aligned to each coil's own overlap phase", legend=:topleft,
        left_margin=13Plots.mm, bottom_margin=7Plots.mm)
    for (j, o) in enumerate(overlaps)
        o.spectrum_norm_t > 0 || continue
        c = real.(conj.(v) .* o.spectrum .* cis(-angle(o.resonant_field_t))) ./ o.spectrum_norm_t
        ms, as = _step_series(m, c)
        plot!(p, ms, as; seriestype=:steppre, lw=2, c=j,
            label="$(o.coil_name)  (sums to $(round(o.resonant_fraction_percent; digits=1))%)")
    end
    hline!(p, [0]; c=:black, ls=:dot, label="")
    return _save(p, save_path)
end

"""
    plot_surface_overlay(ctx, overlaps; ntheta=256, nzeta=180, levels=4, save_path=nothing)
    plot_surface_overlay(h5path, coil_sets; save_path=nothing, kwargs...)

Each coil set's applied normal field on the control surface, with contours of the dominant resonant
mode over it. One panel per coil, each scaled to its own peak so the shapes are comparable.

This is the view that makes a weak overlap physical rather than numerical: a coil driving a
different poloidal region from the one the dominant mode occupies couples poorly however strong it
is. The poloidal angle is cut at 180° so the outboard midplane sits in the middle of each panel
rather than being split across its edges.
"""
function plot_surface_overlay(ctx::EF.ResonantDriveContext, overlaps::AbstractVector{EF.CoilOverlap};
    ntheta::Int=256, nzeta::Int=180, levels::Int=4, save_path=nothing)
    isempty(overlaps) && return _empty("No coil overlaps to plot")
    m, n = ctx.rc.m_modes, ctx.rc.n_modes
    v = ctx.dom.right_singular_vectors[:, first(overlaps).mode]
    vmap = FT.reconstruct_bn(v, m, n; ntheta, nzeta)
    vpeak = maximum(abs, vmap.bn)
    lv = vpeak > 0 ? range(-0.7, 0.7; length=levels + 2)[2:(end-1)] : [0.0]

    k = length(overlaps)
    rows = ceil(Int, k / 2)
    p = plot(; layout=(rows, min(k, 2)), size=(575 * min(k, 2), 430 * rows),
        left_margin=13Plots.mm, bottom_margin=8Plots.mm)
    for (j, o) in enumerate(overlaps)
        bmap = FT.reconstruct_bn(o.spectrum, m, n; ntheta, nzeta)
        peak = maximum(abs, bmap.bn)
        heatmap!(p[j], rad2deg.(bmap.zeta), rad2deg.(bmap.theta), peak > 0 ? bmap.bn ./ peak : bmap.bn;
            c=:balance, clims=(-1, 1), colorbar=false, yticks=-180:90:180,
            xlabel="toroidal angle φ (deg)", ylabel="poloidal angle θ (deg), 0 = outboard midplane",
            title="$(o.coil_name)   (peak $(round(peak; sigdigits=3)) T)")
        vpeak > 0 && contour!(p[j], rad2deg.(vmap.zeta), rad2deg.(vmap.theta), vmap.bn ./ vpeak;
            levels=lv, c=:black, lw=1.2, colorbar=false)
    end
    return _save(p, save_path)
end

for fn in (:plot_applied_spectra, :plot_overlap_contributions, :plot_surface_overlay)
    @eval function $fn(h5path::AbstractString, coil_sets::AbstractVector{FT.CoilSet}; mode::Int=1,
        psi_low::Real=0.0, psi_high::Real=PE.CORE_PSI_HIGH, nzeta_coil=nothing, mtheta_coil=nothing,
        dat_dir=nothing, kwargs...)
        ctx = EF.ResonantDriveContext(h5path; psi_low, psi_high, nzeta_coil, mtheta_coil, dat_dir)
        return $fn(ctx, EF.coil_overlaps(ctx, coil_sets; mode); kwargs...)
    end
end

"""
    plot_phasing_map(h5path, coil_names; psi_low=0.0, psi_high=CORE_PSI_HIGH, mode=1, nphase=180, quantity=:delta_per_kat, save_path=nothing)
    plot_phasing_map(map::EF.PhasingMap; quantity=:delta_per_kat, save_path=nothing)

Contour map of the dominant-mode overlap per kilo-ampere-turn (`:delta_per_kat`) or of the
resonant fraction of the applied field (`:resonant_fraction_percent`) against the relative phases of two
or three coil arrays (`EF.phasing_map`). Two arrays give a line, three a filled contour whose
axes are the phase of the middle array relative to the first and of the third relative to the
middle; the extreme is marked.
"""
function plot_phasing_map(map::EF.PhasingMap; quantity::Symbol=:delta_per_kat, save_path=nothing)
    arr =
        quantity === :delta_per_kat ? map.delta_per_kat :
        quantity === :resonant_fraction_percent ? map.resonant_fraction_percent :
        throw(ArgumentError("quantity must be :delta_per_kat or :resonant_fraction_percent"))
    label = quantity === :delta_per_kat ? "|δ| per kAt" : "resonant fraction [%]"
    names = map.coil_names
    if length(map.phase_deg) == 1
        p = plot(map.phase_deg[1], vec(arr); lw=2, xlabel="Δφ($(names[2]) − $(names[1])) [deg]", ylabel=label, legend=false,
            title="Two-array phasing", left_margin=12Plots.mm, bottom_margin=6Plots.mm)
    elseif length(map.phase_deg) == 2
        p = contourf(map.phase_deg[1], map.phase_deg[2], permutedims(arr); levels=12, xlabel="Δφ($(names[2]) − $(names[1])) [deg]",
            ylabel="Δφ($(names[3]) − $(names[2])) [deg]", title="Three-array phasing: $label", colorbar_title=label, aspect_ratio=:equal,
            left_margin=12Plots.mm, bottom_margin=6Plots.mm, size=(720, 640))
        v, ph = EF.extreme_phasing(map; quantity)
        scatter!(p, [ph[1]], [ph[2]]; marker=:star5, ms=10, c=:white, label="max $(round(v; sigdigits=3))")
    else
        throw(ArgumentError("plot_phasing_map draws maps of one or two relative phases (two or three arrays)"))
    end
    return _save(p, save_path)
end
function plot_phasing_map(h5path::AbstractString, coil_names::AbstractVector{<:AbstractString}; psi_low::Real=0.0, psi_high::Real=PE.CORE_PSI_HIGH, mode::Int=1,
    nphase::Int=180, quantity::Symbol=:delta_per_kat, save_path=nothing)
    return plot_phasing_map(EF.phasing_map(h5path, coil_names; psi_low, psi_high, mode, nphase); quantity, save_path)
end

"""
    plot_efc_ntv_limits(h5path; torque_budget, delta_threshold=nothing, safety_factor=1.0, delta_max=15, profiles=false, save_path=nothing)
    plot_efc_ntv_limits(couplings::Vector{EF.EFCCoupling}; delta_threshold, torque_budget, kwargs...)

Correction current against intrinsic overlap for each correction array of a run
(`ErrorFields/NTV/`): the linear single-mode current and the NTV-limited current whose residual
torque lowers the threshold, with the largest correctable overlap marked; when the run
tabulated the torques against rotation, a second panel shows those tables with the offsets
found and the reference rotation, and the curve uses the torque balance (`model`,
`rotation_exponent` as in `efc_current_curve`). The zero-rotation limits are marked with the
residual torque (circle) and with the whole field's torque (star); nonzero `torque_rtol` or
`budget_rtol` shade the band between the pessimistic and optimistic curves and the range of the
correctable limit. `delta_threshold` defaults to the run's fitted
penetration threshold (`ErrorFields/Risk/threshold_fit`); `torque_budget` is the torque,
N·m, that would bring the reference rotation to rest. `profiles` adds a panel of the torque
integrated from the axis, `T(ψ)` per kAt² at zero rotation shift for the whole field and the
residual, which shows where in the plasma the torque that survives correction is deposited.
"""
function plot_efc_ntv_limits(couplings::Vector{EF.EFCCoupling}; delta_threshold::Real, torque_budget::Real, safety_factor::Real=1.0,
    delta_max::Real=15, model::Symbol=:auto, rotation_exponent::Real=1.0, torque_rtol::Real=0.0, budget_rtol::Real=0.0, profiles::Bool=false,
    save_path=nothing)
    band = torque_rtol > 0 || budget_rtol > 0
    band_label = "NTV band (T₀ ±$(round(Int, 100budget_rtol)) %, torque ±$(round(Int, 100torque_rtol)) %)"
    p = plot(; xlabel="intrinsic overlap δ_EF / δ_thresh", ylabel="correction current [kAt]", legend=:topleft,
        title="Error-field correction against its own NTV torque (budget $(torque_budget) N·m)", left_margin=12Plots.mm, bottom_margin=6Plots.mm)
    for (j, c) in enumerate(couplings)
        curve = EF.efc_current_curve(c; delta_threshold, torque_budget, safety_factor, delta_max, model, rotation_exponent, torque_rtol, budget_rtol)
        x = curve.delta_ef ./ delta_threshold
        tag = curve.model === :torque_balance ? "torque balance" : "linear budget"
        if band
            lo, hi = curve.limits_pessimistic.with_ntv, curve.limits_optimistic.with_ntv
            isfinite(lo) && vspan!(p, [lo, min(hi, x[end] * delta_threshold)] ./ delta_threshold; c=j, alpha=0.12, lw=0, label="")
            plot!(p, x, curve.current_ntv_pessimistic; lw=1, c=j, alpha=0.5, label="$(c.coil_name) $band_label")
            plot!(p, x, curve.current_ntv_optimistic; lw=1, c=j, alpha=0.5, label="")
        end
        plot!(p, x, curve.current_linear; lw=2, c=j, label="$(c.coil_name) single-mode")
        plot!(p, x, curve.current_ntv; lw=2, ls=:dash, c=j, label="$(c.coil_name) with residual NTV ($tag)")
        plot!(p, x, curve.current_ntv_upper; lw=1, ls=:dashdot, c=j, alpha=0.7, label="$(c.coil_name) over-correction edge")
        isfinite(curve.with_ntv) && vline!(p, [curve.with_ntv / delta_threshold]; ls=:dot, c=j, label="$(c.coil_name) correctable limit")
        isfinite(curve.residual_only) && scatter!(p, [curve.residual_only / delta_threshold], [0.0]; marker=:circle, ms=6, c=j,
            label="$(c.coil_name) zero rotation, residual torque")
        isfinite(curve.torque_only) && scatter!(p, [curve.torque_only / delta_threshold], [0.0]; marker=:star5, ms=9, c=j,
            label="$(c.coil_name) zero rotation, whole-field torque")
    end
    panels = [p]
    scanned = filter(EF.has_rotation_scan, couplings)
    isempty(scanned) || push!(panels, _ntv_rotation_panel(scanned))
    profiles && push!(panels, _ntv_profile_panel(couplings))
    length(panels) == 1 && return _save(p, save_path)
    return _save(plot(panels...; layout=(1, length(panels)), size=(750 * length(panels), 550)), save_path)
end

# The tabulated torques against the rotation shift, with the balance's reference rotation and the found offsets.
function _ntv_rotation_panel(scanned)
    q = plot(; xlabel="E×B rotation shift Δω [krad/s]", ylabel="NTV torque per kAt² [N·m]", legend=:topright, title="Torque against rotation",
        left_margin=12Plots.mm, bottom_margin=6Plots.mm)
    hline!(q, [0.0]; c=:gray, label="")
    for (j, c) in enumerate(scanned)
        x = c.rotation_shift ./ 1e3
        plot!(q, x, c.torque_full_scan; lw=2, c=j, marker=:circle, ms=3, label="$(c.coil_name) whole field")
        plot!(q, x, c.torque_residual_scan; lw=2, ls=:dash, c=j, marker=:diamond, ms=3, label="$(c.coil_name) residual")
        for z in EF.torque_zero_crossings(c)
            vline!(q, [z / 1e3]; ls=:dot, c=j, label="")
        end
        isfinite(c.omega_offset_estimate) &&
            vline!(q, [-abs(c.omega_offset_estimate) / 1e3, abs(c.omega_offset_estimate) / 1e3]; ls=:dashdot, c=:gray, label=j == 1 ? "±|rough offset| (offset_factor·ω_*T)" : "")
        isfinite(c.omega_reference) && vline!(q, [-c.omega_reference / 1e3]; ls=:solid, c=:black, alpha=0.4, label=j == 1 ? "rotation brought to rest (−ω_ref)" : "")
    end
    return q
end

# The torque integrated from the axis at zero rotation shift, whole field against residual.
function _ntv_profile_panel(couplings)
    r = plot(; xlabel="normalized flux ψ_N", ylabel="NTV torque per kAt² integrated from the axis [N·m]", legend=:topleft,
        title="Where the torque is deposited (zero rotation shift)", left_margin=12Plots.mm, bottom_margin=6Plots.mm)
    for (j, c) in enumerate(couplings)
        isempty(c.psi) && continue
        i0 = findfirst(==(0.0), c.rotation_shift)
        i0 === nothing && (i0 = argmin(abs.(c.rotation_shift)))
        plot!(r, c.psi, c.torque_full_profile[:, i0]; lw=2, c=j, label="$(c.coil_name) whole field")
        plot!(r, c.psi, c.torque_residual_profile[:, i0]; lw=2, ls=:dash, c=j, label="$(c.coil_name) residual")
    end
    return r
end
function plot_efc_ntv_limits(h5path::AbstractString; torque_budget::Real, delta_threshold=nothing, kwargs...)
    couplings = EF.read_efc_couplings(h5path)
    # A run may carry [ErrorFields.NTV] without [ErrorFields.scenario], in which case there is no
    # stored threshold. Say what to do about it rather than letting HDF5 raise a bare KeyError.
    δt = delta_threshold
    if δt === nothing
        key = "$(EF._RISK_GROUP)/threshold_fit"
        δt = h5open(h5path, "r") do f
            haskey(f, key) || throw(
                ArgumentError(
                    "$h5path has no $key, so there is no fitted penetration threshold to scale by. " *
                    "Pass delta_threshold, or re-run with an [ErrorFields.scenario] table.")
            )
            read(f[key])
        end
    end
    return plot_efc_ntv_limits(couplings; delta_threshold=δt, torque_budget, kwargs...)
end

"""
    plot_correction_requirement(req::EF.CorrectionRequirement; save_path=nothing)
    plot_correction_requirement(h5path, source_name, array_names; save_path=nothing, kwargs...)

Two panels of a `ErrorFields.correction_requirement`: the resonant field on each rational
surface before correction, after each array's dominant-mode correction alone and after the joint
least-squares correction, as stems on a logarithmic axis per surface; then one polar panel per
array with the complex current it needs as a phasor, magnitude as radius in kilo-ampere-turns (or
as a factor on the array's current as given when any array's ampere-turns are unknown) and toroidal phase
as angle: the array's solo dominant-mode factor in grey, the minimum-current member of the set's
dominant-mode family in blue, and the joint least-squares factor in red. A joint factor far from
the solo one, or a surface left with most of its field, says the single-mode picture of the
correction is not the whole story.
"""
function plot_correction_requirement(req::EF.CorrectionRequirement; save_path=nothing)
    n = length(req.rational_psi)
    labels = ["$(m)/$(nn)\nψ=$(round(x; digits=3))" for (m, nn, x) in zip(req.rational_m, req.rational_n, req.rational_psi)]
    # One slot per surface in ψ order rather than a position at ψ itself, so neighbouring edge surfaces do not overprint.
    p = plot(; xlabel="rational surface (m/n, ψ_N)", ylabel="|resonant field| [T]", title="Resonant field per surface: $(req.source_name) and its correction",
        xticks=(1:n, labels), legend=:outerright, size=(1000, 520), left_margin=12Plots.mm, bottom_margin=12Plots.mm, titlefontsize=12)
    vals = vcat(abs.(req.resonant_field_source_t), vec(abs.(req.resonant_field_dominant_t)), abs.(req.resonant_field_least_squares_t))
    ylo = _log_floor!(p, vals)
    ylo === nothing && return _empty("No finite resonant field to draw")
    k = length(req.array_names)
    width = 0.8 / (k + 2)
    offsets = ((1:(k+2)) .- (k + 3) / 2) .* width
    _stems!(p, (1:n) .+ offsets[1], abs.(req.resonant_field_source_t), ylo; c=:black, label="source as built", marker=:circle, ms=6, lw=3)
    for (j, nm) in enumerate(req.array_names)
        _stems!(p, (1:n) .+ offsets[j+1], abs.(req.resonant_field_dominant_t[:, j]), ylo; c=j, label="after $nm, dominant mode only", marker=:diamond, ms=5, lw=2)
    end
    _stems!(p, (1:n) .+ offsets[end], abs.(req.resonant_field_least_squares_t), ylo; c=:red, label="after joint least squares", marker=:star5, ms=7, lw=2)
    # One polar panel per array: a factor is a complex multiple of the array's current, so its magnitude
    # is the radius and its toroidal phase the angle, which bars with printed phases only hint at.
    panels = Plots.Plot[p]
    # In kilo-ampere-turns when every array's ampere-turns are known, so arrays with different pattern
    # currents share one radial unit; otherwise as a factor on each array's current as given.
    in_kat = all(isfinite, req.current_least_squares_kat)
    unit = in_kat ? "kAt" : "× its current as given"
    for (j, nm) in enumerate(req.array_names)
        fd, fm, fl =
            in_kat ? (req.current_dominant_kat[j], req.current_dominant_minimum_norm_kat[j], req.current_least_squares_kat[j]) :
            (req.current_factor_dominant[j], req.current_factor_dominant_minimum_norm[j], req.current_factor_least_squares[j])
        q = plot(; proj=:polar, title="$nm: needed current [$unit]", titlefontsize=10, legend=:outerbottom, legendfontsize=7)
        for (f, lbl, col, w) in
            ((fd, "dominant mode, this array alone", :gray40, 4), (fm, "dominant mode, minimum current over the set", :blue, 2.5), (fl, "joint least squares", :red, 2))
            isfinite(f) || continue
            plot!(q, [angle(f), angle(f)], [0.0, abs(f)]; lw=w, c=col, label="$lbl: $(round(abs(f); sigdigits=3)) $unit ∠ $(round(rad2deg(angle(f)); digits=0))°")
            scatter!(q, [angle(f)], [abs(f)]; ms=6, c=col, label="")
        end
        push!(panels, q)
    end
    n_pol = length(panels) - 1
    return _save(plot(panels...; layout=Plots.grid(1, n_pol + 1; widths=vcat(0.5, fill(0.5 / n_pol, n_pol))), size=(1000 + 420 * n_pol, 540)), save_path)
end
plot_correction_requirement(h5path::AbstractString, source_name::AbstractString, array_names; save_path=nothing, kwargs...) =
    plot_correction_requirement(EF.correction_requirement(h5path, source_name, array_names; kwargs...); save_path)

"""
    plot_needed_current(mc, array::EF.CoilOverlap; coupling=nothing, delta_threshold=nothing, torque_budget=nothing, save_path=nothing)

The distribution of the factor on `array`'s current that the sampled machine needs to cancel its
dominant-mode overlap (`ErrorFields.needed_current_distribution`), with the as-designed
factor marked and, when the array's `EFCCoupling`, the threshold and the torque budget are given,
the NTV-limited correctable overlap as a vertical line and the fraction beyond it
(`ErrorFields.uncorrectable_probability`) in the legend.
"""
function plot_needed_current(mc::EF.MonteCarloResult, array::EF.CoilOverlap; coupling=nothing, delta_threshold=nothing, torque_budget=nothing, save_path=nothing)
    d = EF.needed_current_distribution(mc, array)
    # Kilo-ampere-turns when the array's ampere-turns are known, else a factor on its current as given.
    in_kat = isfinite(array.ampere_turns_kat)
    per_delta = (in_kat ? array.ampere_turns_kat : 1.0) / abs(array.delta)
    c = in_kat ? _centers(d.current_bin_edges_kat) : _centers(d.current_factor_bin_edges)
    xlabel = in_kat ? "needed current on $(array.coil_name) [kAt]" : "needed factor on $(array.coil_name)'s current (|δ| / |δ_$(array.coil_name)|)"
    p = plot(; xlabel, ylabel="probability density", title="Correction current the sampled machine needs", legend=:topright, left_margin=12Plots.mm,
        bottom_margin=6Plots.mm)
    plot!(p, c, in_kat ? d.current_pdf_per_kat : d.current_factor_pdf; lw=2, c=1, label="intrinsic |δ| over the tolerance samples")
    vline!(p, [mc.abs_delta_total_as_designed * per_delta]; ls=:dot, c=:black, label="as designed")
    if coupling !== nothing
        (delta_threshold === nothing || torque_budget === nothing) && throw(ArgumentError("give delta_threshold and torque_budget with the coupling"))
        u = EF.uncorrectable_probability(mc, coupling; delta_threshold, torque_budget)
        isfinite(u.abs_delta_max_correctable) && vline!(p, [u.abs_delta_max_correctable * per_delta]; ls=:dash, c=:red, lw=2,
            label="NTV-limited correctable overlap ($(u.model)); $(round(u.probability_percent; sigdigits=2)) % of samples beyond")
    end
    return _save(p, save_path)
end

"""
    plot_error_field_summary(h5path; save_path=nothing)

Four panels of a run's error-field assessment: coil sensitivities to shift, the tolerance
Monte Carlo distributions, the overlap distribution against the threshold, and the locking
risk against tolerance scale. Panels whose data the run did not produce say so.
"""
function plot_error_field_summary(h5path::AbstractString; save_path=nothing)
    src = _sources(h5path)
    p = plot(plot_coil_sensitivities(src), plot_tolerance_pdf(src), plot_threshold_scaling(src), plot_locking_risk(src);
        layout=(2, 2), size=(1400, 1000))
    return _save(p, save_path)
end

end # module ErrorFields
