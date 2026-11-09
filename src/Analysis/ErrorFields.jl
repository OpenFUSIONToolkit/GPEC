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
using LinearAlgebra
using Random
using TOML

import ...ErrorFields as EF
import ...PerturbedEquilibrium as PE
import ...ForcingTerms as FT
using ..Analysis: register!, _require, _can, _missing, _entry, _why, capabilities, _draw

"""
A label paired with something to plot: a `gpec.h5` path, or a result already in memory
(`ErrorFields.ErrorFieldsResult`, `SensitivityTable`, `CoilSensitivities`, `MonteCarloResult`, `RiskResult`, or a
vector of `CoilOverlap`). Analysing coil geometry that was never part of a run produces the
latter, so the plots accept both.
"""
const Sources = AbstractVector{<:Pair{String,<:Any}}

"""
A single unlabelled source: a `gpec.h5` path, or one in-memory result.
"""
const SingleSource = Union{AbstractString,EF.ErrorFieldsResult,EF.SensitivityTable,EF.CoilSensitivities,EF.MonteCarloResult,EF.RiskResult,
    AbstractVector{EF.CoilOverlap},PE.PlasmaResponse,PE.PerturbedEquilibriumState}

_sources(source) = [_default_label(source) => source]
# A single source is drawn without a label; several are told apart by theirs.
_default_label(_) = ""

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

# The (m, n) of each control-surface harmonic, as the writer records them in Info/mn_index.
_mn_index(ffs) = hcat([(i - 1) % ffs.mpert + ffs.mlow for i in 1:ffs.numpert_total], [(i - 1) ÷ ffs.mpert + ffs.nlow for i in 1:ffs.numpert_total])

# The dominant resonant-coupling modes of a plasma response, or of the response a perturbed equilibrium was driven on.
_dominant(resp::PE.PlasmaResponse) = (; mn_index=_mn_index(resp.ffs), dominant_v=isempty(resp.dominant_singular_values) ? nothing : resp.dominant_right_singular_vectors,
    singular_values=isempty(resp.dominant_singular_values) ? nothing : resp.dominant_singular_values)
_load(resp::PE.PlasmaResponse) = _load(_present(_dominant(resp)))
_load(s::PE.PerturbedEquilibriumState) = _load(s.response)

# The whole assessment: the dominant-mode table and mode labels as the writer records them, from the plasma response it was made on.
function _load(r::EF.ErrorFieldsResult)
    # The dominant-mode table's shift and tilt sensitivities replace the full per-mode arrays, as in the file.
    parts = Any[_present(_load(r.sensitivities)), _present(_load(EF.sensitivity_table(r.sensitivities, r.dominant; mode=1)))]
    r.monte_carlo === nothing || push!(parts, _present(_load(r.monte_carlo)))
    r.risk === nothing || push!(parts, _present(_load(r.risk)))
    sc = r.scan
    sc === nothing || push!(parts,
        (; scan_scale=sc.tolerance_scale, scan_locking_probability_percent=sc.locking_probability_percent,
            scan_locking_probability_efc_percent=sc.locking_probability_efc_percent, scan_locking_probability_spread_percent=sc.locking_probability_spread_percent,
            scan_locking_probability_efc_spread_percent=sc.locking_probability_efc_spread_percent))
    ctrl = r.settings.ctrl
    push!(
        parts,
        _present((; tolerances=r.tolerances, fd_step_shift_m=ctrl.fd_step_shift_m, fd_step_tilt_deg=ctrl.fd_step_tilt_deg, _dominant(r.response)...))
    )
    return _load(merge(parts...))
end

_load(ovs::AbstractVector{EF.CoilOverlap}) = _load((; coil_names=[o.coil_name for o in ovs],
    delta_as_designed=[o.delta for o in ovs], resonant_fraction_percent=[o.resonant_fraction_percent for o in ovs],
    b_t0=isempty(ovs) ? nothing : first(ovs).b_t0))

# The finite-difference steps the run's sensitivities were taken at, from the recorded deck; a
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

_centers(edges) = (edges[1:end-1] .+ edges[2:end]) ./ 2

# A registered source (a path or the assessment) must hold the data a single-source plot draws; other in-memory sources are what they are.
_check(fn, src::Union{AbstractString,EF.ErrorFieldsResult,PE.PlasmaResponse,PE.PerturbedEquilibriumState}) = _require(fn, src)
_check(fn, src) = nothing

# Below this fraction of the largest value, a per-coil value is roundoff (an axisymmetric coil has no n ≠ 0 field).
const _ROUNDOFF = 1e-10

# The NTV correction arrays, which are swept but are not error-field sources.
_efc_names(r::EF.ErrorFieldsResult) = r.couplings === nothing ? String[] : [c.coil_name for c in r.couplings]
_efc_names(h5path::AbstractString) = h5open(f -> haskey(f, EF._NTV_GROUP) ? read(f["$(EF._NTV_GROUP)/coil_name"]) : String[], h5path)
_efc_names(_) = String[]

# The penetration-threshold samples the locking risk used, rebuilt from the recorded scaling, seed and scenario and
# sorted, so the threshold's CDF can be drawn over any |δ| range; `nothing` when the source does not record them.
function _threshold_samples(r::EF.ErrorFieldsResult)
    (r.risk === nothing || r.settings.scenario === nothing) && return nothing
    rc = r.settings.risk
    ffs = r.response.ffs
    n = EF.single_toroidal_mode(ffs.nlow, ffs.nhigh; where="the threshold of this assessment")
    sc = EF.threshold_scaling(; n, year=rc.year, dataset=rc.dataset, fit=rc.fit)
    return sort(EF.threshold_samples(Random.Xoshiro(rc.seed), sc, r.settings.scenario; nsample=rc.nsample_threshold, dist=rc.distribution))
end

function _threshold_samples(h5path::AbstractString)
    deck = h5open(f -> haskey(f, "Input/gpec_toml_raw") ? TOML.parse(read(f["Input/gpec_toml_raw"])) : nothing, h5path)
    deck === nothing && return nothing
    ef = get(deck, "ErrorFields", Dict{String,Any}())
    haskey(ef, "scenario") || return nothing
    rk = get(ef, "Risk", Dict{String,Any}())
    rc = EF.RiskControl()
    sc = EF.threshold_scaling(; n=EF.scaling_toroidal_mode(h5path), year=get(rk, "year", rc.year), dataset=get(rk, "dataset", rc.dataset),
        fit=get(rk, "fit", rc.fit))
    scen = EF.ScenarioParameters(h5path; (Symbol(k) => v for (k, v) in ef["scenario"])...)
    return sort(
        EF.threshold_samples(Random.Xoshiro(get(rk, "seed", rc.seed)), sc, scen; nsample=get(rk, "nsample_threshold", rc.nsample_threshold),
            dist=get(rk, "distribution", rc.distribution))
    )
end

_threshold_samples(_) = nothing

# Drawing functions, defined by the GPECMakieExt extension.
function _plot_coil_sensitivities end
function _plot_tolerance_pdf end
function _plot_overlap_phasors end
function _plot_tolerance_budget end
function _plot_linearity_residuals end
function _plot_linearity_check end
function _plot_locking_risk end
function _plot_threshold_scaling end
function _plot_dominant_mode_spectrum end
function _plot_applied_spectra end
function _plot_overlap_contributions end
function _plot_surface_overlay end
function _plot_phasing_map end
function _plot_efc_ntv_limits end
function _plot_correction_requirement end
function _plot_needed_current end
function _plot_error_field_summary end

# Each loaded source with its label; a single source carries none.
_labelled(sources) = [(lbl, _load(src)) for (lbl, src) in sources]

"""
    plot_coil_sensitivities(sources; quantity=:shift, coils=nothing, yscale=:identity, save_path=nothing)
    plot_coil_sensitivities(source; kwargs...)

Plot each coil set's dominant-mode sensitivity as a stem and dot per row, across runs matched by coil set name.
`quantity` is `:shift` (per millimetre of rigid in-plane shift), `:tilt` (per degree), `:rim` (the same tilt as rim
displacement, the unit mechanical tolerances arrive in), `:as_designed` (`|δ|` as built), or `:fraction` (the resonant
share of the coil's own applied spectrum, `100·|δ|·B_T0/‖b̃‖` in percent). `coils` restricts and orders the coil sets
shown; a name a source does not carry is an error. `yscale = :log10` puts the values on a log axis. Coil sets whose
value is roundoff (an axisymmetric coil has no n ≠ 0 field) are left out and counted in the title.

Each source is a `gpec.h5` path, an `ErrorFieldsResult`, a `SensitivityTable`, or a vector of `CoilOverlap`
(`:as_designed` and `:fraction` only). `:fraction` needs the applied field as well as the overlap, so it is not
available from a `SensitivityTable`.
"""
function plot_coil_sensitivities(sources::Sources; quantity::Symbol=:shift, coils=nothing, yscale::Symbol=:identity, save_path=nothing)
    quantity in (:shift, :tilt, :rim, :as_designed, :fraction) || throw(ArgumentError("quantity must be :shift, :tilt, :rim, :as_designed, or :fraction"))
    yscale in (:identity, :log10) || throw(ArgumentError("yscale must be :identity or :log10"))
    data = _labelled(sources)
    any(d -> d[2].coil_names === nothing, data) && throw(ArgumentError("plot_coil_sensitivities: a source has no coil sensitivities; run with an [ErrorFields] section"))
    names = coils === nothing ? data[1][2].coil_names : String.(collect(coils))
    values = [_sensitivity_values(lbl, d, names, quantity) for (lbl, d) in data]
    return _draw(_plot_coil_sensitivities, first.(data), names, values, quantity, yscale; save_path)
end
plot_coil_sensitivities(source::SingleSource; kwargs...) = (_check(plot_coil_sensitivities, source); plot_coil_sensitivities(_sources(source); kwargs...))

"""
    plot_tolerance_pdf(sources; corrected=true, normalize=false, xscale=:identity, show_batches=false, save_path=nothing)
    plot_tolerance_pdf(source; kwargs...)

Plot the Monte Carlo distributions of the dominant-mode overlap `|δ|` of each run, intrinsic and (when `corrected`) with
error-field correction, with the as-designed overlap marked unless it is roundoff. `show_batches` draws each batch's intrinsic histogram
faintly under the mean, the sampling noise a risk figure inherits.
"""
function plot_tolerance_pdf(sources::Sources; corrected::Bool=true, normalize::Bool=false, xscale::Symbol=:identity, show_batches::Bool=false,
    save_path=nothing)
    xscale in (:identity, :log10) || throw(ArgumentError("xscale must be :identity or :log10"))
    data = filter(d -> d[2].abs_delta_pdf !== nothing, _labelled(sources))
    isempty(data) && throw(ArgumentError("plot_tolerance_pdf: no source has a tolerance Monte Carlo; run with a tolerance_file"))
    return _draw(_plot_tolerance_pdf, data; corrected, normalize, xscale, show_batches, save_path)
end
plot_tolerance_pdf(source::SingleSource; kwargs...) = (_check(plot_tolerance_pdf, source); plot_tolerance_pdf(_sources(source); kwargs...))

"""
    plot_overlap_phasors(sources; coils=nothing, save_path=nothing)
    plot_overlap_phasors(source; kwargs...)

Plot each error-field coil set's as-built dominant-mode overlap `δ` as an arrow in the complex plane, laid head to
tail so the walk ends at the run's total, rotated so the total is real: an arrow along it adds to the error field,
one against it cancels part of another coil's. One panel per source, since the mode phase is arbitrary between runs.
The NTV correction arrays and coil sets with no as-designed n ≠ 0 field are left out; throws when no coil set has
one. On a file the walk's end is checked against the stored Monte Carlo `delta_as_designed`.
"""
function plot_overlap_phasors(sources::Sources; coils=nothing, save_path=nothing)
    panels = map(sources) do (lbl, src)
        d = _load(src)
        d.delta_as_designed === nothing && throw(ArgumentError("plot_overlap_phasors: source \"$lbl\" has no per-coil overlaps; run with an [ErrorFields] section"))
        names = coils === nothing ? [nm for nm in d.coil_names if nm ∉ _efc_names(src)] : String.(collect(coils))
        z = d.delta_as_designed[_coil_indices(lbl, d, names)]
        total = sum(z)
        if coils === nothing && d.abs_delta_total_as_designed !== nothing && !isapprox(abs(total), d.abs_delta_total_as_designed; rtol=1e-6, atol=eps(Float64))
            @warn "$lbl: the head-to-tail total |Σδ| = $(abs(total)) differs from the stored Monte Carlo abs_delta_total_as_designed $(d.abs_delta_total_as_designed)"
        end
        # Roundoff is judged against the run's error-field scale: every coil set's overlap and the Monte Carlo mean.
        scale = max(maximum(abs, d.delta_as_designed), something(d.abs_delta_sampled_mean, 0.0))
        keep = abs.(z) .> _ROUNDOFF * scale
        any(keep) || throw(ArgumentError("plot_overlap_phasors: no error-field coil set$(isempty(lbl) ? "" : " of \"$lbl\"") has an as-designed n ≠ 0 field"))
        (; label=lbl, names=names[keep], delta=z[keep])
    end
    return _draw(_plot_overlap_phasors, panels; save_path)
end
plot_overlap_phasors(source::SingleSource; kwargs...) = (_check(plot_overlap_phasors, source); plot_overlap_phasors(_sources(source); kwargs...))

"""
    plot_tolerance_budget(h5path; psi_low=0.0, psi_high=CORE_PSI_HIGH, mode=1, tolerance_scale=1.0, sort=:total, top=nothing, save_path=nothing)
    plot_tolerance_budget(table, tolerances, coil_sets; monte_carlo=nothing, kwargs...)
    plot_tolerance_budget(terms::NamedTuple; monte_carlo=nothing, sort=:total, top=nothing, save_path=nothing)

Plot the worst-case tolerance budget as stacked horizontal bars, one per coil set, coherent group and the unattributed
budget: the as-designed `|δ|`, the shift term and the tilt term of `ErrorFields.worst_case_terms`, which sum to the
`abs_delta_worst_case` the Monte Carlo sizes its histogram by. `sort` orders the bars by `:total` (largest at the top),
`:name`, or `:none`; `top` keeps only that many largest. With a Monte Carlo result the sample mean of `|δ|` and the
as-designed total are marked.
"""
function plot_tolerance_budget(terms::NamedTuple; monte_carlo=nothing, sort::Symbol=:total, top=nothing, save_path=nothing)
    sort in (:total, :name, :none) || throw(ArgumentError("sort must be :total, :name, or :none"))
    ng = length(terms.group_names)
    nc = length(terms.coil_names)
    labels = vcat(terms.coil_names, ["group $g" for g in terms.group_names], ["unattributed"])
    parts = (
        ("as designed", vcat(terms.abs_delta_as_designed, zeros(ng), 0.0)),
        ("shift tolerance", vcat(terms.abs_delta_shift_tolerance, terms.abs_delta_group_shift_tolerance, 0.0)),
        ("tilt tolerance", vcat(terms.abs_delta_tilt_tolerance, terms.abs_delta_group_tilt_tolerance, 0.0)),
        ("unattributed", vcat(zeros(nc + ng), terms.abs_delta_unattributed))
    )
    total = sum(p[2] for p in parts)
    order = sort === :total ? sortperm(total) : sort === :name ? sortperm(labels; rev=true) : collect(reverse(eachindex(labels)))
    top === nothing || (order = order[max(1, end - top + 1):end])
    mc = monte_carlo === nothing ? nothing : _load(monte_carlo)
    return _draw(_plot_tolerance_budget, labels[order], [(name, v[order]) for (name, v) in parts], terms.abs_delta_worst_case, mc; save_path)
end
function plot_tolerance_budget(table::EF.SensitivityTable, ts::EF.ToleranceSet, coil_sets::AbstractVector{FT.CoilSet}; tolerance_scale::Real=1.0, kwargs...)
    return plot_tolerance_budget(EF.worst_case_terms(table, ts, collect(coil_sets); tolerance_scale); kwargs...)
end
function plot_tolerance_budget(h5path::AbstractString; psi_low::Real=0.0, psi_high::Real=PE.CORE_PSI_HIGH, mode::Int=1, tolerance_scale::Real=1.0, kwargs...)
    terms = EF.worst_case_terms(h5path; psi_low, psi_high, mode, tolerance_scale)
    d = _load(h5path)
    mc = d.abs_delta_pdf === nothing ? nothing : (; abs_delta_sampled_mean=d.abs_delta_sampled_mean, abs_delta_total_as_designed=d.abs_delta_total_as_designed)
    return plot_tolerance_budget(terms; monte_carlo=mc, kwargs...)
end

"""
    plot_linearity_residuals(sources; at=:tolerance, coils=nothing, tolerances=nothing, fd_step_shift_m=nothing, fd_step_tilt_deg=nothing, yscale=:log10, save_path=nothing)
    plot_linearity_residuals(source; kwargs...)

Plot how linear each coil set's field is in its rigid motions: for the six taps (shift and tilt about `x`, `y`, `z`)
the ratio of the central second difference to the largest first difference among the set's taps, as stored in
`ErrorFields/CoilSensitivities/*_linearity_residual`. `at = :step` shows it at the finite-difference steps; `at = :tolerance` rescales each tap by `tolerance / step` to the coil's own tolerance range. Tolerances come from the file's
snapshot or `tolerances`; steps from the recorded deck or `fd_step_shift_m` and `fd_step_tilt_deg`. The 1 % level at
which the sensitivity calculation warns is marked. One panel per source, on a log axis unless `yscale = :identity`.
"""
function plot_linearity_residuals(sources::Sources; at::Symbol=:tolerance, coils=nothing, tolerances=nothing, fd_step_shift_m=nothing,
    fd_step_tilt_deg=nothing, yscale::Symbol=:log10, save_path=nothing)
    at in (:step, :tolerance) || throw(ArgumentError("at must be :step or :tolerance"))
    yscale in (:identity, :log10) || throw(ArgumentError("yscale must be :identity or :log10"))
    panels = map(_labelled(sources)) do (lbl, d)
        d.shift_linearity_residual === nothing && throw(ArgumentError("plot_linearity_residuals: source \"$lbl\" has no linearity residuals; run with an [ErrorFields] section"))
        names = coils === nothing ? d.coil_names : String.(collect(coils))
        (; label=lbl, names, residual=_linearity_matrix(lbl, d, names, at; tolerances, fd_step_shift_m, fd_step_tilt_deg))
    end
    return _draw(_plot_linearity_residuals, panels, at, yscale; save_path)
end
plot_linearity_residuals(source::SingleSource; kwargs...) = (_check(plot_linearity_residuals, source); plot_linearity_residuals(_sources(source); kwargs...))

"""
    plot_linearity_check(check::EF.LinearityCheck; save_path=nothing)
    plot_linearity_check(h5path; scales=(1.0, 2.0), save_path=nothing, kwargs...)

Plot the relative error of the linear sensitivity model at finite displacements, from `ErrorFields.linearity_check`:
one stem per row on a log axis, grouped by coil set or coherent group, shift and tilt told apart by colour and the
scale by marker, with the 1 % level marked. A row whose linear change vanishes has no relative error and is not drawn.
"""
function plot_linearity_check(check::EF.LinearityCheck; save_path=nothing)
    isempty(check.name) && throw(ArgumentError("plot_linearity_check: the tolerance set names no coil or group"))
    any(v -> isfinite(v) && v > 0, check.relative_error) || throw(ArgumentError("plot_linearity_check: no finite relative error to draw"))
    return _draw(_plot_linearity_check, check; save_path)
end
plot_linearity_check(h5path::AbstractString; save_path=nothing, kwargs...) = plot_linearity_check(EF.linearity_check(h5path; kwargs...); save_path)

"""
    plot_locking_risk(sources; corrected=true, target_percent=nothing, save_path=nothing)
    plot_locking_risk(source; kwargs...)

Plot the locking probability against tolerance scale from each run's `ErrorFields/Risk/ToleranceScan/`, intrinsic and
(when `corrected`) with error-field correction, with the batch spread as error bars, on log-log axes. A scale where no
sample locked is left off. With `target_percent` the target is marked and the title gives the allowable scale where
the scan reaches it.
"""
function plot_locking_risk(sources::Sources; corrected::Bool=true, target_percent=nothing, save_path=nothing)
    data = filter(d -> d[2].scan_scale !== nothing, _labelled(sources))
    isempty(data) && throw(ArgumentError("plot_locking_risk: no source has a tolerance scan; set scan_scales in [ErrorFields.Risk]"))
    return _draw(_plot_locking_risk, data; corrected, target_percent, save_path)
end
plot_locking_risk(source::SingleSource; kwargs...) = (_check(plot_locking_risk, source); plot_locking_risk(_sources(source); kwargs...))

"""
    plot_threshold_scaling(sources; save_path=nothing)
    plot_threshold_scaling(source; kwargs...)

Plot each run's overlap distributions against the penetration threshold on a log `|δ|` axis: the threshold as its
cumulative distribution P(lock | δ) on a 0–1 right axis, with the nominal threshold (the fit at the central exponents,
the CDF's median) marked. The CDF is rebuilt from the recorded scaling and scenario when the source records them, and
otherwise drawn from the stored values on the Monte Carlo bins. A source needs the overlap distribution and the
threshold together; pass a `(MonteCarloResult, RiskResult)` tuple for results in memory.
"""
function plot_threshold_scaling(sources::Sources; save_path=nothing)
    data = [
        (lbl, d, _threshold_samples(src)) for (lbl, src) in sources for d in (_load(src),)
        if d.threshold_pdf !== nothing && d.abs_delta_pdf !== nothing && d.abs_delta_bin_edges !== nothing
    ]
    isempty(data) && throw(ArgumentError("plot_threshold_scaling: no source has both the overlap distribution and the threshold; run with an [ErrorFields.scenario] table"))
    return _draw(_plot_threshold_scaling, data; save_path)
end
plot_threshold_scaling(source::SingleSource; kwargs...) = (_check(plot_threshold_scaling, source); plot_threshold_scaling(_sources(source); kwargs...))

"""
    plot_dominant_mode_spectrum(sources; modes=1:3, save_path=nothing)
    plot_dominant_mode_spectrum(source; kwargs...)

Plot the right singular vectors of the dominant resonant-coupling modes, `|V[m, k]|` against poloidal harmonic m for
each mode `k` in `modes` (one step line per toroidal mode number), labelled with their singular values.
"""
function plot_dominant_mode_spectrum(sources::Sources; modes=1:3, save_path=nothing)
    data = filter(d -> d[2].dominant_v !== nothing && d[2].mn_index !== nothing, _labelled(sources))
    isempty(data) && throw(ArgumentError("plot_dominant_mode_spectrum: no source has the dominant coupling modes"))
    return _draw(_plot_dominant_mode_spectrum, data, collect(modes); save_path)
end
plot_dominant_mode_spectrum(source::SingleSource; kwargs...) = (_check(plot_dominant_mode_spectrum, source); plot_dominant_mode_spectrum(_sources(source); kwargs...))

# The coil sets with a significant as-designed overlap; the axisymmetric ones have no n ≠ 0 spectrum to draw.
function _significant_overlaps(overlaps)
    isempty(overlaps) && throw(ArgumentError("no coil overlaps to plot"))
    big = maximum(o -> abs(o.delta), overlaps)
    keep = [o for o in overlaps if abs(o.delta) > _ROUNDOFF * big]
    isempty(keep) && throw(ArgumentError("no coil set has an as-designed n ≠ 0 field"))
    return keep
end

"""
    plot_applied_spectra(ctx, overlaps; normalize=true, save_path=nothing)
    plot_applied_spectra(h5path, coil_sets; normalize=true, save_path=nothing, kwargs...)

Plot each coil set's applied spectrum against the dominant resonant mode. `normalize = true` scales every curve to its
own peak, which says whether a coil drives a different part of the spectrum; `false` keeps the amplitudes, which says
whether it simply drives less.
"""
function plot_applied_spectra(ctx::EF.ResonantDriveContext, overlaps::AbstractVector{EF.CoilOverlap}; normalize::Bool=true, save_path=nothing)
    ovs = _significant_overlaps(overlaps)
    v = abs.(ctx.dom.right_singular_vectors[:, first(ovs).mode])
    return _draw(_plot_applied_spectra, ctx.rc.m_modes, v, [(o.coil_name, abs.(o.spectrum)) for o in ovs]; normalize, save_path)
end

"""
    plot_overlap_contributions(ctx, overlaps; save_path=nothing)
    plot_overlap_contributions(h5path, coil_sets; save_path=nothing, kwargs...)

Plot which poloidal harmonics produce each coil set's overlap, `conj(V[m])·b̃[m]` rotated so the coil's own total is
real; each coil's steps sum to its resonant fraction, given in the legend. It separates a coil that drives the
resonant harmonics weakly from one that drives them strongly and cancels against itself.
"""
function plot_overlap_contributions(ctx::EF.ResonantDriveContext, overlaps::AbstractVector{EF.CoilOverlap}; save_path=nothing)
    ovs = [o for o in _significant_overlaps(overlaps) if o.spectrum_norm_t > 0]
    v = ctx.dom.right_singular_vectors[:, first(ovs).mode]
    curves = [(o.coil_name, o.resonant_fraction_percent, real.(conj.(v) .* o.spectrum .* cis(-angle(o.resonant_field_t))) ./ o.spectrum_norm_t) for o in ovs]
    return _draw(_plot_overlap_contributions, ctx.rc.m_modes, curves; save_path)
end

"""
    plot_surface_overlay(ctx, overlaps; ntheta=256, nzeta=180, top=4, save_path=nothing)
    plot_surface_overlay(h5path, coil_sets; save_path=nothing, kwargs...)

Plot the applied normal field on the control surface of the `top` coil sets with the largest overlap, each scaled to
its own peak, with contours of the dominant resonant mode over it: a coil driving a different poloidal region from the
one the dominant mode occupies couples poorly however strong it is. Toroidal angle φ against poloidal angle θ, with the
outboard midplane at θ = 0.
"""
function plot_surface_overlay(ctx::EF.ResonantDriveContext, overlaps::AbstractVector{EF.CoilOverlap}; ntheta::Int=256, nzeta::Int=180, top::Int=4,
    save_path=nothing)
    ovs = sort(_significant_overlaps(overlaps); by=o -> -abs(o.delta))[1:min(top, end)]
    m, n = ctx.rc.m_modes, ctx.rc.n_modes
    mode = FT.reconstruct_bn(ctx.dom.right_singular_vectors[:, first(ovs).mode], m, n; ntheta, nzeta)
    maps = [(o.coil_name, FT.reconstruct_bn(o.spectrum, m, n; ntheta, nzeta)) for o in ovs]
    return _draw(_plot_surface_overlay, maps, mode; save_path)
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

Plot the dominant-mode overlap per kilo-ampere-turn (`:delta_per_kat`) or the resonant fraction of the applied field
(`:resonant_fraction_percent`) against the relative toroidal phases of two or three coil arrays (`EF.phasing_map`):
a line for two arrays, a filled contour with the extreme marked for three.
"""
function plot_phasing_map(map::EF.PhasingMap; quantity::Symbol=:delta_per_kat, save_path=nothing)
    quantity in (:delta_per_kat, :resonant_fraction_percent) || throw(ArgumentError("quantity must be :delta_per_kat or :resonant_fraction_percent"))
    length(map.phase_deg) in (1, 2) || throw(ArgumentError("plot_phasing_map draws maps of one or two relative phases (two or three arrays)"))
    extreme = length(map.phase_deg) == 2 ? EF.extreme_phasing(map; quantity) : nothing
    return _draw(_plot_phasing_map, map, quantity, extreme; save_path)
end
function plot_phasing_map(h5path::AbstractString, coil_names::AbstractVector{<:AbstractString}; psi_low::Real=0.0, psi_high::Real=PE.CORE_PSI_HIGH, mode::Int=1,
    nphase::Int=180, quantity::Symbol=:delta_per_kat, save_path=nothing)
    return plot_phasing_map(EF.phasing_map(h5path, coil_names; psi_low, psi_high, mode, nphase); quantity, save_path)
end

"""
    plot_efc_ntv_limits(h5path; torque_budget, delta_threshold=nothing, safety_factor=1.0, delta_max=15, profiles=false, save_path=nothing)
    plot_efc_ntv_limits(couplings::Vector{EF.EFCCoupling}; delta_threshold, torque_budget, kwargs...)

Plot the correction current against intrinsic overlap for each correction array (`ErrorFields/NTV/`): the
single-mode current and the NTV-limited current whose residual torque lowers the threshold, with the largest
correctable overlap marked and the zero-rotation limits as points (residual torque: circle, whole field: star). When
the run tabulated the torques against rotation, a second panel shows those tables with the reference rotation; with
`profiles`, a third the torque integrated from the axis at zero rotation shift. Nonzero `torque_rtol` or
`budget_rtol` shade the band between the pessimistic and optimistic curves. `delta_threshold` defaults to the run's
fitted penetration threshold; `torque_budget` is the torque, N·m, that would bring the reference rotation to rest.
"""
function plot_efc_ntv_limits(couplings::Vector{EF.EFCCoupling}; delta_threshold::Real, torque_budget::Real, safety_factor::Real=1.0,
    delta_max::Real=15, model::Symbol=:auto, rotation_exponent::Real=1.0, torque_rtol::Real=0.0, budget_rtol::Real=0.0, profiles::Bool=false,
    save_path=nothing)
    curves = [(c, EF.efc_current_curve(c; delta_threshold, torque_budget, safety_factor, delta_max, model, rotation_exponent, torque_rtol, budget_rtol))
              for c in couplings]
    return _draw(_plot_efc_ntv_limits, curves, Float64(delta_threshold), Float64(torque_budget); band=torque_rtol > 0 || budget_rtol > 0, profiles,
        save_path)
end
function plot_efc_ntv_limits(h5path::AbstractString; torque_budget::Real, delta_threshold=nothing, kwargs...)
    couplings = EF.read_efc_couplings(h5path)
    # A run may carry [ErrorFields.NTV] without [ErrorFields.scenario], in which case there is no stored threshold.
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

Plot an `ErrorFields.correction_requirement`: the resonant field on each rational surface before correction, after
each array's dominant-mode correction alone and after the joint least-squares correction, as stems on a log axis; then
one polar panel per array with the complex current it needs as a phasor (kilo-ampere-turns, or a factor on the array's
current as given when any array's ampere-turns are unknown): the solo dominant-mode factor, the minimum-current member
of the set's dominant-mode family, and the joint least-squares factor.
"""
function plot_correction_requirement(req::EF.CorrectionRequirement; save_path=nothing)
    vals = vcat(abs.(req.resonant_field_source_t), vec(abs.(req.resonant_field_dominant_t)), abs.(req.resonant_field_least_squares_t))
    any(v -> isfinite(v) && v > 0, vals) || throw(ArgumentError("plot_correction_requirement: no finite resonant field to draw"))
    return _draw(_plot_correction_requirement, req; save_path)
end
plot_correction_requirement(h5path::AbstractString, source_name::AbstractString, array_names; save_path=nothing, kwargs...) =
    plot_correction_requirement(EF.correction_requirement(h5path, source_name, array_names; kwargs...); save_path)

"""
    plot_needed_current(mc, array::EF.CoilOverlap; coupling=nothing, delta_threshold=nothing, torque_budget=nothing, save_path=nothing)

Plot the distribution of the current `array` needs to cancel each sampled machine's dominant-mode overlap
(`ErrorFields.needed_current_distribution`), in kilo-ampere-turns when the array's ampere-turns are known, with the
as-designed value marked; with the array's `EFCCoupling`, the threshold and the torque budget, also the NTV-limited
correctable current and the fraction of samples beyond it (`ErrorFields.uncorrectable_probability`).
"""
function plot_needed_current(mc::EF.MonteCarloResult, array::EF.CoilOverlap; coupling=nothing, delta_threshold=nothing, torque_budget=nothing, save_path=nothing)
    limit = nothing
    if coupling !== nothing
        (delta_threshold === nothing || torque_budget === nothing) && throw(ArgumentError("give delta_threshold and torque_budget with the coupling"))
        limit = (EF.uncorrectable_probability(mc, coupling; delta_threshold, torque_budget)..., torque_budget=Float64(torque_budget))
    end
    return _draw(_plot_needed_current, mc, array, EF.needed_current_distribution(mc, array), limit; save_path)
end

"""
    plot_error_field_summary(src; save_path=nothing)

Plot the panels of a run's error-field assessment it supplies: the overlap distribution against the threshold's CDF,
titled with the locking probability; the locking risk against tolerance scale, titled with the allowable scale for 1 %
risk; and the coil sets' shift sensitivity. `src` is an `ErrorFields.ErrorFieldsResult` or the path of a `gpec.h5` it
was saved to.
"""
function plot_error_field_summary(source::Union{AbstractString,EF.ErrorFieldsResult}; save_path=nothing)
    _require(plot_error_field_summary, source)
    caps = capabilities(source)
    shown = [fn for fn in (plot_threshold_scaling, plot_locking_risk, plot_coil_sensitivities) if _can(fn, caps)]
    d = _load(source)
    threshold = plot_threshold_scaling in shown ? _threshold_samples(source) : nothing
    keep = plot_coil_sensitivities in shown ? [nm ∉ _efc_names(source) for nm in d.coil_names] : nothing
    return _draw(_plot_error_field_summary, d, threshold, plot_threshold_scaling in shown, plot_locking_risk in shown, keep; save_path)
end

const _EF_SOURCES = (EF.ErrorFieldsResult, AbstractString)
register!(plot_error_field_summary, _EF_SOURCES, (), "overlap against the threshold, locking risk and coil sensitivities";
    any_of=((:ef_monte_carlo, :ef_risk), (:ef_scan,), (:ef_sensitivities,)))
register!(plot_coil_sensitivities, _EF_SOURCES, (:ef_sensitivities,), "per-coil dominant-mode sensitivities")
register!(plot_overlap_phasors, _EF_SOURCES, (:ef_sensitivities,), "as-designed overlap δ of each coil set in the complex plane")
register!(plot_linearity_residuals, _EF_SOURCES, (:ef_sensitivities,), "linearity residuals of the coil sensitivities")
register!(plot_dominant_mode_spectrum, (_EF_SOURCES..., PE.PlasmaResponse, PE.PerturbedEquilibriumState), (:dominant_mode,),
    "spectra of the dominant resonant-coupling modes")
register!(plot_tolerance_pdf, _EF_SOURCES, (:ef_monte_carlo,), "Monte Carlo distribution of |δ|")
register!(plot_threshold_scaling, _EF_SOURCES, (:ef_monte_carlo, :ef_risk), "|δ| distribution against the threshold's CDF")
register!(plot_locking_risk, _EF_SOURCES, (:ef_scan,), "locking risk against tolerance scale")

end # module ErrorFields
