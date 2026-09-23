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

import ...ErrorFields as EF
import ...PerturbedEquilibrium as PE
import ...ForcingTerms as FT

"""
A label paired with something to plot: a `gpec.h5` path, or a result already in memory
(`ErrorFields.SensitivityTable`, `MonteCarloResult`, `RiskResult`, or a vector of `CoilOverlap`).
Analysing coil geometry that was never part of a run produces the latter, so the plots accept both.
"""
const Sources = AbstractVector{<:Pair{String,<:Any}}

"""A single unlabelled source: a `gpec.h5` path, or one in-memory result."""
const SingleSource = Union{AbstractString,EF.SensitivityTable,EF.MonteCarloResult,EF.RiskResult,AbstractVector{EF.CoilOverlap}}

_sources(source) = [_default_label(source) => source]
_default_label(h5path::AbstractString) = basename(dirname(abspath(h5path)))
_default_label(_) = "in memory"

# Every field the plots read. One template so the HDF5 and in-memory loaders cannot drift apart.
const _BLANK = (; coil_names=nothing, delta_nominal=nothing, delta_per_mm_shift=nothing,
    delta_per_deg_tilt=nothing, delta_per_mm_rim=nothing, shift_sensitivity=nothing, tilt_sensitivity=nothing,
    bin_edges=nothing, pdf=nothing, pdf_efc=nothing, mc_delta_nominal=nothing, threshold_pdf=nothing,
    p_lock_given_delta=nothing, threshold_nominal=nothing, plock=nothing, plock_efc=nothing,
    scan_scale=nothing, scan_plock=nothing, scan_plock_efc=nothing, scan_spread=nothing,
    scan_spread_efc=nothing, dominant_v=nothing, singular_values=nothing, mn_index=nothing)

_load(d::NamedTuple) = merge(_BLANK, d)
_load(h5path::AbstractString) = _load(_load_h5(h5path))

_load(t::EF.SensitivityTable) = _load((; coil_names=t.coil_names, delta_nominal=t.delta_nominal,
    delta_per_mm_shift=t.delta_per_mm_shift, delta_per_deg_tilt=t.delta_per_deg_tilt,
    delta_per_mm_rim=t.delta_per_mm_rim, shift_sensitivity=t.shift, tilt_sensitivity=t.tilt))

_load(r::EF.MonteCarloResult) = _load((; bin_edges=r.bin_edges, pdf=r.pdf, pdf_efc=r.pdf_efc,
    mc_delta_nominal=r.delta_nominal))

_load(r::EF.RiskResult) = _load((; threshold_pdf=r.threshold_pdf, p_lock_given_delta=r.p_lock_given_delta,
    plock=r.plock, plock_efc=r.plock_efc))

_load(ovs::AbstractVector{EF.CoilOverlap}) = _load((; coil_names=[o.coil_name for o in ovs],
    delta_nominal=[o.delta for o in ovs]))

function _load_h5(h5path::AbstractString)
    h5open(h5path, "r") do f
        has(k) = haskey(f, k)
        cs = "ErrorFields/CoilSensitivities"
        mc = "ErrorFields/MonteCarlo"
        rk = "ErrorFields/Risk"
        (
            coil_names=has(cs) ? read(f["$cs/coil_name"]) : nothing,
            delta_nominal=has(cs) ? read(f["$cs/DominantMode/delta_nominal"]) : nothing,
            delta_per_mm_shift=has(cs) ? read(f["$cs/DominantMode/delta_per_mm_shift"]) : nothing,
            delta_per_deg_tilt=has(cs) ? read(f["$cs/DominantMode/delta_per_deg_tilt"]) : nothing,
            delta_per_mm_rim=has(cs) ? read(f["$cs/DominantMode/delta_per_mm_rim"]) : nothing,
            shift_sensitivity=has(cs) ? read(f["$cs/DominantMode/shift_sensitivity"]) : nothing,
            tilt_sensitivity=has(cs) ? read(f["$cs/DominantMode/tilt_sensitivity"]) : nothing,
            bin_edges=has(mc) ? read(f["$mc/bin_edges"]) : nothing,
            pdf=has(mc) ? read(f["$mc/pdf"]) : nothing,
            pdf_efc=has(mc) ? read(f["$mc/pdf_efc"]) : nothing,
            mc_delta_nominal=has(mc) ? read(f["$mc/delta_nominal"]) : nothing,
            threshold_pdf=has(rk) ? read(f["$rk/threshold_pdf"]) : nothing,
            p_lock_given_delta=has(rk) ? read(f["$rk/p_lock_given_delta"]) : nothing,
            threshold_nominal=has(rk) ? read(f["$rk/threshold_nominal"]) : nothing,
            plock=has(rk) ? read(f["$rk/plock_percent"]) : nothing,
            plock_efc=has(rk) ? read(f["$rk/plock_efc_percent"]) : nothing,
            scan_scale=has("$rk/ToleranceScan") ? read(f["$rk/ToleranceScan/scale"]) : nothing,
            scan_plock=has("$rk/ToleranceScan") ? read(f["$rk/ToleranceScan/plock_percent"]) : nothing,
            scan_plock_efc=has("$rk/ToleranceScan") ? read(f["$rk/ToleranceScan/plock_efc_percent"]) : nothing,
            scan_spread=has("$rk/ToleranceScan") ? read(f["$rk/ToleranceScan/plock_spread_percent"]) : nothing,
            scan_spread_efc=has("$rk/ToleranceScan") ? read(f["$rk/ToleranceScan/plock_efc_spread_percent"]) : nothing,
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
    plot_coil_sensitivities(sources; quantity=:shift, coils=nothing, save_path=nothing)
    plot_coil_sensitivities(source; kwargs...)

Grouped bars of the per-coil-set dominant-mode sensitivity across runs, matched by coil set name.
`quantity` is `:shift` (per millimetre of rigid in-plane shift), `:tilt` (per degree), `:rim` (the
same tilt as rim displacement, the unit mechanical tolerances arrive in), or `:nominal` (`|δ|` as
built). `coils` restricts and orders the coil sets shown.

Each source is a `gpec.h5` path or a `SensitivityTable` already in memory, so a sweep of coil
geometry that was never part of a run plots the same way a stored run does.
"""
function plot_coil_sensitivities(sources::Sources; quantity::Symbol=:shift, coils=nothing, save_path=nothing)
    data = [(lbl, _load(src)) for (lbl, src) in sources]
    any(d -> d[2].coil_names === nothing, data) && return _empty("No ErrorFields/CoilSensitivities data — run with an [ErrorFields] section")
    names = coils === nothing ? data[1][2].coil_names : String.(collect(coils))
    field = quantity === :shift ? :delta_per_mm_shift : quantity === :tilt ? :delta_per_deg_tilt :
            quantity === :rim ? :delta_per_mm_rim : quantity === :nominal ? :delta_nominal :
            throw(ArgumentError("quantity must be :shift, :tilt, :rim, or :nominal"))
    value(d, nm) = begin
        i = findfirst(==(nm), d.coil_names)
        (i === nothing || getfield(d, field) === nothing) && return NaN
        abs(getfield(d, field)[i])
    end
    ylabel = quantity === :shift ? "|δ| per mm of shift" : quantity === :tilt ? "|δ| per degree of tilt" :
             quantity === :rim ? "|δ| per mm of rim displacement" : "|δ_nominal|"
    per = quantity === :shift ? "shift" : quantity === :tilt ? "tilt" : "rim displacement"
    title = quantity === :nominal ? "Nominal dominant-mode overlap" : "Dominant-mode error field per $per"
    n = length(names)
    k = length(data)
    width = 0.8 / k
    p = plot(; xlabel="coil set", ylabel=ylabel, title=title, xticks=(1:n, names), xrotation=45, legend=:topright,
        left_margin=12Plots.mm, bottom_margin=8Plots.mm)
    for (j, (lbl, d)) in enumerate(data)
        x = (1:n) .+ (j - (k + 1) / 2) * width
        bar!(p, x, [value(d, nm) for nm in names]; bar_width=width, label=lbl, alpha=0.8)
    end
    return _save(p, save_path)
end
plot_coil_sensitivities(source::SingleSource; kwargs...) = plot_coil_sensitivities(_sources(source); kwargs...)

"""
    plot_tolerance_pdf(sources; corrected=true, normalize=false, xscale=:identity, save_path=nothing)
    plot_tolerance_pdf(h5path; kwargs...)

The Monte Carlo distributions of the dominant-mode overlap `|δ|` of each run, intrinsic and
(when `corrected`) with error-field correction, with the as-designed overlap marked.
"""
function plot_tolerance_pdf(sources::Sources; corrected::Bool=true, normalize::Bool=false, xscale::Symbol=:identity, save_path=nothing)
    p = plot(; xlabel="dominant-mode overlap |δ|", ylabel=normalize ? "probability density (normalized)" : "probability density",
        title="Tolerance Monte Carlo: |δ| over sampled misalignments", legend=:topright, xscale=xscale,
        left_margin=12Plots.mm, bottom_margin=6Plots.mm)
    any_data = false
    for (j, (lbl, path)) in enumerate(sources)
        d = _load(path)
        d.pdf === nothing && continue
        any_data = true
        c = _centers(d.bin_edges)
        keep = xscale === :log10 ? c .> 0 : trues(length(c))
        scale = normalize ? maximum(d.pdf) : 1.0
        plot!(p, c[keep], d.pdf[keep] ./ scale; lw=2, c=j, label="$lbl intrinsic")
        corrected && plot!(p, c[keep], d.pdf_efc[keep] ./ (normalize ? maximum(d.pdf_efc) : 1.0); lw=2, ls=:dash, c=j, label="$lbl corrected")
        vline!(p, [d.mc_delta_nominal]; ls=:dot, c=j, label="$lbl as designed")
    end
    any_data || return _empty("No ErrorFields/MonteCarlo data — run with a tolerance_file")
    return _save(p, save_path)
end
plot_tolerance_pdf(source::SingleSource; kwargs...) = plot_tolerance_pdf(_sources(source); kwargs...)

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
    floor = 1e-4
    for (j, (lbl, path)) in enumerate(sources)
        d = _load(path)
        d.scan_scale === nothing && continue
        any_data = true
        plot!(p, d.scan_scale, max.(d.scan_plock, floor); yerror=d.scan_spread ./ 2, marker=:circle, lw=2, c=j, label="$lbl intrinsic")
        corrected && plot!(p, d.scan_scale, max.(d.scan_plock_efc, floor); yerror=d.scan_spread_efc ./ 2, marker=:square, lw=2, ls=:dash, c=j, label="$lbl corrected")
        if target_percent !== nothing
            scan = EF.ToleranceScan(d.scan_scale, d.scan_plock, d.scan_plock_efc, d.scan_spread, d.scan_spread_efc, 0.0)
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
distributions, on a logarithmic `|δ|` axis, with the nominal threshold marked.
"""
function plot_threshold_scaling(sources::Sources; save_path=nothing)
    p = plot(; xlabel="dominant-mode overlap |δ|", ylabel="normalized density  /  P(lock | δ)", xscale=:log10, legend=:topleft,
        title="Overlap distribution vs penetration threshold", left_margin=12Plots.mm, bottom_margin=6Plots.mm)
    any_data = false
    for (j, (lbl, path)) in enumerate(sources)
        d = _load(path)
        d.threshold_pdf === nothing && continue
        any_data = true
        c = _centers(d.bin_edges)
        keep = c .> 0
        plot!(p, c[keep], d.pdf[keep] ./ maximum(d.pdf); lw=2, c=j, label="$lbl intrinsic |δ|")
        plot!(p, c[keep], d.pdf_efc[keep] ./ maximum(d.pdf_efc); lw=2, ls=:dash, c=j, label="$lbl corrected |δ|")
        plot!(p, c[keep], d.threshold_pdf[keep] ./ maximum(d.threshold_pdf); lw=2, ls=:dot, c=j, label="$lbl threshold")
        plot!(p, d.bin_edges[2:end], d.p_lock_given_delta[2:end]; lw=1.5, ls=:dashdot, c=j, label="$lbl P(lock | δ)")
        vline!(p, [d.threshold_nominal]; ls=:dot, c=:black, label=j == 1 ? "nominal threshold" : "")
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
        o.spectrum_norm > 0 || continue
        c = real.(conj.(v) .* o.spectrum .* cis(-angle(o.raw))) ./ o.spectrum_norm
        ms, as = _step_series(m, c)
        plot!(p, ms, as; seriestype=:steppre, lw=2, c=j,
            label="$(o.coil_name)  (sums to $(round(o.fraction_percent; digits=1))%)")
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
resonant fraction of the applied field (`:overlap_percent`) against the relative phases of two
or three coil arrays (`EF.phasing_map`). Two arrays give a line, three a filled contour whose
axes are the phase of the middle array relative to the first and of the third relative to the
middle; the extreme is marked.
"""
function plot_phasing_map(map::EF.PhasingMap; quantity::Symbol=:delta_per_kat, save_path=nothing)
    arr =
        quantity === :delta_per_kat ? map.delta_per_kat :
        quantity === :overlap_percent ? map.overlap_percent :
        throw(ArgumentError("quantity must be :delta_per_kat or :overlap_percent"))
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
    plot_efc_ntv_limits(h5path; torque_budget, delta_threshold=nothing, safety_factor=1.0, delta_max=15, save_path=nothing)
    plot_efc_ntv_limits(couplings::Vector{EF.EFCCoupling}; delta_threshold, torque_budget, kwargs...)

Correction current against intrinsic overlap for each correction array of a run
(`ErrorFields/NTV/`): the linear single-mode current and the NTV-limited current whose residual
torque lowers the threshold, with the largest correctable overlap marked; when the run
tabulated the torques against rotation, a second panel shows those tables with the offsets
found and the reference rotation, and the curve uses the torque balance (`model`,
`rotation_exponent` as in `efc_current_curve`). `delta_threshold` defaults to the run's nominal
penetration threshold (`ErrorFields/Risk/threshold_nominal`); `torque_budget` is the torque,
N·m, that would bring the reference rotation to rest.
"""
function plot_efc_ntv_limits(couplings::Vector{EF.EFCCoupling}; delta_threshold::Real, torque_budget::Real, safety_factor::Real=1.0,
    delta_max::Real=15, model::Symbol=:auto, rotation_exponent::Real=1.0, save_path=nothing)
    p = plot(; xlabel="intrinsic overlap δ_EF / δ_thresh", ylabel="correction current [kAt]", legend=:topleft,
        title="Error-field correction against its own NTV torque (budget $(torque_budget) N·m)", left_margin=12Plots.mm, bottom_margin=6Plots.mm)
    for (j, c) in enumerate(couplings)
        curve = EF.efc_current_curve(c; delta_threshold, torque_budget, safety_factor, delta_max, model, rotation_exponent)
        x = curve.delta_ef ./ delta_threshold
        tag = curve.model === :torque_balance ? "torque balance" : "linear budget"
        plot!(p, x, curve.current_linear; lw=2, c=j, label="$(c.coil_name) single-mode")
        plot!(p, x, curve.current_ntv; lw=2, ls=:dash, c=j, label="$(c.coil_name) with residual NTV ($tag)")
        isfinite(curve.with_ntv) && vline!(p, [curve.with_ntv / delta_threshold]; ls=:dot, c=j, label="$(c.coil_name) NTV limit")
        isfinite(curve.torque_only) && scatter!(p, [curve.torque_only / delta_threshold], [0.0]; marker=:star5, ms=9, c=j, label="$(c.coil_name) whole-field limit")
    end
    scanned = filter(EF.has_rotation_scan, couplings)
    isempty(scanned) && return _save(p, save_path)
    # The tabulated torques against the rotation shift, with the balance's reference rotation and the found offsets.
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
    return _save(plot(p, q; layout=(1, 2), size=(1500, 550)), save_path)
end
function plot_efc_ntv_limits(h5path::AbstractString; torque_budget::Real, delta_threshold=nothing, kwargs...)
    couplings = EF.read_efc_couplings(h5path)
    δt = delta_threshold === nothing ? h5open(f -> read(f["ErrorFields/Risk/threshold_nominal"]), h5path, "r") : delta_threshold
    return plot_efc_ntv_limits(couplings; delta_threshold=δt, torque_budget, kwargs...)
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
