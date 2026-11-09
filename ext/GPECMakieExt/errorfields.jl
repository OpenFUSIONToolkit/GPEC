# ErrorFields figures. Several runs can be drawn together, told apart by colour and named in the legend; a single run
# carries no label.

const EFA = A.ErrorFields
const _EF = EFA.EF

_with_label(lbl, s) = isempty(lbl) ? s : "$lbl: $s"
_pct(x) = x == 0 ? "0 %" : x < 0.01 ? "<0.01 %" : "$(_num(x)) %"
_steps(m, a) = (vcat(m[1] - 1, m, m[end] + 1), vcat(0.0, a, 0.0))

# Log ticks at whole decades labelled 10ⁿ, with 2× and 5× added when the range spans fewer than three decades.
function _decade_ticks(lo, hi)
    k0, k1 = floor(Int, log10(lo)), ceil(Int, log10(hi))
    vals, labels = Float64[], Any[]
    for k in k0:k1, f in (k1 - k0 <= 3 ? (1, 2, 5) : (1,))
        v = f * 10.0^k
        lo <= v <= hi || continue
        push!(vals, v)
        push!(labels, f == 1 ? rich("10", superscript(string(k))) : rich("$(f)×10", superscript(string(k))))
    end
    return (vals, labels)
end

# The largest and smallest bin centres where any of the densities is above 10⁻³ of its peak, so empty tails are not drawn.
_support_end(c, pdfs...) = maximum(p -> c[findlast(>(1e-3 * maximum(p)), p)], pdfs)
_support_start(c, pdfs...) = minimum(p -> c[findfirst(>(1e-3 * maximum(p)), p)], pdfs)

# Rows of stems and dots: one row per name, one dot per source, on a linear or log value axis. Values that are roundoff
# against the largest are left out; returns how many rows that removed.
function _stem_rows!(ax, names, values, labels; log=false)
    big = maximum(v -> maximum(x -> isfinite(x) ? x : 0.0, v; init=0.0), values; init=0.0)
    keep = [any(v -> isfinite(v[i]) && v[i] > EFA._ROUNDOFF * big, values) for i in eachindex(names)]
    rows = findall(keep)
    isempty(rows) && throw(ArgumentError("no coil set has a value above roundoff"))
    order = rows[sortperm([maximum(v -> isfinite(v[i]) ? v[i] : 0.0, values) for i in rows])]
    k = length(values)
    shown = [v[i] for v in values for i in order if isfinite(v[i]) && v[i] > 0]
    base = log ? minimum(shown) / 3 : 0.0
    ax.yticks = (1:length(order), names[order])
    ax.ygridvisible = false
    for (j, v) in enumerate(values)
        y = (1:length(order)) .+ (k == 1 ? 0.0 : (j - (k + 1) / 2) * 0.22)
        x = [v[i] for i in order]
        ok = [isfinite(xi) && xi > (log ? 0 : -Inf) for xi in x]
        rangebars!(ax, y[ok], fill(base, count(ok)), x[ok]; direction=:x, color=(WONG[j], 0.35), whiskerwidth=0, linewidth=2)
        scatter!(ax, x[ok], y[ok]; color=WONG[j], markersize=10, label=isempty(labels[j]) ? nothing : labels[j])
    end
    log && (ax.xticks = _decade_ticks(base, 1.5 * maximum(shown)))
    xlims!(ax, log ? base : 0.0, (log ? 1.6 : 1.1) * maximum(shown))
    k > 1 && axislegend(ax; position=:rb)
    return length(names) - length(order)
end

const _SENSITIVITY_TEXT = Dict(
    :shift => ("|δ| per mm of shift", "Dominant-mode overlap per mm of shift"),
    :tilt => ("|δ| per degree of tilt", "Dominant-mode overlap per degree of tilt"),
    :rim => ("|δ| per mm of rim displacement", "Dominant-mode overlap per mm of rim displacement"),
    :as_designed => ("as-designed |δ|", "As-designed dominant-mode overlap"),
    :fraction => ("resonant fraction of |b̃| (%)", "Resonant fraction of each coil set's applied field")
)

function EFA._plot_coil_sensitivities(labels, names, values, quantity, yscale)
    return themed() do
        xlabel, title = _SENSITIVITY_TEXT[quantity]
        fig = Figure()
        ax = Axis(fig[1, 1]; xlabel, xscale=yscale === :log10 ? log10 : identity, xminorticksvisible=false, width=640)
        omitted = _stem_rows!(ax, names, values, labels; log=yscale === :log10)
        ax.height = 30 * (length(names) - omitted) + 30
        ax.title = omitted == 0 ? title : "$title ($omitted at roundoff not shown)"
        resize_to_layout!(fig)
        fig
    end
end

function EFA._plot_tolerance_pdf(data; corrected=true, normalize=false, xscale=:identity, show_batches=false)
    return themed() do
        fig = Figure(; size=(820, 460))
        # An as-designed overlap that is roundoff against the sampled mean (axisymmetric coils only) is not marked.
        designed(d) = d.abs_delta_total_as_designed > EFA._ROUNDOFF * d.abs_delta_sampled_mean
        d1 = data[1][2]
        title =
            length(data) == 1 ? "Sampled |δ|: mean $(_num(d1.abs_delta_sampled_mean))" * (designed(d1) ? ", as designed $(_num(d1.abs_delta_total_as_designed))" : "") :
            "Tolerance Monte Carlo"
        xmax = maximum(d -> 1.05 * _support_end(EFA._centers(d.abs_delta_bin_edges), d.abs_delta_pdf, d.abs_delta_efc_pdf), last.(data))
        xmin = xscale === :log10 ? minimum(d -> _support_start(EFA._centers(d.abs_delta_bin_edges), d.abs_delta_pdf, d.abs_delta_efc_pdf), last.(data)) / 1.5 : 0
        ax = Axis(fig[1, 1]; xlabel="dominant-mode overlap |δ|", ylabel=normalize ? "probability density / its peak" : "probability density", title,
            xscale=xscale === :log10 ? log10 : identity, xminorticksvisible=false, limits=((xmin, xmax), (0, nothing)),
            xticks=xscale === :log10 ? _decade_ticks(xmin, xmax) : Makie.automatic, xtickformat=xscale === :log10 ? Makie.automatic : (vs -> _num.(vs)))
        for (j, (lbl, d)) in enumerate(data)
            c = EFA._centers(d.abs_delta_bin_edges)
            keep = xscale === :log10 ? c .> 0 : trues(length(c))
            scale = normalize ? maximum(d.abs_delta_pdf) : 1.0
            if show_batches && d.abs_delta_pdf_batches !== nothing
                nb = size(d.abs_delta_pdf_batches, 2)
                for b in 1:nb
                    lines!(ax, c[keep], d.abs_delta_pdf_batches[keep, b] ./ scale; color=(WONG[j], 0.3), linewidth=1,
                        label=b == 1 ? _with_label(lbl, "$nb batches") : nothing)
                end
            end
            beyond = show_batches && d.clamped_fraction !== nothing && d.clamped_fraction > 0 ? " ($(_pct(100d.clamped_fraction)) beyond the last bin)" : ""
            lines!(ax, c[keep], d.abs_delta_pdf[keep] ./ scale; color=WONG[j], linewidth=2, label=_with_label(lbl, "intrinsic" * beyond))
            corrected && lines!(ax, c[keep], d.abs_delta_efc_pdf[keep] ./ (normalize ? maximum(d.abs_delta_efc_pdf) : 1.0); color=WONG[j], linewidth=2,
                linestyle=:dash, label=_with_label(lbl, "corrected"))
            designed(d) && vlines!(ax, d.abs_delta_total_as_designed; color=WONG[j], linestyle=:dot, linewidth=1.5, label=_with_label(lbl, "as designed"))
        end
        axislegend(ax; position=:rt)
        fig
    end
end

function _phasor_panel!(gl, p)
    total = sum(p.delta)
    w = p.delta .* cis(-angle(total))
    tip = cumsum(w)
    tail = vcat(zero(ComplexF64), tip[1:(end-1)])
    big = sortperm(abs.(w); rev=true)[1:min(6, length(w))]
    grey = RGBAf(0.5, 0.5, 0.5, 0.6)
    colour(k) = k in big ? WONG[findfirst(==(k), big)] : grey
    span = maximum(abs, vcat(0.0, real.(tip), imag.(tip)))
    ax = Axis(gl[1, 1]; aspect=DataAspect(), xlabel="Re δ  (rotated so the total is real)", ylabel="Im δ",
        limits=((-0.1span, 1.1span), (-0.45span, 0.45span)), title=_with_label(p.label, "As-designed overlaps head to tail: |Σδ| = $(_num(abs(total)))"))
    arrows2d!(ax, [Point2f(0, 0)], [Vec2f(abs(total), 0)]; color=(:black, 0.25), shaftwidth=8, tipwidth=18, tiplength=16)
    arrows2d!(ax, Point2f.(real.(tail), imag.(tail)), Vec2f.(real.(w), imag.(w)); color=[colour(k) for k in eachindex(w)], shaftwidth=2.5, tipwidth=10,
        tiplength=10)
    elements = Any[LineElement(; color=colour(k), linewidth=3) for k in big]
    labels = ["$(p.names[k])   |δ| = $(_num(abs(w[k])))" for k in big]
    if length(w) > length(big)
        push!(elements, LineElement(; color=grey, linewidth=3))
        push!(labels, "the other $(length(w) - length(big))")
    end
    push!(elements, LineElement(; color=(:black, 0.25), linewidth=8))
    push!(labels, "total")
    Legend(gl[1, 2], elements, labels; framevisible=false)
    return ax
end

function EFA._plot_overlap_phasors(panels)
    return themed() do
        fig = Figure()
        for (j, p) in enumerate(panels)
            gl = fig[1, j] = GridLayout()
            _phasor_panel!(gl, p)
        end
        for c in contents(fig.layout)
            c isa GridLayout && for a in contents(c)
                a isa Axis && (a.width = 520; a.height = 400)
            end
        end
        resize_to_layout!(fig)
        fig
    end
end

const _BUDGET_COLOURS = (RGBf(0.4, 0.4, 0.4), WONG[1], WONG[2], WONG[3])

function EFA._plot_tolerance_budget(labels, parts, worst, mc)
    return themed() do
        total = sum(p[2] for p in parts)
        factor, scale = _power(total)
        m = length(labels)
        fig = Figure(; size=(900, max(380, 120 + 26 * m)))
        ax = Axis(fig[1, 1]; yticks=(1:m, labels), xlabel=rich("worst-case contribution to |δ|", _unit(scale, "")), ygridvisible=false,
            title="Tolerance budget: Σ = $(_num(worst)) worst case", limits=((0, nothing), (0.4, m + 0.6)))
        x = repeat(1:m; outer=length(parts))
        h = reduce(vcat, (p[2] ./ factor for p in parts))
        g = repeat(1:length(parts); inner=m)
        barplot!(ax, x, h; stack=g, color=[_BUDGET_COLOURS[i] for i in g], direction=:x)
        used = [any(>(EFA._ROUNDOFF * maximum(total)), p[2]) for p in parts]
        elements = Any[PolyElement(; color=c) for c in _BUDGET_COLOURS][used]
        names = [p[1] for p in parts][used]
        if mc !== nothing
            if mc.abs_delta_sampled_mean !== nothing
                vlines!(ax, mc.abs_delta_sampled_mean / factor; color=:black, linestyle=:dash, linewidth=1.5)
                push!(elements, LineElement(; color=:black, linestyle=:dash))
                push!(names, "Monte Carlo mean |δ|: $(_num(mc.abs_delta_sampled_mean))")
            end
            if mc.abs_delta_total_as_designed !== nothing && mc.abs_delta_total_as_designed > EFA._ROUNDOFF * worst
                vlines!(ax, mc.abs_delta_total_as_designed / factor; color=:black, linestyle=:dot, linewidth=1.5)
                push!(elements, LineElement(; color=:black, linestyle=:dot))
                push!(names, "as designed |Σδ|: $(_num(mc.abs_delta_total_as_designed))")
            end
        end
        Legend(fig[1, 2], elements, names; framevisible=false)
        fig
    end
end

const _TAPS = ("shift x", "shift y", "shift z", "tilt x", "tilt y", "tilt z")

function EFA._plot_linearity_residuals(panels, at, yscale)
    return themed() do
        fig = Figure(; size=(820 * length(panels), 500))
        for (j, p) in enumerate(panels)
            # Coil sets whose residuals are all zero have nothing to draw.
            shown = [any(v -> isfinite(v) && v > 0, p.residual[:, i]) for i in eachindex(p.names)]
            names, r = p.names[shown], p.residual[:, shown]
            n = length(names)
            title = _with_label(p.label, "Linearity of the rigid-motion response")
            all(shown) || (title *= " ($(count(!, shown)) with zero residuals not shown)")
            ax = Axis(fig[1, j]; xticks=(1:n, names), xticklabelrotation=π / 6, yscale=yscale === :log10 ? log10 : identity, yminorticksvisible=false,
                ylabel=at === :step ? "2nd difference / largest 1st, at the FD step" : "2nd difference / largest 1st, at the tolerance", title)
            positive = filter(v -> isfinite(v) && v > 0, vcat(vec(r), 1e-2))
            base = yscale === :log10 ? minimum(positive) / 3 : 0.0
            yscale === :log10 && (ax.yticks = _decade_ticks(base, 3 * maximum(positive)))
            for t in 1:6
                x = (1:n) .+ (t - 3.5) * (0.8 / 6)
                ok = [isfinite(v) && v > (yscale === :log10 ? 0 : -Inf) for v in r[t, :]]
                rangebars!(ax, x[ok], fill(base, count(ok)), r[t, ok]; color=(WONG[t], 0.4), whiskerwidth=0, linewidth=2)
                scatter!(ax, x[ok], r[t, ok]; color=WONG[t], markersize=7, label=_TAPS[t])
            end
            hlines!(ax, 1e-2; color=:black, linestyle=:dash, linewidth=1, label="1 %")
            ylims!(ax, base, (yscale === :log10 ? 3 : 1.15) * maximum(positive))
            axislegend(ax; position=:rt, nbanks=2)
        end
        fig
    end
end

const _CHECK_MARKERS = (:circle, :diamond, :utriangle, :rect, :star5, :hexagon)

function EFA._plot_linearity_check(check)
    return themed() do
        names = unique(check.name)
        scales = sort(unique(check.scale))
        positive = filter(v -> isfinite(v) && v > 0, vcat(check.relative_error, 1e-2))
        base = minimum(positive) / 3
        fig = Figure(; size=(1000, 520))
        ax = Axis(fig[1, 1]; xticks=(1:length(names), names), xticklabelrotation=π / 6, yscale=log10, yminorticksvisible=false,
            ylabel=rich("|δ", subscript("actual"), " − δ", subscript("linear"), "| / |δ", subscript("linear"), " − δ", subscript("as designed"), "|"),
            title="Linear model error at finite displacements", limits=(nothing, (base, 3 * maximum(positive))))
        nslot = 4 * length(scales)
        slot = 0
        for (k, kind) in enumerate((:shift, :tilt)), a in 1:2, (si, s) in enumerate(scales)
            slot += 1
            rows = findall(i -> check.kind[i] === kind && check.axis[i] == a && check.scale[i] == s && isfinite(check.relative_error[i]) &&
                             check.relative_error[i] > 0, eachindex(check.name))
            isempty(rows) && continue
            x = [findfirst(==(check.name[i]), names) + (slot - (nslot + 1) / 2) * (0.8 / nslot) for i in rows]
            y = check.relative_error[rows]
            rangebars!(ax, x, fill(base, length(x)), y; color=(WONG[k], 0.35), whiskerwidth=0, linewidth=1.5)
            scatter!(ax, x, y; color=WONG[k], marker=_CHECK_MARKERS[mod1(si, length(_CHECK_MARKERS))], markersize=8,
                label="$kind $(a == 1 ? "x" : "y") × $(_num(s))")
        end
        hlines!(ax, 1e-2; color=:black, linestyle=:dash, linewidth=1, label="1 %")
        Legend(fig[1, 2], ax; framevisible=false, merge=true)
        fig
    end
end

function _risk_panel!(pos, data; corrected=true, target_percent=nothing)
    s = data[1][2].scan_scale
    title = "Locking risk against tolerance scale"
    if target_percent !== nothing && length(data) == 1
        scan = _risk_scan(data[1][2])
        a, ac = _EF.allowable_tolerance(scan, target_percent), _EF.allowable_tolerance(scan, target_percent; corrected=true)
        title =
            "Allowable scale for $(_num(target_percent)) % risk: $(isnan(a) ? "beyond the scan" : _num(a))" *
            (corrected ? " ($(isnan(ac) ? "beyond the scan" : _num(ac)) corrected)" : "")
    end
    # Log-log; a scale where no sample locked has P = 0 and is left off the log axis.
    curves(d) = ((d.scan_locking_probability_percent, d.scan_locking_probability_spread_percent, "intrinsic", :solid, :circle),
        (d.scan_locking_probability_efc_percent, d.scan_locking_probability_efc_spread_percent, "corrected", :dash, :rect))
    shown = [v for (_, d) in data for (p, _, name) in curves(d) if corrected || name == "intrinsic" for v in p if v > 0]
    target_percent === nothing || push!(shown, target_percent)
    logy = !isempty(shown)
    lo, hi = logy ? (minimum(shown) / 3, 3 * maximum(shown)) : (0.0, 1.0)
    ax = Axis(pos; xscale=log10, xminorticksvisible=false, xticks=(s, _num.(s)), xlabel="tolerance scale (1 = design)", ylabel="locking probability (%)",
        title, yscale=logy ? log10 : identity, yminorticksvisible=false, yticks=logy ? _decade_ticks(lo, hi) : Makie.automatic,
        limits=((minimum(s) / 1.3, 1.3 * maximum(s)), (lo, hi)))
    target_percent === nothing || hlines!(ax, target_percent; color=:black, linestyle=:dash, linewidth=1)
    vlines!(ax, 1.0; color=(:black, 0.3), linewidth=1)
    for (j, (lbl, d)) in enumerate(data)
        for (p, spread, name, style, marker) in curves(d)
            (name == "corrected" && !corrected) && continue
            k = logy ? p .> 0 : trues(length(p))
            any(k) || continue
            # The lower bar stops short of zero so it stays on a log axis.
            errorbars!(ax, d.scan_scale[k], p[k], min.(spread[k] ./ 2, 0.9 .* p[k]), spread[k] ./ 2; color=(WONG[j], 0.5), whiskerwidth=6)
            scatterlines!(ax, d.scan_scale[k], p[k]; color=WONG[j], linestyle=style, marker, markersize=8, label=_with_label(lbl, name))
        end
    end
    axislegend(ax; position=:lt)
    return ax
end

_risk_scan(d) = _EF.ToleranceScan(d.scan_scale, d.scan_locking_probability_percent, d.scan_locking_probability_efc_percent,
    d.scan_locking_probability_spread_percent, d.scan_locking_probability_efc_spread_percent, 0.0)

function EFA._plot_locking_risk(data; corrected=true, target_percent=nothing)
    return themed() do
        fig = Figure(; size=(820, 460))
        _risk_panel!(fig[1, 1], data; corrected, target_percent)
        fig
    end
end

# The overlap distributions on a log |δ| axis against the threshold's CDF on a 0–1 right axis, with the nominal
# threshold at the median.
function _threshold_panel!(pos, data)
    lo, hi = Inf, 0.0
    for (_, d, T) in data
        c = EFA._centers(d.abs_delta_bin_edges)
        keep = c .> 0
        support = c[keep][d.abs_delta_pdf[keep].>1e-3*maximum(d.abs_delta_pdf)]
        lo = min(lo, minimum(support) / 3)
        hi = max(hi, T === nothing ? d.abs_delta_bin_edges[end] : 3 * T[ceil(Int, 0.99 * length(T))], 3 * d.threshold_fit)
    end
    d1 = data[1][2]
    title =
        length(data) == 1 && d1.locking_probability_percent !== nothing ?
        rich("P", subscript("lock"), " at design tolerances: $(_pct(d1.locking_probability_percent)) ($(_pct(d1.locking_probability_efc_percent)) corrected)") :
        "Overlap against the penetration threshold"
    ticks = _decade_ticks(lo, hi)
    a = Axis(pos; xscale=log10, xminorticksvisible=false, xticks=ticks, limits=((lo, hi), (0, nothing)), xlabel="dominant-mode overlap |δ|",
        ylabel="probability density of |δ|", title)
    b = Axis(pos; xscale=log10, xminorticksvisible=false, xticks=ticks, limits=((lo, hi), (0, 1)), yaxisposition=:right, ylabel="P(lock | δ)",
        xgridvisible=false, ygridvisible=false)
    hidexdecorations!(b)
    hidespines!(b)
    linkxaxes!(a, b)
    elements, names = Any[], String[]
    threshold_colour = WONG[6]
    for (j, (lbl, d, T)) in enumerate(data)
        c = EFA._centers(d.abs_delta_bin_edges)
        keep = c .> 0
        lines!(a, c[keep], d.abs_delta_pdf[keep]; color=WONG[j], linewidth=2)
        lines!(a, c[keep], d.abs_delta_efc_pdf[keep]; color=WONG[j], linewidth=2, linestyle=:dash)
        append!(elements, [LineElement(; color=WONG[j], linewidth=2), LineElement(; color=WONG[j], linewidth=2, linestyle=:dash)])
        append!(names, [_with_label(lbl, "intrinsic"), _with_label(lbl, "corrected")])
        if T === nothing
            lines!(b, d.abs_delta_bin_edges[2:end], d.locking_probability_given_delta[2:end]; color=threshold_colour, linewidth=2.5)
        else
            xg = exp10.(range(log10(lo), log10(hi); length=400))
            lines!(b, xg, [searchsortedlast(T, x) / length(T) for x in xg]; color=threshold_colour, linewidth=2.5)
        end
        scatter!(b, [d.threshold_fit], [0.5]; color=threshold_colour, marker=:diamond, markersize=12)
        vlines!(b, d.threshold_fit; color=(threshold_colour, 0.5), linestyle=:dot, linewidth=1.5)
    end
    append!(elements, [LineElement(; color=threshold_colour, linewidth=2.5), MarkerElement(; color=threshold_colour, marker=:diamond, markersize=12)])
    append!(names, ["threshold CDF", length(data) == 1 ? "nominal threshold $(_num(d1.threshold_fit))" : "nominal threshold"])
    axislegend(a, elements, names; position=:lt)
    return a
end

function EFA._plot_threshold_scaling(data)
    return themed() do
        fig = Figure(; size=(860, 480))
        _threshold_panel!(fig[1, 1], data)
        fig
    end
end

function EFA._plot_dominant_mode_spectrum(data, modes)
    return themed() do
        fig = Figure(; size=(820, 440))
        ax = Axis(fig[1, 1]; xlabel="poloidal harmonic m", ylabel=rich("|V", subscript("m"), "|  (unit norm)"), title="Dominant resonant-coupling modes",
            limits=(nothing, (0, nothing)))
        styles = (:solid, :dash, :dot, :dashdot)
        for (j, (lbl, d)) in enumerate(data), k in modes
            k <= size(d.dominant_v, 2) || continue
            for n in unique(d.mn_index[:, 2])
                rows = findall(==(n), d.mn_index[:, 2])
                label = _with_label(lbl, "mode $k, σ = $(_num(d.singular_values[k]))")
                stairs!(ax, _steps(d.mn_index[rows, 1], abs.(d.dominant_v[rows, k]))...; step=:center, color=WONG[k], linestyle=styles[mod1(j, 4)],
                    linewidth=k == 1 ? 2.5 : 1.5, label)
            end
        end
        axislegend(ax; position=:rt, merge=true)
        fig
    end
end

function EFA._plot_applied_spectra(m, v, spectra; normalize=true)
    return themed() do
        fig = Figure(; size=(820, 440))
        factor, scale = normalize ? (1.0, "") : _power(reduce(vcat, (s for (_, s) in spectra)))
        ax = Axis(fig[1, 1]; xlabel="poloidal harmonic m", ylabel=normalize ? "amplitude / its own peak" : rich("|b̃|", _unit(scale, "T")),
            title="Applied spectra against the dominant mode", limits=(nothing, (0, normalize ? 1.08 : nothing)))
        if normalize
            sx, sy = _steps(m, v ./ maximum(v))
            px = reduce(vcat, ([x - 0.5, x + 0.5] for x in sx))
            py = reduce(vcat, ([y, y] for y in sy))
            band!(ax, px, zeros(length(py)), py; color=(:black, 0.08))
            stairs!(ax, sx, sy; step=:center, color=:black, linewidth=2, label="dominant mode |V|")
        end
        for (j, (name, a)) in enumerate(spectra)
            stairs!(ax, _steps(m, normalize ? a ./ maximum(a) : a ./ factor)...; step=:center, color=WONG[mod1(j, 7)], linewidth=2, label=name)
        end
        axislegend(ax; position=:rt)
        fig
    end
end

function EFA._plot_overlap_contributions(m, curves)
    return themed() do
        fig = Figure(; size=(820, 440))
        ax = Axis(fig[1, 1]; xlabel="poloidal harmonic m", ylabel="contribution to the resonant fraction (%)", title="Which harmonics make each coil's overlap")
        hlines!(ax, 0; color=:black, linewidth=1)
        for (j, (name, fraction, c)) in enumerate(curves)
            stairs!(ax, _steps(m, 100 .* c)...; step=:center, color=WONG[mod1(j, 7)], linewidth=2, label="$name: sums to $(_num(fraction)) %")
        end
        axislegend(ax; position=:rt)
        fig
    end
end

function EFA._plot_surface_overlay(maps, mode)
    return themed() do
        k = length(maps)
        ncol = k <= 3 ? k : 2
        nrow = cld(k, ncol)
        fig = Figure(; size=(k == 1 ? 720 : 470 * ncol + 100, 340 * nrow + 60))
        mpeak = maximum(abs, mode.bn)
        local hm
        for (j, (name, bm)) in enumerate(maps)
            r, c = fldmod1(j, ncol)
            peak = maximum(abs, bm.bn)
            bottom = r == nrow
            ax = Axis(fig[r, c]; title=rich("$name: peak |b", subscript("n"), "| = $(_num(peak)) T"), yticks=-180:90:180, xticks=0:90:360,
                xlabel=bottom ? "toroidal angle φ (deg)" : "", ylabel=c == 1 ? "poloidal angle θ (deg)" : "", xticklabelsvisible=bottom,
                yticklabelsvisible=c == 1)
            hm = heatmap!(ax, rad2deg.(bm.zeta), rad2deg.(bm.theta), permutedims(bm.bn ./ peak); colormap=DIVERGING, colorrange=(-1, 1))
            mpeak > 0 && contour!(ax, rad2deg.(mode.zeta), rad2deg.(mode.theta), permutedims(mode.bn ./ mpeak); levels=[-0.5, -0.2, 0.2, 0.5], color=:black,
                linewidth=1)
        end
        Colorbar(fig[1:nrow, ncol+1], hm; label=rich("b", subscript("n"), " / its peak   (contours: dominant mode)"))
        fig
    end
end

function EFA._plot_phasing_map(map, quantity, extreme)
    return themed() do
        arr = quantity === :delta_per_kat ? map.delta_per_kat : map.resonant_fraction_percent
        label = quantity === :delta_per_kat ? "|δ| per kAt of the arrays" : "resonant fraction (%)"
        names = map.coil_names
        relative(a, b) = "toroidal phase of $b relative to $a (deg)"
        if length(map.phase_deg) == 1
            flat = maximum(arr) - minimum(arr) <= 1e-6 * maximum(abs, arr)
            fig = Figure(; size=(820, 440))
            ax = Axis(fig[1, 1]; xlabel=relative(names[1], names[2]), ylabel=label, xticks=0:90:360,
                title=flat ? "$label: independent of phase" : "$label: $(_num(minimum(arr)))–$(_num(maximum(arr))) over the phase")
            lines!(ax, map.phase_deg[1], vec(arr); color=WONG[1], linewidth=2)
            flat && ylims!(ax, 0, 1.2 * maximum(arr))
        else
            fig = Figure(; size=(720, 620))
            v, ph = extreme
            ax = Axis(fig[1, 1]; aspect=DataAspect(), xlabel=relative(names[1], names[2]), ylabel=relative(names[2], names[3]),
                title="$label: largest $(_num(v)) at ($(_num(ph[1]))°, $(_num(ph[2]))°)")
            cf = contourf!(ax, map.phase_deg[1], map.phase_deg[2], arr; levels=12, colormap=:viridis)
            scatter!(ax, [ph[1]], [ph[2]]; marker=:star5, markersize=16, color=:white, strokecolor=:black, strokewidth=1)
            Colorbar(fig[1, 2], cf; label)
        end
        fig
    end
end

function EFA._plot_efc_ntv_limits(curves, threshold, budget; band=false, profiles=false)
    return themed() do
        couplings = first.(curves)
        scanned = filter(_EF.has_rotation_scan, couplings)
        npanel = 1 + !isempty(scanned) + profiles
        fig = Figure(; size=(700 + 520 * (npanel - 1), 480))
        ax = Axis(fig[1, 1]; xlabel=rich("intrinsic overlap δ", subscript("EF"), " / nominal threshold"), ylabel="correction current (kAt)",
            title="Correction current with its own NTV torque ($(_num(budget)) N·m budget)", limits=((0, nothing), (0, nothing)))
        xmax, ymax = 0.0, 0.0
        for (j, (c, cv)) in enumerate(curves)
            x = cv.delta_ef ./ threshold
            xmax = max(xmax, isfinite(cv.with_ntv) ? 1.6 * cv.with_ntv / threshold : x[end])
            inview = x .<= xmax
            ymax = max(ymax, maximum(filter(isfinite, vcat(cv.current_linear[inview], cv.current_ntv[inview])); init=0.0))
            if band
                lo, hi = cv.limits_pessimistic.with_ntv, cv.limits_optimistic.with_ntv
                isfinite(lo) && vspan!(ax, lo / threshold, min(hi, cv.delta_ef[end]) / threshold; color=(WONG[j], 0.12))
                band!(ax, x, cv.current_ntv_optimistic, cv.current_ntv_pessimistic; color=(WONG[j], 0.15))
            end
            lines!(ax, x, cv.current_linear; color=WONG[j], linewidth=2, label="$(c.coil_name): single-mode")
            lines!(ax, x, cv.current_ntv; color=WONG[j], linewidth=2, linestyle=:dash, label="$(c.coil_name): with residual NTV")
            lines!(ax, x, cv.current_ntv_upper; color=(WONG[j], 0.7), linewidth=1, linestyle=:dashdot, label="$(c.coil_name): over-correction edge")
            isfinite(cv.with_ntv) && vlines!(ax, cv.with_ntv / threshold; color=WONG[j], linestyle=:dot, linewidth=1.5,
                label="correctable up to $(_num(cv.with_ntv / threshold)) × threshold")
            isfinite(cv.residual_only) && scatter!(ax, [cv.residual_only / threshold], [0.0]; color=WONG[j], marker=:circle, markersize=10,
                label="zero rotation, residual torque")
            isfinite(cv.torque_only) && scatter!(ax, [cv.torque_only / threshold], [0.0]; color=WONG[j], marker=:star5, markersize=13,
                label="zero rotation, whole-field torque")
        end
        xmax > 0 && xlims!(ax, 0, xmax)
        ymax > 0 && ylims!(ax, 0, 1.15ymax)
        axislegend(ax; position=:lt)
        col = 2
        if !isempty(scanned)
            b = Axis(fig[1, col]; xlabel="E×B rotation shift Δω (krad/s)", ylabel="NTV torque per kAt² (N·m)", title="Torque against rotation")
            hlines!(b, 0; color=:black, linewidth=1)
            for (j, c) in enumerate(scanned)
                x = c.rotation_shift ./ 1e3
                scatterlines!(b, x, c.torque_full_scan; color=WONG[j], markersize=7, label="$(c.coil_name): whole field")
                scatterlines!(b, x, c.torque_residual_scan; color=WONG[j], linestyle=:dash, marker=:diamond, markersize=7, label="$(c.coil_name): residual")
                isfinite(c.omega_reference) && vlines!(b, -c.omega_reference / 1e3; color=(:black, 0.4), linewidth=1, label="rotation brought to rest")
            end
            axislegend(b; position=:lt, merge=true)
            col += 1
        end
        if profiles
            p = Axis(fig[1, col]; xlabel=PSIN, ylabel="NTV torque per kAt² from the axis (N·m)", title="Where the torque is deposited (zero rotation shift)")
            for (j, c) in enumerate(couplings)
                isempty(c.psi) && continue
                i0 = something(findfirst(==(0.0), c.rotation_shift), argmin(abs.(c.rotation_shift)))
                lines!(p, c.psi, c.torque_full_profile[:, i0]; color=WONG[j], linewidth=2, label="$(c.coil_name): whole field")
                lines!(p, c.psi, c.torque_residual_profile[:, i0]; color=WONG[j], linewidth=2, linestyle=:dash, label="$(c.coil_name): residual")
            end
            axislegend(p; position=:lt)
        end
        fig
    end
end

function EFA._plot_correction_requirement(req)
    return themed() do
        n = length(req.rational_psi)
        k = length(req.array_names)
        labels = ["$m/$nn" for (m, nn) in zip(req.rational_m, req.rational_n)]
        source = abs.(req.resonant_field_source_t)
        vals = filter(v -> isfinite(v) && v > 0, vcat(source, vec(abs.(req.resonant_field_dominant_t)), abs.(req.resonant_field_least_squares_t)))
        base = minimum(vals) / 3
        fig = Figure(; size=(700 + 360 * k, 560))
        ax = Axis(fig[1, 1]; xticks=(1:n, labels), xlabel="rational surface q = m/n", ylabel="|resonant field| (T)", yscale=log10, yminorticksvisible=false,
            yticks=_decade_ticks(base, 3 * maximum(vals)),
            title="Resonant field of $(req.source_name) and after correction", limits=((0.4, n + 0.6), (base, 3 * maximum(vals))))
        series = vcat([("as built", source, :black, :circle)],
            [("after $nm alone", abs.(req.resonant_field_dominant_t[:, j]), WONG[j], :diamond) for (j, nm) in enumerate(req.array_names)],
            [("after joint least squares", abs.(req.resonant_field_least_squares_t), :firebrick, :star5)])
        w = 0.8 / length(series)
        for (s, (name, v, color, marker)) in enumerate(series)
            x = (1:n) .+ (s - (length(series) + 1) / 2) * w
            ok = [isfinite(y) && y > 0 for y in v]
            rangebars!(ax, x[ok], fill(base, count(ok)), v[ok]; color=(color, 0.35), whiskerwidth=0, linewidth=2)
            scatter!(ax, x[ok], v[ok]; color, marker, markersize=10, label=name)
        end
        Legend(fig[2, 1], ax; orientation=:horizontal, nbanks=2, framevisible=false)
        in_kat = all(isfinite, req.current_least_squares_kat)
        unit = in_kat ? "kAt" : "× its current as given"
        for (j, nm) in enumerate(req.array_names)
            fd, fm, fl =
                in_kat ? (req.current_dominant_kat[j], req.current_dominant_minimum_norm_kat[j], req.current_least_squares_kat[j]) :
                (req.current_factor_dominant[j], req.current_factor_dominant_minimum_norm[j], req.current_factor_least_squares[j])
            pa = PolarAxis(fig[1, j+1]; title="$nm: needed current ($unit)", rticklabelsvisible=false)
            elements, names = Any[], String[]
            for (f, name, color, width) in ((fd, "this array alone", RGBf(0.4, 0.4, 0.4), 4), (fm, "least current over the set", WONG[1], 2.5),
                (fl, "joint least squares", :firebrick, 2))
                isfinite(f) || continue
                lines!(pa, [angle(f), angle(f)], [0.0, abs(f)]; color, linewidth=width)
                scatter!(pa, [angle(f)], [abs(f)]; color, markersize=9)
                push!(elements, LineElement(; color, linewidth=width))
                push!(names, "$name: $(_num(abs(f))) ∠ $(round(Int, rad2deg(angle(f))))°")
            end
            Legend(fig[2, j+1], elements, names; framevisible=false, labelsize=11)
        end
        colsize!(fig.layout, 1, Relative(0.5))
        fig
    end
end

function EFA._plot_needed_current(mc, array, dist, limit)
    return themed() do
        in_kat = isfinite(array.ampere_turns_kat)
        per = (in_kat ? array.ampere_turns_kat : 1.0) / abs(array.delta)
        c = EFA._centers(in_kat ? dist.current_bin_edges_kat : dist.current_factor_bin_edges)
        pdf = in_kat ? dist.current_pdf_per_kat : dist.current_factor_pdf
        fig = Figure(; size=(900, 500))
        designed = mc.abs_delta_total_as_designed * per
        ntv = limit !== nothing && isfinite(limit.abs_delta_max_correctable) ? limit.abs_delta_max_correctable * per : 0.0
        xmax = 1.05 * max(_support_end(c, pdf), designed, ntv)
        ax = Axis(fig[1, 1]; xlabel=in_kat ? "$(array.coil_name) current needed (kAt)" : "factor on $(array.coil_name)'s current needed",
            ylabel=in_kat ? "probability density (per kAt)" : "probability density", limits=((0, xmax), (0, nothing)),
            title="Current that cancels each sampled machine's intrinsic overlap")
        band!(ax, c, zeros(length(c)), pdf; color=(WONG[1], 0.25))
        lines!(ax, c, pdf; color=WONG[1], linewidth=2, label="over the tolerance Monte Carlo")
        designed > 1e-3 * maximum(c) && vlines!(ax, designed; color=:black, linestyle=:dot, linewidth=1.5,
            label="as designed: $(_num(designed))" * (in_kat ? " kAt" : ""))
        if limit !== nothing && isfinite(limit.abs_delta_max_correctable)
            vlines!(ax, limit.abs_delta_max_correctable * per; color=:firebrick, linewidth=2,
                label="NTV limit at $(_num(limit.torque_budget)) N·m ($(limit.model)): $(_pct(limit.probability_percent)) beyond")
        end
        axislegend(ax; position=:rt)
        fig
    end
end

function EFA._plot_error_field_summary(d, T, threshold, risk, keep)
    return themed() do
        fig = Figure(; size=(keep === nothing ? 860 : 1250, threshold && risk ? 820 : 480))
        left = Any[]
        threshold && push!(left, pos -> _threshold_panel!(pos, [("", d, T)]))
        risk && push!(left, pos -> _risk_panel!(pos, [("", d)]; target_percent=1.0))
        for (r, panel!) in enumerate(left)
            panel!(fig[r, 1])
        end
        if keep !== nothing
            names = d.coil_names[keep]
            values = abs.(d.abs_delta_shift_per_mm[keep])
            ax = Axis(fig[1:max(1, length(left)), isempty(left) ? 1 : 2]; xscale=log10, xminorticksvisible=false, xlabel="|δ| per mm of shift")
            omitted = _stem_rows!(ax, names, [values], [""]; log=true)
            ax.title = omitted == 0 ? "Shift sensitivity" : "Shift sensitivity ($omitted with no n ≠ 0 field not shown)"
            isempty(left) || colsize!(fig.layout, 2, Relative(0.32))
        end
        fig
    end
end
