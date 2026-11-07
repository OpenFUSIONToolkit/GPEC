# ForceFreeStates and local-stability figures. Each panel draws into its own nested grid `gl`, so the single plots
# and the summary share one drawing function per panel.

const FFS = A.ForceFreeStates

# A figure holding one panel in a nested grid.
function _single(panel!, size, args...; kwargs...)
    return themed() do
        fig = Figure(; size)
        gl = fig[1, 1] = GridLayout()
        panel!(gl, args...; kwargs...)
        fig
    end
end

# Rational surfaces along the top of a ψ_N axis with the title above them, or just the title without surfaces.
function _titled_rationals!(pos, ax, rat, title)
    isempty(rat.psi) && return (ax.title = title)
    return _surfaces_on_top!(pos, ax, rat.psi, rat.labels; title)
end

_plural(n, word) = "$n $word" * (n == 1 ? "" : "s")

function _critical_determinant!(gl, d, rat)
    ax = Axis(gl[1, 1]; xlabel=PSIN, ylabel="critical determinant", yscale=Makie.pseudolog10, limits=((0, 1), _symlims(d.crit)))
    nz = count(i -> d.crit[i-1] < 0 <= d.crit[i], 2:length(d.crit))   # −→+ crossings, as the integrator counts them
    verdict = nz == 0 ? "Fixed boundary: no zero crossings → stable" : "Fixed boundary: $(_plural(nz, "zero crossing")) → unstable"
    _titled_rationals!(gl[1, 1], ax, rat, verdict)
    hlines!(ax, 0; color=:black, linewidth=1)
    lines!(ax, d.psi, d.crit; color=WONG[1], linewidth=1.8)
    return ax
end

const _ENERGY_STYLE = Dict(
    :total => (:et, rich("total  δW = δW", subscript("p"), " + δW", subscript("v")), :circle, 1),
    :plasma => (:ep, rich("plasma  δW", subscript("p")), :utriangle, 2),
    :vacuum => (:ev, rich("vacuum  δW", subscript("v")), :rect, 3)
)

function _energies!(gl, e; which=(:total, :plasma, :vacuum))
    et = real.(e.et)
    nu = count(<(0), et)
    title = nu == 0 ? rich("Free boundary: stable, least stable δW = ", _num(et[1])) : "Free boundary: $(_plural(nu, "unstable eigenmode"))"
    vals = reduce(vcat, (real.(getfield(e, _ENERGY_STYLE[w][1])) for w in which))
    lims = _symlims(vals)
    ax = Axis(gl[1, 1]; xlabel="eigenmode (most unstable first)", ylabel="eigenvalue of W·v = λN·v", title, yscale=Makie.pseudolog10, limits=(nothing, lims))
    hspan!(ax, lims[1], 0; color=(:firebrick, 0.06))
    hlines!(ax, 0; color=:black, linewidth=1)
    for w in which
        field, label, marker, k = _ENERGY_STYLE[w]
        v = real.(getfield(e, field))
        scatter!(ax, 1:length(v), v; color=WONG[k], marker, markersize=k == 1 ? 9 : 7, label)
    end
    length(which) > 1 && axislegend(ax; position=:rb)
    return ax
end

function _mode_displacement!(gl, d, rat; modes=nothing)
    amp = abs.(d.xi)
    rows = if modes === nothing
        sort(sortperm(vec(maximum(amp; dims=2)); rev=true)[1:min(6, size(amp, 1))])
    else
        [m - d.mlow + 1 for m in modes if 1 <= m - d.mlow + 1 <= size(amp, 1)]
    end
    isempty(rows) && throw(ArgumentError("plot_mode_displacement: no requested m lies in $(d.mlow):$(d.mlow + size(amp, 1) - 1)"))
    scaled, scale = _decade(amp[rows, :])
    ax = Axis(gl[1, 1]; xlabel=PSIN, ylabel=rich("|ξ", superscript("ψ"), subscript("m"), "|", _unit(scale, "")), limits=((0, 1), (0, nothing)))
    title = rich("ξ", superscript("ψ"), subscript("m"), " of the least stable free-boundary eigenmode", d.dW === nothing ? "" : ", δW = " * _num(real(d.dW)))
    _titled_rationals!(gl[1, 1], ax, rat, title)
    colors = Makie.resample_cmap(:viridis, length(rows) + 1)
    for (j, i) in enumerate(rows)
        lines!(ax, d.psi, scaled[j, :]; color=colors[j], linewidth=1.8, label="m = $(d.mlow + i - 1)")
    end
    axislegend(ax; position=:lt, nbanks=2)
    return ax
end

function _eigenvectors!(gl, d; nmodes=12)
    v = abs.(d.wt)
    m = d.mlow .+ (0:(size(v, 1)-1))
    k = min(size(v, 2), nmodes)
    ax = Axis(gl[1, 1]; xlabel="poloidal harmonic m", ylabel="eigenmode", yreversed=true, title="Free-boundary eigenmodes, least stable at top",
        yticks=1:k)
    hm = heatmap!(ax, m, 1:k, v[:, 1:k]; colormap=:viridis)
    Colorbar(gl[1, 2], hm; label=rich("|v", subscript("m"), "| (unit power norm)"))
    return ax
end

function _delta_strip!(gl, dp)
    d = real.([dp.matrix[i, i] for i in axes(dp.matrix, 1)])
    x = 1:length(d)
    lims = _symlims(d)
    npos = count(>(0), d)
    title = npos == 0 ? rich("Tearing: Δ′", subscript("ii"), " < 0 on every surface") : rich("Tearing: Δ′", subscript("ii"), " > 0 on $npos of $(length(d)) surfaces")
    ax = Axis(gl[1, 1]; xticks=(x, dp.labels), xlabel="rational surface q = m/n", ylabel=rich("Re Δ′", subscript("ii")), title,
        yscale=Makie.pseudolog10, limits=((0.4, length(d) + 0.6), lims))
    hspan!(ax, 0, lims[2]; color=(:firebrick, 0.06))
    hlines!(ax, 0; color=:black, linewidth=1)
    rangebars!(ax, x, zeros(length(d)), d; color=(:black, 0.35), whiskerwidth=0)
    scatter!(ax, x, d; color=[v > 0 ? :firebrick : WONG[1] for v in d], markersize=12)
    return ax
end

FFS._plot_critical_determinant(d, rat) = _single(_critical_determinant!, (820, 460), d, rat)
FFS._plot_eigenvalues(e; which=(:total, :plasma, :vacuum)) = _single(_energies!, (820, 460), e; which)
FFS._plot_mode_displacement(d, rat; modes=nothing) = _single(_mode_displacement!, (820, 480), d, rat; modes)
FFS._plot_energy_eigenvectors(d; nmodes=12) = _single(_eigenvectors!, (820, 520), d; nmodes)

function FFS._plot_edge_stability_scan(d)
    return themed() do
        fig = Figure(; size=(820, 460))
        vals = vcat(real.(d.et), real.(d.ep), real.(d.ev))
        lims = _symlims(vals)
        ax = Axis(fig[1, 1]; xlabel="edge truncation q", ylabel="free-boundary energy", yscale=Makie.pseudolog10, limits=(nothing, lims),
            title="Edge scan: truncation at q = $(_num(d.qlim))")
        hspan!(ax, lims[1], 0; color=(:firebrick, 0.06))
        hlines!(ax, 0; color=:black, linewidth=1)
        vlines!(ax, d.qlim; color=(:black, 0.5), linestyle=:dash, linewidth=1.2)
        for (key, w) in ((:et, :total), (:ep, :plasma), (:ev, :vacuum))
            _, label, marker, k = _ENERGY_STYLE[w]
            v = real.(getfield(d, key))
            scatterlines!(ax, d.q, v; color=WONG[k], marker, markersize=6, label)
        end
        axislegend(ax; position=:rb)
        fig
    end
end

# The Δ′ family: complex entries drawn as their real part (default), imaginary part or magnitude.
function _part(z, part)
    part === :real && return real.(z)
    part === :imag && return imag.(z)
    return abs.(z)
end
_part_label(part, sym) = part === :real ? rich("Re ", sym) : part === :imag ? rich("Im ", sym) : rich("|", sym, "|")

# Signed parts on a diverging symmetric-log map, linear within 10⁻³ of the largest entry; magnitudes on a log map
# from the smallest nonzero entry.
function _complex_heatmap!(ax, x, y, v, part)
    lim = max(maximum(abs, v; init=0.0), floatmin())
    if part === :abs
        positive = filter(>(0), v)
        lo = isempty(positive) ? 1e-6lim : minimum(positive)
        return heatmap!(ax, x, y, max.(v, lo); colormap=:viridis, colorscale=log10, colorrange=(lo, lim))
    end
    return heatmap!(ax, x, y, v; colormap=Reverse(:RdBu), colorrange=(-lim, lim), colorscale=Makie.Symlog10(1e-3lim))
end

# Text colour for a value written on its own heatmap cell: white on the dark ends of the map, black elsewhere.
function _cell_text_color(v, v_all, part)
    if part === :abs
        lo, hi = log10.(extrema(filter(>(0), v_all)))
        return hi > lo && (log10(max(v, 10.0^lo)) - lo) / (hi - lo) < 0.35 ? :white : :black
    end
    lim = maximum(abs, v_all)
    lin = 1e-3lim
    return log1p(abs(v) / lin) / log1p(lim / lin) > 0.7 ? :white : :black
end

# Surface/side axes: L and R ticks on the data axis, the q = m/n group label beyond them, white separators.
function _surface_side_axes!(pos, labels; xgroups=true, kwargs...)
    ns = length(labels)
    lr = [isodd(i) ? "L" : "R" for i in 1:(2ns)]
    groups = ["q = " * l for l in labels]
    ax = Axis(pos; yticks=(1:(2ns), lr), yreversed=true, ygridvisible=false, xgridvisible=false, kwargs...)
    g = Axis(pos; yticks=(1.5:2:(2ns), groups), yreversed=true, yticksvisible=false, yticklabelpad=24, ylabel="responding surface and side",
        xgridvisible=false, ygridvisible=false, xticksvisible=false, xticklabelsvisible=false)
    if xgroups
        ax.xticks = (1:(2ns), lr)
        g.xticks = (1.5:2:(2ns), groups)
        g.xticklabelsvisible = true
        g.xticklabelpad = 22
        g.xlabel = "driving surface and side"
    end
    haskey(kwargs, :limits) && (g.limits = kwargs[:limits])
    hidespines!(g)
    linkaxes!(ax, g)
    for s in 1:(ns-1)
        hlines!(ax, 2s + 0.5; color=:white, linewidth=2)
        xgroups && vlines!(ax, 2s + 0.5; color=:white, linewidth=2)
    end
    return ax
end

function FFS._plot_delta_prime(dp; part=:real)
    return themed() do
        v = _part(dp.matrix, part)
        k = length(dp.labels)
        labels = ["q = " * l for l in dp.labels]
        fig = Figure(; size=(620, 520))
        ax = Axis(fig[1, 1]; xticks=(1:k, labels), yticks=(1:k, labels), yreversed=true, aspect=DataAspect(), xgridvisible=false, ygridvisible=false,
            title=rich(_part_label(part, "Δ′"), " between rational surfaces (PEST-3, ", PSIN, "-referenced)"), xlabel="driving surface",
            ylabel="responding surface")
        hm = _complex_heatmap!(ax, 1:k, 1:k, permutedims(v), part)
        for i in 1:k, j in 1:k
            text!(ax, j, i; text=_num(v[i, j]), align=(:center, :center), fontsize=12, color=_cell_text_color(v[i, j], v, part))
        end
        Colorbar(fig[1, 2], hm; label=_part_label(part, sub("Δ′", "ij")))
        resize_to_layout!(fig)
        fig
    end
end

function FFS._plot_delta_prime_raw(dp; part=:real)
    return themed() do
        ns = length(dp.labels)
        fig = Figure(; size=(760, 700))
        ax = _surface_side_axes!(fig[1, 1], dp.labels; aspect=DataAspect(), limits=((0.5, 2ns + 0.5), (0.5, 2ns + 0.5)),
            title=rich(_part_label(part, "D′"), ": outer-region matching matrix"))
        hm = _complex_heatmap!(ax, 1:(2ns), 1:(2ns), permutedims(_part(dp.raw, part)), part)
        Colorbar(fig[1, 2], hm; label=_part_label(part, "D′"))
        resize_to_layout!(fig)
        fig
    end
end

function FFS._plot_delta_ef(dp; part=:real)
    return themed() do
        ns = length(dp.labels)
        me = dp.mlow .+ (0:(size(dp.ef, 1)-1))
        fig = Figure(; size=(900, 560))
        ax = _surface_side_axes!(fig[1, 1], dp.labels; xgroups=false, xlabel="applied edge harmonic m",
            title=rich(_part_label(part, sub("Δ", "EF")), ": surface response to each edge harmonic"),
            limits=((me[1] - 0.5, me[end] + 0.5), (0.5, 2ns + 0.5)))
        hm = _complex_heatmap!(ax, me, 1:(2ns), _part(dp.ef, part), part)
        Colorbar(fig[1, 2], hm; label=_part_label(part, sub("Δ", "EF")))
        fig
    end
end

function FFS._plot_cond_fbar(d, rat)
    return themed() do
        fbar = L"\mathrm{cond}(\bar{F})"
        fig = Figure(; size=(900, 480))
        ax = Axis(fig[1, 1]; xlabel=PSIN, ylabel=fbar, yscale=log10, limits=((0, 1), nothing))
        title = d.kmsing == 0 ? "Kinetic singular surfaces: none above threshold" : "Kinetic singular surfaces: $(d.kmsing) above threshold"
        _titled_rationals!(fig[1, 1], ax, rat, title)
        lines!(ax, d.scan_psi, d.scan_cond; color=WONG[1], linewidth=2, label=fbar)
        d.thr > 0 && hlines!(ax, d.thr; color=:firebrick, linestyle=:dash, linewidth=1.5, label="threshold $(_num(d.thr))")
        isempty(d.k_psi) || vlines!(ax, d.k_psi; color=:firebrick, linewidth=2, label="kinetic singular surface")
        axislegend(ax; position=:lb)
        fig
    end
end

function _ballooning_verdict(alpha, alpha_crit)
    ok = isfinite.(alpha_crit)
    nu = count(ok .& (alpha .> alpha_crit))
    return nu == 0 ? "Infinite-n ballooning: stable on every surface" : "Infinite-n ballooning: unstable on $nu of $(count(ok)) surfaces"
end

function _alpha_top(curves...)
    finite = filter(isfinite, reduce(vcat, curves))
    return isempty(finite) ? 1.0 : 1.3 * maximum(finite)
end

function FFS._plot_ballooning_alpha_boundary(d; psi_min=0.0)
    return themed() do
        unstable = isfinite.(d.alpha_crit) .& (d.alpha .> d.alpha_crit)
        fig = Figure(; size=(820, 460))
        ax = Axis(fig[1, 1]; xlabel=PSIN, ylabel="normalized pressure gradient α", limits=((psi_min, 1), (0, _alpha_top(d.alpha, d.alpha_crit))),
            title=_ballooning_verdict(d.alpha, d.alpha_crit))
        band!(ax, d.psi, d.alpha, ifelse.(unstable, d.alpha_crit, d.alpha); color=(:firebrick, 0.15))
        lines!(ax, d.psi, d.alpha; color=:black, linewidth=2.5, label="equilibrium α")
        lines!(ax, d.psi, d.alpha_crit; color=WONG[6], linewidth=2, linestyle=:dash, label="first stability boundary")
        axislegend(ax; position=:lt)
        fig
    end
end

# Each surface's Δ′(α), signed so that positive is stable everywhere, regridded from its own α = α_ref·scale onto a
# shared α axis; NaN past a surface's scan.
function _signed_delta_prime_map(dpmap)
    psi, scales, aref = dpmap.psi, dpmap.alpha_scales, dpmap.alpha_ref
    signed = dpmap.delta_prime .* sign.(dpmap.delta_prime[:, 1])
    finite_aref = filter(isfinite, aref)
    isempty(finite_aref) && throw(ArgumentError("plot_ballooning_alpha_boundaries: the Δ′ map has no valid surfaces"))
    alpha = collect(range(0.0, maximum(finite_aref) * scales[end]; length=200))
    z = fill(NaN, length(psi), length(alpha))
    for i in eachindex(psi)
        (isfinite(aref[i]) && aref[i] > 0) || continue
        a_s = aref[i] .* scales
        for (j, a) in enumerate(alpha)
            a > a_s[end] && break
            k = min(searchsortedlast(a_s, a), length(a_s) - 1)
            k < 1 && continue
            t = (a - a_s[k]) / (a_s[k+1] - a_s[k])
            z[i, j] = (1 - t) * signed[i, k] + t * signed[i, k+1]
        end
    end
    return psi, alpha, z
end

function FFS._plot_ballooning_alpha_boundaries(bnd, dpmap; psi_min=0.0)
    return themed() do
        fig = Figure(; size=(900, 560))
        top = _alpha_top(bnd.alpha, bnd.alpha_critical1, bnd.alpha_critical2)
        ax = Axis(fig[1, 1]; xlabel=PSIN, ylabel="normalized pressure gradient α", limits=((psi_min, 1), (0, top)),
            title=_ballooning_verdict(bnd.alpha, bnd.alpha_critical1))
        if dpmap !== nothing
            psi, alpha, z = _signed_delta_prime_map(dpmap)
            finite = filter(isfinite, z)
            lim = isempty(finite) ? 1.0 : maximum(abs, finite)
            hm = heatmap!(ax, psi, alpha, z; colormap=:RdBu, colorrange=(-lim, lim), colorscale=Makie.Symlog10(1e-3lim), nan_color=:transparent)
            contour!(ax, psi, alpha, z; levels=[0.0], color=:black, linewidth=1)
            Colorbar(fig[1, 2], hm; label=rich("Δ′ · sign(Δ′", subscript("α=0"), ")   (> 0 stable)"))
        end
        lines!(ax, bnd.psi, bnd.alpha; color=:black, linewidth=2.5, label="equilibrium α")
        lines!(ax, bnd.psi, bnd.alpha_critical1; color=WONG[6], linewidth=2, linestyle=:dash, label="first stability boundary")
        lines!(ax, bnd.psi, bnd.alpha_critical2; color=WONG[2], linewidth=2, linestyle=:dash, label="second stability boundary")
        axislegend(ax; position=:lt)
        fig
    end
end

const _SUMMARY_PANELS = Dict(
    :critical_determinant => _critical_determinant!,
    :energies => _energies!,
    :mode_displacement => _mode_displacement!,
    :delta_prime => _delta_strip!,
    :eigenvectors => _eigenvectors!
)

# Two panels per row; an odd last panel spans both columns, so no cell is empty.
function FFS._plot_ffs_summary(panels)
    return themed() do
        n = length(panels)
        fig = Figure(; size=(n == 1 ? 820 : 1150, 410 * cld(n, 2)))
        for (i, (kind, data...)) in enumerate(panels)
            row = cld(i, 2)
            col = (n == 1 || (i == n && isodd(n))) ? (1:2) : (isodd(i) ? 1 : 2)
            gl = fig[row, col] = GridLayout()
            _SUMMARY_PANELS[kind](gl, data...)
        end
        colgap!(fig.layout, 18)
        rowgap!(fig.layout, 14)
        fig
    end
end
