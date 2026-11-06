# PerturbedEquilibrium figures: resonant coupling per rational surface, response spectrograms and the summary.

const PE = A.PerturbedEquilibrium

# Quantities spanning more than two decades go on a log axis; the rest are scaled by a power of ten.
function _magnitude_row(v, name, unit)
    positive = filter(>(0), v)
    if !isempty(positive) && maximum(positive) / minimum(positive) > 100
        return (v, rich(name, " (", unit, ")"), log10)
    end
    scaled, scale = _decade(v)
    return (scaled, rich(name, _unit(scale, unit)), identity)
end

# Each coupling row: values, label, y scale and reference line (nothing for none).
function _coupling_row(d, key)
    key === :field && return (_magnitude_row(abs.(d.field), rich("|b̄", subscript("res"), "|"), "T")..., nothing)
    key === :current && return (_magnitude_row(abs.(d.current), rich("|I", subscript("res"), "|"), "A")..., nothing)
    key === :island_hw && return (d.island_hw, rich("w/2 (", PSIN, ")"), identity, nothing)
    key === :chirikov && return (d.chirikov, "effective island-overlap\nChirikov parameter", identity, 1.0)
    return (real.(d.delta_prime), "driven Re Δ′", identity, 0.0)
end

const _ROW_TITLES = Dict(
    :field_pen => "Resonant and penetrated field at each rational surface",
    :field => "Resonant field at each rational surface",
    :current => "Resonant current at each rational surface",
    :island_hw => "Island half-width at each rational surface",
    :delta_prime => "Forcing-driven Δ′ at each rational surface"
)

function _coupling_title(d, keys)
    :chirikov in keys || return _ROW_TITLES[only(keys) === :field && d.penetrated !== nothing ? :field_pen : only(keys)]
    k = count(>=(1), d.chirikov)
    return k == 0 ? "Resonant coupling: Chirikov parameter < 1 on every surface" : "Resonant coupling: islands overlap on $k of $(length(d.chirikov)) surfaces"
end

# The field row with the penetrated field: b̄_res and b̄_pen as paired stems on one shared scale.
function _field_pair!(ax, d, x)
    v, ylabel, yscale = _magnitude_row(vcat(abs.(d.field), abs.(d.penetrated)), rich("|b̄|"), "T")
    k = length(x)
    ax.ylabel, ax.yscale = ylabel, yscale
    positive = filter(>(0), v)
    base = yscale === log10 ? minimum(positive) / 3 : 0.0
    yscale === identity && ylims!(ax, 0, 1.12 * maximum(positive))
    for (j, (y, name)) in enumerate(((v[1:k], "res"), (v[(k+1):end], "pen")))
        xs = x .+ (j - 1.5) * 0.24
        rangebars!(ax, xs, fill(base, k), y; color=(WONG[j], 0.35), whiskerwidth=0, linewidth=2)
        scatter!(ax, xs, y; color=WONG[j], markersize=11, label=rich("|b̄", subscript(name), "|"))
    end
    axislegend(ax; position=:lt)
    return ax
end

# Rows of stems and dots sharing a categorical q = m/n axis, one colour per toroidal mode number.
function _coupling_rows!(gl, d, keys)
    labels = ["$m/$n" for (m, n) in zip(d.m, d.n)]
    x = 1:length(labels)
    ns = unique(d.n)
    colors = [WONG[findfirst(==(n), ns)] for n in d.n]
    axes = Axis[]
    for (r, key) in enumerate(keys)
        y, ylabel, yscale, ref = _coupling_row(d, key)
        last = r == length(keys)
        ax = Axis(gl[r, 1]; ylabel, yscale, xticks=(x, labels), limits=((0.4, length(x) + 0.6), nothing), xlabel=last ? "rational surface q = m/n" : "",
            xticklabelsvisible=last, ylabelsize=14, title=r == 1 ? _coupling_title(d, keys) : "")
        if key === :field && d.penetrated !== nothing
            push!(axes, _field_pair!(ax, d, x))
            continue
        end
        ref === nothing || hlines!(ax, ref; color=:black, linewidth=1, linestyle=ref == 1.0 ? :dash : :solid)
        positive = filter(>(0), y)
        base = yscale === log10 ? minimum(positive) / 3 : 0.0
        yscale === identity && ref === nothing && !isempty(positive) && ylims!(ax, 0, 1.12 * maximum(positive))
        rangebars!(ax, x, fill(base, length(y)), y; color=[(c, 0.35) for c in colors], whiskerwidth=0, linewidth=2)
        scatter!(ax, x, y; color=colors, markersize=11)
        if r == 1 && length(ns) > 1
            axislegend(ax, [MarkerElement(; color=WONG[k], marker=:circle) for k in eachindex(ns)], ["n = $n" for n in ns]; position=:rt)
        end
        push!(axes, ax)
    end
    linkxaxes!(axes...)
    rowgap!(gl, 6)
    return axes
end

function PE._plot_resonant_coupling(d, keys)
    return themed() do
        fig = Figure(; size=length(keys) == 1 ? (720, 400) : (620, 150 * length(keys) + 70))
        gl = fig[1, 1] = GridLayout()
        _coupling_rows!(gl, d, keys)
        fig
    end
end

const _COMPONENT_STYLE = Dict(
    :xi_psi => ("Plasma displacement", rich("ξ", superscript("ψ")), ""),
    :b_psi => ("Area-weighted normal field", rich("b̄", superscript("ψ")), "T"),
    :Jb_theta => ("Poloidal field", rich("J b", superscript("θ")), "T m²"),
    :Jb_zeta => ("Toroidal field", rich("J b", superscript("ζ")), "T m²")
)

# Piecewise-linear interpolation of (xs, ys) at x, held at the end values outside xs.
function _lerp(xs, ys, x)
    k = clamp(searchsortedlast(xs, x), 1, length(xs) - 1)
    t = clamp((x - xs[k]) / (xs[k+1] - xs[k]), 0.0, 1.0)
    return (1 - t) * ys[k] + t * ys[k+1]
end

# |harmonic| of the first toroidal mode number on a uniform ψ grid: the response grid holds the integrator's steps,
# which CairoMakie would draw as slanted cells.
function _uniform_harmonics(d)
    mpert = d.mhigh - d.mlow + 1
    order = sortperm(d.psi)
    psi, z = d.psi[order], abs.(d.data[order, 1:mpert])
    keep = [true; diff(psi) .> 0]
    psi, z = psi[keep], z[keep, :]
    pu = collect(range(psi[1], psi[end]; length=600))
    zu = [_lerp(psi, view(z, :, j), p) for p in pu, j in 1:mpert]
    return collect(d.mlow:d.mhigh), psi, z, pu, zu
end

function _spectrogram!(gl, d, component; style=:log_heatmap)
    name, glyph, unit = _COMPONENT_STYLE[component]
    ms, psi, z, pu, zu = _uniform_harmonics(d)
    amp = rich("|", glyph, subscript("m"), "|")
    if style === :lines
        top = sort(sortperm(vec(maximum(z; dims=1)); rev=true)[1:min(8, length(ms))])
        zl, scale = _decade(z[:, top])
        ax = Axis(gl[1, 1]; xlabel=PSIN, ylabel=rich(amp, _unit(scale, unit)), title=rich(name, " ", amp, ", largest harmonics"),
            limits=((0, 1), (0, nothing)))
        isempty(d.rational_psi) || vlines!(ax, d.rational_psi; color=(:black, 0.15), linewidth=1)
        colors = Makie.resample_cmap(:viridis, length(top) + 1)
        for (k, j) in enumerate(top)
            lines!(ax, psi, zl[:, k]; color=colors[k], linewidth=1.8, label="m = $(ms[j])")
        end
        axislegend(ax; position=:lt, nbanks=2)
        return ax
    end
    ax = Axis(gl[1, 1]; xlabel="poloidal harmonic m", ylabel=PSIN, title=rich(name, " ", amp), limits=((ms[1] - 0.5, ms[end] + 0.5), (0, 1)))
    zmax = max(maximum(zu), floatmin())
    if style === :log_heatmap
        hm = heatmap!(ax, ms, pu, permutedims(zu); colormap=:viridis, colorscale=log10, colorrange=(1e-4zmax, zmax), lowclip=Makie.to_colormap(:viridis)[1])
        label = rich(amp, _unit("", unit))
    else
        zs, scale = _decade(zu)
        hm = heatmap!(ax, ms, pu, permutedims(zs); colormap=:viridis, colorrange=(0, maximum(zs)))
        label = rich(amp, _unit(scale, unit))
    end
    isempty(d.rational_psi) || hlines!(ax, d.rational_psi; color=(:white, 0.4), linewidth=1)
    lines!(ax, [d.nhigh * _lerp(d.q_psi, d.q, p) for p in pu], pu; color=:white, linewidth=1.5, linestyle=:dash, label="m = n q(ψ)")
    axislegend(ax; position=:lb)
    Colorbar(gl[1, 2], hm; label)
    return ax
end

function PE._plot_mode_spectrogram(d, component; style=:log_heatmap)
    return themed() do
        fig = Figure(; size=style === :lines ? (820, 480) : (900, 560))
        gl = fig[1, 1] = GridLayout()
        _spectrogram!(gl, d, component; style)
        fig
    end
end

# |b̄^ψ_m| at the outermost surface as a step line, with the resonant harmonics marked.
function _edge_spectrum!(gl, d)
    mpert = d.mhigh - d.mlow + 1
    ms = collect(d.mlow:d.mhigh)
    v, scale = _decade(abs.(d.data[end, 1:mpert]))
    ax = Axis(gl[1, 1]; xlabel="poloidal harmonic m", ylabel=rich("|b̄", superscript("ψ"), subscript("m"), "| at the edge", _unit(scale, "T")),
        title="Edge field spectrum", limits=(nothing, (0, nothing)))
    isempty(d.rational_m) || vlines!(ax, d.rational_m; color=(:black, 0.15), linewidth=1)
    stairs!(ax, vcat(ms[1] - 1, ms, ms[end] + 1), vcat(0.0, v, 0.0); step=:center, color=WONG[1], linewidth=2)
    return ax
end

const _NORMAL_STYLE = Dict(:b_n => (rich("b", subscript("n")), "T"), :xi_n => (rich("ξ", subscript("n")), "m"))

# Outward unit normals of a closed counter-clockwise outline, from central differences.
function _outward_normals(R, Z)
    dR = circshift(R, -1) .- circshift(R, 1)
    dZ = circshift(Z, -1) .- circshift(Z, 1)
    dR[1] = dR[end] = R[2] - R[end-1]
    dZ[1] = dZ[end] = Z[2] - Z[end-1]
    L = hypot.(dR, dZ)
    return dZ ./ L, -dR ./ L
end

# A displacement scale rounded down to one significant figure, so the largest displacement is `frac` of the minor radius.
function _display_scale(R, xi; frac=0.05)
    s = frac * (maximum(R) - minimum(R)) / 2 / max(maximum(abs, xi), floatmin())
    p = 10.0^floor(log10(s))
    return max(1, round(Int, floor(s / p) * p))
end

# b_n or ξ_n on the poloidal plane in `gl`, with its colourbar, and the displaced boundary when given.
function _field_rz_panel!(gl, d, component, boundary)
    glyph, unit = _NORMAL_STYLE[component]
    values, scale = _decade(d.values)
    title =
        boundary === nothing ? rich(glyph, " at φ = ", _num(rad2deg(d.phi)), "°") :
        rich(glyph, " at φ = ", _num(rad2deg(d.phi)), "°: largest boundary |ξ", subscript("n"), "| = ", _num(1e3 * maximum(abs, boundary.xi)), " mm")
    ax = Axis(gl[1, 1]; title)
    hm = _poloidal_slice!(ax, d.R, d.Z, values; outlines=d.outlines)
    Colorbar(gl[1, 2], hm; label=rich(glyph, _unit(scale, unit)))
    # The column takes the plasma's own width-to-height ratio, so the panel leaves no empty band beside it.
    colsize!(gl, 1, Aspect(1, (maximum(d.R) - minimum(d.R)) / (maximum(d.Z) - minimum(d.Z))))
    if boundary !== nothing
        nR, nZ = _outward_normals(boundary.R, boundary.Z)
        s = _display_scale(boundary.R, boundary.xi)
        lines!(ax, boundary.R, boundary.Z; color=(:black, 0.45), linestyle=:dash, linewidth=1.2, label="equilibrium")
        lines!(ax, boundary.R .+ s .* boundary.xi .* nR, boundary.Z .+ s .* boundary.xi .* nZ; color=:black, linewidth=2,
            label=rich("displaced, ξ", subscript("n"), " × $s"))
        Legend(gl[2, 1:2], ax; orientation=:horizontal, framevisible=false)
        rowgap!(gl, 4)
    end
    return ax
end

function PE._plot_perturbed_field_rz(d, component, boundary)
    return themed() do
        fig = Figure(; size=(560, boundary === nothing ? 680 : 720))
        gl = fig[1, 1] = GridLayout()
        _field_rz_panel!(gl, d, component, boundary)
        fig
    end
end

function PE._plot_perturbed_equilibrium_summary(coupling, edge, xi, rz)
    return themed() do
        top = Any[]
        coupling === nothing || push!(top, gl -> _coupling_rows!(gl, coupling, (:field,)))
        edge === nothing || push!(top, gl -> _edge_spectrum!(gl, edge))
        rows = (isempty(top) ? 0 : 1) + (xi === nothing ? 0 : 1)
        ncol = max(1, length(top))
        fig = Figure(; size=(1150 + (rz === nothing ? 0 : 480), rows < 2 ? 560 : 820))
        for (c, draw!) in enumerate(top)
            gl = fig[1, c] = GridLayout()
            draw!(gl)
        end
        if xi !== nothing
            gl = fig[isempty(top) ? 1 : 2, 1:ncol] = GridLayout()
            _spectrogram!(gl, xi, :xi_psi)
            isempty(top) || rowsize!(fig.layout, 1, Relative(0.38))
        end
        # b_n on the poloidal plane, a column spanning the other panels' height.
        if rz !== nothing
            gl = fig[1:max(1, rows), rows == 0 ? 1 : ncol + 1] = GridLayout()
            _field_rz_panel!(gl, rz, :b_n, nothing)
        end
        fig
    end
end

function PE._plot_islands(d, name)
    return themed() do
        chains = d.chains
        isempty(chains) && throw(ArgumentError("no island has a nonzero half-width"))
        colors = Makie.resample_cmap(:viridis, length(chains) + 2)
        fig = Figure(; size=(1250, 640))
        # (θ, ψ_N): the chains in flux coordinates, the outboard midplane central.
        bottom = max(0.0, minimum(c.psis - 1.5c.w for c in chains) - 0.03)
        a = Axis(fig[1, 1]; xlabel="poloidal angle θ (deg)", ylabel=PSIN, xticks=-180:90:180, limits=((-180, 180), (bottom, 1.02)),
            title="$name at φ = $(_num(rad2deg(d.phi)))°")
        for (k, c) in enumerate(chains)
            shift = circshift(1:length(c.theta), length(c.theta) ÷ 2)
            band!(a, sort(c.theta), c.lo[shift], c.hi[shift]; color=(colors[k+1], 0.6))
            hlines!(a, c.psis; color=(colors[k+1], 0.9), linewidth=0.8, linestyle=:dash)
            text!(a, -178, c.psis; text=c.label, align=(:left, :center), fontsize=11)
        end
        # (R, Z): the same chains on the poloidal plane.
        b = Axis(fig[1, 2]; aspect=DataAspect(), xlabel="R (m)", ylabel="Z (m)", title="On the poloidal plane")
        for s in d.surfaces[1:(end-1)]
            lines!(b, s...; color=(:black, 0.15), linewidth=0.8)
        end
        for (k, c) in enumerate(chains)
            poly!(b, Point2f.(c.polygon); color=(colors[k+1], 0.7), strokewidth=0)
        end
        lines!(b, d.surfaces[end]...; color=:black, linewidth=1.5)
        Legend(fig[1, 3], [PolyElement(; color=(colors[k+1], 0.7)) for k in eachindex(chains)], ["$(c.label):  $(_num(c.w))" for c in chains],
            rich("w/2 (", PSIN, ")"); framevisible=false)
        colsize!(fig.layout, 1, Relative(0.5))
        fig
    end
end
