# Equilibrium figures: profiles against ψ_N, flux surfaces in (R, Z), the summary and the Grad-Shafranov residual.

const EQ = A.Equilibrium

# A rational q as "m/n", e.g. 2 → "2/1" and 1.5 → "3/2".
function _rational_label(q)
    r = rationalize(q; tol=1e-6)
    return "$(numerator(r))/$(denominator(r))"
end

# One profile against ψ_N, scaled by a power of ten when needed.
function _profile!(ax, psi, y, label)
    yv, scale = _decade(y)
    ax.ylabel = isempty(scale) ? label : rich(label, " (", scale, ")")
    lines!(ax, psi, yv; color=WONG[1], linewidth=2)
    xlims!(ax, 0, 1)
    return ax
end

# Rational surfaces labelled as m/n along the top of `ax`.
function _rationals_on_top!(fig, slot, ax, d)
    isempty(d.sing_psi) && return nothing
    return _surfaces_on_top!(fig, slot, ax, d.sing_psi, _rational_label.(d.sing_q))
end

function _single_profile(d, y, label)
    fig = Figure(; size=(760, 420))
    ax = Axis(fig[1, 1]; xlabel=PSIN)
    _profile!(ax, d.psi, y, label)
    _rationals_on_top!(fig, (1, 1), ax, d)
    return fig
end

function EQ._plot_qprofile(d; show_singular=true)
    return themed() do
        fig = Figure(; size=(760, 420))
        ax = Axis(fig[1, 1]; xlabel=PSIN)
        _profile!(ax, d.psi, d.q, "q")
        # q₀ labelled at the edge and q₉₅ at the axis, where q is far from each.
        for (v, name, x, align) in ((d.q0, "0", 0.98, (:right, :bottom)), (d.q95, "95", 0.02, (:left, :bottom)))
            hlines!(ax, v; color=(:black, 0.35), linestyle=:dot, linewidth=1)
            text!(ax, x, v; text=rich("q", subscript(name), " = ", _num(v)), align, fontsize=12, color=(:black, 0.6))
        end
        show_singular && _rationals_on_top!(fig, (1, 1), ax, d)
        fig
    end
end

EQ._plot_pressure_profile(d) = themed(() -> _single_profile(d, d.mu0p, rich("μ", subscript("0"), "p (T", superscript("2"), ")")))
EQ._plot_f_profile(d) = themed(() -> _single_profile(d, d.twopif, rich("2πF = RB", subscript("φ"), " (T m)")))

_closed(v) = [v; v[1]]

# Flux surfaces in (R, Z): `n_psi` surfaces, `n_theta` constant-θ spokes, the rational surfaces in colour, the axis.
function _flux_surfaces!(ax, d; n_psi=11, n_theta=0)
    R, Z = EQ._rz_grid(d)
    ax.aspect = DataAspect()
    ax.xlabel = "R (m)"
    ax.ylabel = "Z (m)"
    if n_theta > 0
        for j in round.(Int, range(1, length(d.theta); length=n_theta + 1))[1:(end-1)]
            lines!(ax, R[:, j], Z[:, j]; color=(:black, 0.15), linewidth=0.7)
        end
    end
    for i in round.(Int, range(2, length(d.psi); length=n_psi))
        lines!(ax, _closed(R[i, :]), _closed(Z[i, :]); color=(:black, 0.45), linewidth=0.8)
    end
    colors = Makie.resample_cmap(:viridis, length(d.sing_psi) + 2)
    for (k, (ps, q)) in enumerate(zip(d.sing_psi, d.sing_q))
        i = argmin(abs.(d.psi .- ps))
        lines!(ax, _closed(R[i, :]), _closed(Z[i, :]); color=colors[k+1], linewidth=2, label="q = $(_rational_label(q))")
    end
    scatter!(ax, [d.ro], [d.zo]; marker=:cross, color=:black, markersize=10)
    return ax
end

# The rational-surface legend below a flux-surface axis.
function _surface_legend!(fig, slot, ax, d)
    isempty(d.sing_psi) && return nothing
    return Legend(fig[slot...], ax; orientation=:horizontal, nbanks=cld(length(d.sing_psi), 4), tellheight=true, framevisible=false)
end

function EQ._plot_flux_surfaces(d; n_psi=11, n_theta=18)
    return themed() do
        fig = Figure(; size=(560, 760))
        ax = Axis(fig[1, 1]; title="Flux surfaces")
        _flux_surfaces!(ax, d; n_psi, n_theta)
        _surface_legend!(fig, (2, 1), ax, d)
        fig
    end
end

function EQ._plot_equilibrium_summary(g, prof, geo)
    return themed() do
        fig = Figure(; size=(1150, 760))
        head = Any[]
        for (sym, v, unit) in ((sub("q", "0"), g.q0, ""), (sub("q", "95"), g.q95, ""), (sub("β", "N"), g.betan, ""), ("κ", g.kappa, ""),
            (sub("I", "p"), g.ip, " MA"), (sub("B", "T"), g.bt, " T"))
            isnothing(v) && continue
            push!(head, sym, " = ", _num(v), unit, "    ")
        end
        Label(fig[0, 1:2], rich(head...); fontsize=15, tellwidth=false, halign=:left)
        ax = Axis(fig[1:3, 1]; title="Flux surfaces")
        _flux_surfaces!(ax, geo)
        _surface_legend!(fig, (4, 1), ax, geo)
        rows = ((prof.q, "q"), (prof.mu0p, rich("μ", subscript("0"), "p (T", superscript("2"), ")")), (prof.twopif, rich("2πF = RB", subscript("φ"), " (T m)")))
        axs = Axis[]
        for (r, (y, label)) in enumerate(rows)
            a = Axis(fig[r, 2]; xlabel=r == 3 ? PSIN : "", xticklabelsvisible=r == 3)
            isempty(prof.sing_psi) || vlines!(a, prof.sing_psi; color=(:black, 0.15), linewidth=1)
            _profile!(a, prof.psi, y, label)
            push!(axs, a)
        end
        linkxaxes!(axs...)
        _rationals_on_top!(fig, (1, 2), axs[1], prof)
        colsize!(fig.layout, 1, Relative(0.36))
        rowgap!(fig.layout, 8)
        colgap!(fig.layout, 24)
        fig
    end
end

# θ in degrees on (−180, 180], 0 at the outboard midplane.
_theta_deg(y) = (d = 360y; d > 180 ? d - 360 : d)

function EQ._plot_gse(gse; n_theta_lines=8)
    return themed() do
        idx = round.(Int, range(1, length(gse.ys); length=n_theta_lines + 1))[1:(end-1)]
        emax, imax = findmax(gse.error)
        fig = Figure(; size=(780, 460))
        ax = Axis(fig[1, 1]; yscale=log10, xlabel=PSIN, ylabel="Grad–Shafranov residual",
            title=rich("Grad–Shafranov residual: max ", _num(emax), " at ", PSIN, " = ", _num(gse.xs[imax[1]])))
        for i in idx
            e = gse.error[:, i]
            keep = e .> 0
            lines!(ax, gse.xs[keep], e[keep]; color=_theta_deg(gse.ys[i]), colormap=:romaO, colorrange=(-180, 180), linewidth=1.2)
        end
        keep = gse.errori .> 0
        lines!(ax, gse.xs[keep], gse.errori[keep]; color=:black, linewidth=2.4, label="θ-integrated")
        xlims!(ax, 0, 1)
        axislegend(ax; position=:lt)
        Colorbar(fig[1, 2]; colormap=:romaO, limits=(-180, 180), ticks=-180:90:180, label="poloidal angle θ (deg)")
        fig
    end
end
