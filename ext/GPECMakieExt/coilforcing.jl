# Coil forcing figures: coils in 3D and (R, Z), the forcing spectrum, control-surface maps and the 3D boundary.

const CF = A.CoilForcing

const _SURFACE_GREY = RGBf(0.93, 0.94, 0.96)

# Index runs of a conductor on the camera's side (`near = true`) or the far side of the vertical plane through the axis,
# each run extended by one point so the halves join.
function _coil_runs(x, y, azimuth, near)
    side = [(xi * cos(azimuth) + yi * sin(azimuth) >= 0) == near for (xi, yi) in zip(x, y)]
    runs = UnitRange{Int}[]
    i = 1
    while i <= length(side)
        if side[i]
            j = i
            while j < length(side) && side[j+1]
                j += 1
            end
            push!(runs, max(i - 1, 1):min(j + 1, length(side)))
            i = j + 1
        else
            i += 1
        end
    end
    return runs
end

function CF._plot_coils_3d(coil_sets, surface, field; title=nothing)
    return themed() do
        fig = Figure(; size=(900, 560), figure_padding=(30, 10, 10, 10))
        title = something(title, surface === nothing ? "Coils" : "Plasma boundary and coils")
        azimuth = 1.15π
        ax = Axis3(fig[1, 1]; aspect=:data, title, titlefont=:regular, xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", azimuth, elevation=0.32,
            protrusions=(60, 20, 40, 30))
        # Same map as b_n: a positive current drives positive b_n beneath the coil.
        ilim = isempty(coil_sets) ? 1.0 : max(maximum(abs, (I / 1e3 for cs in coil_sets for I in cs.currents)), floatmin())
        # CairoMakie draws in order with no depth test, so coil runs behind the plasma go before the surface and the rest
        # after it; GLMakie depth-tests either way.
        function coils!(near)
            for cs in coil_sets, j in 1:cs.ncoil, k in 1:cs.s
                x, y, z = cs.x[j, k, :], cs.y[j, k, :], cs.z[j, k, :]
                for r in _coil_runs(x, y, azimuth, near)
                    lines!(ax, x[r], y[r], z[r]; color=fill(cs.currents[j] / 1e3, length(r)), colormap=DIVERGING, colorrange=(-ilim, ilim),
                        linewidth=3)
                end
            end
        end
        coils!(false)
        col = 2
        if surface !== nothing
            X, Y, Z = surface
            if field === nothing
                # Reversed θ order turns the mesh normals outward, so the lit surface is light rather than in shadow.
                surface!(ax, reverse(X; dims=1), reverse(Y; dims=1), reverse(Z; dims=1); color=fill(0.0, size(X)), colormap=[_SURFACE_GREY, _SURFACE_GREY])
            else
                factor, scale = _power(field.values)
                lim = something(field.clim, max(maximum(abs, field.values), floatmin())) / factor
                surface!(ax, X, Y, Z; color=field.values ./ factor, colormap=DIVERGING, colorrange=(-lim, lim), shading=NoShading)
                label = isempty(scale) ? field.label : rich(field.label, "  ×", scale)
                Colorbar(fig[1, col]; colormap=DIVERGING, limits=(-lim, lim), label, height=Relative(0.6))
                col += 1
            end
        end
        coils!(true)
        isempty(coil_sets) || Colorbar(fig[1, col]; colormap=DIVERGING, limits=(-ilim, ilim), label="coil current (kA)", height=Relative(0.6))
        fig
    end
end

function CF._plot_coils_rz(coil_sets, boundary)
    return themed() do
        Rs = [vec(hypot.(cs.x, cs.y)) for cs in coil_sets]
        Zs = [vec(cs.z) for cs in coil_sets]
        Rall = reduce(vcat, Rs; init=boundary === nothing ? Float64[] : boundary[1])
        Zall = reduce(vcat, Zs; init=boundary === nothing ? Float64[] : boundary[2])
        # Room on the right for the set names; the axis width follows the data's aspect so no band is left empty.
        xlo, xhi = minimum(Rall) - 0.15, maximum(Rall) + 0.45
        ylo, yhi = minimum(Zall) - 0.1, maximum(Zall) + 0.1
        height = 620
        fig = Figure()
        ax = Axis(fig[1, 1]; aspect=DataAspect(), height, width=round(Int, height * (xhi - xlo) / (yhi - ylo)), limits=((xlo, xhi), (ylo, yhi)),
            xlabel="R (m)", ylabel="Z (m)", title=boundary === nothing ? "Coils" : "Coils and plasma boundary")
        boundary === nothing || lines!(ax, [boundary[1]; boundary[1][1]], [boundary[2]; boundary[2][1]]; color=:black, linewidth=2)
        for (i, (cs, R, Z)) in enumerate(zip(coil_sets, Rs, Zs))
            color = WONG[mod1(i, length(WONG))]
            # A single toroidal loop is a point in (R, Z): draw it as a marker.
            if maximum(R) - minimum(R) < 0.01 && maximum(Z) - minimum(Z) < 0.01
                scatter!(ax, [sum(R) / length(R)], [sum(Z) / length(Z)]; marker=:rect, markersize=9, color)
            else
                for j in 1:cs.ncoil, k in 1:cs.s
                    lines!(ax, hypot.(cs.x[j, k, :], cs.y[j, k, :]), cs.z[j, k, :]; color, linewidth=2)
                end
            end
            text!(ax, maximum(R) + 0.05, sum(Z) / length(Z); text=cs.name, fontsize=11, align=(:left, :center), color)
        end
        resize_to_layout!(fig)
        fig
    end
end

function CF._plot_forcing_spectrum(forcing_modes, mlow, mhigh, rational_m)
    return themed() do
        ns = sort(unique(f.n for f in forcing_modes))
        factor, scale = _power([abs(f.amplitude) for f in forcing_modes])
        ms = [f.m for f in forcing_modes]
        lo, hi = something(mlow, minimum(ms)), something(mhigh, maximum(ms))
        fig = Figure(; size=(820, 440))
        ax = Axis(fig[1, 1]; xlabel="poloidal harmonic m", ylabel=rich("|Φ", subscript("x"), "(m, n)|", _unit(scale, "")),
            title="Coil forcing spectrum on the control surface", limits=((lo - 1.5, hi + 1.5), (0, nothing)))
        isempty(rational_m) || vlines!(ax, rational_m; color=(:black, 0.15), linewidth=1)
        for (i, n) in enumerate(ns)
            sel = sort([f for f in forcing_modes if f.n == n]; by=f -> f.m)
            m = [f.m for f in sel]
            a = [abs(f.amplitude) / factor for f in sel]
            stairs!(ax, vcat(m[1] - 1, m, m[end] + 1), vcat(0.0, a, 0.0); step=:center, color=WONG[i], linewidth=2, label="n = $n")
        end
        length(ns) > 1 && axislegend(ax; position=:rt)
        fig
    end
end

function CF._plot_surface_map(d; title, arb=false)
    return themed() do
        values, scale = _decade(d.field)
        fig = Figure(; size=(900, 500))
        ax = Axis(fig[1, 1]; title)
        hm = _surface_slice!(ax, d.phi, d.theta, values)
        d.coils === nothing || _coil_traces!(ax, d.coils)
        xlims!(ax, 0, 360)
        ylims!(ax, -180, 180)
        label = arb ? rich("b", subscript("n"), " (arb.)") : rich("b", subscript("n"), _unit(scale, "T"))
        Colorbar(fig[1, 2], hm; label)
        fig
    end
end
