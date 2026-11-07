# The two renderers every 2D field view draws with, and the coil traces laid over a control-surface map.

const DIVERGING = Reverse(:RdBu)

# A real field on the control surface: toroidal φ against poloidal θ in degrees, θ = 0 at the outboard midplane.
function _surface_slice!(ax, phi, theta, values)
    lim = max(maximum(abs, values), floatmin())
    ax.xticks = 0:90:360
    ax.yticks = -180:90:180
    ax.xlabel = "toroidal angle φ (deg)"
    ax.ylabel = "poloidal angle θ (deg)"
    return heatmap!(ax, phi, theta, permutedims(values); colormap=DIVERGING, colorrange=(-lim, lim))
end

# A real field on a poloidal plane, drawn on the flux-surface grid (R, Z) so the map follows the plasma shape.
function _poloidal_slice!(ax, R, Z, values; outlines=())
    lim = max(maximum(abs, values), floatmin())
    ax.aspect = DataAspect()
    ax.xlabel = "R (m)"
    ax.ylabel = "Z (m)"
    hm = surface!(ax, R, Z, zeros(size(R)); color=values, colormap=DIVERGING, colorrange=(-lim, lim), shading=NoShading)
    for o in outlines
        lines!(ax, o...; color=(:black, 0.35), linewidth=1)
    end
    return hm
end

# Coil conductors on a (φ, θ) map: a black outline with a core in the colour of the current's sign, broken where the
# trace wraps in φ or θ.
function _coil_traces!(ax, lines)
    cmap = Makie.to_colormap(DIVERGING)
    for (phi, theta, current) in lines
        breaks = findall(i -> abs(phi[i+1] - phi[i]) > 180 || abs(theta[i+1] - theta[i]) > 180, 1:(length(phi)-1))
        bounds = vcat(0, breaks, length(phi))
        for t in 1:(length(bounds)-1)
            r = (bounds[t]+1):bounds[t+1]
            lines!(ax, phi[r], theta[r]; color=:black, linewidth=2.5)
            lines!(ax, phi[r], theta[r]; color=current > 0 ? cmap[end] : cmap[1], linewidth=1.2)
        end
    end
    return nothing
end
