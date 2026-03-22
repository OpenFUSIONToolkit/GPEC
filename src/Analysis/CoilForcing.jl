"""
    CoilForcing

Visualization functions for coil geometry and the resulting normal field
perturbations on the plasma boundary.

All functions return `Plots.jl` plot objects and accept an optional `save_path`
keyword argument to write the figure to disk.
"""
module CoilForcing

using Plots

"""
    plot_coil_geometry_3d(coil_sets; save_path=nothing, kwargs...)

Plot the 3D wire geometry of all coil sets.

Each coil set is shown with a distinct color; individual conductors are plotted
as lines in Cartesian (X, Y, Z) space in meters.

### Arguments
- `coil_sets`: `Vector{CoilSet}` from `ForcingTerms.load_coil_sets`

### Keyword arguments
- `save_path`: file path to save the figure (default: `nothing`)
- Any extra kwargs are forwarded to `Plots.plot`

### Returns
A `Plots.jl` plot object.
"""
function plot_coil_geometry_3d(coil_sets; save_path=nothing, kwargs...)
    # Collect all currents to set a symmetric colorscale centred at zero
    all_currents = [I for cs in coil_sets for I in cs.currents]
    clim = isempty(all_currents) ? 1.0 : max(maximum(abs, all_currents), eps())

    p = plot(; xlabel="X [m]", ylabel="Y [m]", zlabel="Z [m]",
               title="Coil geometry (3D)",
               colorbar_title=" Current [A]",
               clims=(-clim, clim),
               color=:RdBu,
               camera=(30, 20), kwargs...)

    for cs in coil_sets
        for j in 1:cs.ncoil
            I = cs.currents[j]
            # Map current to [0,1] on a symmetric scale centred at zero
            t = clamp((I + clim) / (2clim), 0.0, 1.0)
            col = get(cgrad(:RdBu), t)
            for k in 1:cs.s
                xs = view(cs.x, j, k, :)
                ys = view(cs.y, j, k, :)
                zs = view(cs.z, j, k, :)
                plot!(p, xs, ys, zs; label="", color=col, lw=1.5)
            end
        end
    end

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_coil_geometry_rz(coil_sets; equil=nothing, save_path=nothing, kwargs...)

Plot coil cross-sections in the (R, Z) meridional plane.

Each coil strand is shown as a closed curve in `(R, Z) = (√(X²+Y²), Z)` space.
If `equil` is provided, the plasma boundary is overplotted for reference.

### Arguments
- `coil_sets`: `Vector{CoilSet}` from `ForcingTerms.load_coil_sets`

### Keyword arguments
- `equil`: optional `PlasmaEquilibrium`; if given, overlays the plasma boundary
- `save_path`: file path to save the figure (default: `nothing`)
- Any extra kwargs are forwarded to `Plots.plot`

### Returns
A `Plots.jl` plot object.
"""
function plot_coil_geometry_rz(coil_sets; equil=nothing, save_path=nothing, kwargs...)
    p = plot(; xlabel="R [m]", ylabel="Z [m]",
               title="Coil cross-sections (R, Z)", legend=:topright,
               aspect_ratio=:equal, kwargs...)

    colors = [:blue, :red, :green, :orange, :purple, :brown, :cyan, :magenta]
    for (ci, cs) in enumerate(coil_sets)
        col = colors[mod1(ci, length(colors))]
        for j in 1:cs.ncoil
            for k in 1:cs.s
                R_strand = sqrt.(view(cs.x, j, k, :).^2 .+ view(cs.y, j, k, :).^2)
                Z_strand = view(cs.z, j, k, :)
                label = (j == 1 && k == 1) ? cs.name : ""
                scatter!(p, R_strand, Z_strand; label=label, color=col,
                         markersize=2, markerstrokewidth=0)
            end
        end
    end

    if !isnothing(equil)
        # Overlay plasma boundary from equilibrium rzphi splines at ψ=1
        mtheta_boundary = length(equil.rzphi_ys)
        hint2d = (Ref(1), Ref(1))
        R_bnd = zeros(mtheta_boundary)
        Z_bnd = zeros(mtheta_boundary)
        for (i, θ_sfl) in enumerate(equil.rzphi_ys)
            r_minor = sqrt(equil.rzphi_rsquared((1.0, θ_sfl); hint=hint2d))
            θ_cyl   = 2π * (θ_sfl + equil.rzphi_offset((1.0, θ_sfl); hint=hint2d))
            R_bnd[i] = equil.ro + r_minor * cos(θ_cyl)
            Z_bnd[i] = equil.zo + r_minor * sin(θ_cyl)
        end
        plot!(p, [R_bnd; R_bnd[1]], [Z_bnd; Z_bnd[1]];
              label="plasma boundary", color=:black, lw=2, linestyle=:dash)
    end

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_bn_contour(bn, mtheta, nzeta; n=nothing, save_path=nothing, kwargs...)

Plot the normal magnetic field `bn(θ, ζ)` on the plasma boundary as a 2D heatmap.

### Arguments
- `bn`: normal field matrix `[mtheta, nzeta]` in Tesla
- `mtheta`: number of poloidal grid points
- `nzeta`: number of toroidal grid points

### Keyword arguments
- `n`: toroidal mode number (used for axis label only)
- `save_path`: file path to save the figure (default: `nothing`)
- Any extra kwargs are forwarded to `Plots.heatmap`

### Returns
A `Plots.jl` plot object.
"""
function plot_bn_contour(bn::Matrix{Float64}, mtheta::Int, nzeta::Int;
                         n=nothing, save_path=nothing, kwargs...)
    # Shift rows so θ=0 (outboard midplane) is centred on the y-axis at 0°
    bn_wrapped = circshift(bn, (mtheta ÷ 2, 0))
    theta_deg = range(-180; step=360.0/mtheta, length=mtheta)
    zeta_deg  = range(0;    step=360.0/nzeta,  length=nzeta)

    n_label = isnothing(n) ? "" : " (n=$n)"
    p = heatmap(zeta_deg, theta_deg, bn_wrapped;
                xlabel="ζ [°]", ylabel="θ [°]",
                ylims=(-180, 180),
                title="Normal field bₙ(θ, ζ)$n_label [T]",
                color=:RdBu, kwargs...)

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_mode_spectrum(forcing_modes; mlow=nothing, mhigh=nothing, save_path=nothing, kwargs...)

Plot the Fourier mode spectrum `|bmn|` vs poloidal mode number m as a step line, grouped by n.

### Arguments
- `forcing_modes`: `Vector{ForcingMode}` from `compute_coil_forcing_modes!`

### Keyword arguments
- `mlow`, `mhigh`: poloidal mode range for the x-axis (e.g. from ForceFreeStates config);
  if provided, sets xlims to span the full stability range even if some modes are zero
- `save_path`: file path to save the figure (default: `nothing`)
- Any extra kwargs are forwarded to `Plots.plot`

### Returns
A `Plots.jl` plot object.
"""
function plot_mode_spectrum(forcing_modes; mlow=nothing, mhigh=nothing, save_path=nothing, kwargs...)
    n_vals = sort(unique(m.n for m in forcing_modes))
    colors = [:blue, :red, :green, :orange, :purple, :brown]

    p = plot(; xlabel="Poloidal mode m", ylabel="|bmn| [T]",
               title="Coil forcing mode spectrum",
               legend=:topright, kwargs...)

    for (ni, n) in enumerate(n_vals)
        modes_n = filter(m -> m.n == n, forcing_modes)
        sort!(modes_n; by=m -> m.m)
        m_vals = [m.m for m in modes_n]
        amps   = abs.([m.amplitude for m in modes_n])
        col = colors[mod1(ni, length(colors))]
        m_ext   = [m_vals[1] - 1; m_vals; m_vals[end] + 1]
        amp_ext = [0.0; amps; 0.0]
        plot!(p, m_ext, amp_ext; label="n=$n", color=col, lw=2, seriestype=:steppre)
    end

    if !isnothing(mlow) && !isnothing(mhigh)
        xlims!(p, mlow - 1, mhigh + 1)
    end

    isnothing(save_path) || savefig(p, save_path)
    return p
end

export plot_coil_geometry_3d, plot_coil_geometry_rz
export plot_bn_contour, plot_mode_spectrum

end # module CoilForcing
