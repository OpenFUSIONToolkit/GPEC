"""
    ForceFreeStates

Post-processing and visualization functions for ForceFreeStates (DCON-style ideal MHD stability)
results stored in GPEC HDF5 output files.
"""
module ForceFreeStates

using HDF5
using Plots

"""
    plot_mode_displacement(h5path; modes=1:5, save_path=nothing)

Plot Im(ξ_ψ) vs ψ_N for the least stable eigenmode, showing one curve per requested
poloidal mode number m.

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file (e.g. `"gpec.h5"`)

### Keyword arguments

  - `modes`: Iterable of m values to plot (default: `1:5`)
  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_mode_displacement(h5path; modes=1:5, save_path=nothing)
    mlow, xi_psi, psi = h5open(h5path, "r") do fid
        read(fid["info/mlow"]), read(fid["integration/xi_psi"]), read(fid["integration/psi"])
    end

    mpert = size(xi_psi, 1)
    mhigh = mlow + mpert - 1

    p = plot(;
        xlabel="ψ_N",
        ylabel="Im(ξ_ψ)",
        title="Least Stable Eigenmode ξ_ψ"
    )
    for m in modes
        mlow <= m <= mhigh || continue
        plot!(p, psi, imag.(xi_psi[m-mlow+1, 1, :]); label="m=$m")
    end

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_eigenmode_summary(h5path; save_path=nothing)

Three-panel summary of the free-boundary energy matrix eigenmodes, analogous to the
DCON summary plot produced by OMFIT GPEC.

Panels:

  - Top: |eigenvector| vs index for the least stable mode
  - Bottom-left: heatmap of |W_t| (eigenvectors) vs mode index
  - Bottom-right: |eigenvalue| on log scale vs mode index

Eigenvectors are scaled by χ₁ = 2π ψ₀ × 10⁻³ to match GPEC conventions.

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file with vacuum data (`vac_flag = true`)

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_eigenmode_summary(h5path; save_path=nothing)
    wt, et, psio, mlow = h5open(h5path, "r") do fid
        read(fid["vacuum/wt"]), read(fid["vacuum/et"]),
        read(fid["equil/psio"]), read(fid["info/mlow"])
    end

    isempty(wt) && error("No vacuum data in $h5path; rerun with vac_flag = true")

    chi1 = 2π * psio
    wt = wt * (chi1 * 1e-3)

    nmn = size(wt, 1)
    nmodes = size(wt, 2)
    m_vals = (0:(nmn-1)) .+ mlow

    p1 = plot(
        m_vals, abs.(wt[:, 1]);
        xlabel="m",
        ylabel="|Eigenvector|",
        title="Mode 1, |λ₁| = $(round(abs(et[1]), digits=3))",
        legend=false
    )

    p2 = heatmap(
        m_vals, 1:nmodes, abs.(wt');
        xlabel="m",
        ylabel="mode index",
        colorbar_title="|Wₜ|"
    )

    p3 = scatter(
        abs.(et), 1:nmodes;
        xlabel="|Eigenvalue|",
        ylabel="mode index",
        xscale=:log10,
        legend=false
    )

    l = @layout [a{0.25h}; b c{0.25w}]
    p = plot(p1, p2, p3; layout=l, size=(900, 700))

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_stability_criterion(h5path; save_path=nothing)

Plot the stability criterion (smallest eigenvalue of W⁻¹, `crit`) vs ψ_N.
A sign change in `crit` during integration indicates an ideal fixed-boundary instability.

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_stability_criterion(h5path; save_path=nothing)
    psi, crit = h5open(h5path, "r") do fid
        read(fid["integration/psi"]), read(fid["integration/crit"])
    end

    p = plot(
        psi, crit;
        xlabel="ψ_N",
        ylabel="crit",
        title="Stability criterion (smallest eigenvalue of W⁻¹) vs ψ_N",
        legend=false
    )

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_edge_scan_comparison(h5paths, labels; q_range=nothing, et_range=nothing, save_path=nothing)

Overplot the edge dW stability scan (total energy vs q) from multiple GPEC runs.

Each run's `edge_scan/psi` is converted to q via the stored `integration/psi` + `integration/q`
arrays (linear interpolation — no runtime spline objects required). The truncation point `info/qlim`
is marked for each run with a vertical dashed line.

### Arguments

  - `h5paths`: Vector of paths to GPEC HDF5 output files
  - `labels`: Vector of legend labels, one per file

### Keyword arguments

  - `q_range`: Optional `(qmin, qmax)` tuple to restrict the x-axis
  - `et_range`: Optional `(etmin, etmax)` tuple for y-axis (default: `(-10, 10)`)
  - `save_path`: If provided, save the figure to this path

### Returns

A `Plots.jl` plot object.
"""
function plot_edge_scan_comparison(h5paths::Vector{<:AbstractString}, labels::Vector{<:AbstractString};
                                   q_range=nothing, et_range=nothing, save_path=nothing)
    colors = palette(:tab10)

    p = plot(;
        xlabel = "q (safety factor)",
        ylabel = "Total energy (et[1])",
        title  = "Edge stability scan comparison",
        legend = :topright,
    )
    hline!(p, [0.0]; color=:black, lw=1, ls=:dot, label="")

    for (i, (h5path, label)) in enumerate(zip(h5paths, labels))
        col = colors[mod1(i, length(colors))]

        psi_scan, et_scan, psi_int, q_int, qlim = h5open(h5path, "r") do f
            read(f["edge_scan/psi"]),
            real.(read(f["edge_scan/total_energy"])),
            read(f["integration/psi"]),
            read(f["integration/q"]),
            read(f["info/qlim"])
        end

        # Convert scan psi → q via linear interpolation on integration grid
        q_scan = map(psi_scan) do psi
            idx = searchsortedfirst(psi_int, psi)
            idx = clamp(idx, 2, length(psi_int))
            t = (psi - psi_int[idx-1]) / (psi_int[idx] - psi_int[idx-1])
            q_int[idx-1] * (1-t) + q_int[idx] * t
        end

        et_lim = isnothing(et_range) ? (-10.0, 10.0) : et_range

        # Filter NaN and out-of-range points so outliers don't compress the y-axis
        valid = .!isnan.(et_scan) .& (et_scan .>= et_lim[1]) .& (et_scan .<= et_lim[2])
        plot!(p, q_scan[valid], et_scan[valid]; label=label, color=col, lw=1.5)
        vline!(p, [qlim]; label="", color=col, lw=1, ls=:dash, alpha=0.6)
    end

    isnothing(q_range) || xlims!(p, q_range)
    ylims!(p, isnothing(et_range) ? (-10.0, 10.0) : et_range)
    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_mode_radial_profiles_comparison(h5paths, labels; modes=1:5, component=imag, save_path=nothing)

Overplot radial profiles of individual poloidal modes ξ_m(ψ) from multiple GPEC runs.
One panel per requested m value; all parameter values overplotted on each panel.

### Arguments

  - `h5paths`: Vector of paths to GPEC HDF5 output files
  - `labels`: Vector of legend labels, one per file

### Keyword arguments

  - `modes`: Iterable of m values to plot (default: `1:5`)
  - `component`: Function to extract the scalar from complex ξ (default: `imag`)
  - `save_path`: If provided, save the figure to this path

### Returns

A `Plots.jl` plot object.
"""
function plot_mode_radial_profiles_comparison(h5paths::Vector{<:AbstractString}, labels::Vector{<:AbstractString};
                                              modes=1:5, component::Function=imag, save_path=nothing)
    colors = palette(:tab10)
    modes_vec = collect(modes)
    npanels = length(modes_vec)

    panels = [plot(; xlabel="ψ_N", ylabel="$(component)(ξ_ψ)",
                   title="m = $(m)", legend=(j == 1 ? :topright : false))
              for (j, m) in enumerate(modes_vec)]

    for (i, (h5path, label)) in enumerate(zip(h5paths, labels))
        col = colors[mod1(i, length(colors))]

        mlow, xi_psi, psi = h5open(h5path, "r") do f
            read(f["info/mlow"]), read(f["integration/xi_psi"]), read(f["integration/psi"])
        end
        mpert = size(xi_psi, 1)
        mhigh = mlow + mpert - 1

        for (j, m) in enumerate(modes_vec)
            mlow <= m <= mhigh || continue
            profile = component.(xi_psi[m - mlow + 1, 1, :])
            plot!(panels[j], psi, profile; label=(j == 1 ? label : ""), color=col, lw=1.5)
        end
    end

    p = plot(panels...; layout=(npanels, 1), size=(900, 280 * npanels), bottom_margin=4Plots.mm)
    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_poloidal_spectrum_at_truncation(h5paths, labels; component=abs, normalize=true, save_path=nothing)

Overplot the poloidal mode spectrum ξ(m) at the edge truncation point from multiple GPEC runs.
The truncation point is the last stored integration step (`info/psilim`), whose ξ_ψ reflects
the eigenmode structure at the integration boundary.

### Arguments

  - `h5paths`: Vector of paths to GPEC HDF5 output files
  - `labels`: Vector of legend labels, one per file

### Keyword arguments

  - `component`: Function to extract the scalar from complex ξ (default: `abs`)
  - `normalize`: If `true`, divide each spectrum by its maximum before plotting (default: `true`)
  - `save_path`: If provided, save the figure to this path

### Returns

A `Plots.jl` plot object.
"""
function plot_poloidal_spectrum_at_truncation(h5paths::Vector{<:AbstractString}, labels::Vector{<:AbstractString};
                                              component::Function=abs, normalize::Bool=true, save_path=nothing)
    colors = palette(:tab10)

    ylabel_str = normalize ? "$(component)(ξ_ψ) / max" : "$(component)(ξ_ψ)"
    p = plot(;
        xlabel = "m (poloidal mode number)",
        ylabel = ylabel_str,
        title  = "Poloidal spectrum at truncation point",
        legend = :topright,
    )

    for (i, (h5path, label)) in enumerate(zip(h5paths, labels))
        col = colors[mod1(i, length(colors))]

        mlow, xi_psi, qlim = h5open(h5path, "r") do f
            read(f["info/mlow"]), read(f["integration/xi_psi"]), read(f["info/qlim"])
        end
        mpert = size(xi_psi, 1)
        m_axis = mlow:(mlow + mpert - 1)

        spectrum = component.(xi_psi[:, 1, end])
        if normalize
            maxval = maximum(abs.(spectrum))
            maxval > 0 && (spectrum ./= maxval)
        end

        plot!(p, collect(m_axis), spectrum;
              label="$label  (q_lim=$(round(qlim, digits=2)))",
              color=col, lw=1.5, markershape=:circle, markersize=3)
    end

    isnothing(save_path) || savefig(p, save_path)
    return p
end

end # module ForceFreeStates
