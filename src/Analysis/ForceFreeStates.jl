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
    plot_edge_stability_scan(h5path; save_path=nothing)

Plot the edge stability scan energy components (et, ep, ev, evonly) vs ψ_N.

The edge scan evaluates `δW_total = δW_plasma + δW_vacuum` at each stored integration step
in the region [psiedge, psilim], with the plasma boundary swept from psiedge to psilim.
A positive et indicates stability; the truncation point is chosen at the peak et.

Four subplots are shown:
  - **Total energy** `et = ep + ev`: total free-boundary energy eigenvalue
  - **Plasma energy** `ep`: plasma contribution to δW
  - **Vacuum energy** `ev`: vacuum (wv) contribution with singfac scaling
  - **Vacuum-only eigenvalue** `evonly`: smallest eigenvalue of wv alone (no plasma response)

A horizontal dashed line at zero marks the stability boundary. A vertical dashed line marks
`psilim` (the final truncation psi, where the peak et was found).

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file produced with `psiedge < psilim`

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object, or `nothing` if no edge scan data is present in the file.
"""
function plot_edge_stability_scan(h5path; save_path=nothing)
    has_scan, psi, et, ep, ev, evonly, psilim = h5open(h5path, "r") do fid
        if !haskey(fid, "integration/edge_scan_psi")
            return false, Float64[], ComplexF64[], ComplexF64[], ComplexF64[], Float64[], NaN
        end
        true,
        read(fid["integration/edge_scan_psi"]),
        read(fid["integration/edge_scan_et"]),
        read(fid["integration/edge_scan_ep"]),
        read(fid["integration/edge_scan_ev"]),
        read(fid["integration/edge_scan_evonly"]),
        read(fid["info/psilim"])
    end

    if !has_scan
        @warn "No edge scan data in $h5path. Run with psiedge < psilim to generate it."
        return nothing
    end

    kw = (legend=false, xlabel="ψ_N")

    p_et = plot(psi, real.(et); ylabel="et = ep + ev", title="Total energy"; kw...)
    hline!(p_et, [0.0]; color=:black, lw=1, ls=:dash)
    vline!(p_et, [psilim]; color=:gray, lw=1, ls=:dash)

    p_ep = plot(psi, real.(ep); ylabel="ep (plasma)"; kw...)
    hline!(p_ep, [0.0]; color=:black, lw=1, ls=:dash)
    vline!(p_ep, [psilim]; color=:gray, lw=1, ls=:dash)

    p_ev = plot(psi, real.(ev); ylabel="ev (vacuum)"; kw...)
    hline!(p_ev, [0.0]; color=:black, lw=1, ls=:dash)
    vline!(p_ev, [psilim]; color=:gray, lw=1, ls=:dash)

    p_evonly = plot(psi, evonly; ylabel="evonly (wv alone)"; kw...)
    hline!(p_evonly, [0.0]; color=:black, lw=1, ls=:dash)
    vline!(p_evonly, [psilim]; color=:gray, lw=1, ls=:dash)

    p = plot(p_et, p_ep, p_ev, p_evonly;
             layout=(4, 1),
             size=(900, 900),
             plot_title="Edge stability scan: $h5path",
             bottom_margin=4Plots.mm)

    isnothing(save_path) || savefig(p, save_path)
    return p
end

end # module ForceFreeStates
