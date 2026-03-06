"""
    ForceFreeStates

Post-processing and visualization functions for ForceFreeStates (DCON-style ideal MHD stability)
results stored in JPEC HDF5 output files.
"""
module ForceFreeStates

using HDF5
using Plots

"""
    plot_mode_displacement(h5path; modes=1:5, save_path=nothing)

Plot Im(ξ_ψ) vs ψ_N for the least stable eigenmode, showing one curve per requested
poloidal mode number m.

### Arguments

  - `h5path`: Path to a JPEC HDF5 output file (e.g. `"jpec.h5"`)

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

  - `h5path`: Path to a JPEC HDF5 output file with vacuum data (`vac_flag = true`)

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

  - `h5path`: Path to a JPEC HDF5 output file

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

end # module ForceFreeStates
