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
        plot!(p, psi, imag.(xi_psi[m-mlow+1, 1, :]); label="m=$m")  # DCON phase convention: ξ_ψ is purely imaginary on the real axis
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
    plot_energy_eigenvectors(h5path; matrix_type=:total, save_path=nothing)

Heatmap of energy eigenvector magnitudes vs (m, mode index).

Only `matrix_type=:total` is supported (the total energy eigenvector matrix `Wₜ` is stored in
`vacuum/wt`). Plasma and vacuum eigenvectors are not stored separately in the HDF5 output.

Eigenvectors are scaled by χ₁ = 2π ψ₀ × 10⁻³ to match GPEC conventions.

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file with vacuum data (`vac_flag = true`)

### Keyword arguments

  - `matrix_type`: Energy matrix to plot; only `:total` is currently supported
  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_energy_eigenvectors(h5path; matrix_type=:total, save_path=nothing)
    matrix_type == :total ||
        error("matrix_type=$matrix_type not supported; only :total has eigenvector matrix stored in HDF5 (ep/ev are eigenvalue vectors, not matrices)")

    wt, psio, mlow = h5open(h5path, "r") do fid
        read(fid["vacuum/wt"]), read(fid["equil/psio"]), read(fid["info/mlow"])
    end

    isempty(wt) && error("No vacuum data in $h5path; rerun with vac_flag = true")

    chi1 = 2π * psio
    wt_scaled = wt * (chi1 * 1e-3)

    nmn = size(wt_scaled, 1)
    nmodes = size(wt_scaled, 2)
    m_vals = (0:(nmn-1)) .+ mlow

    p = heatmap(
        m_vals, 1:nmodes, abs.(wt_scaled');
        xlabel="m",
        ylabel="mode index",
        title="Energy eigenvectors |Wₜ| (total)",
        colorbar_title="|Wₜ|"
    )

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_eigenvalue_spectrum(h5path; matrix_type=:total, save_path=nothing)

Scatter plot of energy eigenvalues vs mode index. Points are colored red (unstable, Re > 0)
or green (stable, Re < 0), with a dashed reference line at zero.

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file with vacuum data (`vac_flag = true`)

### Keyword arguments

  - `matrix_type`: Which eigenvalues to plot: `:total` (`et`), `:plasma` (`ep`), or `:vacuum` (`ev`)
  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_eigenvalue_spectrum(h5path; matrix_type=:total, save_path=nothing)
    dataset = Dict(:total => "vacuum/et", :plasma => "vacuum/ep", :vacuum => "vacuum/ev")
    haskey(dataset, matrix_type) || error("matrix_type must be :total, :plasma, or :vacuum")

    et = h5open(h5path, "r") do fid
        read(fid[dataset[matrix_type]])
    end

    isempty(et) && error("No vacuum data in $h5path; rerun with vac_flag = true")

    nmodes = length(et)
    ev_real = real.(et)
    colors = [v > 0 ? :red : :green for v in ev_real]

    p = scatter(
        ev_real, 1:nmodes;
        xlabel="Re(eigenvalue)",
        ylabel="mode index",
        title="Eigenvalue spectrum ($matrix_type)",
        legend=false,
        color=colors,
        markerstrokewidth=0
    )
    vline!(p, [0]; linestyle=:dash, color=:black, label=nothing)

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_delta_prime(h5path; save_path=nothing)

Bar chart of `|Δ'|` per singular surface, computed from the stored asymptotic coefficients
`ca_left` and `ca_right`. Bars are colored red (tearing unstable, Re(Δ') > 0) or blue
(tearing stable).

Δ' is computed as `(ca_right[resnum,resnum,2,s] - ca_left[resnum,resnum,2,s]) / (4π² ψ₀)`,
where `resnum` is the linear mode index of the (m,n) resonant pair at surface `s`.

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_delta_prime(h5path; save_path=nothing)
    msing, q_sing, ca_l, ca_r, psio, mn_index = h5open(h5path, "r") do fid
        read(fid["singular/msing"]), read(fid["singular/q"]),
        read(fid["singular/ca_left"]), read(fid["singular/ca_right"]),
        read(fid["equil/psio"]), read(fid["info/mn_index"])
    end

    msing == 0 && return plot(; title="No singular surfaces found", legend=false)

    numpert_total = size(ca_l, 1)
    chi1 = 2π * psio

    dp_vals = ComplexF64[]
    labels = String[]

    for s in 1:msing
        q_s = q_sing[s]
        # Find the resonant mode index: (m, n) in mn_index with m/n ≈ q_s
        resnum = findfirst(1:numpert_total) do j
            n_j = mn_index[j, 2]
            n_j != 0 && abs(mn_index[j, 1] / n_j - q_s) < 1e-6
        end
        if isnothing(resnum)
            push!(dp_vals, 0.0 + 0.0im)
        else
            dp = (ca_r[resnum, resnum, 2, s] - ca_l[resnum, resnum, 2, s]) / (2π * chi1)
            push!(dp_vals, dp)
        end
        push!(labels, "q=$(round(q_s, digits=3))")
    end

    colors = [real(v) > 0 ? :red : :steelblue for v in dp_vals]

    p = bar(
        1:msing, abs.(dp_vals);
        xticks=(1:msing, labels),
        xlabel="rational surface",
        ylabel="|Δ'|",
        title="Tearing stability Δ' (FFS asymptotic coefficients)",
        legend=false,
        color=colors,
        xrotation=30
    )

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_dcon_summary(h5path; save_path=nothing)

Four-panel summary of ForceFreeStates (DCON-style) stability results, combining:

  - Energy eigenvector heatmap (`plot_energy_eigenvectors`)
  - Stability criterion vs ψ_N (`plot_stability_criterion`)
  - Eigenvalue spectrum (`plot_eigenvalue_spectrum`)
  - Tearing stability Δ' at each rational surface (`plot_delta_prime`)

If no vacuum data is present (`vac_flag = false`), only the stability criterion and Δ'
panels are shown.

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_dcon_summary(h5path; save_path=nothing)
    has_vac = h5open(h5path, "r") do fid
        haskey(fid, "vacuum/wt") && !isempty(read(fid["vacuum/wt"]))
    end

    p_crit = plot_stability_criterion(h5path)
    p_dp = plot_delta_prime(h5path)

    if has_vac
        p_evec = plot_energy_eigenvectors(h5path; matrix_type=:total)
        p_eval = plot_eigenvalue_spectrum(h5path; matrix_type=:total)
        p = plot(p_evec, p_crit, p_eval, p_dp; layout=(2, 2), size=(1000, 800))
    else
        title!(p_crit, "Stability criterion (no vacuum data — rerun with vac_flag = true)")
        p = plot(p_crit, p_dp; layout=(1, 2), size=(1000, 400))
    end

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_singular_surfaces(h5path; save_path=nothing)

Two-panel summary of singular (rational) surface locations and tearing stability:

  - Left: q(ψ) profile with vertical markers at each rational surface
  - Right: `|Δ'|` bar chart at each rational surface (`plot_delta_prime`)

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_singular_surfaces(h5path; save_path=nothing)
    xs, q_profile, q0, q95, msing, psi_sing, q_sing = h5open(h5path, "r") do fid
        read(fid["splines/profiles/xs"]), read(fid["splines/profiles/q"]),
        read(fid["equil/q0"]), read(fid["equil/q95"]),
        read(fid["singular/msing"]), read(fid["singular/psi"]), read(fid["singular/q"])
    end

    p_q = plot(
        xs, q_profile;
        xlabel="ψ_N",
        ylabel="q",
        title="Safety factor and rational surfaces",
        legend=false
    )
    hline!(p_q, [q0, q95]; linestyle=:dot, color=:gray, label=nothing)
    for s in 1:msing
        vline!(p_q, [psi_sing[s]]; linestyle=:dash, color=:red, label=nothing)
        annotate!(p_q, psi_sing[s], q_sing[s],
            text("  q=$(round(q_sing[s], digits=2))", 7, :left, :red))
    end

    p_dp = plot_delta_prime(h5path)

    p = plot(p_q, p_dp; layout=(1, 2), size=(1000, 400))

    isnothing(save_path) || savefig(p, save_path)
    return p
end

end # module ForceFreeStates
