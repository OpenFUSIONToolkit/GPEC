"""
    ForceFreeStates

Post-processing and visualization functions for ForceFreeStates (DCON-style ideal MHD stability)
results stored in GPEC HDF5 output files.
"""
module ForceFreeStates

using HDF5
using LaTeXStrings
using Plots
using Printf
using Statistics: quantile

"""
    plot_mode_displacement(h5path; modes=1:5, save_path=nothing)

Plot |ξ_ψ| vs ψ_N for the least stable eigenmode, showing one curve per requested
poloidal mode number m. The title includes the first eigenvalue dW = et[1].

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file (e.g. `"gpec.h5"`)

### Keyword arguments

  - `modes`: Iterable of m values to plot (default: `1:5`)
  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_mode_displacement(h5path; modes=1:5, save_path=nothing)
    mlow, xi_psi, psi, et = h5open(h5path, "r") do fid
        read(fid["Info/mlow"]), read(fid["ForceFreeStates/Solutions/ForwardIntegration/xi_psi"]),
        read(fid["ForceFreeStates/Solutions/ForwardIntegration/psi"]), read(fid["ForceFreeStates/FreeBoundaryStability/eigenmode_energies"])
    end

    mpert = size(xi_psi, 1)
    mhigh = mlow + mpert - 1
    dW = isempty(et) ? nothing : et[1]
    title_str = isnothing(dW) ? "Least stable mode" :
                "Least stable mode, δW = $(round(real(dW), sigdigits=4))"

    p = plot(;
        xlims=(0, 1),
        xlabel="Norm. Poloidal Flux",
        ylabel="|ξ^ψ|",
        title=title_str,
        left_margin=10Plots.mm,
        bottom_margin=5Plots.mm
    )
    for m in modes
        mlow <= m <= mhigh || continue
        plot!(p, psi, abs.(xi_psi[m-mlow+1, 1, :]); label="m=$m")
    end

    isnothing(save_path) || savefig(p, save_path)
    return p
end


"""
    plot_fixed_boundary_stability_criterion(h5path; save_path=nothing)

Plot the stability criterion (smallest eigenvalue of W⁻¹, `crit`) vs ψ_N.
A sign change in `crit` during integration indicates an ideal fixed-boundary instability.

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_fixed_boundary_stability_criterion(h5path; save_path=nothing)
    psi, crit = h5open(h5path, "r") do fid
        read(fid["ForceFreeStates/Solutions/ForwardIntegration/psi"]), read(fid["ForceFreeStates/Solutions/ForwardIntegration/crit"])
    end

    p = plot(
        psi, crit;
        xlims=(0, 1),
        xlabel="Norm. Poloidal Flux",
        ylabel="|Dᶜ|",
        title="Fixed-Boundary Stability",
        legend=false,
        left_margin=10Plots.mm,
        bottom_margin=5Plots.mm
    )
    hline!(p, [0.0]; linestyle=:dash, color=:black, label=nothing)

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_energy_eigenvectors(h5path; matrix_type=:total, save_path=nothing)

Heatmap of energy eigenvector magnitudes vs (m, mode index).

Only `matrix_type=:total` is supported (the total energy eigenvector matrix `Wₜ` is stored in
`FreeBoundaryStability/W_freeboundary_eigenmodes`). Plasma and vacuum eigenvectors are not
stored separately in the HDF5 output.

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
        read(fid["ForceFreeStates/FreeBoundaryStability/W_freeboundary_eigenmodes"]), read(fid["Equilibrium/psi_total"]), read(fid["Info/mlow"])
    end

    isempty(wt) && error("No vacuum data in $h5path; rerun with vac_flag = true")

    chi1 = 2π * psio
    wt_scaled = wt * (chi1 * 1e-3)

    nmn = size(wt_scaled, 1)
    nmodes = size(wt_scaled, 2)
    m_vals = (0:(nmn-1)) .+ mlow

    p = heatmap(
        m_vals, 1:nmodes, abs.(wt_scaled');
        xlabel="Poloidal Harmonic",
        ylabel="Eigenmode Index",
        title="Total Energy Eigenvectors",
        colorbar_title="Harmonic Amplitude",
        left_margin=10Plots.mm,
        right_margin=20Plots.mm,
        bottom_margin=5Plots.mm
    )

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_edge_stability_scan(h5path; save_path=nothing, ylims=(-2, 3), kwargs...)

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
  - `ylims`: y-axis limits applied to all panels (default: `(-2, 3)`)
  - `kwargs...`: Additional Plots.jl keyword arguments applied to all line plots (e.g. `lw=2`)

### Returns

A `Plots.jl` plot object, or `nothing` if no `EdgeScan/` group is present in the file.
"""
function plot_edge_stability_scan(h5path; save_path=nothing, ylims=(-2, 3), kwargs...)
    has_scan, q, et, ep, ev, evonly, qlim = h5open(h5path, "r") do fid
        if !haskey(fid, "ForceFreeStates/EdgeScan/psi")
            return false, Float64[], ComplexF64[], ComplexF64[], ComplexF64[], Float64[], NaN
        end
        true,
        read(fid["ForceFreeStates/EdgeScan/q"]),
        read(fid["ForceFreeStates/EdgeScan/total_energy"]),
        read(fid["ForceFreeStates/EdgeScan/plasma_energy"]),
        read(fid["ForceFreeStates/EdgeScan/vacuum_energy"]),
        read(fid["ForceFreeStates/EdgeScan/vacuum_eigenvalue"]),
        read(fid["Info/qlim"])
    end

    if !has_scan
        @warn "No edge_scan group in $h5path. Run with psiedge < psilim to generate it."
        return nothing
    end

    kw_re = (xlabel="q", label="Re", ylims=ylims, kwargs...)
    kw_im = (xlabel="q", label="Im", ls=:dash, ylims=ylims, kwargs...)
    vl_kw = (color=:gray, lw=1, ls=:dot, label=false)
    hl_kw = (color=:black, lw=1, ls=:dash, label=false)

    p_et = plot(q, real.(et); ylabel="Total Energy", title="Edge Stability Scan: δW vs q", kw_re...)
    plot!(p_et, q, imag.(et); kw_im...)
    hline!(p_et, [0.0]; hl_kw...)
    vline!(p_et, [qlim]; vl_kw...)

    p_ep = plot(q, real.(ep); ylabel="Plasma Energy", kw_re...)
    plot!(p_ep, q, imag.(ep); kw_im...)
    hline!(p_ep, [0.0]; hl_kw...)
    vline!(p_ep, [qlim]; vl_kw...)

    p_ev = plot(q, real.(ev); ylabel="Vacuum Energy", kw_re...)
    plot!(p_ev, q, imag.(ev); kw_im...)
    hline!(p_ev, [0.0]; hl_kw...)
    vline!(p_ev, [qlim]; vl_kw...)

    p_evonly = plot(q, evonly; ylabel="Min Vac. Eigenvalue", legend=false, xlabel="q", ylims=ylims, kwargs...)
    hline!(p_evonly, [0.0]; hl_kw...)
    vline!(p_evonly, [qlim]; vl_kw...)

    p = plot(p_et, p_ep, p_ev, p_evonly;
        layout=(4, 1),
        size=(900, 1100),
        plot_title="Edge stability scan: $h5path",
        left_margin=12Plots.mm,
        bottom_margin=4Plots.mm)

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_eigenvalues(h5path; matrix_type=:total, save_path=nothing)

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
function plot_eigenvalues(h5path; matrix_type=:total, save_path=nothing)
    dataset = Dict(
        :total => "ForceFreeStates/FreeBoundaryStability/eigenmode_energies",
        :plasma => "ForceFreeStates/FreeBoundaryStability/eigenmode_plasma_energies",
        :vacuum => "ForceFreeStates/FreeBoundaryStability/eigenmode_vacuum_energies"
    )
    haskey(dataset, matrix_type) || error("matrix_type must be :total, :plasma, or :vacuum")

    et = h5open(h5path, "r") do fid
        read(fid[dataset[matrix_type]])
    end

    isempty(et) && error("No vacuum data in $h5path; rerun with vac_flag = true")

    nmodes = length(et)
    ev_real = real.(et)
    colors = [v < 0 ? :red : :blue for v in ev_real]  # red = negative (unstable), blue = positive (stable)

    p = scatter(
        1:nmodes, ev_real;
        xlabel="mode index",
        ylabel="Re(eigenvalue)",
        title="Eigenvalue spectrum ($matrix_type)",
        legend=false,
        color=colors,
        markerstrokewidth=0,
        left_margin=10Plots.mm,
        bottom_margin=5Plots.mm
    )
    hline!(p, [0]; linestyle=:dash, color=:black, label=nothing)

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_delta_prime(h5path; save_path=nothing)

Scatter plot of `Re(Δ')` per singular surface vs ψ_N, computed from the stored asymptotic
coefficients `ca_left` and `ca_right`. Points are colored red (tearing unstable, Re(Δ') > 0)
or blue (tearing stable). Integer-valued q rational surfaces are annotated.

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
    msing, psi_sing, q_sing, ca_l, ca_r, psio, mn_index = h5open(h5path, "r") do fid
        read(fid["SingularSurfaces/rational_count"]), read(fid["SingularSurfaces/rational_psi"]), read(fid["SingularSurfaces/rational_q"]),
        read(fid["SingularSurfaces/ca_left"]), read(fid["SingularSurfaces/ca_right"]),
        read(fid["Equilibrium/psi_total"]), read(fid["Info/mn_index"])
    end

    msing == 0 && return plot(; title="No singular surfaces found", legend=false)
    # ca_left/ca_right are zero-extent sentinels on kinetic/galerkin-matched runs (never computed there).
    isempty(ca_l) && return plot(; title="No asymptotic coefficients — ca_left/ca_right not computed for this run", legend=false)

    numpert_total = size(ca_l, 1)
    chi1 = 2π * psio

    dp_vals = ComplexF64[]
    for s in 1:msing
        q_s = q_sing[s]
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
    end

    dp_real = real.(dp_vals)
    colors = [v > 0 ? :red : :steelblue for v in dp_real]

    p = scatter(
        psi_sing, dp_real;
        xlims=(0, 1),
        xlabel="Norm. Poloidal Flux",
        ylabel="Re(Δ')",
        title="Tearing stability Δ'",
        legend=false,
        color=colors,
        markersize=7,
        markerstrokewidth=0,
        left_margin=10Plots.mm,
        bottom_margin=5Plots.mm
    )
    hline!(p, [0.0]; linestyle=:dash, color=:black, label=nothing)

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_ballooning_alpha_boundary(h5path; save_path=nothing, psi_min=0.0)

Plot the BALOO-style infinite-n ballooning stability diagram: the experimental
pressure gradient α (solid) and the first stability boundary α_crit (dashed) versus
normalized poloidal flux ψ_N. Surfaces where the experimental α lies above the boundary
are ballooning-unstable. Reads `LocalStability/ballooning_psi`, `LocalStability/alpha`, and
`LocalStability/alpha_critical` (populated when ForceFreeStates runs with
`local_stability_flag = true`).

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file (e.g. `"gpec.h5"`)

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)
  - `psi_min`: Lower ψ_N axis limit; set near the edge (e.g. `0.9`) to focus on the
    pedestal (default: `0.0`)

### Returns

A `Plots.jl` plot object.
"""
function plot_ballooning_alpha_boundary(h5path; save_path=nothing, psi_min=0.0)
    psi, alpha, alpha_crit = h5open(h5path, "r") do fid
        haskey(fid, "LocalStability/alpha") || return (Float64[], Float64[], Float64[])
        read(fid["LocalStability/ballooning_psi"]), read(fid["LocalStability/alpha"]), read(fid["LocalStability/alpha_critical"])
    end

    isempty(alpha) && return plot(; title="No local stability data (set local_stability_flag)", legend=false)

    p = plot(;
        xlims=(psi_min, 1),
        xlabel=L"\psi_N",
        ylabel=L"\alpha",
        title="Infinite-n ballooning stability",
        framestyle=:box,
        legend=:topleft,
        left_margin=10Plots.mm,
        bottom_margin=5Plots.mm
    )

    # NaN entries leave natural gaps over always-stable surfaces; isfinite masking would
    # wrongly connect across such gaps with a straight line.
    plot!(p, psi, alpha; lw=2, color=:black, label="Experimental gradient")
    plot!(p, psi, alpha_crit; lw=2, linestyle=:dash, color=:red, label="1st stability boundary")

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_cond_fbar(h5path; save_path=nothing, zoom=false)

Plot `cond(F̄)` vs ψ from the kinetic-singular-surface scan stored in
`SingularSurfaces/Kinetic/` (populated when ForceFreeStates runs with
`kinetic_factor > 0`, `singfac_min > 0`).

`F̄` is the kinetic Euler-Lagrange matrix formed by Schur-reducing the six
kinetic matrices against the ideal A/B/C/D/E/H blocks (Logan 2015 Appendix
C). Peaks in `cond(F̄)` locate "kinetically-displaced" singular surfaces —
roots of `det(F̄)` that are not at ideal rational surfaces. When a peak
exceeds the threshold stored in `scan_threshold` the ODE integrator stops
there and steps across trapezoidally, mirroring Fortran `ode_kin_cross`.

The plot overlays the ideal rational surfaces (dotted grey, labelled with
their q value) and any accepted kinetic singular surfaces (solid crimson).
If no peak exceeds the threshold, `kmsing = 0` and the kinetic ODE runs as
a single chunk. This diagnostic is useful for anyone asking *where* the
kinetic resonances land relative to the ideal ones.

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file produced with kinetic mode enabled

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)
  - `zoom`: If `true`, auto-scale the y-axis to the scan data (threshold shown
    as annotation only); if `false` (default), always include the threshold
    line in the y-range

### Returns

A `Plots.jl` plot object, or `nothing` if no kinetic scan is stored in the file.
"""
function plot_cond_fbar(h5path; save_path=nothing, zoom=false)
    scan_psi, scan_cond, thr, k_psi, i_psi, i_q, kmsing = h5open(h5path, "r") do fid
        if !(haskey(fid, "SingularSurfaces") && haskey(fid["SingularSurfaces"], "Kinetic"))
            return Float64[], Float64[], 0.0, Float64[], Float64[], Float64[], 0
        end
        kg = fid["SingularSurfaces/Kinetic"]
        (read(kg["scan_psi"]),
            read(kg["scan_cond"]),
            read(kg["scan_threshold"]),
            read(kg["psi"]),
            read(fid["SingularSurfaces/rational_psi"]),
            read(fid["SingularSurfaces/rational_q"]),
            read(kg["rational_count"]))
    end

    if isempty(scan_psi)
        @warn "No kinetic-singular-surface scan in $h5path — rerun with kinetic_factor>0, singfac_min>0"
        return nothing
    end

    finite_cond = filter(isfinite, scan_cond)
    data_max = isempty(finite_cond) ? 1.0 : maximum(finite_cond)
    data_min = isempty(finite_cond) ? 1.0 : max(minimum(finite_cond), 1e-3)

    p = plot(scan_psi, scan_cond;
        yscale=:log10,
        lw=2,
        color=:steelblue,
        label="cond(F̄)",
        xlabel="Norm. Poloidal Flux",
        ylabel="cond(F̄)",
        title="Kinetic F̄ condition number (kmsing = $kmsing)",
        legend=:topleft,
        left_margin=12Plots.mm,
        bottom_margin=4Plots.mm,
        size=(900, 500)
    )

    if zoom
        # Auto-scale y to data and annotate the threshold at the top edge if off-scale.
        ylims!(p, (data_min / 2, data_max * 3))
        if thr > 0 && thr > data_max * 3
            annotate!(p, [(scan_psi[end], data_max * 2.5,
                text(@sprintf("threshold = %.0e (off-scale)", thr), :right, 9, :red))])
        elseif thr > 0
            hline!(p, [thr]; color=:red, linestyle=:dash, lw=1.5,
                label=@sprintf("threshold = %.0e", thr))
        end
    elseif thr > 0
        hline!(p, [thr]; color=:red, linestyle=:dash, lw=1.5,
            label=@sprintf("threshold = %.0e", thr))
    end

    if !isempty(i_psi)
        vline!(p, i_psi; color=:gray, linestyle=:dot, lw=1,
            label="ideal rational (q = m/n)")
        y_label = zoom ? data_max * 1.2 : max(data_max * 1.2, sqrt(data_max * (thr > 0 ? thr : data_max)))
        for (idx, ps) in enumerate(i_psi)
            annotate!(p, [(ps, y_label,
                text(@sprintf("q=%.0f", i_q[idx]), :right, 8, :gray))])
        end
    end

    if !isempty(k_psi)
        vline!(p, k_psi; color=:crimson, lw=1.5, label="accepted kinsing")
    end

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_ffs_summary(h5path; save_path=nothing)

Four-panel summary of ForceFreeStates (DCON-style) stability results, combining:

  - Energy eigenvector heatmap (`plot_energy_eigenvectors`)
  - Fixed-boundary stability criterion |D_c| vs ψ_N (`plot_stability_criterion`)
  - Eigenvalue spectrum (`plot_eigenvalues`)
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
function plot_ffs_summary(h5path; save_path=nothing)
    has_vac = h5open(h5path, "r") do fid
        haskey(fid, "ForceFreeStates/FreeBoundaryStability/W_freeboundary_eigenmodes") &&
            !isempty(read(fid["ForceFreeStates/FreeBoundaryStability/W_freeboundary_eigenmodes"]))
    end

    p_crit = plot_fixed_boundary_stability_criterion(h5path)
    p_dp = plot_delta_prime(h5path)

    if has_vac
        p_evec = plot_energy_eigenvectors(h5path; matrix_type=:total)
        p_modes = plot_mode_displacement(h5path)
        p = plot(p_evec, p_crit, p_modes, p_dp; layout=(2, 2), size=(1100, 900), right_margin=10Plots.mm)
    else
        title!(p_crit, "Stability criterion (no vacuum data — rerun with vac_flag = true)")
        p = plot(p_crit, p_dp; layout=(1, 2), size=(1100, 500), right_margin=10Plots.mm)
    end

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_ballooning_alpha_boundaries(bnd; save_path=nothing, psi_min=0.0)

Plot the BALOO-style infinite-n ballooning stability diagram with first and second stability
boundaries: experimental α (solid black), 1st stability boundary α_crit1 (dashed red), and
2nd stability boundary α_crit2 (dashed blue) versus ψ_N. `NaN` entries in the boundary
arrays leave natural gaps over always-stable surfaces without requiring explicit masking.

### Arguments

  - `bnd`: NamedTuple with fields `psi`, `alpha`, `alpha_critical1`, `alpha_critical2`
    (as returned by `ForceFreeStates.ballooning_alpha_boundaries`)

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)
  - `psi_min`: Lower ψ_N axis limit (default: `0.0`)

### Returns

A `Plots.jl` plot object.
"""
function plot_ballooning_alpha_boundaries(bnd; save_path=nothing, psi_min=0.0)
    p = plot(;
        xlims=(psi_min, 1),
        xlabel=L"\psi_N",
        ylabel=L"\alpha",
        title="Infinite-n ballooning stability",
        framestyle=:box,
        legend=:topleft,
        left_margin=10Plots.mm,
        bottom_margin=5Plots.mm
    )
    plot!(p, bnd.psi, bnd.alpha; lw=2, color=:black, label="Experimental gradient")
    plot!(p, bnd.psi, bnd.alpha_critical1; lw=2, color=:red, marker=:circle, label="1st stability boundary")
    plot!(p, bnd.psi, bnd.alpha_critical2; lw=2, color=:blue, marker=:circle, label="2nd stability boundary")
    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_ballooning_alpha_boundaries(bnd, dpmap; save_path=nothing, psi_min=0.0)

Same diagram drawn over a heatmap of the signed Δ' from
`ForceFreeStates.ballooning_delta_prime_map`: each surface's Δ'(α) is oriented by the
sign of its α=0 (stable) value so that positive is stable everywhere, regridded from
its native physical α = α_ref*scale onto a shared uniform α axis, and shown with the
Δ'=0 contour, the extracted boundaries, and the scan cap `max_alpha_scale*α_exp`.
Color limits are set to the 90th percentile of |Δ'| so the pole regions inside the
unstable band do not wash out the marginal structure.
"""
function plot_ballooning_alpha_boundaries(bnd, dpmap; save_path=nothing, psi_min=0.0)
    psi = dpmap.psi
    scales = dpmap.alpha_scales
    alpha_ref = dpmap.alpha_ref
    signed = dpmap.delta_prime .* sign.(dpmap.delta_prime[:, 1])

    finite_aref = filter(isfinite, alpha_ref)
    isempty(finite_aref) && error("delta-prime map contains no valid surfaces")
    alpha_axis = collect(range(0.0, maximum(finite_aref) * scales[end]; length=200))
    z = fill(NaN, length(alpha_axis), length(psi))
    for i in eachindex(psi)
        (isfinite(alpha_ref[i]) && alpha_ref[i] > 0.0) || continue
        a_surf = alpha_ref[i] .* scales
        for (j, a) in enumerate(alpha_axis)
            a > a_surf[end] && break
            k = min(searchsortedlast(a_surf, a), length(a_surf) - 1)
            k < 1 && continue
            t = (a - a_surf[k]) / (a_surf[k+1] - a_surf[k])
            z[j, i] = (1 - t) * signed[i, k] + t * signed[i, k+1]
        end
    end

    # Color limits track the unstable-side magnitudes; deep stable values and pole
    # regions clamp at the ends instead of washing out the marginal band.
    finite_z = abs.(filter(isfinite, vec(z)))
    finite_neg = abs.(filter(x -> isfinite(x) && x < 0.0, vec(z)))
    lim = isempty(finite_neg) ? (isempty(finite_z) ? 1.0 : quantile(finite_z, 0.9)) : quantile(finite_neg, 0.95)
    lim = max(lim, eps(Float64))

    p = heatmap(
        psi,
        alpha_axis,
        z;
        c=cgrad(:RdBu),
        clims=(-lim, lim),
        # Leading newline offsets the rotated title clear of the colorbar tick labels (GR quirk)
        colorbar_title="\n" * L"\Delta' \times \mathrm{sign}(\Delta'_{\alpha=0})",
        xlims=(psi_min, 1),
        xlabel=L"\psi_N",
        ylabel=L"\alpha",
        title="Infinite-n ballooning stability",
        framestyle=:box,
        legend=:topleft,
        left_margin=10Plots.mm,
        bottom_margin=5Plots.mm,
        right_margin=15Plots.mm
    )
    contour!(p, psi, alpha_axis, z; levels=[0.0], color=:black, linewidth=1, colorbar_entry=false)
    plot!(p, bnd.psi, bnd.alpha; lw=2, color=:black, label="Experimental gradient")
    plot!(p, bnd.psi, bnd.alpha_critical1; lw=2, color=:red, linestyle=:dash, label="1st stability boundary")
    plot!(p, bnd.psi, bnd.alpha_critical2; lw=2, color=:blue, linestyle=:dash, label="2nd stability boundary")
    plot!(p, psi, scales[end] .* alpha_ref; lw=2, color=:gray, linestyle=:dot, label="scan cap")
    isnothing(save_path) || savefig(p, save_path)
    return p
end

end # module ForceFreeStates
