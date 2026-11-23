"""
    ForceFreeStates

Post-processing and visualization functions for ForceFreeStates (DCON-style ideal MHD stability) results. Each plot
takes the `ForceFreeStatesResult` in memory or the path of a `gpec.h5` it was saved to, and throws an `ArgumentError`
naming what to change in the run when the source lacks the data it draws (see `Analysis.available_plots`).
"""
module ForceFreeStates

using HDF5
using LaTeXStrings
using Plots
using Printf
using Statistics: quantile

using ..Analysis: register!, _require, _can, _missing, _entry, _why, capabilities
import ...ForceFreeStates: ForceFreeStatesResult
import ...LocalStability: LocalStabilityResult

const Source = Union{AbstractString,ForceFreeStatesResult}

const _FWD = "ForceFreeStates/Solutions/ForwardIntegration"
const _GAL = "ForceFreeStates/Solutions/GalerkinIntegration"
const _FBS = "ForceFreeStates/FreeBoundaryStability"

_label(h5path::AbstractString) = h5path
_label(::ForceFreeStatesResult) = "in memory"

# Least-stable total energy, or `nothing` without free-boundary energies.
_first_energy(et) = isempty(et) ? nothing : et[1]

function _mode_displacement_data(r::ForceFreeStatesResult)
    sol = r.solution
    return (; mlow=r.mlow, psi=sol.psi_store, xi=sol.u_store[:, 1, 1, :], dW=r.free_boundary === nothing ? nothing : _first_energy(r.free_boundary.et))
end

function _mode_displacement_data(h5path::AbstractString)
    h5open(h5path, "r") do fid
        # Forward integration stores its profiles in ForwardIntegration, a matched Galerkin solve in GalerkinIntegration.
        group = isempty(fid["$_FWD/xi_psi"]) ? _GAL : _FWD
        (; mlow=read(fid["Info/mlow"]), psi=read(fid["$group/psi"]), xi=read(fid["$group/xi_psi"])[:, 1, :],
            dW=_first_energy(read(fid["$_FBS/eigenmode_energies"])))
    end
end

"""
    plot_mode_displacement(src; modes=1:5, save_path=nothing)

Plot |ξ_ψ| vs ψ_N for the least stable eigenmode, showing one curve per requested
poloidal mode number m. The title includes the first eigenvalue dW = et[1].

### Arguments

  - `src`: A `ForceFreeStatesResult` with stored ξ profiles, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `modes`: Iterable of m values to plot (default: `1:5`)
  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_mode_displacement(src::Source; modes=1:5, save_path=nothing)
    _require(plot_mode_displacement, src)
    p = _plot_mode_displacement(_mode_displacement_data(src); modes)
    isnothing(save_path) || savefig(p, save_path)
    return p
end

function _plot_mode_displacement(d; modes=1:5)
    mpert = size(d.xi, 1)
    mhigh = d.mlow + mpert - 1
    title_str = isnothing(d.dW) ? "Least stable mode" :
                "Least stable mode, δW = $(round(real(d.dW), sigdigits=4))"

    p = plot(;
        xlims=(0, 1),
        xlabel="Norm. Poloidal Flux",
        ylabel="|ξ^ψ|",
        title=title_str,
        left_margin=10Plots.mm,
        bottom_margin=5Plots.mm
    )
    for m in modes
        d.mlow <= m <= mhigh || continue
        plot!(p, d.psi, abs.(d.xi[m-d.mlow+1, :]); label="m=$m")
    end
    return p
end

_criterion_data(r::ForceFreeStatesResult) = (; psi=r.diagnostics.psi_store, crit=r.diagnostics.crit_store)
_criterion_data(h5path::AbstractString) = h5open(fid -> (; psi=read(fid["$_FWD/psi"]), crit=read(fid["$_FWD/crit"])), h5path, "r")

"""
    plot_fixed_boundary_stability_criterion(src; save_path=nothing)

Plot the critical determinant (smallest eigenvalue of W⁻¹, `crit`) vs ψ_N.
A sign change in `crit` during integration indicates an ideal fixed-boundary instability.

### Arguments

  - `src`: A Forward or Riccati `ForceFreeStatesResult`, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_fixed_boundary_stability_criterion(src::Source; save_path=nothing)
    _require(plot_fixed_boundary_stability_criterion, src)
    p = _plot_fixed_boundary_stability_criterion(_criterion_data(src))
    isnothing(save_path) || savefig(p, save_path)
    return p
end

function _plot_fixed_boundary_stability_criterion(d)
    p = plot(
        d.psi, d.crit;
        xlims=(0, 1),
        xlabel="Norm. Poloidal Flux",
        ylabel="|Dᶜ|",
        title="Fixed-Boundary Stability",
        legend=false,
        left_margin=10Plots.mm,
        bottom_margin=5Plots.mm
    )
    hline!(p, [0.0]; linestyle=:dash, color=:black, label=nothing)
    return p
end

_energy_eigenvector_data(r::ForceFreeStatesResult) = (; wt=r.free_boundary.wt, psio=r.equil.psio, mlow=r.mlow)
_energy_eigenvector_data(h5path::AbstractString) =
    h5open(fid -> (; wt=read(fid["$_FBS/W_freeboundary_eigenmodes"]), psio=read(fid["Equilibrium/psi_total"]), mlow=read(fid["Info/mlow"])), h5path, "r")

"""
    plot_energy_eigenvectors(src; matrix_type=:total, save_path=nothing)

Heatmap of energy eigenvector magnitudes vs (m, mode index).

Only `matrix_type=:total` is supported (the total energy eigenvector matrix `Wₜ`). Plasma and vacuum eigenvectors
are not stored separately.

Eigenvectors are scaled by χ₁ = 2π ψ₀ × 10⁻³ to match GPEC conventions.

### Arguments

  - `src`: A `ForceFreeStatesResult` with free-boundary energies (`vac_flag = true`), or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `matrix_type`: Energy matrix to plot; only `:total` is currently supported
  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_energy_eigenvectors(src::Source; matrix_type=:total, save_path=nothing)
    matrix_type == :total ||
        throw(ArgumentError("matrix_type=$matrix_type not supported; only :total has an eigenvector matrix (ep/ev are eigenvalue vectors, not matrices)"))
    _require(plot_energy_eigenvectors, src)
    p = _plot_energy_eigenvectors(_energy_eigenvector_data(src))
    isnothing(save_path) || savefig(p, save_path)
    return p
end

function _plot_energy_eigenvectors(d)
    chi1 = 2π * d.psio
    wt_scaled = d.wt * (chi1 * 1e-3)

    nmn = size(wt_scaled, 1)
    nmodes = size(wt_scaled, 2)
    m_vals = (0:(nmn-1)) .+ d.mlow

    return heatmap(
        m_vals, 1:nmodes, abs.(wt_scaled');
        xlabel="Poloidal Harmonic",
        ylabel="Eigenmode Index",
        title="Total Energy Eigenvectors",
        colorbar_title="Harmonic Amplitude",
        left_margin=10Plots.mm,
        right_margin=20Plots.mm,
        bottom_margin=5Plots.mm
    )
end

function _edge_scan_data(r::ForceFreeStatesResult)
    es = r.diagnostics.edge_scan
    return (; q=es.q, et=es.total_eigenvalue, ep=es.plasma_energy, ev=es.vacuum_energy, evonly=es.vacuum_eigenvalue, qlim=r.qlim, label=_label(r))
end

function _edge_scan_data(h5path::AbstractString)
    h5open(h5path, "r") do fid
        (; q=read(fid["ForceFreeStates/EdgeScan/q"]), et=read(fid["ForceFreeStates/EdgeScan/total_energy"]),
            ep=read(fid["ForceFreeStates/EdgeScan/plasma_energy"]), ev=read(fid["ForceFreeStates/EdgeScan/vacuum_energy"]),
            evonly=read(fid["ForceFreeStates/EdgeScan/vacuum_eigenvalue"]), qlim=read(fid["Info/qlim"]), label=_label(h5path))
    end
end

"""
    plot_edge_stability_scan(src; save_path=nothing, ylims=(-2, 3), kwargs...)

Plot the edge stability scan energy components (et, ep, ev, evonly) vs ψ_N.

The edge scan evaluates `δW_total = δW_plasma + δW_vacuum` at each stored integration step
in the region [dW_edge_scan_start, psilim], with the plasma boundary swept from dW_edge_scan_start to psilim.
A positive et indicates stability; the truncation point is chosen at the peak et.

Four subplots are shown:

  - **Total energy** `et = ep + ev`: total free-boundary energy eigenvalue
  - **Plasma energy** `ep`: plasma contribution to δW
  - **Vacuum energy** `ev`: vacuum (wv) contribution with singfac scaling
  - **Vacuum-only eigenvalue** `evonly`: smallest eigenvalue of wv alone (no plasma response)

A horizontal dashed line at zero marks the stability boundary. A vertical dashed line marks
`psilim` (the final truncation psi, where the peak et was found).

### Arguments

  - `src`: A `ForceFreeStatesResult` produced with `dW_edge_scan_start < psilim`, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)
  - `ylims`: y-axis limits applied to all panels (default: `(-2, 3)`)
  - `kwargs...`: Additional Plots.jl keyword arguments applied to all line plots (e.g. `lw=2`)

### Returns

A `Plots.jl` plot object.
"""
function plot_edge_stability_scan(src::Source; save_path=nothing, ylims=(-2, 3), kwargs...)
    _require(plot_edge_stability_scan, src)
    d = _edge_scan_data(src)
    q, qlim = d.q, d.qlim

    kw_re = (xlabel="q", label="Re", ylims=ylims, kwargs...)
    kw_im = (xlabel="q", label="Im", ls=:dash, ylims=ylims, kwargs...)
    vl_kw = (color=:gray, lw=1, ls=:dot, label=false)
    hl_kw = (color=:black, lw=1, ls=:dash, label=false)

    p_et = plot(q, real.(d.et); ylabel="Total Energy", title="Edge Stability Scan: δW vs q", kw_re...)
    plot!(p_et, q, imag.(d.et); kw_im...)
    hline!(p_et, [0.0]; hl_kw...)
    vline!(p_et, [qlim]; vl_kw...)

    p_ep = plot(q, real.(d.ep); ylabel="Plasma Energy", kw_re...)
    plot!(p_ep, q, imag.(d.ep); kw_im...)
    hline!(p_ep, [0.0]; hl_kw...)
    vline!(p_ep, [qlim]; vl_kw...)

    p_ev = plot(q, real.(d.ev); ylabel="Vacuum Energy", kw_re...)
    plot!(p_ev, q, imag.(d.ev); kw_im...)
    hline!(p_ev, [0.0]; hl_kw...)
    vline!(p_ev, [qlim]; vl_kw...)

    p_evonly = plot(q, d.evonly; ylabel="Min Vac. Eigenvalue", legend=false, xlabel="q", ylims=ylims, kwargs...)
    hline!(p_evonly, [0.0]; hl_kw...)
    vline!(p_evonly, [qlim]; vl_kw...)

    p = plot(p_et, p_ep, p_ev, p_evonly;
        layout=(4, 1),
        size=(900, 1100),
        plot_title="Edge stability scan: $(d.label)",
        left_margin=12Plots.mm,
        bottom_margin=4Plots.mm)

    isnothing(save_path) || savefig(p, save_path)
    return p
end

const _ENERGY_FIELDS = Dict(:total => :et, :plasma => :ep, :vacuum => :ev)
const _ENERGY_DATASETS = Dict(:total => "eigenmode_energies", :plasma => "eigenmode_plasma_energies", :vacuum => "eigenmode_vacuum_energies")

_eigenvalue_data(r::ForceFreeStatesResult, matrix_type) = getfield(r.free_boundary, _ENERGY_FIELDS[matrix_type])
_eigenvalue_data(h5path::AbstractString, matrix_type) = h5open(fid -> read(fid["$_FBS/$(_ENERGY_DATASETS[matrix_type])"]), h5path, "r")

"""
    plot_eigenvalues(src; matrix_type=:total, save_path=nothing)

Scatter plot of energy eigenvalues vs mode index. Points are colored red (unstable, Re < 0)
or blue (stable, Re > 0), with a dashed reference line at zero.

### Arguments

  - `src`: A `ForceFreeStatesResult` with free-boundary energies (`vac_flag = true`), or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `matrix_type`: Which eigenvalues to plot: `:total` (`et`), `:plasma` (`ep`), or `:vacuum` (`ev`)
  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_eigenvalues(src::Source; matrix_type=:total, save_path=nothing)
    haskey(_ENERGY_FIELDS, matrix_type) || throw(ArgumentError("matrix_type must be :total, :plasma, or :vacuum"))
    _require(plot_eigenvalues, src)
    et = _eigenvalue_data(src, matrix_type)

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

# Δ′_ii on the surfaces the Δ′ matrix spans: Galerkin solves across its in-band subset of the rational surfaces.
function _delta_prime_data(r::ForceFreeStatesResult)
    dp = r.delta_prime.matrix
    k = minimum(size(dp))
    g = r.galerkin
    psi, q = g !== nothing && length(g.sing_psi) == k ? (g.sing_psi, g.sing_q) : ([s.psifac for s in r.surfaces[1:k]], [s.q for s in r.surfaces[1:k]])
    return (; psi, q, delta_prime=[dp[i, i] for i in 1:k])
end

function _delta_prime_data(h5path::AbstractString)
    h5open(h5path, "r") do fid
        dp = read(fid["SingularSurfaces/Delta_prime_matrix"])
        k = minimum(size(dp))
        group = haskey(fid, "$_GAL/rational_psi") && length(fid["$_GAL/rational_psi"]) == k ? _GAL : "SingularSurfaces"
        (; psi=read(fid["$group/rational_psi"])[1:k], q=read(fid["$group/rational_q"])[1:k], delta_prime=[dp[i, i] for i in 1:k])
    end
end

"""
    plot_delta_prime(src; save_path=nothing)

Scatter plot of the diagonal `Re(Δ′_ii)` of the inter-surface Δ′ matrix (PEST-3 convention, ψ_N-referenced) vs ψ_N
of each rational surface it spans. Points are colored red (tearing unstable, Re(Δ′) > 0) or blue (tearing stable).

### Arguments

  - `src`: A Riccati or Galerkin `ForceFreeStatesResult`, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_delta_prime(src::Source; save_path=nothing)
    _require(plot_delta_prime, src)
    p = _plot_delta_prime(_delta_prime_data(src))
    isnothing(save_path) || savefig(p, save_path)
    return p
end

function _plot_delta_prime(d)
    dp_real = real.(d.delta_prime)
    colors = [v > 0 ? :red : :steelblue for v in dp_real]

    p = scatter(
        d.psi, dp_real;
        xlims=(0, 1),
        xlabel="Norm. Poloidal Flux",
        ylabel="Re(Δ'_ii)",
        title="Tearing stability Δ'",
        legend=false,
        color=colors,
        markersize=7,
        markerstrokewidth=0,
        left_margin=10Plots.mm,
        bottom_margin=5Plots.mm
    )
    hline!(p, [0.0]; linestyle=:dash, color=:black, label=nothing)
    return p
end

_ballooning_data(r::LocalStabilityResult) = (; psi=r.ballooning_psi, alpha=r.alpha, alpha_crit=r.alpha_critical)
_ballooning_data(h5path::AbstractString) =
    h5open(fid -> (; psi=read(fid["LocalStability/ballooning_psi"]), alpha=read(fid["LocalStability/alpha"]), alpha_crit=read(fid["LocalStability/alpha_critical"])), h5path, "r")

"""
    plot_ballooning_alpha_boundary(src; save_path=nothing, psi_min=0.0)

Plot the BALOO-style infinite-n ballooning stability diagram: the experimental
pressure gradient α (solid) and the first stability boundary α_crit (dashed) versus
normalized poloidal flux ψ_N. Surfaces where the experimental α lies above the boundary
are ballooning-unstable.

### Arguments

  - `src`: A `LocalStabilityResult`, or the path of a `gpec.h5` written with a `[LocalStability]` section

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)
  - `psi_min`: Lower ψ_N axis limit; set near the edge (e.g. `0.9`) to focus on the
    pedestal (default: `0.0`)

### Returns

A `Plots.jl` plot object.
"""
function plot_ballooning_alpha_boundary(src::Union{AbstractString,LocalStabilityResult}; save_path=nothing, psi_min=0.0)
    _require(plot_ballooning_alpha_boundary, src)
    d = _ballooning_data(src)

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
    plot!(p, d.psi, d.alpha; lw=2, color=:black, label="Experimental gradient")
    plot!(p, d.psi, d.alpha_crit; lw=2, linestyle=:dash, color=:red, label="1st stability boundary")

    isnothing(save_path) || savefig(p, save_path)
    return p
end

function _cond_fbar_data(r::ForceFreeStatesResult)
    k = r.kinetic
    return (; scan_psi=k.scan_psi, scan_cond=k.scan_cond, thr=k.scan_threshold, k_psi=[s.psifac for s in k.kinsing],
        i_psi=[s.psifac for s in r.surfaces], i_q=[s.q for s in r.surfaces], kmsing=k.kmsing)
end

function _cond_fbar_data(h5path::AbstractString)
    h5open(h5path, "r") do fid
        kg = fid["SingularSurfaces/Kinetic"]
        (; scan_psi=read(kg["scan_psi"]), scan_cond=read(kg["scan_cond"]), thr=read(kg["scan_threshold"]), k_psi=read(kg["rational_psi"]),
            i_psi=read(fid["SingularSurfaces/rational_psi"]), i_q=read(fid["SingularSurfaces/rational_q"]), kmsing=read(kg["rational_count"]))
    end
end

"""
    plot_cond_fbar(src; save_path=nothing, zoom=false)

Plot `cond(F̄)` vs ψ from the kinetic-singular-surface scan (populated when ForceFreeStates runs with
a `kinetic` source and `singfac_min > 0`).

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

  - `src`: A kinetic `ForceFreeStatesResult`, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)
  - `zoom`: If `true`, auto-scale the y-axis to the scan data (threshold shown
    as annotation only); if `false` (default), always include the threshold
    line in the y-range

### Returns

A `Plots.jl` plot object.
"""
function plot_cond_fbar(src::Source; save_path=nothing, zoom=false)
    _require(plot_cond_fbar, src)
    (; scan_psi, scan_cond, thr, k_psi, i_psi, i_q, kmsing) = _cond_fbar_data(src)

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

# The summary's panels in drawing order, each with the plot whose capabilities it needs.
_ffs_panels(src) = (
    plot_energy_eigenvectors => () -> _plot_energy_eigenvectors(_energy_eigenvector_data(src)),
    plot_fixed_boundary_stability_criterion => () -> _plot_fixed_boundary_stability_criterion(_criterion_data(src)),
    plot_mode_displacement => () -> _plot_mode_displacement(_mode_displacement_data(src)),
    plot_delta_prime => () -> _plot_delta_prime(_delta_prime_data(src))
)

"""
    plot_ffs_summary(src; save_path=nothing)

Summary of ForceFreeStates (DCON-style) stability results, with each panel the source can supply:

  - Energy eigenvector heatmap (`plot_energy_eigenvectors`)
  - Critical determinant |D_c| vs ψ_N (`plot_fixed_boundary_stability_criterion`)
  - Least stable mode displacement (`plot_mode_displacement`)
  - Tearing stability Δ' at each rational surface (`plot_delta_prime`)

Panels the source cannot draw are left out; it throws an `ArgumentError` only when none can be drawn.

### Arguments

  - `src`: A `ForceFreeStatesResult`, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_ffs_summary(src::Source; save_path=nothing)
    p = _summary(_ffs_panels(src), capabilities(src), plot_ffs_summary)
    isnothing(save_path) || savefig(p, save_path)
    return p
end

# Draw the panels `caps` allows, two per row; throw naming every reason when none can be drawn.
function _summary(panels, caps, fn)
    drawn = [draw() for (panel_fn, draw) in panels if _can(panel_fn, caps)]
    if isempty(drawn)
        missing = unique(reduce(vcat, (_missing(_entry(panel_fn), caps) for (panel_fn, _) in panels)))
        throw(ArgumentError("$(nameof(fn)): no panel can be drawn: $(_why(missing))"))
    end
    n = length(drawn)
    n <= 2 && return plot(drawn...; layout=(1, n), size=(550n, 500), right_margin=10Plots.mm)
    n == 3 && push!(drawn, plot(; framestyle=:none))
    return plot(drawn...; layout=(2, 2), size=(1100, 900), right_margin=10Plots.mm)
end

register!(plot_ffs_summary, (ForceFreeStatesResult, AbstractString), (), "the stability panels the run supplies")
register!(plot_mode_displacement, (ForceFreeStatesResult, AbstractString), (:solution_profiles,), "|ξ_ψ| per m of the least stable mode")
register!(plot_fixed_boundary_stability_criterion, (ForceFreeStatesResult, AbstractString), (:fixed_boundary_criterion,), "critical determinant against ψ_N")
register!(plot_energy_eigenvectors, (ForceFreeStatesResult, AbstractString), (:free_boundary,), "free-boundary energy eigenvectors per m")
register!(plot_eigenvalues, (ForceFreeStatesResult, AbstractString), (:free_boundary,), "free-boundary energy eigenvalues")
register!(plot_edge_stability_scan, (ForceFreeStatesResult, AbstractString), (:edge_scan,), "edge energies against the truncation q")
register!(plot_delta_prime, (ForceFreeStatesResult, AbstractString), (:delta_prime,), "Δ′_ii at each rational surface")
register!(plot_cond_fbar, (ForceFreeStatesResult, AbstractString), (:kinetic_scan,), "cond(F̄) scan for kinetic singular surfaces")
register!(plot_ballooning_alpha_boundary, (LocalStabilityResult, AbstractString), (:ballooning_scan,), "ballooning α against its first stability boundary")

"""
    plot_ballooning_alpha_boundaries(bnd; save_path=nothing, psi_min=0.0)

Plot the BALOO-style infinite-n ballooning stability diagram with first and second stability
boundaries: experimental α (solid black), 1st stability boundary α_crit1 (dashed red), and
2nd stability boundary α_crit2 (dashed blue) versus ψ_N. `NaN` entries in the boundary
arrays leave natural gaps over always-stable surfaces without requiring explicit masking.

### Arguments

  - `bnd`: NamedTuple with fields `psi`, `alpha`, `alpha_critical1`, `alpha_critical2`
    (as returned by `LocalStability.ballooning_alpha_boundaries`)

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
`LocalStability.ballooning_delta_prime_map`: each surface's Δ'(α) is oriented by the
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
