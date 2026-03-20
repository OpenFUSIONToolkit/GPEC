"""
    Analysis.Equilibrium

Post-processing and visualization functions for GPEC equilibrium objects
(`PlasmaEquilibrium`).

## Public API

  - [`plot_profile_splines`](@ref)  /  `plot_profile_splines!`
  - [`plot_rzphi_splines`](@ref)    /  `plot_rzphi_splines!`
  - [`plot_eqfun_splines`](@ref)    /  `plot_eqfun_splines!`
  - [`plot_equilibrium_contours`](@ref)  /  `plot_equilibrium_contours!`
  - [`plot_flux_surfaces`](@ref)  (simple single-panel convenience plot)

The non-`!` functions create a new figure, set the layout and axis attributes,
then delegate to the `!` variant to add data.  Calling the `!` variant on an
existing figure overlays an additional case without changing the layout or zoom
limits, allowing easy multi-case comparisons.

## Typical usage

```julia
p = plot_profile_splines(pe_ref; color=:steelblue, label="hamada")
plot_profile_splines!(p, pe_blend; color=:darkorange, linestyle=:dash, label="blend")
savefig(p, "profile_comparison.png")
```
"""
module Equilibrium

using Plots

# ─── Defaults ─────────────────────────────────────────────────────────────────
const _PROFILE_NAMES = ["F  (2π·R·Bₜ)", "μ₀·P", "dV/dψ", "q"]
const _RZPHI_NAMES   = ["r²=(R−R₀)²+(Z−Z₀)²", "η/2π−θ  (offset)", "ν  (toroidal)", "Jacobian"]
const _EQFUN_NAMES   = ["|B|  [T]", "metric₁  (e₁·e₂+q e₃·e₁)/JB²", "metric₂  (e₂·e₃+q e₃²)/JB²"]

const _DEFAULT_THETA_VALS   = [0.0, 0.25, 0.5, 0.75]
const _DEFAULT_THETA_COLORS = [:steelblue, :forestgreen, :darkorange, :purple]

# ─── Private helpers ──────────────────────────────────────────────────────────
"Convert (ψ, θ) SFL coordinates to physical (R, Z)."
function _sfl_to_RZ(pe, psi, theta)
    r2   = pe.rzphi_rsquared((psi, theta))
    off  = pe.rzphi_offset((psi, theta))
    rfac = sqrt(max(r2, 0.0))
    η    = 2π * (theta + off)
    return pe.ro + rfac * cos(η), pe.zo + rfac * sin(η)
end

"""
Build the dense ψ evaluation grid and derived quantities used by all three
row panels of the spline figures.

Returns `(psi, mask_core, mask_edge, t_edge)` where `t_edge = −ln(1−ψ)` is
the coordinate used for the edge panel x-axis, which linearises the
separatrix-asymptotic `q ~ A·ln(1/(1−ψ))` behaviour.
"""
function _psi_rows(pe; n_eval, core_cutoff, edge_cutoff)
    psi    = collect(range(pe.rzphi_xs[1], pe.rzphi_xs[end]; length=n_eval))
    m_core = psi .< core_cutoff
    m_edge = psi .> edge_cutoff
    t_edge = -log.(1.0 .- psi[m_edge])
    return psi, m_core, m_edge, t_edge
end

"Create a blank 3×ncols figure with titles, x-axis labels, and log scales pre-set."
function _new_spline_frame(ncols, col_names; figsize)
    p = plot(layout=(3, ncols); size=figsize,
             left_margin=8Plots.mm, bottom_margin=5Plots.mm, top_margin=3Plots.mm)
    for ci in 1:ncols
        plot!(p; title=col_names[ci], xlabel="ψ",
              legend=ci == 1 ? :topleft : :none, subplot=ci)
        plot!(p; xlabel="ψ  [log scale]", xscale=:log10,
              legend=:none, subplot=ncols + ci)
        plot!(p; xlabel="−ln(1−ψ)  [edge zoom]",
              legend=:none, subplot=2ncols + ci)
    end
    return p
end

"""
Add one curve (or one curve per theta value) to all three row-panels of a
single column.

For 1D columns (`fn` is a `Function(pe) → Vector`), pass `theta_vals=nothing`.
For 2D columns (`fn` is a `Function(ψ, θ) → Float64`), pass the theta list
and corresponding colors.
"""
function _add_col!(p, ci, ncols, psi, m_core, m_edge, t_edge, xs_knots;
                   fn, color, linestyle, label,
                   show_knots, core_cutoff, edge_cutoff,
                   theta_vals, theta_colors)
    is_1d = isnothing(theta_vals)
    iters = is_1d ? [(nothing, color, label)] :
                    [(θ, theta_colors[min(ti, end)], ti == 1 ? label : "")
                     for (ti, θ) in enumerate(theta_vals)]

    for (θ, tc, lbl) in iters
        y = is_1d ? fn(psi) : [fn(ψ, θ) for ψ in psi]

        plot!(p, psi, y;           color=tc, linestyle=linestyle, label=lbl,   subplot=ci)
        any(m_core) &&
            plot!(p, psi[m_core], y[m_core]; color=tc, linestyle=linestyle, label="", subplot=ncols + ci)
        any(m_edge) &&
            plot!(p, t_edge,      y[m_edge]; color=tc, linestyle=linestyle, label="", subplot=2ncols + ci)

        if show_knots
            ys_k = is_1d ? fn(xs_knots) : [fn(ψ, θ) for ψ in xs_knots]
            scatter!(p, xs_knots, ys_k; color=tc, ms=3, markerstrokewidth=0,
                     label="", subplot=ci)
            m_kc = xs_knots .< core_cutoff
            any(m_kc) && scatter!(p, xs_knots[m_kc], ys_k[m_kc]; color=tc, ms=3,
                markerstrokewidth=0, label="", subplot=ncols + ci)
            m_ke = xs_knots .> edge_cutoff
            any(m_ke) && scatter!(p, -log.(1.0 .- xs_knots[m_ke]), ys_k[m_ke]; color=tc,
                ms=3, markerstrokewidth=0, label="", subplot=2ncols + ci)
        end
    end
end

"Compute zoom limits for the core and x-point panels of contour plots."
function _contour_zoom_limits(pe, psi_contours; n_theta)
    θs = range(0.0, 1.0; length=n_theta)

    R_core, Z_core = Float64[], Float64[]
    for psi in psi_contours[1:min(4, end)]
        for θ in θs; R, Z = _sfl_to_RZ(pe, psi, θ); push!(R_core, R); push!(Z_core, Z) end
    end
    pad = 0.03
    core_xlims = (minimum(R_core) - pad, maximum(R_core) + pad)
    core_ylims = (minimum(Z_core) - pad, maximum(Z_core) + pad)

    R_edge, Z_edge = Float64[], Float64[]
    for psi in psi_contours[max(1, end-2):end]
        for θ in θs; R, Z = _sfl_to_RZ(pe, psi, θ); push!(R_edge, R); push!(Z_edge, Z) end
    end
    xpt_idx   = argmin(Z_edge)
    Rxpt, Zxpt = R_edge[xpt_idx], Z_edge[xpt_idx]
    xpt_xlims = (Rxpt - 0.20, Rxpt + 0.20)
    xpt_ylims = (Zxpt - 0.05, Zxpt + 0.40)

    return core_xlims, core_ylims, xpt_xlims, xpt_ylims
end

"Create blank 2×3 contour figure with titles and aspect ratios set."
function _new_contour_frame(; figsize)
    p = plot(layout=(2, 3); size=figsize,
             left_margin=8Plots.mm, bottom_margin=5Plots.mm, top_margin=3Plots.mm)
    titles = ("Flux surfaces — full",    "Flux surfaces — core",    "Flux surfaces — x-point",
              "θ contours — full",        "θ contours — core",        "θ contours — x-point")
    for (k, t) in enumerate(titles)
        plot!(p; title=t, xlabel="R [m]", ylabel="Z [m]",
              aspect_ratio=:equal, legend=k == 1 ? :outertopright : :none, subplot=k)
    end
    return p
end

# ─── profile splines ──────────────────────────────────────────────────────────
"""
    plot_profile_splines(pe; kwargs...) → Plots.Plot
    plot_profile_splines!(p, pe; kwargs...)

4-column × 3-row figure of the four 1D equilibrium profile splines
(F = 2π·R·Bₜ, μ₀·P, dV/dψ, q).

**Rows**: full domain (linear ψ) | deep-core zoom (ψ, log-x) | far-edge zoom (−ln(1−ψ))

**Keyword arguments**

  - `color`: line color (default `:steelblue`)
  - `linestyle`: line style (default `:solid`)
  - `label`: legend entry (default `""`)
  - `show_knots`: scatter ψ grid knots (default `true`)
  - `core_cutoff`: upper ψ threshold for core panel (default `0.10`)
  - `edge_cutoff`: lower ψ threshold for edge panel (default `0.98`)
  - `n_eval`: dense evaluation grid size (default `500`)
"""
function plot_profile_splines(pe;
        color=:steelblue, linestyle=:solid, label="", show_knots=true,
        core_cutoff=0.10, edge_cutoff=0.98, n_eval=500)
    p = _new_spline_frame(4, _PROFILE_NAMES; figsize=(1800, 900))
    plot_profile_splines!(p, pe; color, linestyle, label, show_knots,
                          core_cutoff, edge_cutoff, n_eval)
    return p
end

function plot_profile_splines!(p, pe;
        color=:steelblue, linestyle=:solid, label="", show_knots=true,
        core_cutoff=0.10, edge_cutoff=0.98, n_eval=500)
    psi, m_core, m_edge, t_edge = _psi_rows(pe; n_eval, core_cutoff, edge_cutoff)
    splines = (pe.profiles.F_spline, pe.profiles.P_spline,
               pe.profiles.dVdpsi_spline, pe.profiles.q_spline)
    xs_k = pe.rzphi_xs

    for (ci, spl) in enumerate(splines)
        fn_1d = xs -> [spl(ψ) for ψ in xs]
        _add_col!(p, ci, 4, psi, m_core, m_edge, t_edge, xs_k;
                  fn=fn_1d, color, linestyle,
                  label=ci == 1 ? label : "",
                  show_knots, core_cutoff, edge_cutoff,
                  theta_vals=nothing, theta_colors=nothing)
    end
    return p
end

# ─── rzphi splines ────────────────────────────────────────────────────────────
"""
    plot_rzphi_splines(pe; kwargs...) → Plots.Plot
    plot_rzphi_splines!(p, pe; kwargs...)

4-column × 3-row figure of the four 2D rzphi geometric splines
(r², angle offset, ν, Jacobian) evaluated at `theta_vals`.

**Keyword arguments** (beyond those shared with `plot_profile_splines`)

  - `theta_vals`: poloidal angles to show (default `[0, 0.25, 0.5, 0.75]`)
  - `theta_colors`: colors paired with `theta_vals` (default blue/green/orange/purple)
  - `color`: if provided, overrides `theta_colors` for all theta lines
"""
function plot_rzphi_splines(pe;
        color=nothing, linestyle=:solid, label="", show_knots=true,
        theta_vals=_DEFAULT_THETA_VALS, theta_colors=_DEFAULT_THETA_COLORS,
        core_cutoff=0.10, edge_cutoff=0.98, n_eval=500)
    p = _new_spline_frame(4, _RZPHI_NAMES; figsize=(1800, 900))
    plot_rzphi_splines!(p, pe; color, linestyle, label, show_knots,
                        theta_vals, theta_colors, core_cutoff, edge_cutoff, n_eval)
    return p
end

function plot_rzphi_splines!(p, pe;
        color=nothing, linestyle=:solid, label="", show_knots=true,
        theta_vals=_DEFAULT_THETA_VALS, theta_colors=_DEFAULT_THETA_COLORS,
        core_cutoff=0.10, edge_cutoff=0.98, n_eval=500)
    psi, m_core, m_edge, t_edge = _psi_rows(pe; n_eval, core_cutoff, edge_cutoff)
    eff_colors = isnothing(color) ? theta_colors : fill(color, length(theta_vals))
    fns = ((ψ, θ) -> pe.rzphi_rsquared((ψ, θ)),
           (ψ, θ) -> pe.rzphi_offset((ψ, θ)),
           (ψ, θ) -> pe.rzphi_nu((ψ, θ)),
           (ψ, θ) -> pe.rzphi_jac((ψ, θ)))

    for (ci, fn) in enumerate(fns)
        _add_col!(p, ci, 4, psi, m_core, m_edge, t_edge, pe.rzphi_xs;
                  fn, color=nothing, linestyle,
                  label=ci == 1 ? label : "",
                  show_knots, core_cutoff, edge_cutoff,
                  theta_vals, theta_colors=eff_colors)
    end
    return p
end

# ─── eqfun splines ────────────────────────────────────────────────────────────
"""
    plot_eqfun_splines(pe; kwargs...) → Plots.Plot
    plot_eqfun_splines!(p, pe; kwargs...)

3-column × 3-row figure of the three 2D eqfun physics splines
(|B|, metric₁, metric₂) evaluated at `theta_vals`.

Accepts the same keyword arguments as [`plot_rzphi_splines`](@ref).
"""
function plot_eqfun_splines(pe;
        color=nothing, linestyle=:solid, label="", show_knots=true,
        theta_vals=_DEFAULT_THETA_VALS, theta_colors=_DEFAULT_THETA_COLORS,
        core_cutoff=0.10, edge_cutoff=0.98, n_eval=500)
    p = _new_spline_frame(3, _EQFUN_NAMES; figsize=(1400, 900))
    plot_eqfun_splines!(p, pe; color, linestyle, label, show_knots,
                        theta_vals, theta_colors, core_cutoff, edge_cutoff, n_eval)
    return p
end

function plot_eqfun_splines!(p, pe;
        color=nothing, linestyle=:solid, label="", show_knots=true,
        theta_vals=_DEFAULT_THETA_VALS, theta_colors=_DEFAULT_THETA_COLORS,
        core_cutoff=0.10, edge_cutoff=0.98, n_eval=500)
    psi, m_core, m_edge, t_edge = _psi_rows(pe; n_eval, core_cutoff, edge_cutoff)
    eff_colors = isnothing(color) ? theta_colors : fill(color, length(theta_vals))
    fns = ((ψ, θ) -> pe.eqfun_B((ψ, θ)),
           (ψ, θ) -> pe.eqfun_metric1((ψ, θ)),
           (ψ, θ) -> pe.eqfun_metric2((ψ, θ)))

    for (ci, fn) in enumerate(fns)
        _add_col!(p, ci, 3, psi, m_core, m_edge, t_edge, pe.rzphi_xs;
                  fn, color=nothing, linestyle,
                  label=ci == 1 ? label : "",
                  show_knots, core_cutoff, edge_cutoff,
                  theta_vals, theta_colors=eff_colors)
    end
    return p
end

# ─── equilibrium contours ─────────────────────────────────────────────────────
"""
    plot_equilibrium_contours(pe; kwargs...) → Plots.Plot
    plot_equilibrium_contours!(p, pe; kwargs...)

2-row × 3-column figure showing:
- **Row 1**: constant-ψ flux surface contours in (R, Z)
- **Row 2**: constant-θ contour lines in (R, Z) — the key diagnostic for theta grid uniformity

**Columns**: full plasma view | core zoom (innermost surfaces) | x-point zoom

The zoom limits (columns 2 and 3) are fixed by the first `pe` passed to the
non-`!` variant and are not altered by subsequent `plot_equilibrium_contours!`
calls.

**Keyword arguments**

  - `color`: color for all curves in this case (default `:steelblue`)
  - `linestyle`: line style (default `:solid`)
  - `label`: legend entry for this case (default `""`)
  - `show_knots`: scatter (ψᵢ, θⱼ) grid intersections in theta-contour row (default `true`)
  - `n_psi`: number of ψ contours to draw (default `20`)
  - `n_theta_line`: theta resolution for drawing each contour (default `256`)
  - `n_theta_spokes`: approximate number of θ=const lines (default `32`)
"""
function plot_equilibrium_contours(pe;
        color=:steelblue, linestyle=:solid, label="", show_knots=true,
        n_psi=20, n_theta_line=256, n_theta_spokes=32)
    p = _new_contour_frame(; figsize=(2100, 800))

    psi_contours = collect(range(pe.rzphi_xs[1], pe.rzphi_xs[end]; length=n_psi))
    core_xlims, core_ylims, xpt_xlims, xpt_ylims =
        _contour_zoom_limits(pe, psi_contours; n_theta=n_theta_line)

    # Fix zoom limits on panels 2, 3, 5, 6 before adding any data
    for sp in (2, 5)
        plot!(p; xlims=core_xlims, ylims=core_ylims, subplot=sp)
    end
    for sp in (3, 6)
        plot!(p; xlims=xpt_xlims, ylims=xpt_ylims, subplot=sp)
    end

    plot_equilibrium_contours!(p, pe; color, linestyle, label, show_knots,
                               n_psi, n_theta_line, n_theta_spokes)
    return p
end

function plot_equilibrium_contours!(p, pe;
        color=:steelblue, linestyle=:solid, label="", show_knots=true,
        n_psi=20, n_theta_line=256, n_theta_spokes=32)
    θ_line  = range(0.0, 1.0; length=n_theta_line)
    psi_contours = collect(range(pe.rzphi_xs[1], pe.rzphi_xs[end]; length=n_psi))

    # Row 1: flux surface contours (subplots 1, 2, 3)
    for (i, psi) in enumerate(psi_contours)
        lbl = i == 1 ? label : ""
        Rs = [_sfl_to_RZ(pe, psi, θ)[1] for θ in θ_line]
        Zs = [_sfl_to_RZ(pe, psi, θ)[2] for θ in θ_line]
        push!(Rs, Rs[1]); push!(Zs, Zs[1])  # close contour
        for sp in (1, 2, 3)
            plot!(p, Rs, Zs; color=color, linestyle=linestyle, lw=0.9,
                  alpha=0.8, label=sp == 1 ? lbl : "", subplot=sp)
        end
    end

    # Row 2: theta contour lines (subplots 4, 5, 6)
    θ_all  = pe.rzphi_ys
    step   = max(1, length(θ_all) ÷ n_theta_spokes)
    θ_sub  = θ_all[1:step:end]
    psi_dense = range(pe.rzphi_xs[1], pe.rzphi_xs[end]; length=300)

    for (ki, θ) in enumerate(θ_sub)
        lbl = ki == 1 ? label : ""
        Rs = [_sfl_to_RZ(pe, ψ, θ)[1] for ψ in psi_dense]
        Zs = [_sfl_to_RZ(pe, ψ, θ)[2] for ψ in psi_dense]
        for sp in (4, 5, 6)
            plot!(p, Rs, Zs; color=color, linestyle=linestyle, lw=0.9,
                  alpha=0.8, label=sp == 4 ? lbl : "", subplot=sp)
        end
    end

    # Grid dots at (ψᵢ, θⱼ) intersections — the key diagnostic for theta uniformity
    if show_knots
        Rg = [_sfl_to_RZ(pe, ψ, θ)[1] for ψ in pe.rzphi_xs for θ in θ_sub]
        Zg = [_sfl_to_RZ(pe, ψ, θ)[2] for ψ in pe.rzphi_xs for θ in θ_sub]
        for sp in (4, 5, 6)
            scatter!(p, Rg, Zg; color=color, ms=2, alpha=0.7, markerstrokewidth=0,
                     label="", subplot=sp)
        end
    end
    return p
end

# ─── Simple single-panel convenience plot (kept from original) ─────────────────
"""
    plot_flux_surfaces(plasma_eq; n_psi=11, n_theta=13) → Plots.Plot

Simple single-panel plot of constant-ψ (blue) and constant-θ (red) contours
in (R, Z).  For detailed multi-panel diagnostics use
[`plot_equilibrium_contours`](@ref) instead.
"""
function plot_flux_surfaces(plasma_eq; n_psi=11, n_theta=13)
    n_psi_grid   = size(plasma_eq.rzphi_rsquared.nodal_derivs.partials, 2)
    n_theta_grid = size(plasma_eq.rzphi_rsquared.nodal_derivs.partials, 3)

    R_grid = Matrix{Float64}(undef, n_psi_grid, n_theta_grid)
    Z_grid = Matrix{Float64}(undef, n_psi_grid, n_theta_grid)
    for ipsi in 1:n_psi_grid
        rfac  = @. sqrt(max(0.0, plasma_eq.rzphi_rsquared.nodal_derivs.partials[1, ipsi, :]))
        angle = @. 2π * (plasma_eq.rzphi_ys + plasma_eq.rzphi_offset.nodal_derivs.partials[1, ipsi, :])
        R_grid[ipsi, :] = plasma_eq.ro .+ rfac .* cos.(angle)
        Z_grid[ipsi, :] = plasma_eq.zo .+ rfac .* sin.(angle)
    end

    p = plot(; title="Flux Coordinate System Contours in (R, Z)",
               xlabel="R [m]", ylabel="Z [m]",
               aspect_ratio=:equal, legend=:outertopright)

    psi_indices   = round.(Int, range(1, n_psi_grid;   length=n_psi))
    theta_indices = round.(Int, range(1, n_theta_grid; length=n_theta))

    for (i, ipsi) in enumerate(psi_indices)
        plot!(p, [R_grid[ipsi, :]; R_grid[ipsi, 1]], [Z_grid[ipsi, :]; Z_grid[ipsi, 1]];
              color=:blue, lw=1.5, label=i == 1 ? "Constant ψ" : "")
    end
    for (i, itheta) in enumerate(theta_indices)
        plot!(p, R_grid[:, itheta], Z_grid[:, itheta];
              color=:red, lw=1.0, label=i == 1 ? "Constant θ" : "")
    end
    return p
end

end # module Equilibrium
