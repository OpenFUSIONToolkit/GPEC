"""
    Equilibrium

Post-processing and visualization functions for GPEC equilibria. Each plot takes a `PlasmaEquilibrium`, a
`ForceFreeStatesResult` (which adds its rational surfaces as markers), or the path of a `gpec.h5`.
"""
module Equilibrium

using HDF5
using LaTeXStrings
using Plots

using ..Analysis: register!, _require, _can, capabilities
import ...Equilibrium: PlasmaEquilibrium
import ...ForceFreeStates: ForceFreeStatesResult

const Source = Union{AbstractString,PlasmaEquilibrium,ForceFreeStatesResult}

# Rational surfaces (ψ_N, q) to mark: none for a bare equilibrium.
_surfaces(::PlasmaEquilibrium) = (; sing_psi=Float64[], sing_q=Float64[])
_surfaces(r::ForceFreeStatesResult) = (; sing_psi=[s.psifac for s in r.surfaces], sing_q=[s.q for s in r.surfaces])
function _surfaces(fid::HDF5.File)
    haskey(fid, "SingularSurfaces/rational_psi") || return _surfaces(nothing)
    return (; sing_psi=read(fid["SingularSurfaces/rational_psi"]), sing_q=read(fid["SingularSurfaces/rational_q"]))
end
_surfaces(::Nothing) = (; sing_psi=Float64[], sing_q=Float64[])

_equil(eq::PlasmaEquilibrium) = eq
_equil(r::ForceFreeStatesResult) = r.equil

function _profile_data(src::Union{PlasmaEquilibrium,ForceFreeStatesResult})
    eq = _equil(src)
    prof = eq.profiles
    return (; psi=prof.xs, q=prof.q_spline.y, mu0p=prof.P_spline.y, twopif=prof.F_spline.y, q0=eq.params.q0, q95=eq.params.q95, _surfaces(src)...)
end

function _profile_data(h5path::AbstractString)
    h5open(h5path, "r") do fid
        (; psi=read(fid["Equilibrium/Profiles/psi"]), q=read(fid["Equilibrium/Profiles/q"]), mu0p=read(fid["Equilibrium/Profiles/mu0p"]),
            twopif=read(fid["Equilibrium/Profiles/2piF"]), q0=read(fid["Equilibrium/q_axis"]), q95=read(fid["Equilibrium/q_95"]), _surfaces(fid)...)
    end
end

function _geometry_data(src::Union{PlasmaEquilibrium,ForceFreeStatesResult})
    eq = _equil(src)
    return (; rcoords=eq.rzphi_rsquared.nodal_derivs.partials[1, :, :], offset=eq.rzphi_offset.nodal_derivs.partials[1, :, :],
        psi=eq.rzphi_xs, theta=eq.rzphi_ys, ro=eq.ro, zo=eq.zo, _surfaces(src)...)
end

function _geometry_data(h5path::AbstractString)
    h5open(h5path, "r") do fid
        (; rcoords=read(fid["Equilibrium/Geometry/rcoords"]), offset=read(fid["Equilibrium/Geometry/offset"]), psi=read(fid["Equilibrium/Geometry/psi"]),
            theta=read(fid["Equilibrium/Geometry/theta"]), ro=read(fid["Equilibrium/R_axis"]), zo=read(fid["Equilibrium/Z_axis"]), _surfaces(fid)...)
    end
end

# R, Z on the full nodal (ψ, θ) grid of the geometry data.
function _rz_grid(d)
    R = Matrix{Float64}(undef, length(d.psi), length(d.theta))
    Z = similar(R)
    for ipsi in eachindex(d.psi), itheta in eachindex(d.theta)
        rfac = sqrt(max(0.0, d.rcoords[ipsi, itheta]))
        eta = 2π * (d.theta[itheta] + d.offset[ipsi, itheta])
        R[ipsi, itheta] = d.ro + rfac * cos(eta)
        Z[ipsi, itheta] = d.zo + rfac * sin(eta)
    end
    return R, Z
end

function _global_data(src::Union{PlasmaEquilibrium,ForceFreeStatesResult})
    p = _equil(src).params
    return (; q0=p.q0, q95=p.q95, betat=p.betat, betan=p.betan, kappa=p.kappa, li1=p.li1)
end

function _global_data(h5path::AbstractString)
    h5open(h5path, "r") do fid
        (; q0=read(fid["Equilibrium/q_axis"]), q95=read(fid["Equilibrium/q_95"]), betat=read(fid["Equilibrium/beta_t"]),
            betan=read(fid["Equilibrium/beta_N"]), kappa=read(fid["Equilibrium/kappa"]), li1=read(fid["Equilibrium/l_i_1"]))
    end
end

"""
    plot_qprofile(src; show_singular=true, save_path=nothing)

Plot the safety factor q(ψ) profile, with optional vertical markers at each rational surface
and horizontal reference lines at q0 and q95.

### Arguments

  - `src`: A `PlasmaEquilibrium`, a `ForceFreeStatesResult`, or the path of a `gpec.h5`

### Keyword arguments

  - `show_singular`: If `true`, overlay rational surface locations (default: `true`)
  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_qprofile(src::Source; show_singular=true, save_path=nothing)
    _require(plot_qprofile, src)
    p = _plot_qprofile(_profile_data(src); show_singular)
    isnothing(save_path) || savefig(p, save_path)
    return p
end

function _plot_qprofile(d; show_singular=true)
    xs, q, q0, q95 = d.psi, d.q, d.q0, d.q95
    p = plot(
        xs, q;
        xlabel="Norm. Poloidal Flux",
        ylabel="q",
        title="",
        legend=false,
        xlims=(0, 1),
        left_margin=10Plots.mm,
        bottom_margin=5Plots.mm,
        right_margin=8Plots.mm  # annotations at x=1.0 need room on the right
    )
    hline!(p, [q0, q95]; linestyle=:dot, color=:gray, label=nothing)
    annotate!(p, 1.0, q0, text(" q0=$(round(q0, digits=2))", 7, :left, :gray))
    annotate!(p, 1.0, q95, text(" q95=$(round(q95, digits=2))", 7, :left, :gray))

    if show_singular
        for (psi_s, q_s) in zip(d.sing_psi, d.sing_q)
            vline!(p, [psi_s]; linestyle=:dash, color=:red, label=nothing)
            annotate!(p, psi_s, q_s,
                text("  q=$(round(q_s, digits=2))", 7, :left, :red))
        end
    end
    return p
end

"""
    plot_pressure_profile(src; save_path=nothing)

Plot the μ₀p(ψ) pressure profile. Vertical dashed lines mark rational surfaces if present.

### Arguments

  - `src`: A `PlasmaEquilibrium`, a `ForceFreeStatesResult`, or the path of a `gpec.h5`

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_pressure_profile(src::Source; save_path=nothing)
    _require(plot_pressure_profile, src)
    p = _plot_pressure_profile(_profile_data(src))
    isnothing(save_path) || savefig(p, save_path)
    return p
end

function _plot_pressure_profile(d)
    p = plot(
        d.psi, d.mu0p;
        xlabel="Norm. Poloidal Flux",
        ylabel="μ₀p",
        title="",
        legend=false,
        xlims=(0, 1),
        left_margin=10Plots.mm,
        bottom_margin=5Plots.mm
    )
    for psi_s in d.sing_psi
        vline!(p, [psi_s]; linestyle=:dash, color=:red, label=nothing)
    end
    return p
end

"""
    plot_f_profile(src; save_path=nothing)

Plot the toroidal field function 2πF(ψ) profile (F = RBφ/(2π)).

### Arguments

  - `src`: A `PlasmaEquilibrium`, a `ForceFreeStatesResult`, or the path of a `gpec.h5`

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_f_profile(src::Source; save_path=nothing)
    _require(plot_f_profile, src)
    p = _plot_f_profile(_profile_data(src))
    isnothing(save_path) || savefig(p, save_path)
    return p
end

function _plot_f_profile(d)
    p = plot(
        d.psi, d.twopif;
        xlabel="Norm. Poloidal Flux",
        ylabel="2πF",
        title="",
        legend=false,
        xlims=(0, 1),
        left_margin=10Plots.mm,
        bottom_margin=5Plots.mm
    )
    for psi_s in d.sing_psi
        vline!(p, [psi_s]; linestyle=:dash, color=:red, label=nothing)
    end
    return p
end

"""
    plot_flux_surfaces(src; n_psi=11, n_theta=18, save_path=nothing)

Plot flux surface contours (constant ψ, blue) and field-line angle spokes (constant θ, red)
in physical (R, Z) space, from the nodal grid data.

Psi contours are drawn at `n_psi` evenly spaced values between psilow and psihigh.
Theta spokes are drawn at `n_theta` evenly spaced values.

### Arguments

  - `src`: A `PlasmaEquilibrium`, a `ForceFreeStatesResult`, or the path of a `gpec.h5`

### Keyword arguments

  - `n_psi`: Number of constant-ψ contours to draw (default: 11)
  - `n_theta`: Number of constant-θ/θ spokes to draw (default: 18)
  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_flux_surfaces(src::Source; n_psi=11, n_theta=18, save_path=nothing)
    _require(plot_flux_surfaces, src)
    p = _plot_flux_surfaces(_geometry_data(src); n_psi, n_theta)
    isnothing(save_path) || savefig(p, save_path)
    return p
end

function _plot_flux_surfaces(d; n_psi=11, n_theta=18)
    xs_rz, ys_rz, psi_sing, q_sing = d.psi, d.theta, d.sing_psi, d.sing_q

    n_psi_grid = length(xs_rz)
    n_theta_grid = length(ys_rz)
    R_grid, Z_grid = _rz_grid(d)

    p = plot(;
        title="Flux surfaces in (R, Z)",
        xlabel="R [m]",
        ylabel="Z [m]",
        aspect_ratio=:equal,
        legend=:outertopright,
        left_margin=10Plots.mm,
        bottom_margin=5Plots.mm
    )

    psi_indices = round.(Int, range(1, n_psi_grid; length=n_psi))
    theta_indices = round.(Int, range(1, n_theta_grid; length=n_theta))

    for (i, ipsi) in enumerate(psi_indices)
        label = i == 1 ? "Constant ψ" : ""
        plot!(p, [R_grid[ipsi, :]; R_grid[ipsi, 1]], [Z_grid[ipsi, :]; Z_grid[ipsi, 1]];
            color=:steelblue, linewidth=1.5, label=label)
    end
    for (i, itheta) in enumerate(theta_indices)
        label = i == 1 ? "Constant θ" : ""
        plot!(p, R_grid[:, itheta], Z_grid[:, itheta];
            color=:tomato, linewidth=0.8, label=label)
    end
    # Draw rational surface flux contours in red, annotated at the bottom (min Z) of each
    for s in eachindex(psi_sing)
        idx = argmin(abs.(xs_rz .- psi_sing[s]))
        q_label = abs(q_sing[s] - round(q_sing[s])) < 0.05 ?
                  "q=$(round(Int, q_sing[s]))" : "q=$(round(q_sing[s], digits=2))"
        plot!(p, [R_grid[idx, :]; R_grid[idx, 1]], [Z_grid[idx, :]; Z_grid[idx, 1]];
            color=:red, linewidth=1.5,
            label=s == 1 ? "Rational surface" : "")
        itheta_bot = argmin(Z_grid[idx, :])
        annotate!(p, R_grid[idx, itheta_bot], Z_grid[idx, itheta_bot],
            text(" $q_label", 7, :left, :red))
    end
    return p
end

"""
    plot_gse_by_theta(h5path; n_theta_lines=8, save_path=nothing)

Plot Grad-Shafranov error vs ψ_N (log scale) for several θ slices, with the
flux-surface-integrated error overplotted as a thick black line. Reads `gse.h5`
(and optionally `gsei.h5`) from the same directory as `h5path`, written when the equilibrium
configuration sets `diagnose_src = true`.

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file (used to locate `gse.h5`)

### Keyword arguments

  - `n_theta_lines`: Number of evenly-spaced θ slices to overlay (default: 8)
  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_gse_by_theta(h5path::AbstractString; n_theta_lines=8, save_path=nothing)
    _require(plot_gse_by_theta, h5path)
    gse_path = joinpath(dirname(h5path), "gse.h5")

    gse_data = h5open(gse_path, "r") do fid
        read(fid["gse_data"])  # (npsi, ntheta, 7): cols = θ, ψ, flux1, flux2, source, total, error
    end

    _, ntheta, _ = size(gse_data)
    theta_indices = round.(Int, range(1, ntheta; length=n_theta_lines))

    p = plot(;
        xlabel="Norm. Poloidal Flux",
        ylabel="GSE error",
        title="",
        yscale=:log10,
        xlims=(0, 1),
        left_margin=10Plots.mm,
        bottom_margin=5Plots.mm
    )
    for (i, itheta) in enumerate(theta_indices)
        psi_col = gse_data[:, itheta, 2]
        err_col = gse_data[:, itheta, 7]
        plot!(p, psi_col, err_col; label=i == 1 ? "Const. θ" : "", alpha=0.6)
    end

    # Overplot flux-surface-integrated error as thick reference line
    gsei_path = joinpath(dirname(h5path), "gsei.h5")
    if isfile(gsei_path)
        xs_int, errori = h5open(gsei_path, "r") do fid
            read(fid["xs"]), read(fid["errori"])
        end
        plot!(p, xs_int, vec(errori);
            color=:black, linewidth=2.5, linestyle=:solid, label="integrated")
    end

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_gse_integrated(h5path; save_path=nothing)

Plot the flux-surface-integrated Grad-Shafranov error (log scale) vs ψ_N, reading from
`gsei.h5` in the same directory as `h5path`, written when the equilibrium configuration sets
`diagnose_src = true`.

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file (used to locate `gsei.h5`)

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_gse_integrated(h5path::AbstractString; save_path=nothing)
    _require(plot_gse_integrated, h5path)
    gsei_path = joinpath(dirname(h5path), "gsei.h5")

    xs, errlogi = h5open(gsei_path, "r") do fid
        read(fid["xs"]), read(fid["errlogi"])
    end

    p = plot(
        xs, vec(errlogi);
        xlabel="Norm. Poloidal Flux",
        ylabel="log₁₀(integrated GSE)",
        title="Flux-surface-integrated Grad-Shafranov error",
        legend=false,
        left_margin=10Plots.mm,
        bottom_margin=5Plots.mm
    )

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_equilibrium_summary(src; save_path=nothing)

Summary of equilibrium profiles and geometry. The (R, Z) flux surface plot occupies the
full left column. Profile plots are stacked in the right column:

  - q(ψ) safety factor with rational surface markers (`plot_qprofile`)
  - μ₀p(ψ) pressure profile (`plot_pressure_profile`)
  - 2πF(ψ) toroidal field function (`plot_f_profile`)

When the source is a `gpec.h5` path with `gse.h5` beside it (requires `diagnose_src = true`), a combined
Grad-Shafranov error panel (θ slices + integrated, log scale) is appended at the bottom of the right column.

### Arguments

  - `src`: A `PlasmaEquilibrium`, a `ForceFreeStatesResult`, or the path of a `gpec.h5`

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_equilibrium_summary(src::Source; save_path=nothing)
    _require(plot_equilibrium_summary, src)
    (; q0, q95, betat, betan, kappa, li1) = _global_data(src)

    title_str = "q0=$(round(q0,digits=2))  q95=$(round(q95,digits=2))  βₜ=$(round(betat,digits=3))  βₙ=$(round(betan,digits=3))  κ=$(round(kappa,digits=2))  li1=$(round(li1,digits=3))"

    profiles = _profile_data(src)
    p_rz = _plot_flux_surfaces(_geometry_data(src))
    p_q = _plot_qprofile(profiles; show_singular=true)
    p_pres = _plot_pressure_profile(profiles)
    p_f = _plot_f_profile(profiles)
    p_gse = src isa AbstractString && _can(plot_gse_by_theta, capabilities(src)) ? plot_gse_by_theta(src) : nothing  # includes integrated overplot

    # Suppress x-axis labels/ticks on all but the bottom profile panel — they share the
    # same ψ_N axis and labeling every panel wastes vertical space.
    hide_xaxis!(p) = plot!(p; xlabel="", xformatter=_ -> "", bottom_margin=1Plots.mm)

    if isnothing(p_gse)
        hide_xaxis!(p_q)
        hide_xaxis!(p_pres)
        l = @layout [a{0.38w} [b; c; d]]
        p = plot(p_rz, p_q, p_pres, p_f; layout=l, size=(1300, 750),
            plot_title=title_str, plot_titlefontsize=10, top_margin=8Plots.mm)
    else
        hide_xaxis!(p_q)
        hide_xaxis!(p_pres)
        hide_xaxis!(p_f)
        l = @layout [a{0.38w} [b; c; d; e]]
        p = plot(p_rz, p_q, p_pres, p_f, p_gse; layout=l, size=(1300, 1000),
            plot_title=title_str, plot_titlefontsize=10, top_margin=8Plots.mm)
    end

    isnothing(save_path) || savefig(p, save_path)
    return p
end

register!(plot_equilibrium_summary, (PlasmaEquilibrium, ForceFreeStatesResult, AbstractString), (:equilibrium,), "flux surfaces and profiles")
register!(plot_qprofile, (PlasmaEquilibrium, ForceFreeStatesResult, AbstractString), (:equilibrium,), "q against ψ_N with the rational surfaces")
register!(plot_pressure_profile, (PlasmaEquilibrium, ForceFreeStatesResult, AbstractString), (:equilibrium,), "μ₀p against ψ_N")
register!(plot_f_profile, (PlasmaEquilibrium, ForceFreeStatesResult, AbstractString), (:equilibrium,), "2πF against ψ_N")
register!(plot_flux_surfaces, (PlasmaEquilibrium, ForceFreeStatesResult, AbstractString), (:equilibrium,), "flux surfaces and θ spokes in (R, Z)")
register!(plot_gse_by_theta, (AbstractString,), (:gse_files,), "Grad-Shafranov error against ψ_N per θ slice")
register!(plot_gse_integrated, (AbstractString,), (:gse_files,), "flux-surface-integrated Grad-Shafranov error")

end # module Equilibrium
