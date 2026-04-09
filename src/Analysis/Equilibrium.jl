"""
    Equilibrium

Post-processing and visualization functions for GPEC equilibrium objects and HDF5 outputs.
"""
module Equilibrium

using HDF5
using LaTeXStrings
using Plots

"""
    plot_qprofile(h5path; show_singular=true, save_path=nothing)

Plot the safety factor q(ψ) profile, with optional vertical markers at each rational surface
and horizontal reference lines at q0 and q95.

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file

### Keyword arguments

  - `show_singular`: If `true`, overlay rational surface locations (default: `true`)
  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_qprofile(h5path; show_singular=true, save_path=nothing)
    xs, q, q0, q95 = h5open(h5path, "r") do fid
        read(fid["splines/profiles/xs"]), read(fid["splines/profiles/q"]),
        read(fid["equil/q0"]), read(fid["equil/q95"])
    end

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
        msing, psi_sing, q_sing = h5open(h5path, "r") do fid
            read(fid["singular/msing"]), read(fid["singular/psi"]), read(fid["singular/q"])
        end
        for s in 1:msing
            vline!(p, [psi_sing[s]]; linestyle=:dash, color=:red, label=nothing)
            annotate!(p, psi_sing[s], q_sing[s],
                text("  q=$(round(q_sing[s], digits=2))", 7, :left, :red))
        end
    end

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_pressure_profile(h5path; save_path=nothing)

Plot the μ₀p(ψ) pressure profile. Vertical dashed lines mark rational surfaces if present.

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_pressure_profile(h5path; save_path=nothing)
    xs, mu0p, msing, psi_sing = h5open(h5path, "r") do fid
        read(fid["splines/profiles/xs"]), read(fid["splines/profiles/mu0p"]),
        read(fid["singular/msing"]), read(fid["singular/psi"])
    end

    p = plot(
        xs, mu0p;
        xlabel="Norm. Poloidal Flux",
        ylabel="μ₀p",
        title="",
        legend=false,
        xlims=(0, 1),
        left_margin=10Plots.mm,
        bottom_margin=5Plots.mm
    )
    for s in 1:msing
        vline!(p, [psi_sing[s]]; linestyle=:dash, color=:red, label=nothing)
    end

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_f_profile(h5path; save_path=nothing)

Plot the toroidal field function 2πF(ψ) profile (F = RBφ/(2π)).

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_f_profile(h5path; save_path=nothing)
    xs, twopif, msing, psi_sing = h5open(h5path, "r") do fid
        read(fid["splines/profiles/xs"]), read(fid["splines/profiles/2piF"]),
        read(fid["singular/msing"]), read(fid["singular/psi"])
    end

    p = plot(
        xs, twopif;
        xlabel="Norm. Poloidal Flux",
        ylabel="2πF",
        title="",
        legend=false,
        xlims=(0, 1),
        left_margin=10Plots.mm,
        bottom_margin=5Plots.mm
    )
    for s in 1:msing
        vline!(p, [psi_sing[s]]; linestyle=:dash, color=:red, label=nothing)
    end

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_flux_surfaces(h5path; n_psi=11, n_theta=18, save_path=nothing)

Plot flux surface contours (constant ψ, blue) and field-line angle spokes (constant θ, red)
in physical (R, Z) space, reading nodal grid data directly from HDF5.

Psi contours are drawn at `n_psi` evenly spaced values between psilow and psihigh.
Theta spokes are drawn at `n_theta` evenly spaced values.

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file

### Keyword arguments

  - `n_psi`: Number of constant-ψ contours to draw (default: 11)
  - `n_theta`: Number of constant-θ/θ spokes to draw (default: 18)
  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_flux_surfaces(h5path; n_psi=11, n_theta=18, save_path=nothing)
    rcoords, offset_data, xs_rz, ys_rz, ro, zo, msing, psi_sing, q_sing = h5open(h5path, "r") do fid
        read(fid["splines/rzphi/rcoords"]), read(fid["splines/rzphi/offset"]),
        read(fid["splines/rzphi/xs"]), read(fid["splines/rzphi/ys"]),
        read(fid["equil/ro"]), read(fid["equil/zo"]),
        read(fid["singular/msing"]), read(fid["singular/psi"]), read(fid["singular/q"])
    end

    n_psi_grid = length(xs_rz)
    n_theta_grid = length(ys_rz)

    # Build R, Z on the full nodal grid
    R_grid = Matrix{Float64}(undef, n_psi_grid, n_theta_grid)
    Z_grid = Matrix{Float64}(undef, n_psi_grid, n_theta_grid)
    for ipsi in 1:n_psi_grid
        for itheta in 1:n_theta_grid
            rfac = sqrt(max(0.0, rcoords[ipsi, itheta]))
            eta = 2π * (ys_rz[itheta] + offset_data[ipsi, itheta])
            R_grid[ipsi, itheta] = ro + rfac * cos(eta)
            Z_grid[ipsi, itheta] = zo + rfac * sin(eta)
        end
    end

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
    for s in 1:msing
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

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_gse_by_theta(h5path; n_theta_lines=8, save_path=nothing)

Plot Grad-Shafranov error vs ψ_N (log scale) for several θ slices, with the
flux-surface-integrated error overplotted as a thick black line. Reads `gse.h5`
(and optionally `gsei.h5`) from the same directory as `h5path`. Returns `nothing`
if `gse.h5` is not found (requires `diagnose_src = true` in the equilibrium configuration).

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file (used to locate `gse.h5`)

### Keyword arguments

  - `n_theta_lines`: Number of evenly-spaced θ slices to overlay (default: 8)
  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object, or `nothing` if `gse.h5` is absent.
"""
function plot_gse_by_theta(h5path; n_theta_lines=8, save_path=nothing)
    gse_path = joinpath(dirname(h5path), "gse.h5")
    isfile(gse_path) || return nothing

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
`gsei.h5` in the same directory as `h5path`. Returns `nothing` if `gsei.h5` is not found
(requires `diagnose_src = true` in the equilibrium configuration).

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file (used to locate `gsei.h5`)

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object, or `nothing` if `gsei.h5` is absent.
"""
function plot_gse_integrated(h5path; save_path=nothing)
    gsei_path = joinpath(dirname(h5path), "gsei.h5")
    isfile(gsei_path) || return nothing

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
    plot_equilibrium_summary(h5path; save_path=nothing)

Summary of equilibrium profiles and geometry. The (R, Z) flux surface plot occupies the
full left column. Profile plots are stacked in the right column:

  - q(ψ) safety factor with rational surface markers (`plot_qprofile`)
  - μ₀p(ψ) pressure profile (`plot_pressure_profile`)
  - 2πF(ψ) toroidal field function (`plot_f_profile`)

If `gse.h5` is present (requires `diagnose_src = true`), a combined Grad-Shafranov error
panel (θ slices + integrated, log scale) is appended at the bottom of the right column.

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_equilibrium_summary(h5path; save_path=nothing)
    q0, q95, betat, betan, kappa, li1 = h5open(h5path, "r") do fid
        read(fid["equil/q0"]), read(fid["equil/q95"]),
        read(fid["equil/betat"]), read(fid["equil/betan"]),
        read(fid["equil/kappa"]), read(fid["equil/li1"])
    end

    title_str = "q0=$(round(q0,digits=2))  q95=$(round(q95,digits=2))  βₜ=$(round(betat,digits=3))  βₙ=$(round(betan,digits=3))  κ=$(round(kappa,digits=2))  li1=$(round(li1,digits=3))"

    p_rz   = plot_flux_surfaces(h5path)
    p_q    = plot_qprofile(h5path; show_singular=true)
    p_pres = plot_pressure_profile(h5path)
    p_f    = plot_f_profile(h5path)
    p_gse  = plot_gse_by_theta(h5path)  # includes integrated overplot; nothing if absent

    # Suppress x-axis labels/ticks on all but the bottom profile panel — they share the
    # same ψ_N axis and labeling every panel wastes vertical space.
    hide_xaxis!(p) = plot!(p; xlabel="", xformatter=_->"", bottom_margin=1Plots.mm)

    if isnothing(p_gse)
        hide_xaxis!(p_q); hide_xaxis!(p_pres)
        l = @layout [a{0.38w} [b; c; d]]
        p = plot(p_rz, p_q, p_pres, p_f; layout=l, size=(1300, 750),
            plot_title=title_str, plot_titlefontsize=10, top_margin=8Plots.mm)
    else
        hide_xaxis!(p_q); hide_xaxis!(p_pres); hide_xaxis!(p_f)
        l = @layout [a{0.38w} [b; c; d; e]]
        p = plot(p_rz, p_q, p_pres, p_f, p_gse; layout=l, size=(1300, 1000),
            plot_title=title_str, plot_titlefontsize=10, top_margin=8Plots.mm)
    end

    isnothing(save_path) || savefig(p, save_path)
    return p
end

end # module Equilibrium
