"""
    Equilibrium

Post-processing and visualization functions for GPEC equilibrium objects and HDF5 outputs.
"""
module Equilibrium

using HDF5
using Plots

"""
    plot_flux_surfaces(plasma_eq; n_psi=11, n_theta=13)

Plot flux surface contours (constant ψ, blue) and field-line angle contours (constant θ, red)
in physical (R, Z) space.

### Arguments

  - `plasma_eq`: A `GeneralizedPerturbedEquilibrium.Equilibrium.PlasmaEquilibrium` object

### Keyword arguments

  - `n_psi`: Number of constant-ψ contours to draw (default: 11)
  - `n_theta`: Number of constant-θ contours to draw (default: 13)

### Returns

A `Plots.jl` plot object.
"""
function plot_flux_surfaces(plasma_eq; n_psi=11, n_theta=13)
    n_psi_grid = size(plasma_eq.rzphi_rsquared.nodal_derivs.partials, 2)
    n_theta_grid = size(plasma_eq.rzphi_rsquared.nodal_derivs.partials, 3)

    # Build R and Z on the full nodal grid
    R_grid = Matrix{Float64}(undef, n_psi_grid, n_theta_grid)
    Z_grid = Matrix{Float64}(undef, n_psi_grid, n_theta_grid)
    for ipsi in 1:n_psi_grid
        rfac = @. sqrt(max(0.0, plasma_eq.rzphi_rsquared.nodal_derivs.partials[1, ipsi, :]))
        angle = @. 2π * (plasma_eq.rzphi_ys + plasma_eq.rzphi_offset.nodal_derivs.partials[1, ipsi, :])
        R_grid[ipsi, :] = plasma_eq.ro .+ rfac .* cos.(angle)
        Z_grid[ipsi, :] = plasma_eq.zo .+ rfac .* sin.(angle)
    end

    p = plot(;
        title="Flux Coordinate System Contours in (R, Z)",
        xlabel="R [m]",
        ylabel="Z [m]",
        aspect_ratio=:equal,
        legend=:outertopright
    )

    psi_indices = round.(Int, range(1, n_psi_grid; length=n_psi))
    theta_indices = round.(Int, range(1, n_theta_grid; length=n_theta))

    for (i, ipsi) in enumerate(psi_indices)
        label = i == 1 ? "Constant ψ" : ""
        plot!(p, [R_grid[ipsi, :]; R_grid[ipsi, 1]], [Z_grid[ipsi, :]; Z_grid[ipsi, 1]];
            color=:blue, linewidth=1.5, label=label)
    end

    for (i, itheta) in enumerate(theta_indices)
        label = i == 1 ? "Constant θ" : ""
        plot!(p, R_grid[:, itheta], Z_grid[:, itheta];
            color=:red, linewidth=1.0, label=label)
    end

    return p
end

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
        xlabel="ψ_N",
        ylabel="q",
        title="Safety factor q(ψ)",
        legend=false
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
    xs, mu0p, betat, betap1, msing, psi_sing = h5open(h5path, "r") do fid
        read(fid["splines/profiles/xs"]), read(fid["splines/profiles/mu0p"]),
        read(fid["equil/betat"]), read(fid["equil/betap1"]),
        read(fid["singular/msing"]), read(fid["singular/psi"])
    end

    p = plot(
        xs, mu0p;
        xlabel="ψ_N",
        ylabel="μ₀p",
        title="Pressure profile (βₜ=$(round(betat, digits=3)), βₚ=$(round(betap1, digits=3)))",
        legend=false
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
    xs, twopif = h5open(h5path, "r") do fid
        read(fid["splines/profiles/xs"]), read(fid["splines/profiles/2piF"])
    end

    p = plot(
        xs, twopif;
        xlabel="ψ_N",
        ylabel="2πF",
        title="Toroidal field function 2πF(ψ)",
        legend=false
    )

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_equilibrium_summary(h5path; save_path=nothing)

Three-panel summary of equilibrium profiles:

  - q(ψ) safety factor with rational surface markers (`plot_qprofile`)
  - μ₀p(ψ) pressure profile (`plot_pressure_profile`)
  - 2πF(ψ) toroidal field function (`plot_f_profile`)

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

    p_q    = plot_qprofile(h5path; show_singular=true)
    p_pres = plot_pressure_profile(h5path)
    p_f    = plot_f_profile(h5path)

    p = plot(p_q, p_pres, p_f; layout=(1, 3), size=(1200, 400), plot_title=title_str)

    isnothing(save_path) || savefig(p, save_path)
    return p
end

end # module Equilibrium
