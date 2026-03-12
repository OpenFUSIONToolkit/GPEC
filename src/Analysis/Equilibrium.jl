"""
    Equilibrium

Post-processing and visualization functions for GPEC equilibrium objects.
"""
module Equilibrium

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

end # module Equilibrium
