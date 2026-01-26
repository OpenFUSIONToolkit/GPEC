#!/usr/bin/env julia
"""
Visualization script for surface energy geometry
- Plots plasma boundary
- Shows normal vectors (n) as arrows
- Shows curvature vectors (κ) as arrows
- Shows magnetic field magnitude (B) as color
"""

using JPEC
using Plots
using Printf

println("=== Loading equilibrium and computing surface geometry ===")

# Load equilibrium using same approach as Main
equil = JPEC.Equilibrium.setup_equilibrium("equil.toml")

# Parameters
psilim = 0.99  # Close to edge
n_toroidal = 1
n_points = 128  # Points for visualization

println("Computing boundary geometry...")

# Get boundary points
theta = range(0, 2π, length=n_points)
R_vals = zeros(n_points)
Z_vals = zeros(n_points)

for (i, th) in enumerate(theta)
    theta_norm = (th / (2π))  # Normalize to [0, 1]
    
    # Use bicubic spline to get R, Z
    f_val, f_x, f_y = JPEC.Spl.bicube_deriv1!(equil.rzphi, psilim, theta_norm)
    
    rfac = sqrt(f_val[1])
    offset = f_val[2]
    eta = 2π * (theta_norm + offset)
    
    R_vals[i] = equil.ro + rfac * cos(eta)
    Z_vals[i] = equil.zo + rfac * sin(eta)
end

# Compute geometric quantities
normal_R = zeros(n_points)
normal_Z = zeros(n_points)
kappa_R = zeros(n_points)
kappa_Z = zeros(n_points)
B_magnitude = zeros(n_points)
kappa_total = zeros(n_points)

for i in 1:n_points
    th = theta[i]
    theta_norm = th / (2π)
    
    # Get first derivatives using bicubic spline
    f_val, f_x, f_y = JPEC.Spl.bicube_deriv1!(equil.rzphi, psilim, theta_norm)
    
    rfac = sqrt(f_val[1])
    offset = f_val[2]
    eta = 2π * (theta_norm + offset)
    
    R = equil.ro + rfac * cos(eta)
    Z = equil.zo + rfac * sin(eta)
    
    # Get derivatives
    dr2_dtheta = f_y[1]
    doffset_dtheta = f_y[2]
    
    drfac_dtheta = dr2_dtheta / (2.0 * rfac)
    deta_dtheta = 2π * (1.0 + doffset_dtheta)
    
    # Compute second derivatives numerically for curvature
    di = i < n_points ? 1 : -1  # Use forward diff except at last point
    theta_p = theta[i + di]
    theta_norm_p = theta_p / (2π)
    
    f_val_p, _, f_y_p = JPEC.Spl.bicube_deriv1!(equil.rzphi, psilim, theta_norm_p)
    
    rfac_p = sqrt(f_val_p[1])
    offset_p = f_val_p[2]
    eta_p = 2π * (theta_norm_p + offset_p)
    
    dr2_dtheta_p = f_y_p[1]
    doffset_dtheta_p = f_y_p[2]
    
    drfac_dtheta_p = dr2_dtheta_p / (2.0 * rfac_p)
    deta_dtheta_p = 2π * (1.0 + doffset_dtheta_p)
    
    dtheta = theta_p - th
    d2rfac_dtheta2 = (drfac_dtheta_p - drfac_dtheta) / dtheta
    d2eta_dtheta2 = (deta_dtheta_p - deta_dtheta) / dtheta
    
    R_t = drfac_dtheta * cos(eta) - rfac * sin(eta) * deta_dtheta
    Z_t = drfac_dtheta * sin(eta) + rfac * cos(eta) * deta_dtheta
    R_tt = (d2rfac_dtheta2 - rfac * deta_dtheta^2) * cos(eta) - 
           (2 * drfac_dtheta * deta_dtheta + rfac * d2eta_dtheta2) * sin(eta)
    Z_tt = (d2rfac_dtheta2 - rfac * deta_dtheta^2) * sin(eta) + 
           (2 * drfac_dtheta * deta_dtheta + rfac * d2eta_dtheta2) * cos(eta)
    
    # Tangent and normal
    ds = sqrt(R_t^2 + Z_t^2)
    if ds < 1e-10
        continue
    end
    
    # Normal vector (outward pointing)
    normal_R[i] = Z_t / ds
    normal_Z[i] = -R_t / ds
    
    # Poloidal curvature
    ds3 = ds^3
    kappa_p = (R_t * Z_tt - Z_t * R_tt) / ds3
    
    # Toroidal curvature
    kappa_t = -1.0 / R
    
    # Magnetic field components
    sq_vals = JPEC.Spl.spline_eval!(equil.sq, psilim)
    F_psi = sq_vals[1]  # 2π*F
    qa = sq_vals[4]     # q
    
    B_phi = F_psi / (2π * R)
    rfac = sqrt((R - equil.ro)^2 + (Z - equil.zo)^2)
    eps = rfac / equil.ro
    B_theta = eps * B_phi / qa
    B_tot_sq = B_phi^2 + B_theta^2
    B_magnitude[i] = sqrt(B_tot_sq)
    
    # Weighted curvature
    if B_tot_sq > 1e-20
        B_tot = sqrt(B_tot_sq)
        bp_frac = (B_theta / B_tot)^2
        bt_frac = (B_phi / B_tot)^2
        kappa_n = bp_frac * kappa_p + bt_frac * kappa_t
        kappa_total[i] = kappa_n
        
        # Curvature vector (in direction of curvature)
        kappa_R[i] = kappa_n * normal_R[i]
        kappa_Z[i] = kappa_n * normal_Z[i]
    end
end

println("Creating visualization...")

# Create plots
p1 = plot(R_vals, Z_vals, 
    linewidth=2, 
    color=:black, 
    label="Boundary",
    xlabel="R [m]",
    ylabel="Z [m]",
    title="Plasma Boundary and Normal Vectors",
    aspect_ratio=:equal,
    legend=:topleft
)

# Plot normal vectors (subsample for clarity)
skip = 8
for i in 1:skip:n_points
    quiver!(p1, 
        [R_vals[i]], [Z_vals[i]],
        quiver=([normal_R[i]*0.05], [normal_Z[i]*0.05]),
        color=:blue, linewidth=1.5, arrow=true
    )
end

# Add magnetic axis
scatter!(p1, [equil.ro], [equil.zo], 
    marker=:x, markersize=10, color=:red, label="Magnetic Axis")

# Right plot: Curvature vectors and B field
p2 = scatter(R_vals, Z_vals, 
    zcolor=B_magnitude,
    markersize=4,
    markerstrokewidth=0,
    colorbar_title="|B| [T]",
    xlabel="R [m]",
    ylabel="Z [m]",
    title="Curvature Vectors (n·κ) and |B| Field",
    aspect_ratio=:equal,
    legend=false,
    colormap=:viridis
)

# Plot curvature vectors (scaled for visibility)
# Use different colors for positive/negative curvature
kappa_scale = 0.1
for i in 1:skip:n_points
    color = kappa_total[i] > 0 ? :red : :green
    quiver!(p2, 
        [R_vals[i]], [Z_vals[i]],
        quiver=([kappa_R[i]*kappa_scale], [kappa_Z[i]*kappa_scale]),
        color=color, linewidth=1.5, arrow=true
    )
end

# Add magnetic axis
scatter!(p2, [equil.ro], [equil.zo], 
    marker=:x, markersize=10, color=:red)

# Bottom plot: Quantities vs theta
p3 = plot(theta, kappa_total, 
    label="n·κ [m⁻¹]", 
    linewidth=2,
    xlabel="θ [rad]",
    ylabel="Value",
    title="Surface Quantities vs Poloidal Angle",
    legend=:topright
)

plot!(p3, theta, B_magnitude ./ maximum(B_magnitude), 
    label="|B| (normalized)", 
    linewidth=2, 
    linestyle=:dash
)

# Add zero line
hline!(p3, [0.0], color=:black, linestyle=:dot, linewidth=1, label="")

# Combine plots
p_combined = plot(p1, p2, p3, 
    layout=@layout([a b; c{0.4h}]),
    size=(1400, 900)
)

# Statistics
println("\n=== Surface Geometry Statistics ===")
println(@sprintf("  B field range: %.3f - %.3f T", minimum(B_magnitude), maximum(B_magnitude)))
println(@sprintf("  n·κ range: %.3e - %.3e m⁻¹", minimum(kappa_total), maximum(kappa_total)))
println(@sprintf("  Mean n·κ: %.3e m⁻¹", sum(kappa_total)/length(kappa_total)))
println(@sprintf("  Boundary R range: %.3f - %.3f m", minimum(R_vals), maximum(R_vals)))
println(@sprintf("  Boundary Z range: %.3f - %.3f m", minimum(Z_vals), maximum(Z_vals)))

# Check for sign changes
n_positive = count(kappa_total .> 0)
n_negative = count(kappa_total .< 0)
println("\n=== Curvature Sign Distribution ===")
println(@sprintf("  Positive n·κ (destabilizing): %d points (%.1f%%)", 
    n_positive, 100*n_positive/n_points))
println(@sprintf("  Negative n·κ (stabilizing): %d points (%.1f%%)", 
    n_negative, 100*n_negative/n_points))

# Save figure
output_file = "surface_geometry_plot.png"
savefig(p_combined, output_file)
println("\n✓ Plot saved to: $output_file")

display(p_combined)
