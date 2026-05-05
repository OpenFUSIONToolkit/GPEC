using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using LinearAlgebra
using Statistics
using Printf
using GLMakie
using Optim
using CSV
using DataFrames

set_theme!(Theme(
    fontsize = 28, font = :bold,
    Axis = (xlabelsize=32, ylabelsize=32, xticklabelsize=24, yticklabelsize=24,
            titlesize=32, xlabelfont="CMU Serif Bold", ylabelfont="CMU Serif Bold",
            titlefont="CMU Serif Bold", xticklabelfont="CMU Serif", yticklabelfont="CMU Serif"),
    Colorbar = (labelsize=26, ticklabelsize=20,
                labelfont="CMU Serif Bold", ticklabelfont="CMU Serif"),
    Label = (fontsize=32, font="CMU Serif Bold"),
))

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
HALL_DATA_FILE = joinpath(@__DIR__, "hall_probe_rzp_357.csv")

# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════
# Skip comment lines starting with '#'
function load_hall_data(filename::String)
    lines = readlines(filename)
    data_lines = filter(l -> !startswith(strip(l), "#") && !isempty(strip(l)), lines)
    # Rejoin and parse as CSV
    buf = IOBuffer(join(data_lines, "\n"))
    df  = CSV.read(buf, DataFrame)
    return df
end

df = load_hall_data(HALL_DATA_FILE)

# Extract Cartesian and field columns
x    = df.x
y    = df.y
z    = df.z
R    = df.R
phi  = df.phi
Z    = df.Z
B_R  = df.B_R
B_phi = df.B_phi
B_Z  = df.B_Z
Bmag = df.B_mag

# Convert to Cartesian field components
Bx = B_R .* cos.(phi) .- B_phi .* sin.(phi)
By = B_R .* sin.(phi) .+ B_phi .* cos.(phi)
Bz = B_Z

# ═══════════════════════════════════════════════════════════════
# THE 5-DOF MAGNETIC AXIS PROBLEM
#
# Assuming cylindrical symmetry, the magnetic field axis is a
# straight line that may be shifted and tilted relative to the
# measurement (lab) coordinate system.
#
# The 5 degrees of freedom are:
#   1. x0  : x-shift of the magnetic axis at a reference Z plane
#   2. y0  : y-shift of the magnetic axis at a reference Z plane
#   3. Z0  : axial (Z) offset — the reference Z of the magnetic centre
#   4. tx  : tilt of the magnetic axis in the x-Z plane (angle, rad)
#   5. ty  : tilt of the magnetic axis in the y-Z plane (angle, rad)
#
# The magnetic axis in lab Cartesian coordinates is then:
#   X_axis(Z) = x0 + tan(tx)*(Z - Z0)
#   Y_axis(Z) = y0 + tan(ty)*(Z - Z0)
#   Z_axis(Z) = Z   (parameterised by Z)
#
# Physical idea:
#   For a cylindrically symmetric magnet, in coordinates aligned with
#   and centred on the true magnetic axis, B_phi = 0 exactly and
#   B_R(R', Z') depends only on R' and Z' (not phi').
#   Any shift/tilt of the lab frame introduces a phi-dependent B_phi
#   at the probe locations.
#
# Cost function:
#   We transform each probe position into the candidate magnetic-axis
#   frame, recompute the cylindrical components (B_R', B_phi', B_Z'),
#   and minimise the total squared B_phi' (which should vanish for
#   the true axis by symmetry).
# ═══════════════════════════════════════════════════════════════

"""
    transform_to_axis_frame(x, y, z, Bx, By, Bz, params)

Given lab Cartesian probe positions and field vectors, and candidate
axis parameters `params = [x0, y0, Z0, tx, ty]`, compute the
residual B_phi' in the magnetic-axis cylindrical frame.
"""
function compute_Bphi_residual(x, y, z, Bx, By, Bz, params)
    x0, y0, Z0, tx, ty = params

    # Axis direction unit vector (small-angle friendly but exact)
    # The axis points along: (sin(tx), sin(ty), cos(tx)*cos(ty)) ≈ (tx, ty, 1) for small angles
    # We use a proper rotation: first rotate by tx around Y, then by ty around X
    # Axis unit vector in lab frame:
    ax = sin(tx)
    ay = sin(ty) * cos(tx)
    az = cos(tx) * cos(ty)
    e_axis = [ax, ay, az] / norm([ax, ay, az])

    # Reference point on the axis (at Z = Z0)
    p0 = [x0, y0, Z0]

    n = length(x)
    Bphi_sq = 0.0

    for i in 1:n
        # Lab position vector
        r_lab = [x[i], y[i], z[i]]

        # Vector from reference point to probe
        dr = r_lab .- p0

        # Project onto axis to find closest point on axis
        s      = dot(dr, e_axis)          # parameter along axis
        r_perp = dr .- s .* e_axis        # perpendicular vector from axis to probe

        R_prime = norm(r_perp)

        if R_prime < 1e-12
            continue  # probe is on the axis, B_phi = 0 by symmetry
        end

        # Radial unit vector in the plane perpendicular to the axis
        e_R_prime = r_perp ./ R_prime

        # Azimuthal unit vector: e_phi' = e_axis × e_R'
        e_phi_prime = cross(e_axis, e_R_prime)

        # Field vector
        Bvec = [Bx[i], By[i], Bz[i]]

        # B_phi' component
        Bphi_i = dot(Bvec, e_phi_prime)

        Bphi_sq += Bphi_i^2
    end

    return Bphi_sq
end

# ═══════════════════════════════════════════════════════════════
# INITIAL GUESS
# ═══════════════════════════════════════════════════════════════
# x0, y0: field-magnitude-weighted centroid in x,y
w     = Bmag .^ 2
x0_g  = sum(w .* x) / sum(w)
y0_g  = sum(w .* y) / sum(w)
Z0_g  = sum(w .* z) / sum(w)
tx_g  = 0.0
ty_g  = 0.0

p0 = [x0_g, y0_g, Z0_g, tx_g, ty_g]

@printf("Initial guess:\n")
@printf("  x0 = %.6f m\n", p0[1])
@printf("  y0 = %.6f m\n", p0[2])
@printf("  Z0 = %.6f m\n", p0[3])
@printf("  tx = %.6f rad\n", p0[4])
@printf("  ty = %.6f rad\n", p0[5])
@printf("  Initial cost = %.6e T²\n\n",
        compute_Bphi_residual(x, y, z, Bx, By, Bz, p0))

# ═══════════════════════════════════════════════════════════════
# OPTIMISATION
# ═══════════════════════════════════════════════════════════════
cost(p) = compute_Bphi_residual(x, y, z, Bx, By, Bz, p)

result = optimize(
    cost,
    p0,
    NelderMead(),
    Optim.Options(
        x_tol      = 1e-10,
        f_tol      = 1e-16,
        iterations = 100_000,
        show_trace = false,
    )
)

p_opt = Optim.minimizer(result)
x0_opt, y0_opt, Z0_opt, tx_opt, ty_opt = p_opt

@printf("Optimised magnetic axis parameters:\n")
@printf("  x0 (radial shift X) = %+.6f m\n",   x0_opt)
@printf("  y0 (radial shift Y) = %+.6f m\n",   y0_opt)
@printf("  Z0 (axial centre)   = %+.6f m\n",   Z0_opt)
@printf("  tx (tilt in X-Z)    = %+.6f rad  (%+.4f mrad)\n", tx_opt, tx_opt*1e3)
@printf("  ty (tilt in Y-Z)    = %+.6f rad  (%+.4f mrad)\n", ty_opt, ty_opt*1e3)
@printf("  Final cost          = %.6e T²\n\n",  Optim.minimum(result))
@printf("  Converged: %s\n\n", Optim.converged(result) ? "YES" : "NO")

# ═══════════════════════════════════════════════════════════════
# COMPUTE RESIDUAL B_PHI IN EACH FRAME FOR DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════
function get_Bphi_all(x, y, z, Bx, By, Bz, params)
    x0, y0, Z0, tx, ty = params
    ax    = sin(tx)
    ay    = sin(ty) * cos(tx)
    az    = cos(tx) * cos(ty)
    e_axis = [ax, ay, az] / norm([ax, ay, az])
    p0    = [x0, y0, Z0]

    Bphi_arr  = zeros(length(x))
    Rprime_arr = zeros(length(x))

    for i in eachindex(x)
        r_lab  = [x[i], y[i], z[i]]
        dr     = r_lab .- p0
        s      = dot(dr, e_axis)
        r_perp = dr .- s .* e_axis
        R_prime = norm(r_perp)
        Rprime_arr[i] = R_prime
        if R_prime < 1e-12
            continue
        end
        e_R_prime   = r_perp ./ R_prime
        e_phi_prime = cross(e_axis, e_R_prime)
        Bvec        = [Bx[i], By[i], Bz[i]]
        Bphi_arr[i] = dot(Bvec, e_phi_prime)
    end
    return Bphi_arr, Rprime_arr
end

Bphi_lab, R_lab   = get_Bphi_all(x, y, z, Bx, By, Bz, [0.0, 0.0, mean(z), 0.0, 0.0])
Bphi_opt, R_opt   = get_Bphi_all(x, y, z, Bx, By, Bz, p_opt)

@printf("Diagnostics:\n")
@printf("  RMS B_phi (lab frame)       = %.4e T\n", sqrt(mean(Bphi_lab.^2)))
@printf("  RMS B_phi (optimised frame) = %.4e T\n", sqrt(mean(Bphi_opt.^2)))
@printf("  Max |B_phi| (lab frame)     = %.4e T\n", maximum(abs.(Bphi_lab)))
@printf("  Max |B_phi| (opt frame)     = %.4e T\n", maximum(abs.(Bphi_opt)))

# ═══════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════

# --- 1. B_phi before and after, vs probe index ---
fig1 = Figure(resolution=(1400, 600))
ax1a = Axis(fig1[1,1],
    xlabel = "Probe index",
    ylabel = "B_φ  [T]",
    title  = "Azimuthal field residual — lab frame")
ax1b = Axis(fig1[1,2],
    xlabel = "Probe index",
    ylabel = "B_φ  [T]",
    title  = "Azimuthal field residual — magnetic-axis frame")

scatter!(ax1a, 1:length(Bphi_lab), Bphi_lab,
         markersize=4, color=:steelblue)
hlines!(ax1a, [0.0], color=:black, linestyle=:dash, linewidth=1)

scatter!(ax1b, 1:length(Bphi_opt), Bphi_opt,
         markersize=4, color=:tomato)
hlines!(ax1b, [0.0], color=:black, linestyle=:dash, linewidth=1)

display(fig1)
save(joinpath(@__DIR__, "Bphi_residual_comparison.png"), fig1)

# --- 2. B_phi vs phi, coloured by Z plane, after optimisation ---
fig2 = Figure(resolution=(1000, 700))
ax2  = Axis(fig2[1,1],
    xlabel = "φ  [rad]",
    ylabel = "B_φ (magnetic-axis frame)  [T]",
    title  = "B_φ residual vs azimuthal angle after alignment")

Zuniq = sort(unique(round.(Z, digits=4)))
cmap  = cgrad(:viridis, length(Zuniq), categorical=true)

for (k, Zk) in enumerate(Zuniq)
    idx = findall(i -> abs(Z[i] - Zk) < 1e-3, 1:length(Z))
    scatter!(ax2, phi[idx], Bphi_opt[idx],
             label  = @sprintf("Z=%.3f m", Zk),
             color  = cmap[k],
             markersize = 6)
end
axislegend(ax2, position=:rt, labelsize=16)
display(fig2)
save(joinpath(@__DIR__, "Bphi_vs_phi_optimised.png"), fig2)

# --- 3. Axis position as function of Z ---
Z_range = range(minimum(Z) - 0.1, maximum(Z) + 0.1, length=200)
ax_x = x0_opt .+ tan(tx_opt) .* (Z_range .- Z0_opt)
ax_y = y0_opt .+ tan(ty_opt) .* (Z_range .- Z0_opt)

fig3 = Figure(resolution=(1200, 500))
ax3a = Axis(fig3[1,1],
    xlabel = "Z  [m]",
    ylabel = "Axis X position  [m]",
    title  = "Magnetic axis X(Z)")
ax3b = Axis(fig3[1,2],
    xlabel = "Z  [m]",
    ylabel = "Axis Y position  [m]",
    title  = "Magnetic axis Y(Z)")

lines!(ax3a, collect(Z_range), ax_x, color=:steelblue, linewidth=2)
scatter!(ax3a, [Z0_opt], [x0_opt], color=:red, markersize=12,
         label="axis centre")
axislegend(ax3a, position=:lt, labelsize=18)

lines!(ax3b, collect(Z_range), ax_y, color=:darkorange, linewidth=2)
scatter!(ax3b, [Z0_opt], [y0_opt], color=:red, markersize=12,
         label="axis centre")
axislegend(ax3b, position=:lt, labelsize=18)

display(fig3)
save(joinpath(@__DIR__, "magnetic_axis_position.png"), fig3)

# --- 4. Summary table printed ---
@printf("\n══════════════════════════════════════════════\n")
@printf("  MAGNETIC AXIS SUMMARY\n")
@printf("══════════════════════════════════════════════\n")
@printf("  Axis reference point (at Z0):\n")
@printf("    X0 = %+.4f mm\n", x0_opt * 1e3)
@printf("    Y0 = %+.4f mm\n", y0_opt * 1e3)
@printf("    Z0 = %+.4f m\n",  Z0_opt)
@printf("  Axis tilts:\n")
@printf("    θx (tilt in XZ plane) = %+.4f mrad\n", tx_opt * 1e3)
@printf("    θy (tilt in YZ plane) = %+.4f mrad\n", ty_opt * 1e3)
@printf("  Axis direction (lab unit vector):\n")
ax_vec = [sin(tx_opt), sin(ty_opt)*cos(tx_opt), cos(tx_opt)*cos(ty_opt)]
ax_vec /= norm(ax_vec)
@printf("    e = (%.6f, %.6f, %.6f)\n", ax_vec[1], ax_vec[2], ax_vec[3])
@printf("══════════════════════════════════════════════\n")

REFERENCE_COIL_FILES   = [joinpath(@__DIR__, "sparc_pf1u.dat")]
TRANSFORMED_COIL_FILES = [joinpath(@__DIR__, "sparc_pf1u_r357_rzp.dat")]

COIL_CURRENT = 100.0   # A per conductor; must match Biot-Savart generator

name        = "test_357_cyl_nonlin"
OUTPUT_FILE = joinpath(@__DIR__, "$name.txt")
PLOT_FILE   = joinpath(@__DIR__, "$name.png")

# ═══════════════════════════════════════════════════════════════
# CSV reader
# ═══════════════════════════════════════════════════════════════
function read_hall_probe_csv(filepath::String)
    isfile(filepath) || error("Not found: $filepath")
    println("Reading: $filepath")
    lines_raw = readlines(filepath)
    data_lines = String[]; header_line = ""
    for line in lines_raw
        s = strip(line)
        startswith(s, "#") && continue
        if isempty(header_line); header_line = s; else push!(data_lines, s); end
    end
    N = length(data_lines)
    x=zeros(N); y=zeros(N); z=zeros(N)
    R=zeros(N); phi=zeros(N); Z=zeros(N)
    B_R=zeros(N); B_phi=zeros(N); B_Z=zeros(N); B_mag=zeros(N)
    for (i, line) in enumerate(data_lines)
        v = parse.(Float64, split(line, ","))
        x[i]=v[1]; y[i]=v[2]; z[i]=v[3]; R[i]=v[4]; phi[i]=v[5]; Z[i]=v[6]
        B_R[i]=v[7]; B_phi[i]=v[8]; B_Z[i]=v[9]; B_mag[i]=v[10]
    end
    println("  $N probes")
    println("  R ∈ [$(round(minimum(R),digits=4)), $(round(maximum(R),digits=4))] m")
    println("  Z ∈ [$(round(minimum(Z),digits=4)), $(round(maximum(Z),digits=4))] m")
    return (; x, y, z, R, phi, Z, B_R, B_phi, B_Z, B_mag, N)
end

# ═══════════════════════════════════════════════════════════════
# .dat geometry → cylindrical descriptors (benchmark only)
# ═══════════════════════════════════════════════════════════════
function coil_geometry_cyl(coil_files::Vector{String}; label="Coil")
    all_x = Float64[]; all_y = Float64[]; all_z = Float64[]
    for fp in coil_files
        cs = _read_coil_dat(fp)
        append!(all_x, vec(cs.x)); append!(all_y, vec(cs.y)); append!(all_z, vec(cs.z))
    end
    cx = mean(all_x); cy = mean(all_y); cz = mean(all_z)
    R_all = sqrt.(all_x.^2 .+ all_y.^2)
    R_coil = mean(R_all)
    P = hcat(all_x .- cx, all_y .- cy, all_z .- cz)
    _, _, V = svd(P)
    n = V[:, 3]; if n[3] < 0; n = -n; end
    γ_geom = acos(clamp(n[3], -1.0, 1.0))
    φ_tilt_geom = hypot(n[1], n[2]) > 1e-12 ? atan(n[1], -n[2]) : 0.0
    cR = sqrt(cx^2 + cy^2)
    cφ = atan(cy, cx)
    println("  $label:")
    @printf("    Centroid (R, φ, Z) = (%.4f mm, %.3f°, %.4f mm)\n",
            cR*1000, rad2deg(cφ), cz*1000)
    @printf("    R_coil = %.4f mm\n", R_coil*1000)
    @printf("    Tilt γ = %.4f° about φ_tilt = %.3f°\n",
            rad2deg(γ_geom), rad2deg(φ_tilt_geom))
    return (; X0=cx, Y0=cy, Z0=cz, R_coil,
              cR, cφ, γ=γ_geom, φ_tilt=φ_tilt_geom, normal=n)
end

# ═══════════════════════════════════════════════════════════════
# Forward model: ideal circular loop field
# ═══════════════════════════════════════════════════════════════
const μ₀ = 4π * 1e-7

function ellipke(m::Float64)
    m = clamp(m, 0.0, 1.0 - 1e-15)
    a₀ = 1.0
    b₀ = sqrt(1.0 - m)
    s = m
    p = 1.0
    for _ in 1:50
        a₁ = 0.5 * (a₀ + b₀)
        b₁ = sqrt(a₀ * b₀)
        c₁ = 0.5 * (a₀ - b₀)
        s += p * c₁^2 * 2
        p *= 2
        if abs(c₁) < 1e-15 * abs(a₁); a₀, b₀ = a₁, b₁; break; end
        a₀, b₀ = a₁, b₁
    end
    K = π / (2 * a₀)
    E = K * (1 - 0.5 * s)
    return K, E
end

"""
Field of single circular loop, radius a, current I, in x-y plane at origin.
Returns (B_ρ, B_z) at cylindrical (ρ, z).
"""
function loop_field_cyl(ρ::Float64, z::Float64, a::Float64, I::Float64)
    if ρ < 1e-12
        Bz = μ₀ * I * a^2 / (2 * (a^2 + z^2)^1.5)
        return 0.0, Bz
    end
    denom1 = (a + ρ)^2 + z^2
    denom2 = (a - ρ)^2 + z^2
    k2 = 4 * a * ρ / denom1
    K, E = ellipke(k2)
    sd1 = sqrt(denom1)
    Bρ = (μ₀ * I) / (2π) * (z / (ρ * sd1)) *
         (-K + (a^2 + ρ^2 + z^2) / denom2 * E)
    Bz = (μ₀ * I) / (2π) * (1.0 / sd1) *
         (K + (a^2 - ρ^2 - z^2) / denom2 * E)
    return Bρ, Bz
end

function build_R(γ::Float64, φ_tilt::Float64)
    c = cos(γ); s = sin(γ); C = 1 - c
    ax = cos(φ_tilt); ay = sin(φ_tilt); az = 0.0
    return [c + ax*ax*C       ax*ay*C - az*s    ax*az*C + ay*s ;
            ay*ax*C + az*s    c + ay*ay*C       ay*az*C - ax*s ;
            az*ax*C - ay*s    az*ay*C + ax*s    c + az*az*C    ]
end

"""
Field at lab Cartesian observation point of a loop at (X0,Y0,Z0)
with tilt (γ, φ_tilt), radius R_coil, current I.  Returns lab (Bx,By,Bz).
"""
function field_at_lab(x_obs, y_obs, z_obs;
                       X0, Y0, Z0, γ, φ_tilt, R_coil, I)
    Rmat = build_R(γ, φ_tilt)
    dx = x_obs - X0; dy = y_obs - Y0; dz = z_obs - Z0

    # Coil-frame coords = Rᵀ · Δlab
    xc = Rmat[1,1]*dx + Rmat[2,1]*dy + Rmat[3,1]*dz
    yc = Rmat[1,2]*dx + Rmat[2,2]*dy + Rmat[3,2]*dz
    zc = Rmat[1,3]*dx + Rmat[2,3]*dy + Rmat[3,3]*dz

    ρc = sqrt(xc^2 + yc^2)
    Bρc, Bzc = loop_field_cyl(ρc, zc, R_coil, I)
    cosφc = ρc > 1e-12 ? xc/ρc : 1.0
    sinφc = ρc > 1e-12 ? yc/ρc : 0.0
    Bxc = Bρc * cosφc
    Byc = Bρc * sinφc

    Bx = Rmat[1,1]*Bxc + Rmat[1,2]*Byc + Rmat[1,3]*Bzc
    By = Rmat[2,1]*Bxc + Rmat[2,2]*Byc + Rmat[2,3]*Bzc
    Bz = Rmat[3,1]*Bxc + Rmat[3,2]*Byc + Rmat[3,3]*Bzc
    return Bx, By, Bz
end

# ═══════════════════════════════════════════════════════════════
# Predict (B_R, B_φ, B_Z) at every probe
# ═══════════════════════════════════════════════════════════════
function predict_field_cyl(probes, params)
    X0, Y0, Z0, γ, φ_tilt, R_coil = params
    N = probes.N
    bR  = zeros(N); bφ = zeros(N); bZ = zeros(N)
    for i in 1:N
        Bx, By, Bz = field_at_lab(probes.x[i], probes.y[i], probes.z[i];
                                    X0=X0, Y0=Y0, Z0=Z0, γ=γ, φ_tilt=φ_tilt,
                                    R_coil=R_coil, I=COIL_CURRENT)
        cosφ = cos(probes.phi[i]); sinφ = sin(probes.phi[i])
        bR[i] =  Bx*cosφ + By*sinφ
        bφ[i] = -Bx*sinφ + By*cosφ
        bZ[i] =  Bz
    end
    return bR, bφ, bZ
end

# ═══════════════════════════════════════════════════════════════
# Cost & fit
# ═══════════════════════════════════════════════════════════════
function cost(params, probes)
    X0, Y0, Z0, γ, φ_tilt, R_coil = params
    R_coil < 0.05 && return 1e10
    γ < 0 && return 1e10
    γ > π/4 && return 1e10
    abs(X0) > 1.0 && return 1e10
    abs(Y0) > 1.0 && return 1e10
    bR, bφ, bZ = predict_field_cyl(probes, params)
    return sum((bR .- probes.B_R).^2) +
           sum((bφ .- probes.B_phi).^2) +
           sum((bZ .- probes.B_Z).^2)
end

function initial_guess(probes)
    imax = argmax(probes.B_mag)
    R_coil0 = probes.R[imax]
    Z00 = probes.Z[imax]
    return [0.0, 0.0, Z00, 0.0, 0.0, R_coil0]
end

function fit_nonlinear(probes; verbose=true)
    println("\n─── Nonlinear forward fit (X₀, Y₀, Z₀, γ, φ_tilt, R_coil) ───")
    p0 = initial_guess(probes)
    if verbose
        @printf("  Initial: X₀=%.3f mm, Y₀=%.3f mm, Z₀=%.3f mm, γ=%.3f°, φ_tilt=%.2f°, R_coil=%.3f mm\n",
                p0[1]*1000, p0[2]*1000, p0[3]*1000,
                rad2deg(p0[4]), rad2deg(p0[5]), p0[6]*1000)
    end

    f = p -> cost(p, probes)

    res1 = optimize(f, p0, NelderMead(), Optim.Options(iterations=4000, g_tol=1e-14))
    p1 = Optim.minimizer(res1)
    println("  Stage 1 cost = $(round(Optim.minimum(res1), sigdigits=4))")

    res2 = optimize(f, p1, NelderMead(), Optim.Options(iterations=8000, g_tol=1e-16))
    p2 = Optim.minimizer(res2)
    println("  Stage 2 cost = $(round(Optim.minimum(res2), sigdigits=4))")

    X0, Y0, Z0, γ, φ_tilt, R_coil = p2
    cR = sqrt(X0^2 + Y0^2); cφ = atan(Y0, X0)

    bR_p, bφ_p, bZ_p = predict_field_cyl(probes, p2)
    data_n = sqrt(sum(probes.B_R.^2) + sum(probes.B_phi.^2) + sum(probes.B_Z.^2))
    res_n  = sqrt(sum((bR_p .- probes.B_R).^2) +
                  sum((bφ_p .- probes.B_phi).^2) +
                  sum((bZ_p .- probes.B_Z).^2))
    rel_res = res_n / data_n

    println("\n  ── Recovered ──")
    @printf("    Centroid: R = %.4f mm at φ = %.3f°,  Z = %.4f mm\n",
            cR*1000, rad2deg(cφ), Z0*1000)
    @printf("    R_coil    = %.4f mm\n", R_coil*1000)
    @printf("    Tilt γ    = %.4f°  about φ_tilt = %.3f°\n",
            rad2deg(γ), rad2deg(φ_tilt))
    @printf("    Relative residual: %.4f%%\n", rel_res*100)

    return (; X0, Y0, Z0, γ, φ_tilt, R_coil, cR, cφ, residual=rel_res, p_fit=p2)
end

# ═══════════════════════════════════════════════════════════════
# Benchmark print + write
# ═══════════════════════════════════════════════════════════════
function benchmark_print(fit, geo_ref, geo_trans)
    println("\n╔══════════════════════════════════════════════════════════════════╗")
    println("║          RECOVERED 5 DOFs (cylindrical)                          ║")
    println("╠══════════════════════════════════════════════════════════════════╣")
    @printf("║  Centroid: R = %8.3f mm,  φ = %8.3f °,  Z = %8.3f mm     ║\n",
            fit.cR*1000, rad2deg(fit.cφ), fit.Z0*1000)
    @printf("║  Tilt:     γ = %8.4f °,  φ_tilt = %8.3f °                  ║\n",
            rad2deg(fit.γ), rad2deg(fit.φ_tilt))
    @printf("║  R_coil  = %8.3f mm                                          ║\n",
            fit.R_coil*1000)
    println("╚══════════════════════════════════════════════════════════════════╝")

    if geo_trans !== nothing
        println("\n═══ BENCHMARK vs. ground-truth (.dat) ═══\n")
        println("  Parameter       │  True       │  Recovered  │  Error")
        println("  ────────────────┼─────────────┼─────────────┼────────────")
        @printf("  R_centroid [mm] │ %10.3f  │ %10.3f  │ %10.3f\n",
                geo_trans.cR*1000, fit.cR*1000, (fit.cR - geo_trans.cR)*1000)
        @printf("  φ_centroid  [°] │ %10.3f  │ %10.3f  │ %10.3f\n",
                rad2deg(geo_trans.cφ), rad2deg(fit.cφ),
                rad2deg(fit.cφ - geo_trans.cφ))
        @printf("  Z_centroid [mm] │ %10.3f  │ %10.3f  │ %10.3f\n",
                geo_trans.Z0*1000, fit.Z0*1000, (fit.Z0 - geo_trans.Z0)*1000)
        @printf("  γ tilt     [°]  │ %10.4f  │ %10.4f  │ %10.4f\n",
                rad2deg(geo_trans.γ), rad2deg(fit.γ),
                rad2deg(fit.γ - geo_trans.γ))
        @printf("  φ_tilt     [°]  │ %10.3f  │ %10.3f  │ %10.3f\n",
                rad2deg(geo_trans.φ_tilt), rad2deg(fit.φ_tilt),
                rad2deg(fit.φ_tilt - geo_trans.φ_tilt))
        @printf("  R_coil    [mm]  │ %10.3f  │ %10.3f  │ %10.3f\n",
                geo_trans.R_coil*1000, fit.R_coil*1000,
                (fit.R_coil - geo_trans.R_coil)*1000)
    end
end

function write_results(filepath, fit, geo_ref, geo_trans)
    open(filepath, "w") do io
        println(io, "═══════════════════════════════════════════════════════")
        println(io, " 5-DOF Cylindrical Coil Centering — Nonlinear Fit")
        println(io, "═══════════════════════════════════════════════════════\n")
        for (label, geo) in [("Reference", geo_ref), ("Transformed (truth)", geo_trans)]
            geo === nothing && continue
            println(io, "── $label coil ──")
            @printf(io, "  Centroid: R = %.4f mm at φ = %.3f°,  Z = %.4f mm\n",
                    geo.cR*1000, rad2deg(geo.cφ), geo.Z0*1000)
            @printf(io, "  R_coil  = %.4f mm\n", geo.R_coil*1000)
            @printf(io, "  Tilt γ = %.4f° about φ_tilt = %.3f°\n\n",
                    rad2deg(geo.γ), rad2deg(geo.φ_tilt))
        end
        println(io, "── Fit (Hall-data only) ──")
        @printf(io, "  Centroid: R = %.4f mm at φ = %.3f°,  Z = %.4f mm\n",
                fit.cR*1000, rad2deg(fit.cφ), fit.Z0*1000)
        @printf(io, "  R_coil  = %.4f mm\n", fit.R_coil*1000)
        @printf(io, "  Tilt γ = %.4f° about φ_tilt = %.3f°\n",
                rad2deg(fit.γ), rad2deg(fit.φ_tilt))
        @printf(io, "  Relative residual: %.4f%%\n", fit.residual*100)
    end
    println("  Wrote: $filepath")
end

# ═══════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════
function plot_diagnostics(probes, fit, geo_trans)
    figs = []
    bR, bφ, bZ = predict_field_cyl(probes, fit.p_fit)

    fig1 = Figure(size=(1500, 550))
    Label(fig1[0,1:3], "Predicted vs Measured Field"; fontsize=26)
    rowsize!(fig1.layout, 0, Fixed(34))
    for (i, (data, pred, comp)) in enumerate([
            (probes.B_R, bR, "B_R"),
            (probes.B_phi, bφ, "B_φ"),
            (probes.B_Z, bZ, "B_Z")])
        ax = Axis(fig1[1,i]; xlabel="measured [T]", ylabel="predicted [T]",
                  title=comp, aspect=DataAspect())
        scatter!(ax, data, pred; markersize=6, color=:steelblue)
        m = max(maximum(abs.(data)), maximum(abs.(pred)), 1e-12)
        lines!(ax, [-m, m], [-m, m]; color=:red, linewidth=2)
    end
    push!(figs, ("pred_vs_meas", fig1))

    fig2 = Figure(size=(1100, 950))
    Label(fig2[0,1:2], "R-Z probes + fitted coil"; fontsize=26)
    rowsize!(fig2.layout, 0, Fixed(34))
    ax2 = Axis(fig2[1,1]; xlabel="R [m]", ylabel="Z [m]", aspect=DataAspect())
    sc = scatter!(ax2, probes.R, probes.Z;
                  color=probes.B_mag, colormap=:inferno, markersize=10,
                  colorrange=(0.0, max(quantile(probes.B_mag, 0.98), 1e-30)))
    Colorbar(fig2[1,2], sc; label="|B| [T]", width=18)
    scatter!(ax2, [fit.R_coil], [fit.Z0]; color=:lime, marker=:cross, markersize=28,
             strokecolor=:black, strokewidth=2, label="Fit")
    if geo_trans !== nothing
        scatter!(ax2, [geo_trans.R_coil], [geo_trans.Z0]; color=:orange,
                 marker=:utriangle, markersize=24, strokecolor=:black, strokewidth=2,
                 label="True")
    end
    axislegend(ax2; position=:lt, labelsize=14)
    push!(figs, ("rz", fig2))

    fig3 = Figure(size=(1000, 950))
    Label(fig3[0,1], "Top-down (X, Y) centroid"; fontsize=26)
    rowsize!(fig3.layout, 0, Fixed(34))
    ax3 = Axis(fig3[1,1]; xlabel="X [mm]", ylabel="Y [mm]", aspect=DataAspect())
    scatter!(ax3, [fit.X0*1000], [fit.Y0*1000]; color=:lime, marker=:cross,
             markersize=30, strokecolor=:black, strokewidth=2, label="Fit")
    if geo_trans !== nothing
        scatter!(ax3, [geo_trans.X0*1000], [geo_trans.Y0*1000]; color=:orange,
                 marker=:utriangle, markersize=26, strokecolor=:black, strokewidth=2,
                 label="True")
    end
    scatter!(ax3, [0.0], [0.0]; color=:white, marker=:diamond, markersize=20,
             strokecolor=:black, strokewidth=1.5, label="Origin")
    axislegend(ax3; position=:lt, labelsize=14)
    push!(figs, ("xy_top", fig3))

    return figs
end

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
function main()
    println("═══════════════════════════════════════════════════════")
    println(" 5-DOF Cylindrical Coil Centering — Nonlinear Forward Fit")
    println("═══════════════════════════════════════════════════════\n")

    probes = read_hall_probe_csv(HALL_DATA_FILE)

    println("\n── Reference geometry (benchmark only) ──")
    geo_ref = isempty(REFERENCE_COIL_FILES) ? nothing :
              coil_geometry_cyl(REFERENCE_COIL_FILES; label="Reference")

    println("\n── Transformed geometry (ground truth) ──")
    geo_trans = isempty(TRANSFORMED_COIL_FILES) ? nothing :
                coil_geometry_cyl(TRANSFORMED_COIL_FILES; label="Transformed")

    fit = fit_nonlinear(probes)
    benchmark_print(fit, geo_ref, geo_trans)
    write_results(OUTPUT_FILE, fit, geo_ref, geo_trans)

    figs = plot_diagnostics(probes, fit, geo_trans)
    for (name, f) in figs; display(GLMakie.Screen(), f); end

    println("\nPress Enter to save and exit...")
    readline()
    for (name, f) in figs
        outpath = replace(PLOT_FILE, ".png" => "_$(name).png")
        save(outpath, f)
        println("Saved $name → $outpath")
    end
end

main()