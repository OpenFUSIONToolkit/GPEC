using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using GeneralizedPerturbedEquilibrium
using LinearAlgebra
using Statistics
using GLMakie

# ═══════════════════════════════════════════════════════════════
# Global theme — large, bold, serif fonts
# ═══════════════════════════════════════════════════════════════
set_theme!(Theme(
    fontsize = 36,
    font     = :bold,
    Axis = (
        xlabelsize     = 32,
        ylabelsize     = 32,
        xticklabelsize = 24,
        yticklabelsize = 24,
        titlesize      = 36,
        xlabelfont     = "CMU Serif Bold",
        ylabelfont     = "CMU Serif Bold",
        titlefont      = "CMU Serif Bold",
        xticklabelfont = "CMU Serif",
        yticklabelfont = "CMU Serif",
    ),
    Colorbar = (
        labelsize     = 28,
        ticklabelsize = 22,
        labelfont     = "CMU Serif Bold",
        ticklabelfont = "CMU Serif",
    ),
    Label = (
        fontsize = 36,
        font     = "CMU Serif Bold",
    ),
))

const FT = GeneralizedPerturbedEquilibrium.ForcingTerms
const compute_biot_savart_boundary! = FT.compute_biot_savart_boundary!
const read_coil_dat                 = FT.read_coil_dat

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
COIL_FILES = [
    (joinpath(@__DIR__, "sparc_pf1u.dat"), 100.0),
]

R0 = 1.85
a  = 0.57

N_THETA = 120
N_PHI   = 180

# ── Field component selection ────────────────────────────
# :B_R, :B_phi, :B_Z, :B_mag
FIELD_COMPONENT = :B_R

# ── Units ────────────────────────────────────────────────
# :T or :G  (1 T = 10000 G)
FIELD_UNITS = :G

# ── Plot toggles ─────────────────────────────────────────
ENABLE_TORUS_PLOT        = true
ENABLE_HALL_PROBES       = true
ENABLE_HALL_PERTURBATION = true
ENABLE_Z_SLICES          = true

# ── Hall probe grid settings ─────────────────────────────
HALL_MARGIN = 0.5
HALL_N_R    = 8
HALL_N_PHI  = 24
HALL_N_Z    = 8

# ── Perturbation visibility threshold ────────────────────
PERT_VISIBILITY_THRESHOLD = 0.005

# ── Z-slice settings ─────────────────────────────────────
# :auto → one slice through the Z-center of each coil set
# or provide explicit list of Z values, e.g. [0.0, 0.5, -0.3]
Z_SLICE_POSITIONS    = :auto
Z_SLICE_N_XY         = 200
Z_SLICE_MARGIN       = 0.5
Z_SLICE_SHOW_COILS   = true    # overplot coil geometry on Z-slices
Z_SLICE_COIL_PROJECT = true    # true  = project full coil onto XY plane
                                # false = only show points near the Z slice
Z_SLICE_COIL_TOL_FRAC = 0.05  # only used when Z_SLICE_COIL_PROJECT = false

# ── Hall probe Z-slice settings ──────────────────────────
ENABLE_HALL_Z_SLICES      = true    # 2D scatter of hall probes at Z slices
HALL_Z_SLICE_POSITIONS    = :auto   # :auto or explicit [z1, z2, ...]
HALL_Z_SLICE_TOL          = 0.15    # [m] half-thickness of slab to capture probes
HALL_Z_SLICE_SHOW_COILS   = true
HALL_Z_SLICE_COIL_PROJECT = true
HALL_Z_SLICE_CENTERING    = true   # if true, calculate center of Bfield and mark on plot also print difference from 0,0

# Output
PLOT_FILE = joinpath(@__DIR__, "b_3d_colorplot.png")

# ═══════════════════════════════════════════════════════════════
# Units
# ═══════════════════════════════════════════════════════════════
const T_TO_G = 1e4

unit_scale() = FIELD_UNITS == :G ? T_TO_G : 1.0
unit_str()   = FIELD_UNITS == :G ? "G"     : "T"

# ═══════════════════════════════════════════════════════════════
# Field component helpers
# ═══════════════════════════════════════════════════════════════
function field_label(c::Symbol)
    Dict(:B_R => "Bᵣ", :B_phi => "Bφ", :B_Z => "Bz", :B_mag => "|B|")[c]
end

function pert_label(c::Symbol)
    Dict(:B_R => "δBᵣ", :B_phi => "δBφ", :B_Z => "δBz", :B_mag => "|δB|")[c]
end

field_cb_label(c::Symbol)  = "$(field_label(c)) [$(unit_str())]"
pert_cb_label(c::Symbol)   = "$(pert_label(c)) (n≠0) [$(unit_str())]"
field_is_signed(c::Symbol) = c in (:B_R, :B_phi, :B_Z)

function select_component(B_R, B_phi, B_Z, c::Symbol)
    c == :B_R   && return B_R
    c == :B_phi && return B_phi
    c == :B_Z   && return B_Z
    return sqrt.(B_R.^2 .+ B_phi.^2 .+ B_Z.^2)
end

function symmetric_clim(data; q=0.98)
    max(quantile(abs.(vec(data)), q), 1e-30)
end

# ═══════════════════════════════════════════════════════════════
# Core functions
# ═══════════════════════════════════════════════════════════════
function load_all_coil_sets(coil_file_specs)
    coil_sets = FT.CoilSet[]
    for (filepath, current) in coil_file_specs
        isfile(filepath) || error("Coil file not found: $filepath")
        println("Loading $filepath ...")
        cs = read_coil_dat(filepath)
        cs.currents .= current
        push!(coil_sets, cs)
        R_all = sqrt.(cs.x.^2 .+ cs.y.^2)
        println("  ncoil=$(cs.ncoil), s=$(cs.s), nsec=$(cs.nsec), nw=$(cs.nw)")
        println("  R ∈ [$(round(minimum(R_all), digits=3)), $(round(maximum(R_all), digits=3))]")
        println("  Z ∈ [$(round(minimum(cs.z), digits=3)), $(round(maximum(cs.z), digits=3))]")
        println("  I = $current A per conductor")
    end
    return coil_sets
end

function compute_field_on_torus(coil_sets, R0, a, n_theta, n_phi)
    theta_range = range(0, 2π, length=n_theta)
    phi_range   = range(0, 2π, length=n_phi)

    obs_R   = vec([(R0 + a*cos(θ)) for φ in phi_range, θ in theta_range])
    obs_phi = vec([φ               for φ in phi_range, θ in theta_range])
    obs_Z   = vec([a*sin(θ)        for φ in phi_range, θ in theta_range])

    nobs  = length(obs_R)
    B_R   = zeros(nobs)
    B_phi = zeros(nobs)
    B_Z   = zeros(nobs)

    println("Computing Biot-Savart on torus ($nobs points, $(length(coil_sets)) coil sets)...")
    compute_biot_savart_boundary!(B_R, B_phi, B_Z, obs_R, obs_phi, obs_Z, coil_sets)

    s = unit_scale()
    C = reshape(select_component(B_R, B_phi, B_Z, FIELD_COMPONENT) .* s, n_phi, n_theta)
    X = [(R0 + a*cos(θ))*cos(φ) for φ in phi_range, θ in theta_range]
    Y = [(R0 + a*cos(θ))*sin(φ) for φ in phi_range, θ in theta_range]
    Z = [a*sin(θ)               for φ in phi_range, θ in theta_range]

    return X, Y, Z, C
end

function compute_coil_bounding_box(coil_sets)
    R_min =  Inf; R_max = -Inf
    Z_min =  Inf; Z_max = -Inf
    for cs in coil_sets
        R_all = sqrt.(cs.x.^2 .+ cs.y.^2)
        R_min = min(R_min, minimum(R_all))
        R_max = max(R_max, maximum(R_all))
        Z_min = min(Z_min, minimum(cs.z))
        Z_max = max(Z_max, maximum(cs.z))
    end
    return R_min, R_max, Z_min, Z_max
end

function coil_set_z_centers(coil_sets)
    [(minimum(cs.z) + maximum(cs.z)) / 2.0 for cs in coil_sets]
end

function compute_hall_probes(coil_sets, coil_bbox, margin, n_r, n_phi, n_z)
    R_min_coil, R_max_coil, Z_min_coil, Z_max_coil = coil_bbox

    R_min_probe = max(0.05, R_min_coil - margin)
    R_max_probe = R_max_coil + margin
    Z_min_probe = Z_min_coil - margin
    Z_max_probe = Z_max_coil + margin

    R_range   = range(R_min_probe, R_max_probe, length=n_r)
    phi_range = range(0, 2π, length=n_phi + 1)[1:end-1]
    Z_range   = range(Z_min_probe, Z_max_probe, length=n_z)

    println("\n  Hall probe grid:")
    println("    R   ∈ [$(round(R_min_probe, digits=3)), $(round(R_max_probe, digits=3))] m  ($n_r points)")
    println("    φ   ∈ [0, 2π)  ($n_phi points)")
    println("    Z   ∈ [$(round(Z_min_probe, digits=3)), $(round(Z_max_probe, digits=3))] m  ($n_z points)")

    nobs    = n_r * n_phi * n_z
    obs_R   = zeros(nobs)
    obs_phi = zeros(nobs)
    obs_Z   = zeros(nobs)

    idx = 1
    for iz in 1:n_z, ip in 1:n_phi, ir in 1:n_r
        obs_R[idx]   = R_range[ir]
        obs_phi[idx] = phi_range[ip]
        obs_Z[idx]   = Z_range[iz]
        idx += 1
    end

    B_R   = zeros(nobs)
    B_phi = zeros(nobs)
    B_Z   = zeros(nobs)

    println("    Computing Biot-Savart at $nobs probe locations...")
    compute_biot_savart_boundary!(B_R, B_phi, B_Z, obs_R, obs_phi, obs_Z, coil_sets)

    B_mag = sqrt.(B_R.^2 .+ B_phi.^2 .+ B_Z.^2)

    probe_x = obs_R .* cos.(obs_phi)
    probe_y = obs_R .* sin.(obs_phi)
    probe_z = obs_Z

    println("    |B| range at probes: [$(round(minimum(B_mag), sigdigits=4)), $(round(maximum(B_mag), sigdigits=4))] T")

    return (probe_x=probe_x, probe_y=probe_y, probe_z=probe_z,
            B_R=B_R, B_phi=B_phi, B_Z=B_Z, B_mag=B_mag,
            n_r=n_r, n_phi=n_phi, n_z=n_z)
end

function subtract_axisymmetric(field_flat::Vector{Float64}, n_r::Int, n_phi::Int, n_z::Int)
    field_3d = reshape(copy(field_flat), n_r, n_phi, n_z)
    for iz in 1:n_z, ir in 1:n_r
        n0 = mean(field_3d[ir, :, iz])
        field_3d[ir, :, iz] .-= n0
    end
    println("    Perturbation range: [$(round(minimum(field_3d), sigdigits=4)), $(round(maximum(field_3d), sigdigits=4))]")
    return vec(field_3d)
end

# ═══════════════════════════════════════════════════════════════
# Z-slice computation
# ═══════════════════════════════════════════════════════════════
function compute_z_slice_xy(coil_sets, z_val, coil_bbox, margin, n_xy)
    _, R_max_coil, _, _ = coil_bbox
    extent = R_max_coil + margin

    x_range = range(-extent, extent, length=n_xy)
    y_range = range(-extent, extent, length=n_xy)

    R_min_safe = 0.02

    obs_R   = Float64[]
    obs_phi = Float64[]
    obs_Z   = Float64[]
    valid_idx = Int[]

    for (iy, yv) in enumerate(y_range)
        for (ix, xv) in enumerate(x_range)
            R = sqrt(xv^2 + yv^2)
            if R >= R_min_safe
                push!(obs_R, R)
                push!(obs_phi, atan(yv, xv))
                push!(obs_Z, z_val)
                push!(valid_idx, (iy - 1) * n_xy + ix)
            end
        end
    end

    nobs  = length(obs_R)
    B_R   = zeros(nobs)
    B_phi = zeros(nobs)
    B_Z   = zeros(nobs)

    println("    Computing Biot-Savart at $nobs points (Z = $(round(z_val, digits=4)) m)...")
    compute_biot_savart_boundary!(B_R, B_phi, B_Z, obs_R, obs_phi, obs_Z, coil_sets)

    field_vals = select_component(B_R, B_phi, B_Z, FIELD_COMPONENT) .* unit_scale()

    field_2d = fill(NaN, n_xy, n_xy)
    for (k, li) in enumerate(valid_idx)
        field_2d[li] = field_vals[k]
    end

    return collect(x_range), collect(y_range), field_2d
    
end

# ═══════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════
function plot_coils!(ax, coil_sets, set_palettes)
    total = 0
    for (s_idx, cs) in enumerate(coil_sets)
        palette = set_palettes[mod1(s_idx, length(set_palettes))]
        for j in 1:cs.ncoil
            cc = palette[mod1(j, length(palette))]
            for k in 1:cs.s
                xl = vec(cs.x[j, k, :])
                yl = vec(cs.y[j, k, :])
                zl = vec(cs.z[j, k, :])
                if (xl[1]-xl[end])^2 + (yl[1]-yl[end])^2 + (zl[1]-zl[end])^2 > 1e-10
                    push!(xl, xl[1]); push!(yl, yl[1]); push!(zl, zl[1])
                end
                lines!(ax, xl, yl, zl; color=cc, linewidth=4)
                total += 1
            end
        end
    end
    return total
end

"""
    plot_coils_xy_at_z!(ax, coil_sets, z_val, set_palettes;
                        project=true, tol=0.05)

Overlay coil geometry on a 2D (X, Y) axis.

If `project=true`: draw the full coil outline projected onto the XY plane
(ignoring Z), so the complete coil footprint is always visible regardless
of whether the coil passes through the slice plane.

If `project=false`: only show coil points within `tol` of `z_val`.
"""
function plot_coils_xy_at_z!(ax, coil_sets, z_val, set_palettes;
                              project::Bool=true, tol=0.05)
    for (s_idx, cs) in enumerate(coil_sets)
        palette = set_palettes[mod1(s_idx, length(set_palettes))]
        for j in 1:cs.ncoil
            cc = palette[mod1(j, length(palette))]
            for k in 1:cs.s
                xl = vec(cs.x[j, k, :])
                yl = vec(cs.y[j, k, :])
                zl = vec(cs.z[j, k, :])

                if project
                    # Close the loop
                    if (xl[1]-xl[end])^2 + (yl[1]-yl[end])^2 > 1e-10
                        push!(xl, xl[1])
                        push!(yl, yl[1])
                    end
                    # Draw full coil outline projected onto XY
                    lines!(ax, xl, yl; color=cc, linewidth=2.5)
                else
                    mask = abs.(zl .- z_val) .< tol
                    any(mask) || continue
                    scatter!(ax, xl[mask], yl[mask];
                             color=cc, markersize=6,
                             strokecolor=:black, strokewidth=0.5)
                end
            end
        end
    end
end

function make_3d_figure(; title="")
    fig = Figure(size=(1400, 1100), backgroundcolor=:white)
    if !isempty(title)
        Label(fig[0, 1:2], title; fontsize=24, halign=:center)
        rowsize!(fig.layout, 0, Fixed(30))
    end
    ax = LScene(fig[1, 1]; show_axis=false)
    colsize!(fig.layout, 1, Relative(0.92))
    return fig, ax
end

function make_hall_scatter!(fig, ax, px, py, pz, color_data, component, is_perturbation)
    s = unit_scale()
    color_display = color_data .* s
    signed = field_is_signed(component) || is_perturbation

    if signed
        clim_hi = symmetric_clim(color_display)
        clim_lo = -clim_hi
        cmap = :RdBu
    else
        clim_lo = 0.0
        clim_hi = max(quantile(color_display, 0.98), 1e-30)
        cmap = :inferno
    end

    cb_label = is_perturbation ? pert_cb_label(component) : field_cb_label(component)

    hp = scatter!(ax, px, py, pz;
                  color=color_display, colormap=cmap,
                  colorrange=(clim_lo, clim_hi),
                  markersize=12)
    Colorbar(fig[1, 2], hp; label=cb_label, width=20)

    return clim_lo, clim_hi
end

function make_hall_scatter_filtered!(fig, ax, px, py, pz, color_data, component, threshold_frac)
    s = unit_scale()
    u = unit_str()
    color_display = color_data .* s

    peak      = maximum(abs.(color_display))
    threshold = threshold_frac * peak
    mask      = abs.(color_display) .>= threshold

    n_total   = length(color_display)
    n_visible = count(mask)
    println("    Visibility filter: threshold = $(round(threshold, sigdigits=4)) $u " *
            "($(round(threshold_frac*100, digits=1))% of peak $(round(peak, sigdigits=4)) $u)")
    println("    Showing $n_visible / $n_total probes ($(n_total - n_visible) hidden)")

    cd_f    = color_display[mask]
    clim_hi = symmetric_clim(cd_f)
    clim_lo = -clim_hi

    hp = scatter!(ax, px[mask], py[mask], pz[mask];
                  color=cd_f, colormap=:RdBu,
                  colorrange=(clim_lo, clim_hi),
                  markersize=14)
    Colorbar(fig[1, 2], hp; label=pert_cb_label(component), width=20)

    return clim_lo, clim_hi
end

"""
    plot_hall_z_slice(hall_data, z_val, tol, component,
                      coil_sets, coil_bbox, set_palettes;
                      show_coils, coil_project, coil_tol_frac)

Create a 2D scatter plot of hall probe measurements near Z = z_val ± tol,
projected onto the (X, Y) plane.
"""
function plot_hall_z_slice(hall_data, z_val, tol, component,
                            coil_sets, coil_bbox, set_palettes;
                            show_coils=true, coil_project=true, coil_tol_frac=0.05)
    s = unit_scale()

    # Select probes within the slab
    mask      = abs.(hall_data.probe_z .- z_val) .< tol
    n_in_slab = count(mask)

    px = hall_data.probe_x[mask]
    py = hall_data.probe_y[mask]
    color_raw = select_component(hall_data.B_R[mask], hall_data.B_phi[mask],
                                  hall_data.B_Z[mask], component)
    color_display = color_raw .* s

    println("    Hall Z-slice at Z=$(round(z_val, digits=3)) m ± $(tol) m: $n_in_slab probes")

    fig = Figure(size=(1000, 900), backgroundcolor=:white)

    Label(fig[0, 1:2],
          "Hall Probes: $(field_label(component)) at Z = $(round(z_val, digits=3)) m";
          fontsize=22, halign=:center)
    rowsize!(fig.layout, 0, Fixed(30))

    ax = Axis(fig[1, 1];
              xlabel="X [m]", ylabel="Y [m]",
              aspect=DataAspect())

    if n_in_slab == 0
        println("    WARNING: No probes found in slab — increase HALL_Z_SLICE_TOL or HALL_N_Z")
        return fig
    end

    if field_is_signed(component)
        clim = symmetric_clim(color_display)
        hp = scatter!(ax, px, py;
                      color=color_display, colormap=:RdBu,
                      colorrange=(-clim, clim),
                      markersize=18)
    else
        clim_hi = max(quantile(color_display, 0.98), 1e-30)
        hp = scatter!(ax, px, py;
                      color=color_display, colormap=:inferno,
                      colorrange=(0.0, clim_hi),
                      markersize=18)
    end

    Colorbar(fig[1, 2], hp; label=field_cb_label(component), width=20)

    if show_coils
        z_extent = coil_bbox[4] - coil_bbox[3]
        ct = max(0.05, z_extent * coil_tol_frac)
        plot_coils_xy_at_z!(ax, coil_sets, z_val, set_palettes;
                             project=coil_project, tol=ct)
    end

    return fig
end

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
function main()
    fl = field_label(FIELD_COMPONENT)
    println("Field component: $fl ($(FIELD_COMPONENT)), units: $(unit_str())")
    println("Loading coil sets...")
    coil_sets = load_all_coil_sets(COIL_FILES)

    println("\nTorus: R0 = $R0 m, a = $a m")
    X_torus, Y_torus, Z_torus, C_torus = compute_field_on_torus(coil_sets, R0, a, N_THETA, N_PHI)

    hall_data  = nothing
    coil_bbox  = compute_coil_bounding_box(coil_sets)
    if ENABLE_HALL_PROBES || ENABLE_HALL_PERTURBATION || ENABLE_HALL_Z_SLICES
        println("\nComputing synthetic Hall probe measurements...")
        hall_data = compute_hall_probes(
            coil_sets, coil_bbox, HALL_MARGIN,
            HALL_N_R, HALL_N_PHI, HALL_N_Z
        )
    end

    println("\nCreating plots...")

    set_palettes = [
        [:royalblue, :dodgerblue, :steelblue, :cornflowerblue, :navy, :mediumblue],
        [:crimson, :firebrick, :indianred, :salmon, :darkred, :tomato],
        [:forestgreen, :seagreen, :limegreen, :darkgreen, :olive, :springgreen],
        [:darkorange, :orange, :goldenrod, :gold, :peru, :sandybrown],
        [:purple, :mediumpurple, :blueviolet, :indigo, :orchid, :plum],
        [:deeppink, :hotpink, :mediumvioletred, :palevioletred, :magenta, :pink],
    ]

    cam_eye    = Vec3f(2.5, -3.0, 2.5)
    cam_lookat = Vec3f(0, 0, 0)
    cam_up     = Vec3f(0, 0, 1)

    figures = []

    # ── Figure 1: Torus surface ──────────────────────────
    if ENABLE_TORUS_PLOT
        fig1, ax1 = make_3d_figure(title="$fl on Torus Surface")

        if field_is_signed(FIELD_COMPONENT)
            clim = symmetric_clim(C_torus)
            sf = surface!(ax1, X_torus, Y_torus, Z_torus;
                          color=C_torus, colormap=:RdBu, colorrange=(-clim, clim))
        else
            clim_hi = max(quantile(vec(C_torus), 0.98), 1e-30)
            sf = surface!(ax1, X_torus, Y_torus, Z_torus;
                          color=C_torus, colormap=:inferno, colorrange=(0.0, clim_hi))
        end
        Colorbar(fig1[1, 2], sf; label=field_cb_label(FIELD_COMPONENT), width=20)

        plot_coils!(ax1, coil_sets, set_palettes)
        cam3d!(ax1.scene; lookat=cam_lookat, eyeposition=cam_eye, upvector=cam_up)
        push!(figures, ("torus", fig1))
    end

    # ── Figure 2: Hall probes — total field ──────────────
    if ENABLE_HALL_PROBES && hall_data !== nothing
        probe_color = select_component(hall_data.B_R, hall_data.B_phi, hall_data.B_Z, FIELD_COMPONENT)

        fig2, ax2 = make_3d_figure(title="Synthetic Hall Probes — Total $fl")

        lo, hi = make_hall_scatter!(fig2, ax2,
            hall_data.probe_x, hall_data.probe_y, hall_data.probe_z,
            probe_color, FIELD_COMPONENT, false)

        plot_coils!(ax2, coil_sets, set_palettes)
        cam3d!(ax2.scene; lookat=cam_lookat, eyeposition=cam_eye, upvector=cam_up)

        println("  Hall total: $(length(probe_color)) probes")
        push!(figures, ("hall_total", fig2))
    end

    # ── Figure 3: Hall probes — non-axisymmetric, filtered ──
    if ENABLE_HALL_PERTURBATION && hall_data !== nothing
        println("\n  Subtracting axisymmetric (n=0) component...")

        B_R_pert   = subtract_axisymmetric(hall_data.B_R,   hall_data.n_r, hall_data.n_phi, hall_data.n_z)
        B_phi_pert = subtract_axisymmetric(hall_data.B_phi, hall_data.n_r, hall_data.n_phi, hall_data.n_z)
        B_Z_pert   = subtract_axisymmetric(hall_data.B_Z,   hall_data.n_r, hall_data.n_phi, hall_data.n_z)

        pert_color = select_component(B_R_pert, B_phi_pert, B_Z_pert, FIELD_COMPONENT)
        pl = pert_label(FIELD_COMPONENT)

        fig3, ax3 = make_3d_figure(title="Synthetic Hall Probes — Non-Axisymmetric $pl (filtered)")

        lo, hi = make_hall_scatter_filtered!(fig3, ax3,
            hall_data.probe_x, hall_data.probe_y, hall_data.probe_z,
            pert_color, FIELD_COMPONENT, PERT_VISIBILITY_THRESHOLD)

        plot_coils!(ax3, coil_sets, set_palettes)
        cam3d!(ax3.scene; lookat=cam_lookat, eyeposition=cam_eye, upvector=cam_up)

        println("  Hall perturbation: $pl ∈ [$lo, $hi]")
        push!(figures, ("hall_perturbation", fig3))
    end

    # ── Z-slice 2D plots in (X, Y) ──────────────────────
    if ENABLE_Z_SLICES
        z_positions = if Z_SLICE_POSITIONS == :auto
            unique(round.(coil_set_z_centers(coil_sets), digits=4))
        else
            Float64.(Z_SLICE_POSITIONS)
        end

        println("\n  Computing Z-slices at: $(z_positions) m")

        for (si, z_val) in enumerate(z_positions)
            x_range, y_range, field_2d = compute_z_slice_xy(
                coil_sets, z_val, coil_bbox, Z_SLICE_MARGIN, Z_SLICE_N_XY
            )

            fig_s = Figure(size=(1000, 900), backgroundcolor=:white)

            Label(fig_s[0, 1:2],
                  "$(field_label(FIELD_COMPONENT)) at Z = $(round(z_val, digits=3)) m";
                  fontsize=22, halign=:center)
            rowsize!(fig_s.layout, 0, Fixed(30))

            ax_s = Axis(fig_s[1, 1];
                        xlabel="X [m]", ylabel="Y [m]",
                        aspect=DataAspect())

            valid = filter(!isnan, vec(field_2d))

            if field_is_signed(FIELD_COMPONENT)
                clim = symmetric_clim(valid)
                hm = heatmap!(ax_s, x_range, y_range, field_2d;
                              colormap=:RdBu, colorrange=(-clim, clim),
                              nan_color=:transparent)
            else
                clim_hi = max(quantile(valid, 0.98), 1e-30)
                hm = heatmap!(ax_s, x_range, y_range, field_2d;
                              colormap=:inferno, colorrange=(0.0, clim_hi),
                              nan_color=:transparent)
            end

            Colorbar(fig_s[1, 2], hm; label=field_cb_label(FIELD_COMPONENT), width=20)

            # Overlay coil geometry if enabled
            if Z_SLICE_SHOW_COILS
                z_extent  = coil_bbox[4] - coil_bbox[3]
                slice_tol = max(0.05, z_extent * Z_SLICE_COIL_TOL_FRAC)
                plot_coils_xy_at_z!(ax_s, coil_sets, z_val, set_palettes;
                                     project=Z_SLICE_COIL_PROJECT, tol=slice_tol)
            end

            push!(figures, ("zslice_$(si)", fig_s))
        end
    end

    # ── Hall probe Z-slice 2D plots ──────────────────────
    if ENABLE_HALL_Z_SLICES && hall_data !== nothing
        hz_positions = if HALL_Z_SLICE_POSITIONS == :auto
            unique(round.(coil_set_z_centers(coil_sets), digits=4))
        else
            Float64.(HALL_Z_SLICE_POSITIONS)
        end

        println("\n  Computing Hall probe Z-slices at: $(hz_positions) m")

        for (si, z_val) in enumerate(hz_positions)
            fig_hz = plot_hall_z_slice(
                hall_data, z_val, HALL_Z_SLICE_TOL, FIELD_COMPONENT,
                coil_sets, coil_bbox, set_palettes;
                show_coils=HALL_Z_SLICE_SHOW_COILS,
                coil_project=HALL_Z_SLICE_COIL_PROJECT,
                coil_tol_frac=Z_SLICE_COIL_TOL_FRAC
            )
            push!(figures, ("hall_zslice_$(si)", fig_hz))
        end
    end

    # ── Display & save ───────────────────────────────────
    for (name, fig) in figures
        display(GLMakie.Screen(), fig)
    end

    println("\nPress Enter to save and exit...")
    readline()

    for (name, fig) in figures
        outpath = name == "torus" ? PLOT_FILE : replace(PLOT_FILE, ".png" => "_$(name).png")
        save(outpath, fig)
        println("Saved $name → $outpath")
    end
end

main()