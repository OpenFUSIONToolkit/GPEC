using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using GeneralizedPerturbedEquilibrium
using LinearAlgebra
using Statistics
using GLMakie

const FT = GeneralizedPerturbedEquilibrium.ForcingTerms
const compute_biot_savart_boundary! = FT.compute_biot_savart_boundary!
const read_coil_dat = FT.read_coil_dat

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Each entry: (filepath, current_per_conductor [A])
COIL_FILES = [
    (joinpath(@__DIR__, "sparc_pf1u.dat"), 1000.0),
    (joinpath(@__DIR__, "sparc_pf3u.dat"), 1000.0),
]

# Plasma torus parameters
R0 = 1.85
a  = .57

# Torus grid resolution
N_THETA = 120
N_PHI   = 180

# ── Field component selection ────────────────────────────
# Choose which component to display on ALL plots:
#   :B_R, :B_phi, :B_Z, :B_mag
FIELD_COMPONENT = :B_R

# ── Plot toggles ─────────────────────────────────────────
ENABLE_TORUS_PLOT       = true    # Fig 1: field on torus surface
ENABLE_HALL_PROBES      = true    # Fig 2: total field at probe grid
ENABLE_HALL_PERTURBATION = true   # Fig 3: non-axisymmetric (n≠0) only

# ── Hall probe grid settings ─────────────────────────────
HALL_MARGIN = 0.5
HALL_N_R    = 8
HALL_N_PHI  = 24
HALL_N_Z    = 8

# Output
PLOT_FILE = joinpath(@__DIR__, "b_3d_colorplot.png")

# ═══════════════════════════════════════════════════════════════
# Field component helpers
# ═══════════════════════════════════════════════════════════════

const FIELD_LABELS = Dict(
    :B_R   => ("Bᵣ",  "Bᵣ [T]"),
    :B_phi => ("Bφ",  "Bφ [T]"),
    :B_Z   => ("Bz",  "Bz [T]"),
    :B_mag => ("|B|", "|B| [T]"),
)

const PERT_LABELS = Dict(
    :B_R   => ("δBᵣ",  "δBᵣ (n≠0) [T]"),
    :B_phi => ("δBφ",  "δBφ (n≠0) [T]"),
    :B_Z   => ("δBz",  "δBz (n≠0) [T]"),
    :B_mag => ("|δB|", "|δB| (n≠0) [T]"),
)

function field_is_signed(component::Symbol)
    return component in (:B_R, :B_phi, :B_Z)
end

function select_component(B_R, B_phi, B_Z, component::Symbol)
    if component == :B_R;      return B_R
    elseif component == :B_phi; return B_phi
    elseif component == :B_Z;   return B_Z
    else                        return sqrt.(B_R.^2 .+ B_phi.^2 .+ B_Z.^2)
    end
end

function symmetric_clim(data; q=0.98)
    c = quantile(abs.(vec(data)), q)
    return max(c, 1e-30)
end

# ═══════════════════════════════════════════════════════════════
# Core functions
# ═══════════════════════════════════════════════════════════════

function load_all_coil_sets(coil_file_specs)
    coil_sets = FT.CoilSet[]
    for (filepath, current) in coil_file_specs
        isfile(filepath) || error("Coil file not found: $filepath")
        println("  Loading $filepath ...")
        cs = read_coil_dat(filepath)
        cs.currents .= current
        push!(coil_sets, cs)
        R_all = sqrt.(cs.x.^2 .+ cs.y.^2)
        println("    ncoil=$(cs.ncoil), s=$(cs.s), nsec=$(cs.nsec), nw=$(cs.nw)")
        println("    R ∈ [$(round(minimum(R_all),digits=3)), $(round(maximum(R_all),digits=3))]")
        println("    Z ∈ [$(round(minimum(cs.z),digits=3)), $(round(maximum(cs.z),digits=3))]")
        println("    I = $current A per conductor")
    end
    return coil_sets
end

function compute_field_on_torus(coil_sets, R0, a, n_theta, n_phi)
    theta_range = range(0, 2π, length=n_theta)
    phi_range   = range(0, 2π, length=n_phi)

    obs_R   = vec([(R0 + a*cos(θ)) for φ in phi_range, θ in theta_range])
    obs_phi = vec([φ               for φ in phi_range, θ in theta_range])
    obs_Z   = vec([a*sin(θ)        for φ in phi_range, θ in theta_range])

    nobs = length(obs_R)
    B_R   = zeros(nobs)
    B_phi = zeros(nobs)
    B_Z   = zeros(nobs)

    println("Computing Biot-Savart on torus ($nobs points, $(length(coil_sets)) coil sets)...")
    compute_biot_savart_boundary!(B_R, B_phi, B_Z, obs_R, obs_phi, obs_Z, coil_sets)

    println("  B_R   range: [$(minimum(B_R)), $(maximum(B_R))]")
    println("  B_phi range: [$(minimum(B_phi)), $(maximum(B_phi))]")
    println("  B_Z   range: [$(minimum(B_Z)), $(maximum(B_Z))]")

    C = reshape(select_component(B_R, B_phi, B_Z, FIELD_COMPONENT), n_phi, n_theta)
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
    println("    R   ∈ [$(round(R_min_probe,digits=3)), $(round(R_max_probe,digits=3))] m  ($n_r points)")
    println("    φ   ∈ [0, 2π)  ($n_phi points)")
    println("    Z   ∈ [$(round(Z_min_probe,digits=3)), $(round(Z_max_probe,digits=3))] m  ($n_z points)")

    nobs = n_r * n_phi * n_z
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
    for iz in 1:n_z
        for ir in 1:n_r
            n0 = mean(field_3d[ir, :, iz])
            field_3d[ir, :, iz] .-= n0
        end
    end
    println("    Perturbation range: [$(round(minimum(field_3d), sigdigits=4)), $(round(maximum(field_3d), sigdigits=4))]")
    return vec(field_3d)
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
    signed = field_is_signed(component) || is_perturbation

    if signed
        clim_hi = symmetric_clim(color_data)
        clim_lo = -clim_hi
        cmap = :RdBu
    else
        clim_lo = 0.0
        clim_hi = max(quantile(color_data, 0.98), 1e-30)
        cmap = :inferno
    end

    labels = is_perturbation ? PERT_LABELS : FIELD_LABELS
    _, cb_label = labels[component]

    hp = scatter!(ax, px, py, pz;
                  color=color_data, colormap=cmap,
                  colorrange=(clim_lo, clim_hi),
                  markersize=12)
    Colorbar(fig[1, 2], hp; label=cb_label, width=20)

    return clim_lo, clim_hi
end

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

function main()
    short_name, cb_label = FIELD_LABELS[FIELD_COMPONENT]
    println("Field component: $short_name ($FIELD_COMPONENT)")
    println("Loading coil sets...")
    coil_sets = load_all_coil_sets(COIL_FILES)

    println("\nTorus: R0 = $R0 m, a = $a m")
    X_torus, Y_torus, Z_torus, C_torus = compute_field_on_torus(coil_sets, R0, a, N_THETA, N_PHI)

    hall_data = nothing
    if ENABLE_HALL_PROBES || ENABLE_HALL_PERTURBATION
        println("\nComputing synthetic Hall probe measurements...")
        coil_bbox = compute_coil_bounding_box(coil_sets)
        hall_data = compute_hall_probes(
            coil_sets, coil_bbox, HALL_MARGIN,
            HALL_N_R, HALL_N_PHI, HALL_N_Z
        )
    end

    println("\nCreating 3D plots...")

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
        fig1, ax1 = make_3d_figure(title="$short_name on Torus Surface")

        if field_is_signed(FIELD_COMPONENT)
            clim = symmetric_clim(C_torus)
            sf = surface!(ax1, X_torus, Y_torus, Z_torus;
                          color=C_torus, colormap=:RdBu, colorrange=(-clim, clim))
        else
            clim_hi = max(quantile(vec(C_torus), 0.98), 1e-30)
            sf = surface!(ax1, X_torus, Y_torus, Z_torus;
                          color=C_torus, colormap=:inferno, colorrange=(0.0, clim_hi))
        end
        Colorbar(fig1[1, 2], sf; label=cb_label, width=20)

        plot_coils!(ax1, coil_sets, set_palettes)
        cam3d!(ax1.scene; lookat=cam_lookat, eyeposition=cam_eye, upvector=cam_up)
        push!(figures, ("torus", fig1))
    end

    # ── Figure 2: Hall probes — total field ──────────────
    if ENABLE_HALL_PROBES && hall_data !== nothing
        probe_color = select_component(hall_data.B_R, hall_data.B_phi, hall_data.B_Z, FIELD_COMPONENT)

        fig2, ax2 = make_3d_figure(title="Synthetic Hall Probes — Total $short_name")

        lo, hi = make_hall_scatter!(fig2, ax2,
            hall_data.probe_x, hall_data.probe_y, hall_data.probe_z,
            probe_color, FIELD_COMPONENT, false)

        plot_coils!(ax2, coil_sets, set_palettes)
        cam3d!(ax2.scene; lookat=cam_lookat, eyeposition=cam_eye, upvector=cam_up)

        println("  Hall total: $(length(probe_color)) probes, $short_name ∈ [$lo, $hi]")
        push!(figures, ("hall_total", fig2))
    end

    # ── Figure 3: Hall probes — non-axisymmetric ─────────
    if ENABLE_HALL_PERTURBATION && hall_data !== nothing
        println("\n  Subtracting axisymmetric (n=0) component...")

        B_R_pert   = subtract_axisymmetric(hall_data.B_R,   hall_data.n_r, hall_data.n_phi, hall_data.n_z)
        B_phi_pert = subtract_axisymmetric(hall_data.B_phi, hall_data.n_r, hall_data.n_phi, hall_data.n_z)
        B_Z_pert   = subtract_axisymmetric(hall_data.B_Z,   hall_data.n_r, hall_data.n_phi, hall_data.n_z)

        pert_color = select_component(B_R_pert, B_phi_pert, B_Z_pert, FIELD_COMPONENT)

        pert_short, _ = PERT_LABELS[FIELD_COMPONENT]

        fig3, ax3 = make_3d_figure(title="Synthetic Hall Probes — Non-Axisymmetric $pert_short")

        lo, hi = make_hall_scatter!(fig3, ax3,
            hall_data.probe_x, hall_data.probe_y, hall_data.probe_z,
            pert_color, FIELD_COMPONENT, true)

        plot_coils!(ax3, coil_sets, set_palettes)
        cam3d!(ax3.scene; lookat=cam_lookat, eyeposition=cam_eye, upvector=cam_up)

        println("  Hall perturbation: $pert_short ∈ [$lo, $hi]")
        push!(figures, ("hall_perturbation", fig3))
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