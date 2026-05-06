using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using GeneralizedPerturbedEquilibrium
using LinearAlgebra
using Statistics
using GLMakie
using DelimitedFiles
using FFTW

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
    (joinpath(@__DIR__, "sparc_pf1u_axisymmetric.dat"), 100.0),
]

R0 = 1.85
a  = 0.57

N_THETA = 120
N_PHI   = 180

out_name = "pf1_AS_small"

# ── Field component selection ────────────────────────────
# :B_R, :B_phi, :B_Z, :B_mag
FIELD_COMPONENT = :B_phi

# ── Units ────────────────────────────────────────────────
# :T or :G  (1 T = 10000 G)
FIELD_UNITS = :G

# ── Plot toggles ─────────────────────────────────────────
ENABLE_TORUS_PLOT        = true
ENABLE_HALL_PROBES       = true
ENABLE_HALL_PERTURBATION = true
ENABLE_Z_SLICES          = true
ENABLE_FOURIER_ANALYSIS  = true    # NEW: Fourier tilt/center analysis

# ── Hall probe shell settings ─────────────────────────────
# Two concentric cylindrical shells of fixed R, uniform φ, multiple Z
# Inner shell: between axis and coil, high Z resolution
# Outer shell: outside coil, moderate Z resolution
# Total probes: N_PHI_INNER × N_Z_INNER + N_PHI_OUTER × N_Z_OUTER
HALL_MARGIN          = 0.4    # [m] axial margin beyond coil Z extents
HALL_N_PHI_INNER     = 24    # φ points on inner shell
HALL_N_Z_INNER       = 22    # Z levels on inner shell
HALL_N_PHI_OUTER     = 20    # φ points on outer shell
HALL_N_Z_OUTER       = 10    # Z levels on outer shell
HALL_R_INNER_FRAC    = 0.80  # inner shell R = this fraction × R_min_coil
HALL_R_OUTER_FRAC    = 1.20  # outer shell R = this fraction × R_max_coil

# ── Perturbation visibility threshold ────────────────────
PERT_VISIBILITY_THRESHOLD = 0.0005

# ── Z-slice settings ─────────────────────────────────────
Z_SLICE_POSITIONS     = :auto
Z_SLICE_N_XY          = 200
Z_SLICE_MARGIN        = 0.5
Z_SLICE_SHOW_COILS    = true
Z_SLICE_COIL_PROJECT  = false
Z_SLICE_COIL_TOL_FRAC = 0.05

# ── Hall probe Z-slice settings ──────────────────────────
ENABLE_HALL_Z_SLICES      = true
HALL_Z_SLICE_POSITIONS    = :auto
HALL_Z_SLICE_TOL          = 0.15
HALL_Z_SLICE_SHOW_COILS   = false
HALL_Z_SLICE_COIL_PROJECT = false
HALL_Z_SLICE_CENTERING    = true

# ── Hall probe data export ───────────────────────────────
ENABLE_HALL_EXPORT      = true
HALL_EXPORT_FILE        = joinpath(@__DIR__, "hall_probe_$out_name.csv")
HALL_COIL_EXCLUSION_PAD = 0.00

# Output
PLOT_FILE = joinpath(@__DIR__, "b_3d_colorplot_$out_name.png")

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
# Coil-interior exclusion
# ═══════════════════════════════════════════════════════════════

"""
    build_coil_cross_sections(coil_sets)

For each coil winding, compute the (R, Z) bounding box of its
cross-section. Returns a vector of (R_min, R_max, Z_min, Z_max) tuples.
"""
function build_coil_cross_sections(coil_sets)
    sections = NTuple{4, Float64}[]
    for cs in coil_sets
        for j in 1:cs.ncoil
            all_x = Float64[]
            all_y = Float64[]
            all_z = Float64[]
            for k in 1:cs.s
                append!(all_x, vec(cs.x[j, k, :]))
                append!(all_y, vec(cs.y[j, k, :]))
                append!(all_z, vec(cs.z[j, k, :]))
            end
            R_all = sqrt.(all_x.^2 .+ all_y.^2)
            push!(sections, (minimum(R_all), maximum(R_all),
                             minimum(all_z),  maximum(all_z)))
        end
    end
    return sections
end

"""
    is_inside_any_coil(R, Z, coil_sections; pad=0.02)

Returns true if the point (R, Z) lies inside any coil cross-section
bounding box expanded by pad on each side.
"""
function is_inside_any_coil(R::Float64, Z::Float64,
                             coil_sections::Vector{NTuple{4, Float64}};
                             pad::Float64=0.02)
    for (Rlo, Rhi, Zlo, Zhi) in coil_sections
        if (Rlo - pad) <= R <= (Rhi + pad) &&
           (Zlo - pad) <= Z <= (Zhi + pad)
            return true
        end
    end
    return false
end

"""
    filter_probes_outside_coils(hall_data, coil_sets; pad=0.02)

Return index mask of probes that are NOT inside any coil cross-section.
"""
function filter_probes_outside_coils(hall_data, coil_sets; pad::Float64=0.02)
    sections = build_coil_cross_sections(coil_sets)
    probe_R  = hall_data.obs_R
    N        = length(probe_R)
    outside  = BitVector(undef, N)
    for i in 1:N
        outside[i] = !is_inside_any_coil(probe_R[i], hall_data.probe_z[i],
                                          sections; pad=pad)
    end
    n_in  = N - count(outside)
    n_out = count(outside)
    println("    Coil exclusion filter: $n_out probes outside coils, " *
            "$n_in removed (pad=$(pad*1000) mm)")
    return outside
end

# ═══════════════════════════════════════════════════════════════
# Hall probe data export
# ═══════════════════════════════════════════════════════════════

"""
    export_hall_probes(filepath, hall_data, outside_mask)

Save measurable hall probe data to a CSV file.
Columns: x, y, z, R, phi, shell_id, B_R, B_phi, B_Z, B_mag (SI units)
"""
function export_hall_probes(filepath::String, hall_data, outside_mask::BitVector)
    px       = hall_data.probe_x[outside_mask]
    py       = hall_data.probe_y[outside_mask]
    pz       = hall_data.probe_z[outside_mask]
    pR       = hall_data.obs_R[outside_mask]
    pphi     = hall_data.obs_phi[outside_mask]
    pshell   = hall_data.shell_id[outside_mask]
    bR       = hall_data.B_R[outside_mask]
    bphi     = hall_data.B_phi[outside_mask]
    bZ       = hall_data.B_Z[outside_mask]
    bmag     = hall_data.B_mag[outside_mask]

    N = count(outside_mask)

    open(filepath, "w") do io
        println(io, "# Hall probe data — measurable points (outside coil geometry)")
        println(io, "# Coordinates in meters, fields in Tesla")
        println(io, "# Shell 1 = inner (R=$(round(hall_data.R_inner,digits=4)) m), " *
                    "Shell 2 = outer (R=$(round(hall_data.R_outer,digits=4)) m)")
        println(io, "# N_probes = $N")
        println(io, "# Columns: x,y,z,R,phi,shell_id,B_R,B_phi,B_Z,B_mag")
        println(io, "x,y,z,R,phi,shell_id,B_R,B_phi,B_Z,B_mag")
        for i in 1:N
            println(io, "$(px[i]),$(py[i]),$(pz[i]),$(pR[i]),$(pphi[i])," *
                        "$(pshell[i]),$(bR[i]),$(bphi[i]),$(bZ[i]),$(bmag[i])")
        end
    end
    println("    Exported $N measurable probes → $filepath")
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

# ═══════════════════════════════════════════════════════════════
# Hall probe shell layout  (replaces old volumetric grid)
# ═══════════════════════════════════════════════════════════════

"""
    compute_hall_probes_shells(coil_sets, coil_bbox, margin,
                               n_phi_inner, n_z_inner,
                               n_phi_outer, n_z_outer,
                               r_inner_frac, r_outer_frac)

Place sensors on two concentric cylindrical shells of fixed R.
Each shell has uniform φ spacing (DFT-ready) and uniform Z spacing.

Shell 1 (inner): R = r_inner_frac × R_min_coil, wide Z range
Shell 2 (outer): R = r_outer_frac × R_max_coil, narrower Z range

Returns a named tuple compatible with the rest of the pipeline,
with additional fields: obs_R, obs_phi, shell_id, R_inner, R_outer,
Z_inner, Z_outer, n_phi_inner, n_z_inner, n_phi_outer, n_z_outer.
"""
function compute_hall_probes_shells(coil_sets, coil_bbox, margin,
                                    n_phi_inner, n_z_inner,
                                    n_phi_outer, n_z_outer,
                                    r_inner_frac, r_outer_frac)

    R_min_coil, R_max_coil, Z_min_coil, Z_max_coil = coil_bbox

    # ── Shell radii ──────────────────────────────────────
    R_inner = R_min_coil * r_inner_frac
    R_outer = R_max_coil * r_outer_frac

    # ── Z ranges ─────────────────────────────────────────
    # Inner shell: full axial span with margin
    Z_lo_inner = Z_min_coil - margin
    Z_hi_inner = Z_max_coil + margin

    # Outer shell: slightly narrower (field decays faster radially)
    Z_lo_outer = Z_min_coil - margin * 0.5
    Z_hi_outer = Z_max_coil + margin * 0.5

    # ── φ grids: uniform, no repeated endpoint (FFT-ready) ──
    phi_inner = collect(range(0, 2π, length=n_phi_inner+1)[1:end-1])
    phi_outer = collect(range(0, 2π, length=n_phi_outer+1)[1:end-1])

    # ── Z grids: uniform ─────────────────────────────────
    Z_inner = collect(range(Z_lo_inner, Z_hi_inner, length=n_z_inner))
    Z_outer = collect(range(Z_lo_outer, Z_hi_outer, length=n_z_outer))

    n_inner = n_phi_inner * n_z_inner
    n_outer = n_phi_outer * n_z_outer
    n_total = n_inner + n_outer

    println("\n  Hall probe shell layout:")
    println("    Inner shell: R = $(round(R_inner, digits=4)) m  " *
            "($(round(r_inner_frac*100, digits=0))% of R_min_coil=$(round(R_min_coil,digits=3)) m)")
    println("    Inner shell: $(n_phi_inner) φ × $(n_z_inner) Z = $n_inner probes")
    println("    Inner shell: Z ∈ [$(round(Z_lo_inner,digits=3)), $(round(Z_hi_inner,digits=3))] m")
    println("    Outer shell: R = $(round(R_outer, digits=4)) m  " *
            "($(round(r_outer_frac*100, digits=0))% of R_max_coil=$(round(R_max_coil,digits=3)) m)")
    println("    Outer shell: $(n_phi_outer) φ × $(n_z_outer) Z = $n_outer probes")
    println("    Outer shell: Z ∈ [$(round(Z_lo_outer,digits=3)), $(round(Z_hi_outer,digits=3))] m")
    println("    Total probes: $n_total")

    # ── Assemble observation points ───────────────────────
    # Loop order: Z outer, φ inner — so that reshape(field, n_phi, n_z) works
    obs_R    = zeros(n_total)
    obs_phi  = zeros(n_total)
    obs_Z    = zeros(n_total)
    shell_id = zeros(Int, n_total)

    idx = 1
    for iz in 1:n_z_inner, ip in 1:n_phi_inner
        obs_R[idx]    = R_inner
        obs_phi[idx]  = phi_inner[ip]
        obs_Z[idx]    = Z_inner[iz]
        shell_id[idx] = 1
        idx += 1
    end

    for iz in 1:n_z_outer, ip in 1:n_phi_outer
        obs_R[idx]    = R_outer
        obs_phi[idx]  = phi_outer[ip]
        obs_Z[idx]    = Z_outer[iz]
        shell_id[idx] = 2
        idx += 1
    end

    # ── Biot-Savart ───────────────────────────────────────
    B_R   = zeros(n_total)
    B_phi = zeros(n_total)
    B_Z   = zeros(n_total)

    println("    Computing Biot-Savart at $n_total probe locations...")
    compute_biot_savart_boundary!(B_R, B_phi, B_Z, obs_R, obs_phi, obs_Z, coil_sets)

    B_mag = sqrt.(B_R.^2 .+ B_phi.^2 .+ B_Z.^2)

    probe_x = obs_R .* cos.(obs_phi)
    probe_y = obs_R .* sin.(obs_phi)

    println("    |B| range: [$(round(minimum(B_mag), sigdigits=4)), " *
            "$(round(maximum(B_mag), sigdigits=4))] T")

    return (probe_x      = probe_x,
            probe_y      = probe_y,
            probe_z      = obs_Z,
            obs_R        = obs_R,
            obs_phi      = obs_phi,
            B_R          = B_R,
            B_phi        = B_phi,
            B_Z          = B_Z,
            B_mag        = B_mag,
            shell_id     = shell_id,
            n_phi_inner  = n_phi_inner,
            n_z_inner    = n_z_inner,
            n_phi_outer  = n_phi_outer,
            n_z_outer    = n_z_outer,
            R_inner      = R_inner,
            R_outer      = R_outer,
            Z_inner      = Z_inner,
            Z_outer      = Z_outer,
            # Legacy compatibility fields
            n_r          = 1,
            n_phi        = n_phi_inner,
            n_z          = n_z_inner)
end

# ═══════════════════════════════════════════════════════════════
# Axisymmetric subtraction  (operates per shell)
# ═══════════════════════════════════════════════════════════════

"""
    subtract_axisymmetric_shells(hall_data)

For each shell independently, subtract the φ-mean (n=0 component)
at every (R, Z) point. Returns B_R_pert, B_phi_pert, B_Z_pert vectors
of the same length as the hall_data fields.
"""
function subtract_axisymmetric_shells(hall_data)
    n_total   = length(hall_data.B_R)
    B_R_pert   = copy(hall_data.B_R)
    B_phi_pert = copy(hall_data.B_phi)
    B_Z_pert   = copy(hall_data.B_Z)

    for shell in 1:2
        mask = hall_data.shell_id .== shell
        n_phi = shell == 1 ? hall_data.n_phi_inner : hall_data.n_phi_outer
        n_z   = shell == 1 ? hall_data.n_z_inner   : hall_data.n_z_outer

        # reshape: [n_phi × n_z] (phi varies fastest per our loop order)
        bR_3d   = reshape(hall_data.B_R[mask],   n_phi, n_z)
        bphi_3d = reshape(hall_data.B_phi[mask], n_phi, n_z)
        bZ_3d   = reshape(hall_data.B_Z[mask],   n_phi, n_z)

        for iz in 1:n_z
            bR_3d[:,   iz] .-= mean(bR_3d[:,   iz])
            bphi_3d[:, iz] .-= mean(bphi_3d[:, iz])
            bZ_3d[:,   iz] .-= mean(bZ_3d[:,   iz])
        end

        idxs = findall(mask)
        B_R_pert[idxs]   = vec(bR_3d)
        B_phi_pert[idxs] = vec(bphi_3d)
        B_Z_pert[idxs]   = vec(bZ_3d)
    end

    println("    B_R   perturbation range: " *
            "[$(round(minimum(B_R_pert),   sigdigits=4)), $(round(maximum(B_R_pert),   sigdigits=4))] T")
    println("    B_phi perturbation range: " *
            "[$(round(minimum(B_phi_pert), sigdigits=4)), $(round(maximum(B_phi_pert), sigdigits=4))] T")
    println("    B_Z   perturbation range: " *
            "[$(round(minimum(B_Z_pert),   sigdigits=4)), $(round(maximum(B_Z_pert),   sigdigits=4))] T")

    return B_R_pert, B_phi_pert, B_Z_pert
end

# ═══════════════════════════════════════════════════════════════
# Fourier / tilt analysis
# ═══════════════════════════════════════════════════════════════

"""
    fourier_analyze_shell(hall_data; shell=1, component=:B_phi)

For each Z level on the selected shell, compute the DFT of the chosen
field component around the φ ring and extract:
  - A0   : DC (mean) value
  - A1   : 1st harmonic amplitude  → proportional to offset magnitude
  - phi1 : 1st harmonic phase      → points toward offset direction
  - A2   : 2nd harmonic amplitude  → ellipticity / nonlinearity check
  - phi2 : 2nd harmonic phase

Apparent centre (x_app, y_app) at each Z is estimated from A1 alone,
normalised by the mean |B| magnitude at that Z level (not A0 of the
chosen component, which can be near zero for Bphi).

This makes the calibration valid for Bphi (zero DC), Br, and Bz alike.

The calibration used is:
    δ(Z) ≈ (A1(Z) / <|B|(Z)>) × R_shell

where <|B|(Z)> is the mean of B_mag around the ring at height Z.
This is approximate; replace with a Biot-Savart-derived sensitivity
matrix for quantitative precision.
"""
function fourier_analyze_shell(hall_data; shell::Int=1, component::Symbol=:B_phi)

    if shell == 1
        n_phi    = hall_data.n_phi_inner
        n_z      = hall_data.n_z_inner
        Z_levels = hall_data.Z_inner
        R_shell  = hall_data.R_inner
    else
        n_phi    = hall_data.n_phi_outer
        n_z      = hall_data.n_z_outer
        Z_levels = hall_data.Z_outer
        R_shell  = hall_data.R_outer
    end

    mask = hall_data.shell_id .== shell

    field_flat = select_component(
        hall_data.B_R[mask],
        hall_data.B_phi[mask],
        hall_data.B_Z[mask],
        component
    )

    # Also grab B_mag for the same probes — used as normalisation
    Bmag_flat = hall_data.B_mag[mask]

    # Reshape to [n_phi × n_z]
    field_2d = reshape(field_flat, n_phi, n_z)
    Bmag_2d  = reshape(Bmag_flat,  n_phi, n_z)

    A0        = zeros(n_z)
    A1        = zeros(n_z)
    phi1      = zeros(n_z)
    A2        = zeros(n_z)
    phi2      = zeros(n_z)
    Bmag_mean = zeros(n_z)   # mean |B| at each Z ring

    for iz in 1:n_z
        ring   = field_2d[:, iz]
        F      = fft(ring) ./ n_phi

        A0[iz]   =  real(F[1])
        A1[iz]   = 2*abs(F[2])
        phi1[iz] =  angle(F[2])
        A2[iz]   = 2*abs(F[3])
        phi2[iz] =  angle(F[3])

        Bmag_mean[iz] = mean(Bmag_2d[:, iz])
    end

    # ── Apparent centre calibration ───────────────────────
    # Normalise by mean |B|, not by A0 of the selected component.
    # This is safe for Bphi (A0 ≈ 0) and consistent across components.
    # δ ≈ (A1 / <|B|>) × R_shell   [approximate, linear regime]
    safe_Bmag = max.(Bmag_mean, 1e-30)
    frac      = A1 ./ safe_Bmag

    x_app = @. R_shell * frac * cos(phi1)
    y_app = @. R_shell * frac * sin(phi1)

    # ── Diagnostics ───────────────────────────────────────
    println("\n  Fourier analysis — Shell $shell  " *
            "(R = $(round(R_shell, digits=4)) m, component = $component):")
    println("    A0      range : [$(round(minimum(A0),        sigdigits=4)), " *
            "$(round(maximum(A0),        sigdigits=4))] T")
    println("    A1      range : [$(round(minimum(A1),        sigdigits=4)), " *
            "$(round(maximum(A1),        sigdigits=4))] T")
    println("    <|B|>   range : [$(round(minimum(Bmag_mean), sigdigits=4)), " *
            "$(round(maximum(Bmag_mean), sigdigits=4))] T")
    println("    A1/<|B|> max  : $(round(maximum(frac), sigdigits=3))  " *
            "(dimensionless offset fraction)")
    println("    φ₁      range : [$(round(rad2deg(minimum(phi1)), digits=2))°, " *
            "$(round(rad2deg(maximum(phi1)), digits=2))°]")
    println("    A2      range : [$(round(minimum(A2),        sigdigits=4)), " *
            "$(round(maximum(A2),        sigdigits=4))] T")
    println("    A2/A1   max   : $(round(maximum(A2 ./ max.(A1, 1e-30)), sigdigits=3))  " *
            "(should be ≪ 1 for simple tilt)")
    println("    x_app   range : [$(round(minimum(x_app)*1e3, digits=3)), " *
            "$(round(maximum(x_app)*1e3, digits=3))] mm")
    println("    y_app   range : [$(round(minimum(y_app)*1e3, digits=3)), " *
            "$(round(maximum(y_app)*1e3, digits=3))] mm")

    return (Z         = Z_levels,
            A0        = A0,
            A1        = A1,
            phi1      = phi1,
            A2        = A2,
            phi2      = phi2,
            Bmag_mean = Bmag_mean,
            x_app     = x_app,
            y_app     = y_app,
            R_shell   = R_shell,
            shell     = shell,
            component = component)
end

"""
    fit_tilt_axis(fr)

Fit a 3D line through the apparent centres (x_app, y_app, Z) extracted
by fourier_analyze_shell. Uses least-squares linear regression in X(Z)
and Y(Z) independently.

Returns a named tuple with:
  direction      — unit vector along the fitted axis
  x0, y0         — axis position at Z = 0
  tilt_deg       — tilt angle from vertical [degrees]
  tilt_azimuth_deg — azimuthal direction of the tilt [degrees]
  x_fit, y_fit   — fitted x/y values at each Z (for plotting residuals)
"""
function fit_tilt_axis(fr)
    Z  = fr.Z
    xc = fr.x_app
    yc = fr.y_app

    Z_mean  = mean(Z)
    xc_mean = mean(xc)
    yc_mean = mean(yc)

    dZ  = Z .- Z_mean
    SSZ = sum(dZ.^2)

    # Slope of apparent centre vs Z
    dx = sum(dZ .* (xc .- xc_mean)) / SSZ
    dy = sum(dZ .* (yc .- yc_mean)) / SSZ

    # Axis intercept at Z = 0
    x0 = xc_mean - dx * Z_mean
    y0 = yc_mean - dy * Z_mean

    # Fitted values
    x_fit = x0 .+ dx .* Z
    y_fit = y0 .+ dy .* Z

    # Unit direction vector
    dir = normalize([dx, dy, 1.0])

    tilt_deg         = acosd(abs(dir[3]))
    tilt_azimuth_deg = atand(dir[2], dir[1])

    # RMS residual of apparent centres from fitted line
    rms_res = sqrt(mean((xc .- x_fit).^2 .+ (yc .- y_fit).^2))

    println("\n  Tilt axis fit (shell $(fr.shell), component $(fr.component)):")
    println("    Axis direction vector : " *
            "($(round(dir[1], sigdigits=4)), $(round(dir[2], sigdigits=4)), " *
            "$(round(dir[3], sigdigits=4)))")
    println("    Axis at Z = 0         : " *
            "($(round(x0*1e3, digits=3)), $(round(y0*1e3, digits=3))) mm")
    println("    Tilt from vertical    : $(round(tilt_deg, digits=4))°")
    println("    Tilt azimuthal dir    : $(round(tilt_azimuth_deg, digits=2))°")
    println("    RMS residual          : $(round(rms_res*1e3, digits=4)) mm")

    return (direction         = dir,
            x0                = x0,
            y0                = y0,
            tilt_deg          = tilt_deg,
            tilt_azimuth_deg  = tilt_azimuth_deg,
            x_fit             = x_fit,
            y_fit             = y_fit,
            rms_residual_mm   = rms_res * 1e3)
end

"""
    plot_fourier_analysis(fr1, fr2, tilt1, tilt2)

Produce a 4-panel diagnostic figure showing:
  Panel 1: A1(Z) for both shells  — offset magnitude vs height
  Panel 2: φ1(Z) for both shells  — offset direction vs height
  Panel 3: Apparent centre track in XY plane
  Panel 4: A2/A1 ratio            — nonlinearity / self-consistency check
"""
function plot_fourier_analysis(fr1, fr2, tilt1, tilt2)
    fig = Figure(size=(1800, 1100), backgroundcolor=:white)

    Label(fig[0, 1:2],
          "Fourier Tilt / Centre Analysis — $(fr1.component)";
          fontsize=22, halign=:center)
    rowsize!(fig.layout, 0, Fixed(35))

    # ── Panel 1: A1(Z) ───────────────────────────────────
    ax1 = Axis(fig[1, 1];
               xlabel     = "Z [m]",
               ylabel     = "A₁ [T]",
               title      = "1st Harmonic Amplitude A₁(Z)",
               titlesize  = 18,
               xlabelsize = 16,
               ylabelsize = 16,
               xticklabelsize = 14,
               yticklabelsize = 14)

    lines!(ax1, fr1.Z, fr1.A1;
           color=:royalblue, linewidth=2,
           label="Shell 1 (R=$(round(fr1.R_shell, digits=3)) m)")
    scatter!(ax1, fr1.Z, fr1.A1; color=:royalblue, markersize=8)
    lines!(ax1, fr2.Z, fr2.A1;
           color=:crimson, linewidth=2,
           label="Shell 2 (R=$(round(fr2.R_shell, digits=3)) m)")
    scatter!(ax1, fr2.Z, fr2.A1; color=:crimson, markersize=8)
    axislegend(ax1; position=:lt, labelsize=13)

    # ── Panel 2: φ1(Z) ───────────────────────────────────
    ax2 = Axis(fig[1, 2];
               xlabel     = "Z [m]",
               ylabel     = "φ₁ [degrees]",
               title      = "1st Harmonic Phase φ₁(Z)",
               titlesize  = 18,
               xlabelsize = 16,
               ylabelsize = 16,
               xticklabelsize = 14,
               yticklabelsize = 14)

    lines!(ax2, fr1.Z, rad2deg.(fr1.phi1);
           color=:royalblue, linewidth=2, label="Shell 1")
    scatter!(ax2, fr1.Z, rad2deg.(fr1.phi1);
             color=:royalblue, markersize=8)
    lines!(ax2, fr2.Z, rad2deg.(fr2.phi1);
           color=:crimson, linewidth=2, label="Shell 2")
    scatter!(ax2, fr2.Z, rad2deg.(fr2.phi1);
             color=:crimson, markersize=8)
    hlines!(ax2, [0.0]; color=:gray, linestyle=:dash, linewidth=1)
    axislegend(ax2; position=:lt, labelsize=13)

    # ── Panel 3: Apparent centre track in XY ─────────────
    ax3 = Axis(fig[2, 1];
               xlabel     = "X apparent centre [mm]",
               ylabel     = "Y apparent centre [mm]",
               title      = "Apparent Field Centre Track (X, Y) vs Z",
               titlesize  = 18,
               xlabelsize = 16,
               ylabelsize = 16,
               xticklabelsize = 14,
               yticklabelsize = 14)

    sc1 = scatter!(ax3,
                   fr1.x_app .* 1e3,
                   fr1.y_app .* 1e3;
                   color=fr1.Z, colormap=:viridis,
                   markersize=10, label="Shell 1")
    lines!(ax3, tilt1.x_fit .* 1e3, tilt1.y_fit .* 1e3;
           color=:royalblue, linewidth=2, linestyle=:dash)
    scatter!(ax3,
             fr2.x_app .* 1e3,
             fr2.y_app .* 1e3;
             color=fr2.Z, colormap=:plasma,
             markersize=10, marker=:diamond, label="Shell 2")
    lines!(ax3, tilt2.x_fit .* 1e3, tilt2.y_fit .* 1e3;
           color=:crimson, linewidth=2, linestyle=:dash)
    scatter!(ax3, [0.0], [0.0];
             color=:black, markersize=14,
             marker=:cross, strokewidth=3, label="Origin")
    axislegend(ax3; position=:lt, labelsize=13)

    Colorbar(fig[2, 2], sc1;
             label="Z [m] (Shell 1)",
             labelsize=14,
             ticklabelsize=12,
             width=18,
             tellheight=false,
             tellwidth=true)

    # ── Panel 4: A2/A1 self-consistency ──────────────────
    ax4 = Axis(fig[3, 1];
               xlabel     = "Z [m]",
               ylabel     = "A₂ / A₁  [—]",
               title      = "2nd/1st Harmonic Ratio  (< 0.1 → linear regime)",
               titlesize  = 18,
               xlabelsize = 16,
               ylabelsize = 16,
               xticklabelsize = 14,
               yticklabelsize = 14)

    ratio1 = fr1.A2 ./ max.(fr1.A1, 1e-30)
    ratio2 = fr2.A2 ./ max.(fr2.A1, 1e-30)

    lines!(ax4, fr1.Z, ratio1;
           color=:royalblue, linewidth=2, label="Shell 1")
    scatter!(ax4, fr1.Z, ratio1; color=:royalblue, markersize=8)
    lines!(ax4, fr2.Z, ratio2;
           color=:crimson, linewidth=2, label="Shell 2")
    scatter!(ax4, fr2.Z, ratio2; color=:crimson, markersize=8)
    hlines!(ax4, [0.1];
            color=:orange, linestyle=:dash,
            linewidth=2, label="0.1 threshold")
    axislegend(ax4; position=:lt, labelsize=13)

    # ── Panel 5: Tilt summary text box ───────────────────
    ax5 = Axis(fig[3, 2]; backgroundcolor=:gray95)
    hidedecorations!(ax5)
    hidespines!(ax5)
    xlims!(ax5, 0, 1)
    ylims!(ax5, 0, 1)

    summary_lines = [
        "Shell 1  (R = $(round(fr1.R_shell, digits=3)) m)",
        "  Tilt angle    : $(round(tilt1.tilt_deg,         digits=4))°",
        "  Tilt azimuth  : $(round(tilt1.tilt_azimuth_deg, digits=2))°",
        "  Axis at Z=0   : ($(round(tilt1.x0*1e3, digits=3)), $(round(tilt1.y0*1e3, digits=3))) mm",
        "  RMS residual  : $(round(tilt1.rms_residual_mm,  digits=4)) mm",
        "",
        "Shell 2  (R = $(round(fr2.R_shell, digits=3)) m)",
        "  Tilt angle    : $(round(tilt2.tilt_deg,         digits=4))°",
        "  Tilt azimuth  : $(round(tilt2.tilt_azimuth_deg, digits=2))°",
        "  Axis at Z=0   : ($(round(tilt2.x0*1e3, digits=3)), $(round(tilt2.y0*1e3, digits=3))) mm",
        "  RMS residual  : $(round(tilt2.rms_residual_mm,  digits=4)) mm",
        "",
        "Cross-shell",
        "  |Δ tilt|    : $(round(abs(tilt1.tilt_deg - tilt2.tilt_deg),                 digits=4))°",
        "  |Δ azimuth| : $(round(abs(tilt1.tilt_azimuth_deg - tilt2.tilt_azimuth_deg), digits=2))°",
    ]

    for (i, ln) in enumerate(summary_lines)
        text!(ax5, 0.05, 0.97 - (i-1) * 0.064;
              text     = ln,
              fontsize = 14,
              font     = "CMU Serif",
              color    = :black,
              align    = (:left, :top),
              space    = :data)
    end

    # ── Layout: fixed pixel sizes that fit a standard screen ─
    rowsize!(fig.layout, 1, Fixed(280))
    rowsize!(fig.layout, 2, Fixed(340))
    rowsize!(fig.layout, 3, Fixed(280))

    colsize!(fig.layout, 1, Fixed(1250))
    colsize!(fig.layout, 2, Fixed(430))

    rowgap!(fig.layout, 1, Fixed(25))
    rowgap!(fig.layout, 2, Fixed(25))
    rowgap!(fig.layout, 3, Fixed(25))
    colgap!(fig.layout, Fixed(20))

    resize_to_layout!(fig)

    return fig
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

    println("    Computing Biot-Savart at $nobs points " *
            "(Z = $(round(z_val, digits=4)) m)...")
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
    plot_coils_xy_at_z!(ax, coil_sets, z_val, set_palettes; project, tol)

Overlay coil geometry on a 2D (X, Y) axis.
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
                    if (xl[1]-xl[end])^2 + (yl[1]-yl[end])^2 > 1e-10
                        push!(xl, xl[1])
                        push!(yl, yl[1])
                    end
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
    compute_field_weighted_center(px, py, B_mag)

Compute the |B|-weighted centroid of 2D probe positions.
Returns (cx, cy).
"""
function compute_field_weighted_center(px, py, B_mag)
    w = abs.(B_mag)
    W = sum(w)
    if W < 1e-30
        return mean(px), mean(py)
    end
    cx = sum(w .* px) / W
    cy = sum(w .* py) / W
    return cx, cy
end

"""
    plot_hall_z_slice(hall_data, z_val, tol, component,
                      coil_sets, coil_bbox, set_palettes;
                      show_coils, coil_project, coil_tol_frac, show_centering)

Create a 2D scatter plot of hall probe measurements near Z = z_val ± tol,
projected onto the (X, Y) plane. Optionally mark the field-weighted centre.
"""
function plot_hall_z_slice(hall_data, z_val, tol, component,
                            coil_sets, coil_bbox, set_palettes;
                            show_coils=true, coil_project=true,
                            coil_tol_frac=0.05, show_centering=false)    
                            s = unit_scale()

                            # Select probes within the slab
                            mask      = abs.(hall_data.probe_z .- z_val) .< tol
                            n_in_slab = count(mask)
                        
                            px = hall_data.probe_x[mask]
                            py = hall_data.probe_y[mask]
                            color_raw = select_component(hall_data.B_R[mask], hall_data.B_phi[mask],
                                                          hall_data.B_Z[mask], component)
                            color_display = color_raw .* s
                            B_mag_slab = hall_data.B_mag[mask]
                        
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
                        
                            # ── Field-weighted centre ────────────────────────────
                            if show_centering && n_in_slab > 0
                                cx, cy = compute_field_weighted_center(px, py, B_mag_slab)
                                cR     = sqrt(cx^2 + cy^2)
                                cphi   = atan(cy, cx)
                        
                                println("    Field-weighted centre: (X, Y) = " *
                                        "($(round(cx, digits=5)), $(round(cy, digits=5))) m")
                                println("    Field-weighted centre: (R, φ) = " *
                                        "($(round(cR, digits=5)) m, $(round(rad2deg(cphi), digits=3))°)")
                                println("    Offset from origin:    " *
                                        "ΔX=$(round(cx*1000, digits=3)) mm, " *
                                        "ΔY=$(round(cy*1000, digits=3)) mm, " *
                                        "|Δ|=$(round(sqrt(cx^2+cy^2)*1000, digits=3)) mm")
                        
                                scatter!(ax, [cx], [cy];
                                         color=:lime, markersize=22,
                                         marker=:cross, strokecolor=:black, strokewidth=2)
                                scatter!(ax, [0.0], [0.0];
                                         color=:white, markersize=18,
                                         marker=:diamond, strokecolor=:black, strokewidth=2)
                        
                                scatter!(ax, [NaN], [NaN];
                                         color=:lime, markersize=14, marker=:cross,
                                         label="|B|-centre ($(round(cx*1000,digits=1)), " *
                                               "$(round(cy*1000,digits=1))) mm")
                                scatter!(ax, [NaN], [NaN];
                                         color=:white, markersize=14, marker=:diamond,
                                         strokecolor=:black, strokewidth=1,
                                         label="Origin (0,0)")
                                axislegend(ax; position=:lt, labelsize=16)
                            end
                        
                            return fig
                        end
                        
                        # ═══════════════════════════════════════════════════════════════
                        # Main
                        # ═══════════════════════════════════════════════════════════════
                        function main()
                            fl = field_label(FIELD_COMPONENT)
                            println("Field component : $fl ($(FIELD_COMPONENT)), units: $(unit_str())")
                            println("Loading coil sets...")
                            coil_sets = load_all_coil_sets(COIL_FILES)
                        
                            println("\nTorus: R0 = $R0 m, a = $a m")
                            X_torus, Y_torus, Z_torus, C_torus =
                                compute_field_on_torus(coil_sets, R0, a, N_THETA, N_PHI)
                        
                            coil_bbox = compute_coil_bounding_box(coil_sets)
                        
                            # ── Hall probe shell layout ───────────────────────────
                            hall_data = nothing
                            if ENABLE_HALL_PROBES      ||
                               ENABLE_HALL_PERTURBATION ||
                               ENABLE_HALL_Z_SLICES    ||
                               ENABLE_HALL_EXPORT      ||
                               ENABLE_FOURIER_ANALYSIS
                        
                                println("\nComputing synthetic Hall probe measurements (shell layout)...")
                                hall_data = compute_hall_probes_shells(
                                    coil_sets, coil_bbox, HALL_MARGIN,
                                    HALL_N_PHI_INNER, HALL_N_Z_INNER,
                                    HALL_N_PHI_OUTER, HALL_N_Z_OUTER,
                                    HALL_R_INNER_FRAC, HALL_R_OUTER_FRAC
                                )
                            end
                        
                            # ── Filter probes inside coil geometry ───────────────
                            outside_mask = nothing
                            if hall_data !== nothing
                                println("\n  Filtering probes inside coil geometry...")
                                outside_mask = filter_probes_outside_coils(
                                    hall_data, coil_sets; pad=HALL_COIL_EXCLUSION_PAD)
                        
                                if ENABLE_HALL_EXPORT
                                    export_hall_probes(HALL_EXPORT_FILE, hall_data, outside_mask)
                                end
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
                        
                            # ── Figure 1: Torus surface ───────────────────────────
                            if ENABLE_TORUS_PLOT
                                fig1, ax1 = make_3d_figure(title="$fl on Torus Surface")
                        
                                if field_is_signed(FIELD_COMPONENT)
                                    clim = symmetric_clim(C_torus)
                                    sf = surface!(ax1, X_torus, Y_torus, Z_torus;
                                                  color=C_torus, colormap=:RdBu,
                                                  colorrange=(-clim, clim))
                                else
                                    clim_hi = max(quantile(vec(C_torus), 0.98), 1e-30)
                                    sf = surface!(ax1, X_torus, Y_torus, Z_torus;
                                                  color=C_torus, colormap=:inferno,
                                                  colorrange=(0.0, clim_hi))
                                end
                                Colorbar(fig1[1, 2], sf; label=field_cb_label(FIELD_COMPONENT), width=20)
                        
                                plot_coils!(ax1, coil_sets, set_palettes)
                                cam3d!(ax1.scene;
                                       lookat=cam_lookat, eyeposition=cam_eye, upvector=cam_up)
                                push!(figures, ("torus", fig1))
                            end  # ── end Figure 1 ──
                        
                            # ── Figure 2: Hall probes — total field ───────────────
                            if ENABLE_HALL_PROBES && hall_data !== nothing
                                probe_color = select_component(
                                    hall_data.B_R, hall_data.B_phi, hall_data.B_Z, FIELD_COMPONENT)
                        
                                fig2, ax2 = make_3d_figure(title="Synthetic Hall Probes — Total $fl")
                        
                                lo, hi = make_hall_scatter!(fig2, ax2,
                                    hall_data.probe_x, hall_data.probe_y, hall_data.probe_z,
                                    probe_color, FIELD_COMPONENT, false)
                        
                                plot_coils!(ax2, coil_sets, set_palettes)
                                cam3d!(ax2.scene;
                                       lookat=cam_lookat, eyeposition=cam_eye, upvector=cam_up)
                        
                                println("  Hall total: $(length(probe_color)) probes")
                                push!(figures, ("hall_total", fig2))
                            end  # ── end Figure 2 ──
                        
                            # ── Figure 3: Hall probes — non-axisymmetric, filtered ─
                            if ENABLE_HALL_PERTURBATION && hall_data !== nothing
                                println("\n  Subtracting axisymmetric (n=0) component (per shell)...")
                        
                                B_R_pert, B_phi_pert, B_Z_pert =
                                    subtract_axisymmetric_shells(hall_data)
                        
                                pert_color = select_component(B_R_pert, B_phi_pert, B_Z_pert,
                                                              FIELD_COMPONENT)
                                pl = pert_label(FIELD_COMPONENT)
                        
                                fig3, ax3 = make_3d_figure(
                                    title="Synthetic Hall Probes — Non-Axisymmetric $pl (filtered)")
                        
                                lo, hi = make_hall_scatter_filtered!(fig3, ax3,
                                    hall_data.probe_x, hall_data.probe_y, hall_data.probe_z,
                                    pert_color, FIELD_COMPONENT, PERT_VISIBILITY_THRESHOLD)
                        
                                plot_coils!(ax3, coil_sets, set_palettes)
                                cam3d!(ax3.scene;
                                       lookat=cam_lookat, eyeposition=cam_eye, upvector=cam_up)
                        
                                println("  Hall perturbation: $pl ∈ [$lo, $hi]")
                                push!(figures, ("hall_perturbation", fig3))
                            end  # ── end Figure 3 ──
                        
                            # ── Figure 4: Fourier tilt/centre analysis ────────────
                            if ENABLE_FOURIER_ANALYSIS && hall_data !== nothing
                                println("\n  Running Fourier tilt/centre analysis...")
                        
                                fr1   = fourier_analyze_shell(hall_data; shell=1, component=:B_phi)
                                fr2   = fourier_analyze_shell(hall_data; shell=2, component=:B_phi)
                                tilt1 = fit_tilt_axis(fr1)
                                tilt2 = fit_tilt_axis(fr2)
                        
                                Δtilt = abs(tilt1.tilt_deg - tilt2.tilt_deg)
                                Δazi  = abs(tilt1.tilt_azimuth_deg - tilt2.tilt_azimuth_deg)
                                println("\n  Cross-shell consistency:")
                                println("    |Δ tilt angle|   = $(round(Δtilt, digits=4))°  " *
                                        "(should be ≈ 0 for a rigid tilt)")
                                println("    |Δ tilt azimuth| = $(round(Δazi,  digits=2))°  " *
                                        "(should be ≈ 0 for a rigid tilt)")
                        
                                println("  Building Fourier analysis figure...")
                                fig4 = plot_fourier_analysis(fr1, fr2, tilt1, tilt2)
                                println("  Fourier figure built successfully")
                                push!(figures, ("fourier_analysis", fig4))
                        
                                println("\n  Cross-component consistency check (Br, Bz on shell 1)...")
                                fr1_Br = fourier_analyze_shell(hall_data; shell=1, component=:B_R)
                                fr1_Bz = fourier_analyze_shell(hall_data; shell=1, component=:B_Z)
                                println("    φ₁ (Bphi) mean: $(round(rad2deg(mean(fr1.phi1)),    digits=2))°")
                                println("    φ₁ (Br)   mean: $(round(rad2deg(mean(fr1_Br.phi1)), digits=2))°")
                                println("    φ₁ (Bz)   mean: $(round(rad2deg(mean(fr1_Bz.phi1)), digits=2))°")
                            end  # ── end Figure 4 ──
                        
                            # ── Figure 5: Z-slice 2D heatmaps ────────────────────
                            if ENABLE_Z_SLICES
                                z_positions = if Z_SLICE_POSITIONS == :auto
                                    unique(round.(coil_set_z_centers(coil_sets), digits=4))
                                else
                                    Float64.(Z_SLICE_POSITIONS)
                                end
                        
                                println("\n  Computing Z-slices at: $(z_positions) m")
                        
                                for (si, z_val) in enumerate(z_positions)
                                    x_range, y_range, field_2d = compute_z_slice_xy(
                                        coil_sets, z_val, coil_bbox, Z_SLICE_MARGIN, Z_SLICE_N_XY)
                        
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
                        
                                    Colorbar(fig_s[1, 2], hm;
                                             label=field_cb_label(FIELD_COMPONENT), width=20)
                        
                                    if Z_SLICE_SHOW_COILS
                                        z_extent  = coil_bbox[4] - coil_bbox[3]
                                        slice_tol = max(0.05, z_extent * Z_SLICE_COIL_TOL_FRAC)
                                        plot_coils_xy_at_z!(ax_s, coil_sets, z_val, set_palettes;
                                                            project=Z_SLICE_COIL_PROJECT,
                                                            tol=slice_tol)
                                    end
                        
                                    push!(figures, ("zslice_$(si)", fig_s))
                                end
                            end  # ── end Figure 5 ──
                        
                            # ── Figure 6: Hall probe Z-slice 2D scatter ───────────
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
                                        show_coils     = HALL_Z_SLICE_SHOW_COILS,
                                        coil_project   = HALL_Z_SLICE_COIL_PROJECT,
                                        coil_tol_frac  = Z_SLICE_COIL_TOL_FRAC,
                                        show_centering = HALL_Z_SLICE_CENTERING)
                        
                                    push!(figures, ("hall_zslice_$(si)", fig_hz))
                                end
                            end  # ── end Figure 6 ──
                        
                            # ── Display all figures ───────────────────────────────
                            println("\n  Displaying $(length(figures)) figures:")
                            for (name, fig) in figures
                                println("    displaying: $name")
                                display(GLMakie.Screen(), fig)
                            end
                        
                            println("\nwPress Enter to save and exit...")
                            readline()
                        
                            # ── Save all figures ──────────────────────────────────
                            for (name, fig) in figures
                                outpath = replace(PLOT_FILE, ".png" => "_$(name).png")
                                save(outpath, fig)
                                println("Saved $name → $outpath")
                            end
                        
                            println("\nDone.")
                        end
                        
                        main()
                        