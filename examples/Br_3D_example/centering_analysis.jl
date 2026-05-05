using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using GeneralizedPerturbedEquilibrium
using LinearAlgebra
using Statistics
using DelimitedFiles
using Printf
using FFTW
using GLMakie

# ═══════════════════════════════════════════════════════════════
# Global theme
# ═══════════════════════════════════════════════════════════════
set_theme!(Theme(
    fontsize = 28,
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

const _FT            = GeneralizedPerturbedEquilibrium.ForcingTerms
const _read_coil_dat = _FT.read_coil_dat

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
HALL_DATA_FILE = joinpath(@__DIR__, "hall_probe_test_357.csv")

REFERENCE_COIL_FILES = [
    joinpath(@__DIR__, "sparc_pf1u.dat"),
]

TRANSFORMED_COIL_FILES = [
    joinpath(@__DIR__, "sparc_pf1u_357_test.dat"),
]
name = "test_357"
OUTPUT_FILE = joinpath(@__DIR__, "$name.txt")
PLOT_FILE   = joinpath(@__DIR__, "$name.png")

# ═══════════════════════════════════════════════════════════════
# Read hall probe CSV
# ═══════════════════════════════════════════════════════════════
function read_hall_probe_csv(filepath::String)
    isfile(filepath) || error("Not found: $filepath")
    println("Reading: $filepath")

    lines_raw = readlines(filepath)
    data_lines = String[]
    header_line = ""
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

    println("  $N probes, R∈[$(round(minimum(R),digits=4)), $(round(maximum(R),digits=4))] m, " *
            "Z∈[$(round(minimum(Z),digits=4)), $(round(maximum(Z),digits=4))] m")

    return (; x, y, z, R, phi, Z, B_R, B_phi, B_Z, B_mag, N)
end

# ═══════════════════════════════════════════════════════════════
# Build structured 3D grid from flat probe data
# ═══════════════════════════════════════════════════════════════
function build_probe_grid(probes)
    R_vals   = sort(unique(round.(probes.R,   digits=8)))
    phi_vals = sort(unique(round.(probes.phi, digits=8)))
    Z_vals   = sort(unique(round.(probes.Z,   digits=8)))

    n_r   = length(R_vals)
    n_phi = length(phi_vals)
    n_z   = length(Z_vals)

    println("  Probe grid: $n_r R × $n_phi φ × $n_z Z = $(n_r*n_phi*n_z) points")

    R_idx   = Dict(round(r, digits=8) => i for (i, r) in enumerate(R_vals))
    phi_idx = Dict(round(p, digits=8) => i for (i, p) in enumerate(phi_vals))
    Z_idx   = Dict(round(z, digits=8) => i for (i, z) in enumerate(Z_vals))

    # 3D arrays: (R, phi, Z)
    BR_3d   = fill(NaN, n_r, n_phi, n_z)
    Bphi_3d = fill(NaN, n_r, n_phi, n_z)
    BZ_3d   = fill(NaN, n_r, n_phi, n_z)
    Bmag_3d = fill(NaN, n_r, n_phi, n_z)

    for i in 1:probes.N
        ir = get(R_idx,   round(probes.R[i],   digits=8), 0)
        ip = get(phi_idx, round(probes.phi[i], digits=8), 0)
        iz = get(Z_idx,   round(probes.Z[i],   digits=8), 0)
        (ir == 0 || ip == 0 || iz == 0) && continue
        BR_3d[ir, ip, iz]   = probes.B_R[i]
        Bphi_3d[ir, ip, iz] = probes.B_phi[i]
        BZ_3d[ir, ip, iz]   = probes.B_Z[i]
        Bmag_3d[ir, ip, iz] = probes.B_mag[i]
    end

    return (; R_vals, phi_vals, Z_vals, n_r, n_phi, n_z,
              BR_3d, Bphi_3d, BZ_3d, Bmag_3d)
end

# ═══════════════════════════════════════════════════════════════
# Coil geometry (R₀, Z₀, normal, tilt)
# ═══════════════════════════════════════════════════════════════
function coil_geometry(coil_files::Vector{String}; label="Coil")
    all_x = Float64[]; all_y = Float64[]; all_z = Float64[]
    for fp in coil_files
        isfile(fp) || error("Not found: $fp")
        cs = _read_coil_dat(fp)
        append!(all_x, vec(cs.x)); append!(all_y, vec(cs.y)); append!(all_z, vec(cs.z))
    end

    all_R = sqrt.(all_x.^2 .+ all_y.^2)
    R0 = mean(all_R); Z0 = mean(all_z)
    cx = mean(all_x); cy = mean(all_y); cz = mean(all_z)

    P = hcat(all_x .- cx, all_y .- cy, all_z .- cz)
    _, _, V = svd(P)
    normal = V[:, 3]
    if normal[3] < 0; normal = -normal; end

    tilt_from_z = acos(clamp(abs(normal[3]), 0.0, 1.0))
    tilt_x = atan(normal[1], normal[3])
    tilt_y = atan(normal[2], normal[3])

    # Coil junction: find the largest gap in toroidal angle
    all_phi = atan.(all_y, all_x)
    phi_sorted = sort(all_phi)
    dphi = diff(phi_sorted)
    push!(dphi, (phi_sorted[1] + 2π) - phi_sorted[end])
    junction_idx = argmax(dphi)
    if junction_idx <= length(phi_sorted)
        junction_phi = phi_sorted[junction_idx] + dphi[junction_idx] / 2
    else
        junction_phi = phi_sorted[end] + dphi[end] / 2
    end
    junction_phi = mod(junction_phi + π, 2π) - π  # normalize to [-π, π]

    println("  $label: R₀=$(round(R0*1000, digits=3)) mm, Z₀=$(round(Z0*1000, digits=3)) mm")
    println("    Normal: $(round.(normal, digits=6)), Tilt from ẑ: $(round(rad2deg(tilt_from_z), digits=4))°")
    println("    Junction φ ≈ $(round(rad2deg(junction_phi), digits=2))°")

    return (; R0, Z0, x=cx, y=cy, z=cz,
              normal, tilt_from_z, tilt_x, tilt_y,
              junction_phi, all_x, all_y, all_z, all_R)
end

# ═══════════════════════════════════════════════════════════════
# FIT 1: Z₀ from B_Z sign change
#
# B_Z from a PF coil reverses sign as you cross the coil
# midplane in Z.  The zero-crossing Z(R, φ) gives the local
# effective Z-center.
# ═══════════════════════════════════════════════════════════════
function fit_Z_center_from_sign_change(grid)
    println("\n═══ Z₀ from B_Z sign change ═══")

    Z_vals = grid.Z_vals
    n_r    = grid.n_r
    n_phi  = grid.n_phi
    n_z    = grid.n_z

    # For each (R, φ) column, find the Z where B_Z changes sign
    Z_crossings = fill(NaN, n_r, n_phi)

    for ir in 1:n_r, ip in 1:n_phi
        bz_col = grid.BZ_3d[ir, ip, :]

        # Check for NaN
        any(isnan.(bz_col)) && continue

        # Find sign changes
        for iz in 1:(n_z - 1)
            if bz_col[iz] * bz_col[iz + 1] < 0
                # Linear interpolation for zero crossing
                z1 = Z_vals[iz]; z2 = Z_vals[iz + 1]
                b1 = bz_col[iz]; b2 = bz_col[iz + 1]
                Z_crossings[ir, ip] = z1 - b1 * (z2 - z1) / (b2 - b1)
                break  # take the first crossing
            end
        end
    end

    valid = .!isnan.(Z_crossings)
    n_valid = count(valid)
    println("  Found $n_valid / $(n_r * n_phi) zero crossings")

    if n_valid == 0
        println("  WARNING: No B_Z sign changes found. Is the probe grid spanning the coil Z center?")
        return nothing
    end

    # ── Overall Z₀: mean of all crossings ────────────────
    Z0_mean = mean(Z_crossings[valid])
    Z0_std  = std(Z_crossings[valid])
    println("  Z₀ (mean of crossings) = $(round(Z0_mean*1000, digits=3)) ± $(round(Z0_std*1000, digits=3)) mm")

    # ── Z₀(φ): the crossings at each toroidal angle ─────
    # For a tilted coil, Z₀(φ) = Z₀_mean + A_cos·cos(φ) + A_sin·sin(φ)
    # where A_cos and A_sin encode the tilt direction and magnitude.

    # Average over R at each φ
    Z_vs_phi = zeros(n_phi)
    Z_vs_phi_valid = falses(n_phi)

    for ip in 1:n_phi
        col = Z_crossings[:, ip]
        mask = .!isnan.(col)
        if any(mask)
            Z_vs_phi[ip] = mean(col[mask])
            Z_vs_phi_valid[ip] = true
        end
    end

    return (; Z0=Z0_mean, Z0_std,
              Z_crossings, Z_vs_phi, Z_vs_phi_valid)
end

# ═══════════════════════════════════════════════════════════════
# FIT 2: X/Y tilt from sinusoidal Z₀(φ) variation
#
# A coil tilted by angle γ about an axis at toroidal angle φ₀
# creates:  Z_eff(φ) = Z₀ + R₀·tan(γ)·sin(φ − φ₀)
#                     = Z₀ + A·cos(φ) + B·sin(φ)
# where A = −R₀·tan(γ)·sin(φ₀), B = R₀·tan(γ)·cos(φ₀)
#
# Fit the n=1 Fourier component of Z₀(φ).
# ═══════════════════════════════════════════════════════════════
function fit_tilt_from_Z_variation(z_fit, grid, R0_est)
    println("\n═══ X/Y Tilt from Z₀(φ) sinusoidal variation ═══")

    z_fit === nothing && (println("  No Z crossing data."); return nothing)

    phi_vals = grid.phi_vals
    Z_vs_phi = z_fit.Z_vs_phi
    valid    = z_fit.Z_vs_phi_valid

    n_valid = count(valid)
    if n_valid < 4
        println("  WARNING: Need ≥ 4 valid φ points for tilt fit (have $n_valid)")
        return nothing
    end

    # Fit:  Z(φ) = Z₀ + A·cos(φ) + B·sin(φ)
    phi_v = phi_vals[valid]
    Z_v   = Z_vs_phi[valid]

    # Design matrix
    M = hcat(ones(n_valid), cos.(phi_v), sin.(phi_v))
    coeffs = M \ Z_v

    Z0_fit = coeffs[1]
    A_cos  = coeffs[2]
    A_sin  = coeffs[3]

    # Amplitude and phase
    tilt_amplitude = sqrt(A_cos^2 + A_sin^2)  # = R₀·tan(γ)
    tilt_phase     = atan(-A_cos, A_sin)        # φ₀ (tilt axis direction)

    # Recover tilt angle
    if R0_est > 0.01
        tilt_angle = atan(tilt_amplitude / R0_est)
    else
        tilt_angle = NaN
    end

    # Decompose into X and Y tilts
    # Tilt about X-axis → Z variation ~ sin(φ)  → A_sin component
    # Tilt about Y-axis → Z variation ~ cos(φ)  → A_cos component
    tilt_about_x = atan(A_sin / R0_est)    # radians
    tilt_about_y = atan(-A_cos / R0_est)   # radians (negative because cos convention)

    # Fit residual
    Z_fitted = M * coeffs
    residual = sqrt(mean((Z_v .- Z_fitted).^2))

    println("  Z₀(φ) fit: Z₀=$(round(Z0_fit*1000, digits=3)) mm")
    println("    + $(round(A_cos*1000, digits=4))·cos(φ) + $(round(A_sin*1000, digits=4))·sin(φ) mm")
    println("  Tilt amplitude: $(round(tilt_amplitude*1000, digits=4)) mm (R₀·tan γ)")
    println("  Tilt angle γ:   $(round(rad2deg(tilt_angle), digits=4))°")
    println("  Tilt axis φ₀:   $(round(rad2deg(tilt_phase), digits=2))°")
    println("  Tilt about X:   $(round(rad2deg(tilt_about_x), digits=4))°")
    println("  Tilt about Y:   $(round(rad2deg(tilt_about_y), digits=4))°")
    println("  Fit residual:   $(round(residual*1000, digits=4)) mm")

    return (; Z0=Z0_fit, A_cos, A_sin,
              tilt_amplitude, tilt_angle, tilt_phase,
              tilt_about_x, tilt_about_y,
              residual, Z_fitted, phi_v, Z_v)
end

# ═══════════════════════════════════════════════════════════════
# FIT 3: R₀ from toroidal FFT of |B| profile
#
# At each Z, the n=0 (axisymmetric) component of |B|(R) peaks
# at the conductor radius R₀.  We toroidally average (take n=0),
# then find the R where |B|_n=0 is maximum.
# ═══════════════════════════════════════════════════════════════
function fit_R_center_from_FFT(grid)
    println("\n═══ R₀ from toroidal FFT (n=0 profile) ═══")

    R_vals  = grid.R_vals
    Z_vals  = grid.Z_vals
    n_r     = grid.n_r
    n_phi   = grid.n_phi
    n_z     = grid.n_z

    # ── Toroidally average |B| to get n=0 component ──────
    Bmag_n0 = zeros(n_r, n_z)   # n=0 of |B| at each (R, Z)
    counts  = zeros(Int, n_r, n_z)

    for iz in 1:n_z, ir in 1:n_r
        vals = grid.Bmag_3d[ir, :, iz]
        mask = .!isnan.(vals)
        if any(mask)
            Bmag_n0[ir, iz] = mean(vals[mask])
            counts[ir, iz] = count(mask)
        end
    end

    # ── At each Z, find R where Bmag_n0(R) is maximum ───
    # Use quadratic interpolation around the peak for sub-grid accuracy
    R0_at_Z = fill(NaN, n_z)

    for iz in 1:n_z
        profile = Bmag_n0[:, iz]
        valid = counts[:, iz] .> 0
        n_valid = count(valid)
        n_valid < 3 && continue

        # Find peak
        peak_ir = argmax(profile)

        if peak_ir == 1 || peak_ir == n_r
            # Peak at edge — just use the edge value
            R0_at_Z[iz] = R_vals[peak_ir]
        else
            # Quadratic interpolation: fit parabola through 3 points
            r1 = R_vals[peak_ir - 1]; b1 = profile[peak_ir - 1]
            r2 = R_vals[peak_ir];     b2 = profile[peak_ir]
            r3 = R_vals[peak_ir + 1]; b3 = profile[peak_ir + 1]

            # Parabola vertex
            denom = 2.0 * ((r2-r1)*(b2-b3) - (r2-r3)*(b2-b1))
            if abs(denom) > 1e-30
                R_peak = r2 - ((r2-r1)^2*(b2-b3) - (r2-r3)^2*(b2-b1)) / denom
                # Sanity check
                if r1 <= R_peak <= r3
                    R0_at_Z[iz] = R_peak
                else
                    R0_at_Z[iz] = R_vals[peak_ir]
                end
            else
                R0_at_Z[iz] = R_vals[peak_ir]
            end
        end
    end

    valid_z = .!isnan.(R0_at_Z)
    n_valid = count(valid_z)
    println("  Found R₀ peak at $n_valid / $n_z Z slices")

    if n_valid == 0
        println("  WARNING: Could not find |B| peak at any Z")
        return nothing
    end

    R0_mean = mean(R0_at_Z[valid_z])
    R0_std  = n_valid > 1 ? std(R0_at_Z[valid_z]) : 0.0

    println("  R₀ = $(round(R0_mean*1000, digits=3)) ± $(round(R0_std*1000, digits=3)) mm")

    # ── Also compute n=1 amplitude vs R to check for lateral shift ──
    n1_amplitude = zeros(n_r, n_z)
    for iz in 1:n_z, ir in 1:n_r
        vals = grid.Bmag_3d[ir, :, iz]
        mask = .!isnan.(vals)
        count(mask) < 3 && continue

        phi_v = grid.phi_vals[mask]
        b_v   = vals[mask]
        nc    = mean(b_v .* cos.(phi_v))
        ns    = mean(b_v .* sin.(phi_v))
        n1_amplitude[ir, iz] = 2.0 * sqrt(nc^2 + ns^2)
    end

    return (; R0=R0_mean, R0_std, R0_at_Z, Bmag_n0, n1_amplitude, counts)
end

# ═══════════════════════════════════════════════════════════════
# FIT 4: Junction/rotation from non-axisymmetric field spike
#
# The coil winding has a junction (start/end) that breaks
# perfect toroidal symmetry.  This creates a localized spike
# in the non-axisymmetric field.  Find its toroidal angle.
# ═══════════════════════════════════════════════════════════════
function fit_junction_from_field_spike(grid)
    println("\n═══ Junction/Rotation from field asymmetry ═══")

    n_r   = grid.n_r
    n_phi = grid.n_phi
    n_z   = grid.n_z

    # Compute non-axisymmetric perturbation: δB(R, φ, Z) = B(R,φ,Z) - <B>_φ(R,Z)
    # Then look at |δB| as a function of φ

    perturbation_vs_phi = zeros(n_phi)
    total_weight = zeros(n_phi)

    for iz in 1:n_z, ir in 1:n_r
        # Get the toroidal profile at this (R, Z)
        bm = grid.Bmag_3d[ir, :, iz]
        mask = .!isnan.(bm)
        count(mask) < 3 && continue

        # Axisymmetric average
        b_mean = mean(bm[mask])
        b_mean < 1e-20 && continue

        # Perturbation
        for ip in 1:n_phi
            mask[ip] || continue
            δb = abs(bm[ip] - b_mean) / b_mean   # relative perturbation
            perturbation_vs_phi[ip] += δb * b_mean^2   # weight by B²
            total_weight[ip] += b_mean^2
        end
    end

    # Normalize
    valid_phi = total_weight .> 1e-30
    perturbation_vs_phi[valid_phi] ./= total_weight[valid_phi]

    if !any(valid_phi)
        println("  WARNING: No valid perturbation data")
        return nothing
    end

    # Find the peak perturbation angle
    peak_ip = argmax(perturbation_vs_phi)
    junction_phi_detected = grid.phi_vals[peak_ip]

    # Refine with quadratic interpolation
    if 2 <= peak_ip <= n_phi - 1
        p1 = perturbation_vs_phi[peak_ip - 1]
        p2 = perturbation_vs_phi[peak_ip]
        p3 = perturbation_vs_phi[peak_ip + 1]
        φ1 = grid.phi_vals[peak_ip - 1]
        φ2 = grid.phi_vals[peak_ip]
        φ3 = grid.phi_vals[peak_ip + 1]

        denom = 2.0 * ((φ2-φ1)*(p2-p3) - (φ2-φ3)*(p2-p1))
        if abs(denom) > 1e-30
            φ_peak = φ2 - ((φ2-φ1)^2*(p2-p3) - (φ2-φ3)^2*(p2-p1)) / denom
            if φ1 <= φ_peak <= φ3
                junction_phi_detected = φ_peak
            end
        end
    end

    peak_perturbation = perturbation_vs_phi[peak_ip]
    mean_perturbation = mean(perturbation_vs_phi[valid_phi])

    # SNR: how much does the peak stand out?
    snr = peak_perturbation / max(mean_perturbation, 1e-30)

    println("  Junction φ (detected): $(round(rad2deg(junction_phi_detected), digits=2))°")
    println("  Peak relative perturbation: $(round(peak_perturbation*100, digits=4))%")
    println("  Mean relative perturbation: $(round(mean_perturbation*100, digits=4))%")
    println("  SNR (peak/mean): $(round(snr, digits=2))")

    if snr < 1.5
        println("  ⚠  Low SNR — junction detection may be unreliable")
    end

    return (; junction_phi=junction_phi_detected, peak_perturbation,
              mean_perturbation, snr, perturbation_vs_phi)
end

# ═══════════════════════════════════════════════════════════════
# Combined fit summary
# ═══════════════════════════════════════════════════════════════
function combined_fit_summary(r_fit, z_fit, tilt_fit, junction_fit)
    println("\n╔══════════════════════════════════════════════════════════════════╗")
    println("║              COMBINED FIT RESULTS                               ║")
    println("╠══════════════════════════════════════════════════════════════════╣")

    if r_fit !== nothing
        @printf("║  R₀ (from |B| n=0 peak)    = %10.3f ± %6.3f mm             ║\n",
                r_fit.R0*1000, r_fit.R0_std*1000)
    end

    if z_fit !== nothing
        @printf("║  Z₀ (from B_Z sign change) = %10.3f ± %6.3f mm             ║\n",
                z_fit.Z0*1000, z_fit.Z0_std*1000)
    end

    if tilt_fit !== nothing
        @printf("║  Tilt about X              = %10.4f°                        ║\n",
                rad2deg(tilt_fit.tilt_about_x))
        @printf("║  Tilt about Y              = %10.4f°                        ║\n",
                rad2deg(tilt_fit.tilt_about_y))
        @printf("║  Total tilt angle γ        = %10.4f°                        ║\n",
                rad2deg(tilt_fit.tilt_angle))
        @printf("║  Tilt axis direction φ₀    = %10.2f°                        ║\n",
                rad2deg(tilt_fit.tilt_phase))
    end

    if junction_fit !== nothing
        @printf("║  Junction φ (detected)     = %10.2f°   (SNR=%.1f)            ║\n",
                rad2deg(junction_fit.junction_phi), junction_fit.snr)
    end

    println("╚══════════════════════════════════════════════════════════════════╝")
end

# ═══════════════════════════════════════════════════════════════
# Write results
# ═══════════════════════════════════════════════════════════════
function write_results(filepath, geo_ref, geo_trans,
                       r_fit, z_fit, tilt_fit, junction_fit)
    open(filepath, "w") do io
        println(io, "═══════════════════════════════════════════════════════")
        println(io, " PF Coil Centering Analysis — Full Fit Results")
        println(io, "═══════════════════════════════════════════════════════\n")

        for (label, geo) in [("Reference", geo_ref), ("Transformed", geo_trans)]
            geo === nothing && continue
            println(io, "── $label Coil Geometry ──")
            @printf(io, "  R₀ = %.4f m (%.3f mm)\n", geo.R0, geo.R0*1000)
            @printf(io, "  Z₀ = %.4f m (%.3f mm)\n", geo.Z0, geo.Z0*1000)
            println(io, "  Normal: $(round.(geo.normal, digits=6))")
            @printf(io, "  Tilt from ẑ: %.4f°\n", rad2deg(geo.tilt_from_z))
            @printf(io, "  Tilt X: %.4f°, Tilt Y: %.4f°\n", rad2deg(geo.tilt_x), rad2deg(geo.tilt_y))
            @printf(io, "  Junction φ: %.2f°\n\n", rad2deg(geo.junction_phi))
        end

        println(io, "── Fitted Parameters (from Hall probe data) ──")
        if r_fit !== nothing
            @printf(io, "  R₀ = %.4f ± %.4f m (%.3f ± %.3f mm)\n",
                    r_fit.R0, r_fit.R0_std, r_fit.R0*1000, r_fit.R0_std*1000)
        end
        if z_fit !== nothing
            @printf(io, "  Z₀ = %.4f ± %.4f m (%.3f ± %.3f mm)\n",
                    z_fit.Z0, z_fit.Z0_std, z_fit.Z0*1000, z_fit.Z0_std*1000)
        end
        if tilt_fit !== nothing
            @printf(io, "  Tilt about X: %.4f°\n", rad2deg(tilt_fit.tilt_about_x))
            @printf(io, "  Tilt about Y: %.4f°\n", rad2deg(tilt_fit.tilt_about_y))
            @printf(io, "  Total tilt γ: %.4f°\n", rad2deg(tilt_fit.tilt_angle))
            @printf(io, "  Tilt axis φ₀: %.2f°\n", rad2deg(tilt_fit.tilt_phase))
            @printf(io, "  Fit residual: %.4f mm\n", tilt_fit.residual*1000)
        end
        if junction_fit !== nothing
            @printf(io, "  Junction φ: %.2f° (SNR=%.1f)\n",
                    rad2deg(junction_fit.junction_phi), junction_fit.snr)
        end

        # ── Benchmarking table ───────────────────────────
        if geo_ref !== nothing && geo_trans !== nothing
            println(io, "\n═══ BENCHMARKING ═══")
            println(io, "")
            println(io, "  Parameter     │   True      │  Recovered  │  Error")
            println(io, "  ──────────────┼─────────────┼─────────────┼──────────")

            true_dR = (geo_trans.R0 - geo_ref.R0) * 1000
            true_dZ = (geo_trans.Z0 - geo_ref.Z0) * 1000
            true_tilt_x = rad2deg(geo_trans.tilt_x - geo_ref.tilt_x)
            true_tilt_y = rad2deg(geo_trans.tilt_y - geo_ref.tilt_y)
            true_tilt_total = rad2deg(acos(clamp(dot(geo_ref.normal, geo_trans.normal), -1.0, 1.0)))
            true_djunc = rad2deg(geo_trans.junction_phi - geo_ref.junction_phi)

            if r_fit !== nothing
                rec_dR = (r_fit.R0 - geo_ref.R0) * 1000
                @printf(io, "  ΔR [mm]       │ %11.3f │ %11.3f │ %9.3f\n", true_dR, rec_dR, rec_dR - true_dR)
            end
            if z_fit !== nothing
                rec_dZ = (z_fit.Z0 - geo_ref.Z0) * 1000
                @printf(io, "  ΔZ [mm]       │ %11.3f │ %11.3f │ %9.3f\n", true_dZ, rec_dZ, rec_dZ - true_dZ)
            end
            if tilt_fit !== nothing
                rec_tx = rad2deg(tilt_fit.tilt_about_x)
                rec_ty = rad2deg(tilt_fit.tilt_about_y)
                @printf(io, "  Tilt X [°]    │ %11.4f │ %11.4f │ %9.4f\n", true_tilt_x, rec_tx, rec_tx - true_tilt_x)
                @printf(io, "  Tilt Y [°]    │ %11.4f │ %11.4f │ %9.4f\n", true_tilt_y, rec_ty, rec_ty - true_tilt_y)
            end
            if junction_fit !== nothing
                rec_djunc = rad2deg(junction_fit.junction_phi - geo_ref.junction_phi)
                @printf(io, "  Δφ_junc [°]   │ %11.2f │ %11.2f │ %9.2f\n", true_djunc, rec_djunc, rec_djunc - true_djunc)
            end
        end
    end
    println("  Written: $filepath")
end

# ═══════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════
function plot_results(probes, grid, geo_ref, geo_trans,
                      r_fit, z_fit, tilt_fit, junction_fit)
    println("\n  Creating plots...")
    figures = []

    # ── Plot 1: R-Z with fitted center ───────────────────
    fig1 = Figure(size=(1200, 1000), backgroundcolor=:white)
    Label(fig1[0, 1:2], "PF Coil Centering — R-Z Plane"; fontsize=28, halign=:center)
    rowsize!(fig1.layout, 0, Fixed(35))

    ax1 = Axis(fig1[1, 1]; xlabel="R [m]", ylabel="Z [m]", aspect=DataAspect())

    hp = scatter!(ax1, probes.R, probes.Z;
                  color=probes.B_mag, colormap=:inferno,
                  colorrange=(0.0, max(quantile(probes.B_mag, 0.98), 1e-30)),
                  markersize=12)
    Colorbar(fig1[1, 2], hp; label="|B| [T]", width=20)

    if geo_ref !== nothing
        scatter!(ax1, [geo_ref.R0], [geo_ref.Z0]; color=:cyan, markersize=30,
                 marker=:diamond, strokecolor=:black, strokewidth=2.5)
        scatter!(ax1, [NaN], [NaN]; color=:cyan, markersize=16, marker=:diamond,
                 strokecolor=:black, strokewidth=1, label="Reference")
    end
    if geo_trans !== nothing
        scatter!(ax1, [geo_trans.R0], [geo_trans.Z0]; color=:orange, markersize=30,
                 marker=:utriangle, strokecolor=:black, strokewidth=2.5)
        scatter!(ax1, [NaN], [NaN]; color=:orange, markersize=16, marker=:utriangle,
                 strokecolor=:black, strokewidth=1, label="Transformed")
    end

    # Fitted center
    if r_fit !== nothing && z_fit !== nothing
        scatter!(ax1, [r_fit.R0], [z_fit.Z0]; color=:lime, markersize=28,
                 marker=:cross, strokecolor=:black, strokewidth=2)
        scatter!(ax1, [NaN], [NaN]; color=:lime, markersize=16, marker=:cross,
                 label="Fitted (R₀, Z₀)")
    end

    axislegend(ax1; position=:lt, labelsize=14)
    push!(figures, ("rz_centering", fig1))

    # ── Plot 2: Z₀(φ) with tilt fit ─────────────────────
    if tilt_fit !== nothing
        fig2 = Figure(size=(1100, 700), backgroundcolor=:white)
        Label(fig2[0, 1], "Z₀(φ) — Tilt Signature"; fontsize=28, halign=:center)
        rowsize!(fig2.layout, 0, Fixed(35))

        ax2 = Axis(fig2[1, 1]; xlabel="φ [°]", ylabel="Z crossing [mm]")

        phi_deg = rad2deg.(tilt_fit.phi_v)
        Z_mm   = tilt_fit.Z_v .* 1000

        scatter!(ax2, phi_deg, Z_mm; markersize=10, color=:steelblue, label="Data")

        # Fitted curve
        phi_dense = range(0, 360, length=200)
        Z_fit_dense = (tilt_fit.Z0 .+ tilt_fit.A_cos .* cos.(deg2rad.(phi_dense)) .+
                       tilt_fit.A_sin .* sin.(deg2rad.(phi_dense))) .* 1000
        lines!(ax2, phi_dense, Z_fit_dense; color=:red, linewidth=3,
               label="Fit: Z₀ + A cos φ + B sin φ")

        # Reference line
        if z_fit !== nothing
            hlines!(ax2, [z_fit.Z0 * 1000]; color=:gray50, linewidth=1, linestyle=:dash,
                    label="Mean Z₀")
        end

        axislegend(ax2; position=:lt, labelsize=14)
        push!(figures, ("z_vs_phi", fig2))
    end

    # ── Plot 3: |B|_n=0 vs R at different Z ─────────────
    if r_fit !== nothing
        fig3 = Figure(size=(1100, 700), backgroundcolor=:white)
        Label(fig3[0, 1], "|B|ₙ₌₀ vs R (toroidally averaged)"; fontsize=28, halign=:center)
        rowsize!(fig3.layout, 0, Fixed(35))

        ax3 = Axis(fig3[1, 1]; xlabel="R [m]", ylabel="|B|ₙ₌₀ [T]")

        n_z = grid.n_z
        colors = range(colorant"blue", colorant"red", length=n_z)

        for iz in 1:n_z
            profile = r_fit.Bmag_n0[:, iz]
            valid = r_fit.counts[:, iz] .> 0
            any(valid) || continue
            lines!(ax3, grid.R_vals[valid], profile[valid];
                   color=colors[iz], linewidth=2,
                   label=iz == 1 || iz == n_z ? "Z=$(round(grid.Z_vals[iz], digits=3)) m" : "")
        end

        vlines!(ax3, [r_fit.R0]; color=:lime, linewidth=3, linestyle=:dash,
                label="Fitted R₀")

        if geo_trans !== nothing
            vlines!(ax3, [geo_trans.R0]; color=:orange, linewidth=2, linestyle=:dot,
                    label="True R₀")
        end

        axislegend(ax3; position=:rt, labelsize=14)
        push!(figures, ("B_vs_R", fig3))
    end

    # ── Plot 4: Perturbation vs φ (junction detection) ───
    if junction_fit !== nothing
        fig4 = Figure(size=(1100, 600), backgroundcolor=:white)
        Label(fig4[0, 1], "Non-Axisymmetric Perturbation vs φ"; fontsize=28, halign=:center)
        rowsize!(fig4.layout, 0, Fixed(35))

        ax4 = Axis(fig4[1, 1]; xlabel="φ [°]", ylabel="Relative perturbation [%]")

        phi_deg = rad2deg.(grid.phi_vals)
        pert_pct = junction_fit.perturbation_vs_phi .* 100

        barplot!(ax4, phi_deg, pert_pct; color=:steelblue, strokecolor=:black, strokewidth=0.5)
        vlines!(ax4, [rad2deg(junction_fit.junction_phi)]; color=:red, linewidth=3,
                linestyle=:dash, label="Detected junction")

        if geo_trans !== nothing
            vlines!(ax4, [rad2deg(geo_trans.junction_phi)]; color=:orange, linewidth=2,
                    linestyle=:dot, label="True junction")
        end

        axislegend(ax4; position=:rt, labelsize=14)
        push!(figures, ("junction_detection", fig4))
    end

    # ── Plot 5: Benchmarking bar chart ───────────────────
    if geo_ref !== nothing && geo_trans !== nothing
        labels = String[]
        true_vals = Float64[]
        rec_vals  = Float64[]

        if r_fit !== nothing
            push!(labels, "ΔR [mm]")
            push!(true_vals, (geo_trans.R0 - geo_ref.R0) * 1000)
            push!(rec_vals, (r_fit.R0 - geo_ref.R0) * 1000)
        end
        if z_fit !== nothing
            push!(labels, "ΔZ [mm]")
            push!(true_vals, (geo_trans.Z0 - geo_ref.Z0) * 1000)
            push!(rec_vals, (z_fit.Z0 - geo_ref.Z0) * 1000)
        end
        if tilt_fit !== nothing
            true_tx = rad2deg(geo_trans.tilt_x - geo_ref.tilt_x)
            true_ty = rad2deg(geo_trans.tilt_y - geo_ref.tilt_y)
            push!(labels, "Tilt X [°]"); push!(true_vals, true_tx); push!(rec_vals, rad2deg(tilt_fit.tilt_about_x))
            push!(labels, "Tilt Y [°]"); push!(true_vals, true_ty); push!(rec_vals, rad2deg(tilt_fit.tilt_about_y))
        end

        if !isempty(labels)
            n = length(labels)
            fig5 = Figure(size=(1100, 700), backgroundcolor=:white)
            Label(fig5[0, 1], "Benchmarking: True vs Recovered"; fontsize=28, halign=:center)
            rowsize!(fig5.layout, 0, Fixed(35))

            ax5 = Axis(fig5[1, 1]; xlabel="", ylabel="Value",
                       xticks=(1:n, labels))

            barplot!(ax5, (1:n) .- 0.2, true_vals;
                     color=:gray70, strokecolor=:black, strokewidth=1, width=0.35, label="True")
            barplot!(ax5, (1:n) .+ 0.2, rec_vals;
                     color=:steelblue, strokecolor=:black, strokewidth=1, width=0.35, label="Recovered")

            hlines!(ax5, [0.0]; color=:black, linewidth=1)
            axislegend(ax5; position=:lt, labelsize=16)
            push!(figures, ("benchmark", fig5))
        end
    end

    return figures
end

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
function main()
    println("═══════════════════════════════════════════════════════")
    println(" PF Coil Centering — Full Parameter Fit")
    println(" (R₀, Z₀, Tilt X/Y, Junction φ)")
    println("═══════════════════════════════════════════════════════\n")

    # ── Read data ────────────────────────────────────────
    probes = read_hall_probe_csv(HALL_DATA_FILE)
    grid   = build_probe_grid(probes)

    # ── Geometry ─────────────────────────────────────────
    geo_ref = nothing
    if !isempty(REFERENCE_COIL_FILES)
        println("\n═══ Reference ═══")
        geo_ref = coil_geometry(REFERENCE_COIL_FILES; label="Reference")
    end

    geo_trans = nothing
    if !isempty(TRANSFORMED_COIL_FILES)
        println("\n═══ Transformed ═══")
        geo_trans = coil_geometry(TRANSFORMED_COIL_FILES; label="Transformed")
    end

    # Sanity check
    if geo_ref !== nothing && geo_trans !== nothing
        if abs(geo_ref.R0 - geo_trans.R0) < 1e-10 && abs(geo_ref.Z0 - geo_trans.Z0) < 1e-10
            println("\n  ⚠  WARNING: Reference and Transformed are IDENTICAL!")
        end
    end

    # ── FIT 1: Z₀ from B_Z sign change ──────────────────
    z_fit = fit_Z_center_from_sign_change(grid)

    # ── FIT 2: R₀ from toroidal FFT n=0 peak ────────────
    r_fit = fit_R_center_from_FFT(grid)

    # ── FIT 3: X/Y tilt from Z₀(φ) sinusoid ─────────────
    R0_est = r_fit !== nothing ? r_fit.R0 : (geo_ref !== nothing ? geo_ref.R0 : 1.0)
    tilt_fit = fit_tilt_from_Z_variation(z_fit, grid, R0_est)

    # ── FIT 4: Junction from field spike ─────────────────
    junction_fit = fit_junction_from_field_spike(grid)

    # ── Summary ──────────────────────────────────────────
    combined_fit_summary(r_fit, z_fit, tilt_fit, junction_fit)

    # ── Benchmarking against known geometry ──────────────
    if geo_ref !== nothing && geo_trans !== nothing
        println("\n═══ BENCHMARKING ═══")
        true_dR = (geo_trans.R0 - geo_ref.R0) * 1000
        true_dZ = (geo_trans.Z0 - geo_ref.Z0) * 1000
        true_tx = rad2deg(geo_trans.tilt_x - geo_ref.tilt_x)
        true_ty = rad2deg(geo_trans.tilt_y - geo_ref.tilt_y)
        true_djunc = rad2deg(geo_trans.junction_phi - geo_ref.junction_phi)

        println("  True:      ΔR=$(round(true_dR, digits=3)) mm, ΔZ=$(round(true_dZ, digits=3)) mm")
        println("             Tilt X=$(round(true_tx, digits=4))°, Tilt Y=$(round(true_ty, digits=4))°")
        println("             Δφ_junction=$(round(true_djunc, digits=2))°")

        if r_fit !== nothing
            rec_dR = (r_fit.R0 - geo_ref.R0) * 1000
            @printf("  Recovered: ΔR=%.3f mm (error=%.3f mm)\n", rec_dR, rec_dR - true_dR)
        end
        if z_fit !== nothing
            rec_dZ = (z_fit.Z0 - geo_ref.Z0) * 1000
            @printf("  Recovered: ΔZ=%.3f mm (error=%.3f mm)\n", rec_dZ, rec_dZ - true_dZ)
        end
        if tilt_fit !== nothing
            rec_tx = rad2deg(tilt_fit.tilt_about_x)
            rec_ty = rad2deg(tilt_fit.tilt_about_y)
            @printf("  Recovered: Tilt X=%.4f° (error=%.4f°)\n", rec_tx, rec_tx - true_tx)
            @printf("  Recovered: Tilt Y=%.4f° (error=%.4f°)\n", rec_ty, rec_ty - true_ty)
        end
        if junction_fit !== nothing
            rec_djunc = rad2deg(junction_fit.junction_phi - geo_ref.junction_phi)
            @printf("  Recovered: Δφ_junction=%.2f° (error=%.2f°)\n", rec_djunc, rec_djunc - true_djunc)
        end
    end

    # ── Output ───────────────────────────────────────────
    write_results(OUTPUT_FILE, geo_ref, geo_trans, r_fit, z_fit, tilt_fit, junction_fit)

    figures = plot_results(probes, grid, geo_ref, geo_trans,
                           r_fit, z_fit, tilt_fit, junction_fit)

    for (name, fig) in figures
        display(GLMakie.Screen(), fig)
    end

    println("\nPress Enter to save and exit...")
    readline()

    for (name, fig) in figures
        outpath = replace(PLOT_FILE, ".png" => "_$(name).png")
        save(outpath, fig)
        println("Saved $name → $outpath")
    end
end

main()