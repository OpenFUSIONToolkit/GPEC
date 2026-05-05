using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using GeneralizedPerturbedEquilibrium
using LinearAlgebra
using Statistics
using Printf
using GLMakie

const FT = GeneralizedPerturbedEquilibrium.ForcingTerms
const compute_biot_savart_boundary! = FT.compute_biot_savart_boundary!
const read_coil_dat = FT.read_coil_dat

# ═══════════════════════════════════════════════════════════════
# USER INPUTS / ASSUMPTIONS
# ═══════════════════════════════════════════════════════════════

# Coil file used only to generate synthetic Hall measurements.
# The estimators below use only Hall probe positions and fields.
COIL_FILE = joinpath(@__DIR__, "sparc_pf1u.dat")
COIL_CURRENT_A = 100.0

OUT_NAME = "sparc_pf1u"
HALL_CSV = joinpath(@__DIR__, "hall_probe_$OUT_NAME.csv")

# Expected values are only for comparison printing.
# Set to NaN if unknown.
EXPECTED_DX_MM = 0
EXPECTED_DY_MM = 0
EXPECTED_DZ_MM = 0
EXPECTED_TILT_X_DEG = 0.0
EXPECTED_TILT_Y_DEG = 0.0

# Hall probe layout.
# These are user-specified sensor locations.
HALL_R_INNER_M = 0.736061
HALL_R_OUTER_M = 1.11807

HALL_N_PHI_INNER = 24
HALL_N_Z_INNER   = 22

HALL_N_PHI_OUTER = 20
HALL_N_Z_OUTER   = 10

# Z ranges for synthetic sensors.
# Choose a range that brackets the expected coil midplane.
#HALL_Z_CENTER_M = EXPECTED_DZ_MM * 1e-3
#HALL_Z_HALFSPAN_INNER_M = 0.40
#HALL_Z_HALFSPAN_OUTER_M = 0.20

HALL_Z_CENTER_M = 0.0
HALL_Z_HALFSPAN_INNER_M = 10.0
HALL_Z_HALFSPAN_OUTER_M = 10.0

# Small-misalignment assumptions:
#   Bφ(φ,Z) ≈ <B_R>(Z)/R * (dx*sinφ - dy*cosφ)
# for a nearly axisymmetric circular PF coil.
#
# If dx,dy come out with both signs flipped, set this to -1.
BPHI_SIGN = 1.0

# Tilt conversion assumption.
# B_R zero crossing gives:
#   Z_cross(φ) = Z0 + B*cosφ + C*sinφ
#
# Raw observable is amp = sqrt(B²+C²).
# To convert amp to angle, need an effective radius.
# If nominal design coil radius is known, use it here.
# Set to NaN to report only raw amplitude and shell-radius proxy.
TILT_CALIBRATION_RADIUS_M = 0.9259

# Analysis cuts.
BR_MIN_FRAC_FOR_XY = 0.10
BR_ZERO_NOISE_FRAC = 0.02

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

function wmean(x, w)
    W = sum(w)
    W <= 0 && return mean(x)
    return sum(w .* x) / W
end

function wstd(x, w)
    μ = wmean(x, w)
    W = sum(w)
    W <= 0 && return std(x)
    return sqrt(sum(w .* (x .- μ).^2) / W)
end

function fit_cossin(phi, y)
    M = [ones(length(phi)) cos.(phi) sin.(phi)]
    c = M \ y
    return c[1], c[2], c[3]
end

function find_zero_crossings_z(vals, z)
    out = Float64[]
    for i in 1:length(vals)-1
        if vals[i] == 0
            push!(out, z[i])
        elseif vals[i] * vals[i+1] < 0
            t = vals[i] / (vals[i] - vals[i+1])
            push!(out, z[i] + t * (z[i+1] - z[i]))
        end
    end
    return out
end

# ═══════════════════════════════════════════════════════════════
# Synthetic Hall data
# ═══════════════════════════════════════════════════════════════

function load_coil()
    isfile(COIL_FILE) || error("Coil file not found: $COIL_FILE")

    cs = read_coil_dat(COIL_FILE)
    cs.currents .= COIL_CURRENT_A

    println("\nLoaded coil:")
    println("  file = $COIL_FILE")
    println("  current = $COIL_CURRENT_A A")
    println("  ncoil=$(cs.ncoil), s=$(cs.s), nsec=$(cs.nsec)")

    return [cs]
end

function make_hall_data(coil_sets)
    phi_i = collect(range(0, 2π, length=HALL_N_PHI_INNER+1)[1:end-1])
    phi_o = collect(range(0, 2π, length=HALL_N_PHI_OUTER+1)[1:end-1])

    Zi = collect(range(
        HALL_Z_CENTER_M - HALL_Z_HALFSPAN_INNER_M,
        HALL_Z_CENTER_M + HALL_Z_HALFSPAN_INNER_M,
        length=HALL_N_Z_INNER,
    ))

    Zo = collect(range(
        HALL_Z_CENTER_M - HALL_Z_HALFSPAN_OUTER_M,
        HALL_Z_CENTER_M + HALL_Z_HALFSPAN_OUTER_M,
        length=HALL_N_Z_OUTER,
    ))

    N = HALL_N_PHI_INNER*HALL_N_Z_INNER + HALL_N_PHI_OUTER*HALL_N_Z_OUTER

    R = zeros(N)
    phi = zeros(N)
    Z = zeros(N)
    shell = zeros(Int, N)

    k = 1

    for iz in eachindex(Zi), ip in eachindex(phi_i)
        R[k] = HALL_R_INNER_M
        phi[k] = phi_i[ip]
        Z[k] = Zi[iz]
        shell[k] = 1
        k += 1
    end

    for iz in eachindex(Zo), ip in eachindex(phi_o)
        R[k] = HALL_R_OUTER_M
        phi[k] = phi_o[ip]
        Z[k] = Zo[iz]
        shell[k] = 2
        k += 1
    end

    B_R = zeros(N)
    B_phi = zeros(N)
    B_Z = zeros(N)

    println("\nComputing synthetic Hall measurements:")
    println("  inner: R=$HALL_R_INNER_M m, $HALL_N_PHI_INNER φ × $HALL_N_Z_INNER Z")
    println("  outer: R=$HALL_R_OUTER_M m, $HALL_N_PHI_OUTER φ × $HALL_N_Z_OUTER Z")
    println("  total: $N probes")

    compute_biot_savart_boundary!(B_R, B_phi, B_Z, R, phi, Z, coil_sets)

    B_mag = sqrt.(B_R.^2 .+ B_phi.^2 .+ B_Z.^2)

    println("  |B| range = [$(minimum(B_mag)), $(maximum(B_mag))] T")
    println("  Bφ range  = [$(minimum(B_phi)), $(maximum(B_phi))] T")

    return (
        R=R,
        phi=phi,
        Z=Z,
        x=R .* cos.(phi),
        y=R .* sin.(phi),
        shell=shell,
        B_R=B_R,
        B_phi=B_phi,
        B_Z=B_Z,
        B_mag=B_mag,
        Zi=Zi,
        Zo=Zo,
        phi_i=phi_i,
        phi_o=phi_o,
    )
end

function export_hall_csv(h)
    open(HALL_CSV, "w") do io
        println(io, "x,y,z,R,phi,shell,B_R,B_phi,B_Z,B_mag")
        for i in eachindex(h.R)
            println(io,
                "$(h.x[i]),$(h.y[i]),$(h.Z[i]),$(h.R[i]),$(h.phi[i]),$(h.shell[i])," *
                "$(h.B_R[i]),$(h.B_phi[i]),$(h.B_Z[i]),$(h.B_mag[i])"
            )
        end
    end
    println("\nExported Hall CSV:")
    println("  $HALL_CSV")
end

# ═══════════════════════════════════════════════════════════════
# XY shift from Bφ / <BR>
# ═══════════════════════════════════════════════════════════════

function estimate_xy_shift(h; shell_id::Int)
    if shell_id == 1
        nphi = HALL_N_PHI_INNER
        nz = HALL_N_Z_INNER
        R_shell = HALL_R_INNER_M
        Zgrid = h.Zi
        phigrid = h.phi_i
    else
        nphi = HALL_N_PHI_OUTER
        nz = HALL_N_Z_OUTER
        R_shell = HALL_R_OUTER_M
        Zgrid = h.Zo
        phigrid = h.phi_o
    end

    m = h.shell .== shell_id

    BR = reshape(h.B_R[m], nphi, nz)
    BP = reshape(h.B_phi[m], nphi, nz)

    Br0 = [mean(BR[:, iz]) for iz in 1:nz]
    Br_cut = BR_MIN_FRAC_FOR_XY * maximum(abs.(Br0))

    dx = fill(NaN, nz)
    dy = fill(NaN, nz)
    valid = falses(nz)

    for iz in 1:nz
        abs(Br0[iz]) < Br_cut && continue

        _, a, b = fit_cossin(phigrid, BP[:, iz])

        # Bφ = <BR>/R * (dx*sinφ - dy*cosφ)
        dx[iz] =  BPHI_SIGN * b * R_shell / Br0[iz]
        dy[iz] = -BPHI_SIGN * a * R_shell / Br0[iz]

        valid[iz] = true
    end

    w = abs.(Br0[valid]).^2

    dxm = wmean(dx[valid], w)
    dym = wmean(dy[valid], w)
    dm = hypot(dxm, dym)
    ph = atan(dym, dxm)

    dxs = wstd(dx[valid], w)
    dys = wstd(dy[valid], w)

    println("\nXY shift estimate, shell $shell_id:")
    @printf("  dx = %.6f mm\n", dxm*1e3)
    @printf("  dy = %.6f mm\n", dym*1e3)
    @printf("  |d| = %.6f mm\n", dm*1e3)
    @printf("  φ = %.6f deg\n", rad2deg(ph))
    @printf("  scatter dx = %.6f mm\n", dxs*1e3)
    @printf("  scatter dy = %.6f mm\n", dys*1e3)

    if !isnan(EXPECTED_DX_MM) && !isnan(EXPECTED_DY_MM)
        @printf("  error dx = %.6f mm\n", dxm*1e3 - EXPECTED_DX_MM)
        @printf("  error dy = %.6f mm\n", dym*1e3 - EXPECTED_DY_MM)
    end

    return (
        shell=shell_id,
        Z=Zgrid,
        valid=valid,
        dx=dx,
        dy=dy,
        dx_mean=dxm,
        dy_mean=dym,
        d_mean=dm,
        phi_mean=ph,
        dx_std=dxs,
        dy_std=dys,
    )
end

# ═══════════════════════════════════════════════════════════════
# Z shift and tilt from BR zero crossings
# ═══════════════════════════════════════════════════════════════

function estimate_z_and_tilt(h; shell_id::Int=1)
    if shell_id == 1
        nphi = HALL_N_PHI_INNER
        nz = HALL_N_Z_INNER
        R_shell = HALL_R_INNER_M
        Zgrid = h.Zi
        phigrid = h.phi_i
    else
        nphi = HALL_N_PHI_OUTER
        nz = HALL_N_Z_OUTER
        R_shell = HALL_R_OUTER_M
        Zgrid = h.Zo
        phigrid = h.phi_o
    end

    m = h.shell .== shell_id
    BR = reshape(h.B_R[m], nphi, nz)

    noise = BR_ZERO_NOISE_FRAC * maximum(abs.(BR))

    zcross = fill(NaN, nphi)
    valid = falses(nphi)

    z_guess = mean(Zgrid)

    # ── Method 1: per-φ BR zero crossings ─────────────────
    for ip in 1:nphi
        col = BR[ip, :]

        maximum(abs.(col)) < noise && continue

        crossings = find_zero_crossings_z(col, Zgrid)
        isempty(crossings) && continue

        zcross[ip] = crossings[argmin(abs.(crossings .- z_guess))]
        valid[ip] = true
    end

    println("\nZ estimate / tilt from BR zero crossings, shell $shell_id:")
    println("  per-φ valid crossings = $(count(valid)) / $nphi")
    @printf("  sampled Z range = [%.3f, %.3f] mm\n", minimum(Zgrid)*1e3, maximum(Zgrid)*1e3)

    # ── If enough φ crossings exist, fit tilt sinusoid ─────
    if count(valid) >= 4
        phi_v = phigrid[valid]
        z_v = zcross[valid]

        M = [ones(length(phi_v)) cos.(phi_v) sin.(phi_v)]
        coeffs = M \ z_v

        Z0, Bcos, Csin = coeffs

        amp = hypot(Bcos, Csin)
        phase = atan(Csin, Bcos)
        zfit = M * coeffs
        rms = sqrt(mean((z_v .- zfit).^2))

        tilt_shell = atand(amp, R_shell)

        if isnan(TILT_CALIBRATION_RADIUS_M)
            tilt_cal = NaN
            tilt_x = NaN
            tilt_y = NaN
        else
            tilt_cal = atand(amp, TILT_CALIBRATION_RADIUS_M)

            # Small-angle approximation:
            # Z_cross ≈ Z0 - θy*Rcal*cosφ + θx*Rcal*sinφ
            tilt_x = atand(Csin, TILT_CALIBRATION_RADIUS_M)
            tilt_y = atand(-Bcos, TILT_CALIBRATION_RADIUS_M)
        end

        @printf("  Z0 = %.6f mm\n", Z0*1e3)
        @printf("  sinusoid amp = %.6f mm\n", amp*1e3)
        @printf("  phase = %.6f deg\n", rad2deg(phase))
        @printf("  tilt proxy using Hall radius = %.8f deg\n", tilt_shell)

        if !isnan(tilt_cal)
            @printf("  calibrated tilt magnitude = %.8f deg\n", tilt_cal)
            @printf("  calibrated tilt_x ≈ %.8f deg\n", tilt_x)
            @printf("  calibrated tilt_y ≈ %.8f deg\n", tilt_y)
        end

        @printf("  RMS zero-crossing fit = %.6f mm\n", rms*1e3)

        return (
            shell=shell_id,
            method="per_phi_BR_zero_crossing",
            Z0=Z0,
            Bcos=Bcos,
            Csin=Csin,
            amp=amp,
            phase=phase,
            tilt_shell=tilt_shell,
            tilt_cal=tilt_cal,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            rms=rms,
            zcross=zcross,
            valid=valid,
        )
    end

    # ── Method 2: fallback to <BR>(Z) zero crossing ─────────
    BR0 = [mean(BR[:, iz]) for iz in 1:nz]
    avg_crossings = find_zero_crossings_z(BR0, Zgrid)

    if !isempty(avg_crossings)
        Z0 = avg_crossings[argmin(abs.(avg_crossings .- z_guess))]

        println("  WARNING: not enough per-φ crossings for tilt.")
        println("  Falling back to <B_R>(Z) zero crossing.")
        @printf("  Z0 = %.6f mm\n", Z0*1e3)

        if !isnan(EXPECTED_DZ_MM)
            @printf("  error Z0 = %.6f mm\n", Z0*1e3 - EXPECTED_DZ_MM)
        end

        return (
            shell=shell_id,
            method="mean_BR_zero_crossing",
            Z0=Z0,
            Bcos=NaN,
            Csin=NaN,
            amp=NaN,
            phase=NaN,
            tilt_shell=NaN,
            tilt_cal=NaN,
            tilt_x=NaN,
            tilt_y=NaN,
            rms=NaN,
            zcross=zcross,
            valid=valid,
        )
    end

    # ── Method 3: no crossing available ────────────────────
    println("  WARNING: no usable BR zero crossing found.")
    println("  Cannot determine absolute Z0 from BR zero crossing in this Z range.")
    println("  Increase HALL_Z_HALFSPAN_INNER_M / OUTER_M or use a reference map.")

    return (
        shell=shell_id,
        method="failed",
        Z0=NaN,
        Bcos=NaN,
        Csin=NaN,
        amp=NaN,
        phase=NaN,
        tilt_shell=NaN,
        tilt_cal=NaN,
        tilt_x=NaN,
        tilt_y=NaN,
        rms=NaN,
        zcross=zcross,
        valid=valid,
    )
end

function print_summary_table(xy1, xy2, zt)
    rows = String[]

    function addrow(name, est, exp, unit)
        if isnan(exp)
            push!(rows, @sprintf("%-24s %14.6f %14s %14s %8s",
                name, est, "—", "—", unit))
        else
            err = est - exp
            push!(rows, @sprintf("%-24s %14.6f %14.6f %14.6f %8s",
                name, est, exp, err, unit))
        end
    end

    function separator!()
        push!(rows, "─"^86)
    end

    exp_d =
        isnan(EXPECTED_DX_MM) || isnan(EXPECTED_DY_MM) ?
        NaN :
        hypot(EXPECTED_DX_MM, EXPECTED_DY_MM)

    avg_dx = 0.5 * (xy1.dx_mean + xy2.dx_mean)
    avg_dy = 0.5 * (xy1.dy_mean + xy2.dy_mean)
    avg_d  = hypot(avg_dx, avg_dy)
    avg_phi = atan(avg_dy, avg_dx)

    println("\n" * "═"^86)
    println("SUMMARY TABLE")
    println("═"^86)
    println(@sprintf("%-24s %14s %14s %14s %8s",
        "quantity", "estimate", "expected", "error", "unit"))
    println("─"^86)

    addrow("shell1 dx", xy1.dx_mean * 1e3, EXPECTED_DX_MM, "mm")
    addrow("shell1 dy", xy1.dy_mean * 1e3, EXPECTED_DY_MM, "mm")
    addrow("shell1 |d|", xy1.d_mean * 1e3, exp_d, "mm")

    separator!()

    addrow("shell2 dx", xy2.dx_mean * 1e3, EXPECTED_DX_MM, "mm")
    addrow("shell2 dy", xy2.dy_mean * 1e3, EXPECTED_DY_MM, "mm")
    addrow("shell2 |d|", xy2.d_mean * 1e3, exp_d, "mm")

    separator!()

    addrow("avg dx", avg_dx * 1e3, EXPECTED_DX_MM, "mm")
    addrow("avg dy", avg_dy * 1e3, EXPECTED_DY_MM, "mm")
    addrow("avg |d|", avg_d * 1e3, exp_d, "mm")

    if isnan(EXPECTED_DX_MM) || isnan(EXPECTED_DY_MM)
        addrow("avg φ", rad2deg(avg_phi), NaN, "deg")
    else
        exp_phi = rad2deg(atan(EXPECTED_DY_MM, EXPECTED_DX_MM))
        addrow("avg φ", rad2deg(avg_phi), exp_phi, "deg")
    end

    separator!()

    addrow("Z0", zt.Z0 * 1e3, EXPECTED_DZ_MM, "mm")

    if !isnan(zt.tilt_cal)
        exp_tilt_mag =
            isnan(EXPECTED_TILT_X_DEG) || isnan(EXPECTED_TILT_Y_DEG) ?
            NaN :
            hypot(EXPECTED_TILT_X_DEG, EXPECTED_TILT_Y_DEG)

        addrow("tilt magnitude", zt.tilt_cal, exp_tilt_mag, "deg")
        addrow("tilt_x", zt.tilt_x, EXPECTED_TILT_X_DEG, "deg")
        addrow("tilt_y", zt.tilt_y, EXPECTED_TILT_Y_DEG, "deg")
    else
        addrow("tilt amp", zt.amp * 1e3, NaN, "mm")
    end

    addrow("BR ZC RMS", zt.rms * 1e3, 0.0, "mm")

    println(join(rows, "\n"))
    println("═"^86)
end
# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

function main()
    println("\nSynthetic Hall Probe Misalignment Analysis")
    println("  coil file = $COIL_FILE")
    println("  expected dx,dy,dz = $EXPECTED_DX_MM, $EXPECTED_DY_MM, $EXPECTED_DZ_MM mm")
    println("  expected tilt_x,tilt_y = $EXPECTED_TILT_X_DEG, $EXPECTED_TILT_Y_DEG deg")

    coils = load_coil()
    h = make_hall_data(coils)
    export_hall_csv(h)

    xy1 = estimate_xy_shift(h; shell_id=1)
    xy2 = estimate_xy_shift(h; shell_id=2)
    zt  = estimate_z_and_tilt(h; shell_id=1)

    print_summary_table(xy1, xy2, zt)
    println("\nDone.")
end

main()