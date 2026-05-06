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
# USER INPUTS
# ═══════════════════════════════════════════════════════════════

RUN_NAME = "sparc_pf1u_357_test"

# Physical/baseline coil: nominal geometry.
PHYSICAL_COIL_FILE = joinpath(@__DIR__, "sparc_pf1u.dat")

# Field coil: actual/instantiated coil used for the field calculation.
FIELD_COIL_FILE = joinpath(@__DIR__, "$(RUN_NAME).dat") # PHYSICAL_COIL_FILE

COIL_CURRENT_A = 100.0

HALL_CSV_FIELD    = joinpath(@__DIR__, "hall_probe_$(RUN_NAME)_FIELD_coil.csv")
HALL_CSV_PHYSICAL = joinpath(@__DIR__, "hall_probe_$(RUN_NAME)_PHYSICAL_coil.csv")
AXIS_PLOT_FILE    = joinpath(@__DIR__, "axis_comparison_$(RUN_NAME)_field_vs_physical.png")

# Hall input mode:
#   :internal -> compute Hall data internally from Biot-Savart and optionally export CSV
#   :csv      -> read precomputed Hall data from CSV
FIELD_HALL_MODE = :internal
#FIELD_HALL_MODE = :csv

FIELD_HALL_INPUT_CSV = joinpath(@__DIR__, "hall_probe_sparc_pf1u_357.csv") #HALL_CSV_FIELD

# Optional matching control for physical-coil field.
# Usually leave this as :internal.
PHYSICAL_HALL_MODE = :internal
# PHYSICAL_HALL_MODE = :csv

PHYSICAL_HALL_INPUT_CSV = HALL_CSV_PHYSICAL

HALL_R_INNER_FRAC = 0.40
HALL_R_OUTER_FRAC = 0.50
HALL_N_PHI_INNER = 24
HALL_N_Z_INNER   = 22
HALL_N_PHI_OUTER = 20
HALL_N_Z_OUTER   = 10
HALL_Z_HALFSPAN_INNER_M = 0.60
HALL_Z_HALFSPAN_OUTER_M = 0.40

# If NaN, use fitted physical baseline coil radius for converting Z sinusoid
# amplitude to tilt.
TILT_CALIBRATION_RADIUS_M = NaN

# Analysis settings
BR_ZERO_NOISE_FRAC = 0.0
BR_MIN_FRAC_FOR_FOURIER_XY = 0.10
BPHI_SIGN = 1.0
XY_MIN_STEP0_M = 0.050
XY_MIN_TOL_M = 1e-8

# Forward-model bias correction
BIAS_CORRECTION_N_ITER = 4
BIAS_CORRECTION_DAMPING = 0.6

# Plot controls
ENABLE_AXIS_PLOT = true
DISPLAY_AXIS_PLOT = true
SHOW_PHYSICAL_COILS = true
SHOW_FIELD_COILS = true
SHOW_HALL_PROBES = true
AXIS_PLOT_ZMIN_M = NaN
AXIS_PLOT_ZMAX_M = NaN

ENABLE_DETAILED_FIELD_SUMMARY = true

# ═══════════════════════════════════════════════════════════════
# Small helpers
# ═══════════════════════════════════════════════════════════════

wmean(x, w) = sum(w) <= 0 ? mean(x) : sum(w .* x) / sum(w)

function meanfinite(v)
    u = filter(isfinite, collect(v))
    return isempty(u) ? NaN : mean(u)
end

function fit_cossin(φ, y)
    c = hcat(ones(length(φ)), cos.(φ), sin.(φ)) \ y
    return c[1], c[2], c[3]
end

function zcrossings(vals, z)
    out = Float64[]

    for i in 1:(length(vals) - 1)
        vals[i] == 0 && push!(out, z[i])

        vals[i] * vals[i + 1] < 0 || continue

        t = vals[i] / (vals[i] - vals[i + 1])
        push!(out, z[i] + t * (z[i + 1] - z[i]))
    end

    return out
end

function axis_line(x0, y0, z0, tx_deg, ty_deg, z)
    tx = tand(tx_deg)
    ty = tand(ty_deg)

    x = x0 .+ ty .* (z .- z0)
    y = y0 .- tx .* (z .- z0)

    return x, y, z
end

shellmeta(h, shell) = shell == 1 ?
    (length(h.phi_i), h.nzi, h.R_inner, h.Zi, h.phi_i) :
    (length(h.phi_o), h.nzo, h.R_outer, h.Zo, h.phi_o)

# ═══════════════════════════════════════════════════════════════
# Coil geometry helpers
# ═══════════════════════════════════════════════════════════════

function segment_midpoints_lengths(coils)
    mx = Float64[]
    my = Float64[]
    mz = Float64[]
    w  = Float64[]

    for cs in coils, j in 1:cs.ncoil, k in 1:cs.s, l in 1:(cs.nsec - 1)
        x1 = cs.x[j, k, l]
        y1 = cs.y[j, k, l]
        z1 = cs.z[j, k, l]

        x2 = cs.x[j, k, l + 1]
        y2 = cs.y[j, k, l + 1]
        z2 = cs.z[j, k, l + 1]

        dlx = x2 - x1
        dly = y2 - y1
        dlz = z2 - z1

        ds = sqrt(dlx^2 + dly^2 + dlz^2)
        ds <= 0 && continue

        push!(mx, 0.5 * (x1 + x2))
        push!(my, 0.5 * (y1 + y2))
        push!(mz, 0.5 * (z1 + z2))
        push!(w, ds)
    end

    return mx, my, mz, w
end

function plane_basis_from_normal(n)
    ref = abs(n[3]) < 0.9 ? [0.0, 0.0, 1.0] : [1.0, 0.0, 0.0]

    e1 = normalize(cross(ref, n))
    e2 = cross(n, e1)

    return e1, e2
end

function coil_pose_from_segments(coils)
    mx, my, mz, w = segment_midpoints_lengths(coils)

    W = sum(w)
    W <= 0 && error("No valid coil segments found")

    cx = sum(w .* mx) / W
    cy = sum(w .* my) / W
    cz = sum(w .* mz) / W

    P = hcat(mx .- cx, my .- cy, mz .- cz)
    Pw = P .* sqrt.(w)

    _, _, V = svd(Pw)

    n = Vector(V[:, 3])
    n[3] < 0 && (n = -n)

    return (
        cx=cx,
        cy=cy,
        cz=cz,
        n=n,
        mx=mx,
        my=my,
        mz=mz,
        w=w,
    )
end

function collect_coil_xyz(coils)
    x = Float64[]
    y = Float64[]
    z = Float64[]

    for cs in coils, j in 1:cs.ncoil, k in 1:cs.s
        append!(x, vec(cs.x[j, k, :]))
        append!(y, vec(cs.y[j, k, :]))
        append!(z, vec(cs.z[j, k, :]))
    end

    return x, y, z
end

function _axis_from_coil_impl(coils)
    pose = coil_pose_from_segments(coils)

    cx, cy, cz = pose.cx, pose.cy, pose.cz
    n = pose.n

    # Convention:
    # normal = [tan(tilt_y), -tan(tilt_x), 1] / norm
    tx = atand(-n[2], n[3])
    ty = atand( n[1], n[3])

    e1, e2 = plane_basis_from_normal(n)

    dx = pose.mx .- cx
    dy = pose.my .- cy
    dz = pose.mz .- cz

    u = dx .* e1[1] .+ dy .* e1[2] .+ dz .* e1[3]
    v = dx .* e2[1] .+ dy .* e2[2] .+ dz .* e2[3]

    ρ = sqrt.(u.^2 .+ v.^2)
    rfit = sum(pose.w .* ρ) / sum(pose.w)

    x, y, z = collect_coil_xyz(coils)
    R = sqrt.(x.^2 .+ y.^2)

    return (
        x0=cx,
        y0=cy,
        z0=cz,
        tilt_x=tx,
        tilt_y=ty,
        tilt_mag=hypot(tx, ty),
        Rfit=rfit,
        Rmin=minimum(R),
        Rmax=maximum(R),
        Zmin=minimum(z),
        Zmax=maximum(z),
    )
end

function axis_from_coil(coils; label="Coil geometric axis")
    a = _axis_from_coil_impl(coils)

    println("\n$label:")
    @printf("  center = (%.3f, %.3f, %.3f) mm\n",
            a.x0 * 1e3, a.y0 * 1e3, a.z0 * 1e3)
    @printf("  tilt_x = %.6f deg, tilt_y = %.6f deg\n",
            a.tilt_x, a.tilt_y)
    @printf("  fit radius = %.3f mm\n", a.Rfit * 1e3)
    @printf("  R range = [%.3f, %.3f] mm\n",
            a.Rmin * 1e3, a.Rmax * 1e3)
    @printf("  Z range = [%.3f, %.3f] mm\n",
            a.Zmin * 1e3, a.Zmax * 1e3)

    return a
end

axis_from_coil_silent(coils) = _axis_from_coil_impl(coils)

# ═══════════════════════════════════════════════════════════════
# Rotation and coil placement
# ═══════════════════════════════════════════════════════════════

function rotation_between_vectors(a::Vector{Float64}, b::Vector{Float64})
    a = normalize(a)
    b = normalize(b)

    cosθ = clamp(dot(a, b), -1.0, 1.0)

    abs(cosθ - 1.0) < 1e-12 && return Matrix{Float64}(I, 3, 3)

    if abs(cosθ + 1.0) < 1e-10
        perp = abs(a[1]) < 0.9 ? [1.0, 0.0, 0.0] : [0.0, 1.0, 0.0]
        ax = normalize(cross(a, perp))
        return 2.0 * (ax * ax') - Matrix{Float64}(I, 3, 3)
    end

    k = normalize(cross(a, b))
    sinθ = sqrt(max(1.0 - cosθ^2, 0.0))

    K = [
         0.0  -k[3]   k[2]
         k[3]  0.0   -k[1]
        -k[2]  k[1]   0.0
    ]

    return Matrix{Float64}(I, 3, 3) + sinθ * K + (1.0 - cosθ) * K * K
end

function place_coil_at_axis(
    coils,
    target_x0::Float64,
    target_y0::Float64,
    target_z0::Float64,
    target_tx_deg::Float64,
    target_ty_deg::Float64,
)
    coils_new = deepcopy(coils)

    pose = coil_pose_from_segments(coils_new)

    cx, cy, cz = pose.cx, pose.cy, pose.cz
    n_orig = copy(pose.n)

    n_target = normalize([tand(target_ty_deg), -tand(target_tx_deg), 1.0])

    z_hat = [0.0, 0.0, 1.0]

    R_undo  = rotation_between_vectors(n_orig, z_hat)
    R_apply = rotation_between_vectors(z_hat, n_target)
    R_total = R_apply * R_undo

    for cs in coils_new
        for j in 1:cs.ncoil, k in 1:cs.s, l in 1:cs.nsec
            p = [
                cs.x[j, k, l] - cx,
                cs.y[j, k, l] - cy,
                cs.z[j, k, l] - cz,
            ]

            q = R_total * p

            cs.x[j, k, l] = q[1] + target_x0
            cs.y[j, k, l] = q[2] + target_y0
            cs.z[j, k, l] = q[3] + target_z0
        end
    end

    return coils_new
end

# ═══════════════════════════════════════════════════════════════
# Coil loading
# ═══════════════════════════════════════════════════════════════

function load_coils(file::AbstractString; label::String, current_A::Float64=COIL_CURRENT_A)
    isfile(file) || error("Coil file not found for $label: $file")

    cs = read_coil_dat(file)
    cs.currents .= current_A

    println("\nLoaded $label coil:")
    println("  file = $file")
    println("  ncoil=$(cs.ncoil), s=$(cs.s), nsec=$(cs.nsec), I=$current_A A")

    return [cs]
end

# ═══════════════════════════════════════════════════════════════
# Hall data generation, export, and CSV import
# ═══════════════════════════════════════════════════════════════

function make_hall_data(source_coils, phys_baseline; label="FIELD")
    R_inner = HALL_R_INNER_FRAC * phys_baseline.Rmin
    R_outer = HALL_R_OUTER_FRAC * phys_baseline.Rmax
    Zc = phys_baseline.z0

    φi = collect(range(0, 2π, length=HALL_N_PHI_INNER + 1)[1:end-1])
    φo = collect(range(0, 2π, length=HALL_N_PHI_OUTER + 1)[1:end-1])

    Zi = collect(range(
        Zc - HALL_Z_HALFSPAN_INNER_M,
        Zc + HALL_Z_HALFSPAN_INNER_M;
        length=HALL_N_Z_INNER,
    ))

    Zo = collect(range(
        Zc - HALL_Z_HALFSPAN_OUTER_M,
        Zc + HALL_Z_HALFSPAN_OUTER_M;
        length=HALL_N_Z_OUTER,
    ))

    N = HALL_N_PHI_INNER * HALL_N_Z_INNER +
        HALL_N_PHI_OUTER * HALL_N_Z_OUTER

    R = zeros(N)
    φ = zeros(N)
    Z = zeros(N)
    shell = zeros(Int, N)

    k = 1

    for iz in eachindex(Zi), ip in eachindex(φi)
        R[k] = R_inner
        φ[k] = φi[ip]
        Z[k] = Zi[iz]
        shell[k] = 1
        k += 1
    end

    for iz in eachindex(Zo), ip in eachindex(φo)
        R[k] = R_outer
        φ[k] = φo[ip]
        Z[k] = Zo[iz]
        shell[k] = 2
        k += 1
    end

    BR = zeros(N)
    BP = zeros(N)
    BZ = zeros(N)

    println("\nComputing synthetic Hall data from $label coil:")
    println("  probes = $N")
    @printf("  R1 = %.3f mm, Z1=[%.3f, %.3f] mm\n",
            R_inner * 1e3, first(Zi) * 1e3, last(Zi) * 1e3)
    @printf("  R2 = %.3f mm, Z2=[%.3f, %.3f] mm\n",
            R_outer * 1e3, first(Zo) * 1e3, last(Zo) * 1e3)

    compute_biot_savart_boundary!(BR, BP, BZ, R, φ, Z, source_coils)

    Bmag = sqrt.(BR.^2 .+ BP.^2 .+ BZ.^2)

    return (
        label=label,
        R=R,
        phi=φ,
        Z=Z,
        x=R .* cos.(φ),
        y=R .* sin.(φ),
        shell=shell,
        B_R=BR,
        B_phi=BP,
        B_Z=BZ,
        B_mag=Bmag,
        Zi=Zi,
        Zo=Zo,
        phi_i=φi,
        phi_o=φo,
        nzi=HALL_N_Z_INNER,
        nzo=HALL_N_Z_OUTER,
        R_inner=R_inner,
        R_outer=R_outer,
    )
end

function recompute_hall_field(h, coils; label=h.label)
    N = length(h.R)

    BR = zeros(N)
    BP = zeros(N)
    BZ = zeros(N)

    compute_biot_savart_boundary!(BR, BP, BZ, h.R, h.phi, h.Z, coils)

    Bmag = sqrt.(BR.^2 .+ BP.^2 .+ BZ.^2)

    return (
        label=label,
        R=h.R,
        phi=h.phi,
        Z=h.Z,
        x=h.x,
        y=h.y,
        shell=h.shell,
        B_R=BR,
        B_phi=BP,
        B_Z=BZ,
        B_mag=Bmag,
        Zi=h.Zi,
        Zo=h.Zo,
        phi_i=h.phi_i,
        phi_o=h.phi_o,
        nzi=h.nzi,
        nzo=h.nzo,
        R_inner=h.R_inner,
        R_outer=h.R_outer,
    )
end

function export_hall_csv(h, path::AbstractString)
    open(path, "w") do io
        println(io, "x,y,z,R,phi,shell,B_R,B_phi,B_Z,B_mag")

        for i in eachindex(h.R)
            println(io,
                "$(h.x[i]),$(h.y[i]),$(h.Z[i]),$(h.R[i]),$(h.phi[i])," *
                "$(h.shell[i]),$(h.B_R[i]),$(h.B_phi[i]),$(h.B_Z[i]),$(h.B_mag[i])"
            )
        end
    end

    println("\nExported Hall CSV: $path")
end

function _csv_col(header, choices; required=true)
    hlo = lowercase.(strip.(header))

    for name in choices
        j = findfirst(==(lowercase(name)), hlo)
        j !== nothing && return j
    end

    required && error("Missing required CSV column. Tried names: $(join(choices, ", "))")
    return nothing
end

function _csv_float_col(rows, idx)
    return [parse(Float64, rows[i][idx]) for i in eachindex(rows)]
end

function _csv_int_col(rows, idx)
    return [Int(round(parse(Float64, rows[i][idx]))) for i in eachindex(rows)]
end

function _sorted_unique(v)
    return sort(unique(collect(v)))
end

function read_hall_csv(path::AbstractString; label="FIELD-FIELD")
    isfile(path) || error("Hall CSV not found: $path")

    raw = readlines(path)
    raw = filter(line -> !isempty(strip(line)), raw)
    length(raw) >= 2 || error("Hall CSV has no data rows: $path")

    header = strip.(split(raw[1], ","))

    rows = Vector{Vector{SubString{String}}}()

    for line in raw[2:end]
        push!(rows, strip.(split(line, ",")))
    end

    N = length(rows)

    xidx    = _csv_col(header, ["x"]; required=false)
    yidx    = _csv_col(header, ["y"]; required=false)
    zidx    = _csv_col(header, ["z"], required=true)
    Ridx    = _csv_col(header, ["R", "r"], required=false)
    phiidx  = _csv_col(header, ["phi", "φ"], required=false)
    shidx   = _csv_col(header, ["shell"], required=true)
    BRidx   = _csv_col(header, ["B_R", "BR", "B_r", "br"], required=true)
    BPidx   = _csv_col(header, ["B_phi", "BPHI", "BP", "B_P", "Bphi", "b_phi"], required=true)
    BZidx   = _csv_col(header, ["B_Z", "BZ", "B_z", "bz"], required=true)
    Bmagidx = _csv_col(header, ["B_mag", "Bmag", "B", "B_abs"], required=false)

    Z = _csv_float_col(rows, zidx)
    shell = _csv_int_col(rows, shidx)

    if Ridx === nothing || phiidx === nothing
        xidx === nothing && error("CSV must contain either R/phi or x/y. Missing x.")
        yidx === nothing && error("CSV must contain either R/phi or x/y. Missing y.")

        x = _csv_float_col(rows, xidx)
        y = _csv_float_col(rows, yidx)

        R = sqrt.(x.^2 .+ y.^2)
        phi = mod.(atan.(y, x), 2π)
    else
        R = _csv_float_col(rows, Ridx)
        phi = mod.(_csv_float_col(rows, phiidx), 2π)

        x = xidx === nothing ? R .* cos.(phi) : _csv_float_col(rows, xidx)
        y = yidx === nothing ? R .* sin.(phi) : _csv_float_col(rows, yidx)
    end

    BR = _csv_float_col(rows, BRidx)
    BP = _csv_float_col(rows, BPidx)
    BZ = _csv_float_col(rows, BZidx)

    Bmag = Bmagidx === nothing ?
        sqrt.(BR.^2 .+ BP.^2 .+ BZ.^2) :
        _csv_float_col(rows, Bmagidx)

    # Reorder into make_hall_data convention:
    # shell, then Z, then phi. This matches reshape(..., nphi, nz).
    perm = sortperm(1:N; by=i -> (shell[i], Z[i], phi[i]))

    R = R[perm]
    phi = phi[perm]
    Z = Z[perm]
    x = x[perm]
    y = y[perm]
    shell = shell[perm]
    BR = BR[perm]
    BP = BP[perm]
    BZ = BZ[perm]
    Bmag = Bmag[perm]

    mask1 = shell .== 1
    mask2 = shell .== 2

    any(mask1) || error("Hall CSV has no shell == 1 rows.")
    any(mask2) || error("Hall CSV has no shell == 2 rows.")

    Zi = _sorted_unique(Z[mask1])
    Zo = _sorted_unique(Z[mask2])
    phi_i = _sorted_unique(phi[mask1])
    phi_o = _sorted_unique(phi[mask2])

    nzi = length(Zi)
    nzo = length(Zo)

    expected1 = length(phi_i) * nzi
    expected2 = length(phi_o) * nzo

    count(mask1) == expected1 || error(
        "Shell 1 CSV grid is not rectangular: count=$(count(mask1)), expected=$expected1"
    )

    count(mask2) == expected2 || error(
        "Shell 2 CSV grid is not rectangular: count=$(count(mask2)), expected=$expected2"
    )

    R_inner = mean(R[mask1])
    R_outer = mean(R[mask2])

    println("\nLoaded precomputed Hall CSV for $label:")
    println("  file = $path")
    println("  probes = $N")
    @printf("  shell 1: nphi=%d, nz=%d, R≈%.3f mm\n",
            length(phi_i), nzi, R_inner * 1e3)
    @printf("  shell 2: nphi=%d, nz=%d, R≈%.3f mm\n",
            length(phi_o), nzo, R_outer * 1e3)

    return (
        label=label,
        R=R,
        phi=phi,
        Z=Z,
        x=x,
        y=y,
        shell=shell,
        B_R=BR,
        B_phi=BP,
        B_Z=BZ,
        B_mag=Bmag,
        Zi=Zi,
        Zo=Zo,
        phi_i=phi_i,
        phi_o=phi_o,
        nzi=nzi,
        nzo=nzo,
        R_inner=R_inner,
        R_outer=R_outer,
    )
end

# ═══════════════════════════════════════════════════════════════
# Magnetic Z and tilt from B_R zero crossings
# ═══════════════════════════════════════════════════════════════

function magnetic_z_tilt(h, phys_ref; shell=1, verbose=true)
    nφ, nz, Rsh, Zgrid, φgrid = shellmeta(h, shell)
    BR = reshape(h.B_R[h.shell .== shell], nφ, nz)

    zc = fill(NaN, nφ)
    valid = falses(nφ)

    noise = BR_ZERO_NOISE_FRAC * maximum(abs.(BR))
    zguess = phys_ref.z0

    for ip in 1:nφ
        maximum(abs.(BR[ip, :])) < noise && continue

        c = zcrossings(BR[ip, :], Zgrid)
        isempty(c) && continue

        zc[ip] = c[argmin(abs.(c .- zguess))]
        valid[ip] = true
    end

    Rcal = isfinite(TILT_CALIBRATION_RADIUS_M) ?
        TILT_CALIBRATION_RADIUS_M :
        phys_ref.Rfit

    if count(valid) >= 4
        M = hcat(ones(count(valid)), cos.(φgrid[valid]), sin.(φgrid[valid]))
        Z0, Bc, Cs = M \ zc[valid]

        fit = M * [Z0, Bc, Cs]

        amp = hypot(Bc, Cs)
        rms = sqrt(mean((zc[valid] .- fit).^2))

        tx = atand(Cs, Rcal)
        ty = atand(-Bc, Rcal)
        tmag = atand(amp, Rcal)

        if verbose
            println("\nB_R zero-crossing sinusoid, $(h.label), shell $shell:")
            @printf("  valid = %d/%d, Z0 = %.6f mm, tilt_x = %.6f deg, tilt_y = %.6f deg\n",
                    count(valid), nφ, Z0 * 1e3, tx, ty)
        end

        return (
            shell=shell,
            R=Rsh,
            x0=NaN,
            y0=NaN,
            z0=Z0,
            tilt_x=tx,
            tilt_y=ty,
            tilt_mag=tmag,
            Bcos=Bc,
            Csin=Cs,
            amp=amp,
            rms=rms,
            zcross=zc,
            valid=valid,
            method="per_phi",
        )
    end

    BR0 = [mean(BR[:, iz]) for iz in 1:nz]
    c = zcrossings(BR0, Zgrid)
    Z0 = isempty(c) ? NaN : c[argmin(abs.(c .- zguess))]

    if verbose
        println("\nB_R zero-crossing fallback, $(h.label), shell $shell:")
        if isnan(Z0)
            @printf("  valid = %d/%d, Z0 = NaN\n", count(valid), nφ)
        else
            @printf("  valid = %d/%d, Z0 = %.6f mm\n", count(valid), nφ, Z0 * 1e3)
        end
    end

    return (
        shell=shell,
        R=Rsh,
        x0=NaN,
        y0=NaN,
        z0=Z0,
        tilt_x=NaN,
        tilt_y=NaN,
        tilt_mag=NaN,
        Bcos=NaN,
        Csin=NaN,
        amp=NaN,
        rms=NaN,
        zcross=zc,
        valid=valid,
        method=isempty(c) ? "failed" : "mean_BR",
    )
end

# ═══════════════════════════════════════════════════════════════
# Magnetic X/Y from local B_phi minimization
# ═══════════════════════════════════════════════════════════════

function B_cartesian(h)
    c = cos.(h.phi)
    s = sin.(h.phi)

    Bx = h.B_R .* c .- h.B_phi .* s
    By = h.B_R .* s .+ h.B_phi .* c
    Bz = h.B_Z

    return Bx, By, Bz
end

function local_bphi_objective(h, idxs, x0, y0, z0, tx_deg, ty_deg)
    Bx, By, Bz = B_cartesian(h)

    d = normalize([tand(ty_deg), -tand(tx_deg), 1.0])

    acc = 0.0
    n = 0

    for i in idxs
        qx = h.x[i] - x0
        qy = h.y[i] - y0
        qz = h.Z[i] - z0

        s = qx * d[1] + qy * d[2] + qz * d[3]

        rx = qx - s * d[1]
        ry = qy - s * d[2]
        rz = qz - s * d[3]

        ρ = sqrt(rx^2 + ry^2 + rz^2)
        ρ < 1e-9 && continue

        rhat = [rx / ρ, ry / ρ, rz / ρ]
        eφ = cross(d, rhat)

        bp = Bx[i] * eφ[1] + By[i] * eφ[2] + Bz[i] * eφ[3]
        bn = max(h.B_mag[i], 1e-30)

        acc += (bp / bn)^2
        n += 1
    end

    return acc / max(n, 1)
end

function magnetic_xy_minimize(h, phys_ref, zt; shell=1, verbose=true)
    idxs = findall(h.shell .== shell)

    z0 = isfinite(zt.z0) ? zt.z0 : phys_ref.z0
    tx = isfinite(zt.tilt_x) ? zt.tilt_x : phys_ref.tilt_x
    ty = isfinite(zt.tilt_y) ? zt.tilt_y : phys_ref.tilt_y

    x = phys_ref.x0
    y = phys_ref.y0

    step = XY_MIN_STEP0_M
    best = local_bphi_objective(h, idxs, x, y, z0, tx, ty)

    dirs = [
        ( 1.0,  0.0),
        (-1.0,  0.0),
        ( 0.0,  1.0),
        ( 0.0, -1.0),
        ( 1.0,  1.0),
        ( 1.0, -1.0),
        (-1.0,  1.0),
        (-1.0, -1.0),
    ]

    while step > XY_MIN_TOL_M
        improved = false

        for (dx, dy) in dirs
            xn = x + step * dx
            yn = y + step * dy

            val = local_bphi_objective(h, idxs, xn, yn, z0, tx, ty)

            if val < best
                x, y, best = xn, yn, val
                improved = true
            end
        end

        improved || (step *= 0.5)
    end

    if verbose
        println("\nLocal B_phi minimization, $(h.label), shell $shell:")
        @printf("  x0 = %.6f mm, y0 = %.6f mm, objective = %.6e\n",
                x * 1e3, y * 1e3, best)
    end

    return (
        shell=shell,
        R=zt.R,
        x0=x,
        y0=y,
        z0=z0,
        tilt_x=tx,
        tilt_y=ty,
        tilt_mag=hypot(tx, ty),
        objective=best,
    )
end

# ═══════════════════════════════════════════════════════════════
# Fourier magnetic X/Y
# ═══════════════════════════════════════════════════════════════

function magnetic_xy_fourier(h; shell=1, verbose=true)
    nφ, nz, Rsh, Zgrid, φgrid = shellmeta(h, shell)

    BR = reshape(h.B_R[h.shell .== shell], nφ, nz)
    BP = reshape(h.B_phi[h.shell .== shell], nφ, nz)

    BR0 = [mean(BR[:, iz]) for iz in 1:nz]
    cut = BR_MIN_FRAC_FOR_FOURIER_XY * maximum(abs.(BR0))

    x = fill(NaN, nz)
    y = fill(NaN, nz)
    valid = falses(nz)

    for iz in 1:nz
        abs(BR0[iz]) < cut && continue

        _, a, b = fit_cossin(φgrid, BP[:, iz])

        # B_phi ≈ <B_R>/R * (x0*sinφ - y0*cosφ)
        x[iz] =  BPHI_SIGN * b * Rsh / BR0[iz]
        y[iz] = -BPHI_SIGN * a * Rsh / BR0[iz]

        valid[iz] = true
    end

    count(valid) == 0 && return (
        shell=shell,
        R=Rsh,
        x0=NaN,
        y0=NaN,
    )

    w = abs.(BR0[valid]).^2

    x0 = wmean(x[valid], w)
    y0 = wmean(y[valid], w)

    if verbose
        println("\nFourier B_phi n=1 XY, $(h.label), shell $shell:")
        @printf("  x0 = %.6f mm, y0 = %.6f mm\n", x0 * 1e3, y0 * 1e3)
    end

    return (
        shell=shell,
        R=Rsh,
        x0=x0,
        y0=y0,
    )
end

function combine_axis(xy, zt)
    return (
        shell=xy.shell,
        R=xy.R,
        x0=xy.x0,
        y0=xy.y0,
        z0=zt.z0,
        tilt_x=zt.tilt_x,
        tilt_y=zt.tilt_y,
        tilt_mag=hypot(zt.tilt_x, zt.tilt_y),
    )
end

function average_axes(a1, a2)
    tx = meanfinite([a1.tilt_x, a2.tilt_x])
    ty = meanfinite([a1.tilt_y, a2.tilt_y])

    return (
        shell=0,
        R=NaN,
        x0=meanfinite([a1.x0, a2.x0]),
        y0=meanfinite([a1.y0, a2.y0]),
        z0=meanfinite([a1.z0, a2.z0]),
        tilt_x=tx,
        tilt_y=ty,
        tilt_mag=hypot(tx, ty),
    )
end

# ═══════════════════════════════════════════════════════════════
# Complete magnetic-axis analysis for one source coil / Hall dataset
# ═══════════════════════════════════════════════════════════════

function run_analysis_pipeline(h, phys_ref; verbose=false)
    zt1 = magnetic_z_tilt(h, phys_ref; shell=1, verbose=verbose)
    zt2 = magnetic_z_tilt(h, phys_ref; shell=2, verbose=verbose)

    sin1 = magnetic_xy_minimize(h, phys_ref, zt1; shell=1, verbose=verbose)
    sin2 = magnetic_xy_minimize(h, phys_ref, zt2; shell=2, verbose=verbose)
    sinavg = average_axes(sin1, sin2)

    fxy1 = magnetic_xy_fourier(h; shell=1, verbose=verbose)
    fxy2 = magnetic_xy_fourier(h; shell=2, verbose=verbose)

    fou1 = combine_axis(fxy1, zt1)
    fou2 = combine_axis(fxy2, zt2)
    fouavg = average_axes(fou1, fou2)

    return (
        zt1=zt1,
        zt2=zt2,
        sin1=sin1,
        sin2=sin2,
        sinavg=sinavg,
        fxy1=fxy1,
        fxy2=fxy2,
        fou1=fou1,
        fou2=fou2,
        fouavg=fouavg,
    )
end

function analyze_source_coil(
    label::String,
    source_coils,
    phys_baseline;
    csv_path::Union{Nothing,AbstractString}=nothing,
    hall_mode::Symbol=:internal,
    hall_input_csv::Union{Nothing,AbstractString}=nothing,
    verbose=true,
)
    h = if hall_mode == :internal
        h_internal = make_hall_data(source_coils, phys_baseline; label=label)

        csv_path !== nothing && export_hall_csv(h_internal, csv_path)

        h_internal

    elseif hall_mode == :csv
        input_path = hall_input_csv === nothing ? csv_path : hall_input_csv
        input_path === nothing && error("hall_mode=:csv requires hall_input_csv or csv_path.")

        read_hall_csv(input_path; label=label)

    else
        error("Unknown hall_mode=$hall_mode. Use :internal or :csv.")
    end

    results = run_analysis_pipeline(h, phys_baseline; verbose=verbose)

    return (
        label=label,
        h=h,
        zt1=results.zt1,
        zt2=results.zt2,
        sin1=results.sin1,
        sin2=results.sin2,
        sinavg=results.sinavg,
        fxy1=results.fxy1,
        fxy2=results.fxy2,
        fou1=results.fou1,
        fou2=results.fou2,
        fouavg=results.fouavg,
    )
end

# ═══════════════════════════════════════════════════════════════
# Forward-model bias correction
# ═══════════════════════════════════════════════════════════════

function bias_correct_axis(
    model_coils,
    h,
    measured_zt1,
    measured_zt2,
    measured_sin1,
    measured_sin2;
    n_iter=BIAS_CORRECTION_N_ITER,
    damping=BIAS_CORRECTION_DAMPING,
    label="",
)
    println("\n" * "─"^70)

    if isempty(label)
        println("Forward-model bias correction using ASSUMED coil model  ($n_iter iterations, α=$damping)")
    else
        println("Forward-model bias correction for $label using ASSUMED coil model  ($n_iter iterations, α=$damping)")
    end

    println("─"^70)

    meas_x  = meanfinite([measured_sin1.x0,    measured_sin2.x0])
    meas_y  = meanfinite([measured_sin1.y0,    measured_sin2.y0])
    meas_z  = meanfinite([measured_zt1.z0,     measured_zt2.z0])
    meas_tx = meanfinite([measured_zt1.tilt_x, measured_zt2.tilt_x])
    meas_ty = meanfinite([measured_zt1.tilt_y, measured_zt2.tilt_y])

    if any(x -> !isfinite(x), [meas_x, meas_y, meas_z, meas_tx, meas_ty])
        @warn "bias_correct_axis: measured values contain NaN — skipping correction"
        return average_axes(measured_sin1, measured_sin2)
    end

    est_x  = meas_x
    est_y  = meas_y
    est_z  = meas_z
    est_tx = meas_tx
    est_ty = meas_ty

    corrected = (
        shell=0,
        R=NaN,
        x0=meas_x,
        y0=meas_y,
        z0=meas_z,
        tilt_x=meas_tx,
        tilt_y=meas_ty,
        tilt_mag=hypot(meas_tx, meas_ty),
    )

    for iter in 1:n_iter
        synth_coils = place_coil_at_axis(
            model_coils,
            est_x,
            est_y,
            est_z,
            est_tx,
            est_ty,
        )

        h_synth = recompute_hall_field(h, synth_coils; label="BIAS-MODEL-SYNTH")
        geom_synth = axis_from_coil_silent(synth_coils)

        synth_results = run_analysis_pipeline(h_synth, geom_synth; verbose=false)

        bias_x  = meanfinite([synth_results.sin1.x0,    synth_results.sin2.x0])    - geom_synth.x0
        bias_y  = meanfinite([synth_results.sin1.y0,    synth_results.sin2.y0])    - geom_synth.y0
        bias_z  = meanfinite([synth_results.zt1.z0,     synth_results.zt2.z0])     - geom_synth.z0
        bias_tx = meanfinite([synth_results.zt1.tilt_x, synth_results.zt2.tilt_x]) - geom_synth.tilt_x
        bias_ty = meanfinite([synth_results.zt1.tilt_y, synth_results.zt2.tilt_y]) - geom_synth.tilt_y

        corr_x  = meas_x  - bias_x
        corr_y  = meas_y  - bias_y
        corr_z  = meas_z  - bias_z
        corr_tx = meas_tx - bias_tx
        corr_ty = meas_ty - bias_ty

        @printf("  iter %d | bias     : Δx=%+.4f mm  Δy=%+.4f mm  Δz=%+.4f mm  Δtx=%+.6f°  Δty=%+.6f°\n",
                iter, bias_x * 1e3, bias_y * 1e3, bias_z * 1e3, bias_tx, bias_ty)
        @printf("  iter %d | corrected: x=%+.4f mm   y=%+.4f mm   z=%+.4f mm   tx=%+.6f°   ty=%+.6f°\n",
                iter, corr_x * 1e3, corr_y * 1e3, corr_z * 1e3, corr_tx, corr_ty)

        corrected = (
            shell=0,
            R=NaN,
            x0=corr_x,
            y0=corr_y,
            z0=corr_z,
            tilt_x=corr_tx,
            tilt_y=corr_ty,
            tilt_mag=hypot(corr_tx, corr_ty),
        )

        est_x  = est_x  + damping * (corr_x  - est_x)
        est_y  = est_y  + damping * (corr_y  - est_y)
        est_z  = est_z  + damping * (corr_z  - est_z)
        est_tx = est_tx + damping * (corr_tx - est_tx)
        est_ty = est_ty + damping * (corr_ty - est_ty)

        @printf("  iter %d | next est : x=%+.4f mm   y=%+.4f mm   z=%+.4f mm   tx=%+.6f°   ty=%+.6f°\n\n",
                iter, est_x * 1e3, est_y * 1e3, est_z * 1e3, est_tx, est_ty)
    end

    println("─"^70)

    return corrected
end

function print_axis_row(method, label, R, ref, a)
    dx = (a.x0 - ref.x0) * 1e3
    dy = (a.y0 - ref.y0) * 1e3
    dz = (a.z0 - ref.z0) * 1e3
    dtx = a.tilt_x - ref.tilt_x
    dty = a.tilt_y - ref.tilt_y

    Rmm = isfinite(R) ? R * 1e3 : NaN

    @printf(
        "%-18s %-12s %9.1f %11.3f %11.3f %11.3f %10.5f %10.5f %11.3f %11.3f %11.3f %10.5f %10.5f\n",
        method,
        label,
        Rmm,
        a.x0 * 1e3,
        a.y0 * 1e3,
        a.z0 * 1e3,
        a.tilt_x,
        a.tilt_y,
        dx,
        dy,
        dz,
        dtx,
        dty,
    )
end

function print_detailed_field_summary(phys_geom, field_geom, field_analysis, field_corrected)
    println("\n" * "═"^165)
    println("DETAILED FIELD-COIL ANALYSIS")
    println("Deltas in this table are relative to physical baseline geometry.")
    println("═"^165)

    @printf(
        "%-18s %-12s %9s %11s %11s %11s %10s %10s %11s %11s %11s %10s %10s\n",
        "method",
        "radius",
        "R[mm]",
        "x",
        "y",
        "z",
        "tilt_x",
        "tilt_y",
        "Δx",
        "Δy",
        "Δz",
        "Δtx",
        "Δty",
    )

    println("─"^165)

    print_axis_row("physical-base", "-", phys_geom.Rfit, phys_geom, phys_geom)
    print_axis_row("field-geom", "-", field_geom.Rfit, phys_geom, field_geom)

    println("─"^165)

    print_axis_row("sinusoid/local", "R1", field_analysis.sin1.R, phys_geom, field_analysis.sin1)
    print_axis_row("sinusoid/local", "R2", field_analysis.sin2.R, phys_geom, field_analysis.sin2)
    print_axis_row("sinusoid/local", "average", NaN, phys_geom, field_analysis.sinavg)

    println("─"^165)

    print_axis_row("fourier", "R1", field_analysis.fou1.R, phys_geom, field_analysis.fou1)
    print_axis_row("fourier", "R2", field_analysis.fou2.R, phys_geom, field_analysis.fou2)
    print_axis_row("fourier", "average", NaN, phys_geom, field_analysis.fouavg)

    println("─"^165)

    print_axis_row("bias-corrected", "field", NaN, phys_geom, field_corrected)

    println("─"^165)

    rms1_str = isfinite(field_analysis.zt1.rms) ? @sprintf("%.3f", field_analysis.zt1.rms * 1e3) : "NaN"
    rms2_str = isfinite(field_analysis.zt2.rms) ? @sprintf("%.3f", field_analysis.zt2.rms * 1e3) : "NaN"

    @printf("%-18s %-12s %9s %11s %11s %11s\n", "BR RMS", "R1", "", "", "", rms1_str)
    @printf("%-18s %-12s %9s %11s %11s %11s\n", "BR RMS", "R2", "", "", "", rms2_str)

    println("═"^165)
end

# ═══════════════════════════════════════════════════════════════
# Final one-line output table
# ═══════════════════════════════════════════════════════════════

function print_final_field_bias_one_line(
    physical_corrected,
    field_corrected,
)
    # One line = field-coil field bias-corrected result.
    # Deltas are relative to physical-coil field bias-corrected result.
    ref = physical_corrected
    a = field_corrected

    dx = (a.x0 - ref.x0) * 1e3
    dy = (a.y0 - ref.y0) * 1e3
    dz = (a.z0 - ref.z0) * 1e3
    dtx = a.tilt_x - ref.tilt_x
    dty = a.tilt_y - ref.tilt_y

    println("\n" * "═"^145)
    println("FINAL FIELD BIAS-CORRECTED AXIS")
    println("One-line table. Deltas are relative to physical-coil field bias-corrected axis.")
    println("Positions in mm, tilts in deg.")
    println("═"^145)

    @printf(
        "%-36s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s\n",
        "row",
        "x",
        "y",
        "z",
        "tilt_x",
        "tilt_y",
        "Δx_phys",
        "Δy_phys",
        "Δz_phys",
        "Δtx_phys",
        "Δty_phys",
    )

    println("─"^145)

    @printf(
        "%-36s %12.4f %12.4f %12.4f %12.6f %12.6f %12.4f %12.4f %12.4f %12.6f %12.6f\n",
        "field bias corrected",
        a.x0 * 1e3,
        a.y0 * 1e3,
        a.z0 * 1e3,
        a.tilt_x,
        a.tilt_y,
        dx,
        dy,
        dz,
        dtx,
        dty,
    )

    println("═"^145)
end

# ═══════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════

const PHYSICAL_COIL_COLORS = [:royalblue, :dodgerblue, :steelblue, :navy]
const FIELD_COIL_COLORS    = [:crimson, :firebrick, :salmon, :darkred]

function plot_coils!(
    ax,
    coils;
    colors=PHYSICAL_COIL_COLORS,
    linewidth=4,
    alpha=1.0,
    label="coil geometry",
)
    for cs in coils, j in 1:cs.ncoil, k in 1:cs.s
        c = colors[mod1(j, length(colors))]

        x = collect(vec(cs.x[j, k, :])) .* 1e3
        y = collect(vec(cs.y[j, k, :])) .* 1e3
        z = collect(vec(cs.z[j, k, :])) .* 1e3

        if (x[1] - x[end])^2 + (y[1] - y[end])^2 + (z[1] - z[end])^2 > 1e-10
            push!(x, x[1])
            push!(y, y[1])
            push!(z, z[1])
        end

        lines!(ax, x, y, z; color=(c, alpha), linewidth=linewidth)
    end

    lines!(
        ax,
        [NaN, NaN],
        [NaN, NaN],
        [NaN, NaN];
        color=(colors[1], alpha),
        linewidth=linewidth,
        label=label,
    )
end

function plot_axis(
    h,
    phys_geom,
    field_geom,
    physical_mag,
    field_mag,
    field_corrected,
    physical_coils,
    field_coils,
)
    zmin = isnan(AXIS_PLOT_ZMIN_M) ?
        minimum([minimum(h.Z), phys_geom.Zmin, field_geom.Zmin]) :
        AXIS_PLOT_ZMIN_M

    zmax = isnan(AXIS_PLOT_ZMAX_M) ?
        maximum([maximum(h.Z), phys_geom.Zmax, field_geom.Zmax]) :
        AXIS_PLOT_ZMAX_M

    z = collect(range(zmin, zmax, length=200))

    fig = Figure(size=(1400, 950), backgroundcolor=:white)

    ax = Axis3(
        fig[1, 1],
        xlabel="X [mm]",
        ylabel="Y [mm]",
        zlabel="Z [mm]",
        title="Physical, Field, and Bias-Corrected Magnetic Axes",
        aspect=:data,
    )

    if SHOW_PHYSICAL_COILS
        plot_coils!(
            ax,
            physical_coils;
            colors=PHYSICAL_COIL_COLORS,
            linewidth=4,
            alpha=0.75,
            label="physical baseline coil",
        )
    end

    if SHOW_FIELD_COILS
        plot_coils!(
            ax,
            field_coils;
            colors=FIELD_COIL_COLORS,
            linewidth=3,
            alpha=0.75,
            label="field calculation coil",
        )
    end

    if SHOW_HALL_PROBES
        scatter!(
            ax,
            h.x .* 1e3,
            h.y .* 1e3,
            h.Z .* 1e3;
            color=(:gray, 0.85),
            markersize=4,
            label="Hall probes",
        )
    end

    xp, yp, zp = axis_line(
        phys_geom.x0,
        phys_geom.y0,
        phys_geom.z0,
        phys_geom.tilt_x,
        phys_geom.tilt_y,
        z,
    )

    xg, yg, zg = axis_line(
        field_geom.x0,
        field_geom.y0,
        field_geom.z0,
        field_geom.tilt_x,
        field_geom.tilt_y,
        z,
    )

    xpm, ypm, zpm = axis_line(
        physical_mag.x0,
        physical_mag.y0,
        physical_mag.z0,
        physical_mag.tilt_x,
        physical_mag.tilt_y,
        z,
    )

    xfm, yfm, zfm = axis_line(
        field_mag.x0,
        field_mag.y0,
        field_mag.z0,
        field_mag.tilt_x,
        field_mag.tilt_y,
        z,
    )

    xc, yc, zc = axis_line(
        field_corrected.x0,
        field_corrected.y0,
        field_corrected.z0,
        field_corrected.tilt_x,
        field_corrected.tilt_y,
        z,
    )

    lines!(ax, xp .* 1e3, yp .* 1e3, zp .* 1e3;
           color=:black, linewidth=6, linestyle=:dash, label="physical-base geom")

    lines!(ax, xg .* 1e3, yg .* 1e3, zg .* 1e3;
           color=:royalblue, linewidth=4, linestyle=:dashdot, label="field-coil geom")

    lines!(ax, xpm .* 1e3, ypm .* 1e3, zpm .* 1e3;
           color=:purple, linewidth=4, label="physical-coil field magnetic")

    lines!(ax, xfm .* 1e3, yfm .* 1e3, zfm .* 1e3;
           color=:crimson, linewidth=5, label="field-coil field magnetic")

    lines!(ax, xc .* 1e3, yc .* 1e3, zc .* 1e3;
           color=:darkorange, linewidth=5, label="field bias-corrected")

    scatter!(ax, [phys_geom.x0 * 1e3], [phys_geom.y0 * 1e3], [phys_geom.z0 * 1e3];
             color=:black, marker=:utriangle, markersize=20, label="physical-base center")

    scatter!(ax, [physical_mag.x0 * 1e3], [physical_mag.y0 * 1e3], [physical_mag.z0 * 1e3];
             color=:purple, marker=:circle, markersize=16, label="physical-field center")

    scatter!(ax, [field_mag.x0 * 1e3], [field_mag.y0 * 1e3], [field_mag.z0 * 1e3];
             color=:crimson, marker=:diamond, markersize=18, label="field-field center")

    scatter!(ax, [field_corrected.x0 * 1e3], [field_corrected.y0 * 1e3], [field_corrected.z0 * 1e3];
             color=:darkorange, marker=:star5, markersize=22, label="field bias-corr center")

    lines!(ax, zeros(length(z)), zeros(length(z)), z .* 1e3;
           color=(:black, 0.3), linestyle=:dot, linewidth=2, label="machine Z")

    axislegend(ax; position=:lt)

    return fig
end

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

function main()
    println("\nPhysical/Field Coil Magnetic Axis Comparison")
    println("Physical baseline coil file:")
    println("  $PHYSICAL_COIL_FILE")
    println("Field calculation coil file:")
    println("  $FIELD_COIL_FILE")

    println("\nHall input modes:")
    println("  PHYSICAL_HALL_MODE = $PHYSICAL_HALL_MODE")
    println("  FIELD_HALL_MODE    = $FIELD_HALL_MODE")

    if PHYSICAL_HALL_MODE == :csv
        println("  PHYSICAL_HALL_INPUT_CSV = $PHYSICAL_HALL_INPUT_CSV")
    end

    if FIELD_HALL_MODE == :csv
        println("  FIELD_HALL_INPUT_CSV    = $FIELD_HALL_INPUT_CSV")
    end

    # 1. Nominal physical baseline.
    physical_coils = load_coils(
        PHYSICAL_COIL_FILE;
        label="PHYSICAL BASELINE",
        current_A=COIL_CURRENT_A,
    )

    # 2. Actual instantiated coil used to generate or compare field-coil field.
    # If FIELD_HALL_MODE=:csv, this is still used for field geometry and plotting,
    # while magnetic field values come from FIELD_HALL_INPUT_CSV.
    field_coils = load_coils(
        FIELD_COIL_FILE;
        label="FIELD CALCULATION",
        current_A=COIL_CURRENT_A,
    )

    # 3. Bias-correction model.
    # Realistic mode: only the physical baseline model is assumed available.
    bias_model_coils = load_coils(
        PHYSICAL_COIL_FILE;
        label="BIAS-CORRECTION MODEL",
        current_A=COIL_CURRENT_A,
    )

    # Geometric axes.
    phys_geom = axis_from_coil(
        physical_coils;
        label="Physical baseline geometric axis",
    )

    field_geom = axis_from_coil(
        field_coils;
        label="Field-coil geometric axis",
    )

    # Magnetic axes from physical-coil field.
    physical_analysis = analyze_source_coil(
        "PHYSICAL-FIELD",
        physical_coils,
        phys_geom;
        csv_path=HALL_CSV_PHYSICAL,
        hall_mode=PHYSICAL_HALL_MODE,
        hall_input_csv=PHYSICAL_HALL_INPUT_CSV,
        verbose=true,
    )

    # Magnetic axes from field-coil field.
    #
    # FIELD_HALL_MODE=:internal:
    #   uses field_coils to compute Hall data and writes HALL_CSV_FIELD.
    #
    # FIELD_HALL_MODE=:csv:
    #   reads FIELD_HALL_INPUT_CSV and does not recompute the field Hall data.
    field_analysis = analyze_source_coil(
        "FIELD-FIELD",
        field_coils,
        phys_geom;
        csv_path=HALL_CSV_FIELD,
        hall_mode=FIELD_HALL_MODE,
        hall_input_csv=FIELD_HALL_INPUT_CSV,
        verbose=true,
    )

    # Bias-correct the physical-coil field result.
    physical_corrected = bias_correct_axis(
        bias_model_coils,
        physical_analysis.h,
        physical_analysis.zt1,
        physical_analysis.zt2,
        physical_analysis.sin1,
        physical_analysis.sin2;
        label="PHYSICAL-FIELD",
    )

    # Bias-correct the field-coil field result.
    field_corrected = bias_correct_axis(
        bias_model_coils,
        field_analysis.h,
        field_analysis.zt1,
        field_analysis.zt2,
        field_analysis.sin1,
        field_analysis.sin2;
        label="FIELD-FIELD",
    )

    if ENABLE_DETAILED_FIELD_SUMMARY
        print_detailed_field_summary(
            phys_geom,
            field_geom,
            field_analysis,
            field_corrected,
        )
    end

    # Final requested one-line output table.
    # Deltas are relative to physical-coil field bias-corrected.
    # This is deliberately before plot display/readline so it always appears.
    print_final_field_bias_one_line(
        physical_corrected,
        field_corrected,
    )

    if ENABLE_AXIS_PLOT
        fig = plot_axis(
            field_analysis.h,
            phys_geom,
            field_geom,
            physical_analysis.sinavg,
            field_analysis.sinavg,
            field_corrected,
            physical_coils,
            field_coils,
        )

        save(AXIS_PLOT_FILE, fig)
        println("\nSaved plot: $AXIS_PLOT_FILE")

        if DISPLAY_AXIS_PLOT
            display(fig)
            println("\nPress Enter to exit...")
            readline()
        end
    end

    println("\nDone.")
end

main()