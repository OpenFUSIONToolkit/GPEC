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
OUT_NAME = "sparc_pf1u_357_35T"

COIL_FILE = joinpath(@__DIR__, "$OUT_NAME.dat")
COIL_CURRENT_A = 100.0

HALL_CSV = joinpath(@__DIR__, "hall_probe_$OUT_NAME.csv")
AXIS_PLOT_FILE = joinpath(@__DIR__, "axis_comparison_$OUT_NAME.png")

HALL_R_INNER_FRAC = 0.40
HALL_R_OUTER_FRAC = 0.50
HALL_N_PHI_INNER = 24
HALL_N_Z_INNER   = 22
HALL_N_PHI_OUTER = 20
HALL_N_Z_OUTER   = 10
HALL_Z_HALFSPAN_INNER_M = 0.60
HALL_Z_HALFSPAN_OUTER_M = 0.40

# If NaN, use fitted physical coil radius for converting Z sinusoid amplitude to tilt.
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
SHOW_COILS = true
SHOW_HALL_PROBES = true
AXIS_PLOT_ZMIN_M = NaN
AXIS_PLOT_ZMAX_M = NaN

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

wmean(x, w) = sum(w) <= 0 ? mean(x) : sum(w .* x) / sum(w)

function fit_cossin(φ, y)
    c = hcat(ones(length(φ)), cos.(φ), sin.(φ)) \ y
    return c[1], c[2], c[3]
end

function zcrossings(vals, z)
    out = Float64[]
    for i in 1:length(vals)-1
        vals[i] == 0 && push!(out, z[i])
        vals[i] * vals[i+1] < 0 || continue
        t = vals[i] / (vals[i] - vals[i+1])
        push!(out, z[i] + t * (z[i+1] - z[i]))
    end
    return out
end

function fit_circle_xy(x, y)
    A = hcat(2 .* x, 2 .* y, ones(length(x)))
    b = x.^2 .+ y.^2
    c = A \ b
    cx, cy = c[1], c[2]
    r = sqrt(max(c[3] + cx^2 + cy^2, 0.0))
    return cx, cy, r
end

function axis_line(x0, y0, z0, tx_deg, ty_deg, z)
    tx = tand(tx_deg)
    ty = tand(ty_deg)
    x = x0 .+ ty .* (z .- z0)
    y = y0 .- tx .* (z .- z0)
    return x, y, z
end

meanfinite(v) = begin
    u = filter(isfinite, collect(v))
    isempty(u) ? NaN : mean(u)
end

shellmeta(h, shell) = shell == 1 ?
    (HALL_N_PHI_INNER, h.nzi, h.R_inner, h.Zi, h.phi_i) :
    (HALL_N_PHI_OUTER, h.nzo, h.R_outer, h.Zo, h.phi_o)

# ═══════════════════════════════════════════════════════════════
# Geometry helpers for robust coil pose
# ═══════════════════════════════════════════════════════════════

function segment_midpoints_lengths(coils)
    mx = Float64[]
    my = Float64[]
    mz = Float64[]
    w  = Float64[]

    for cs in coils, j in 1:cs.ncoil, k in 1:cs.s, l in 1:(cs.nsec - 1)
        x1 = cs.x[j, k, l];   x2 = cs.x[j, k, l+1]
        y1 = cs.y[j, k, l];   y2 = cs.y[j, k, l+1]
        z1 = cs.z[j, k, l];   z2 = cs.z[j, k, l+1]

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

    # Weighted SVD for best-fit plane normal
    P = hcat(mx .- cx, my .- cy, mz .- cz)
    Pw = P .* sqrt.(w)
    _, _, V = svd(Pw)
    n = Vector(V[:, 3])
    n[3] < 0 && (n = -n)

    return (cx=cx, cy=cy, cz=cz, n=n, mx=mx, my=my, mz=mz, w=w)
end

# ═══════════════════════════════════════════════════════════════
# Rotation helpers
# ═══════════════════════════════════════════════════════════════

function rotation_x(θ::Float64)
    c, s = cos(θ), sin(θ)
    return [1.0 0.0 0.0;
            0.0 c   -s;
            0.0 s    c]
end

function rotation_y(θ::Float64)
    c, s = cos(θ), sin(θ)
    return [ c   0.0  s;
            0.0  1.0 0.0;
             -s  0.0  c]
end

function rotation_z(θ::Float64)
    c, s = cos(θ), sin(θ)
    return [c  -s  0.0;
            s   c  0.0;
            0.0 0.0 1.0]
end

function build_rotation_matrix(tx_deg::Float64, ty_deg::Float64, tz_deg::Float64)
    Rx = rotation_x(deg2rad(tx_deg))
    Ry = rotation_y(deg2rad(ty_deg))
    Rz = rotation_z(deg2rad(tz_deg))
    return Rz * Ry * Rx
end

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

    K = [  0.0  -k[3]   k[2];
          k[3]   0.0   -k[1];
         -k[2]   k[1]   0.0 ]

    return I + sinθ * K + (1.0 - cosθ) * K * K
end

# ═══════════════════════════════════════════════════════════════
# Coil placement: exact target axis in lab frame
# ═══════════════════════════════════════════════════════════════

function place_coil_at_axis(coils,
                             target_x0::Float64, target_y0::Float64, target_z0::Float64,
                             target_tx_deg::Float64, target_ty_deg::Float64)

    coils_new = deepcopy(coils)

    pose = coil_pose_from_segments(coils_new)
    cx, cy, cz = pose.cx, pose.cy, pose.cz
    n_orig = copy(pose.n)

    # Convention consistent with _physical_axis_from_coil_impl:
    # normal = [tan(ty), -tan(tx), 1] up to normalization
    n_target = normalize([tand(target_ty_deg), -tand(target_tx_deg), 1.0])

    z_hat   = [0.0, 0.0, 1.0]
    R_undo  = rotation_between_vectors(n_orig, z_hat)
    R_apply = rotation_between_vectors(z_hat, n_target)
    R_total = R_apply * R_undo

    for cs in coils_new
        for j in 1:cs.ncoil, k in 1:cs.s, l in 1:cs.nsec
            p = [cs.x[j, k, l] - cx,
                 cs.y[j, k, l] - cy,
                 cs.z[j, k, l] - cz]
            q = R_total * p
            cs.x[j, k, l] = q[1] + target_x0
            cs.y[j, k, l] = q[2] + target_y0
            cs.z[j, k, l] = q[3] + target_z0
        end
    end

    return coils_new
end

# ═══════════════════════════════════════════════════════════════
# Coil geometry: physical axis
# ═══════════════════════════════════════════════════════════════

function load_coils()
    isfile(COIL_FILE) || error("Coil file not found: $COIL_FILE")
    cs = read_coil_dat(COIL_FILE)
    cs.currents .= COIL_CURRENT_A
    println("\nLoaded coil: $COIL_FILE")
    println("  ncoil=$(cs.ncoil), s=$(cs.s), nsec=$(cs.nsec), I=$COIL_CURRENT_A A")
    return [cs]
end

function collect_coil_xyz(coils)
    x = Float64[]; y = Float64[]; z = Float64[]
    for cs in coils, j in 1:cs.ncoil, k in 1:cs.s
        append!(x, vec(cs.x[j, k, :]))
        append!(y, vec(cs.y[j, k, :]))
        append!(z, vec(cs.z[j, k, :]))
    end
    return x, y, z
end

function _physical_axis_from_coil_impl(coils)
    pose = coil_pose_from_segments(coils)
    cx, cy, cz = pose.cx, pose.cy, pose.cz
    n = pose.n

    # normal = [tan(ty), -tan(tx), 1] / norm
    tx = atand(-n[2], n[3])
    ty = atand( n[1], n[3])

    # Radius estimate in coil plane using weighted segment midpoints
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
        x0=cx, y0=cy, z0=cz,
        tilt_x=tx, tilt_y=ty,
        Rfit=rfit,
        Rmin=minimum(R), Rmax=maximum(R),
        Zmin=minimum(z), Zmax=maximum(z),
    )
end

function physical_axis_from_coil(coils)
    phys = _physical_axis_from_coil_impl(coils)
    println("\nPhysical axis from coil geometry:")
    @printf("  center = (%.3f, %.3f, %.3f) mm\n",
            phys.x0 * 1e3, phys.y0 * 1e3, phys.z0 * 1e3)
    @printf("  tilt_x = %.6f deg, tilt_y = %.6f deg\n", phys.tilt_x, phys.tilt_y)
    @printf("  fit radius = %.3f mm\n", phys.Rfit * 1e3)
    @printf("  R range = [%.3f, %.3f] mm\n", phys.Rmin * 1e3, phys.Rmax * 1e3)
    @printf("  Z range = [%.3f, %.3f] mm\n", phys.Zmin * 1e3, phys.Zmax * 1e3)
    return phys
end

physical_axis_from_coil_silent(coils) = _physical_axis_from_coil_impl(coils)

# ═══════════════════════════════════════════════════════════════
# Synthetic Hall measurements
# ═══════════════════════════════════════════════════════════════

function make_hall_data(coils, phys)
    R_inner = HALL_R_INNER_FRAC * phys.Rmin
    R_outer = HALL_R_OUTER_FRAC * phys.Rmax
    Zc = phys.z0

    φi = collect(range(0, 2π, length=HALL_N_PHI_INNER + 1)[1:end-1])
    φo = collect(range(0, 2π, length=HALL_N_PHI_OUTER + 1)[1:end-1])
    Zi = collect(range(Zc - HALL_Z_HALFSPAN_INNER_M, Zc + HALL_Z_HALFSPAN_INNER_M, length=HALL_N_Z_INNER))
    Zo = collect(range(Zc - HALL_Z_HALFSPAN_OUTER_M, Zc + HALL_Z_HALFSPAN_OUTER_M, length=HALL_N_Z_OUTER))

    N = HALL_N_PHI_INNER * HALL_N_Z_INNER + HALL_N_PHI_OUTER * HALL_N_Z_OUTER

    R = zeros(N); φ = zeros(N); Z = zeros(N); shell = zeros(Int, N)
    k = 1

    for iz in eachindex(Zi), ip in eachindex(φi)
        R[k] = R_inner; φ[k] = φi[ip]; Z[k] = Zi[iz]; shell[k] = 1; k += 1
    end
    for iz in eachindex(Zo), ip in eachindex(φo)
        R[k] = R_outer; φ[k] = φo[ip]; Z[k] = Zo[iz]; shell[k] = 2; k += 1
    end

    BR = zeros(N); BP = zeros(N); BZ = zeros(N)

    println("\nComputing synthetic Hall data:")
    println("  probes = $N")
    @printf("  R1 = %.3f mm, Z1=[%.3f, %.3f] mm\n", R_inner * 1e3, first(Zi) * 1e3, last(Zi) * 1e3)
    @printf("  R2 = %.3f mm, Z2=[%.3f, %.3f] mm\n", R_outer * 1e3, first(Zo) * 1e3, last(Zo) * 1e3)

    compute_biot_savart_boundary!(BR, BP, BZ, R, φ, Z, coils)
    Bmag = sqrt.(BR.^2 .+ BP.^2 .+ BZ.^2)

    return (
        R=R, phi=φ, Z=Z,
        x=R .* cos.(φ), y=R .* sin.(φ),
        shell=shell,
        B_R=BR, B_phi=BP, B_Z=BZ, B_mag=Bmag,
        Zi=Zi, Zo=Zo, phi_i=φi, phi_o=φo,
        nzi=HALL_N_Z_INNER, nzo=HALL_N_Z_OUTER,
        R_inner=R_inner, R_outer=R_outer,
    )
end

function recompute_hall_field(h, coils)
    N = length(h.R)
    BR = zeros(N); BP = zeros(N); BZ = zeros(N)
    compute_biot_savart_boundary!(BR, BP, BZ, h.R, h.phi, h.Z, coils)
    Bmag = sqrt.(BR.^2 .+ BP.^2 .+ BZ.^2)
    return (
        R=h.R, phi=h.phi, Z=h.Z,
        x=h.x, y=h.y,
        shell=h.shell,
        B_R=BR, B_phi=BP, B_Z=BZ, B_mag=Bmag,
        Zi=h.Zi, Zo=h.Zo, phi_i=h.phi_i, phi_o=h.phi_o,
        nzi=h.nzi, nzo=h.nzo,
        R_inner=h.R_inner, R_outer=h.R_outer,
    )
end

function export_hall_csv(h)
    open(HALL_CSV, "w") do io
        println(io, "x,y,z,R,phi,shell,B_R,B_phi,B_Z,B_mag")
        for i in eachindex(h.R)
            println(io, "$(h.x[i]),$(h.y[i]),$(h.Z[i]),$(h.R[i]),$(h.phi[i]),$(h.shell[i]),$(h.B_R[i]),$(h.B_phi[i]),$(h.B_Z[i]),$(h.B_mag[i])")
        end
    end
    println("\nExported Hall CSV: $HALL_CSV")
end

# ═══════════════════════════════════════════════════════════════
# Magnetic Z/tilt: n=1 sinusoid of B_R zero-crossings
# ═══════════════════════════════════════════════════════════════

function magnetic_z_tilt(h, phys; shell=1, verbose=true)
    nφ, nz, Rsh, Zgrid, φgrid = shellmeta(h, shell)
    BR = reshape(h.B_R[h.shell .== shell], nφ, nz)

    zc = fill(NaN, nφ)
    valid = falses(nφ)
    noise = BR_ZERO_NOISE_FRAC * maximum(abs.(BR))
    zguess = phys.z0

    for ip in 1:nφ
        maximum(abs.(BR[ip, :])) < noise && continue
        c = zcrossings(BR[ip, :], Zgrid)
        isempty(c) && continue
        zc[ip] = c[argmin(abs.(c .- zguess))]
        valid[ip] = true
    end

    Rcal = isfinite(TILT_CALIBRATION_RADIUS_M) ? TILT_CALIBRATION_RADIUS_M : phys.Rfit

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
            println("\nB_R zero-crossing sinusoid, shell $shell:")
            @printf("  valid = %d/%d, Z0 = %.6f mm, tilt_x = %.6f deg, tilt_y = %.6f deg\n",
                    count(valid), nφ, Z0 * 1e3, tx, ty)
        end

        return (
            shell=shell, R=Rsh,
            x0=NaN, y0=NaN, z0=Z0,
            tilt_x=tx, tilt_y=ty, tilt_mag=tmag,
            Bcos=Bc, Csin=Cs, amp=amp, rms=rms,
            zcross=zc, valid=valid, method="per_phi",
        )
    end

    BR0 = [mean(BR[:, iz]) for iz in 1:nz]
    c = zcrossings(BR0, Zgrid)
    Z0 = isempty(c) ? NaN : c[argmin(abs.(c .- zguess))]

    if verbose
        println("\nB_R zero-crossing fallback, shell $shell:")
        @printf("  valid = %d/%d, Z0 = %s\n",
                count(valid), nφ, isnan(Z0) ? "NaN" : @sprintf("%.6f mm", Z0 * 1e3))
    end

    return (
        shell=shell, R=Rsh,
        x0=NaN, y0=NaN, z0=Z0,
        tilt_x=NaN, tilt_y=NaN, tilt_mag=NaN,
        Bcos=NaN, Csin=NaN, amp=NaN, rms=NaN,
        zcross=zc, valid=valid,
        method=isempty(c) ? "failed" : "mean_BR",
    )
end

# ═══════════════════════════════════════════════════════════════
# Magnetic X/Y: local B_phi minimization
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

function magnetic_xy_minimize(h, phys, zt; shell=1, verbose=true)
    idxs = findall(h.shell .== shell)

    z0 = isfinite(zt.z0) ? zt.z0 : phys.z0
    tx = isfinite(zt.tilt_x) ? zt.tilt_x : phys.tilt_x
    ty = isfinite(zt.tilt_y) ? zt.tilt_y : phys.tilt_y

    x = phys.x0
    y = phys.y0
    step = XY_MIN_STEP0_M

    best = local_bphi_objective(h, idxs, x, y, z0, tx, ty)

    dirs = [(1.0,0.0),(-1.0,0.0),(0.0,1.0),(0.0,-1.0),
            (1.0,1.0),(1.0,-1.0),(-1.0,1.0),(-1.0,-1.0)]

    while step > XY_MIN_TOL_M
        improved = false

        for (dx,dy) in dirs
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
        println("\nLocal B_phi minimization, shell $shell:")
        @printf("  x0 = %.6f mm, y0 = %.6f mm, objective = %.6e\n", x * 1e3, y * 1e3, best)
    end

    return (
        shell=shell, R=zt.R,
        x0=x, y0=y, z0=z0,
        tilt_x=tx, tilt_y=ty, tilt_mag=hypot(tx, ty),
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

    count(valid) == 0 && return (shell=shell, R=Rsh, x0=NaN, y0=NaN)

    w = abs.(BR0[valid]).^2
    x0 = wmean(x[valid], w)
    y0 = wmean(y[valid], w)

    if verbose
        println("\nFourier B_phi n=1 XY, shell $shell:")
        @printf("  x0 = %.6f mm, y0 = %.6f mm\n", x0 * 1e3, y0 * 1e3)
    end

    return (shell=shell, R=Rsh, x0=x0, y0=y0)
end

function combine_axis(xy, zt)
    return (
        shell=xy.shell, R=xy.R,
        x0=xy.x0, y0=xy.y0, z0=zt.z0,
        tilt_x=zt.tilt_x, tilt_y=zt.tilt_y,
        tilt_mag=hypot(zt.tilt_x, zt.tilt_y),
    )
end

function average_axes(a1, a2)
    return (
        shell=0,
        R=NaN,
        x0=meanfinite([a1.x0, a2.x0]),
        y0=meanfinite([a1.y0, a2.y0]),
        z0=meanfinite([a1.z0, a2.z0]),
        tilt_x=meanfinite([a1.tilt_x, a2.tilt_x]),
        tilt_y=meanfinite([a1.tilt_y, a2.tilt_y]),
        tilt_mag=hypot(meanfinite([a1.tilt_x, a2.tilt_x]),
                       meanfinite([a1.tilt_y, a2.tilt_y])),
    )
end

# ═══════════════════════════════════════════════════════════════
# Analysis pipeline
# ═══════════════════════════════════════════════════════════════

function run_analysis_pipeline(h, phys_ref; verbose=false)
    zt1 = magnetic_z_tilt(h, phys_ref; shell=1, verbose=verbose)
    zt2 = magnetic_z_tilt(h, phys_ref; shell=2, verbose=verbose)
    sin1 = magnetic_xy_minimize(h, phys_ref, zt1; shell=1, verbose=verbose)
    sin2 = magnetic_xy_minimize(h, phys_ref, zt2; shell=2, verbose=verbose)
    return average_axes(sin1, sin2), zt1, zt2, sin1, sin2
end

# ═══════════════════════════════════════════════════════════════
# Forward-model bias correction
# ═══════════════════════════════════════════════════════════════

function bias_correct_axis(coils, phys, h,
                            measured_sinavg,
                            measured_zt1, measured_zt2,
                            measured_sin1, measured_sin2;
                            n_iter=BIAS_CORRECTION_N_ITER,
                            damping=BIAS_CORRECTION_DAMPING)

    println("\n" * "─"^70)
    println("Forward-model bias correction  ($n_iter iterations, α=$damping)")
    println("─"^70)

    meas_x  = meanfinite([measured_sin1.x0,    measured_sin2.x0])
    meas_y  = meanfinite([measured_sin1.y0,    measured_sin2.y0])
    meas_z  = meanfinite([measured_zt1.z0,     measured_zt2.z0])
    meas_tx = meanfinite([measured_zt1.tilt_x, measured_zt2.tilt_x])
    meas_ty = meanfinite([measured_zt1.tilt_y, measured_zt2.tilt_y])

    if any(!isfinite, [meas_x, meas_y, meas_z, meas_tx, meas_ty])
        @warn "bias_correct_axis: measured values contain NaN — skipping correction"
        return measured_sinavg
    end

    est_x  = meas_x
    est_y  = meas_y
    est_z  = meas_z
    est_tx = meas_tx
    est_ty = meas_ty

    corrected_avg = measured_sinavg

    for iter in 1:n_iter
        synth_coils = place_coil_at_axis(coils, est_x, est_y, est_z, est_tx, est_ty)
        h_synth = recompute_hall_field(h, synth_coils)
        phys_synth = physical_axis_from_coil_silent(synth_coils)

        _, szt1, szt2, ssin1, ssin2 = run_analysis_pipeline(h_synth, phys_synth; verbose=false)

        bias_x  = meanfinite([ssin1.x0,     ssin2.x0])    - phys_synth.x0
        bias_y  = meanfinite([ssin1.y0,     ssin2.y0])    - phys_synth.y0
        bias_z  = meanfinite([szt1.z0,      szt2.z0])     - phys_synth.z0
        bias_tx = meanfinite([szt1.tilt_x,  szt2.tilt_x]) - phys_synth.tilt_x
        bias_ty = meanfinite([szt1.tilt_y,  szt2.tilt_y]) - phys_synth.tilt_y

        corr_x  = meas_x  - bias_x
        corr_y  = meas_y  - bias_y
        corr_z  = meas_z  - bias_z
        corr_tx = meas_tx - bias_tx
        corr_ty = meas_ty - bias_ty

        @printf("  iter %d | bias     : Δx=%+.4f mm  Δy=%+.4f mm  Δz=%+.4f mm  Δtx=%+.6f°  Δty=%+.6f°\n",
                iter, bias_x * 1e3, bias_y * 1e3, bias_z * 1e3, bias_tx, bias_ty)
        @printf("  iter %d | corrected: x=%+.4f mm   y=%+.4f mm   z=%+.4f mm   tx=%+.6f°   ty=%+.6f°\n",
                iter, corr_x * 1e3, corr_y * 1e3, corr_z * 1e3, corr_tx, corr_ty)
        println()

        corrected_avg = (
            shell=0, R=NaN,
            x0=corr_x, y0=corr_y, z0=corr_z,
            tilt_x=corr_tx, tilt_y=corr_ty,
            tilt_mag=hypot(corr_tx, corr_ty),
        )

        est_x  = est_x  + damping * (corr_x  - est_x)
        est_y  = est_y  + damping * (corr_y  - est_y)
        est_z  = est_z  + damping * (corr_z  - est_z)
        est_tx = est_tx + damping * (corr_tx - est_tx)
        est_ty = est_ty + damping * (corr_ty - est_ty)

        @printf("  iter %d | next est : x=%+.4f mm   y=%+.4f mm   z=%+.4f mm   tx=%+.6f°   ty=%+.6f°\n",
                iter, est_x * 1e3, est_y * 1e3, est_z * 1e3, est_tx, est_ty)
        println()
    end

    println("─"^70)
    return corrected_avg
end

# ═══════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════

function print_axis_row(method, label, R, phys, a)
    dx = (a.x0 - phys.x0) * 1e3
    dy = (a.y0 - phys.y0) * 1e3
    dz = (a.z0 - phys.z0) * 1e3
    dtx = a.tilt_x - phys.tilt_x
    dty = a.tilt_y - phys.tilt_y

    @printf("%-14s %-10s %9.1f %11.3f %11.3f %11.3f %10.5f %10.5f %11.3f %11.3f %11.3f %10.5f %10.5f\n",
        method, label, R * 1e3,
        a.x0 * 1e3, a.y0 * 1e3, a.z0 * 1e3, a.tilt_x, a.tilt_y,
        dx, dy, dz, dtx, dty)
end

function print_summary(phys, sin1, sin2, sinavg, fou1, fou2, fouavg, zt1, zt2, corrected)
    println("\n" * "═"^155)
    println("PHYSICAL AXIS VS MAGNETIC AXES")
    println("═"^155)
    println("Columns: x/y/z in mm, tilts in deg, deltas = magnetic - physical")
    println("─"^155)
    @printf("%-14s %-10s %9s %11s %11s %11s %10s %10s %11s %11s %11s %10s %10s\n",
        "method", "radius", "R[mm]", "x", "y", "z", "tilt_x", "tilt_y",
        "Δx", "Δy", "Δz", "Δtx", "Δty")
    println("─"^155)

    print_axis_row("physical", "-", phys.Rfit, phys, phys)

    println("─"^155)
    print_axis_row("sinusoid", "R1", sin1.R, phys, sin1)
    print_axis_row("sinusoid", "R2", sin2.R, phys, sin2)
    print_axis_row("sinusoid", "average", NaN, phys, sinavg)

    println("─"^155)
    print_axis_row("fourier", "R1", fou1.R, phys, fou1)
    print_axis_row("fourier", "R2", fou2.R, phys, fou2)
    print_axis_row("fourier", "average", NaN, phys, fouavg)

    println("─"^155)
    print_axis_row("bias-corrected", "average", NaN, phys, corrected)

    println("─"^155)
    rms1_str = isfinite(zt1.rms) ? @sprintf("%.3f", zt1.rms * 1e3) : "NaN"
    rms2_str = isfinite(zt2.rms) ? @sprintf("%.3f", zt2.rms * 1e3) : "NaN"
    @printf("%-14s %-10s %9s %11s %11s %11s %10s %10s %11s %11s %11s %10s %10s\n",
        "BR RMS", "R1", "", "", "", rms1_str, "", "", "", "", "", "", "")
    @printf("%-14s %-10s %9s %11s %11s %11s %10s %10s %11s %11s %11s %10s %10s\n",
        "BR RMS", "R2", "", "", "", rms2_str, "", "", "", "", "", "", "")
    println("═"^155)
end

# ═══════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════

const COIL_PALETTES = [
    [:royalblue,:dodgerblue,:steelblue,:navy],
    [:crimson,:firebrick,:salmon,:darkred],
    [:forestgreen,:seagreen,:limegreen,:darkgreen],
    [:darkorange,:orange,:gold],
    [:purple,:mediumpurple,:blueviolet],
    [:deeppink,:hotpink,:magenta],
]

function plot_coils!(ax, coils)
    for (si, cs) in enumerate(coils), j in 1:cs.ncoil, k in 1:cs.s
        pal = COIL_PALETTES[mod1(si, length(COIL_PALETTES))]
        c = pal[mod1(j, length(pal))]

        x = collect(vec(cs.x[j, k, :])) .* 1e3
        y = collect(vec(cs.y[j, k, :])) .* 1e3
        z = collect(vec(cs.z[j, k, :])) .* 1e3

        if (x[1] - x[end])^2 + (y[1] - y[end])^2 + (z[1] - z[end])^2 > 1e-10
            push!(x, x[1]); push!(y, y[1]); push!(z, z[1])
        end

        lines!(ax, x, y, z; color=c, linewidth=4)
    end

    lines!(ax, [NaN,NaN], [NaN,NaN], [NaN,NaN];
           color=:royalblue, linewidth=4, label="coil geometry")
end

function plot_axis(h, phys, sinavg, fouavg, corrected, coils)
    zmin = isnan(AXIS_PLOT_ZMIN_M) ? minimum([minimum(h.Z), phys.Zmin]) : AXIS_PLOT_ZMIN_M
    zmax = isnan(AXIS_PLOT_ZMAX_M) ? maximum([maximum(h.Z), phys.Zmax]) : AXIS_PLOT_ZMAX_M
    z = collect(range(zmin, zmax, length=200))

    fig = Figure(size=(1400,950), backgroundcolor=:white)

    ax = Axis3(
        fig[1,1],
        xlabel="X [mm]",
        ylabel="Y [mm]",
        zlabel="Z [mm]",
        title="Physical Axis vs Magnetic Axis Methods",
        aspect=:data,
    )

    SHOW_COILS && plot_coils!(ax, coils)

    if SHOW_HALL_PROBES
        scatter!(ax, h.x .* 1e3, h.y .* 1e3, h.Z .* 1e3;
                 color=(:gray,0.6), markersize=4, label="Hall probes")
    end

    xp, yp, zp = axis_line(phys.x0, phys.y0, phys.z0, phys.tilt_x, phys.tilt_y, z)
    xs, ys, zs = axis_line(sinavg.x0, sinavg.y0, sinavg.z0, sinavg.tilt_x, sinavg.tilt_y, z)
    xf, yf, zf = axis_line(fouavg.x0, fouavg.y0, fouavg.z0, fouavg.tilt_x, fouavg.tilt_y, z)
    xc, yc, zc_l = axis_line(corrected.x0, corrected.y0, corrected.z0, corrected.tilt_x, corrected.tilt_y, z)

    lines!(ax, xp .* 1e3, yp .* 1e3, zp .* 1e3;
           color=:black, linewidth=6, linestyle=:dash, label="physical axis")
    scatter!(ax, [phys.x0 * 1e3], [phys.y0 * 1e3], [phys.z0 * 1e3];
             color=:black, marker=:utriangle, markersize=20, label="physical center")

    lines!(ax, xs .* 1e3, ys .* 1e3, zs .* 1e3;
           color=:crimson, linewidth=4, label="sinusoid/local-Bφ")
    scatter!(ax, [sinavg.x0 * 1e3], [sinavg.y0 * 1e3], [sinavg.z0 * 1e3];
             color=:crimson, markersize=16, label="sinusoid center")

    lines!(ax, xf .* 1e3, yf .* 1e3, zf .* 1e3;
           color=:forestgreen, linewidth=4, label="Fourier n=1")
    scatter!(ax, [fouavg.x0 * 1e3], [fouavg.y0 * 1e3], [fouavg.z0 * 1e3];
             color=:forestgreen, marker=:diamond, markersize=16, label="Fourier center")

    lines!(ax, xc .* 1e3, yc .* 1e3, zc_l .* 1e3;
           color=:darkorange, linewidth=6, label="bias-corrected")
    scatter!(ax, [corrected.x0 * 1e3], [corrected.y0 * 1e3], [corrected.z0 * 1e3];
             color=:darkorange, marker=:star5, markersize=22, label="bias-corr center")

    lines!(ax, zeros(length(z)), zeros(length(z)), z .* 1e3;
           color=(:black,0.3), linestyle=:dot, linewidth=2, label="machine Z")

    axislegend(ax; position=:lt)
    return fig
end

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

function main()
    println("\nPhysical vs Magnetic Axis Analysis")

    coils = load_coils()
    phys = physical_axis_from_coil(coils)

    h = make_hall_data(coils, phys)
    export_hall_csv(h)

    # Standard analysis
    zt1 = magnetic_z_tilt(h, phys; shell=1)
    zt2 = magnetic_z_tilt(h, phys; shell=2)

    sin1 = magnetic_xy_minimize(h, phys, zt1; shell=1)
    sin2 = magnetic_xy_minimize(h, phys, zt2; shell=2)
    sinavg = average_axes(sin1, sin2)

    fxy1 = magnetic_xy_fourier(h; shell=1)
    fxy2 = magnetic_xy_fourier(h; shell=2)
    fou1 = combine_axis(fxy1, zt1)
    fou2 = combine_axis(fxy2, zt2)
    fouavg = average_axes(fou1, fou2)

    # Forward-model bias correction
    corrected = bias_correct_axis(coils, phys, h, sinavg, zt1, zt2, sin1, sin2)

    # Summary
    print_summary(phys, sin1, sin2, sinavg, fou1, fou2, fouavg, zt1, zt2, corrected)

    # Plot
    if ENABLE_AXIS_PLOT
        fig = plot_axis(h, phys, sinavg, fouavg, corrected, coils)
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