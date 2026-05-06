
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

COIL_FILE = joinpath(@__DIR__, "sparc_pf1u_AS_9501_xy3T.dat")
COIL_CURRENT_A = 100.0

OUT_NAME = "sparc_pf1u_AS_9501_xy3T"
HALL_CSV = joinpath(@__DIR__, "hall_probe_$OUT_NAME.csv")
AXIS_PLOT_FILE = joinpath(@__DIR__, "axis_comparison_$OUT_NAME.png")

# Hall sensors are placed using approximate coil geometry.
# Final magnetic estimates use Hall fields + Hall positions.
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

# Analysis settings.
BR_ZERO_NOISE_FRAC = 0.0#2
BR_MIN_FRAC_FOR_FOURIER_XY = 0.10
BPHI_SIGN = 1.0                 # set -1 if Fourier dx/dy signs are both flipped
XY_MIN_STEP0_M = 0.050
XY_MIN_TOL_M = 1e-8

# Plot controls.
ENABLE_AXIS_PLOT = true
DISPLAY_AXIS_PLOT = true
SHOW_COILS = true
SHOW_HALL_PROBES = true
AXIS_PLOT_ZMIN_M = NaN
AXIS_PLOT_ZMAX_M = NaN

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

wmean(x,w) = sum(w) <= 0 ? mean(x) : sum(w .* x) / sum(w)

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
        push!(out, z[i] + t*(z[i+1]-z[i]))
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
    x=Float64[]; y=Float64[]; z=Float64[]
    for cs in coils, j in 1:cs.ncoil, k in 1:cs.s
        append!(x, vec(cs.x[j,k,:]))
        append!(y, vec(cs.y[j,k,:]))
        append!(z, vec(cs.z[j,k,:]))
    end
    return x, y, z
end

function physical_axis_from_coil(coils)
    x,y,z = collect_coil_xyz(coils)
    R = sqrt.(x.^2 .+ y.^2)

    cx, cy, rfit = fit_circle_xy(x, y)

    # Plane model: z = z0 + A(x-cx) + B(y-cy)
    # Axis convention: x(Z)=x0+tan(tilt_y)(Z-z0), y(Z)=y0-tan(tilt_x)(Z-z0)
    X = x .- cx
    Y = y .- cy
    c = hcat(ones(length(x)), X, Y) \ z
    z0, A, B = c

    tx = atand(B)
    ty = atand(-A)

    phys = (
        x0=cx, y0=cy, z0=z0,
        tilt_x=tx, tilt_y=ty,
        Rfit=rfit,
        Rmin=minimum(R), Rmax=maximum(R),
        Zmin=minimum(z), Zmax=maximum(z),
    )

    println("\nPhysical axis from coil geometry:")
    @printf("  center = (%.3f, %.3f, %.3f) mm\n", cx*1e3, cy*1e3, z0*1e3)
    @printf("  tilt_x = %.6f deg, tilt_y = %.6f deg\n", tx, ty)
    @printf("  fit radius = %.3f mm\n", rfit*1e3)
    @printf("  R range = [%.3f, %.3f] mm\n", phys.Rmin*1e3, phys.Rmax*1e3)
    @printf("  Z range = [%.3f, %.3f] mm\n", phys.Zmin*1e3, phys.Zmax*1e3)

    return phys
end

# ═══════════════════════════════════════════════════════════════
# Synthetic Hall measurements
# ═══════════════════════════════════════════════════════════════

function make_hall_data(coils, phys)
    R_inner = HALL_R_INNER_FRAC * phys.Rmin
    R_outer = HALL_R_OUTER_FRAC * phys.Rmax
    Zc = phys.z0

    φi = collect(range(0, 2π, length=HALL_N_PHI_INNER+1)[1:end-1])
    φo = collect(range(0, 2π, length=HALL_N_PHI_OUTER+1)[1:end-1])
    Zi = collect(range(Zc-HALL_Z_HALFSPAN_INNER_M, Zc+HALL_Z_HALFSPAN_INNER_M, length=HALL_N_Z_INNER))
    Zo = collect(range(Zc-HALL_Z_HALFSPAN_OUTER_M, Zc+HALL_Z_HALFSPAN_OUTER_M, length=HALL_N_Z_OUTER))

    N = HALL_N_PHI_INNER*HALL_N_Z_INNER + HALL_N_PHI_OUTER*HALL_N_Z_OUTER

    R=zeros(N); φ=zeros(N); Z=zeros(N); shell=zeros(Int,N)
    k = 1

    for iz in eachindex(Zi), ip in eachindex(φi)
        R[k]=R_inner; φ[k]=φi[ip]; Z[k]=Zi[iz]; shell[k]=1; k+=1
    end
    for iz in eachindex(Zo), ip in eachindex(φo)
        R[k]=R_outer; φ[k]=φo[ip]; Z[k]=Zo[iz]; shell[k]=2; k+=1
    end

    BR=zeros(N); BP=zeros(N); BZ=zeros(N)

    println("\nComputing synthetic Hall data:")
    println("  probes = $N")
    @printf("  R1 = %.3f mm, Z1=[%.3f, %.3f] mm\n", R_inner*1e3, first(Zi)*1e3, last(Zi)*1e3)
    @printf("  R2 = %.3f mm, Z2=[%.3f, %.3f] mm\n", R_outer*1e3, first(Zo)*1e3, last(Zo)*1e3)

    compute_biot_savart_boundary!(BR, BP, BZ, R, φ, Z, coils)
    Bmag = sqrt.(BR.^2 .+ BP.^2 .+ BZ.^2)

    return (
        R=R, phi=φ, Z=Z,
        x=R.*cos.(φ), y=R.*sin.(φ),
        shell=shell,
        B_R=BR, B_phi=BP, B_Z=BZ, B_mag=Bmag,
        Zi=Zi, Zo=Zo, phi_i=φi, phi_o=φo,
        nzi=HALL_N_Z_INNER, nzo=HALL_N_Z_OUTER,
        R_inner=R_inner, R_outer=R_outer,
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

function magnetic_z_tilt(h, phys; shell=1)
    nφ,nz,Rsh,Zgrid,φgrid = shellmeta(h, shell)
    BR = reshape(h.B_R[h.shell .== shell], nφ, nz)

    zc = fill(NaN,nφ)
    valid = falses(nφ)
    noise = BR_ZERO_NOISE_FRAC * maximum(abs.(BR))
    zguess = phys.z0

    for ip in 1:nφ
        maximum(abs.(BR[ip,:])) < noise && continue
        c = zcrossings(BR[ip,:], Zgrid)
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

        println("\nB_R zero-crossing sinusoid, shell $shell:")
        @printf("  valid = %d/%d, Z0 = %.6f mm, tilt_x = %.6f deg, tilt_y = %.6f deg\n",
                count(valid), nφ, Z0*1e3, tx, ty)

        return (
            shell=shell, R=Rsh,
            x0=NaN, y0=NaN, z0=Z0,
            tilt_x=tx, tilt_y=ty, tilt_mag=tmag,
            Bcos=Bc, Csin=Cs, amp=amp, rms=rms,
            zcross=zc, valid=valid, method="per_phi",
        )
    end

    BR0 = [mean(BR[:,iz]) for iz in 1:nz]
    c = zcrossings(BR0, Zgrid)
    Z0 = isempty(c) ? NaN : c[argmin(abs.(c .- zguess))]

    println("\nB_R zero-crossing fallback, shell $shell:")
    @printf("  valid = %d/%d, Z0 = %s\n",
            count(valid), nφ, isnan(Z0) ? "NaN" : @sprintf("%.6f mm", Z0*1e3))

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
# Current magnetic X/Y: local B_phi minimization around candidate axis
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

        s = qx*d[1] + qy*d[2] + qz*d[3]

        rx = qx - s*d[1]
        ry = qy - s*d[2]
        rz = qz - s*d[3]

        ρ = sqrt(rx^2 + ry^2 + rz^2)
        ρ < 1e-9 && continue

        rhat = [rx/ρ, ry/ρ, rz/ρ]
        eφ = cross(d, rhat)

        bp = Bx[i]*eφ[1] + By[i]*eφ[2] + Bz[i]*eφ[3]
        bn = max(h.B_mag[i], 1e-30)

        acc += (bp / bn)^2
        n += 1
    end

    return acc / max(n, 1)
end

function magnetic_xy_minimize(h, phys, zt; shell=1)
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
            xn = x + step*dx
            yn = y + step*dy
            val = local_bphi_objective(h, idxs, xn, yn, z0, tx, ty)

            if val < best
                x, y, best = xn, yn, val
                improved = true
            end
        end

        improved || (step *= 0.5)
    end

    println("\nLocal B_phi minimization, shell $shell:")
    @printf("  x0 = %.6f mm, y0 = %.6f mm, objective = %.6e\n", x*1e3, y*1e3, best)

    return (
        shell=shell, R=zt.R,
        x0=x, y0=y, z0=z0,
        tilt_x=tx, tilt_y=ty, tilt_mag=hypot(tx,ty),
        objective=best,
    )
end

# ═══════════════════════════════════════════════════════════════
# Fourier magnetic X/Y: B_phi n=1 / <B_R>
# ═══════════════════════════════════════════════════════════════

function magnetic_xy_fourier(h; shell=1)
    nφ,nz,Rsh,Zgrid,φgrid = shellmeta(h, shell)
    BR = reshape(h.B_R[h.shell .== shell], nφ, nz)
    BP = reshape(h.B_phi[h.shell .== shell], nφ, nz)

    BR0 = [mean(BR[:,iz]) for iz in 1:nz]
    cut = BR_MIN_FRAC_FOR_FOURIER_XY * maximum(abs.(BR0))

    x = fill(NaN, nz)
    y = fill(NaN, nz)
    valid = falses(nz)

    for iz in 1:nz
        abs(BR0[iz]) < cut && continue
        _, a, b = fit_cossin(φgrid, BP[:,iz])

        # B_phi ≈ <B_R>/R * (x0*sinφ - y0*cosφ)
        x[iz] =  BPHI_SIGN * b * Rsh / BR0[iz]
        y[iz] = -BPHI_SIGN * a * Rsh / BR0[iz]

        valid[iz] = true
    end

    if count(valid) == 0
        return (shell=shell, R=Rsh, x0=NaN, y0=NaN)
    end

    w = abs.(BR0[valid]).^2
    x0 = wmean(x[valid], w)
    y0 = wmean(y[valid], w)

    println("\nFourier B_phi n=1 XY, shell $shell:")
    @printf("  x0 = %.6f mm, y0 = %.6f mm\n", x0*1e3, y0*1e3)

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
# Summary table
# ═══════════════════════════════════════════════════════════════

function print_axis_row(method, label, R, phys, a)
    dx = (a.x0 - phys.x0)*1e3
    dy = (a.y0 - phys.y0)*1e3
    dz = (a.z0 - phys.z0)*1e3
    dtx = a.tilt_x - phys.tilt_x
    dty = a.tilt_y - phys.tilt_y

    @printf("%-12s %-10s %9.1f %11.3f %11.3f %11.3f %10.5f %10.5f %11.3f %11.3f %11.3f %10.5f %10.5f\n",
        method, label, R*1e3,
        a.x0*1e3, a.y0*1e3, a.z0*1e3, a.tilt_x, a.tilt_y,
        dx, dy, dz, dtx, dty)
end

function print_summary(phys, sin1, sin2, sinavg, fou1, fou2, fouavg, zt1, zt2)
    println("\n" * "═"^150)
    println("PHYSICAL AXIS VS MAGNETIC AXES")
    println("═"^150)
    println("Columns: x/y/z in mm, tilts in deg, deltas = magnetic - physical")
    println("─"^150)
    @printf("%-12s %-10s %9s %11s %11s %11s %10s %10s %11s %11s %11s %10s %10s\n",
        "method", "radius", "R[mm]", "x", "y", "z", "tilt_x", "tilt_y",
        "Δx", "Δy", "Δz", "Δtx", "Δty")
    println("─"^150)

    print_axis_row("physical", "-", phys.Rfit, phys, phys)

    println("─"^150)
    print_axis_row("sinusoid", "R1", sin1.R, phys, sin1)
    print_axis_row("sinusoid", "R2", sin2.R, phys, sin2)
    print_axis_row("sinusoid", "average", NaN, phys, sinavg)

    println("─"^150)
    print_axis_row("fourier", "R1", fou1.R, phys, fou1)
    print_axis_row("fourier", "R2", fou2.R, phys, fou2)
    print_axis_row("fourier", "average", NaN, phys, fouavg)

    println("─"^150)
    @printf("%-12s %-10s %9s %11s %11s %11s %10s %10s %11s %11s %11s %10s %10s\n",
        "BR RMS", "R1", "", "", "", @sprintf("%.3f", zt1.rms*1e3), "", "", "", "", "", "", "")
    @printf("%-12s %-10s %9s %11s %11s %11s %10s %10s %11s %11s %11s %10s %10s\n",
        "BR RMS", "R2", "", "", "", @sprintf("%.3f", zt2.rms*1e3), "", "", "", "", "", "", "")
    println("═"^150)
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
    for (si,cs) in enumerate(coils), j in 1:cs.ncoil, k in 1:cs.s
        pal = COIL_PALETTES[mod1(si,length(COIL_PALETTES))]
        c = pal[mod1(j,length(pal))]

        x=collect(vec(cs.x[j,k,:])) .* 1e3
        y=collect(vec(cs.y[j,k,:])) .* 1e3
        z=collect(vec(cs.z[j,k,:])) .* 1e3

        if (x[1]-x[end])^2 + (y[1]-y[end])^2 + (z[1]-z[end])^2 > 1e-10
            push!(x,x[1]); push!(y,y[1]); push!(z,z[1])
        end

        lines!(ax,x,y,z; color=c, linewidth=4)
    end

    lines!(ax,[NaN,NaN],[NaN,NaN],[NaN,NaN];
           color=:royalblue, linewidth=4, label="coil geometry")
end

function plot_axis(h, phys, sinavg, fouavg, coils)
    zmin = isnan(AXIS_PLOT_ZMIN_M) ? minimum([minimum(h.Z), phys.Zmin]) : AXIS_PLOT_ZMIN_M
    zmax = isnan(AXIS_PLOT_ZMAX_M) ? maximum([maximum(h.Z), phys.Zmax]) : AXIS_PLOT_ZMAX_M
    z = collect(range(zmin, zmax, length=200))

    fig = Figure(size=(1300,950), backgroundcolor=:white)

    ax = Axis3(
        fig[1,1],
        xlabel="X [mm]",
        ylabel="Y [mm]",
        zlabel="Z [mm]",
        title="Physical Axis vs Sinusoid Magnetic vs Fourier Magnetic",
        aspect=:data,
    )

    SHOW_COILS && plot_coils!(ax, coils)

    if SHOW_HALL_PROBES
        scatter!(ax, h.x.*1e3, h.y.*1e3, h.Z.*1e3;
                 color=(:gray,0.86), markersize=5, label="Hall probes")
    end

    xp,yp,zp = axis_line(phys.x0, phys.y0, phys.z0, phys.tilt_x, phys.tilt_y, z)
    xs,ys,zs = axis_line(sinavg.x0, sinavg.y0, sinavg.z0, sinavg.tilt_x, sinavg.tilt_y, z)
    xf,yf,zf = axis_line(fouavg.x0, fouavg.y0, fouavg.z0, fouavg.tilt_x, fouavg.tilt_y, z)

    lines!(ax, xp.*1e3, yp.*1e3, zp.*1e3;
           color=:black, linewidth=6, linestyle=:dash, label="physical axis")
    scatter!(ax, [phys.x0*1e3], [phys.y0*1e3], [phys.z0*1e3];
             color=:black, marker=:utriangle, markersize=20, label="physical center")

    lines!(ax, xs.*1e3, ys.*1e3, zs.*1e3;
           color=:crimson, linewidth=6, label="sinusoid/local-Bφ magnetic")
    scatter!(ax, [sinavg.x0*1e3], [sinavg.y0*1e3], [sinavg.z0*1e3];
             color=:crimson, markersize=20, label="sinusoid mag center")

    lines!(ax, xf.*1e3, yf.*1e3, zf.*1e3;
           color=:forestgreen, linewidth=6, label="Fourier n=1 magnetic")
    scatter!(ax, [fouavg.x0*1e3], [fouavg.y0*1e3], [fouavg.z0*1e3];
             color=:forestgreen, marker=:diamond, markersize=20, label="Fourier mag center")

    lines!(ax, zeros(length(z)), zeros(length(z)), z.*1e3;
           color=(:black,0.35), linestyle=:dot, linewidth=2, label="machine Z")

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

    # Z/tilt from B_R zero-crossing sinusoid, at both radii.
    zt1 = magnetic_z_tilt(h, phys; shell=1)
    zt2 = magnetic_z_tilt(h, phys; shell=2)

    # Current magnetic method:
    #   Z/tilt from B_R sinusoid + X/Y from local B_phi minimization.
    sin1 = magnetic_xy_minimize(h, phys, zt1; shell=1)
    sin2 = magnetic_xy_minimize(h, phys, zt2; shell=2)
    sinavg = average_axes(sin1, sin2)

    # Fourier magnetic method:
    #   X/Y from B_phi n=1 / <B_R>, Z/tilt from same B_R sinusoid.
    fxy1 = magnetic_xy_fourier(h; shell=1)
    fxy2 = magnetic_xy_fourier(h; shell=2)
    fou1 = combine_axis(fxy1, zt1)
    fou2 = combine_axis(fxy2, zt2)
    fouavg = average_axes(fou1, fou2)

    print_summary(phys, sin1, sin2, sinavg, fou1, fou2, fouavg, zt1, zt2)

    if ENABLE_AXIS_PLOT
        fig = plot_axis(h, phys, sinavg, fouavg, coils)
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
