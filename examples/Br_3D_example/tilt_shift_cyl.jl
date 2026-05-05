using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using GeneralizedPerturbedEquilibrium
using LinearAlgebra
using Statistics
using Printf
using GLMakie
using Optim
using DelimitedFiles

set_theme!(Theme(
    fontsize = 28, font = :bold,
    Axis = (xlabelsize=32, ylabelsize=32, xticklabelsize=24, yticklabelsize=24,
            titlesize=32, xlabelfont="CMU Serif Bold", ylabelfont="CMU Serif Bold",
            titlefont="CMU Serif Bold", xticklabelfont="CMU Serif", yticklabelfont="CMU Serif"),
    Colorbar = (labelsize=26, ticklabelsize=20,
                labelfont="CMU Serif Bold", ticklabelfont="CMU Serif"),
    Label = (fontsize=32, font="CMU Serif Bold"),
))

const _FT            = GeneralizedPerturbedEquilibrium.ForcingTerms
const _read_coil_dat = _FT.read_coil_dat

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
HALL_DATA_FILE = joinpath(@__DIR__, "hall_probe_rzp_357.csv")

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
"""
    load_hall_data(path) -> NamedTuple

Read the Hall-probe CSV, skipping comment lines (#) and the header row.
Returns positions in cartesian (x,y,z) and the field components in
cylindrical form (B_R, B_phi, B_Z) along with B in cartesian (Bx,By,Bz).
"""
function load_hall_data(path::AbstractString)
    # Read all lines, drop comment lines starting with '#'
    raw = readlines(path)
    data_lines = filter(l -> !startswith(strip(l), "#") && !isempty(strip(l)), raw)
    # First non-comment line is the header
    header = split(data_lines[1], ',')
    body   = data_lines[2:end]

    n = length(body)
    M = Matrix{Float64}(undef, n, length(header))
    for (i, line) in enumerate(body)
        M[i, :] = parse.(Float64, split(line, ','))
    end

    col(name) = findfirst(==(name), strip.(header))
    x   = M[:, col("x")]
    y   = M[:, col("y")]
    z   = M[:, col("z")]
    R   = M[:, col("R")]
    phi = M[:, col("phi")]
    Z   = M[:, col("Z")]
    BR  = M[:, col("B_R")]
    Bph = M[:, col("B_phi")]
    BZ  = M[:, col("B_Z")]
    Bm  = M[:, col("B_mag")]

    # Reconstruct cartesian B from cylindrical components
    cphi = cos.(phi); sphi = sin.(phi)
    Bx = BR .* cphi .- Bph .* sphi
    By = BR .* sphi .+ Bph .* cphi
    Bz = BZ

    return (x=x, y=y, z=z, R=R, phi=phi, Z=Z,
            BR=BR, Bphi=Bph, BZ=BZ, Bmag=Bm,
            Bx=Bx, By=By, Bz=Bz)
end

# ═══════════════════════════════════════════════════════════════
# COORDINATE TRANSFORM
# ═══════════════════════════════════════════════════════════════
"""
    rotation_matrix(αx, αy) -> SMatrix{3,3}

Build a rotation matrix Ry(αy) * Rx(αx) that takes the measurement-frame
ẑ axis into the magnetic-field symmetry axis direction.
αx is rotation about x-axis, αy is rotation about y-axis.
A rotation about the symmetry axis itself (αz) is unobservable under
cylindrical symmetry and is therefore omitted (5 DOF total).
"""
function rotation_matrix(αx::Real, αy::Real)
    cx, sx = cos(αx), sin(αx)
    cy, sy = cos(αy), sin(αy)
    Rx = [1.0  0.0  0.0;
          0.0  cx  -sx;
          0.0  sx   cx]
    Ry = [ cy  0.0  sy;
          0.0  1.0  0.0;
          -sy  0.0  cy]
    return Ry * Rx
end

"""
    transform_to_field_frame(x,y,z, Bx,By,Bz, x0,y0,z0, αx,αy)

Translate the probe positions by (-x0,-y0,-z0), then rotate both
positions and field vectors by Rᵀ so that the proposed magnetic-axis
direction becomes the new ẑ′. Returns (R′, Z′, BR′, Bφ′, BZ′).
"""
function transform_to_field_frame(x, y, z, Bx, By, Bz,
                                  x0, y0, z0, αx, αy)
    Rmat  = rotation_matrix(αx, αy)
    Rinv  = transpose(Rmat)            # rotation is orthogonal

    n = length(x)
    Rp  = Vector{Float64}(undef, n)
    Zp  = Vector{Float64}(undef, n)
    BRp = Vector{Float64}(undef, n)
    Bφp = Vector{Float64}(undef, n)
    BZp = Vector{Float64}(undef, n)

    @inbounds for i in 1:n
        # Translate position
        dx = x[i] - x0
        dy = y[i] - y0
        dz = z[i] - z0
        # Rotate position into field frame
        xp = Rinv[1,1]*dx + Rinv[1,2]*dy + Rinv[1,3]*dz
        yp = Rinv[2,1]*dx + Rinv[2,2]*dy + Rinv[2,3]*dz
        zp = Rinv[3,1]*dx + Rinv[3,2]*dy + Rinv[3,3]*dz
        # Rotate field vector into field frame
        bxp = Rinv[1,1]*Bx[i] + Rinv[1,2]*By[i] + Rinv[1,3]*Bz[i]
        byp = Rinv[2,1]*Bx[i] + Rinv[2,2]*By[i] + Rinv[2,3]*Bz[i]
        bzp = Rinv[3,1]*Bx[i] + Rinv[3,2]*By[i] + Rinv[3,3]*Bz[i]

        rp  = hypot(xp, yp)
        cφ  = rp > 0 ? xp/rp : 1.0
        sφ  = rp > 0 ? yp/rp : 0.0

        Rp[i]  = rp
        Zp[i]  = zp
        BRp[i] =  bxp*cφ + byp*sφ
        Bφp[i] = -bxp*sφ + byp*cφ
        BZp[i] = bzp
    end
    return Rp, Zp, BRp, Bφp, BZp
end

# ═══════════════════════════════════════════════════════════════
# AXISYMMETRY OBJECTIVE
# ═══════════════════════════════════════════════════════════════
"""
    axisymmetry_residual(params, data)

For a candidate (x0, y0, z0, αx, αy), transform all probes into the
proposed field frame, then bin them by (R′, Z′) and measure how much
the field components vary at fixed (R′, Z′). For a perfectly
axisymmetric field, B_R′ and B_Z′ depend only on (R′, Z′) and B_φ′ = 0.
The cost is the sum of variances of (B_R′, B_Z′) across phi at each
(R′,Z′) bin, plus the mean square of B_φ′.
"""
function axisymmetry_residual(params, data)
    x0, y0, z0, αx, αy = params
    Rp, Zp, BRp, Bφp, BZp = transform_to_field_frame(
        data.x, data.y, data.z, data.Bx, data.By, data.Bz,
        x0, y0, z0, αx, αy)

    # Bin by (R, Z) using rounded values; tolerance chosen from the
    # original sampling spacing (~0.22 m in R, ~0.265 m in Z). We use
    # a finer rounding than the spacing so original concentric rings
    # remain grouped after small perturbations.
    keyR = round.(Rp; digits=2)
    keyZ = round.(Zp; digits=2)

    bins = Dict{Tuple{Float64,Float64}, Vector{Int}}()
    for i in eachindex(keyR)
        k = (keyR[i], keyZ[i])
        push!(get!(bins, k, Int[]), i)
    end

    cost = 0.0
    nbins = 0
    for (_, idxs) in bins
        length(idxs) < 2 && continue
        # Variance of B_R' and B_Z' across phi at this (R',Z')
        vR = var(@view BRp[idxs])
        vZ = var(@view BZp[idxs])
        # B_phi' should be zero at every probe
        mφ2 = mean(abs2, @view Bφp[idxs])
        cost += vR + vZ + mφ2
        nbins += 1
    end
    return nbins == 0 ? Inf : cost / nbins
end

# ═══════════════════════════════════════════════════════════════
# CENTER ALONG THE AXIS (B_R = 0 plane)
# ═══════════════════════════════════════════════════════════════
"""
    estimate_axial_center(Rp, Zp, BRp; nR=4)

After axisymmetrization, the magnetic 'center' along Z′ is the plane
where B_R′ vanishes for all R′ (the mid-plane of a solenoid-like coil).
We estimate it by, for each of the smallest few R-rings, finding the
zero-crossing of B_R′(Z′) and averaging.
"""
function estimate_axial_center(Rp, Zp, BRp; nR::Int=4)
    keyR = round.(Rp; digits=3)
    uR   = sort(unique(keyR))
    rings = uR[1:min(nR, length(uR))]

    z_centers = Float64[]
    for r in rings
        idx = findall(==(r), keyR)
        zs  = Zp[idx]
        brs = BRp[idx]
        # Average over phi at each Z
        keyZ = round.(zs; digits=3)
        uZ   = sort(unique(keyZ))
        br_avg = [mean(brs[findall(==(z), keyZ)]) for z in uZ]
        # Find zero-crossing by linear interpolation
        for i in 1:length(uZ)-1
            if br_avg[i] * br_avg[i+1] < 0
                t = br_avg[i] / (br_avg[i] - br_avg[i+1])
                push!(z_centers, uZ[i] + t*(uZ[i+1]-uZ[i]))
                break
            end
        end
    end
    return isempty(z_centers) ? NaN : mean(z_centers)
end

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
function main()
    println("Loading Hall-probe data from: $HALL_DATA_FILE")
    data = load_hall_data(HALL_DATA_FILE)
    @printf("  N probes        = %d\n", length(data.x))
    @printf("  Z range         = [%.4f, %.4f] m\n", minimum(data.Z), maximum(data.Z))
    @printf("  R range         = [%.4f, %.4f] m\n", minimum(data.R), maximum(data.R))

    # Initial guess: no shift, no tilt, center at mid-Z
    z_mid0 = 0.5*(minimum(data.z) + maximum(data.z))
    p0 = [0.0, 0.0, z_mid0, 0.0, 0.0]

    println("\nInitial residual = ",
            axisymmetry_residual(p0, data))

    # ── Stage 1: coarse Nelder–Mead in all 5 DOF ────────────────
    println("\nStage 1: Nelder–Mead in (x0, y0, z0, αx, αy) ...")
    obj = p -> axisymmetry_residual(p, data)
    res1 = optimize(obj, p0, NelderMead(),
                    Optim.Options(iterations=5000,
                                  g_tol=1e-12,
                                  show_trace=false))
    p1 = Optim.minimizer(res1)
    @printf("  cost after stage 1 = %.6e\n", Optim.minimum(res1))

    # ── Stage 2: refine with LBFGS via finite differences ───────
    println("\nStage 2: refining with LBFGS ...")
    res2 = try
        optimize(obj, p1, LBFGS(),
                 Optim.Options(iterations=500, g_tol=1e-14);
                 autodiff = :finite)
    catch err
        @warn "LBFGS failed, keeping Nelder-Mead result" exception=err
        res1
    end
    pbest = Optim.minimizer(res2)
    cost  = Optim.minimum(res2)

    x0, y0, z0, αx, αy = pbest

    # ── Recover axis direction in measurement frame ─────────────
    Rmat = rotation_matrix(αx, αy)
    axis_hat = Rmat[:, 3]                        # ẑ′ in measurement frame

    # ── Center along the axis (B_R'=0 plane) in field frame ────
    Rp, Zp, BRp, Bφp, BZp = transform_to_field_frame(
        data.x, data.y, data.z, data.Bx, data.By, data.Bz,
        x0, y0, z0, αx, αy)
    zc_field = estimate_axial_center(Rp, Zp, BRp)
    # Convert that field-frame center back to measurement frame
    center_meas = [x0, y0, z0] .+ axis_hat .* zc_field

    # Cylindrical (R, phi, Z) of the field center
    Rc = hypot(center_meas[1], center_meas[2])
    φc = atan(center_meas[2], center_meas[1])
    Zc = center_meas[3]

    # ── Report ──────────────────────────────────────────────────
    println("\n" * "═"^65)
    println("  MAGNETIC-FIELD COORDINATE FIT  (5 DOF, cyl. symmetry)")
    println("═"^65)
    @printf("  final residual                = %.6e\n", cost)
    println()
    println("  Shifts (origin of field frame, measurement Cartesian):")
    @printf("    x0 = %+.6f  m\n", x0)
    @printf("    y0 = %+.6f  m\n", y0)
    @printf("    z0 = %+.6f  m\n", z0)
    println()
    println("  Tilt angles (about measurement x and y axes):")
    @printf("    αx = %+.6e rad  (%+.4f°)\n", αx, rad2deg(αx))
    @printf("    αy = %+.6e rad  (%+.4f°)\n", αy, rad2deg(αy))
    println()
    println("  Magnetic axis direction (unit vector, meas. frame):")
    @printf("    ê = [%+.8f, %+.8f, %+.8f]\n",
            axis_hat[1], axis_hat[2], axis_hat[3])
    println()
    println("  Magnetic-field center in measurement Cartesian:")
    @printf("    (x,y,z) = (%+.6f, %+.6f, %+.6f) m\n",
            center_meas[1], center_meas[2], center_meas[3])
    println("  Magnetic-field center in measurement cylindrical (R, φ, Z):")
    @printf("    R   = %.6f  m\n", Rc)
    @printf("    φ   = %+.6f rad (%+.4f°)\n", φc, rad2deg(φc))
    @printf("    Z   = %.6f  m\n", Zc)
    println("═"^65)

    # ── Visualisation: residual axisymmetry check ───────────────
    fig = Figure(size=(1400, 600))

    ax1 = Axis(fig[1,1]; xlabel="R′ [m]", ylabel="Z′ [m]",
               title="B_R′ in fitted field frame")
    sc1 = scatter!(ax1, Rp, Zp; color=BRp,
                   colormap=:RdBu, markersize=8)
    Colorbar(fig[1,2], sc1; label="B_R′ [T]")

    ax2 = Axis(fig[1,3]; xlabel="R′ [m]", ylabel="Z′ [m]",
               title="B_φ′ residual (should be ≈ 0)")
    sc2 = scatter!(ax2, Rp, Zp; color=Bφp,
                   colormap=:RdBu, markersize=8)
    Colorbar(fig[1,4], sc2; label="B_φ′ [T]")

    display(fig)

    return (x0=x0, y0=y0, z0=z0, αx=αx, αy=αy,
            axis=axis_hat, center=center_meas,
            R=Rc, phi=φc, Z=Zc, cost=cost,
            fig=fig)
end

result = main()