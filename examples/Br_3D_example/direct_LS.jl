using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using CSV
using DataFrames
using LinearAlgebra
using Statistics
using Printf

# ============================================================
# User settings
# ============================================================

input_csv = joinpath(@__DIR__, "hall_probe_pf1_AS_357_noT_small.csv")
output_csv = joinpath(@__DIR__, "magnetic_axis_two_shell_global_LS.csv")

# Nominal / assumed axis.
# If you know the design major radius, put it here.
# Otherwise leave R_nom = 0.0 and interpret the printed mean R0.
R_nom = 0.0
Z_nom = 0.0

# Recommended first settings for your file.
# If this runs well, try field_fourier_order = 2 and axis_fourier_order = 2.
field_fourier_order = 1
axis_fourier_order = 1

# true = include [dR^2, dR*dZ, dZ^2] terms.
# If Newton fails or condition number is huge, set this false.
use_quadratic_RZ = false

# Number of toroidal points where reconstructed center is evaluated.
nphi_eval = 72

# Newton solve settings.
newton_maxiter = 30
newton_tol = 1e-12

# Initial guess for magnetic center.
# Midpoint between the two measured shells:
#
#     R1 = 0.736061370832623
#     R2 = 1.1180696929404714
#
R_initial = 0.9270655318865472

# Strongly suggested by z symmetry in your file.
Z_initial = 0.007

# ============================================================
# CSV reader
# ============================================================

function read_field_csv(path::AbstractString)
    if !isfile(path)
        error("Input CSV does not exist: $path")
    end

    lines = readlines(path)

    possible_headers = [
        "x,y,z,R,phi,shell,B_R,B_phi,B_Z,B_mag",
        "x,y,z,R,phi,shell_id,B_R,B_phi,B_Z,B_mag",
    ]

    normalized_lines = replace.(strip.(lines), " " => "")

    header_idx = findfirst(line -> line in possible_headers, normalized_lines)

    if header_idx === nothing
        error("""
        Could not find expected CSV header.

        Expected one of:

            x,y,z,R,phi,shell,B_R,B_phi,B_Z,B_mag
            x,y,z,R,phi,shell_id,B_R,B_phi,B_Z,B_mag

        Found first few lines:

        $(join(lines[1:min(end, 8)], "\n"))
        """)
    end

    csv_text = join(lines[header_idx:end], "\n")

    df = CSV.read(
        IOBuffer(csv_text),
        DataFrame;
        delim = ',',
        header = 1,
        normalizenames = false,
        ignorerepeated = false,
        stripwhitespace = true
    )

    return df
end

function ensure_numeric_columns!(df::DataFrame, cols::Vector{Symbol})
    for c in cols
        if !hasproperty(df, c)
            error("Missing required column: $c. Found columns: $(names(df))")
        end

        df[!, c] = Float64.(df[!, c])
    end

    return df
end

function shell_column_name(df::DataFrame)
    if hasproperty(df, :shell_id)
        return :shell_id
    elseif hasproperty(df, :shell)
        return :shell
    else
        return nothing
    end
end

function print_dataset_summary(df::DataFrame)
    println("============================================================")
    println("Dataset summary")
    println("============================================================")
    println("Rows      : ", nrow(df))
    println("Columns   : ", names(df))
    println("R range   : ", extrema(df.R))
    println("z range   : ", extrema(df.z))
    println("phi range : ", extrema(df.phi))
    println("B_R range : ", extrema(df.B_R))
    println("B_Z range : ", extrema(df.B_Z))

    shell_col = shell_column_name(df)

    if shell_col !== nothing
        println()
        println("Shell summary:")

        for g in groupby(df, shell_col)
            sid = g[1, shell_col]
            println(
                "  shell $(sid): ",
                "rows=$(nrow(g)), ",
                "R unique=$(sort(unique(g.R))), ",
                "z count=$(length(unique(g.z))), ",
                "phi count=$(length(unique(g.phi)))"
            )
        end
    end

    println()
end

# ============================================================
# Fourier utilities
# ============================================================

function fourier_basis(phi::Real, N::Int)
    fb = ones(1 + 2N)

    for k in 1:N
        fb[2k]     = cos(k * phi)
        fb[2k + 1] = sin(k * phi)
    end

    return fb
end

function fourier_matrix(phi::AbstractVector, N::Int)
    A = ones(length(phi), 1 + 2N)

    for k in 1:N
        A[:, 2k]     .= cos.(k .* phi)
        A[:, 2k + 1] .= sin.(k .* phi)
    end

    return A
end

function fourier_eval(phi::AbstractVector, coeff::AbstractVector, N::Int)
    y = fill(coeff[1], length(phi))

    for k in 1:N
        y .+= coeff[2k]     .* cos.(k .* phi)
        y .+= coeff[2k + 1] .* sin.(k .* phi)
    end

    return y
end

function fourier_derivative(phi::AbstractVector, coeff::AbstractVector, N::Int)
    dy = zeros(length(phi))

    for k in 1:N
        dy .+= -k .* coeff[2k]     .* sin.(k .* phi)
        dy .+=  k .* coeff[2k + 1] .* cos.(k .* phi)
    end

    return dy
end

# ============================================================
# R/Z polynomial basis
#
# Linear:
#
#     1, dR, dZ
#
# Quadratic:
#
#     1, dR, dZ, dR^2, dR*dZ, dZ^2
#
# ============================================================

function rz_basis(
    R::Real,
    Z::Real,
    Rref::Real,
    Zref::Real;
    quadratic::Bool = true
)
    dR = R - Rref
    dZ = Z - Zref

    if quadratic
        return [1.0, dR, dZ, dR^2, dR * dZ, dZ^2]
    else
        return [1.0, dR, dZ]
    end
end

function rz_basis_dR(
    R::Real,
    Z::Real,
    Rref::Real,
    Zref::Real;
    quadratic::Bool = true
)
    dR = R - Rref
    dZ = Z - Zref

    if quadratic
        return [0.0, 1.0, 0.0, 2.0 * dR, dZ, 0.0]
    else
        return [0.0, 1.0, 0.0]
    end
end

function rz_basis_dZ(
    R::Real,
    Z::Real,
    Rref::Real,
    Zref::Real;
    quadratic::Bool = true
)
    dR = R - Rref
    dZ = Z - Zref

    if quadratic
        return [0.0, 0.0, 1.0, 0.0, dR, 2.0 * dZ]
    else
        return [0.0, 0.0, 1.0]
    end
end

# ============================================================
# Global field model:
#
#     B(R,Z,phi) =
#         sum_i sum_j c_ij * rz_i(R,Z) * fourier_j(phi)
#
# Fit both B_R and B_Z with the same design matrix.
# ============================================================

function build_design(
    df::DataFrame;
    Nphi::Int,
    Rref::Float64,
    Zref::Float64,
    quadratic::Bool
)
    n = nrow(df)

    nrz = quadratic ? 6 : 3
    nfb = 1 + 2Nphi

    X = zeros(n, nrz * nfb)

    for i in 1:n
        rb = rz_basis(
            df.R[i],
            df.z[i],
            Rref,
            Zref;
            quadratic = quadratic
        )

        fb = fourier_basis(df.phi[i], Nphi)

        col = 1

        for a in 1:nrz
            for b in 1:nfb
                X[i, col] = rb[a] * fb[b]
                col += 1
            end
        end
    end

    return X
end

function fit_field_model(
    df::DataFrame;
    Nphi::Int = 1,
    quadratic::Bool = true
)
    Rref = mean(df.R)
    Zref = mean(df.z)

    X = build_design(
        df;
        Nphi = Nphi,
        Rref = Rref,
        Zref = Zref,
        quadratic = quadratic
    )

    coeff_BR = X \ df.B_R
    coeff_BZ = X \ df.B_Z

    BR_fit = X * coeff_BR
    BZ_fit = X * coeff_BZ

    rms_BR = sqrt(mean((BR_fit .- df.B_R).^2))
    rms_BZ = sqrt(mean((BZ_fit .- df.B_Z).^2))

    println("============================================================")
    println("Global two-shell field model")
    println("============================================================")
    println("Fourier order Nphi : ", Nphi)
    println("Quadratic R/Z      : ", quadratic)
    println("Reference Rref     : ", Rref)
    println("Reference Zref     : ", Zref)
    println("Design size        : ", size(X))
    println("Design rank        : ", rank(X))
    println("Design cond        : ", cond(X))
    println("RMS residual B_R   : ", rms_BR)
    println("RMS residual B_Z   : ", rms_BZ)
    println()

    return (
        Nphi = Nphi,
        quadratic = quadratic,
        Rref = Rref,
        Zref = Zref,
        coeff_BR = coeff_BR,
        coeff_BZ = coeff_BZ,
        rms_BR = rms_BR,
        rms_BZ = rms_BZ
    )
end

function eval_model_component(coeff, R, Z, phi, model)
    rb = rz_basis(
        R,
        Z,
        model.Rref,
        model.Zref;
        quadratic = model.quadratic
    )

    fb = fourier_basis(phi, model.Nphi)

    nrz = length(rb)
    nfb = length(fb)

    y = 0.0
    col = 1

    for a in 1:nrz
        for b in 1:nfb
            y += coeff[col] * rb[a] * fb[b]
            col += 1
        end
    end

    return y
end

function eval_model_component_derivs(coeff, R, Z, phi, model)
    rb_R = rz_basis_dR(
        R,
        Z,
        model.Rref,
        model.Zref;
        quadratic = model.quadratic
    )

    rb_Z = rz_basis_dZ(
        R,
        Z,
        model.Rref,
        model.Zref;
        quadratic = model.quadratic
    )

    fb = fourier_basis(phi, model.Nphi)

    nrz = length(rb_R)
    nfb = length(fb)

    dBdR = 0.0
    dBdZ = 0.0
    col = 1

    for a in 1:nrz
        for b in 1:nfb
            c = coeff[col]
            dBdR += c * rb_R[a] * fb[b]
            dBdZ += c * rb_Z[a] * fb[b]
            col += 1
        end
    end

    return dBdR, dBdZ
end

function eval_BR_BZ_J(model, R, Z, phi)
    BR = eval_model_component(model.coeff_BR, R, Z, phi, model)
    BZ = eval_model_component(model.coeff_BZ, R, Z, phi, model)

    dBRdR, dBRdZ = eval_model_component_derivs(
        model.coeff_BR,
        R,
        Z,
        phi,
        model
    )

    dBZdR, dBZdZ = eval_model_component_derivs(
        model.coeff_BZ,
        R,
        Z,
        phi,
        model
    )

    J = [
        dBRdR  dBRdZ;
        dBZdR  dBZdZ
    ]

    return BR, BZ, J
end

# ============================================================
# Newton solve for magnetic center at each phi
# ============================================================

function solve_center_at_phi(
    model,
    phi;
    R0::Float64,
    Z0::Float64,
    maxiter::Int = 30,
    tol::Float64 = 1e-12
)
    R = R0
    Z = Z0

    converged = false
    final_res = Inf
    final_cond = Inf
    final_det = NaN

    for iter in 1:maxiter
        BR, BZ, J = eval_BR_BZ_J(model, R, Z, phi)

        F = [BR, BZ]

        final_res = norm(F)
        final_cond = cond(J)
        final_det = det(J)

        if final_res < tol
            converged = true
            return R, Z, converged, iter, final_res, final_cond, final_det
        end

        if abs(final_det) < 1e-24
            return R, Z, false, iter, final_res, final_cond, final_det
        end

        Δ = -J \ F

        # Damping to avoid large Newton jumps.
        α = 1.0
        improved = false

        for _ in 1:10
            Rtrial = R + α * Δ[1]
            Ztrial = Z + α * Δ[2]

            BRt, BZt, _ = eval_BR_BZ_J(model, Rtrial, Ztrial, phi)

            if norm([BRt, BZt]) < final_res
                R = Rtrial
                Z = Ztrial
                improved = true
                break
            end

            α *= 0.5
        end

        if !improved
            R += Δ[1]
            Z += Δ[2]
        end

        if norm(Δ) < tol
            BR2, BZ2, J2 = eval_BR_BZ_J(model, R, Z, phi)
            converged = norm([BR2, BZ2]) < 10tol
            return R, Z, converged, iter, norm([BR2, BZ2]), cond(J2), det(J2)
        end
    end

    BR, BZ, J = eval_BR_BZ_J(model, R, Z, phi)

    return R, Z, converged, maxiter, norm([BR, BZ]), cond(J), det(J)
end

function reconstruct_axis(
    model;
    nphi::Int = 72,
    R_initial::Float64,
    Z_initial::Float64,
    R_nom::Float64 = 0.0,
    Z_nom::Float64 = 0.0,
    maxiter::Int = 30,
    tol::Float64 = 1e-12
)
    out = DataFrame(
        phi = Float64[],
        Rc = Float64[],
        Zc = Float64[],
        dR = Float64[],
        dZ = Float64[],
        converged = Bool[],
        iterations = Int[],
        residual = Float64[],
        cond_J = Float64[],
        det_J = Float64[]
    )

    R_guess = R_initial
    Z_guess = Z_initial

    for j in 0:(nphi - 1)
        φ = 2π * j / nphi

        Rc, Zc, conv, iters, res, cJ, dJ = solve_center_at_phi(
            model,
            φ;
            R0 = R_guess,
            Z0 = Z_guess,
            maxiter = maxiter,
            tol = tol
        )

        push!(
            out,
            (
                phi = φ,
                Rc = Rc,
                Zc = Zc,
                dR = Rc - R_nom,
                dZ = Zc - Z_nom,
                converged = conv,
                iterations = iters,
                residual = res,
                cond_J = cJ,
                det_J = dJ
            )
        )

        # Continue from previous good solution.
        if conv
            R_guess = Rc
            Z_guess = Zc
        end
    end

    return out
end

# ============================================================
# Axis Fourier smoothing and reports
# ============================================================

function fit_axis_fourier(
    axis::DataFrame;
    N::Int = 1,
    R_nom = 0.0,
    Z_nom = 0.0
)
    good = axis[axis.converged .== true, :]

    if nrow(good) == 0
        error("No converged axis points available for Fourier fit.")
    end

    if nrow(good) < 1 + 2N
        N = max(0, floor(Int, (nrow(good) - 1) / 2))
        @warn "Reducing axis Fourier order to N=$N due to limited converged points"
    end

    A = fourier_matrix(good.phi, N)

    coeff_R = A \ good.Rc
    coeff_Z = A \ good.Zc

    Rc_fit = fourier_eval(axis.phi, coeff_R, N)
    Zc_fit = fourier_eval(axis.phi, coeff_Z, N)

    dRc_dphi = fourier_derivative(axis.phi, coeff_R, N)
    dZc_dphi = fourier_derivative(axis.phi, coeff_Z, N)

    R_for_tilt = map(
        r -> abs(r) < 1e-12 ? copysign(1e-12, r == 0 ? 1.0 : r) : r,
        Rc_fit
    )

    radial_tilt_rad = atan.(dRc_dphi ./ R_for_tilt)
    vertical_tilt_rad = atan.(dZc_dphi ./ R_for_tilt)

    result = copy(axis)

    result.Rc_fit = Rc_fit
    result.Zc_fit = Zc_fit

    result.dR_fit = Rc_fit .- R_nom
    result.dZ_fit = Zc_fit .- Z_nom

    result.dRc_dphi = dRc_dphi
    result.dZc_dphi = dZc_dphi

    result.radial_tilt_rad = radial_tilt_rad
    result.vertical_tilt_rad = vertical_tilt_rad

    result.radial_tilt_mrad = 1e3 .* radial_tilt_rad
    result.vertical_tilt_mrad = 1e3 .* vertical_tilt_rad

    result.xc = Rc_fit .* cos.(axis.phi)
    result.yc = Rc_fit .* sin.(axis.phi)
    result.zc = Zc_fit

    return result, coeff_R, coeff_Z, N
end

function print_toroidal_alignment_report(coeff_R, coeff_Z)
    println()
    println("============================================================")
    println("Toroidal magnetic-axis alignment report")
    println("============================================================")

    R0 = coeff_R[1]
    Z0 = coeff_Z[1]

    # For a shifted circular/toroidal axis:
    #
    #     R(phi) ≈ R0 + x0*cos(phi) + y0*sin(phi)
    #
    x_shift = length(coeff_R) >= 2 ? coeff_R[2] : 0.0
    y_shift = length(coeff_R) >= 3 ? coeff_R[3] : 0.0

    # For a tilted toroidal plane:
    #
    #     Z(phi) ≈ Z0 + tx*R0*cos(phi) + ty*R0*sin(phi)
    #
    z_cos = length(coeff_Z) >= 2 ? coeff_Z[2] : 0.0
    z_sin = length(coeff_Z) >= 3 ? coeff_Z[3] : 0.0

    dzdx = z_cos / R0
    dzdy = z_sin / R0

    tilt_x_rad = atan(dzdx)
    tilt_y_rad = atan(dzdy)

    @printf("Mean magnetic major radius R0        = %+ .9e m\n", R0)
    @printf("Mean vertical center Z0              = %+ .9e m\n", Z0)
    println()

    @printf("Horizontal x shift                   = %+ .9e m\n", x_shift)
    @printf("Horizontal y shift                   = %+ .9e m\n", y_shift)
    @printf("Horizontal shift magnitude           = %+ .9e m\n", hypot(x_shift, y_shift))
    @printf("Horizontal shift angle               = %+ .9e rad\n", atan(y_shift, x_shift))
    println()

    @printf("Z cos(phi) coefficient               = %+ .9e m\n", z_cos)
    @printf("Z sin(phi) coefficient               = %+ .9e m\n", z_sin)
    @printf("Approx dz/dx plane slope             = %+ .9e\n", dzdx)
    @printf("Approx dz/dy plane slope             = %+ .9e\n", dzdy)
    @printf("Approx x-plane tilt                  = %+ .9e rad = %+ .6f mrad\n", tilt_x_rad, 1e3 * tilt_x_rad)
    @printf("Approx y-plane tilt                  = %+ .9e rad = %+ .6f mrad\n", tilt_y_rad, 1e3 * tilt_y_rad)
    println()

    if length(coeff_R) >= 5
        println("Higher-order R Fourier terms:")
        for k in 2:((length(coeff_R) - 1) ÷ 2)
            c = coeff_R[2k]
            s = coeff_R[2k + 1]
            @printf(
                "  R m=%d: cos = %+ .9e m, sin = %+ .9e m, amp = %+ .9e m\n",
                k,
                c,
                s,
                hypot(c, s)
            )
        end
        println()
    end

    if length(coeff_Z) >= 5
        println("Higher-order Z Fourier terms:")
        for k in 2:((length(coeff_Z) - 1) ÷ 2)
            c = coeff_Z[2k]
            s = coeff_Z[2k + 1]
            @printf(
                "  Z m=%d: cos = %+ .9e m, sin = %+ .9e m, amp = %+ .9e m\n",
                k,
                c,
                s,
                hypot(c, s)
            )
        end
        println()
    end
end

# ============================================================
# Main
# ============================================================

df = read_field_csv(input_csv)

ensure_numeric_columns!(
    df,
    [:x, :y, :z, :R, :phi, :B_R, :B_phi, :B_Z, :B_mag]
)

print_dataset_summary(df)

model = fit_field_model(
    df;
    Nphi = field_fourier_order,
    quadratic = use_quadratic_RZ
)

axis_raw = reconstruct_axis(
    model;
    nphi = nphi_eval,
    R_initial = R_initial,
    Z_initial = Z_initial,
    R_nom = R_nom,
    Z_nom = Z_nom,
    maxiter = newton_maxiter,
    tol = newton_tol
)

println("============================================================")
println("Axis reconstruction solve status")
println("============================================================")
println("Total phi points      : ", nrow(axis_raw))
println("Converged phi points  : ", count(axis_raw.converged))
println("Max residual          : ", maximum(axis_raw.residual))
println("Median residual       : ", median(axis_raw.residual))
println("Max cond(J)           : ", maximum(axis_raw.cond_J))
println("Median cond(J)        : ", median(axis_raw.cond_J))
println()

if count(axis_raw.converged) == 0
    error("No magnetic-axis points converged.")
end

axis_fit, coeff_R, coeff_Z, actual_axis_N = fit_axis_fourier(
    axis_raw;
    N = axis_fourier_order,
    R_nom = R_nom,
    Z_nom = Z_nom
)

CSV.write(output_csv, axis_fit)

println("============================================================")
println("Magnetic center reconstruction complete")
println("============================================================")
println("Wrote: ", output_csv)
println()

println("Axis Fourier order actually used: ", actual_axis_N)
println()

println("Fourier coefficients for Rc(phi):")
println(coeff_R)
println()

println("Fourier coefficients for Zc(phi):")
println(coeff_Z)
println()

print_toroidal_alignment_report(coeff_R, coeff_Z)

println("============================================================")
println("Per-phi reconstructed magnetic center summary")
println("============================================================")

summary_cols = [
    :phi,
    :Rc_fit,
    :Zc_fit,
    :dR_fit,
    :dZ_fit,
    :radial_tilt_mrad,
    :vertical_tilt_mrad,
    :converged,
    :residual,
    :cond_J
]

show(axis_fit[:, summary_cols], allrows = true, allcols = true)
println()
println()

println("Done.")