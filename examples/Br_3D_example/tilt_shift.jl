#!/usr/bin/env julia
# ═══════════════════════════════════════════════════════════════
#
# Reads a coil .dat file, applies user-specified translations
# and rotations (tilts), and writes a new .dat file.
#
# This version uses the SAME pose definition as coil_centering_full.jl:
#   • center = segment-length-weighted centroid of conductor path
#   • orientation = weighted best-fit plane normal from segment midpoints
#
# Transform order:
#   1. Compute weighted coil pose
#   2. Center coil about weighted pose center
#   3. Apply rotation (tilt_x, tilt_y, tilt_z) — extrinsic XYZ Euler angles
#   4. Un-center
#   5. Apply translation
#
# Usage:
#   julia tilt_shift.jl
#
# ═══════════════════════════════════════════════════════════════

using LinearAlgebra
using Printf
using Statistics

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — edit these values
# ═══════════════════════════════════════════════════════════════

INPUT_COIL_FILE  = joinpath(@__DIR__, "sparc_pf1u.dat")
OUTPUT_COIL_FILE = joinpath(@__DIR__, "sparc_pf1u_105070_2010T.dat")

# Translation AFTER rotation, units: meters
SHIFT_X = 0.01
SHIFT_Y = 0.05
SHIFT_Z = 0.07

# Extrinsic XYZ rotations, units: degrees
TILT_X_DEG = 20
TILT_Y_DEG = 10
TILT_Z_DEG = 0.0

VERBOSE = true

# ═══════════════════════════════════════════════════════════════
# Rotation matrices
# ═══════════════════════════════════════════════════════════════

function rotation_x(θ::Float64)
    c, s = cos(θ), sin(θ)
    return [1.0  0.0  0.0;
            0.0   c   -s;
            0.0   s    c]
end

function rotation_y(θ::Float64)
    c, s = cos(θ), sin(θ)
    return [ c   0.0   s;
            0.0  1.0  0.0;
            -s   0.0   c]
end

function rotation_z(θ::Float64)
    c, s = cos(θ), sin(θ)
    return [c   -s   0.0;
            s    c   0.0;
            0.0  0.0  1.0]
end

"""
    build_rotation_matrix(tilt_x_deg, tilt_y_deg, tilt_z_deg)

Build the combined rotation matrix for extrinsic XYZ Euler rotations.
Applied as R = Rz * Ry * Rx  (i.e. Rx first, then Ry, then Rz).
"""
function build_rotation_matrix(tilt_x_deg::Float64, tilt_y_deg::Float64, tilt_z_deg::Float64)
    Rx = rotation_x(deg2rad(tilt_x_deg))
    Ry = rotation_y(deg2rad(tilt_y_deg))
    Rz = rotation_z(deg2rad(tilt_z_deg))
    return Rz * Ry * Rx
end

# ═══════════════════════════════════════════════════════════════
# Coil file I/O
# ═══════════════════════════════════════════════════════════════

function read_coil_file(filepath::String)
    isfile(filepath) || error("Input coil file not found: $filepath")

    lines = readlines(filepath)
    isempty(lines) && error("Coil file is empty: $filepath")

    header_line = lines[1]

    x = Float64[]
    y = Float64[]
    z = Float64[]

    for i in 2:length(lines)
        stripped = strip(lines[i])
        isempty(stripped) && continue

        tokens = split(stripped)
        if length(tokens) >= 3
            try
                push!(x, parse(Float64, tokens[1]))
                push!(y, parse(Float64, tokens[2]))
                push!(z, parse(Float64, tokens[3]))
            catch
                if VERBOSE
                    println("  Skipping unparseable line $i: \"$stripped\"")
                end
            end
        end
    end

    return header_line, x, y, z
end

function write_coil_file(filepath::String, header_line::String,
                         x::Vector{Float64}, y::Vector{Float64}, z::Vector{Float64})
    N = length(x)
    open(filepath, "w") do io
        println(io, header_line)
        for i in 1:N
            @printf(io, "  %14.4e   %14.4e   %14.4e\n", x[i], y[i], z[i])
        end
    end
end

# ═══════════════════════════════════════════════════════════════
# Geometry helpers: same pose definitions as coil_centering_full.jl
# ═══════════════════════════════════════════════════════════════

"""
Return true if the polyline is explicitly closed.
"""
function is_closed_curve(x::Vector{Float64}, y::Vector{Float64}, z::Vector{Float64}; tol=1e-12)
    length(x) >= 2 || return false
    return (x[1]-x[end])^2 + (y[1]-y[end])^2 + (z[1]-z[end])^2 <= tol^2
end

"""
Build segment midpoints and segment lengths from a polyline.
If the curve is not explicitly closed, include the closing segment from N → 1.
"""
function segment_midpoints_lengths(x::Vector{Float64}, y::Vector{Float64}, z::Vector{Float64})
    N = length(x)
    N >= 2 || error("Need at least 2 points")

    closed = is_closed_curve(x, y, z)
    lastseg = closed ? N - 1 : N

    mx = Float64[]
    my = Float64[]
    mz = Float64[]
    w  = Float64[]

    for i in 1:lastseg
        j = (i == N ? 1 : i + 1)

        x1, y1, z1 = x[i], y[i], z[i]
        x2, y2, z2 = x[j], y[j], z[j]

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

function plane_basis_from_normal(n::Vector{Float64})
    ref = abs(n[3]) < 0.9 ? [0.0, 0.0, 1.0] : [1.0, 0.0, 0.0]
    e1 = normalize(cross(ref, n))
    e2 = cross(n, e1)
    return e1, e2
end

"""
Compute the coil pose using the same definition as coil_centering_full.jl:
  - center = segment-length-weighted centroid of segment midpoints
  - normal = weighted SVD best-fit plane normal
  - tilts from normal using:
      normal ∝ [tan(tilt_y), -tan(tilt_x), 1]
"""
function coil_pose_from_points(x::Vector{Float64}, y::Vector{Float64}, z::Vector{Float64})
    mx, my, mz, w = segment_midpoints_lengths(x, y, z)
    W = sum(w)
    W <= 0 && error("No valid segments found")

    cx = sum(w .* mx) / W
    cy = sum(w .* my) / W
    cz = sum(w .* mz) / W

    P = hcat(mx .- cx, my .- cy, mz .- cz)
    Pw = P .* sqrt.(w)
    _, _, V = svd(Pw)
    n = Vector(V[:,3])
    n[3] < 0 && (n = -n)

    tilt_x = atand(-n[2], n[3])
    tilt_y = atand( n[1], n[3])

    e1, e2 = plane_basis_from_normal(n)
    dx = mx .- cx
    dy = my .- cy
    dz = mz .- cz
    u = dx .* e1[1] .+ dy .* e1[2] .+ dz .* e1[3]
    v = dx .* e2[1] .+ dy .* e2[2] .+ dz .* e2[3]
    ρ = sqrt.(u.^2 .+ v.^2)
    rfit = sum(w .* ρ) / W

    cR   = sqrt(cx^2 + cy^2)
    cphi = rad2deg(atan(cy, cx))

    return (
        x0 = cx, y0 = cy, z0 = cz,
        R = cR, phi = cphi,
        tilt_x = tilt_x, tilt_y = tilt_y,
        normal = n,
        Rfit = rfit,
        weights = w,
        mx = mx, my = my, mz = mz,
    )
end

# ═══════════════════════════════════════════════════════════════
# Main transform function
# ═══════════════════════════════════════════════════════════════

"""
    transform_coil(input_file, output_file; ...)

Read a coil file, apply shift and tilt, and write the result.

Rotation is applied about the SAME weighted pose center used by
coil_centering_full.jl.
"""
function transform_coil(input_file::String, output_file::String;
                        shift_x::Float64=0.0, shift_y::Float64=0.0, shift_z::Float64=0.0,
                        tilt_x_deg::Float64=0.0, tilt_y_deg::Float64=0.0, tilt_z_deg::Float64=0.0,
                        verbose::Bool=true)

    if verbose
        println("═══════════════════════════════════════════════════════")
        println(" Coil Transformation Utility")
        println("═══════════════════════════════════════════════════════")
        println("  Input:  $input_file")
        println("  Output: $output_file")
    end

    header_line, x, y, z = read_coil_file(input_file)
    N = length(x)

    if verbose
        println("\n  Read $N conductor points")
        println("  Header: \"$(strip(header_line))\"")
    end

    # ── Original pose using same definition as coil_centering_full.jl ─────
    pose = coil_pose_from_points(x, y, z)
    cx, cy, cz = pose.x0, pose.y0, pose.z0

    if verbose
        println("\n  ── Original coil pose ──")
        println("  Cartesian center:   ($(@sprintf("%.4f", cx)), $(@sprintf("%.4f", cy)), $(@sprintf("%.4f", cz))) m")
        println("  Cylindrical center: R=$(@sprintf("%.5f", pose.R)) m, φ=$(@sprintf("%.3f", pose.phi))°, Z=$(@sprintf("%.5f", cz)) m")
        println("  Tilt:              θx=$(@sprintf("%.6f", pose.tilt_x))°, θy=$(@sprintf("%.6f", pose.tilt_y))°")
        println("  Fit radius:        $(@sprintf("%.5f", pose.Rfit)) m")

        R_all = sqrt.(x.^2 .+ y.^2)
        println("  R range: [$(@sprintf("%.4f", minimum(R_all))), $(@sprintf("%.4f", maximum(R_all)))] m")
        println("  Z range: [$(@sprintf("%.4f", minimum(z))), $(@sprintf("%.4f", maximum(z)))] m")
    end

    if verbose
        println("\n  ── Requested transformation ──")
        println("  Shift:  ΔX=$(@sprintf("%.4f", shift_x*1000)) mm, " *
                        "ΔY=$(@sprintf("%.4f", shift_y*1000)) mm, " *
                        "ΔZ=$(@sprintf("%.4f", shift_z*1000)) mm")
        println("  Tilt:   θx=$(@sprintf("%.4f", tilt_x_deg))°, " *
                        "θy=$(@sprintf("%.4f", tilt_y_deg))°, " *
                        "θz=$(@sprintf("%.4f", tilt_z_deg))°")
    end

    R_mat = build_rotation_matrix(tilt_x_deg, tilt_y_deg, tilt_z_deg)

    has_rotation = !(tilt_x_deg == 0.0 && tilt_y_deg == 0.0 && tilt_z_deg == 0.0)
    has_shift    = !(shift_x == 0.0 && shift_y == 0.0 && shift_z == 0.0)

    if verbose && has_rotation
        println("\n  Rotation matrix:")
        for row in 1:3
            println("    [$(@sprintf("%12.8f", R_mat[row,1]))  " *
                    "$(@sprintf("%12.8f", R_mat[row,2]))  " *
                    "$(@sprintf("%12.8f", R_mat[row,3]))]")
        end
    end

    # ── Apply transformation about weighted pose center ───────────────────
    x_new = copy(x)
    y_new = copy(y)
    z_new = copy(z)

    if has_rotation
        x_new .-= cx
        y_new .-= cy
        z_new .-= cz

        for i in 1:N
            p = R_mat * [x_new[i], y_new[i], z_new[i]]
            x_new[i] = p[1]
            y_new[i] = p[2]
            z_new[i] = p[3]
        end

        x_new .+= cx
        y_new .+= cy
        z_new .+= cz
    end

    if has_shift
        x_new .+= shift_x
        y_new .+= shift_y
        z_new .+= shift_z
    end

    # ── New pose with same definition ──────────────────────────────────────
    pose_new = coil_pose_from_points(x_new, y_new, z_new)
    cx_new, cy_new, cz_new = pose_new.x0, pose_new.y0, pose_new.z0

    if verbose
        println("\n  ── Transformed coil pose ──")
        println("  Cartesian center:   ($(@sprintf("%.4f", cx_new)), $(@sprintf("%.4f", cy_new)), $(@sprintf("%.4f", cz_new))) m")
        println("  Cylindrical center: R=$(@sprintf("%.5f", pose_new.R)) m, φ=$(@sprintf("%.3f", pose_new.phi))°, Z=$(@sprintf("%.5f", cz_new)) m")
        println("  Tilt:              θx=$(@sprintf("%.6f", pose_new.tilt_x))°, θy=$(@sprintf("%.6f", pose_new.tilt_y))°")
        println("  Fit radius:        $(@sprintf("%.5f", pose_new.Rfit)) m")

        R_all_new = sqrt.(x_new.^2 .+ y_new.^2)
        println("  R range: [$(@sprintf("%.4f", minimum(R_all_new))), $(@sprintf("%.4f", maximum(R_all_new)))] m")
        println("  Z range: [$(@sprintf("%.4f", minimum(z_new))), $(@sprintf("%.4f", maximum(z_new)))] m")

        dcx = (cx_new - cx) * 1000
        dcy = (cy_new - cy) * 1000
        dcz = (cz_new - cz) * 1000
        dc_total = sqrt(dcx^2 + dcy^2 + dcz^2)

        println("\n  ── Pose-center displacement ──")
        println("  ΔX=$(@sprintf("%.4f", dcx)) mm, ΔY=$(@sprintf("%.4f", dcy)) mm, ΔZ=$(@sprintf("%.4f", dcz)) mm")
        println("  |Δ|=$(@sprintf("%.4f", dc_total)) mm")

        if has_rotation
            n_orig = pose.normal
            n_new  = pose_new.normal
            angle_between = rad2deg(acos(clamp(dot(n_orig, n_new), -1.0, 1.0)))

            println("\n  ── Normal vector verification ──")
            println("  Original normal:    [$(@sprintf("%.6f", n_orig[1])), $(@sprintf("%.6f", n_orig[2])), $(@sprintf("%.6f", n_orig[3]))]")
            println("  Transformed normal: [$(@sprintf("%.6f", n_new[1])), $(@sprintf("%.6f", n_new[2])), $(@sprintf("%.6f", n_new[3]))]")
            println("  Angle between normals: $(@sprintf("%.6f", angle_between))°")

            expected_angle = rad2deg(acos(clamp(
                cos(deg2rad(tilt_x_deg)) * cos(deg2rad(tilt_y_deg)), -1.0, 1.0)))
            println("  Expected tilt magnitude: ~$(@sprintf("%.6f", expected_angle))°")
        end
    end

    write_coil_file(output_file, header_line, x_new, y_new, z_new)

    if verbose
        println("\n  ✓ Transformed coil written to: $output_file")
        println("═══════════════════════════════════════════════════════")
    end

    return (;
        original_pose = pose,
        transformed_pose = pose_new,
        n_points = N,
    )
end

# ═══════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════

function main()
    transform_coil(
        INPUT_COIL_FILE, OUTPUT_COIL_FILE;
        shift_x    = Float64(SHIFT_X),
        shift_y    = Float64(SHIFT_Y),
        shift_z    = Float64(SHIFT_Z),
        tilt_x_deg = Float64(TILT_X_DEG),
        tilt_y_deg = Float64(TILT_Y_DEG),
        tilt_z_deg = Float64(TILT_Z_DEG),
        verbose    = VERBOSE,
    )
end

main()