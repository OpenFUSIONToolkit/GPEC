#!/usr/bin/env julia
# ═══════════════════════════════════════════════════════════════
#
# Reads a coil .dat file, applies user-specified translations
# and rotations (tilts), and writes a new .dat file.
#
# The transformations are applied about the coil's geometric
# centroid, so a pure tilt rotates the coil in place.
#
# Order of operations:
#   1. Compute centroid
#   2. Center the coil (subtract centroid)
#   3. Apply rotation (tilt_x, tilt_y, tilt_z) — extrinsic XYZ Euler angles
#   4. Un-center (add centroid back)
#   5. Apply translation (shift_x, shift_y, shift_z)
#
# Usage:
#   julia transform_coil.jl
#   (edit the CONFIGURATION section below, or import and call transform_coil())
# ═══════════════════════════════════════════════════════════════

using LinearAlgebra
using Printf
using Statistics

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — edit these values
# ═══════════════════════════════════════════════════════════════

# Input / output files
INPUT_COIL_FILE  = joinpath(@__DIR__, "sparc_pf1u.dat")
OUTPUT_COIL_FILE = joinpath(@__DIR__, "sparc_pf1u_200_xy29T.dat")

# ── Translation (applied AFTER rotation) ─────────────────────
# Units: meters
SHIFT_X = .002#0.001    # 1 mm shift in X
SHIFT_Y = .00#0.000    # no shift in Y
SHIFT_Z = .00#0.002    # 2 mm shift in Z

# ── Rotation / tilt (applied about the coil centroid) ────────
# Units: degrees
# These are extrinsic rotations applied in order: Rx then Ry then Rz
#   TILT_X = rotation about the X-axis (pitches the coil in the Y-Z plane)
#   TILT_Y = rotation about the Y-axis (pitches the coil in the X-Z plane)
#   TILT_Z = rotation about the Z-axis (rotates the coil in the X-Y plane)
TILT_X_DEG = .2 # 0.1    # 0.1° tilt about X
TILT_Y_DEG = .9 # 0.0    # no tilt about Y
TILT_Z_DEG = 0 # 0.0    # no tilt about Z

# EQUIVALENTLY: R, Z, Phi 

# ── Verbose output ───────────────────────────────────────────
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
Applied as R = Rz * Ry * Rx  (i.e., Rx first, then Ry, then Rz).
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

"""
    read_coil_file(filepath)

Read a coil .dat file.  Returns:
  - header_line::String   (the first line, e.g. "    1    1 30600    1.00")
  - x, y, z :: Vector{Float64}  (the conductor point coordinates)
"""
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

        # Try to parse as 3 floats
        tokens = split(stripped)
        if length(tokens) >= 3
            try
                xv = parse(Float64, tokens[1])
                yv = parse(Float64, tokens[2])
                zv = parse(Float64, tokens[3])
                push!(x, xv)
                push!(y, yv)
                push!(z, zv)
            catch
                # Skip lines that don't parse (e.g. additional headers)
                if VERBOSE
                    println("  Skipping unparseable line $i: \"$stripped\"")
                end
            end
        end
    end

    return header_line, x, y, z
end

"""
    write_coil_file(filepath, header_line, x, y, z)

Write a coil .dat file with the same format as the input.
"""
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
# Main transform function
# ═══════════════════════════════════════════════════════════════

"""
    transform_coil(input_file, output_file;
                    shift_x=0.0, shift_y=0.0, shift_z=0.0,
                    tilt_x_deg=0.0, tilt_y_deg=0.0, tilt_z_deg=0.0,
                    verbose=true)

Read a coil file, apply shift and tilt, write the result.

Tilt is applied about the coil's geometric centroid.
Shift is applied after tilt.
"""
function transform_coil(input_file::String, output_file::String;
                         shift_x::Float64=0.0, shift_y::Float64=0.0, shift_z::Float64=0.0,
                         tilt_x_deg::Float64=0.0, tilt_y_deg::Float64=0.0, tilt_z_deg::Float64=0.0,
                         verbose::Bool=true)

    # ── Read ─────────────────────────────────────────────
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

    # ── Compute centroid ─────────────────────────────────
    cx = mean(x)
    cy = mean(y)
    cz = mean(z)

    cR   = sqrt(cx^2 + cy^2)
    cphi = rad2deg(atan(cy, cx))

    if verbose
        println("\n  ── Original coil centroid ──")
        println("  Cartesian:   ($(@sprintf("%.4f", cx)), $(@sprintf("%.4f", cy)), $(@sprintf("%.4f", cz))) m")
        println("  Cylindrical: R=$(@sprintf("%.5f", cR)) m, φ=$(@sprintf("%.3f", cphi))°, Z=$(@sprintf("%.5f", cz)) m")

        R_all = sqrt.(x.^2 .+ y.^2)
        println("  R range: [$(@sprintf("%.4f", minimum(R_all))), $(@sprintf("%.4f", maximum(R_all)))] m")
        println("  Z range: [$(@sprintf("%.4f", minimum(z))), $(@sprintf("%.4f", maximum(z)))] m")
    end

    # ── Requested transformation ─────────────────────────
    if verbose
        println("\n  ── Requested transformation ──")
        println("  Shift:  ΔX=$(@sprintf("%.4f", shift_x*1000)) mm, " *
                         "ΔY=$(@sprintf("%.4f", shift_y*1000)) mm, " *
                         "ΔZ=$(@sprintf("%.4f", shift_z*1000)) mm")
        println("  Tilt:   θx=$(@sprintf("%.4f", tilt_x_deg))°, " *
                         "θy=$(@sprintf("%.4f", tilt_y_deg))°, " *
                         "θz=$(@sprintf("%.4f", tilt_z_deg))°")
    end

    # ── Build rotation matrix ────────────────────────────
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

    # ── Apply transformation ─────────────────────────────
    x_new = copy(x)
    y_new = copy(y)
    z_new = copy(z)

    if has_rotation
        # Step 1: Center about centroid
        x_new .-= cx
        y_new .-= cy
        z_new .-= cz

        # Step 2: Rotate
        for i in 1:N
            p = R_mat * [x_new[i], y_new[i], z_new[i]]
            x_new[i] = p[1]
            y_new[i] = p[2]
            z_new[i] = p[3]
        end

        # Step 3: Un-center
        x_new .+= cx
        y_new .+= cy
        z_new .+= cz
    end

    if has_shift
        # Step 4: Translate
        x_new .+= shift_x
        y_new .+= shift_y
        z_new .+= shift_z
    end

    # ── Report new centroid ──────────────────────────────
    cx_new = mean(x_new)
    cy_new = mean(y_new)
    cz_new = mean(z_new)

    cR_new   = sqrt(cx_new^2 + cy_new^2)
    cphi_new = rad2deg(atan(cy_new, cx_new))

    if verbose
        println("\n  ── Transformed coil centroid ──")
        println("  Cartesian:   ($(@sprintf("%.4f", cx_new)), $(@sprintf("%.4f", cy_new)), $(@sprintf("%.4f", cz_new))) m")
        println("  Cylindrical: R=$(@sprintf("%.5f", cR_new)) m, φ=$(@sprintf("%.3f", cphi_new))°, Z=$(@sprintf("%.5f", cz_new)) m")

        R_all_new = sqrt.(x_new.^2 .+ y_new.^2)
        println("  R range: [$(@sprintf("%.4f", minimum(R_all_new))), $(@sprintf("%.4f", maximum(R_all_new)))] m")
        println("  Z range: [$(@sprintf("%.4f", minimum(z_new))), $(@sprintf("%.4f", maximum(z_new)))] m")

        # Centroid shift
        dcx = (cx_new - cx) * 1000
        dcy = (cy_new - cy) * 1000
        dcz = (cz_new - cz) * 1000
        dc_total = sqrt(dcx^2 + dcy^2 + dcz^2)
        println("\n  ── Centroid displacement ──")
        println("  ΔX=$(@sprintf("%.4f", dcx)) mm, ΔY=$(@sprintf("%.4f", dcy)) mm, ΔZ=$(@sprintf("%.4f", dcz)) mm")
        println("  |Δ|=$(@sprintf("%.4f", dc_total)) mm")

        # Verify rotation via SVD-based normal comparison
        if has_rotation
            # Original normal
            P_orig = hcat(x .- cx, y .- cy, z .- cz)
            _, _, V_orig = svd(P_orig)
            n_orig = V_orig[:, 3]
            if n_orig[3] < 0; n_orig = -n_orig; end

            # Transformed normal
            P_new = hcat(x_new .- cx_new, y_new .- cy_new, z_new .- cz_new)
            _, _, V_new = svd(P_new)
            n_new = V_new[:, 3]
            if n_new[3] < 0; n_new = -n_new; end

            angle_between = rad2deg(acos(clamp(dot(n_orig, n_new), -1.0, 1.0)))

            println("\n  ── Normal vector verification ──")
            println("  Original normal: [$(@sprintf("%.6f", n_orig[1])), $(@sprintf("%.6f", n_orig[2])), $(@sprintf("%.6f", n_orig[3]))]")
            println("  Transformed normal: [$(@sprintf("%.6f", n_new[1])), $(@sprintf("%.6f", n_new[2])), $(@sprintf("%.6f", n_new[3]))]")
            println("  Angle between normals: $(@sprintf("%.6f", angle_between))°")

            expected_angle = rad2deg(acos(clamp(
                cos(deg2rad(tilt_x_deg)) * cos(deg2rad(tilt_y_deg)), -1.0, 1.0)))
            println("  Expected tilt magnitude: ~$(@sprintf("%.6f", expected_angle))°")
        end
    end

    # ── Write ────────────────────────────────────────────
    write_coil_file(output_file, header_line, x_new, y_new, z_new)

    if verbose
        println("\n  ✓ Transformed coil written to: $output_file")
        println("═══════════════════════════════════════════════════════")
    end

    return (;
        original_centroid  = (x=cx, y=cy, z=cz, R=cR, phi=cphi),
        transformed_centroid = (x=cx_new, y=cy_new, z=cz_new, R=cR_new, phi=cphi_new),
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