# tilt_shift_notebook.jl
# notebook-safe version, identical math/logic, only fixes mutability + cell usage

using LinearAlgebra
using Printf
using Statistics

# ───────────────────────────────────────────────────────────────
# Rotation matrices
# ───────────────────────────────────────────────────────────────

rotation_x(θ) = [1.0 0 0; 0 cos(θ) -sin(θ); 0 sin(θ) cos(θ)]
rotation_y(θ) = [cos(θ) 0 sin(θ); 0 1 0; -sin(θ) 0 cos(θ)]
rotation_z(θ) = [cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1]

build_rotation_matrix(tx, ty, tz) =
    rotation_z(deg2rad(tz)) * rotation_y(deg2rad(ty)) * rotation_x(deg2rad(tx))

# ───────────────────────────────────────────────────────────────
# I/O
# ───────────────────────────────────────────────────────────────

function read_coil_file(f)
    lines = readlines(f)
    h = lines[1]
    x = Float64[]
    y = Float64[]
    z = Float64[]

    for l in lines[2:end]
        t = split(strip(l))
        length(t) < 3 && continue
        push!(x, parse(Float64, t[1]))
        push!(y, parse(Float64, t[2]))
        push!(z, parse(Float64, t[3]))
    end

    return h, x, y, z
end

function write_coil_file(f, h, x, y, z)
    open(f, "w") do io
        println(io, h)
        for i in eachindex(x)
            @printf(io, "  %14.4e   %14.4e   %14.4e\n", x[i], y[i], z[i])
        end
    end
end

# ───────────────────────────────────────────────────────────────
# Geometry (same logic, fixed mutability bug only)
# ───────────────────────────────────────────────────────────────

is_closed(x, y, z) =
    (x[1]-x[end])^2 + (y[1]-y[end])^2 + (z[1]-z[end])^2 < 1e-24

function seg_mid_len(x, y, z)
    N = length(x)
    last = is_closed(x, y, z) ? N-1 : N

    mx = Float64[]
    my = Float64[]
    mz = Float64[]
    w  = Float64[]

    for i in 1:last
        j = i == N ? 1 : i + 1

        dx = x[j] - x[i]
        dy = y[j] - y[i]
        dz = z[j] - z[i]

        ds = sqrt(dx^2 + dy^2 + dz^2)
        ds == 0 && continue

        push!(mx, (x[i] + x[j]) / 2)
        push!(my, (y[i] + y[j]) / 2)
        push!(mz, (z[i] + z[j]) / 2)
        push!(w, ds)
    end

    return mx, my, mz, w
end

function coil_pose(x, y, z)
    mx, my, mz, w = seg_mid_len(x, y, z)
    W = sum(w)

    cx = sum(w .* mx) / W
    cy = sum(w .* my) / W
    cz = sum(w .* mz) / W

    P = hcat(mx .- cx, my .- cy, mz .- cz)
    _, _, V = svd(P .* sqrt.(w))

    n = V[:, 3]
    n[3] < 0 && (n = -n)

    return (x0 = cx, y0 = cy, z0 = cz, normal = n)
end

# ───────────────────────────────────────────────────────────────
# Core transform (unchanged math)
# ───────────────────────────────────────────────────────────────

function transform_coil(input_file, output_file;
    shift_x = 0.0, shift_y = 0.0, shift_z = 0.0,
    tilt_x_deg = 0.0, tilt_y_deg = 0.0, tilt_z_deg = 0.0)

    h, x, y, z = read_coil_file(input_file)

    p = coil_pose(x, y, z)
    cx, cy, cz = p.x0, p.y0, p.z0

    R = build_rotation_matrix(tilt_x_deg, tilt_y_deg, tilt_z_deg)

    x2 = copy(x)
    y2 = copy(y)
    z2 = copy(z)

    # center
    x2 .-= cx
    y2 .-= cy
    z2 .-= cz

    # rotate
    for i in eachindex(x2)
        v = R * [x2[i], y2[i], z2[i]]
        x2[i] = v[1]
        y2[i] = v[2]
        z2[i] = v[3]
    end

    # translate back
    x2 .+= cx + shift_x
    y2 .+= cy + shift_y
    z2 .+= cz + shift_z

    write_coil_file(output_file, h, x2, y2, z2)

    return x2, y2, z2
end

# ───────────────────────────────────────────────────────────────
# Notebook-friendly wrapper (no side effects at import time)
# ───────────────────────────────────────────────────────────────

function run_tilt_shift_notebook(;
    input_file  = "sparc_pf1u.dat",
    output_file = "out.dat",
    shift_x = 0.003,
    shift_y = 0.005,
    shift_z = 0.007,
    tilt_x_deg = 0.0,
    tilt_y_deg = 0.0,
    tilt_z_deg = 0.0
)

    return transform_coil(
        input_file,
        output_file;
        shift_x = shift_x,
        shift_y = shift_y,
        shift_z = shift_z,
        tilt_x_deg = tilt_x_deg,
        tilt_y_deg = tilt_y_deg,
        tilt_z_deg = tilt_z_deg,
    )
end