# synth_hall_cyl.jl — cylindrical-coords synthetic Hall-probe generator.
#
# Given a coil .dat file, a 5- or 6-DOF rigid pose (shift in (x,y,z), tilt
# about (x,y,z) in degrees), and Hall probe coordinates expressed in
# cylindrical (R, φ, Z), compute the field components (B_R, B_φ, B_Z) at each
# probe using the GPEC Biot-Savart kernel.
#
# Two entry points:
#   synth_hall_cyl(coil_file, dx, dy, dz; ...)
#       -> NamedTuple (R, phi, Z, B_R, B_phi, B_Z, applied_pose)
#   synth_hall_cyl_to_file(output_path, coil_file, dx, dy, dz; ...)
#       -> writes .csv (or .h5) and returns the same NamedTuple

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GeneralizedPerturbedEquilibrium
using LinearAlgebra
using Statistics
using Printf
using Dates
using Random

const FT = GeneralizedPerturbedEquilibrium.ForcingTerms
const compute_biot_savart_boundary! = FT.compute_biot_savart_boundary!
const read_coil_dat = FT.read_coil_dat

# ───────────────────────────────────────────────────────────────────────────
# Geometric helpers
# ───────────────────────────────────────────────────────────────────────────

"""
    coil_centroid(coils) -> (cx, cy, cz)

Mean of every segment endpoint across the coil set.
"""
function coil_centroid(coils)
    sx, sy, sz, n = 0.0, 0.0, 0.0, 0
    for cs in coils, j in 1:cs.ncoil, k in 1:cs.s, l in 1:cs.nsec
        sx += cs.x[j, k, l]; sy += cs.y[j, k, l]; sz += cs.z[j, k, l]; n += 1
    end
    return sx / n, sy / n, sz / n
end

"""
    rotation_xyz(tx_deg, ty_deg, tz_deg) -> 3x3 matrix

Rotation matrix `Rz · Ry · Rx`, with each factor a right-handed rotation
about the corresponding axis. All inputs in degrees.
"""
function rotation_xyz(tx_deg, ty_deg, tz_deg)
    tx = deg2rad(tx_deg); ty = deg2rad(ty_deg); tz = deg2rad(tz_deg)
    Rx = [1.0 0.0 0.0; 0.0 cos(tx) -sin(tx); 0.0 sin(tx) cos(tx)]
    Ry = [cos(ty) 0.0 sin(ty); 0.0 1.0 0.0; -sin(ty) 0.0 cos(ty)]
    Rz = [cos(tz) -sin(tz) 0.0; sin(tz) cos(tz) 0.0; 0.0 0.0 1.0]
    return Rz * Ry * Rx
end

"""
    place_coil!(coils, dx, dy, dz; tx_deg=0.0, ty_deg=0.0, tz_deg=0.0)

Rigid transform of every segment in place: rotate about the coil's
geometric centroid by the requested tilt, then translate by (dx,dy,dz)
in metres. Default tilts make this a pure translation (5-DOF when
`tz_deg=0`; full 6-DOF when supplied).
"""
function place_coil!(coils, dx, dy, dz; tx_deg=0.0, ty_deg=0.0, tz_deg=0.0)
    cx, cy, cz = coil_centroid(coils)
    R = rotation_xyz(tx_deg, ty_deg, tz_deg)
    for cs in coils, j in 1:cs.ncoil, k in 1:cs.s, l in 1:cs.nsec
        p = (cs.x[j, k, l] - cx, cs.y[j, k, l] - cy, cs.z[j, k, l] - cz)
        qx = R[1, 1]*p[1] + R[1, 2]*p[2] + R[1, 3]*p[3]
        qy = R[2, 1]*p[1] + R[2, 2]*p[2] + R[2, 3]*p[3]
        qz = R[3, 1]*p[1] + R[3, 2]*p[2] + R[3, 3]*p[3]
        cs.x[j, k, l] = qx + cx + dx
        cs.y[j, k, l] = qy + cy + dy
        cs.z[j, k, l] = qz + cz + dz
    end
    return coils
end

# ───────────────────────────────────────────────────────────────────────────
# Probe-grid convenience
# ───────────────────────────────────────────────────────────────────────────

"""
    dual_shell_grid(R_inner, R_outer, Z_center, half_inner, half_outer;
                    nphi_inner=24, nz_inner=22, nphi_outer=20, nz_outer=10)

Build flat `(R, phi, Z)` vectors for two concentric cylindrical shells of
Hall probes centred at `Z_center`. Returns three `Vector{Float64}` of equal
length.
"""
function dual_shell_grid(R_inner, R_outer, Z_center, half_inner, half_outer;
                         nphi_inner::Int=24, nz_inner::Int=22,
                         nphi_outer::Int=20, nz_outer::Int=10)
    φi = collect(range(0, 2π; length=nphi_inner + 1)[1:end-1])
    φo = collect(range(0, 2π; length=nphi_outer + 1)[1:end-1])
    Zi = collect(range(Z_center - half_inner, Z_center + half_inner; length=nz_inner))
    Zo = collect(range(Z_center - half_outer, Z_center + half_outer; length=nz_outer))
    R = Float64[]; phi = Float64[]; Z = Float64[]
    for z in Zi, p in φi
        push!(R, R_inner); push!(phi, p); push!(Z, z)
    end
    for z in Zo, p in φo
        push!(R, R_outer); push!(phi, p); push!(Z, z)
    end
    return R, phi, Z
end

# ───────────────────────────────────────────────────────────────────────────
# Main entry points
# ───────────────────────────────────────────────────────────────────────────

"""
    synth_hall_cyl(coil_file, dx, dy, dz;
                   tx_deg=0.0, ty_deg=0.0, tz_deg=0.0,
                   current_A=100.0,
                   R_probes, phi_probes, Z_probes,
                   noise_floor_T=0.0, noise_rel_frac=0.0,
                   noise_seed=nothing)
        -> (R, phi, Z, B_R, B_phi, B_Z, applied_pose, noise, ...)

Load `coil_file`, apply the rigid pose, set current to `current_A`, and
compute cylindrical field components at each probe. Probes are cylindrical
coordinates supplied as three equal-length flat vectors.

Noise (applied independently to each of `B_R, B_phi, B_Z` at every probe):

    σ_i = sqrt(noise_floor_T^2 + (noise_rel_frac * |B_clean|_i)^2)

`noise_floor_T` is an absolute Gaussian noise floor in Tesla.
`noise_rel_frac` is a fractional gain/calibration error scaled by the local
clean |B|. Pass `noise_seed::Int` for reproducible noise (the function
seeds a local `MersenneTwister` rather than the global RNG).
"""
function synth_hall_cyl(coil_file::AbstractString, dx, dy, dz;
        tx_deg::Real=0.0, ty_deg::Real=0.0, tz_deg::Real=0.0,
        current_A::Real=100.0,
        R_probes::AbstractVector, phi_probes::AbstractVector,
        Z_probes::AbstractVector,
        noise_floor_T::Real=0.0, noise_rel_frac::Real=0.0,
        noise_seed::Union{Integer,Nothing}=nothing)
    @assert length(R_probes) == length(phi_probes) == length(Z_probes) "R_probes, phi_probes, Z_probes must have equal length"
    cs = read_coil_dat(coil_file)
    cs.currents .= current_A
    coils = [cs]
    place_coil!(coils, dx, dy, dz; tx_deg=tx_deg, ty_deg=ty_deg, tz_deg=tz_deg)

    N = length(R_probes)
    BR = zeros(N); BP = zeros(N); BZ = zeros(N)
    R  = collect(Float64, R_probes)
    P  = collect(Float64, phi_probes)
    Z  = collect(Float64, Z_probes)
    compute_biot_savart_boundary!(BR, BP, BZ, R, P, Z, coils)

    if noise_floor_T > 0 || noise_rel_frac > 0
        rng = noise_seed === nothing ? MersenneTwister() : MersenneTwister(noise_seed)
        floor2 = Float64(noise_floor_T)^2
        rel    = Float64(noise_rel_frac)
        @inbounds for i in 1:N
            Bmag = sqrt(BR[i]^2 + BP[i]^2 + BZ[i]^2)
            σ = sqrt(floor2 + (rel * Bmag)^2)
            BR[i] += randn(rng) * σ
            BP[i] += randn(rng) * σ
            BZ[i] += randn(rng) * σ
        end
    end

    return (R=R, phi=P, Z=Z, B_R=BR, B_phi=BP, B_Z=BZ,
            applied_pose=(dx=dx, dy=dy, dz=dz,
                          tx_deg=tx_deg, ty_deg=ty_deg, tz_deg=tz_deg),
            noise=(floor_T=Float64(noise_floor_T),
                   rel_frac=Float64(noise_rel_frac),
                   seed=noise_seed),
            current_A=Float64(current_A), coil_file=String(coil_file))
end

"""
    synth_hall_cyl_to_file(output_path, coil_file, dx, dy, dz; kwargs...)

Externally-callable wrapper: runs `synth_hall_cyl(...)` and writes the
result to `output_path`. Format is selected from the path extension:

- `.csv` (default): comment header recording the pose + coil file + timestamp,
  then `R,phi,Z,x,y,B_R,B_phi,B_Z,B_mag` rows.
- `.h5`: HDF5 file with datasets `R, phi, Z, B_R, B_phi, B_Z` and a
  `metadata` group recording the pose. Requires HDF5 (already in Project.toml).

Returns the same NamedTuple as `synth_hall_cyl`.
"""
function synth_hall_cyl_to_file(output_path::AbstractString, coil_file::AbstractString,
        dx, dy, dz; kwargs...)
    out = synth_hall_cyl(coil_file, dx, dy, dz; kwargs...)
    ext = lowercase(splitext(output_path)[2])
    if ext == ".h5" || ext == ".hdf5"
        _write_h5(output_path, out)
    else
        _write_csv(output_path, out)
    end
    println("Wrote ", output_path, "  (", length(out.R), " probes)")
    return out
end

function _write_csv(path::AbstractString, out)
    mkpath(dirname(abspath(path)))
    p = out.applied_pose
    n = out.noise
    open(path, "w") do io
        println(io, "# synth_hall_cyl output")
        println(io, "# generated: ", Dates.now())
        println(io, "# coil_file: ", out.coil_file)
        println(io, "# current_A: ", out.current_A)
        @printf(io, "# pose: dx=%.6e m, dy=%.6e m, dz=%.6e m, ", p.dx, p.dy, p.dz)
        @printf(io, "tilt_x=%.6e deg, tilt_y=%.6e deg, tilt_z=%.6e deg\n",
                p.tx_deg, p.ty_deg, p.tz_deg)
        @printf(io, "# noise: floor_T=%.6e, rel_frac=%.6e, seed=%s\n",
                n.floor_T, n.rel_frac, n.seed === nothing ? "none" : string(n.seed))
        println(io, "R,phi,Z,x,y,B_R,B_phi,B_Z,B_mag")
        for i in eachindex(out.R)
            x = out.R[i] * cos(out.phi[i])
            y = out.R[i] * sin(out.phi[i])
            Bm = sqrt(out.B_R[i]^2 + out.B_phi[i]^2 + out.B_Z[i]^2)
            @printf(io, "%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n",
                    out.R[i], out.phi[i], out.Z[i], x, y,
                    out.B_R[i], out.B_phi[i], out.B_Z[i], Bm)
        end
    end
end

function _write_h5(path::AbstractString, out)
    @eval Main using HDF5
    mkpath(dirname(abspath(path)))
    HDF5 = Main.HDF5
    HDF5.h5open(path, "w") do f
        for (name, arr) in (("R", out.R), ("phi", out.phi), ("Z", out.Z),
                            ("B_R", out.B_R), ("B_phi", out.B_phi), ("B_Z", out.B_Z))
            f[name] = arr
        end
        p = out.applied_pose
        n = out.noise
        g = HDF5.create_group(f, "metadata")
        g["coil_file"]      = out.coil_file
        g["current_A"]      = out.current_A
        g["dx"]             = Float64(p.dx)
        g["dy"]             = Float64(p.dy)
        g["dz"]             = Float64(p.dz)
        g["tx_deg"]         = Float64(p.tx_deg)
        g["ty_deg"]         = Float64(p.ty_deg)
        g["tz_deg"]         = Float64(p.tz_deg)
        g["noise_floor_T"]  = n.floor_T
        g["noise_rel_frac"] = n.rel_frac
        g["noise_seed"]     = n.seed === nothing ? "none" : string(n.seed)
        g["generated"]      = string(Dates.now())
    end
end

# ───────────────────────────────────────────────────────────────────────────
# Demo
# ───────────────────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    coil_file = joinpath(@__DIR__, "..", "examples", "Br_3D_example", "sparc_pf1u.dat")
    R_probes, phi_probes, Z_probes = dual_shell_grid(0.5, 0.8, 2.31, 0.6, 0.4;
        nphi_inner=24, nz_inner=14, nphi_outer=16, nz_outer=8)
    out = synth_hall_cyl_to_file(
        joinpath(@__DIR__, "csvs", "synth_hall_demo.csv"),
        coil_file,
        0.005, -0.002, 0.001;
        tx_deg=0.30, ty_deg=-0.10,
        current_A=100.0,
        R_probes=R_probes, phi_probes=phi_probes, Z_probes=Z_probes,
    )
    @printf("|B_R|   median: %.3e T,  max: %.3e T\n",
            median(abs.(out.B_R)),  maximum(abs.(out.B_R)))
    @printf("|B_phi| median: %.3e T,  max: %.3e T\n",
            median(abs.(out.B_phi)), maximum(abs.(out.B_phi)))
    @printf("|B_Z|   median: %.3e T,  max: %.3e T\n",
            median(abs.(out.B_Z)),  maximum(abs.(out.B_Z)))
end
