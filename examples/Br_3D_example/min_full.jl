using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

module MinFull

using GeneralizedPerturbedEquilibrium
using LinearAlgebra
using Statistics
using Printf

const FT = GeneralizedPerturbedEquilibrium.ForcingTerms
const compute_biot_savart_boundary! = FT.compute_biot_savart_boundary!
const read_coil_dat = FT.read_coil_dat

export run_min_centering, fit_pose_to_hall, field_residual,
       axis_from_coil, load_coils, make_hall_data, place_coil_at_axis

# ═══════════════════════════════════════════════════════════════
# Coil geometry helpers (copied from coil_centering_full.jl)
# ═══════════════════════════════════════════════════════════════

function segment_midpoints_lengths(coils)
    mx = Float64[]; my = Float64[]; mz = Float64[]; w = Float64[]
    for cs in coils, j in 1:cs.ncoil, k in 1:cs.s, l in 1:(cs.nsec - 1)
        x1, y1, z1 = cs.x[j,k,l], cs.y[j,k,l], cs.z[j,k,l]
        x2, y2, z2 = cs.x[j,k,l+1], cs.y[j,k,l+1], cs.z[j,k,l+1]
        ds = sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
        ds <= 0 && continue
        push!(mx, 0.5*(x1+x2)); push!(my, 0.5*(y1+y2))
        push!(mz, 0.5*(z1+z2)); push!(w, ds)
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
    cx = sum(w .* mx) / W; cy = sum(w .* my) / W; cz = sum(w .* mz) / W
    P = hcat(mx .- cx, my .- cy, mz .- cz)
    _, _, V = svd(P .* sqrt.(w))
    n = Vector(V[:, 3])
    n[3] < 0 && (n = -n)
    return (cx=cx, cy=cy, cz=cz, n=n, mx=mx, my=my, mz=mz, w=w)
end

function collect_coil_xyz(coils)
    x = Float64[]; y = Float64[]; z = Float64[]
    for cs in coils, j in 1:cs.ncoil, k in 1:cs.s
        append!(x, vec(cs.x[j,k,:])); append!(y, vec(cs.y[j,k,:])); append!(z, vec(cs.z[j,k,:]))
    end
    return x, y, z
end

"""
    axis_from_coil(coils; verbose=true, label="Coil geometric axis")

SVD-based geometric pose of a coil: center (x0,y0,z0), tilt angles in deg about
machine x and y axes, fit radius in the coil plane, and R/Z extents.
"""
function axis_from_coil(coils; verbose::Bool=true, label::AbstractString="Coil geometric axis")
    pose = coil_pose_from_segments(coils)
    cx, cy, cz, n = pose.cx, pose.cy, pose.cz, pose.n
    tx = atand(-n[2], n[3]); ty = atand(n[1], n[3])
    e1, e2 = plane_basis_from_normal(n)
    dx = pose.mx .- cx; dy = pose.my .- cy; dz = pose.mz .- cz
    u = dx .* e1[1] .+ dy .* e1[2] .+ dz .* e1[3]
    v = dx .* e2[1] .+ dy .* e2[2] .+ dz .* e2[3]
    rfit = sum(pose.w .* sqrt.(u.^2 .+ v.^2)) / sum(pose.w)
    x, y, z = collect_coil_xyz(coils)
    R = sqrt.(x.^2 .+ y.^2)
    a = (x0=cx, y0=cy, z0=cz, tilt_x=tx, tilt_y=ty, tilt_mag=hypot(tx,ty),
         Rfit=rfit, Rmin=minimum(R), Rmax=maximum(R), Zmin=minimum(z), Zmax=maximum(z))
    if verbose
        println("\n$label:")
        @printf("  center = (%.3f, %.3f, %.3f) mm\n", a.x0*1e3, a.y0*1e3, a.z0*1e3)
        @printf("  tilt_x = %.6f deg, tilt_y = %.6f deg\n", a.tilt_x, a.tilt_y)
        @printf("  fit radius = %.3f mm\n", a.Rfit*1e3)
        @printf("  R range = [%.3f, %.3f] mm\n", a.Rmin*1e3, a.Rmax*1e3)
        @printf("  Z range = [%.3f, %.3f] mm\n", a.Zmin*1e3, a.Zmax*1e3)
    end
    return a
end

# ═══════════════════════════════════════════════════════════════
# Rotation and rigid coil placement
# ═══════════════════════════════════════════════════════════════

function rotation_between_vectors(a::Vector{Float64}, b::Vector{Float64})
    a = normalize(a); b = normalize(b)
    cosθ = clamp(dot(a, b), -1.0, 1.0)
    abs(cosθ - 1.0) < 1e-12 && return Matrix{Float64}(I, 3, 3)
    if abs(cosθ + 1.0) < 1e-10
        perp = abs(a[1]) < 0.9 ? [1.0,0.0,0.0] : [0.0,1.0,0.0]
        ax = normalize(cross(a, perp))
        return 2.0*(ax*ax') - Matrix{Float64}(I, 3, 3)
    end
    k = normalize(cross(a, b)); sinθ = sqrt(max(1.0 - cosθ^2, 0.0))
    K = [0.0 -k[3] k[2]; k[3] 0.0 -k[1]; -k[2] k[1] 0.0]
    return Matrix{Float64}(I, 3, 3) + sinθ*K + (1.0 - cosθ)*K*K
end

"""
    place_coil_at_axis(coils, x0, y0, z0, tx_deg, ty_deg)

Rigid shift+rotation of all coil segments so the coil plane normal points along
the axis defined by tilts (tx_deg, ty_deg) and the geometric center sits at
(x0, y0, z0). Returns a deep-copied coil set; the input is not modified.
"""
function place_coil_at_axis(coils, target_x0, target_y0, target_z0, target_tx_deg, target_ty_deg)
    coils_new = deepcopy(coils)
    pose = coil_pose_from_segments(coils_new)
    cx, cy, cz, n_orig = pose.cx, pose.cy, pose.cz, copy(pose.n)
    n_target = normalize([tand(target_ty_deg), -tand(target_tx_deg), 1.0])
    z_hat = [0.0, 0.0, 1.0]
    R_total = rotation_between_vectors(z_hat, n_target) * rotation_between_vectors(n_orig, z_hat)
    for cs in coils_new, j in 1:cs.ncoil, k in 1:cs.s, l in 1:cs.nsec
        p = [cs.x[j,k,l]-cx, cs.y[j,k,l]-cy, cs.z[j,k,l]-cz]
        q = R_total * p
        cs.x[j,k,l] = q[1]+target_x0; cs.y[j,k,l] = q[2]+target_y0; cs.z[j,k,l] = q[3]+target_z0
    end
    return coils_new
end

# ═══════════════════════════════════════════════════════════════
# Coil loading and synthetic Hall data
# ═══════════════════════════════════════════════════════════════

function load_coils(file::AbstractString; label::String="COIL", current_A::Float64=100.0, verbose::Bool=true)
    isfile(file) || error("Coil file not found for $label: $file")
    cs = read_coil_dat(file)
    cs.currents .= current_A
    if verbose
        println("\nLoaded $label coil:")
        println("  file = $file")
        println("  ncoil=$(cs.ncoil), s=$(cs.s), nsec=$(cs.nsec), I=$current_A A")
    end
    return [cs]
end

"""
    make_hall_data(source_coils, phys_baseline; kwargs...)

Build a dual-shell Hall probe grid sized to `phys_baseline` (a NamedTuple from
`axis_from_coil`), evaluate Biot-Savart from `source_coils` at every probe, and
return a NamedTuple containing probe coordinates and field components. Defaults
match `coil_centering_full.jl`'s `run_coil_axis_comparison`.
"""
function make_hall_data(source_coils, phys_baseline; label::AbstractString="FIELD",
        hall_r_inner_frac::Float64=0.40, hall_r_outer_frac::Float64=0.50,
        hall_n_phi_inner::Int=24, hall_n_z_inner::Int=22,
        hall_n_phi_outer::Int=20, hall_n_z_outer::Int=10,
        hall_z_halfspan_inner_m::Float64=0.60, hall_z_halfspan_outer_m::Float64=0.40,
        add_noise_frac::Float64=0.0,
        noise_floor_T::Float64=0.0, noise_rel_frac::Float64=0.0,
        verbose::Bool=true)

    R_inner = hall_r_inner_frac * phys_baseline.Rmin
    R_outer = hall_r_outer_frac * phys_baseline.Rmax
    Zc = phys_baseline.z0

    φi = collect(range(0, 2π, length=hall_n_phi_inner+1)[1:end-1])
    φo = collect(range(0, 2π, length=hall_n_phi_outer+1)[1:end-1])
    Zi = collect(range(Zc - hall_z_halfspan_inner_m, Zc + hall_z_halfspan_inner_m; length=hall_n_z_inner))
    Zo = collect(range(Zc - hall_z_halfspan_outer_m, Zc + hall_z_halfspan_outer_m; length=hall_n_z_outer))

    N = hall_n_phi_inner*hall_n_z_inner + hall_n_phi_outer*hall_n_z_outer
    R = zeros(N); φ = zeros(N); Z = zeros(N); shell = zeros(Int, N)

    k = 1
    for iz in eachindex(Zi), ip in eachindex(φi)
        R[k]=R_inner; φ[k]=φi[ip]; Z[k]=Zi[iz]; shell[k]=1; k+=1
    end
    for iz in eachindex(Zo), ip in eachindex(φo)
        R[k]=R_outer; φ[k]=φo[ip]; Z[k]=Zo[iz]; shell[k]=2; k+=1
    end

    BR=zeros(N); BP=zeros(N); BZ=zeros(N)
    compute_biot_savart_boundary!(BR, BP, BZ, R, φ, Z, source_coils)
    if add_noise_frac > 0
        σ = add_noise_frac * maximum(sqrt.(BR.^2 .+ BP.^2 .+ BZ.^2))
        BR .+= randn(N) .* σ; BP .+= randn(N) .* σ; BZ .+= randn(N) .* σ
    end
    # Per-probe noise: σ_i = sqrt(σ_floor² + (rel * |B_clean|_i)²), applied
    # independently to each of the 3 components. σ_floor sets a constant
    # Hall electronics noise (Tesla); rel sets the gain/calibration error.
    if noise_floor_T > 0 || noise_rel_frac > 0
        Bmag_clean = sqrt.(BR.^2 .+ BP.^2 .+ BZ.^2)
        @inbounds for i in 1:N
            σi = sqrt(noise_floor_T^2 + (noise_rel_frac * Bmag_clean[i])^2)
            BR[i] += randn() * σi
            BP[i] += randn() * σi
            BZ[i] += randn() * σi
        end
    end
    Bmag = sqrt.(BR.^2 .+ BP.^2 .+ BZ.^2)

    if verbose
        println("\nSynthetic Hall data ($label): $N probes")
        @printf("  shell 1: R = %.3f mm, Z = [%.3f, %.3f] mm, nphi=%d, nz=%d\n",
                R_inner*1e3, first(Zi)*1e3, last(Zi)*1e3, hall_n_phi_inner, hall_n_z_inner)
        @printf("  shell 2: R = %.3f mm, Z = [%.3f, %.3f] mm, nphi=%d, nz=%d\n",
                R_outer*1e3, first(Zo)*1e3, last(Zo)*1e3, hall_n_phi_outer, hall_n_z_outer)
    end

    return (label=label, R=R, phi=φ, Z=Z, x=R.*cos.(φ), y=R.*sin.(φ), shell=shell,
            B_R=BR, B_phi=BP, B_Z=BZ, B_mag=Bmag,
            Zi=Zi, Zo=Zo, phi_i=φi, phi_o=φo, nzi=hall_n_z_inner, nzo=hall_n_z_outer,
            R_inner=R_inner, R_outer=R_outer)
end

# ═══════════════════════════════════════════════════════════════
# Forward-model residual and 5D coordinate-descent fit
# ═══════════════════════════════════════════════════════════════

"""
    field_residual(coils, hall, buf)

Sum-of-squares mismatch between Biot-Savart from `coils` and the stored field
in `hall` (`B_R`, `B_phi`, `B_Z`), evaluated at the Hall probe coordinates.
`buf` is a NamedTuple of three preallocated `Vector{Float64}` of length
`length(hall.R)` used as scratch — the function zeros them before each call.
"""
function field_residual(coils, hall, buf)
    fill!(buf.BR, 0.0); fill!(buf.BP, 0.0); fill!(buf.BZ, 0.0)
    compute_biot_savart_boundary!(buf.BR, buf.BP, buf.BZ, hall.R, hall.phi, hall.Z, coils)
    s = 0.0
    @inbounds for i in eachindex(hall.R)
        s += (buf.BR[i] - hall.B_R[i])^2 +
             (buf.BP[i] - hall.B_phi[i])^2 +
             (buf.BZ[i] - hall.B_Z[i])^2
    end
    return s
end

"""
    fit_pose_to_hall(baseline_coils, baseline_geom, hall; kwargs...)

Coordinate-descent fit of a rigid pose `(x0, y0, z0, tilt_x_deg, tilt_y_deg)`
that minimizes the sum-of-squares mismatch between Biot-Savart from
`baseline_coils` placed at the candidate pose and the field stored in `hall`.

The initial guess is the baseline's own geometric pose. Each pass tries `±step`
in each of the 5 directions; any improvement is accepted greedily; when no
direction improves in a pass, all steps are halved. Iteration terminates when
both translation and tilt steps fall below their tolerances or `max_iter` is
reached.
"""
function fit_pose_to_hall(baseline_coils, baseline_geom, hall;
        step0_m::Float64=0.050, step0_deg::Float64=0.5,
        tol_m::Float64=1e-6, tol_deg::Float64=1e-4,
        max_iter::Int=200, verbose::Bool=true)

    N = length(hall.R)
    buf = (BR=zeros(N), BP=zeros(N), BZ=zeros(N))

    pose = [baseline_geom.x0, baseline_geom.y0, baseline_geom.z0,
            baseline_geom.tilt_x, baseline_geom.tilt_y]
    steps = [step0_m, step0_m, step0_m, step0_deg, step0_deg]

    best = field_residual(baseline_coils, hall, buf)
    n_eval = 1
    iter = 0
    converged = false

    while iter < max_iter
        translation_done = steps[1] <= tol_m && steps[2] <= tol_m && steps[3] <= tol_m
        tilt_done = steps[4] <= tol_deg && steps[5] <= tol_deg
        if translation_done && tilt_done
            converged = true
            break
        end

        improved = false
        for i in 1:5, sgn in (1.0, -1.0)
            trial = copy(pose); trial[i] += sgn * steps[i]
            cand = place_coil_at_axis(baseline_coils, trial[1], trial[2], trial[3], trial[4], trial[5])
            val = field_residual(cand, hall, buf)
            n_eval += 1
            if val < best
                pose, best = trial, val
                improved = true
            end
        end
        improved || (steps .*= 0.5)
        iter += 1
    end

    result = (x0=pose[1], y0=pose[2], z0=pose[3], tilt_x=pose[4], tilt_y=pose[5],
              tilt_mag=hypot(pose[4], pose[5]),
              objective=best, n_eval=n_eval, n_iter=iter, converged=converged)

    if verbose
        println("\nForward-model coordinate-descent fit:")
        @printf("  iterations  = %d (max %d)\n", iter, max_iter)
        @printf("  evaluations = %d\n", n_eval)
        @printf("  objective   = %.6e\n", best)
        @printf("  converged   = %s\n", converged)
        @printf("  pose: x = %.4f mm, y = %.4f mm, z = %.4f mm, tilt_x = %.6f deg, tilt_y = %.6f deg\n",
                pose[1]*1e3, pose[2]*1e3, pose[3]*1e3, pose[4], pose[5])
    end
    return result
end

# ═══════════════════════════════════════════════════════════════
# User-facing entry point
# ═══════════════════════════════════════════════════════════════

"""
    run_min_centering(; baseline_file, perturbed_file, coil_current_a=100.0,
                       hall_kwargs..., fit_kwargs..., verbose=true)

Minimal forward-model coil centering. Loads a baseline coil and a perturbed
coil from .dat files, generates synthetic Hall probes from the perturbed coil
on a grid sized to the baseline, then fits a rigid pose to the baseline coil
that reproduces the perturbed Hall field.

Returns a NamedTuple with `baseline_geom`, `perturbed_geom`, `hall`, and
`recovered_pose`.
"""
function run_min_centering(;
        baseline_file::AbstractString,
        perturbed_file::AbstractString,
        coil_current_a::Float64=100.0,

        hall_r_inner_frac::Float64=0.40, hall_r_outer_frac::Float64=0.50,
        hall_n_phi_inner::Int=24, hall_n_z_inner::Int=22,
        hall_n_phi_outer::Int=20, hall_n_z_outer::Int=10,
        hall_z_halfspan_inner_m::Float64=0.60, hall_z_halfspan_outer_m::Float64=0.40,
        add_noise_frac::Float64=0.0,

        step0_m::Float64=0.050, step0_deg::Float64=0.5,
        tol_m::Float64=1e-6, tol_deg::Float64=1e-4,
        max_iter::Int=200,

        verbose::Bool=true)

    baseline_coils  = load_coils(baseline_file;  label="BASELINE",  current_A=coil_current_a, verbose=verbose)
    perturbed_coils = load_coils(perturbed_file; label="PERTURBED", current_A=coil_current_a, verbose=verbose)

    baseline_geom  = axis_from_coil(baseline_coils;  verbose=verbose, label="Baseline geometric axis")
    perturbed_geom = axis_from_coil(perturbed_coils; verbose=verbose, label="Perturbed geometric axis")

    hall = make_hall_data(perturbed_coils, baseline_geom;
        label="PERTURBED-SYNTH",
        hall_r_inner_frac=hall_r_inner_frac, hall_r_outer_frac=hall_r_outer_frac,
        hall_n_phi_inner=hall_n_phi_inner, hall_n_z_inner=hall_n_z_inner,
        hall_n_phi_outer=hall_n_phi_outer, hall_n_z_outer=hall_n_z_outer,
        hall_z_halfspan_inner_m=hall_z_halfspan_inner_m,
        hall_z_halfspan_outer_m=hall_z_halfspan_outer_m,
        add_noise_frac=add_noise_frac, verbose=verbose)

    recovered = fit_pose_to_hall(baseline_coils, baseline_geom, hall;
        step0_m=step0_m, step0_deg=step0_deg,
        tol_m=tol_m, tol_deg=tol_deg,
        max_iter=max_iter, verbose=verbose)

    if verbose
        println("\n" * "─"^88)
        println("Ground-truth offset vs. recovered offset (recovered − baseline geometric pose)")
        println("─"^88)
        @printf("  %-10s %16s %16s %16s\n", "param", "ground truth", "recovered", "error")
        for (name, gt, rec) in (
                ("Δx [mm]",    (perturbed_geom.x0 - baseline_geom.x0)*1e3,  (recovered.x0 - baseline_geom.x0)*1e3),
                ("Δy [mm]",    (perturbed_geom.y0 - baseline_geom.y0)*1e3,  (recovered.y0 - baseline_geom.y0)*1e3),
                ("Δz [mm]",    (perturbed_geom.z0 - baseline_geom.z0)*1e3,  (recovered.z0 - baseline_geom.z0)*1e3),
                ("Δtx [deg]",  perturbed_geom.tilt_x - baseline_geom.tilt_x, recovered.tilt_x - baseline_geom.tilt_x),
                ("Δty [deg]",  perturbed_geom.tilt_y - baseline_geom.tilt_y, recovered.tilt_y - baseline_geom.tilt_y),
            )
            @printf("  %-10s %16.6f %16.6f %16.6f\n", name, gt, rec, rec - gt)
        end
        println("─"^88)
    end

    return (baseline_geom=baseline_geom, perturbed_geom=perturbed_geom,
            hall=hall, recovered_pose=recovered,
            baseline_coils=baseline_coils, perturbed_coils=perturbed_coils)
end

end # module MinFull

using .MinFull

# ─── Run when executed directly (julia min_full.jl) ─────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    run_min_centering(
        baseline_file  = joinpath(@__DIR__, "sparc_pf1u.dat"),
        perturbed_file = joinpath(@__DIR__, "out_3mm.dat"),
        coil_current_a = 100.0,
    )
end
