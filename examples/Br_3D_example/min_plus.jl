# Hybrid coil-centering method ("plus"):
#   1. Build a B_R zero-crossing sinusoid initial guess for (z0, tilt_x, tilt_y)
#      from each Hall shell (as in the original coil_centering_full pipeline).
#   2. Hand the initial pose to the same 5D coordinate-descent forward-model
#      fit used by the "new" method (MinFull.field_residual).
#
# Self-contained: includes min_full.jl which sets up the MinFull module with
# load_coils, axis_from_coil, place_coil_at_axis, make_hall_data,
# field_residual, fit_pose_to_hall.

include(joinpath(@__DIR__, "min_full.jl"))

using Statistics
using Printf
using LinearAlgebra

# ─── B_R zero-crossing helpers (matching coil_centering_full.jl) ────────────

function _mp_zcrossings(vals, z)
    out = Float64[]
    for i in 1:length(vals)-1
        vals[i] == 0 && push!(out, z[i])
        vals[i] * vals[i+1] < 0 || continue
        t = vals[i] / (vals[i] - vals[i+1])
        push!(out, z[i] + t * (z[i+1] - z[i]))
    end
    return out
end

_mp_shellmeta(h, shell) = shell == 1 ?
    (length(h.phi_i), h.nzi, h.R_inner, h.Zi, h.phi_i) :
    (length(h.phi_o), h.nzo, h.R_outer, h.Zo, h.phi_o)

"""
    sinusoid_initial_pose(hall, phys_ref) -> NamedTuple

Initial pose for the seeded forward-model fit. `z0`, `tilt_x`, `tilt_y` come
from a per-φ B_R zero-crossing position fitted to `Z0 + B·cos(φ) + C·sin(φ)`
on each Hall shell, then averaged across shells. `x0`, `y0` are returned at
the baseline geometric values (the sinusoid does not constrain XY) —
coordinate descent finds those.

If the sinusoid fails on a shell (fewer than 4 valid zero crossings), that
shell is dropped. If both shells fail, all components fall back to the
baseline geometric pose so the seeded fit becomes equivalent to a cold start.
"""
function sinusoid_initial_pose(hall, phys_ref)
    function shell_zt(shell)
        nφ, nz, _, Zgrid, φgrid = _mp_shellmeta(hall, shell)
        BR = reshape(hall.B_R[hall.shell .== shell], nφ, nz)
        zc = fill(NaN, nφ)
        valid = falses(nφ)
        for ip in 1:nφ
            c = _mp_zcrossings(BR[ip, :], Zgrid)
            isempty(c) && continue
            zc[ip] = c[argmin(abs.(c .- phys_ref.z0))]
            valid[ip] = true
        end
        count(valid) >= 4 || return (z0=NaN, tilt_x=NaN, tilt_y=NaN)
        M = hcat(ones(count(valid)), cos.(φgrid[valid]), sin.(φgrid[valid]))
        Z0, Bc, Cs = M \ zc[valid]
        return (z0=Z0, tilt_x=atand(Cs, phys_ref.Rfit), tilt_y=atand(-Bc, phys_ref.Rfit))
    end
    a = shell_zt(1)
    b = shell_zt(2)
    meanfin(v) = (u = filter(isfinite, v); isempty(u) ? NaN : mean(u))
    z0 = meanfin([a.z0, b.z0])
    tx = meanfin([a.tilt_x, b.tilt_x])
    ty = meanfin([a.tilt_y, b.tilt_y])
    return (
        x0 = phys_ref.x0,
        y0 = phys_ref.y0,
        z0 = isfinite(z0) ? z0 : phys_ref.z0,
        tilt_x = isfinite(tx) ? tx : phys_ref.tilt_x,
        tilt_y = isfinite(ty) ? ty : phys_ref.tilt_y,
        sinusoid_valid = isfinite(z0) && isfinite(tx) && isfinite(ty),
    )
end

"""
    fit_pose_to_hall_seeded(baseline_coils, init_pose, hall; ...) -> NamedTuple

5D coordinate-descent forward-model fit, identical to
`MinFull.fit_pose_to_hall` but starting from an explicit `init_pose` (a
NamedTuple with `x0, y0, z0, tilt_x, tilt_y`) rather than the baseline
geometric pose. Pair with `sinusoid_initial_pose` for a warm start.
"""
function fit_pose_to_hall_seeded(baseline_coils, init_pose, hall;
        step0_m::Float64=0.050, step0_deg::Float64=0.5,
        tol_m::Float64=1e-7, tol_deg::Float64=1e-5,
        max_iter::Int=200, verbose::Bool=true)

    N = length(hall.R)
    buf = (BR=zeros(N), BP=zeros(N), BZ=zeros(N))

    pose = [init_pose.x0, init_pose.y0, init_pose.z0, init_pose.tilt_x, init_pose.tilt_y]
    steps = [step0_m, step0_m, step0_m, step0_deg, step0_deg]

    seed_coils = MinFull.place_coil_at_axis(baseline_coils, pose[1], pose[2], pose[3], pose[4], pose[5])
    best = MinFull.field_residual(seed_coils, hall, buf)
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
            trial = copy(pose)
            trial[i] += sgn * steps[i]
            cand = MinFull.place_coil_at_axis(baseline_coils, trial[1], trial[2], trial[3], trial[4], trial[5])
            val = MinFull.field_residual(cand, hall, buf)
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
        println("\nSeeded coordinate-descent fit (plus method):")
        @printf("  init pose: x=%.4f mm, y=%.4f mm, z=%.4f mm, tx=%.5f°, ty=%.5f°\n",
                init_pose.x0*1e3, init_pose.y0*1e3, init_pose.z0*1e3,
                init_pose.tilt_x, init_pose.tilt_y)
        @printf("  iters=%d, evals=%d, converged=%s, obj=%.3e\n",
                iter, n_eval, converged, best)
        @printf("  final pose: x=%.4f mm, y=%.4f mm, z=%.4f mm, tx=%.5f°, ty=%.5f°\n",
                pose[1]*1e3, pose[2]*1e3, pose[3]*1e3, pose[4], pose[5])
    end
    return result
end

"""
    run_min_plus_centering(; baseline_file, perturbed_file, ...) -> NamedTuple

End-to-end wrapper: loads two coils, generates synthetic Hall probes from
the perturbed coil, computes a sinusoid initial pose, and runs the seeded
coordinate-descent fit. Same shape as `MinFull.run_min_centering`, with an
extra `initial_pose` field.
"""
function run_min_plus_centering(;
        baseline_file::AbstractString,
        perturbed_file::AbstractString,
        coil_current_a::Float64=100.0,

        hall_r_inner_frac::Float64=0.40, hall_r_outer_frac::Float64=0.50,
        hall_n_phi_inner::Int=24, hall_n_z_inner::Int=22,
        hall_n_phi_outer::Int=20, hall_n_z_outer::Int=10,
        hall_z_halfspan_inner_m::Float64=0.60, hall_z_halfspan_outer_m::Float64=0.40,
        add_noise_frac::Float64=0.0,
        noise_floor_T::Float64=0.0, noise_rel_frac::Float64=0.0,

        step0_m::Float64=0.050, step0_deg::Float64=0.5,
        tol_m::Float64=1e-7, tol_deg::Float64=1e-5,
        max_iter::Int=200, verbose::Bool=true)

    baseline_coils  = MinFull.load_coils(baseline_file;  label="BASELINE",  current_A=coil_current_a, verbose=verbose)
    perturbed_coils = MinFull.load_coils(perturbed_file; label="PERTURBED", current_A=coil_current_a, verbose=verbose)
    baseline_geom   = MinFull.axis_from_coil(baseline_coils;  verbose=verbose, label="Baseline geometric axis")
    perturbed_geom  = MinFull.axis_from_coil(perturbed_coils; verbose=verbose, label="Perturbed geometric axis")

    hall = MinFull.make_hall_data(perturbed_coils, baseline_geom; verbose=verbose,
        hall_r_inner_frac=hall_r_inner_frac, hall_r_outer_frac=hall_r_outer_frac,
        hall_n_phi_inner=hall_n_phi_inner, hall_n_z_inner=hall_n_z_inner,
        hall_n_phi_outer=hall_n_phi_outer, hall_n_z_outer=hall_n_z_outer,
        hall_z_halfspan_inner_m=hall_z_halfspan_inner_m,
        hall_z_halfspan_outer_m=hall_z_halfspan_outer_m,
        add_noise_frac=add_noise_frac,
        noise_floor_T=noise_floor_T, noise_rel_frac=noise_rel_frac)

    init = sinusoid_initial_pose(hall, baseline_geom)
    rec  = fit_pose_to_hall_seeded(baseline_coils, init, hall;
        step0_m=step0_m, step0_deg=step0_deg, tol_m=tol_m, tol_deg=tol_deg,
        max_iter=max_iter, verbose=verbose)

    return (baseline_geom=baseline_geom, perturbed_geom=perturbed_geom,
            hall=hall, initial_pose=init, recovered_pose=rec,
            baseline_coils=baseline_coils, perturbed_coils=perturbed_coils)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_min_plus_centering(
        baseline_file  = joinpath(@__DIR__, "sparc_pf1u.dat"),
        perturbed_file = joinpath(@__DIR__, "out_3mm.dat"),
        coil_current_a = 100.0,
    )
end
