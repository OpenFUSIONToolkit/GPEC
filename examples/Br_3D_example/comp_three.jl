# Compares three coil-centering methods (old / new / plus) across a grid of
# perturbations (shifts and tilts) and Hall-probe noise levels. Prints
# everything as a table to stdout. No notebook required.
#
#   - old  : coil_centering_full.jl pipeline — B_R zero-crossing sinusoid for
#            Z+tilt → local-B_phi grid search for XY → 4-step damped
#            fixed-point bias correction.
#   - new  : min_full.jl — pure 5D coordinate-descent forward-model fit,
#            cold-started from the baseline geometric pose.
#   - plus : min_plus.jl — B_R sinusoid initial guess for (z0, tilt_x, tilt_y)
#            (XY initialized from baseline), then the same 5D coordinate
#            descent.

include(joinpath(@__DIR__, "min_plus.jl"))            # brings in MinFull + plus functions
include(joinpath(@__DIR__, "coil_centering_full.jl")) # brings in old-method primitives

using Printf
using Random
using LinearAlgebra
using Statistics

const BASELINE_FILE  = joinpath(@__DIR__, "sparc_pf1u.dat")
const COIL_CURRENT_A = 100.0

# Small Hall grid keeps the scan tractable (~180 probes vs the notebook's 728).
const HALL_KW = (
    hall_r_inner_frac        = 0.40, hall_r_outer_frac        = 0.50,
    hall_n_phi_inner         = 12,   hall_n_z_inner           = 10,
    hall_n_phi_outer         = 10,   hall_n_z_outer           = 6,
    hall_z_halfspan_inner_m  = 0.60, hall_z_halfspan_outer_m  = 0.40,
    add_noise_frac           = 0.0,
)

const FIT_KW = (
    step0_m  = 0.050, step0_deg = 0.5,
    tol_m    = 1e-6,  tol_deg   = 1e-4,
    max_iter = 80,
)

const NOISE_LEVELS = [
    (label="0.00%", floor=0.0,   rel=0.0),
    (label="0.10%", floor=1.0e-5, rel=1.0e-3),
    (label="1.00%", floor=1.0e-5, rel=1.0e-2),
]
const SHIFTS_MM = [0.1, 1.0, 10.0, 100.0]
const TILTS_DEG = [0.01, 0.1, 1.0]

baseline_coils = MinFull.load_coils(BASELINE_FILE; current_A=COIL_CURRENT_A, verbose=false)
baseline_geom  = MinFull.axis_from_coil(baseline_coils; verbose=false)

function perturbed_coils_for(dx, dy, dz, dtx, dty)
    MinFull.place_coil_at_axis(baseline_coils,
        baseline_geom.x0 + dx, baseline_geom.y0 + dy, baseline_geom.z0 + dz,
        baseline_geom.tilt_x + dtx, baseline_geom.tilt_y + dty)
end

function make_hall(perturbed_coils, noise_floor_T, noise_rel_frac)
    MinFull.make_hall_data(perturbed_coils, baseline_geom;
        verbose=false, HALL_KW...,
        noise_floor_T=noise_floor_T, noise_rel_frac=noise_rel_frac)
end

function method_old(hall)
    redirect_stdout(devnull) do
        zt1  = magnetic_z_tilt(hall, baseline_geom; shell=1, verbose=false)
        zt2  = magnetic_z_tilt(hall, baseline_geom; shell=2, verbose=false)
        sin1 = magnetic_xy_minimize(hall, baseline_geom, zt1; shell=1, verbose=false)
        sin2 = magnetic_xy_minimize(hall, baseline_geom, zt2; shell=2, verbose=false)
        bias_correct_axis(baseline_coils, hall, zt1, zt2, sin1, sin2;
            n_iter=4, damping=0.6)
    end
end

method_new(hall)  = MinFull.fit_pose_to_hall(baseline_coils, baseline_geom, hall;
                                             verbose=false, FIT_KW...)

function method_plus(hall)
    init = sinusoid_initial_pose(hall, baseline_geom)
    fit_pose_to_hall_seeded(baseline_coils, init, hall; verbose=false, FIT_KW...)
end

delta_from_baseline(p) = (
    dx_mm   = (p.x0 - baseline_geom.x0) * 1e3,
    dy_mm   = (p.y0 - baseline_geom.y0) * 1e3,
    dz_mm   = (p.z0 - baseline_geom.z0) * 1e3,
    dtx_deg =  p.tilt_x - baseline_geom.tilt_x,
    dty_deg =  p.tilt_y - baseline_geom.tilt_y,
)

function run_one_case(dx, dy, dz, dtx, dty, noise_floor_T, noise_rel_frac; seed=42)
    p = perturbed_coils_for(dx, dy, dz, dtx, dty)
    Random.seed!(seed)
    hall = make_hall(p, noise_floor_T, noise_rel_frac)
    t_old  = @elapsed res_old  = method_old(hall)
    t_new  = @elapsed res_new  = method_new(hall)
    t_plus = @elapsed res_plus = method_plus(hall)
    return (
        old   = delta_from_baseline(res_old),
        new   = delta_from_baseline(res_new),
        plus  = delta_from_baseline(res_plus),
        t_old=t_old, t_new=t_new, t_plus=t_plus,
    )
end

n_probes = HALL_KW.hall_n_phi_inner*HALL_KW.hall_n_z_inner +
           HALL_KW.hall_n_phi_outer*HALL_KW.hall_n_z_outer

println("="^150)
println("Three-method comparison: OLD / NEW / PLUS")
@printf("Hall grid: %d probes (shell-1: %d×%d, shell-2: %d×%d)\n",
        n_probes,
        HALL_KW.hall_n_phi_inner, HALL_KW.hall_n_z_inner,
        HALL_KW.hall_n_phi_outer, HALL_KW.hall_n_z_outer)
@printf("Forward-model max_iter=%d, step0=(%.3f m, %.2f°), tol=(%g m, %g°)\n",
        FIT_KW.max_iter, FIT_KW.step0_m, FIT_KW.step0_deg, FIT_KW.tol_m, FIT_KW.tol_deg)
println("Noise model: σ_i = sqrt(floor² + (rel·|B|)²) per component per probe")
println("Each scan point shares the same Hall data across all three methods.")
println("="^150)

println("\n--- PURE X-SHIFT SCAN (Δx, no tilt) ---")
@printf("%-9s %-7s | %-22s | %-22s | %-22s | %s\n",
        "true Δx", "noise", "old: recovered (err)", "new: recovered (err)", "plus: recovered (err)",
        "t [s] old/new/plus")
println("-"^150)
for dx in SHIFTS_MM, ns in NOISE_LEVELS
    r = run_one_case(dx*1e-3, 0.0, 0.0, 0.0, 0.0, ns.floor, ns.rel)
    @printf("%6.3f mm %-7s | %+9.4f mm (%+9.4f) | %+9.4f mm (%+9.4f) | %+9.4f mm (%+9.4f) | %5.2f / %5.2f / %5.2f\n",
            dx, ns.label,
            r.old.dx_mm,  r.old.dx_mm  - dx,
            r.new.dx_mm,  r.new.dx_mm  - dx,
            r.plus.dx_mm, r.plus.dx_mm - dx,
            r.t_old, r.t_new, r.t_plus)
end

println("\n--- PURE TX-TILT SCAN (Δtx, no shift) ---")
@printf("%-12s %-7s | %-24s | %-24s | %-24s | %s\n",
        "true Δtx", "noise", "old: recovered (err)", "new: recovered (err)", "plus: recovered (err)",
        "t [s] old/new/plus")
println("-"^150)
for dt in TILTS_DEG, ns in NOISE_LEVELS
    r = run_one_case(0.0, 0.0, 0.0, dt, 0.0, ns.floor, ns.rel)
    @printf("%9.4f deg %-7s | %+10.5f° (%+10.5f) | %+10.5f° (%+10.5f) | %+10.5f° (%+10.5f) | %5.2f / %5.2f / %5.2f\n",
            dt, ns.label,
            r.old.dtx_deg,  r.old.dtx_deg  - dt,
            r.new.dtx_deg,  r.new.dtx_deg  - dt,
            r.plus.dtx_deg, r.plus.dtx_deg - dt,
            r.t_old, r.t_new, r.t_plus)
end

println("\n" * "="^150)
println("Note: 'err' columns are signed (recovered − true) in mm or deg, respectively.")
println("Cases where |err| ≫ true value indicate the perturbation is below the noise-induced detection floor.")
println("="^150)
