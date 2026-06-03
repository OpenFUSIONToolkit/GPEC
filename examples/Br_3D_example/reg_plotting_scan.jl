# Full 3D-crossing scan driver for the three centering methods (old / new / plus).
#
# Loops over: Hall grid size × prescribed perturbation × noise_rel_frac
#             × noise_floor_T × method.  Total ~3240 fits.
#
# Each scan cell uses a unique seed so noise realizations are independent
# across cells. Method calls are wrapped in try/catch; failed cells write a
# NaN row with an `error_message` column.
#
# CSV path is the first positional argument (defaults to
# reg_plotting/scan_results.csv). Run from the repo root via reg_plot.py.

include(joinpath(@__DIR__, "min_plus.jl"))             # MinFull + plus helpers
include(joinpath(@__DIR__, "coil_centering_full.jl"))  # old-method primitives

using Printf, Random, LinearAlgebra, Statistics

const BASELINE_FILE  = joinpath(@__DIR__, "sparc_pf1u.dat")
const COIL_CURRENT_A = 100.0

# Unique seed per scan cell: seed = MEASUREMENT_SEED_BASE + cell_index.
# Methods themselves are deterministic; no additional RNG inside the methods.
const MEASUREMENT_SEED_BASE = 100_000

const GRID_CONFIGS = [
    (n_probes=50,   nφi=8,  nzi=4,  nφo=6,  nzo=3),    #  32 +  18 =   50
    (n_probes=100,  nφi=10, nzi=7,  nφo=8,  nzo=4),    #  70 +  32 =  102
    (n_probes=200,  nφi=14, nzi=10, nφo=12, nzo=5),    # 140 +  60 =  200
    (n_probes=500,  nφi=20, nzi=16, nφo=15, nzo=12),   # 320 + 180 =  500
    (n_probes=1000, nφi=28, nzi=24, nφo=20, nzo=16),   # 672 + 320 =  992
]
const SHIFTS_MM  = [0.01, 0.1, 1.0, 10.0, 100.0]
const TILTS_DEG  = [0.01, 0.1, 1.0, 5.0]
const NOISE_REL_FRACS = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5]
const NOISE_FLOOR_TS  = [0.0, 1.0e-6, 1.0e-5, 1.0e-4]

const FIT_KW = (step0_m=0.050, step0_deg=0.5,
                tol_m=1.0e-6, tol_deg=1.0e-4,
                max_iter=50)

baseline_coils = MinFull.load_coils(BASELINE_FILE; current_A=COIL_CURRENT_A, verbose=false)
baseline_geom  = MinFull.axis_from_coil(baseline_coils; verbose=false)

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
method_new(hall) = MinFull.fit_pose_to_hall(baseline_coils, baseline_geom, hall;
                                            verbose=false, FIT_KW...)
function method_plus(hall)
    init = sinusoid_initial_pose(hall, baseline_geom)
    fit_pose_to_hall_seeded(baseline_coils, init, hall; verbose=false, FIT_KW...)
end

function build_hall(perturbed_coils, cfg, noise_rel, noise_floor)
    MinFull.make_hall_data(perturbed_coils, baseline_geom; verbose=false,
        hall_r_inner_frac=0.40, hall_r_outer_frac=0.50,
        hall_n_phi_inner=cfg.nφi, hall_n_z_inner=cfg.nzi,
        hall_n_phi_outer=cfg.nφo, hall_n_z_outer=cfg.nzo,
        hall_z_halfspan_inner_m=0.60, hall_z_halfspan_outer_m=0.40,
        add_noise_frac=0.0,
        noise_floor_T=noise_floor, noise_rel_frac=noise_rel)
end

# CSV-safe error string: no newlines, no commas, capped length.
function clean_err(err)
    s = sprint(showerror, err)
    s = replace(s, '\n' => ' ')
    s = replace(s, '\r' => ' ')
    s = replace(s, ',' => ';')
    length(s) > 200 ? first(s, 200) * "..." : s
end

function row_ok(io, cfg, kind, dx_mm, dtx_deg, nr, nf, seed, method, r, t)
    @printf(io, "%d,%d,%d,%d,%d,%s,%.9e,%.9e,%.9e,%.9e,%d,%s,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%s\n",
        cfg.n_probes, cfg.nφi, cfg.nzi, cfg.nφo, cfg.nzo, kind,
        dx_mm, dtx_deg, nr, nf, seed, method,
        (r.x0 - baseline_geom.x0) * 1e3,
        (r.y0 - baseline_geom.y0) * 1e3,
        (r.z0 - baseline_geom.z0) * 1e3,
         r.tilt_x - baseline_geom.tilt_x,
         r.tilt_y - baseline_geom.tilt_y,
        t, "")
end

function row_fail(io, cfg, kind, dx_mm, dtx_deg, nr, nf, seed, method, err)
    @printf(io, "%d,%d,%d,%d,%d,%s,%.9e,%.9e,%.9e,%.9e,%d,%s,%s,%s,%s,%s,%s,%s,%s\n",
        cfg.n_probes, cfg.nφi, cfg.nzi, cfg.nφo, cfg.nzo, kind,
        dx_mm, dtx_deg, nr, nf, seed, method,
        "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", clean_err(err))
end

# Enumerate perturbations once
perturbations = Tuple{String,Float64,Float64}[]
for dx in SHIFTS_MM
    push!(perturbations, ("shift", dx, 0.0))
end
for dt in TILTS_DEG
    push!(perturbations, ("tilt", 0.0, dt))
end

csv_path = length(ARGS) >= 1 ? ARGS[1] :
           joinpath(@__DIR__, "..", "..", "reg_plotting", "scan_results.csv")
mkpath(dirname(csv_path))
io = open(csv_path, "w")
println(io,
    "n_probes,n_phi_inner,n_z_inner,n_phi_outer,n_z_outer,scan_kind,",
    "dx_true_mm,dtx_true_deg,noise_rel_frac,noise_floor_T,seed,method,",
    "dx_rec_mm,dy_rec_mm,dz_rec_mm,dtx_rec_deg,dty_rec_deg,runtime_s,error_message")

n_total = length(GRID_CONFIGS) * length(perturbations) *
          length(NOISE_REL_FRACS) * length(NOISE_FLOOR_TS) *
          3   # methods
@printf(stderr, "Total scan: %d fits across %d cells\n",
        n_total, n_total ÷ 3)

cell_index = 0
fails = Dict("old" => 0, "new" => 0, "plus" => 0)
t_total = @elapsed begin
    for cfg in GRID_CONFIGS
        @printf(stderr, ">>> Grid %d probes (%dx%d + %dx%d)\n",
                cfg.n_probes, cfg.nφi, cfg.nzi, cfg.nφo, cfg.nzo)
        for (kind, dx_mm, dtx_deg) in perturbations
            for nr in NOISE_REL_FRACS, nf in NOISE_FLOOR_TS
                global cell_index += 1
                seed = MEASUREMENT_SEED_BASE + cell_index
                Random.seed!(seed)

                p = MinFull.place_coil_at_axis(baseline_coils,
                    baseline_geom.x0 + dx_mm*1e-3,
                    baseline_geom.y0,
                    baseline_geom.z0,
                    baseline_geom.tilt_x + dtx_deg,
                    baseline_geom.tilt_y)
                hall = build_hall(p, cfg, nr, nf)

                for (mname, mfn) in (("old", method_old),
                                     ("new", method_new),
                                     ("plus", method_plus))
                    try
                        t = @elapsed r = mfn(hall)
                        row_ok(io, cfg, kind, dx_mm, dtx_deg, nr, nf, seed, mname, r, t)
                    catch err
                        row_fail(io, cfg, kind, dx_mm, dtx_deg, nr, nf, seed, mname, err)
                        fails[mname] += 1
                    end
                end
                flush(io)
            end
            @printf(stderr, "    %s %s done (cell %d / %d)\n",
                    kind, kind == "shift" ?
                    @sprintf("Δx=%.3f mm", dx_mm) :
                    @sprintf("Δtx=%.3f deg", dtx_deg),
                    cell_index, n_total ÷ 3)
        end
    end
end
close(io)
@printf(stderr, "\nWrote %s (total wall: %.1f s = %.2f h)\n",
        csv_path, t_total, t_total / 3600)
@printf(stderr, "Convergence failures: old=%d, new=%d, plus=%d  (out of %d cells each)\n",
        fails["old"], fails["new"], fails["plus"], cell_index)
