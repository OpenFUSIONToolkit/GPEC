================================================================================
Coil_Axis_Example — README
================================================================================

Self-contained example folder for the GPEC repository.  It demonstrates the
full coil-axis-recovery story end-to-end in Julia:

  1.  GENERATE noisy synthetic Hall-probe measurements for a known coil pose
      using the GPEC Biot-Savart kernel.

  2.  RECOVER that pose pose-blind by iteratively shifting/tilting the coil
      to match the noisy Hall data.

  3.  VISUALISE the coil + field in 3D (GLMakie).

  4.  DIAGNOSE how the recovery error scales with input measurement noise.

All scripts read coil geometry from `../examples/Br_3D_example/sparc_pf1u.dat`
and `sparc_pf1u_axisymmetric.dat` (those .dat files are NOT duplicated here —
the relative path keeps a single source of truth).


================================================================================
DIRECTORY LAYOUT
================================================================================

Coil_Axis_Example/
├── README.txt                    ← this file
│
├── synth_hall_cyl.jl             ← GENERATOR: place a coil at a 5- or 6-DOF
│                                   pose, compute B_R, B_phi, B_Z at user-
│                                   supplied cylindrical Hall probes, add
│                                   optional Gaussian noise, write CSV / HDF5.
│
├── synth_hall_cyl_test.jl        ← TEST SUITE (6 @testsets, 16 @tests) for
│                                   the generator, run against the idealised
│                                   axisymmetric coil to assert physical
│                                   symmetries (axisymmetry, z-translation
│                                   invariance, tilt sinusoid, centroid
│                                   invariance under rotation, …).
│
├── fit_hall_cyl.jl               ← INVERSE: takes a CSV (no pose metadata)
│                                   and a coil .dat, runs 5-DOF coordinate-
│                                   descent forward-model fit using only B_R
│                                   and B_Z residuals.
│
├── plot_3d.jl                    ← 3D VISUALISATION via GLMakie.  Coil
│                                   geometry + probes coloured by field
│                                   component + 3D arrows + toroidal slice
│                                   "blades".  Interactive walk-through with
│                                   ENTER between plots.
│
├── synth_hall_cyl.ipynb          ← IJulia (Julia kernel) NOTEBOOK that
│                                   stitches every step together: configure
│                                   pose + noise + grid → generate → reload
│                                   pose-blind → fit → diagnostics → noise-
│                                   to-error sweep.  Uses Plots.jl/GR for
│                                   inline figures.
│
├── csvs/                         ← AUTO-POPULATED CSV outputs from the
│   │                               generator / notebook / scan.
│   ├── synth_hall_demo.csv             (from `julia synth_hall_cyl.jl`)
│   ├── synth_hall_notebook.csv         (from notebook "generate" cell)
│   ├── synth_hall_for_fit.csv          (from notebook "load_blind" cell)
│   └── synth_hall_noise_scan.csv       (scratch, overwritten each sweep)
│
└── plots/                        ← AUTO-POPULATED PNG outputs.
    ├── 3d_probes_Bmag.png              (GLMakie, from plot_3d.jl)
    ├── 3d_probes_BR.png                (GLMakie)
    ├── 3d_probes_BZ.png                (GLMakie)
    ├── 3d_probes_Bphi.png              (GLMakie — tiny because near-axisym.)
    ├── 3d_field_vectors.png            (GLMakie arrows!)
    ├── 3d_slices_Bmag.png              (GLMakie — toroidal "blade" slices)
    ├── 3d_slices_BR.png                (GLMakie)
    ├── 3d_slices_BZ.png                (GLMakie)
    ├── 3d_slices_eight_blades.png      (GLMakie — 8 toroidal blades)
    ├── fit_residual_vs_iter.png        (notebook — log-y, descent curve)
    ├── fit_pose_trajectories.png       (notebook — 5 panels, one per DOF)
    ├── fit_residual_histogram.png      (notebook — ΔB_R, ΔB_Z residual hist)
    ├── fit_bphi_sanity.png             (notebook — "B magnitude comparison")
    └── noise_to_error.png              (notebook — sweep over noise_rel)


================================================================================
FILE-BY-FILE EXPLANATION
================================================================================

────────────────────────────────────────────────────────────────────────────
1. synth_hall_cyl.jl                                            [generator]
────────────────────────────────────────────────────────────────────────────

Self-contained Julia script.  At include time it activates the repo's
Project.toml and pulls in `GeneralizedPerturbedEquilibrium.ForcingTerms`
to access `read_coil_dat` and `compute_biot_savart_boundary!`.

Public helpers
  • coil_centroid(coils)             → (cx, cy, cz)
  • rotation_xyz(tx_deg, ty_deg, tz_deg)
        → 3x3 matrix Rz·Ry·Rx
  • place_coil!(coils, dx, dy, dz; tx_deg, ty_deg, tz_deg)
        in-place rigid transform of every segment;
        rotation about the coil centroid (NOT origin).
  • dual_shell_grid(R_inner, R_outer, Z_center,
                    half_inner, half_outer; nphi_*, nz_*)
        convenience two-shell cylindrical probe grid.

Main entry points
  • synth_hall_cyl(coil_file, dx, dy, dz;
                   tx_deg, ty_deg, tz_deg, current_A,
                   R_probes, phi_probes, Z_probes,
                   noise_floor_T, noise_rel_frac, noise_seed)
        loads coil, applies pose, runs Biot-Savart at the cylindrical
        probes, returns NamedTuple
        (R, phi, Z, B_R, B_phi, B_Z, applied_pose, noise, current_A,
         coil_file).
        Optional Gaussian noise per probe per component:
            σ_i = sqrt(noise_floor_T² + (noise_rel_frac · |B|)²)
        noise_seed picks a local MersenneTwister (does not touch the
        global RNG).

  • synth_hall_cyl_to_file(output_path, coil_file, dx, dy, dz; …)
        wrapper that calls synth_hall_cyl and writes the result.
        Format chosen from path extension:
          .csv  → comment header (coil file, pose, current, noise,
                                   timestamp) + rows
                  R,phi,Z,x,y,B_R,B_phi,B_Z,B_mag
          .h5   → HDF5 datasets + metadata group

Demo (`julia --project=. Coil_Axis_Example/synth_hall_cyl.jl`)
  Loads sparc_pf1u.dat, applies an example 5-DOF pose
  (5 mm, −2 mm, 1 mm, 0.30°, −0.10°) at 100 A,
  writes csvs/synth_hall_demo.csv (464 probes), prints |B| summaries.


────────────────────────────────────────────────────────────────────────────
2. synth_hall_cyl_test.jl                                        [tests]
────────────────────────────────────────────────────────────────────────────

Asserts physical symmetries against the IDEALISED axisymmetric coil
(`sparc_pf1u_axisymmetric.dat` — a 32-segment circular loop) so that
the assertions are exact.

`julia --project=. Coil_Axis_Example/synth_hall_cyl_test.jl`
→ 6 @testset blocks, 16 @tests, all expected to pass:
   1. zero-pose axisymmetry         (φ-variation < 1 ppm of |B|)
   2. tilt_z invariance             (Rz wired, axisym keeps |B| ≈ const)
   3. z-shift ↔ probe-Z shift       (translational invariance)
   4. x-shift → cos(φ) mode in B_R  (n=1 only, n≥2 < 5 %)
   5. tilt_x → sin(φ) mode in B_R   (n=1 only, b1 dominates a1)
   6. centroid invariance under
      rotation                      (place_coil! rotates about centroid)


────────────────────────────────────────────────────────────────────────────
3. fit_hall_cyl.jl                                            [inverse fit]
────────────────────────────────────────────────────────────────────────────

`include("synth_hall_cyl.jl")` at top → reuses place_coil!, coil_centroid,
read_coil_dat, compute_biot_savart_boundary!.  Adds:

Public
  • load_hall_cyl_csv(path) → (R, phi, Z, B_R, B_phi, B_Z)
        Strips '#' comment header.  POSE METADATA IS IGNORED so the
        fitter cannot "cheat".

  • residual_BR_BZ(BR_pred, BZ_pred, BR_meas, BZ_meas) → Float64
        Σ over probes of (ΔB_R² + ΔB_Z²).  B_phi is intentionally
        EXCLUDED because for near-axisymmetric coils it is two orders
        of magnitude smaller than B_R / B_Z, so noise on B_phi would
        dominate that channel.

  • fit_pose_BR_BZ(coil_file, hall;
                   init_pose, current_A, step0_m, step0_deg,
                   tol_m, tol_deg, max_iter, verbose,
                   fwd_noise_floor_T, fwd_noise_rel_frac,
                   fwd_noise_seed)
        5-DOF coordinate-descent (x, y, z, tilt_x_deg, tilt_y_deg)
        with halving step on no-improvement passes.  Tilt_z is NOT
        searched (B_R and B_Z don't constrain it for axisymmetric
        coils).
        fwd_noise_* is a Monte-Carlo objective hook (default off);
        fwd_noise_seed is stored in the result regardless so callers
        can document that it differs from the input-data seed.
        Returns (pose, history, n_eval, n_iter, converged,
                 residual, fwd_noise).  history is a Vector of
        NamedTuples (iter, n_eval, residual, pose) for plotting.

Demo
  `julia --project=. Coil_Axis_Example/fit_hall_cyl.jl`
  reads csvs/synth_hall_demo.csv and recovers the pose to ~µm /
  ~10 µdeg on the noiseless demo input.


────────────────────────────────────────────────────────────────────────────
4. plot_3d.jl                                                 [GLMakie 3D]
────────────────────────────────────────────────────────────────────────────

`include("synth_hall_cyl.jl"); using GLMakie`.  Produces all 3D figures
via GLMakie for proper depth-cueing, true 3D arrows, and interactive
rotation.  Notebook plots stay on Plots.jl per user request.

Helpers
  • coil_xyz(coil_file; pose, current_A, stride)
        flat (xs, ys, zs) for `lines!`, optionally sub-sampled.
  • plot_coil_only!(ax, coil_file; pose, current_A, …)
        overlay coil on any Axis3.

Plot makers (each returns a Figure; saves PNG if `outpath` supplied)
  • plot_probes_colored(hall; component, …)
        3D scatter of the dual-shell Hall probes coloured by
        :Bmag, :B_R, :B_phi, or :B_Z.  Zero-is-white colormap:
            signed components → Reverse(:RdBu), symmetric clims
            |B|               → :Blues, clims (0, max)

  • plot_field_slices(coil_file; phi_values, R/Z ranges, nR, nZ,
                       component, …)
        Dense (nR=80, nZ=100) scatter on flat (R,Z) "blades" at
        the requested toroidal angles — appears as solid coloured
        planes intersecting along the z-axis.

  • plot_field_vectors(hall; coil_file, pose, current_A, scale)
        True 3D `arrows!` for B at every probe.  Length and colour
        scale with |B|.

Demo (`julia --project=. Coil_Axis_Example/plot_3d.jl`)
  Generates the same canonical (pose, grid, current) data the
  notebook uses, then walks through 9 plots:
      Bmag / BR / BZ / Bphi probe scatters,
      3D arrow field,
      |B| / BR / BZ four-blade slices,
      eight-blade |B| slice.
  After each `display(fig)` the script blocks on `readline()` so
  the GLMakie window stays open; press ENTER to advance, or
  q+ENTER to quit early.  Each figure is also saved to plots/.


────────────────────────────────────────────────────────────────────────────
5. synth_hall_cyl.ipynb                                       [notebook]
────────────────────────────────────────────────────────────────────────────

IJulia kernel (Julia 1.11).  Cells in order:

  1.  intro            — markdown overview.
  2.  setup            — activate project, include synth_hall_cyl.jl,
                          using Plots; gr().
  3.  config-md / config — pick coil_file, pose, current_A, noise model,
                            probe grid (dual_shell_grid).
  4.  gen-md / generate  — synth_hall_cyl_to_file → csvs/synth_hall_notebook.csv.
  5.  inspect-md / inspect — print pose, noise model, |B| per component.
  6.  plot-md / plot   — 3D scatter of probes by |B| (Plots.jl).
  7.  (markdown 89e8f2cd) — "Fit a Hall file back to a coil pose
                              (pose-blind)" — explains the dual-seed
                              convention.
  8.  load_blind (3873452b) — include fit_hall_cyl.jl; noise_seed_input=42,
                                noise_seed_fit=137, asserted different;
                                regenerate input via synth_hall_cyl_to_file
                                into csvs/synth_hall_for_fit.csv;
                                load_hall_cyl_csv pose-blind.
  9.  fit (f75d1832)   — fit_pose_BR_BZ; prints recovered vs. true pose
                          with errors and final residual.
 10.  residual_vs_iter (bed6444a) — log-y plot of fit residual through
                                     iterations; writes
                                     plots/fit_residual_vs_iter.png.
 11.  diagnostics (3f394080) — 3-panel figure:
                                 pose-component trajectories vs iter
                                   (plots/fit_pose_trajectories.png),
                                 per-probe ΔB_R / ΔB_Z histogram
                                   (plots/fit_residual_histogram.png),
                                 "B magnitude comparison" — |B_phi|
                                   scatter with median |B_R|, |B_Z| as
                                   reference lines
                                   (plots/fit_bphi_sanity.png).
 12.  noise→error (9e07c6c3 / 398747f9) — sweep noise_rel_frac
                                            ∈ [1e-4 … 2e-1] at fixed
                                            floor=1e-5 T; refit each
                                            level with independent
                                            seed; log-log plot of shift
                                            and tilt error vs noise
                                            with linear-σ reference;
                                            writes plots/noise_to_error.png.


================================================================================
INPUTS REUSED FROM ELSEWHERE
================================================================================

  ../examples/Br_3D_example/sparc_pf1u.dat
      Helical 30600-segment SPARC PF1U coil.  Used by every script and
      the notebook as the "realistic" coil.

  ../examples/Br_3D_example/sparc_pf1u_axisymmetric.dat
      Idealised 32-segment circular loop at R = 0.926 m, Z = 0.  Used
      ONLY by synth_hall_cyl_test.jl so symmetries are exact and
      assertions are tight.

The .dat files are NOT duplicated into this folder so the source of
truth stays single.


================================================================================
HOW TO RUN
================================================================================

From the repo root:

  # Tests (≈ 1–2 s)
  julia --project=. Coil_Axis_Example/synth_hall_cyl_test.jl

  # Generator demo  → csvs/synth_hall_demo.csv
  julia --project=. Coil_Axis_Example/synth_hall_cyl.jl

  # Pose-blind fitter demo  (needs the CSV from the line above)
  julia --project=. Coil_Axis_Example/fit_hall_cyl.jl

  # Interactive 3D walk-through  → plots/3d_*.png
  julia --project=. Coil_Axis_Example/plot_3d.jl
  # → press ENTER between plots, q+ENTER to quit.

  # Notebook  → all plots/*.png and csvs/synth_hall_*.csv
  jupyter notebook Coil_Axis_Example/synth_hall_cyl.ipynb
  # (uses the IJulia kernel; cells must be run top-to-bottom)


================================================================================
DESIGN NOTES
================================================================================

* Pose convention:  pose components 1..3 are RELATIVE shifts (dx, dy, dz)
  applied AFTER rotation about the coil's geometric centroid.  Default
  pose = (0,0,0,0,0) means "leave the coil where the .dat file put it".

* B_phi is dropped from the fit residual.  For SPARC PF1U,
  median |B_phi| / median |B_R| ≈ 1e-2; including B_phi would let
  noise on a small signal dominate the objective without adding pose
  sensitivity.

* Dual-seed convention.  Input data noise uses `noise_seed_input`
  (default 42); the fitter is plumbed for `noise_seed_fit` (default
  137).  Currently the forward model is deterministic, so the fit
  seed is unused, but the asymmetry is enforced (`@assert
  noise_seed_input != noise_seed_fit`) so the fit can never share a
  noise realisation with the data.

* Plotting backends.  All inline / static figures use Plots.jl
  (GR backend).  Only the 3D plots in plot_3d.jl use GLMakie, for
  proper depth-cueing, true 3D arrows, and interactive rotation.

================================================================================
