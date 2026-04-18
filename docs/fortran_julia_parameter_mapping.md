# Fortran GPEC (STRIDE) to Julia GPEC Parameter Mapping

This document provides a comprehensive mapping between all Fortran GPEC namelist parameters and their Julia `gpec.toml` equivalents.

---

## 1. Equilibrium Parameters

**Fortran file:** `equil.in`
**Fortran namelists:** `equil_control`, `equil_output`
**Fortran source:** `equil/equil.f`, `equil/global.f`, `equil/direct.f`, `equil/read_eq.f`, `equil/local.f`
**Julia section:** `[Equilibrium]` in `gpec.toml`
**Julia struct:** `EquilibriumConfig` in `src/Equilibrium/EquilibriumTypes.jl`

### equil_control Namelist

| Fortran Parameter | Fortran Default | Julia Parameter | Julia Default | Notes |
|---|---|---|---|---|
| `eq_type` | `"fluxgrid"` | `eq_type` | `"efit"` | **Different defaults.** Fortran supports fluxgrid, efit, chease, lar, sol, etc. Julia supports efit, chease, chease2, lar, sol. |
| `eq_filename` | `"fluxgrid.dat"` | `eq_filename` | `"mypath"` | Path to equilibrium input file |
| `jac_type` | `"hamada"` | `jac_type` | `"hamada"` | Same. Options: hamada, pest, equal_arc, boozer, park (Julia adds "park" and "other") |
| `power_bp` | `0` | `power_bp` | `0` | Poloidal field power exponent |
| `power_b` | `0` | `power_b` | `0` | Total field power exponent |
| `power_r` | `0` | `power_r` | `0` | Major radius power exponent |
| -- | -- | `power_rc` | `0` | **Julia-only.** Minor radius power exponent (rfac) |
| `grid_type` | `"ldp"` | `grid_type` | `"log_asymptotic"` | **Different defaults.** Fortran: ldp, rho, pow1, pow2, original. Julia: log_asymptotic, ldp. Julia adds log_asymptotic grid type. |
| `psilow` | `1e-4` | `psilow` | `1e-2` | **Different defaults.** Lower limit of normalized flux. |
| `psihigh` | `1 - 1e-6` (~0.999999) | `psihigh` | `0.994` | **Different defaults.** Upper limit of normalized flux. Clamped to 1.0 in both. |
| `mpsi` | `128` | `mpsi` | `0` | **Different defaults.** Fortran: fixed at 128. Julia: 0 means auto-compute from psi_accuracy. |
| `mtheta` | `128` | `mtheta` | `256` | **Different defaults.** Number of poloidal grid points. |
| `newq0` | `0` | `newq0` | `0` | Override for on-axis safety factor |
| `etol` | `1e-8` | `etol` | `1e-7` | **Different defaults.** Error tolerance for equilibrium solver. |
| `input_only` | `.FALSE.` | `force_termination` | `false` | **Renamed.** Terminate after equilibrium setup. |
| `use_galgrid` | `.TRUE.` | `use_galgrid` | `true` | Use Galerkin grid |
| `use_classic_splines` | `.FALSE.` | -- | -- | **Fortran-only.** Julia uses FastInterpolations exclusively. |
| `jac_method` | `1` | -- | -- | **Fortran-only.** Jacobian computation method selector. |
| `convert_type` | `"none"` | -- | -- | **Fortran-only.** Coordinate conversion type. |
| `ns1` | `1` | -- | -- | **Fortran-only.** Used in read_eq. |
| `sp_pfac` | `1` | -- | -- | **Fortran-only.** Spline packing factor. |
| `sp_nx` | (unset) | -- | -- | **Fortran-only.** Spline grid count. |
| `sp_dx1` | `1e-3` | -- | -- | **Fortran-only.** Spline spacing parameter 1. |
| `sp_dx2` | `1e-3` | -- | -- | **Fortran-only.** Spline spacing parameter 2. |
| `enstep` | `16384` | -- | -- | **Fortran-only.** Max integration steps in direct equilibrium solver. |
| `power_flag` | `.TRUE.` | -- | -- | **Fortran-only.** Power flag for equilibrium. |
| -- | -- | `r0exp` | `1.0` | **Julia-only.** Major radius normalization for CHEASE/EQDSK [m]. |
| -- | -- | `b0exp` | `1.0` | **Julia-only.** On-axis toroidal field normalization [T]. |
| -- | -- | `psi_accuracy` | `0.001` | **Julia-only.** Target absolute error in q for auto-mpsi (when mpsi=0). |

### equil_output Namelist

| Fortran Parameter | Fortran Default | Julia Parameter | Julia Default | Notes |
|---|---|---|---|---|
| `out_eq_1d` | `.FALSE.` | -- | -- | **Fortran-only.** ASCII output of 1D equilibrium data. |
| `bin_eq_1d` | `.FALSE.` | -- | -- | **Fortran-only.** Binary output of 1D equilibrium data. |
| `out_eq_2d` | `.FALSE.` | -- | -- | **Fortran-only.** ASCII output of 2D equilibrium data. |
| `bin_eq_2d` | `.FALSE.` | -- | -- | **Fortran-only.** Binary output of 2D equilibrium data. |
| `out_2d` | `.FALSE.` | -- | -- | **Fortran-only.** ASCII output of processed 2D data. |
| `bin_2d` | `.FALSE.` | -- | -- | **Fortran-only.** Binary output of processed 2D data. |
| `out_fl` | `.FALSE.` | -- | -- | **Fortran-only.** ASCII field line output. |
| `bin_fl` | `.FALSE.` | -- | -- | **Fortran-only.** Binary field line output. |
| `interp` | `.FALSE.` | -- | -- | **Fortran-only.** Interpolation output flag. |
| `gse_flag` | `.FALSE.` | -- | -- | **Fortran-only.** Grad-Shafranov equation diagnostic. |
| `dump_flag` | `.FALSE.` | -- | -- | **Fortran-only.** Binary dump of basic equilibrium. |
| `verbose` | `.TRUE.` | -- | -- | **Fortran-only.** In Julia, verbose is on ForceFreeStatesControl. |

---

## 2. Stability / STRIDE Parameters

**Fortran file:** `stride.in`
**Fortran namelists:** `stride_control`, `stride_output`, `stride_params`
**Fortran source:** `stride/stride.F`, `stride/dcon_mod.f`, `stride/ode.F`, `stride/sing.F`
**Julia section:** `[ForceFreeStates]` in `gpec.toml`
**Julia struct:** `ForceFreeStatesControl` in `src/ForceFreeStates/ForceFreeStatesStructs.jl`

### stride_control Namelist

| Fortran Parameter | Fortran Default | Julia Parameter | Julia Default | Notes |
|---|---|---|---|---|
| `bal_flag` | `.FALSE.` | `bal_flag` | `false` | Ballooning mode analysis |
| `mat_flag` | `.FALSE.` | `mat_flag` | `false` | Matrix output |
| `ode_flag` | `.FALSE.` | `ode_flag` | `false` | ODE integration diagnostics |
| `vac_flag` | `.FALSE.` | `vac_flag` | `false` | Vacuum region calculation |
| `mer_flag` | `.FALSE.` | `mer_flag` | `false` | Mercier stability criterion |
| `fft_flag` | `.FALSE.` | `fft_flag` | `false` | Fourier transform analysis |
| `node_flag` | `.FALSE.` | -- | -- | **Fortran-only.** Node detection flag. |
| `mthvac` | `480` | `mthvac` | `480` | Number of vacuum poloidal grid points |
| `sing_start` | `0` | `sing_start` | `0` | Start integration at N-th singular surface |
| `nn` | (no default) | `nn_low` / `nn_high` | `0` / `0` | **Restructured.** Fortran uses single `nn` for one toroidal mode. Julia uses `nn_low`/`nn_high` range to support multi-n. |
| `delta_mlow` | `0` | `delta_mlow` | `0` | Expands lower bound of Fourier harmonics |
| `delta_mhigh` | `0` | `delta_mhigh` | `0` | Expands upper bound of Fourier harmonics |
| `delta_mband` | `0` | `delta_mband` | `0` | Band width in m,m' diagonal |
| `thmax0` | `1` | `thmax0` | `1.0` | Linear multiplier on theta integration bounds |
| `nstep` | `HUGE(0)` | `nstep` | `typemax(Int)` | Max number of integration steps |
| `ksing` | `-1` | `ksing` | `-1` | Singular surface handling parameter |
| `tol_nr` | `1e-5` | `eulerlagrange_tolerance` | `1e-7` | **Renamed, different defaults.** Fortran has separate `tol_nr`/`tol_r`; Julia unifies to single tolerance. |
| `tol_r` | `1e-5` | `eulerlagrange_tolerance` | `1e-7` | **Merged.** Fortran near-rational tolerance merged into Julia's single tolerance. |
| `crossover` | `1e-2` | -- | -- | **Fortran-only.** Fractional distance where tolerance switches from `tol_nr` to `tol_r`. |
| `ucrit` | `1e4` | `ucrit` | `1e4` | Critical unorm ratio for normalization |
| `singfac_min` | `1e-5` (ode.F) | `singfac_min` | `0.0` | **Different defaults.** Fractional distance from rational q for ideal jump condition. |
| `singfac_max` | `1e-4` (ode.F) | -- | -- | **Fortran-only.** Upper bound of singular factor. Julia uses only singfac_min. |
| `cyl_flag` | `.FALSE.` | `cyl_flag` | `false` | Cylindrical mode truncation |
| `sas_flag` / `lim_flag` | `.FALSE.` | `set_psilim_via_dmlim` | `false` | **Renamed.** Same behavior: determines psilim from outermost rational + dmlim. |
| `dmlim` | `0.5` | `dmlim` | `0.2` | **Different defaults.** Distance beyond last rational surface. |
| `sing_order` | `2` | `sing_order` | `2` | Order of singular layer expansion |
| `qlow` | `0` | `qlow` | `0.0` | Integration q lower limit |
| `qhigh` | `1e3` | `qhigh` | `1e3` | Integration q upper limit |
| `use_classic_splines` | `.FALSE.` | -- | -- | **Fortran-only.** Julia uses FastInterpolations. |
| `use_notaknot_splines` | `.TRUE.` (stride.F) | -- | -- | **Fortran-only.** Julia uses its own spline BC logic. |
| `reform_eq_with_psilim` | `.FALSE.` (stride.F) | `reform_eq_with_psilim` | `false` | Reform equilibrium with computed psilim |
| -- | -- | `nzvac` | `1` | **Julia-only.** Number of vacuum toroidal grid points (for 3D vacuum). |
| -- | -- | `verbose` | `true` | **Julia-only.** Verbose output control. |
| -- | -- | `numsteps_init` | `4000` | **Julia-only.** Initial array size for ODE data storage. |
| -- | -- | `numunorms_init` | `100` | **Julia-only.** Initial array size for normalization data. |
| -- | -- | `psiedge` | `1.0` | **Julia-only.** Edge dW scan boundary parameter. |
| -- | -- | `diagnose` | `false` | **Julia-only.** Diagnostic output flag. |
| -- | -- | `diagnose_ca` | `false` | **Julia-only.** Asymptotic coefficient diagnostics. |

### stride_output Namelist

| Fortran Parameter | Fortran Default | Julia Parameter | Julia Default | Notes |
|---|---|---|---|---|
| `interp` | `.FALSE.` | -- | -- | **Fortran-only.** Interpolation output. |
| `crit_break` | `.TRUE.` | -- | -- | **Fortran-only.** Color crit curve at singular surfaces. |
| `out_bal1` | `.FALSE.` | -- | -- | **Fortran-only.** ASCII ballooning output. |
| `bin_bal1` | `.FALSE.` | -- | -- | **Fortran-only.** Binary ballooning output. |
| `out_bal2` | `.FALSE.` | -- | -- | **Fortran-only.** ASCII ballooning functions. |
| `bin_bal2` | `.FALSE.` | -- | -- | **Fortran-only.** Binary ballooning functions. |
| `out_metric` | `.FALSE.` | -- | -- | **Fortran-only.** ASCII metric output. |
| `bin_metric` | `.FALSE.` | -- | -- | **Fortran-only.** Binary metric output. |
| `out_fmat` | `.FALSE.` | -- | -- | **Fortran-only.** ASCII F-matrix output. |
| `bin_fmat` | `.FALSE.` | -- | -- | **Fortran-only.** Binary F-matrix output. |
| `out_gmat` | `.FALSE.` | -- | -- | **Fortran-only.** ASCII G-matrix output. |
| `bin_gmat` | `.FALSE.` | -- | -- | **Fortran-only.** Binary G-matrix output. |
| `out_kmat` | `.FALSE.` | -- | -- | **Fortran-only.** ASCII K-matrix output. |
| `bin_kmat` | `.FALSE.` | -- | -- | **Fortran-only.** Binary K-matrix output. |
| `out_sol` | `.FALSE.` | -- | -- | **Fortran-only.** ASCII solution output. |
| `out_sol_min` | (unset) | -- | -- | **Fortran-only.** Min solution output index. |
| `out_sol_max` | (unset) | -- | -- | **Fortran-only.** Max solution output index. |
| `bin_sol` | `.FALSE.` | -- | -- | **Fortran-only.** Binary solution output. |
| `bin_sol_min` | (unset) | -- | -- | **Fortran-only.** Min binary solution index. |
| `bin_sol_max` | (unset) | -- | -- | **Fortran-only.** Max binary solution index. |
| `out_fl` | `.FALSE.` | -- | -- | **Fortran-only.** ASCII field line output. |
| `bin_fl` | `.FALSE.` | -- | -- | **Fortran-only.** Binary field line output. |
| `out_evals` | `.FALSE.` | -- | -- | **Fortran-only.** ASCII eigenvalue output. |
| `bin_evals` | `.FALSE.` | -- | -- | **Fortran-only.** Binary eigenvalue output. |
| `bin_euler` | `.FALSE.` | -- | -- | **Fortran-only.** Dummy variable (does nothing). |
| `euler_stride` | `1` | `save_interval` | `3` | **Renamed, different defaults.** Save every Nth ODE step. Fortran: every step. Julia: every 3rd. |
| `ahb_flag` | `.FALSE.` | -- | -- | **Fortran-only.** Boundary eigenfunction output. |
| `mthsurf0` | `1` | -- | -- | **Fortran-only.** Boundary surface point multiplier. |
| `msol_ahb` | (unset) | -- | -- | **Fortran-only.** Number of eigenfunctions for ahb. |
| `netcdf_out` | `.TRUE.` | -- | -- | **Fortran-only.** Julia outputs HDF5 natively. |
| `out_ahg2msc` | `.TRUE.` | -- | -- | **Fortran-only.** Deprecated vacuum communication file. |
| -- | -- | `write_outputs_to_HDF5` | `true` | **Julia-only.** Write results to HDF5. |
| -- | -- | `HDF5_filename` | `"gpec.h5"` | **Julia-only.** Output filename. |
| -- | -- | `force_wv_symmetry` | `true` | **Julia-only.** Enforce vacuum matrix symmetry. |
| -- | -- | `force_termination` | `false` | **Julia-only.** Skip perturbed equilibrium after stability. |

### stride_params Namelist

| Fortran Parameter | Fortran Default | Julia Parameter | Julia Default | Notes |
|---|---|---|---|---|
| `grid_packing` | (no default) | -- | -- | **Fortran-only.** "singularities" or "naive". Julia always uses singularity-aware grid. |
| `asymp_at_sing` | (no default) | -- | -- | **Fortran-only.** Julia always uses asymptotic expansions. |
| `integrate_riccati` | (no default) | `use_riccati` | `false` | **Renamed.** Riccati reformulation of the EL ODE. |
| `calc_delta_prime` | (no default) | -- | -- | **Fortran-only.** Julia computes delta prime automatically when asymptotic data is available. |
| `calc_dp_with_vac` | (no default) | -- | -- | **Fortran-only.** Julia couples vacuum to delta prime automatically when vac_flag=true. |
| `axis_mid_pt_skew` | (no default) | -- | -- | **Fortran-only.** Interval distribution skew. |
| `big_soln_err_tol` | (no default) | -- | -- | **Fortran-only.** Delta prime BVP error threshold. |
| `kill_big_soln_for_ideal_dW` | (no default) | -- | -- | **Fortran-only.** Remove big solution for ideal dW. |
| `ric_dt` | (no default) | -- | -- | **Fortran-only.** Riccati initial step size. |
| `ric_tol` | (no default) | -- | -- | **Fortran-only.** Riccati relative tolerance. |
| `riccati_bounce` | (no default) | -- | -- | **Fortran-only.** Riccati integration mode toggle. |
| `riccati_match_hamiltonian_evals` | (no default) | -- | -- | **Fortran-only.** Riccati Hamiltonian eigenvalue matching. |
| `verbose_riccati_output` | (no default) | -- | -- | **Fortran-only.** Riccati diagnostic output. |
| `verbose_performance_output` | (no default) | -- | -- | **Fortran-only.** Timing information. |
| `fourfit_metric_parallel` | (no default) | -- | -- | **Fortran-only.** Parallel metric computation. |
| `vac_parallel` | `.TRUE.` | -- | -- | **Fortran-only.** Julia parallelism is handled differently. |
| `nIntervalsTot` | `33` | -- | -- | **Fortran-only.** Number of radial intervals for parallel decomposition. |
| `nThreads` | `0` | -- | -- | **Fortran-only.** Julia uses `Threads.nthreads()`. |
| -- | -- | `use_parallel` | `false` | **Julia-only.** Parallel fundamental matrix integration via Threads.@threads. |

### Kinetic Parameters (in stride_control or ForceFreeStatesControl)

| Fortran Parameter | Fortran Default | Julia Parameter | Julia Default | Notes |
|---|---|---|---|---|
| (in separate kinetic module) | -- | `kin_flag` | `false` | Kinetic EL equation flag |
| -- | -- | `con_flag` | `false` | Continue through layers |
| -- | -- | `kinfac1` | `1.0` | Energy scale factor |
| -- | -- | `kinfac2` | `1.0` | Torque scale factor |
| -- | -- | `kingridtype` | `0` | Grid type |
| -- | -- | `ktanh_flag` | `false` | Hyperbolic tangent profile |
| -- | -- | `passing_flag` | `false` | Passing particle effects |
| -- | -- | `trapped_flag` | `true` | Trapped particle effects |
| -- | -- | `ion_flag` | `true` | Ion kinetic effects |
| -- | -- | `electron_flag` | `false` | Electron kinetic effects |
| -- | -- | `ktc` | `0.1` | Kinetic collision parameter |
| -- | -- | `ktw` | `50.0` | Kinetic width parameter |

---

## 3. Vacuum Parameters

**Fortran file:** `vac.in`
**Fortran namelists:** `modes`, `debugs`, `vacdat`, `shape`, `diagns`
**Fortran source:** `vacuum/vacuum_io.f`, `vacuum/vacuum_global.f`, `vacuum/vacuum_ma.f`
**Julia section:** `[Wall]` in `gpec.toml` (wall shape), `[ForceFreeStates]` (mthvac)
**Julia structs:** `WallShapeSettings` and `VacuumInput` in `src/Vacuum/DataTypes.jl`

### modes Namelist

| Fortran Parameter | Fortran Default | Julia Parameter | Julia Default | Notes |
|---|---|---|---|---|
| `mfel` | (unset) | -- | -- | **Fortran-only.** Finite element count per period. |
| `m` | (unset) | -- | -- | **Fortran-only.** Number of modes. Set by STRIDE. |
| `mth` | (set from `nths0`) | `mthvac` | `480` | **Mapped differently.** Fortran reads from vac.in but STRIDE overrides with `mthvac`. Julia uses `mthvac` in `[ForceFreeStates]`. |
| `n` | (unset, real) | `nn_low` / `nn_high` | `0` / `0` | Toroidal mode number. Fortran: real-valued `n`. Julia: integer range. |
| `mdiv` | (set to 2 in vacuum_ma.f) | -- | -- | **Fortran-only.** Mode subdivision parameter. |
| `lsymz` | `.FALSE.` | -- | -- | **Fortran-only.** Up-down symmetry flag. |
| `lfunin` | (unset) | -- | -- | **Fortran-only.** Function input flag. |
| `xiin(1:9)` | `(0,0,0,0,0,0,0,1,0)` | -- | -- | **Fortran-only.** Boundary condition input array. |
| `leqarcw` | (unset) | `equal_arc_wall` | `true` | **Renamed.** Fortran: integer `leqarcw=1` for equal-arc wall. Julia: bool. |
| `lpest1` | (unset) | -- | -- | **Fortran-only.** PEST1 flag. |
| `lnova` | (unset) | -- | -- | **Fortran-only.** NOVA flag. |
| `ladj` | (unset) | -- | -- | **Fortran-only.** Adjacent coupling flag. |
| `ldcon` | (unset) | -- | -- | **Fortran-only.** DCON interface flag (set internally). |
| `lgato` | (unset) | -- | -- | **Fortran-only.** GATO interface flag. |
| `lrgato` | (unset) | -- | -- | **Fortran-only.** R-GATO interface flag. |
| `lspark` | (unset) | -- | -- | **Fortran-only.** SPARC interface flag. |
| `ismth` | (unset) | -- | -- | **Fortran-only.** Smoothing parameter. |
| `lzio` | (unset) | -- | -- | **Fortran-only.** Z I/O flag. |
| `mp0` | (unset) | -- | -- | **Fortran-only.** Mode parameter 0. |
| `mp1` | (unset) | -- | -- | **Fortran-only.** Mode parameter 1. |

### debugs Namelist

| Fortran Parameter | Fortran Default | Julia Parameter | Julia Default | Notes |
|---|---|---|---|---|
| `checkd` | `.FALSE.` | -- | -- | **Fortran-only.** Debug check D. |
| `checke` | `.FALSE.` | -- | -- | **Fortran-only.** Debug check E. |
| `check1` | `.FALSE.` | -- | -- | **Fortran-only.** Debug check 1. |
| `check2` | `.FALSE.` | -- | -- | **Fortran-only.** Debug check 2. |
| `checks` | `.FALSE.` | -- | -- | **Fortran-only.** Debug check S. |
| `wall` | `.FALSE.` | -- | -- | **Fortran-only.** Wall debug flag. |
| `lkplt` | `0` | -- | -- | **Fortran-only.** Plot flag. |
| `verbose_timer_output` | (unset) | -- | -- | **Fortran-only.** Verbose timer. |

### vacdat Namelist

| Fortran Parameter | Fortran Default | Julia Parameter | Julia Default | Notes |
|---|---|---|---|---|
| `ishape` | (unset) | `shape` | `"nowall"` | **Restructured.** Fortran: integer `ishape` (0=no wall, 6=dee, etc.). Julia: string in `[Wall]` section. |
| `aw` | (unset) | `aw` | `0.05` | Wall half-thickness (Dee shape) |
| `bw` | (unset) | `bw` | `1.5` | Wall elongation |
| `cw` | (unset) | `cw` | `0.0` | Wall center offset |
| `dw` | (unset) | `dw` | `0.5` | Wall triangularity |
| `tw` | (unset) | `tw` | `0.05` | Wall corner sharpness |
| `nsing` | (unset) | -- | -- | **Fortran-only.** Number of singular points in Green's function integration. |
| `epsq` | `1e-5` (vacuum_ma.f) | -- | -- | **Fortran-only.** Epsilon for quadrature. |
| `noutv` | (unset) | -- | -- | **Fortran-only.** Output verbosity level for vacuum. |
| `delg` | (unset) | -- | -- | **Fortran-only.** Green's function integration parameter. |
| `idgt` | `0` (vacuum_ma.f) | -- | -- | **Fortran-only.** Digit accuracy parameter. |
| `idot` | (unset) | -- | -- | **Fortran-only.** Dot product flag. |
| `delfac` | (unset) | -- | -- | **Fortran-only.** Delta factor for Green's function. |
| `idsk` | `1` (vacuum_ma.f) | -- | -- | **Fortran-only.** Disk I/O flag. |
| `cn0` | (unset) | -- | -- | **Fortran-only.** Normalization constant. |
| `use_legacy_greens_function` | `.FALSE.` | -- | -- | **Fortran-only.** Use old Green's function method. Julia always uses improved method. |

### shape Namelist

| Fortran Parameter | Fortran Default | Julia Parameter | Julia Default | Notes |
|---|---|---|---|---|
| `ipshp` | (unset) | -- | -- | **Fortran-only.** Plasma shape type. |
| `xpl` | (unset) | -- | -- | **Fortran-only.** Plasma X parameter. |
| `apl` | (unset) | -- | -- | **Fortran-only.** Plasma A parameter. |
| `bpl` | (unset) | -- | -- | **Fortran-only.** Plasma B parameter. |
| `dpl` | (unset) | -- | -- | **Fortran-only.** Plasma D parameter. |
| `a` | (unset) | `a` | `0.3` | Wall distance / shape parameter |
| `b` | (unset) | -- | -- | **Fortran-only.** Shape B parameter. |
| `r` | (overridden to 0) | -- | -- | **Fortran-only.** Always overridden to 0 for DCON. |
| `abulg` | (unset) | -- | -- | **Fortran-only.** Bulge A parameter. |
| `bbulg` | (unset) | -- | -- | **Fortran-only.** Bulge B parameter. |
| `tbulg` | (unset) | -- | -- | **Fortran-only.** Bulge T parameter. |
| `qain` | (unset) | -- | -- | **Fortran-only.** Edge safety factor input. |

### diagns Namelist

| Fortran Parameter | Fortran Default | Julia Parameter | Julia Default | Notes |
|---|---|---|---|---|
| `lkdis` | (unset) | -- | -- | **Fortran-only.** Dispersion relation flag. |
| `ieig` | (unset) | -- | -- | **Fortran-only.** Eigenvalue output flag. |
| `iloop` | (unset) | -- | -- | **Fortran-only.** Loop diagnostic. |
| `nloop` | (unset) | -- | -- | **Fortran-only.** Number of loop points. |
| `nloopr` | (unset) | -- | -- | **Fortran-only.** Loop radial count. |
| `lpsub` | (unset) | -- | -- | **Fortran-only.** Subplot flag. |
| `nphil` | (unset) | -- | -- | **Fortran-only.** Number of phi levels. |
| `nphse` | (unset) | -- | -- | **Fortran-only.** Number of phases. |
| `mx` | (unset) | -- | -- | **Fortran-only.** X grid for diagnostics. |
| `mz` | (unset) | -- | -- | **Fortran-only.** Z grid for diagnostics. |
| `nph` | (unset) | -- | -- | **Fortran-only.** Phase count. |
| `xofsl` | (unset) | -- | -- | **Fortran-only.** X offset. |
| `aloop` | (unset) | -- | -- | **Fortran-only.** Loop minor radius. |
| `bloop` | (unset) | -- | -- | **Fortran-only.** Loop elongation. |
| `dloop` | (unset) | -- | -- | **Fortran-only.** Loop triangularity. |
| `rloop` | (unset) | -- | -- | **Fortran-only.** Loop major radius. |
| `ntloop` | (unset) | -- | -- | **Fortran-only.** Loop theta count. |
| `deloop` | (unset) | -- | -- | **Fortran-only.** Loop delta. |
| `nxlpin` | (unset) | -- | -- | **Fortran-only.** X line point count. |
| `nzlpin` | (unset) | -- | -- | **Fortran-only.** Z line point count. |
| `epslp` | (unset) | -- | -- | **Fortran-only.** Line epsilon. |
| `xlpmin` | (unset) | -- | -- | **Fortran-only.** X line min. |
| `xlpmax` | (unset) | -- | -- | **Fortran-only.** X line max. |
| `zlpmin` | (unset) | -- | -- | **Fortran-only.** Z line min. |
| `zlpmax` | (unset) | -- | -- | **Fortran-only.** Z line max. |
| `linterior` | (unset) | -- | -- | **Fortran-only.** Interior calculation level. |

---

## 4. Perturbed Equilibrium Parameters

**Fortran:** Part of the broader GPEC suite (separate modules, not in STRIDE namelists)
**Julia section:** `[PerturbedEquilibrium]` in `gpec.toml`
**Julia struct:** `PerturbedEquilibriumControl` in `src/PerturbedEquilibrium/PerturbedEquilibriumStructs.jl`

| Fortran Parameter | Fortran Default | Julia Parameter | Julia Default | Notes |
|---|---|---|---|---|
| (various, in separate GPEC PE module) | -- | `fixed_boundary` | `false` | Fixed boundary conditions |
| -- | -- | `output_eigenmodes` | `true` | Output eigenmode fields |
| -- | -- | `compute_response` | `true` | Compute plasma response |
| -- | -- | `compute_singular_coupling` | `true` | Compute singular coupling metrics |
| -- | -- | `verbose` | `true` | Verbose logging |
| -- | -- | `output_filename` | `""` | Output file (empty = use FFS filename) |
| -- | -- | `write_outputs_to_HDF5` | `true` | Write to HDF5 |
| -- | -- | `filter_modes` | `false` | Mode filtering |
| -- | -- | `singular_point_method` | `"standard"` | Singular point treatment method |

---

## 5. Forcing Terms Parameters

**Fortran:** Handled via `coil.in` and related files in the Fortran GPEC suite
**Julia section:** `[ForcingTerms]` in `gpec.toml`
**Julia struct:** `ForcingTermsControl` in `src/ForcingTerms/ForcingTerms.jl`

| Fortran Parameter | Fortran Default | Julia Parameter | Julia Default | Notes |
|---|---|---|---|---|
| (coil.in file) | -- | `forcing_data_file` | `"forcing.dat"` | Path to forcing data |
| -- | -- | `forcing_data_format` | `"ascii"` | "ascii" or "hdf5" |

---

## 6. Large Aspect Ratio (LAR) Equilibrium Parameters

**Fortran:** Handled internally in LAR module
**Julia section:** `[LAR_INPUT]` in `gpec.toml` (for LAR equilibria)
**Julia struct:** `LargeAspectRatioConfig` in `src/Equilibrium/EquilibriumTypes.jl`

| Fortran Parameter | Fortran Default | Julia Parameter | Julia Default | Notes |
|---|---|---|---|---|
| -- | -- | `lar_r0` | `10.0` | Major radius [m] |
| -- | -- | `lar_a` | `1.0` | Minor radius [m] |
| -- | -- | `beta0` | `1e-3` | On-axis beta |
| -- | -- | `q0` | `1.5` | On-axis safety factor |
| -- | -- | `p_pres` | `2.0` | Pressure profile exponent |
| -- | -- | `p_sig` | `1.0` | Current profile exponent |
| -- | -- | `sigma_type` | `"default"` | Sigma profile type |
| -- | -- | `mtau` | `128` | Poloidal grid points |
| -- | -- | `ma` | `128` | Radial grid points |
| -- | -- | `zeroth` | `false` | Neglect Shafranov shift |

---

## 7. Solovev Equilibrium Parameters

**Fortran:** Handled internally in SOL module
**Julia section:** `[SOL_INPUT]` in `gpec.toml` (for Solovev equilibria)
**Julia struct:** `SolovevConfig` in `src/Equilibrium/EquilibriumTypes.jl`

| Fortran Parameter | Fortran Default | Julia Parameter | Julia Default | Notes |
|---|---|---|---|---|
| -- | -- | `mr` | `128` | Radial grid zones |
| -- | -- | `mz` | `128` | Axial grid zones |
| -- | -- | `ma` | `128` | Flux grid zones |
| -- | -- | `e` | `1.6` | Elongation |
| -- | -- | `a` | `0.33` | Minor radius |
| -- | -- | `r0` | `1.0` | Major radius |
| -- | -- | `q0` | `1.9` | Safety factor at O-point |
| -- | -- | `p0fac` | `1.0` | Pressure scale factor |
| -- | -- | `b0fac` | `1.0` | Toroidal field scale factor |
| -- | -- | `f0fac` | `1.0` | Toroidal field scale (constant pressure) |

---

## Summary of Key Behavioral Differences

1. **Toroidal mode number**: Fortran uses a single `nn` (integer). Julia uses `nn_low`/`nn_high` range to support multi-n calculations.

2. **ODE tolerance**: Fortran has separate `tol_nr` (away from rationals) and `tol_r` (near rationals) with `crossover` parameter. Julia unifies these into a single `eulerlagrange_tolerance`.

3. **Grid type**: Fortran defaults to `"ldp"` radial packing. Julia defaults to `"log_asymptotic"` which auto-selects mpsi based on `psi_accuracy`.

4. **Radial resolution**: Fortran uses fixed `mpsi=128`. Julia uses `mpsi=0` (auto-compute) by default with `psi_accuracy=0.001`.

5. **Wall shape specification**: Fortran uses integer `ishape` codes (0, 6, etc.) and parameters spread across `vacdat` and `shape` namelists. Julia uses a descriptive string `shape` in a dedicated `[Wall]` section.

6. **Output format**: Fortran outputs to NetCDF (`netcdf_out`), ASCII, and binary files. Julia outputs exclusively to HDF5 (`gpec.h5`).

7. **Splines**: Fortran has `use_classic_splines` and `use_notaknot_splines` toggles. Julia uses FastInterpolations exclusively with no user-facing spline method selection.

8. **Parallelism**: Fortran uses OpenMP with `nThreads` and `nIntervalsTot` for interval-based parallelism. Julia uses `Threads.@threads` with `use_parallel` flag; chunk decomposition is automatic.

9. **Delta prime**: Fortran requires explicit `calc_delta_prime=t` and `asymp_at_sing=t`. Julia computes delta prime automatically when asymptotic data is available.

10. **Vacuum communication**: Fortran originally communicated between STRIDE and VACUUM via `ahg2msc.out` files (deprecated, replaced by in-memory). Julia has no intermediate files; vacuum is called as a library function.

11. **Singfac_min default**: Fortran defaults to `1e-5`. Julia defaults to `0.0` (no explicit jump condition unless specified).

12. **Save interval**: Fortran `euler_stride=1` (save every step). Julia `save_interval=3` (save every 3rd step).

13. **Wall Fortran ishape integer to Julia shape string mapping**:
    - `ishape=0` -> `shape="nowall"` (roughly; ishape=0 may mean analytical plasma shape)
    - `ishape=6` -> `shape="dee"` (Dee-shaped wall)
    - Other shapes mapped via `a`, `aw`, `bw`, `cw`, `dw`, `tw` parameters.
