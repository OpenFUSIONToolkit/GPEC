# JPEC – Kinetic Version Porting Plan

## Porting Map (GPEC → JPEC)
| Module | GPEC (Fortran) Dir | Function / Routine | JPEC Dir | Function | Notes | Status |
| --- | --- | --- | --- | --- | --- | --- |
| DCON | `dcon/forfit.F` | `fourfit_action_matrix()` | `src/DCON/Fourfit.jl` | `make_metric` | `fourfit_action_matrix()` Renamed `action_matrices()`. Isn't this different than `make_metric` | In progress |
|  |  | `fourfit_kinetic_matrix()` |  |  `make_kinetic_matrix`  | conneted with pentrc | In progress - mostly coded by Claude so far |
| PENTRC | `pentrc/torque.F90` |`tpsi()` | src/ | | | TODO |
| PENTRC | `pentrc/torque.F90` |`tintgrl_grid` | src/ | | | TODO |
| PENTRC | `pentrc/torque.F90` |`tintgrl_lsode` | src/ | | | TODO |
| DCON | `dcon/sing.f` | `ksing_find()` | `src/DCON/Sing.jl` | Option 1) `sing_find!`<br>Option 2) `sing_king_fing!` |  | In progress |
|  |  | `sing_get_f_det()` | `src/DCON/Sing.jl` | `sing_get_f_det()` | Maybe include inside `ksing_find()` | TODO |
|  |  | `sing_der()` | `src/DCON/Sing.jl` | `sing_der!` |  | TODO |
| DCON | `dcon/ode.f` | `ode_kin_cross()` | `src/DCON/Ode.jl` | `ode_kin_cross()` | for `ode_run()` | TODO |
|  |  | `ode_axis_init()` |  | `ode_axis_init()` |  | TODO |
|  |  | `ode_step()` |  | `compute_tols()` | Kinetic part needed only in `ode_step` | TODO |
| PENTRC | `pentrc/pentrc_interface.f90` | `initialize_pentrc()` |  |  |  | TODO |
| PENTRC | `pentrc/dcon_interface.f` | `set_eq()` |  |  |  | TODO |
|  |  | `set_geom()` |  |  | used in `set_eq()` | TODO |
| PENTRC | `pentrc/inputs.f90` | `read_kin()` |  |  |  | TODO |
|  |  | `read_equil()` |  |  |  | TODO |
|  |  | `set_peq()` |  |  |  | TODO |
| PENTRC | `pentrc/utilities` | `*` |  |  |  | TODO |
| PENTRC | `pentrc/*` | some more things |  |  | esp parmas.f90/ㅣlsode.f90 | TODO |

More things to change :
- use pitch_integration, only : lambdaintgrl_lsode,kappaintgrl,kappadjsum
- use energy_integration, only : xintgrl_lsode,qt
- use special, only : ellipk,ellipe
- use grid_mod, only : powspace_sub,linspace_sub
- I added `fourfit.F` to this repo in `src/dcon/fortran` right now. We will need to remove this later, but I wanted GitHub Copilot to have access to it.

---

## Additional TODOs

* [ ] Replace `OMP_SET_NUM_THREADS(dcon_kin_threads)`
  → Julia multithreading
  → [https://docs.julialang.org/en/v1/manual/multi-threading/](https://docs.julialang.org/en/v1/manual/multi-threading/)

* [ ] Remove unused or duplicated Julia files
  (e.g., `idcon_read`, `idcon_build`, …)
