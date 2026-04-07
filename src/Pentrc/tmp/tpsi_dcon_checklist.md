# Tpsi DCON Checklist

Goal: make the `tpsi!` path used by DCON complete before widening the standalone PENTRC scope.

Scope: prioritize the `tpsi -> method kernel -> lsode helpers` path that DCON needs.

## Working Order

1. LSODE-layer replacements
2. Method kernels
3. DCON-facing initialization and input

## Definition Of Done

- Implemented: the Julia function is wired into the active calculation path.
- Julia-tested: at least one fixed input case runs without exceptions.
- Source-parity checked: compared against a direct Fortran-source translation or direct Fortran runtime output.
- Runtime-parity checked: compared against the actual Fortran executable output when available.
- Solver recorded: the solver used in Julia is explicitly noted in the log entry.
- Tolerance recorded: absolute and relative tolerances are written down with the verdict.

## Phase 1. LSODE Layer

### 1.1 `xintgrnd!`

- [x] Implemented
- [x] Julia-tested
- [x] Source-parity checked
- [ ] Runtime-parity checked
- Target file: `src/Pentrc/energy.jl`
- Fortran reference: `src/Pentrc/energy.f90`
- Solver to record: `none` (closed-form integrand evaluation)
- Key checks:
  - collision operator branch `xnutype`
  - zeroth-order distribution branch `xf0type`
  - complex contour `ximag`
  - heat-flux modifier `qt`

### 1.2 `xintgrl_lsode`

- [x] Implemented
- [x] Julia-tested
- [x] Source-parity checked
- [ ] Runtime-parity checked
- Target file: `src/Pentrc/energy.jl`
- Fortran reference: `src/Pentrc/energy.f90`
- Solver to record: `OrdinaryDiffEq.Tsit5()`
- Key checks:
  - imaginary-axis contour segment
  - real-axis segment
  - record path `op_record`
  - failure diagnostics

### 1.3 `lintgrnd!`

- [x] Implemented
- [x] Julia-tested
- [x] Source-parity checked
- [ ] Runtime-parity checked
- Target file: `src/Pentrc/Pitch.jl` or a replacement pitch implementation file
- Fortran reference: `src/Pentrc/pitch.f90`
- Solver to record: `none` (integrand)
- Key checks:
  - circulating vs trapped branch
  - `xintgrl_lsode` call pattern
  - `flambda` quantity combination

### 1.4 `lambdaintgrl_lsode`

- [x] Implemented
- [x] Julia-tested
- [x] Source-parity checked
- [ ] Runtime-parity checked
- Target file: `src/Pentrc/Pitch.jl` or a replacement pitch implementation file
- Fortran reference: `src/Pentrc/pitch.f90`
- Solver to record: `OrdinaryDiffEq.Tsit5()`
- Dependencies:
  - `xintgrl_lsode`
  - `lintgrnd!`
- Key checks:
  - `flambda` spline copy/evaluation
  - LSODE normalization removal
  - pitch-record path

### 1.5 `tintgrl_lsode`

- [x] Implemented
- [x] Julia-tested
- [x] Source-parity checked
- [ ] Runtime-parity checked
- Target file: `src/Pentrc/energy.jl`
- Fortran reference: `src/Pentrc/torque.F90`
- Solver to record: `OrdinaryDiffEq.Tsit5()`
- Dependencies:
  - `tpsi!`
  - `xintgrl_lsode`
  - `lambdaintgrl_lsode`
- Key checks:
  - adaptive `psi` integration
  - `-nl:nl` harmonic-state accumulation
  - `psi` interval clipping against equilibrium/input domains
  - method dispatch used by DCON
  - `tpsi!` wrapper behavior

## Phase 2. Method Kernels

### 2.1 `calculate_fcgl`

- [x] Implemented
- [x] Julia-tested
- [x] Source-parity checked
- [ ] Runtime-parity checked
- Target file: `src/Pentrc/torque.jl`
- Fortran reference: `src/Pentrc/torque.F90`
- Solver to record: not applicable unless it calls a lower-level solver

### 2.2 `calculate_rlar`

- [x] Implemented
- [x] Julia-tested
- [x] Source-parity checked
- [ ] Runtime-parity checked
- Target file: `src/Pentrc/torque.jl`
- Fortran reference: `src/Pentrc/torque.F90`
- Solver to record: `OrdinaryDiffEq.Tsit5()` in `xintgrl_lsode`, spline quadrature in `kappaintgrl`
- Dependencies:
  - `xintgrl_lsode`
  - `kappaintgrl`
  - `read_fnml`

### 2.3 `calculate_clar`

- [x] Implemented
- [x] Julia-tested
- [ ] Source-parity checked
- [ ] Runtime-parity checked
- Target file: `src/Pentrc/torque.jl`
- Fortran reference: `src/Pentrc/torque.F90`
- Solver to record: `OrdinaryDiffEq.Tsit5()` in `lambdaintgrl_lsode`
- Dependencies:
  - `lambdaintgrl_lsode`
  - elliptic helpers
  - `flambda` generation

### 2.4 `calculate_gar`

- [ ] Implemented
- [ ] Julia-tested
- [ ] Source-parity checked
- [ ] Runtime-parity checked
- Target file: `src/Pentrc/torque.jl`
- Fortran reference: `src/Pentrc/torque.F90`
- Dependencies:
  - `lambdaintgrl_lsode`
  - orbit/pitch helper quantities
  - `kappadjsum`

### 2.5 Helper Kernels

#### `kappaintgrl`

- [x] Implemented
- [x] Julia-tested
- [x] Source-parity checked
- [ ] Runtime-parity checked
- Target file: `src/Pentrc/torque.jl`
- Fortran reference: `src/Pentrc/pitch.f90`
- Solver to record: spline quadrature over `kappa`

#### `kappadjsum`

- [x] Implemented
- [x] Julia-tested
- [x] Source-parity checked
- [ ] Runtime-parity checked
- Target file: `src/Pentrc/torque.jl`
- Fortran reference: `src/Pentrc/pitch.f90`
- Solver to record: none (pointwise sum)

#### `flambda` generation

- [x] Implemented
- [x] Julia-tested
- [ ] Source-parity checked
- [ ] Runtime-parity checked
- Target file: `src/Pentrc/torque.jl` or `src/Pentrc/Pitch.jl`
- Fortran reference: `src/Pentrc/torque.F90`, `src/Pentrc/pitch.f90`

## Phase 3. Initialization / Input For DCON

### 3.1 `set_eq`

- [ ] Implemented
- [ ] Julia-tested
- [ ] Source-parity checked
- [ ] Runtime-parity checked
- Target file: `src/Pentrc/Input.jl`
- Key checks:
  - `sq`, `geom`, `eqfun`, `rzphi`
  - `chi1`, `bo`, `ro`, `theta`, `mthsurf`

### 3.2 `read_kin`

- [ ] Implemented
- [ ] Julia-tested
- [ ] Source-parity checked
- [ ] Runtime-parity checked
- Target file: `src/Pentrc/Input.jl`

### 3.3 `read_peq`

- [ ] Implemented
- [ ] Julia-tested
- [ ] Source-parity checked
- [ ] Runtime-parity checked
- Target file: `src/Pentrc/Input.jl`

### 3.4 `read_gpec_peq`

- [ ] Implemented
- [ ] Julia-tested
- [ ] Source-parity checked
- [ ] Runtime-parity checked
- Target file: `src/Pentrc/Input.jl`

### 3.5 `read_pmodb`

- [ ] Implemented
- [ ] Julia-tested
- [ ] Source-parity checked
- [ ] Runtime-parity checked
- Target file: `src/Pentrc/Input.jl`

### 3.6 `read_fnml`

- [ ] Implemented
- [ ] Julia-tested
- [ ] Source-parity checked
- [ ] Runtime-parity checked
- Target file: `src/Pentrc/Input.jl`

### 3.7 `set_mode_numbers!` / `active_mode_numbers`

- [ ] Implemented
- [ ] Julia-tested
- [ ] Source-parity checked
- [ ] Runtime-parity checked
- Target file: `src/Pentrc/pentrc.jl`

## Verification Log

Append one entry here after each implementation step.

#### 2026-04-02 22:58 KST - `xintgrnd!`

- status: source-parity-pass
- solver used: `none`
- files:
  - `src/Pentrc/energy.jl`
  - `src/Pentrc/tmp/verification_script.jl`
  - `src/Pentrc/tmp/xint_verification.md`
- input cases:
  - `case1_maxwellian_harmonic`
  - `case2_jkp_krook_imag`

#### 2026-04-03 13:40 KST - `calculate_rlar` / `kappaintgrl`

- status: source-parity-pass, runtime-baseline-blocked
- solver used:
  - `OrdinaryDiffEq.Tsit5()` for `xintgrl_lsode`
  - spline quadrature for `kappaintgrl`
- files:
  - `src/Pentrc/torque.jl`
  - `src/Pentrc/tmp/rlar_runtime_probe.jl`
  - `src/Pentrc/tmp/rlar_components_probe.jl`
  - `src/Pentrc/tmp/rlar_verification.md`
- runtime notes:
  - Julia RLAR path runs and returns nonzero values
  - current Fortran RLAR runtime cases in `runtime_rlar_case` and `runtime_rlar_regularized_case` produce identically zero `dTdpsi_rlar` / `T_rlar`
  - direct runtime parity is therefore not yet a useful pass/fail signal
- diagnostic notes:
  - `load_pmodb!` versus Fortran `pentrc_pmodb_n1.out` still shows mode-by-mode complex mismatch
  - simple global complex conjugation does not change RLAR, so the blocker is not a trivial conjugation convention

#### 2026-04-03 14:20 KST - `calculate_clar`

- status: local-runtime-mismatch-recorded
- solver used:
  - `OrdinaryDiffEq.Tsit5()` for `lambdaintgrl_lsode`
  - `OrdinaryDiffEq.Tsit5()` for nested `xintgrl_lsode`
- files:
  - `src/Pentrc/torque.jl`
  - `src/Pentrc/tmp/method_runtime_probe.jl`
  - `src/Pentrc/tmp/clar_verification.md`
- runtime notes:
  - Fortran CLAR baseline is nonzero and usable
  - current Julia local `ell=1` values are systematically high by about `18%` to `19%`
  - mismatch looks like a shared scaling / normalization issue rather than a noisy pointwise failure

#### 2026-04-03 01:26 KST - `calculate_fcgl`

- status: runtime-case-built, parity-fail
- solver used:
  - `calculate_fcgl`: `none`
  - Julia radial wrapper: `OrdinaryDiffEq.Tsit5()`
- files:
  - `src/Pentrc/torque.jl`
  - `src/Pentrc/Input.jl`
  - `src/Pentrc/tmp/runtime_fcgl_case/pentrc.in`
  - `src/Pentrc/tmp/runtime_fcgl_case/gpec.in`
  - `src/Pentrc/tmp/runtime_fcgl_case/equil.toml`
  - `src/Pentrc/tmp/fcgl_runtime_probe.jl`
  - `src/Pentrc/tmp/fcgl_verification.md`
- Fortran runtime:
  - `gpec` succeeded and produced `gpec_pmodb_n1.out`
  - `pentrc` succeeded and produced `pentrc_output_n1.nc`
- verdict:
  - Julia `fcgl` now runs on the same `kin` + `pmodb` case
  - runtime parity is not yet achieved; Julia result is near zero while Fortran is `O(1)`
  - `case3_cgl_zero`
- Julia result:
  - sampled pointwise integrand values matched the reference translation exactly
- comparison basis:
  - direct line-by-line translation of `src/Pentrc/energy.f90`
- error:
  - max abs error: `0.0`
- notes:
  - bundled output stored only in `xint_verification.md`

#### 2026-04-03 12:28 KST - `load_kinetic_profiles!`, `load_fnml!`, `load_equilibrium!`, `load_pmodb!`

- status: input-layer-progress, parity-improved
- solver used:
  - `load_kinetic_profiles!`: `none`
  - `load_fnml!`: `none`
  - `load_equilibrium!`: `none`
  - `load_pmodb!`: `none`
- files:
  - `src/Pentrc/Input.jl`
  - `src/Pentrc/pentrc.jl`
  - `src/Pentrc/tmp/fcgl_runtime_probe.jl`
  - `src/Pentrc/tmp/fcgl_verification.md`
- verification:
  - removed all legacy `set_eq/read_*` callsites from Julia sources
  - `load_fnml!` now reads the original `/Users/iseonjae/Desktop/GPEC/pentrc/fkmnl.dat` as Fortran unformatted records
  - `load_gpec_peq!` binary path now uses the same Fortran record reader
  - `load_kinetic_profiles!` now follows the Fortran structure:
    - `readtable` input
    - temporary spline on the experimental grid
    - regular-grid resampling
    - natural-log Coulomb log
    - spline-derivative-based `wdian/wdiat`
  - `load_equilibrium!` now stores jacobian metadata and builds `geom` from actual flux-surface `R,Z` contours instead of a circular placeholder
- fcgl probe result:
  - before input-layer fixes:
    - `tpsi(0.1, 0.5, 0.9) ≈ (2.63e-17, 1.93e-16, 1.76e-15)i`
  - after input-layer fixes:
    - `tpsi(0.1, 0.5, 0.9) ≈ (3.18e-1, 9.22e-1, 3.74e0)i`
- verdict:
  - input parity improved materially; Julia `fcgl` is no longer collapsing to numerical zero
  - runtime parity is still not achieved, and the remaining mismatch now looks like a normalization / radial-integration issue rather than a dead input path

#### 2026-04-03 13:05 KST - `calculate_rlar`, `calculate_clar`, `calculate_gar`

- status: scalar-kernel-progress, smoke-pass
- solver used:
  - `calculate_rlar`: `xintgrl_lsode -> OrdinaryDiffEq.Tsit5()`
  - `calculate_clar`: `lambdaintgrl_lsode -> OrdinaryDiffEq.Tsit5()`
  - `calculate_gar`: `lambdaintgrl_lsode -> OrdinaryDiffEq.Tsit5()`
- files:
  - `src/Pentrc/torque.jl`
  - `src/Pentrc/tmp/rlar_runtime_probe.jl`
  - `src/Pentrc/tmp/method_runtime_probe.jl`
- implementation notes:
  - `kappaintgrl` and `kappadjsum` now use the original `fnml` spline instead of placeholder RMS scaling
  - `calculate_clar` now builds a lambda-function spline and integrates through `lambdaintgrl_lsode`
  - `calculate_gar` now uses the same lambda-integral structure for the scalar torque path
  - GAR theta-space bounce/action integrals were switched to cumulative trapezoid integration to avoid complex-spline integration crashes
- Julia smoke results:
  - `rlar`:
    - `psi=0.1 -> 2.39e-05 + 6.55e-03i`
    - `psi=0.5 -> 1.10e-04 + 1.90e-02i`
    - `psi=0.9 -> -5.78e-04 + 1.85e-01i`
  - `clar`:
    - `psi=0.5 -> 5.00e-05 + 1.07e-02i`
    - `psi=0.9 -> -4.48e-04 + 9.98e-02i`
  - `fgar`:
    - `psi=0.1 -> 3.05e-05 - 1.10e-04i`
    - `psi=0.5 -> 9.29e-05 - 1.58e-04i`
    - `psi=0.9 -> 1.02e-04 - 5.16e-04i`
- caveats:
  - `clar` and `fgar` currently trigger many spline-root refinement warnings from the spline library
  - `wmats` / full matrix-element parity is still pending

#### 2026-04-02 22:58 KST - `xintgrl_lsode`

- status: source-parity-pass
- solver used: `OrdinaryDiffEq.Tsit5()`
- files:
  - `src/Pentrc/energy.jl`
  - `src/Pentrc/tmp/verification_script.jl`
  - `src/Pentrc/tmp/xint_verification.md`
- input cases:
  - `case1_maxwellian_harmonic`
  - `case2_jkp_krook_imag`
  - `case3_cgl_zero`
- Julia result:
  - case1: `-0.00022767029132538398 - 0.97627242895539im`
  - case2: `-0.004665501540899422 - 0.8907382161764089im`
  - case3: `0.0 - 3.3233509706387108im`
- comparison basis:
  - contour-integral reference translated from `src/Pentrc/energy.f90`
- error:
  - case1 rel err: `1.29e-11`
  - case2 rel err: `6.23e-9`
  - case3 rel err: `5.76e-11`
- notes:
  - runtime parity against the actual Fortran executable is still pending

#### 2026-04-02 23:13 KST - `lintgrnd!`

- status: source-parity-pass
- solver used: `none`
- files:
  - `src/Pentrc/Pitch.jl`
  - `src/Pentrc/tmp/verification_script.jl`
  - `src/Pentrc/tmp/lambda_verification.md`
- input cases:
  - synthetic `flambda` spline with mixed circulating/trapped lambda range
- Julia result:
  - sampled pointwise lambda integrand values matched the reference translation exactly
- comparison basis:
  - direct source translation of `lintgrnd` from `src/Pentrc/pitch.f90`
- error:
  - max sample error: `0.0`
  - mean sample error: `0.0`
- notes:
  - comparison reused the already-verified Julia `xintgrl_lsode` for the nested energy-space integral

#### 2026-04-02 23:13 KST - `lambdaintgrl_lsode`

- status: source-parity-pass
- solver used: `OrdinaryDiffEq.Tsit5()`
- files:
  - `src/Pentrc/Pitch.jl`
  - `src/Pentrc/tmp/verification_script.jl`
  - `src/Pentrc/tmp/lambda_verification.md`
- input cases:
  - synthetic `flambda` spline case spanning both circulating and trapped regions
- Julia result:
  - `ComplexF64[0.4242446835920722 - 1.4188072200547126im, -0.22445590623342068 - 0.2802004115122173im, 0.0563560344722568 - 0.11299271475850577im]`
- comparison basis:
  - direct source translation of `lambdaintgrl_lsode` logic, with lambda integration performed independently in the verification script
- error:
  - vector norm abs err: `8.33e-4`
  - vector norm rel err: `5.45e-4`
- notes:
  - runtime parity against the actual Fortran executable is still pending
  - current comparison uses a synthetic spline case rather than a full equilibrium-derived `flambda`

#### 2026-04-02 23:26 KST - `tintgrl_lsode`

- status: source-parity-pass
- solver used: `OrdinaryDiffEq.Tsit5()`
- files:
  - `src/Pentrc/energy.jl`
  - `src/Pentrc/tmp/verification_script.jl`
  - `src/Pentrc/tmp/tint_verification.md`
- input cases:
  - synthetic smooth `tpsi` surrogate over clipped interval `psi in [0.18, 0.84]`
  - `nn=3`, `nl=2`, ion branch, `method=clar`
- Julia result:
  - total: `1.1742233238152797 - 0.4783504068046298im`
- comparison basis:
  - direct source translation of `tintgrl_lsode/tintgrnd` wrapper behavior, with independent high-resolution quadrature over each `l` harmonic
- error:
  - component vector abs err: `9.28e-6`
  - component vector rel err: `1.30e-5`
  - total rel err: `3.54e-6`
- notes:
  - runtime parity against the actual Fortran executable is still pending
  - verification currently targets wrapper behavior and harmonic accumulation, not a full equilibrium-driven `tpsi!` case

### Template

```md
#### YYYY-MM-DD HH:MM KST - <function name>

- status: implemented | tested | source-parity-pass | source-parity-fail | runtime-parity-pass | runtime-parity-fail
- solver used:
- files:
  - `path/to/file`
- input cases:
  - method:
  - psi:
  - n:
  - l:
  - species:
- Julia result:
  - real:
  - imag:
- comparison basis:
  - direct Fortran runtime | direct source translation | analytical reference
- Fortran result:
  - real:
  - imag:
- error:
  - abs(real):
  - abs(imag):
  - rel(real):
  - rel(imag):
- notes:
```

## First Execution Plan

1. `xintgrnd!`
2. `xintgrl_lsode`
3. `lintgrnd!`
4. `lambdaintgrl_lsode`
5. `tintgrl_lsode`
6. `calculate_fcgl`
7. `kappaintgrl`
8. `calculate_rlar`
9. `flambda` generation
10. `calculate_clar`
11. `kappadjsum`
12. `calculate_gar`
13. `set_eq`
14. `read_kin`
15. `read_fnml`
16. `read_peq`
17. `read_gpec_peq`
18. `read_pmodb`
19. `set_mode_numbers!` / `active_mode_numbers`

## Current Notes

- The priority is `tpsi!` parity for DCON, not the standalone output path.
- Record the solver used after each implementation step.
- Keep bundled verification output in one file per topic when possible.
