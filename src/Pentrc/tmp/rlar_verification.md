# RLAR Verification

Primary parity target: actual Fortran runtime output when it is numerically usable.

## Variable Mapping

- Julia local surface kernel:
  - `tpsi!(..., "rlar")`
- Fortran local profile:
  - `real(dTdpsi_rlar)` / `imag(dTdpsi_rlar)` in `pentrc_output_n1.nc`
- Julia radial integral:
  - `tintgrl_lsode([0.0, psi], ..., "rlar")`
- Fortran cumulative profile:
  - `real(T_rlar)` / `imag(T_rlar)` in `pentrc_output_n1.nc`

## Current Status

- `calculate_rlar` is wired into the active Julia path.
- `xintgrl_lsode` and `kappaintgrl` are both called from the live RLAR kernel.
- Julia smoke test passes for the DIII-D runtime case.
- Runtime parity is currently blocked by the available Fortran baseline case returning identically zero RLAR output.

## Julia Smoke Test

Case:
- `src/Pentrc/tmp/runtime_rlar_case`

Probe:
- `src/Pentrc/tmp/rlar_runtime_probe.jl`

Representative Julia outputs:

- `psi = 0.1`
  - `tpsi = 2.394087437308e-05 + 6.545292536939e-03i`
- `psi = 0.5`
  - `tpsi = 1.101767472930e-04 + 1.901768369764e-02i`
- `psi = 0.9`
  - `tpsi = -5.783169619856e-04 + 1.849045568329e-01i`

Solver path:

- `xintgrl_lsode`: `OrdinaryDiffEq.Tsit5()`
- `kappaintgrl`: spline quadrature over `kappa`

## Fortran Runtime Baseline

### `runtime_rlar_case`

Fortran output file:
- `src/Pentrc/tmp/runtime_rlar_case/pentrc_output_n1.nc`

Observed result:
- `dTdpsi_rlar = 0` at every recorded point
- `T_rlar = 0` at every recorded point
- global attributes:
  - `T_total_rlar = 0`
  - `dW_total_rlar = 0`

### `runtime_rlar_regularized_case`

Fortran output file:
- `src/Pentrc/tmp/runtime_rlar_regularized_case/pentrc_output_n1.nc`

Observed result:
- `dTdpsi_rlar_equil = 0` on the equilibrium grid
- `T_rlar_equil = 0` on the equilibrium grid
- global attributes:
  - `T_total_rlar_equil = 0`
  - `dW_total_rlar_equil = 0`

Conclusion:
- the currently prepared Fortran RLAR runtime cases are not yet a usable nonzero parity baseline.

## Input-Side Diagnostic

Debug comparison case:
- `src/Pentrc/tmp/runtime_rlar_debug_case`

Probe:
- `src/Pentrc/tmp/pmodb_compare_probe.jl`

Fortran reference:
- `src/Pentrc/tmp/runtime_rlar_debug_case/pentrc_pmodb_n1.out`

Comparison target:
- Julia `dbob_m.fs` / `divx_m.fs`
- Fortran post-processed `deltaB/B` / `divxprp` after `read_pmodb`

Current mismatch summary:

- `dbob_max_abs_err = 1.357786353456e-02`
- `divx_max_abs_err = 2.191815493865e-02`
- `dbob_mean_abs_err = 2.230202038395e-04`
- `divx_mean_abs_err = 4.141225200307e-04`

Worst samples:

- `dbob`
  - `psi = 9.911576920000e-01`, `m = 6`
  - Fortran: `2.369471870000e-03 - 6.656667520000e-03i`
  - Julia: `1.621284997063e-03 + 6.900566510819e-03i`
- `divx`
  - `psi = 9.374130520000e-01`, `m = 3`
  - Fortran: `1.256419940000e-03 + 1.101335440000e-02i`
  - Julia: `1.431350030898e-03 - 1.090410246384e-02i`

Interpretation:
- FCGL mainly depends on magnitude-like combinations, so this mismatch stayed mostly hidden there.
- RLAR is phase-sensitive through `kappaintgrl`, so the remaining `pmodb`/geometry phase difference is a plausible blocker.
- The present Julia probes rebuild equilibrium from `equil.toml`, while Fortran uses `euler.bin`; that geometry difference may be part of the remaining phase mismatch.

Conjugation sanity check:

- Conjugating every `dbob_m` / `divx_m` mode in the Julia probe did not change the RLAR result.
- This is expected for `kappaintgrl` because the quadratic combination uses `real(db_i * conj(db_j))`.
- Therefore the remaining mismatch is not explained by a simple global complex conjugation convention.

## Next Steps

1. Establish a nonzero Fortran RLAR reference case.
2. Compare Julia against Fortran on `dTdpsi_rlar` / `T_rlar`, not only on local probe values.
3. Narrow the remaining input mismatch by comparing:
   - Fortran `read_pmodb` debug output
   - Julia `load_pmodb!`
   - and, if needed, the equilibrium geometry path used by `idcon_coords`.
