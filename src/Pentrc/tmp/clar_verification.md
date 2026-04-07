# CLAR Verification

Primary parity target: Fortran runtime NetCDF output from [`runtime_clar_case`](/Users/iseonjae/Desktop/JPEC/src/Pentrc/tmp/runtime_clar_case).

## Variable Mapping

- Julia local surface kernel:
  - `tpsi!(..., "clar")`
- Fortran local profile:
  - `dTdpsi_clar(psi_clar, ell, i)` in [`pentrc_output_n1.nc`](/Users/iseonjae/Desktop/JPEC/src/Pentrc/tmp/runtime_clar_case/pentrc_output_n1.nc)
- Julia radial integral:
  - `tintgrl_lsode([0.0, psi], ..., "clar")`
- Fortran cumulative profile:
  - `T_clar(psi_clar, ell, i)` in [`pentrc_output_n1.nc`](/Users/iseonjae/Desktop/JPEC/src/Pentrc/tmp/runtime_clar_case/pentrc_output_n1.nc)

For the current local comparisons:

- `n = 1`
- `ell = 1`
- species = thermal ion

## Fortran Baseline

Case:
- [`runtime_clar_case`](/Users/iseonjae/Desktop/JPEC/src/Pentrc/tmp/runtime_clar_case)

Fortran global totals:

- `T_total_clar = 3.941997090007e-01`
- `dW_total_clar = 2.905562956303e-02`

So unlike the current RLAR runtime cases, CLAR has a usable nonzero Fortran baseline.

## Julia Local Comparison

Probe:
- [`method_runtime_probe.jl`](/Users/iseonjae/Desktop/JPEC/src/Pentrc/tmp/method_runtime_probe.jl)

Nearest `psi_clar` points used from Fortran:

1. `psi = 0.091046810178`
2. `psi = 0.547552122729`
3. `psi = 0.903715836303`

Local `ell = 1` comparison:

- `psi = 0.091046810178`
  - Fortran `dTdpsi_clar = 9.975408353071e-06 + 3.515342538989e-03i`
  - Julia `tpsi = 1.304761488378e-05 + 4.196282571542e-03i`
  - relative error: `1.937064e-01`
  - magnitude ratio `|Julia|/|Fortran| = 1.193706`

- `psi = 0.547552122729`
  - Fortran `dTdpsi_clar = 5.724944496874e-05 + 1.167131022718e-02i`
  - Julia `tpsi = 7.918795589943e-05 + 1.394239496792e-02i`
  - relative error: `1.945937e-01`
  - magnitude ratio `|Julia|/|Fortran| = 1.194592`

- `psi = 0.903715836303`
  - Fortran `dTdpsi_clar = -5.546392089105e-04 + 8.924132853031e-02i`
  - Julia `tpsi = -7.411928389401e-04 + 1.050691986318e-01i`
  - relative error: `1.773692e-01`
  - magnitude ratio `|Julia|/|Fortran| = 1.177367`

## Current Interpretation

- CLAR is not zero and is directly comparable to Fortran.
- The current mismatch is systematic rather than random.
- Across the sampled points, Julia is consistently about `18%` to `19%` larger in magnitude than Fortran.
- That pattern suggests a shared normalization, geometry, or pitch-weighting convention difference rather than a branch-local bug at one `psi`.

## Known Diagnostics

- The Julia CLAR path currently emits many spline root-refinement warnings while finding turning points.
- Despite that warning flood, the computation completes and returns stable values.
- These warnings should be reduced, but they do not by themselves explain the near-uniform scaling mismatch.

## Next Steps

1. Compare cumulative Julia `tintgrl_lsode(..., "clar")` against Fortran `T_clar`.
2. Inspect CLAR normalization:
   - `tnorm`
   - `fbnce_norm`
   - `djdj` scaling from `kappadjsum`
3. Check whether the remaining factor is inherited from the same equilibrium / `pmodb` convention gap seen earlier.
