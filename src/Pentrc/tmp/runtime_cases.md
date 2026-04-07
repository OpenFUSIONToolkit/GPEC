# Runtime Cases

Shared assets come from [`runtime_fcgl_case`](/Users/iseonjae/Desktop/JPEC/src/Pentrc/tmp/runtime_fcgl_case).

Each case directory contains:

- `pentrc.in` for a single method or branch
- `equil.toml` symlink
- symlinks to:
  - `g147131.02300_DIIID_KEFIT`
  - `g147131.02300_DIIID_KEFIT.kin`
  - `gpec_pmodb_n1.out`
  - `euler.bin`

## Cases

- [`runtime_fcgl_case`](/Users/iseonjae/Desktop/JPEC/src/Pentrc/tmp/runtime_fcgl_case): `fcgl` only, runtime-verified
- [`runtime_rlar_case`](/Users/iseonjae/Desktop/JPEC/src/Pentrc/tmp/runtime_rlar_case): `rlar` only, runtime-verified
- [`runtime_clar_case`](/Users/iseonjae/Desktop/JPEC/src/Pentrc/tmp/runtime_clar_case): `clar` only, runtime-verified
- [`runtime_fgar_case`](/Users/iseonjae/Desktop/JPEC/src/Pentrc/tmp/runtime_fgar_case): `fgar` only, runtime-verified
- [`runtime_tgar_case`](/Users/iseonjae/Desktop/JPEC/src/Pentrc/tmp/runtime_tgar_case): `tgar` only, runtime-verified
- [`runtime_pgar_case`](/Users/iseonjae/Desktop/JPEC/src/Pentrc/tmp/runtime_pgar_case): `pgar` only, runtime-verified

## Intended Use

Run original Fortran directly inside each case directory:

```bash
/Users/iseonjae/Desktop/GPEC/bin/pentrc
```

These cases exist to accumulate reproducible runtime baselines before Julia parity work on each method kernel.
