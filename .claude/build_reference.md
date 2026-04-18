# Build & Runtime Reference

Quick lookup for compilers, paths, and build commands across the codebase.

## Julia

- **Binary**: `/Users/danielburgess/.juliaup/bin/julia`
- **Version target**: 1.11
- **Project activation**: `julia --project=.` from repo root (`/Users/danielburgess/Desktop/plasma/julia_GPEC`)
- **Run example**: `julia --project=. examples/DIIID-like_ideal_example/run_example.jl`
- **Run tests**: `julia --project=. -e 'include("test/runtests.jl")'`

## Fortran GPEC (STRIDE)

- **Repo**: `/Users/danielburgess/Desktop/plasma/GPEC`
- **Binary**: `/Users/danielburgess/Desktop/plasma/GPEC/stride/stride`
- **Build**: `cd /Users/danielburgess/Desktop/plasma/GPEC/install && make stride`
- **Clean**: `cd /Users/danielburgess/Desktop/plasma/GPEC/install && make clean`
- **Input files**: `equil.in`, `stride.in`, `vac.in` (Fortran namelists) in the working directory
- **Output**: `stride_output_n<N>.nc` (NetCDF), `stride.out` (ASCII), `delta_prime.out` (ASCII)
- **Run**: `cd <working_dir_with_inputs> && /Users/danielburgess/Desktop/plasma/GPEC/stride/stride`
- **Compiler**: gfortran (arm64 macOS)
- **Dependencies**: LAPACK, NetCDF, HDF5 (all via Homebrew at `/opt/homebrew/lib`)

## Python

- **Binary**: `python3` (system, `/Library/Developer/CommandLineTools/usr/bin/python3`)
- **Packages**: numpy, matplotlib, netCDF4 (installed)
- **pip**: `pip3 install <pkg>` (version 21.2.4)

## Key Paths

| Item | Path |
|------|------|
| Julia GPEC repo | `/Users/danielburgess/Desktop/plasma/julia_GPEC` |
| Fortran GPEC repo | `/Users/danielburgess/Desktop/plasma/GPEC` |
| IDA_run data | `/Users/danielburgess/Desktop/plasma/IDA_run` |
| STRIDE binary | `/Users/danielburgess/Desktop/plasma/GPEC/stride/stride` |
| LAR epsilon scan (Julia) | `examples/LAR_epsilon_scan/` |
| LAR beta scan (Julia) | `examples/LAR_beta_scan/` |
| LAR scan Fortran reference | Fortran GPEC `docs/scans/outputs/` |
| DIII-D comparison | `examples/DIIID_stride_comparison/` |
| Regression harness | `regression-harness/regress.jl` |

## DIII-D Equilibrium Files

| Shot | geqdsk path |
|------|-------------|
| 147131 | `~/Desktop/plasma/julia_GPEC/examples/DIIID-like_ideal_example/TkMkr_D3Dlike_Hmode.geqdsk` |
| 147131 (original) | `~/Desktop/plasma/GPEC/docs/examples/DIIID_ideal_example/g147131.02300_DIIID_KEFIT` |
| 204441 | `~/Desktop/plasma/IDA_run/MSE_constraint/test_data/204441_4400_withEr/g204441.04409.geqdsk` |
| 153833 | `~/Desktop/plasma/IDA_run/IDA_workflow_examples/153833_3450/g153833.3450_results_baseline.geqdsk` |
| 153072 | `~/Desktop/plasma/IDA_run/MSE_constraint/test_data/153072_3415/g153072.03415.geqdsk` |

## Fortran STRIDE Input File Templates

### equil.in (key parameters)
```
&EQUIL_CONTROL
    eq_type="efit"
    eq_filename = "<geqdsk>"
    jac_type="hamada"
    grid_type="ldp"
    psilow=1e-4
    psihigh=0.993
    mpsi=128
    mtheta=256
/
```

### stride.in (key parameters)
```
&stride_control
    bal_flag=t, mat_flag=t, ode_flag=t, vac_flag=t, mer_flag=t
    sas_flag=t, dmlim=0.2, qlow=1.02, qhigh=1e3, sing_start=0
    nn=1
    delta_mlow=8, delta_mhigh=8, delta_mband=0
    mthvac=512, thmax0=1
    tol_nr=1e-8, tol_r=1e-8, crossover=1e-2
    singfac_min=1e-4, ucrit=1e4, sing_order=6
/
&stride_output
    netcdf_out=t
/
&stride_params
    nThreads=8
    nIntervalsTot=33
    grid_packing="singularities"
    asymp_at_sing=t
    calc_delta_prime=t
    calc_dp_with_vac=t
/
```

### vac.in (key parameters)
```
&MODES
   mth = 480
/
&VACDAT
   ishape = 6
   a = 20
   aw = 0.05
/
```
