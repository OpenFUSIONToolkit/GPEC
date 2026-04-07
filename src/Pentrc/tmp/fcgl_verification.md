# FCGL Runtime Verification

Case directory: `src/Pentrc/tmp/runtime_fcgl_case`

Goal: compare the Julia `fcgl` path against the original Fortran runtime using the same `equil + kin + pmodb` inputs.

## Primary Rule

Use the Fortran NetCDF output as the primary parity target.

- Reference file:
  - `src/Pentrc/tmp/runtime_fcgl_case/pentrc_output_n1.nc`
- Fortran terminal `T_phi` lines are only a secondary sanity check.
- The main comparison must be Julia variables against Fortran `fcgl` variables stored in the NetCDF file.

## Variable Mapping

### 1. Local FCGL surface kernel

- Julia:
  - `tpsi!(ts, psi, 1, 0, 1, 2, 1.0, 1.0, false, "fcgl")`
- Fortran NetCDF:
  - `imag(dTdpsi_fcgl(:, 1, :))`
  - In the file layout used here, the last index `i=2` is the imaginary part

Meaning:

- Local value on a single flux surface
- This is the correct comparison for Julia `tpsi!`

### 2. Cumulative FCGL radial integral

- Julia:
  - `tintgrl_lsode([0.0, psi], 1, 0, 1, 2, 1.0, 1.0, false, "fcgl")`
- Fortran NetCDF:
  - `imag(T_fcgl(:, 1, :))`
  - Again, the last index `i=2` is the imaginary part

Meaning:

- Integral from `psi=0` up to the current `psi`
- This is the correct comparison for the Fortran terminal `psi -> T_phi` progress lines

### 3. Total integrated FCGL output

- Julia:
  - `tintgrl_lsode([0.0, 1.0], 1, 0, 1, 2, 1.0, 1.0, false, "fcgl")`
- Fortran NetCDF global attribute:
  - `dW_total_fcgl`

Meaning:

- Final integrated FCGL energy-like output
- For `fcgl`, `T_total_fcgl = 0` and the relevant total comparison is `dW_total_fcgl`

### 4. Flux-function sanity checks from perturbation input

- Julia reconstructed from loaded splines:
  - `|dB/B|^2` surface average from `dbob_m`
  - `|div xi_perp|^2` surface average from `divx_m`
- Fortran NetCDF:
  - `sqdBoB_L_SA_fcgl`
  - `sqdivxi_perp_SA_fcgl`

Meaning:

- Checks whether `load_pmodb!` is feeding the `fcgl` kernel correctly

## Things That Must Not Be Compared Directly

- Julia `tpsi!(..., "fcgl")`
- Fortran terminal `psi -> T_phi`

Reason:

- The Julia quantity is local-in-`psi`
- The Fortran terminal quantity is cumulative-in-`psi`

## Runtime Setup

- Original executables:
  - `/Users/iseonjae/Desktop/GPEC/bin/gpec`
  - `/Users/iseonjae/Desktop/GPEC/bin/pentrc`
- Julia probe:
  - `src/Pentrc/tmp/fcgl_runtime_probe.jl`
- Julia solver record:
  - `calculate_fcgl`: no ODE solver
  - `tintgrl_lsode`: `OrdinaryDiffEq.Tsit5()`

## Comparison Points

The cleanest comparison uses the actual `psi_fcgl` points stored in the Fortran NetCDF file:

- `psi = 0.10224598621083623`
- `psi = 0.4996307469131832`
- `psi = 0.9093384445179171`

## NetCDF-Based Comparison

### Local kernel: Julia `tpsi!` vs Fortran `imag(dTdpsi_fcgl)`

- `psi = 0.102245986211`
  - Fortran `imag(dTdpsi_fcgl) = 3.281744649922e-01`
  - Julia `tpsi! = 3.289357362466e-01 i`

- `psi = 0.499630746913`
  - Fortran `imag(dTdpsi_fcgl) = 9.296402267633e-01`
  - Julia `tpsi! = 9.156338376095e-01 i`

- `psi = 0.909338444518`
  - Fortran `imag(dTdpsi_fcgl) = 5.279396286625e+00`
  - Julia `tpsi! = 5.249829289113e+00 i`

### Cumulative profile: Julia `tintgrl_lsode([0, psi])` vs Fortran `imag(T_fcgl)`

- `psi = 0.102245986211`
  - Fortran `imag(T_fcgl) = 5.839598672747e-02`
  - Julia cumulative = `5.691373730185e-02 i`

- `psi = 0.499630746913`
  - Fortran `imag(T_fcgl) = 2.286018734504e-01`
  - Julia cumulative = `2.220156456440e-01 i`

- `psi = 0.909338444518`
  - Fortran `imag(T_fcgl) = 1.051380959751e+00`
  - Julia cumulative = `1.030275897007e+00 i`

### Final total: Julia total vs Fortran `dW_total_fcgl`

- Fortran `dW_total_fcgl = 7.092171503892e-01`
- Julia `tintgrl_lsode([0, 1.0], ...) = 1.314888409648e+00 i`

Note:

- The full-total convention still needs a final check.
- The `psi<=0.9` cumulative comparison is already much tighter than the earlier mistaken comparison.

## Input-Layer Sanity Checks

These checks suggest the main remaining gap is no longer a gross input-read failure.

### Equilibrium globals

- Fortran NetCDF globals:
  - `R0 = 1.75888031905042`
  - `B0 = 1.69026883381664`
  - `chi1 = 1.64798888117825`
- Julia:
  - `ro = 1.7588803190504143`
  - `bo = 1.726611705188328`
  - `chi1 = 1.647988881178246`

### Kinetic profiles near `psi ≈ 0.5`

- Fortran NetCDF at `psi_n ≈ 0.496895187378`
  - `n_i = 3.194418123642e+19`
  - `n_e = 4.083275812429e+19`
  - `T_i = 2.164646028603e+03 eV`
  - `T_e = 1.872041706824e+03 eV`
  - `omega_E = 4.579980843680e+04`
  - `omega_N = 4.745787308235e+03`
  - `omega_T = 1.207642945890e+04`
  - `omega_trans = 1.491847421884e+05`
  - `omega_gyro = 8.095516536651e+07`

- Julia at `psi = 0.5`
  - `n_i = 3.188737313432e+19`
  - `n_e = 4.075763516356e+19`
  - `T_i = 3.452068336578e-16 J`
  - `T_e = 2.989799048767e-16 J`
  - `omega_E = 4.557929988554e+04`
  - `omega_N = 4.694366895122e+03`
  - `omega_T = 1.198293172301e+04`
  - `omega_trans = 1.482168235333e+05`
  - `omega_gyro = 8.269579661463e+07`

### Perturbation flux functions near `psi ≈ 0.5`

- Fortran NetCDF at `psi_fcgl ≈ 0.499630746913`
  - `sqdBoB_L_SA_fcgl = 1.519761433145e-06`
  - `sqdivxi_perp_SA_fcgl = 5.716745258843e-06`

- Julia reconstructed from loaded splines at the same `psi`
  - `|dB/B|^2 surface average = 1.557895284507e-06`
  - `|div xi_perp|^2 surface average = 5.777292074906e-06`

## Current Verdict

- The comparison target is now properly defined.
- Primary parity checks should use the Fortran NetCDF file, not the terminal log.
- With the correct variable pairing, `fcgl` is already reasonably close on both:
  - local kernel: `tpsi!` vs `imag(dTdpsi_fcgl)`
  - cumulative profile: `tintgrl_lsode([0, psi])` vs `imag(T_fcgl)`
- The remaining open question is mainly the final total normalization / output convention, not a gross mismatch in `Input.jl`.

## Electron Scan

Variant case:

- `src/Pentrc/tmp/runtime_fcgl_electron_case`
- same inputs as the ion case, except:
  - `electron = .true.`

### Commands

Fortran:

```bash
/Users/iseonjae/Desktop/GPEC/bin/pentrc
```

Julia:

```bash
PENTRC_CASE=src/Pentrc/tmp/runtime_fcgl_electron_case PENTRC_ELECTRON=true julia --project src/Pentrc/tmp/fcgl_runtime_probe.jl
```

### Fortran Runtime Summary

Observed terminal reference:

```text
 psi = 0.1  -> T_phi =   0.00E+00  5.15E-02j
 psi = 0.5  -> T_phi =   0.00E+00  2.17E-01j
 psi = 0.9  -> T_phi =   0.00E+00  1.08E+00j
 Total Kinetic Energy =  7.710E-001
```

NetCDF total:

- `dW_total_fcgl = 7.710435007776e-01`

### NetCDF-Based Electron Comparison

Local kernel: Julia `tpsi!` vs Fortran `imag(dTdpsi_fcgl)`

- `psi = 0.099462539092`
  - Fortran `imag(dTdpsi_fcgl) = 2.375490119066e-01`
  - Julia `tpsi! = 2.381668195268e-01 i`

- `psi = 0.485948450687`
  - Fortran `imag(dTdpsi_fcgl) = 8.412658136523e-01`
  - Julia `tpsi! = 8.296593787563e-01 i`

- `psi = 0.894490985540`
  - Fortran `imag(dTdpsi_fcgl) = 3.385901141156e+00`
  - Julia `tpsi! = 3.380797380622e+00 i`

Cumulative profile: Julia `tintgrl_lsode([0, psi])` vs Fortran `imag(T_fcgl)`

- `psi = 0.099462539092`
  - Fortran `imag(T_fcgl) = 4.092380036977e-02`
  - Julia cumulative = `3.951675863797e-02 i`

- `psi = 0.485948450687`
  - Fortran `imag(T_fcgl) = 1.814152770951e-01`
  - Julia cumulative = `1.885209445032e-01 i`

- `psi = 0.894490985540`
  - Fortran `imag(T_fcgl) = 1.037580923122e+00`
  - Julia cumulative = `9.597996968261e-01 i`

Julia probe total on `[0, 1]`:

- `tintgrl_lsode([0, 1.0], ..., electron=true) = 1.476760700850e+00 i`

### Electron-Case Interpretation

- The electron scan shows the same broad pattern as the ion scan:
  - local `fcgl` kernel matches closely
  - cumulative profile is still reasonably close
- Local errors remain small, at roughly the percent level.
- The cumulative mismatch grows somewhat more near the edge for electrons.
- This again points more toward radial accumulation / final-output convention than a broken local `calculate_fcgl` kernel.
