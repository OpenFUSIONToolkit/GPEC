# ForcingTerms Module

The ForcingTerms module specifies external magnetic field perturbations for the perturbed
equilibrium calculation. It supports reading from ASCII or HDF5 files, as well as computing
perturbations directly from coil Biot-Savart calculations.

## Normalization conventions

Two normalization tags are supported, specified inside the file (not in `gpec.toml`):

| Tag | Quantity | Units | Notes |
|-----|----------|-------|-------|
| `"normal_field_T"` | Fourier modes of B·n̂ | Tesla | **Default** — most intuitive for users |
| `"sfl_flux_Wb"` | Fourier modes of R×(B_R ∂Z/∂θ − B_Z ∂R/∂θ) | T·m² | Julia-native; output of coil code |

### Angle convention (important!)

Julia uses the **2π-angle convention** throughout: θ ∈ [0, 2π] and ζ ∈ [0, 2π].
This differs from Fortran GPEC, which uses unit-normalized angles θ_norm ∈ [0, 1],
introducing an extra (2π)² factor in the Fortran Phi_x quantity:

```
Fortran Phi_x ≈ (2π)² × Julia sfl_flux_Wb
```

This (2π)² factor is internally consistent within Julia — both the coil forcing code and the
permeability matrix use the same 2π-angle convention, so it cancels in the plasma response.
Users specifying modes in `normal_field_T` should use **2π-angle Fourier amplitudes**
(e.g. the amplitude of cos(mθ − nζ) where θ and ζ are both in [0, 2π]).

### Automatic conversion

Files tagged `normal_field_T` are automatically converted to `sfl_flux_Wb` during
loading using the plasma boundary geometry. This conversion accounts for the R-variation
of the Jacobian and is a mode-mixing operation. The coil Biot-Savart code (`forcing_data_format = "coil"`) always produces `sfl_flux_Wb` directly — no conversion is applied.

## ASCII file format

```
# normalization: normal_field_T
# Format: n  m  real(amplitude)  imag(amplitude)
1  6  1.0e-4  0.0
1  7  5.0e-5  2.0e-5
```

- The `# normalization: <tag>` line is optional; default is `normal_field_T`.
- Lines starting with `#` are comments and are ignored by the data parser.
- Columns: toroidal mode n, poloidal mode m, real part of amplitude, imaginary part (optional, default 0).
- Amplitudes are in Tesla (for `normal_field_T`) or T·m² (for `sfl_flux_Wb`).

## HDF5 file format

Required datasets:
- `n`: integer array of toroidal mode numbers
- `m`: integer array of poloidal mode numbers  
- `amplitude_real`: float array of real parts
- `amplitude_imag`: float array of imaginary parts (optional)

Optional root attribute:
- `normalization`: string, either `"normal_field_T"` (default) or `"sfl_flux_Wb"`

Example (Julia):
```julia
using HDF5
h5open("forcing.h5", "w") do f
    f["n"] = [1, 1, 1]
    f["m"] = [5, 6, 7]
    f["amplitude_real"] = [1e-4, 2e-4, 1e-4]
    f["amplitude_imag"] = [0.0, 0.0, 0.0]
    HDF5.attrs(f)["normalization"] = "normal_field_T"
end
```

## Coil Biot-Savart format

Set `forcing_data_format = "coil"` in `[ForcingTerms]` TOML to compute the field
directly from coil geometry. Output is always in `sfl_flux_Wb` convention.

The coil pipeline:
1. `sample_boundary_grid`: evaluate plasma boundary geometry at (mtheta × nzeta) grid
2. `compute_biot_savart_boundary!`: compute B at all grid points via Biot-Savart law
3. `project_normal_flux!`: compute flux element R×(B_R ∂Z/∂θ − B_Z ∂R/∂θ)
4. `fourier_decompose_bn`: Fourier decompose to get bmn in sfl_flux_Wb convention

The toroidal grid direction follows the Fortran convention: `phi_j = -helicity × 2π×j/nzeta`
where `helicity = sign(Bt) × sign(Ip)`, derived from `equil.params.bt_sign` and
`equil.params.crnt`. For DIII-D standard operation (Bt < 0, Ip > 0): phi increases
with j (same as Julia's default positive direction).

## Comparison with Fortran GPEC

| Quantity | Julia convention | Fortran GPEC convention |
|----------|-----------------|------------------------|
| Angle range | θ ∈ [0, 2π], ζ ∈ [0, 2π] | θ_norm ∈ [0, 1], ζ_norm ∈ [0, 1] |
| Coil forcing output | `sfl_flux_Wb` = R×(B_R ∂Z/∂θ − B_Z ∂R/∂θ) [T·m²] | `Phi_x` [Wb] with unit-angle Jacobian |
| Ratio | Julia sfl_flux_Wb ≈ Fortran Phi_x / (2π)² | — |

## API Reference

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.ForcingTerms]
```
