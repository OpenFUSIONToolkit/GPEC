# ForcingTerms Module

The ForcingTerms module specifies external magnetic field perturbations for the perturbed
equilibrium calculation. It supports reading from ASCII or HDF5 files, as well as computing
perturbations directly from coil Biot-Savart calculations.

## Normalization conventions

Two normalization tags are supported, specified inside the file (not in `gpec.toml`):

| Tag | Quantity | Units | Notes |
|-----|----------|-------|-------|
| `"normal_field_T"` | Fourier modes of B·n̂ | Tesla | **Default** — most intuitive for users |
| `"sfl_flux_Wb"` | Fourier modes of R×(B_R ∂Z/∂θ − B_Z ∂R/∂θ) | T·m² | 2π-angle SFL flux (user input) |

### Angle convention

Julia's ForcingTerms use the **unit-norm angle convention** (θ_norm ∈ [0, 1], ζ_norm ∈ [0, 1])
internally, matching Fortran GPEC and the Equilibrium/ForceFreeStates modules.

The coil Biot-Savart pipeline produces mode amplitudes directly equal to Fortran's `Phi_x`:

```
Julia output ≡ Fortran Phi_x
```

The formula is `Phi_x(θ_norm, ζ_norm) = 2π × R × (B_R ∂Z/∂θ_norm − B_Z ∂R/∂θ_norm)`, where
the `2π` comes from the toroidal Jacobian ∂r/∂ζ_norm = 2π·R·ê_φ.

Users who provide forcing files may use the more intuitive **2π-angle Fourier amplitudes**
(e.g. the amplitude of cos(mθ − nζ) where θ ∈ [0, 2π]). These are tagged `"sfl_flux_Wb"`
and multiplied by (2π)² internally to reach unit-norm convention.

### Automatic conversion

Files tagged `normal_field_T` are automatically converted to unit-norm Phi_x convention
during loading using the plasma boundary geometry. The conversion accounts for the
R-variation of the Jacobian and is a mode-mixing operation.

Files tagged `sfl_flux_Wb` (2π-angle SFL flux) are scaled by (2π)² to reach unit-norm.

The coil Biot-Savart code (`forcing_data_format = "coil"`) always produces unit-norm
amplitudes directly — no conversion is applied.

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
3. `project_normal_flux!`: compute Phi_x = 2π×R×(B_R ∂Z/∂θ_norm − B_Z ∂R/∂θ_norm)
4. `fourier_decompose_bn`: Fourier decompose to get bmn in unit-norm (= Phi_x) convention

The toroidal angle for each observation point at SFL grid coordinate (θ_i, ζ_j) is:
```
φ_phys(i, j) = -helicity × (2π × ζ_j + ν(ψ, θ_i))
```
where `helicity = sign(Bt) × sign(Ip)` and `ν(ψ, θ)` is the SFL toroidal coordinate correction
stored in `equil.rzphi_nu`. This matches Fortran's `phi = -helicity*(twopi*czeta + crzphi_f(3))`.
For DIII-D (Bt < 0, Ip > 0 → helicity = -1), ν ranges from about -3.4 to +3.4 radians at the
boundary — omitting it completely scrambles the Fourier mode decomposition.

## Comparison with Fortran GPEC

| Quantity | Julia convention | Fortran GPEC convention |
|----------|-----------------|------------------------|
| Angle range | θ_norm ∈ [0, 1], ζ_norm ∈ [0, 1] | θ_norm ∈ [0, 1], ζ_norm ∈ [0, 1] |
| Coil forcing output | unit-norm = 2π×R×(B_R ∂Z/∂θ_norm − B_Z ∂R/∂θ_norm) | `Phi_x` [Wb] |
| Ratio | Julia output ≡ Fortran Phi_x (directly comparable) | — |

## API Reference

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.ForcingTerms]
```
