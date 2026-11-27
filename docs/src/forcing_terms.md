# Forcing Terms

`ForcingTerms` specifies the external magnetic perturbation that drives the perturbed equilibrium: the spectrum of the normal field on the control surface at `psilim`, given either as a file of (m, n) amplitudes or computed by Biot–Savart from coil geometry.
The `[ForcingTerms]` section is read when `[PerturbedEquilibrium]` runs. Every source it holds (a `forcing_data_file`, coil sets, magnetic materials) drives the plasma, and their spectra are summed into the `ForcingField` that `perturbed_equilibrium` applies to the plasma response.

## Normalization

Internally every forcing is the unit-norm flux spectrum ``\Phi_x``, the Fourier amplitude of ``2\pi R\,(B_R\,\partial Z/\partial\theta - B_Z\,\partial R/\partial\theta)`` over the unit-norm angles ``\theta, \zeta \in [0, 1]``, the quantity the Fortran GPEC calls `Phi_x`.
A forcing file declares its own normalization by a tag; a coil set needs none, because the Biot–Savart pipeline produces ``\Phi_x`` directly.

| Tag | Quantity | Units | On load |
|---|---|---|---|
| `"normal_field_T"` (default) | Fourier modes of ``B\cdot\hat n`` over ``\theta \in [0, 2\pi]`` | T | weighted by the area element ``J\lvert\nabla\psi\rvert`` and projected onto every harmonic of the run; a poloidally varying weight mixes modes, so the file must cover its sidebands (a warning asks for a wider m range when an edge harmonic carries more than 10% of the flux) |
| `"sfl_flux_Wb"` | the straight-field-line flux ``R\,(B_R\,\partial Z/\partial\theta - B_Z\,\partial R/\partial\theta)`` over ``\theta \in [0, 2\pi]`` | T·m² | multiplied by ``(2\pi)^2`` |
| `"unit_flux"` | ``\Phi_x`` itself, as a run records its forcing for replay | T·m² | loaded as is |

## File formats

ASCII (`forcing_data_format = "ascii"`): an optional `# normalization: <tag>` line, comment lines starting with `#`, and one `n  m  real(amplitude)  imag(amplitude)` row per mode; all four columns are required:

```
# normalization: normal_field_T
1  6  1.0e-4  0.0
1  7  5.0e-5  2.0e-5
```

HDF5 (`forcing_data_format = "hdf5"`): datasets `n`, `m`, `amplitude_real` and `amplitude_imag`, all required, with the optional root attribute `normalization`:

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

## Coils

Coil sets compute the field from coil geometry: the plasma boundary is sampled on an `mtheta_coil × nzeta_coil` grid, the Biot–Savart field of every coil set is evaluated there, projected onto the normal flux and Fourier decomposed.
The toroidal angle of each observation point includes the straight-field-line offset ``\nu(\psi, \theta)`` and the helicity ``\mathrm{sign}(B_T)\,\mathrm{sign}(I_p)`` of the equilibrium, without which the decomposition would be wrong.
Each `[[ForcingTerms.coil_set]]` block supplies one coil set, selected by `source`; shifts and tilts (`shiftx/y/z`, `tiltx/y/z`, `n_tilt`, …) apply to every source, which is what the [Error Fields](error_fields.md) stage sweeps.
From `examples/DIIID-like_ideal_example/gpec.toml` (the first block) and the other sources:

```toml
[ForcingTerms]
machine = "d3d"
mtheta_coil = 480

# ASCII .dat file, resolved to {dat_dir}/{machine}_{name}.dat
[[ForcingTerms.coil_set]]
name = "c"
currents = [1000.0, 500.0, -500.0, -1000.0, -500.0, 500.0]

# HDF5 file; `name` selects one set of a multi-set file
[[ForcingTerms.coil_set]]
source = "file"
h5_file = "/path/to/my_coils.h5"
name = "il"
currents = [1e4, 1e4, 1e4, 1e4, 1e4, 1e4]

# Analytic PF hoop; a tilt breaks axisymmetry to give n ≠ 0 forcing
[[ForcingTerms.coil_set]]
source = "pf_hoop"
radius = 1.8
height = 0.9
currents = [5e3]
tiltx = [2.0]

# Analytic window-pane array, by two opposite [R, Z] corners ...
[[ForcingTerms.coil_set]]
source = "window_pane"
n_coils = 8
rz_corners = [[2.2, -0.6], [2.2, 0.6]]
gap_fraction = 0.15
currents = [1e3, -1e3, 1e3, -1e3, 1e3, -1e3, 1e3, -1e3]

# ... or by standoff off the plasma surface at a poloidal angle (degrees)
[[ForcingTerms.coil_set]]
source = "window_pane"
n_coils = 2
standoff = 0.25
poloidal_angle = 0.0
poloidal_length = 1.2
poloidal_tilt = 0.0
gap_fraction = 0.15
currents = [1e4, -1e4]

# Analytic helical array on a circular torus, pitch m_hel/n_hel
[[ForcingTerms.coil_set]]
source = "helical"
R0 = 1.7
a = 0.6
m_hel = 4
n_hel = 1
n_coils = 4
currents = [1e3, 1e3, 1e3, 1e3]
```

A coil HDF5 file stores one subgroup per coil set under a parent group (default `coils`): `<group>/<set>/x`, `y`, `z` as `ncoil × s × nsec` Cartesian metres, an optional `currents` dataset, and the winding multiplier as the attribute `nw`; `convert_coil_dat_to_h5` and `convert_coil_h5_to_dat` translate the ASCII format, `read_coil_h5` and `write_coil_h5` read and write it.

The field is linear in the coils, so a `CoilForcingGrid` built once per equilibrium and n evaluates coil sets one at a time, for comparing coil variants against one plasma without re-sampling the boundary:

```julia
forcing_grid = CoilForcingGrid(equil, cfg, n; psi=ffs.psilim)
spectra = [coil_forcing_modes(cs, forcing_grid, n, m_low, m_high) for cs in coil_sets]
```

## Magnetic materials

Each `[[ForcingTerms.magnetic_material]]` block adds a set of ferritic elements, closed triangulated polyhedra read from an HDF5 geometry file (`magnetic_material/<name>/vertices`, `triangles`, `element_offsets`; see `read_magnetic_material_h5`).
Every element is saturated along the equilibrium field at its centroid, ``M = f\,\mu_0 M_s\,\hat B_0/\mu_0``, and its field is the exact field of a uniformly magnetized polyhedron: a surface charge ``M\cdot\hat n`` on each face, each triangle integrated in closed form (Van Oosterom & Strackee, IEEE Trans. Biomed. Eng. 30, 125 (1983)).
That field is sampled on the same `mtheta_coil × nzeta_coil` boundary grid as the coils.
The equilibrium must carry its flux map outside the plasma (a g-file or IMAS), every vertex must lie inside that map's grid and outside the separatrix, and no control-surface point may lie inside an element.

```toml
[[ForcingTerms.magnetic_material]]
name = "iter_inserts"          # set to read from the geometry file
h5_file = "iter_inserts.h5"    # magnetic_material/<name>/ geometry
saturation_field = 1.7         # μ₀M_s, in tesla
fill_fraction = 1.0            # magnetic fraction of each element's volume
```

## Forcing sources

Each coil set, forcing file, spectrum and magnetic material set is one forcing source, and the run records each source's contribution beside the total.
Give several `[[ForcingTerms.coil_set]]` or `[[ForcingTerms.magnetic_material]]` blocks the same `group = "<label>"` to report them as one source named by the label, for example the seven region sets of the test blanket modules as `group = "tbm"`; a block without `group` is its own source, named after the set.
From the API, `ForcingField(materials; groups=[...])` takes one label per material set.

## Outputs

The applied spectrum is written with the response: `PerturbedEquilibrium/ForcingModes/` (`m`, `n`, `amplitude`) and the control-surface spectra `PerturbedEquilibrium/forcing_b`, `forcing_b_root_area` and `forcing_b_area` (see [Perturbed Equilibrium](perturbed_equilibrium.md)).
Each source's root-area-weighted field and its overlap with the dominant modes go to `PerturbedEquilibrium/ForcingSources/` (`source_name`, `forcing_b_root_area` on `(mode, source)`, `dominant_forcing_overlap` on `(rank, source)`); their columns sum to the totals, and the torque response's `T_source` uses the same sources.
A forcing file is captured raw with its tag under `Input/RawInputs/ForcingTerms/`, and the coil geometry as used, after shifts and tilts, under `Input/RawInputs/Coils/`, so that `main(["gpec.h5"])` can replay the run without the original files; see the [Scripting API](api.md).

## API reference

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.ForcingTerms]
```
