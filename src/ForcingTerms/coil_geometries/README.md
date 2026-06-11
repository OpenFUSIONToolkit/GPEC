# Coil Geometry Files

3D coil winding geometry files for various fusion devices, ported from the
Fortran GPEC coil library.

## File Format

Each `.dat` file uses the Fortran GPEC ASCII format:
- **Header line**: `ncoil  s  nsec  nw`
  - `ncoil`: number of coils in the toroidal array
  - `s`: number of strands per coil
  - `nsec`: number of cross-section points per strand
  - `nw`: number of winding turns
- **Data lines**: `ncoil × s × nsec` rows, each with `X  Y  Z` in meters (Cartesian lab frame)

All `ncoil` conductors are stored explicitly (the Biot-Savart computation iterates over
every one); a toroidal array is written out as its full set of conductors, not a single
representative coil.

## HDF5 format

Coil geometry can also be stored in HDF5, which is the modern format and the one used for
the self-contained `gpec.h5` rerun snapshot. One file holds one **subgroup per coil set**
under a parent group (default `coils`), and any unrelated content in the file is ignored:

```
<group>/<set_name>/x         Float64[ncoil, s, nsec]   Cartesian metres
<group>/<set_name>/y         Float64[ncoil, s, nsec]
<group>/<set_name>/z         Float64[ncoil, s, nsec]
<group>/<set_name>/currents  Float64[ncoil]            (optional)
<group>/<set_name>  attr nw  Float64                   (winding multiplier; default 1.0)
```

The bundled device library is kept as `.dat` files (human-readable and diff-able in git);
use `convert_coil_dat_to_h5` / `convert_coil_h5_to_dat` to move between formats.

## Naming Convention

Files are named `{device}_{coil_name}.dat`:
- `d3d_c` / `d3d_il` / `d3d_iu` — DIII-D C-coil, I-coil lower/upper
- `nstx_*` / `nstx_u*` — NSTX and NSTX-U coils
- `iter_*` — ITER ELM/VS/blanket coils
- `kstar_*` — KSTAR IVCC/FEC coils
- `jet_*` — JET EFCC/saddle coils
- `mast_*` / `mastu_*` — MAST and MAST-U coils
- `aug_*` — ASDEX Upgrade B-coils
- `east_*` — EAST RMP coils
- `compass_*` — COMPASS saddle coils
- `rfxmod_*` — RFX-mod coils
- `jtext_*` — J-TEXT coils
