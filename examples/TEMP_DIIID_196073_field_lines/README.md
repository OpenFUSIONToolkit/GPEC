# DIII-D shot 196073 — vacuum / plasma / total field-line tracing

Field-line tracing of the n=1 perturbed field for **DIII-D shot 196073** at three time
slices (2840, 2845, 2850 ms), driven by the shot's I-coil (`iu`, `il`) and C-coil (`c`)
currents. Equilibria are the EFIT g-files and coil currents from the shot's OMFIT save.

## Layout

```
g196073.02840 / .02845 / .02850     EFIT equilibria (one per time slice)
run_all.jl                          driver: generates configs, runs JPEC, makes figures
t<slice>_<source>/                  one directory per (time slice × perturbation source)
    gpec.toml                       generated config
    gpec.h5                         full JPEC output incl. field_line_tracing/ group
    poincare_RZ_<tag>.png           Poincaré section in (R,Z)
    poincare_flux_<tag>.png         Poincaré section in (θ, ψ_N) — island chains
    connection_length_<tag>.png     connection-length / laminar map
    island_widths_<tag>.png         island half-widths + Chirikov overlap
    summary_<tag>.png               2×2 composite of the above
```

9 directories: `{2840,2845,2850} × {vacuum,plasma,total}`.

- **vacuum** — externally applied I-coil+C-coil field only (Φ_x)
- **plasma** — plasma-response part only
- **total** — applied field + plasma response (Φ_tot = P·Φ_x)

## Reproduce

```bash
julia --project=. examples/TEMP_DIIID_196073_field_lines/run_all.jl
```

## Island half-widths (w½ in ψ_N), n·q = m surfaces m = 2,3,4,5

| Slice | vacuum | plasma | total |
|---|---|---|---|
| 2840 ms | 0.0124, 0.0117, 0.0113, 0.0099 | 0.0075, 0.0058, 0.0077, 0.0096 | 0.0099, 0.0102, 0.0083, 0.0037 |
| 2845 ms | 0.0115, 0.0115, 0.0115, 0.0096 | 0.0068, 0.0052, 0.0073, 0.0093 | 0.0093, 0.0103, 0.0089, 0.0035 |
| 2850 ms | 0.0104, 0.0111, 0.0113, 0.0101 | 0.0058, 0.0048, 0.0072, 0.0091 | 0.0089, 0.0100, 0.0088, 0.0046 |

The Chirikov overlap stays well below 1 at all surfaces, so the islands do not overlap
into a stochastic layer for these currents — the Poincaré sections show isolated island
chains on otherwise nested surfaces.

## Notes

- Tracing runs in flux coordinates (`tracing_coords = "flux"`, the interior default), so
  the HDF5 `footprints/` group is empty; divertor footprints require real-space tracing
  with a wall.
- This is a `TEMP_` scratch example (equilibria + generated outputs); it is not intended
  to be committed to the repository.
