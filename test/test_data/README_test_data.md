Put relevant data for various tests here, most likely outputs from the Fortran code used to validate Julia outputs

# TODO: store as hdf5 files instead?

## `TJ_circular_axis_newton_{regression,cycling}.geqdsk`

Two 257×257 geqdsks written by the TJ code (headers dated 2026-09-04 and 2026-09-05): synthetic
circular equilibria, R₀ = 2 m, B = 12 T, two points of one β scan. Their ψ has a curvature spike in
the grid cells at the magnetic axis (∂²ψ/∂R² ≈ 13–19 against ≈ 3–10 on the neighbouring cells), a
defect of how the file maps flux coordinates onto the R–Z grid near r = 0. They exercise the
fallback path of the magnetic-axis search in `runtests_equil_axis_newton.jl`; they are not smooth
equilibria and should not be used as physics references.
