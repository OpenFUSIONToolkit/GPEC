Put relevant data for various tests here, most likely outputs from the Fortran code used to validate Julia outputs

# TODO: store as hdf5 files instead?

## regression_rampup_fixed_axis/

Fixture for `test/runtests_fixed_axis.jl`: a synthetic, diverted DIII-D-like ramp-up
slice (TokaMaker free-boundary solve, 129×129 geqdsk, q0 = 3.42, q95 = 5.9, no wall,
n = 1) on which the Frobenius free-axis start (`fixed_axis = false`) produces a spurious
`et[1]` ≈ −1e5 while the DCON fixed-axis start (`fixed_axis = true`, the default) gives
`et[1]` = +1.50 and a plasma-matrix stiff eigenvalue of +1.609e4, matching Fortran DCON
(v1.5.5-323) on the 257×257 parent equilibrium. `gpec.toml` carries the production settings
(mpsi 128, mtheta 256, mthvac 480, `set_psilim_via_dmlim = true`); the test injects
`fixed_axis` itself. Analytic profiles, not experimental data.
