# Fourier modes in the CGL torque path

`KineticForcesInternal.mfac` is a vector of mode indices. At base 0304fd54c,
`src/KineticForces/Torque.jl:291` calls scalar `exp` on this vector and
raises `MethodError` before evaluating torque. Broadcast the exponential
so each mode receives its complex Fourier phase.

The regression in `test/runtests_kinetic.jl` calls the actual CGL path with
two modes and an independent zero-curvature double-adiabatic pressure-work
oracle. `gen/cgl_fourier_modes.py` proves that oracle with a FortSym native
zero test. The CGL curvature coefficient is a separate correction; the
zero-curvature fixture tests this runtime defect independently of it.
