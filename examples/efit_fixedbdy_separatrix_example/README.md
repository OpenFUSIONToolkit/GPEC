# Fixed-boundary EFIT separatrix-find fixture (PR #296, issue #294)

`eq_eps0.0500000_k1.000_d0.000.geqdsk` is a fixed-boundary TokaMaker aspect-scan equilibrium
(circular, R/a = 20, eps = 0.05) whose computational box hugs the last closed flux surface.

In a fixed-boundary solve the LCFS shape is prescribed and the flux **outside** it is physically
unconstrained: the coil-vacuum flux turns back above the boundary value (`sibry`) before reaching
the grid edge. The old bracketed `Roots.Brent()` separatrix finder used the grid edge as its outer
bracket point, so `f(axis)` and `f(edge)` ended up the same sign and `direct_position!` threw
`ArgumentError: not a bracketing interval`. The Newton finder (PR #296, faithful to Fortran
`direct.f direct_position`) walks inward to the first `psi = sibry` crossing — the actual LCFS — and
loads it regardless of box size.

This g-file is the fixture for the `efit_fixedbdy_separatrix` regression case
(`regression-harness/cases/efit_fixedbdy_separatrix.toml`, a `kind = "computed"` case that calls
`setup_equilibrium` directly). It is **not** a runnable example deck — there is no `gpec.toml`.

Free-boundary equilibria (real coil currents) and standard diverted EFITs are unaffected: their
outboard midplane flux keeps dropping past the LCFS, so the bracket was never ambiguous.
