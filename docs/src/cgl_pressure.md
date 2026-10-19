# CGL pressure work and field-strength convention

Verified: `dbob=delta B_L(xi_perp)/B=-(D+K)`, with
`D=div xi_perp` and `K=kappa.xi_perp`. Thus `K=-(dbob+D)`.
Recovering K with an extra factor of one half is inconsistent with the input
contract (`pentrc/inputs.f90:856` or Julia
`src/KineticForces/KineticForcesStructs.jl:437`).

The double-adiabatic pressure response is `dp_perp/p=-D+dbob` and
`dp_parallel/p=-3D-2dbob`. Its work is
`W/p=-Re[(dp_parallel-dp_perp)*conj(dbob)+dp_parallel*conj(D)]/(2p)`.
FortSym proves this equals `(5|D|^2/3+|D+3K|^2/3)/2`.
The additional half in the poloidal integrand is the real-Fourier-mode
quadratic normalization and remains in place.

Run `PYTHONPATH=$FORTSYM/python python3 gen/cgl_pressure.py` using a
native FortSym build (verified at ae635c80). No numerical zero test is used.
The regression exercises the actual CGL torque against this independent
pressure closure, including a complex perturbation with nonzero divergence
and curvature. The general kinetic calculation already uses the consistent
perpendicular field-strength/Jacobian pair; this fix affects CGL only.

References: Logan and Park, PoP 20 122507 (2013), Eqs. 4-5; Park,
PoP 18 110702 (2011), Eqs. 4, 10-11. Neither perpendicular field strength
nor the working-chart lagb is automatically a perturbed Boozer scalar.
