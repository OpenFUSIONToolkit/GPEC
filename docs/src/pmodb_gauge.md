# Field strength and displacement gauge

**Verified at Julia base 0304fd54c.** `xi_s` is the surface displacement of
Glasser (2016), PoP 23 072505, Eq. 9:

```math
\xi_s=\chi_1(q\xi^\theta-\xi^\zeta),\qquad
\chi_1=2\pi\psi_o,\qquad
\xi^\alpha=\xi_s/\chi_1.
```

Eq. 18 describes elimination of this displacement in the Euler-Lagrange
system. Calling it a toroidal displacement confuses it with `xi^zeta`.
The corrected docstrings preserve the definition and numerical behavior.
See `src/PerturbedEquilibrium/FieldReconstruction.jl` and the conversion to
`clebsch_alpha` before the geometric matrix products in
`src/KineticForces/KineticForcesStructs.jl:407`.

**Verified.** The kinetic matrix input uses the perpendicular displacement:
`dbob=delta B_L(xi_perp)/B=-(divx+kapx)`
(`src/KineticForces/KineticForcesStructs.jl:435`). The curvature matrices
supply `kapx=kappa.xi_perp`, and the divergence matrices supply `divx` after
removing their `J B^2` weight. FortSym G7 and G8 verify these normalizations,
including the `xi_s/chi1` conversion. The paired `dbob,divx` is the correct
contract for the explicit Jacobian term in the general kinetic action;
it is not by itself the input to a code that accepts only perturbed-Boozer
field strength. Park (2011), PoP 18 110702, Eqs. 4, 10 and 11, and Logan
and Park (2013), PoP 20 122507, Eqs. 4 and 5, give those contracts.

The CGL path separately used `kapx=-(dbob+divx)/2` at
`src/KineticForces/Torque.jl:295`. **Verified defect:** the extra half is
inconsistent with the matrix input. Its correction and independent
pressure-work regression are a separate topic branch.

For comparison, Fortran GPEC's `pmodb` lagb is

```math
b_g=\delta B_E+\xi^\psi\partial_\psi B+
       \frac{\xi_s}{\chi_1 q}\partial_\theta B,
\qquad b_g-b_\perp=-\frac{\xi_\perp^\zeta}{q}\partial_\theta B.
```

It uses `xi_g^zeta=0` in the working chart; `jac_out` only relabels
harmonics. Fortran diagnostic llagb is `b_perp=delta B_E+xi_perp.grad B`.
Under ideal induction and equilibrium force balance,
`b_perp=-B(div xi_perp+kappa.xi_perp)`. With a force residual
`F=(curl B) cross B-mu0 grad p`, the force-balance curvature representation
instead leaves `-xi_perp.F/B`. Regularized numerical outputs need not be
an exact ideal-induction pair near a rational surface.

**Verified target-gauge equations.** In angles in turns put
`D=partial_theta+q partial_zeta`. On a matching equilibrium chart
`J B^a=H(psi)`, `a=0` for Hamada and `a=2` for Boozer. Given one consistent
pair `b_X,d_X`, form `T=d_X+a b_X/B` and solve

```math
D w=-T,\qquad w_m=\frac{iT_m}{2\pi(m-nq)},\qquad
b_{target}=b_X+w\partial_\theta B.
```

A nonzero resonant source has no periodic solution. For a Hamada scalar
already relabeled onto Boozer equilibrium angles, `T=2 b_H/B`; this is the
rmp_torque c3614334 Eq. 4.14 kernel. Relabeling the equilibrium chart and
changing the perturbed displacement gauge are distinct operations.
For PEST, `J/R^2` is a flux function and its linearized constraint is
`div xi-2 xi_R/R`; there is no automatic Boozer/Hamada interpretation.

**Verification:** run
`PYTHONPATH=$FORTSYM/python python3 gen/pmodb_gauge/derive.py`.
All accepted identities use FortSym's native zero test (ae635c80), with
no numerical fallback. `fortsym_verdicts.json` records the exact labels.
The optional direct curvilinear contraction G6 remains UNKNOWN for the
large rational metric jet; C1/C2 prove the same invariant relation on
Cartesian jets and G3-G5 map it to the columns. Negative controls have
exact rational nonzero witnesses. #84/#85/#86 are available; accepted
proofs do not depend on the open coefficient/radical gaps #91/#92.

**Hypothesis limit:** the original ITER edge discrepancies cannot be
quantitatively assigned to force error, interpolation or regularization
without an independent equilibrium-residual calculation. The pressure
coefficient near 0.168 is explained numerically by a missing division by B
in the old fitting regressor, plus the omitted parallel gauge term;
the correct pressure coefficient is one (FortSym P1).
