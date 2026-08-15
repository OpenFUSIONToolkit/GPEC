Written BEFORE the runs, 2026-08-15.

H1 "node-noise floor": each flux surface's coefficients are built by an independent field-line
integration to tolerance etol, so node values carry an independent O(etol-ish) error. Refining the
grid does not reduce that error, so the interpolant's derivatives pick it up amplified by 1/dpsi^k.
  - etol A/B at fixed mpsi=512: nstep_total(1e-8) > nstep_total(1e-10) > nstep_total(1e-12).
  - near-axis |f''|/|f| of K,G tracks etol; r1 moves toward -2/3 as etol loosens.
  - ladder: near-axis |f''|/|f| grows ~4x per mpsi doubling (dpsi^-2); r1 stays negative.

H2 "pure cubic-spline C2 structure with exact data": roughness comes only from interpolating an
exactly-known smooth function, so
  - etol A/B: nstep_total independent of etol.
  - ladder: |f''|/|f| converges to the physical curvature; r1 stays positive at all mpsi.

--- Added before the abstol experiment, 2026-08-15 ---

Remedy 0 candidate: DirectEquilibrium.jl:294 traces each flux surface with
  solve(..., reltol=equil_config.etol, abstol=1e-8)
u0 starts at zeros and its components become the nu / eta-offset / r^2 node data measured as
noise-floored in section 10. A hard-coded abstol=1e-8 would bind for components near zero,
which would also explain why the etol (reltol) sweep was null.

If abstol is the binding constraint, tightening it to 1e-14 predicts:
  - nu / offset / rcoords psi-direction fit residuals drop well below their current floors,
    and their r1 turns positive;
  - C/E/H residuals drop toward the A/B/D level; K/G follow;
  - accepted steps at mpsi=1024 (stripped deck) fall well below 7638;
  - et[1] unchanged to ~8 digits and Delta' within ~1e-4 (both were already converged).
If abstol is NOT binding, all of the above stay put.

RESULT: abstol 1e-8 -> 1e-14 is NULL. accepted steps 3977 -> 4071 (mpsi=512),
7638 -> 7474 (mpsi=1024). Same magnitude as the etol sweep. The ODE error control
is not what sets eps; remedy 0 is refuted.

--- Added before the capped-differentiation experiment ---

The amplification, not the origin of eps, is the fixable part: eps is fixed-amplitude in psi,
and all the damage comes from differentiating it on an ever-finer psi grid. So compute the
psi-derivative channel (fx1/fx2/fx3 -> g11/g12/g31) at a resolution the data actually supports,
while leaving values and theta-derivatives untouched.

PoC: per theta column, build a cubic interpolant of r^2 / offset / nu over a psi SUBGRID
(every k-th node, k = mpsi/256) and take its derivative at every node. Values, theta-derivatives,
jac and the coordinate mapping are unchanged.

Predicts, at mpsi=1024 stripped: accepted steps fall from 7638 toward the ~2300-4000 band of the
coarser grids; et[1] unchanged to ~6-8 digits; near-axis K r1 turns positive and C/E/H residuals
drop toward A/B/D. If the psi-derivative amplification is NOT the mechanism, steps stay ~7638.
