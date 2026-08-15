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
