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

--- Phase 0 (route (a) plan): Fill Site B natural experiment, written before the runs ---

Fill Site A (DirectEquilibrium equilibrium_solver; efit + sol) resamples each surface from the ODE
solver's OWN adaptive step points, so the remap error is uncorrelated between neighbouring
surfaces. Fill Site B (InverseEquilibrium; lar/chease/tj_analytic) derives its SFL abscissae from a
cumulative integral of a spline on a UNIFORM theta grid, so its abscissae vary smoothly with psi.

Running the mpsi ladder on LAR_ideal_match_test (eq_type = "tj_analytic" -> Fill Site B):

H-A "the adaptive abscissae are the noise source":
  - LAR nu/offset/rcoords psi-direction fit residuals FALL with mpsi (converge), r1 positive.
  - contrast: Fill Site A residuals are flat (nu near-axis 2.41e-5 / 2.11e-5 / 2.12e-5).

Then the step-count half decides what route (a) can be worth:
  - LAR eps clean AND LAR accepted steps flatten with mpsi -> noise drives the step growth;
    route (a) is strongly motivated.
  - LAR eps clean BUT LAR steps still grow ~1.7-1.9x per mpsi doubling -> even a perfect route (a)
    cannot decouple nstep; the larger issue is NOT a noise problem and route (a) is a correctness
    fix only.
  - LAR eps floored -> the noise is not specific to Fill Site A's construction; stop and
    re-diagnose before touching src/.

Caveat: LAR is a different equilibrium (analytic large-aspect-ratio, grid_type "ldp", mtheta=512,
populate_dense_xi=false), so only the SCALING with mpsi is comparable to the DIIID/Solovev ladders,
never the absolute magnitudes.

Phase 0 RESULT (LAR, Fill Site B): geometry residuals converge (nu mid 1.25e-7 -> 3.25e-8 ->
5.24e-9, r1 rising +0.95 -> +0.99) and accepted steps are nearly FLAT: 591 / 672 / 739 for
mpsi 256/512/1024 (ratios 1.14x, 1.10x) vs Fill Site A's 2309 / 3977 / 7638 (1.72x, 1.92x).
et[1] identical to 8 digits across the ladder.

CONFOUND, acknowledged before drawing conclusions: LAR also uses grid_type="ldp" (packs toward
rationals, NOT toward the axis) and psilow=0.01 instead of 1e-4, so it largely avoids the
near-axis region where 2/3 of the Fill Site A steps live. Section 5's uniform-grid probe already
showed that un-packing the axis alone cuts near-axis steps 1891 -> 559 at fixed mpsi. So flat LAR
scaling could be a grid-packing effect rather than a geometry-cleanliness effect.

CONTROL: run the SAME DIII-D geqdsk through Fill Site B via eq_type = "efit_by_inversion",
holding grid_type/psilow/psihigh/mtheta and everything else identical to the m*_e1e-10 baselines.
  - If geometry residuals converge AND steps flatten -> geometry cleanliness is the cause,
    confound eliminated, route (a) strongly motivated.
  - If geometry converges but steps still grow ~1.7-1.9x -> the LAR flatness was grid packing;
    route (a) is a correctness fix only and will not decouple nstep.

Phase 0 CONTROL RESULTS:
  - efit_by_inversion (same DIIID equilibrium+grid, Fill Site B): steps 2058/3629/7116, ratios
    1.76x/1.96x -- scales just as badly as Fill Site A. Its geometry is NOT clean either (nu mid
    residual RISES 1.30e-6 -> 3.17e-6 -> 5.57e-6): marching-squares contour extraction is its own
    per-surface noise source. So fill site alone does not determine cleanliness.
  - Solovev (ANALYTIC input, Fill Site A): nu residual dead flat at 0.19 with r1 = -0.65, i.e.
    pure white noise, with a perfectly smooth input. This removes the input variable and confirms
    Fill Site A's adaptive-abscissae trace+remap manufactures the noise by itself.

Across all four cases geometry cleanliness and step flatness correlate perfectly (only LAR is
clean, only LAR is flat). But LAR is ALSO the only case without axis packing (ldp, psilow=0.01).

DECIDING TEST: give LAR -- the cleanest geometry available -- a DIIID-style axis-packing grid
(grid_type="auto", psilow=1e-4), holding its analytic input and Fill Site B path.
  - steps stay flat (~1.1x per doubling) -> geometry cleanliness governs nstep; route (a) is
    strongly motivated and should largely decouple nstep from mpsi.
  - steps grow ~1.7-2.0x -> axis packing governs nstep regardless of cleanliness; route (a)
    cannot solve the larger issue and is a correctness fix only.

DECIDING TEST RESULT: LAR with the DIIID-style axis-packing grid (grid_type="auto", psilow=1e-4)
keeps clean, CONVERGING geometry (near-axis nu residual 8.99e-6 -> 9.95e-8, r1 +0.938 -> +0.984)
and its accepted steps stay FLAT: 753 / 826 / 951 for mpsi 256/512/1024 (ratios 1.10x, 1.15x).
et[1] identical to 6-7 digits across the ladder.

=> Axis packing does NOT cause the step blowup. Geometry cleanliness governs nstep scaling.
   Route (a) is strongly motivated. This also refutes the section 11 speculation that a good share
   of the near-axis steps might be legitimate resolution of real structure, and shows the ~20%
   ceiling was a limit of that PARTIAL repair, not of fixing the source.
