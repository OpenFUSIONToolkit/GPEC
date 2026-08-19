# Issue #376 investigation results — why does large mpsi slow ForceFreeStates?

All runs on `examples/DIIID-like_ideal_example` (ideal path, `kinetic_factor=0`), branch
`performance/regression-harness-threading` @ e7bb7cb8, single Julia thread, Vern9,
`eulerlagrange_tolerance = 1e-10` unless noted. "Warm" numbers are the second `main()` call in
one process (JIT excluded).

## Conclusion

**Spline evaluation cost is NOT the mechanism.** The hinted lookup and the series-spline kernel
are knot-count-independent, exactly as the issue hoped. The slowdown is entirely **adaptive step
count**: the integrator's accepted step size is slaved to the *local knot spacing* (median
dψ ≈ 0.12·knotΔ, invariant across mpsi), so `nstep` scales with the number of knots in the
regions where steps concentrate — dominated by the near-axis region where log-family grids pack
hardest. Per-step cost is flat (~6 ms, Npert-sized LAPACK dominates).

**Why the step size follows the knot spacing — partial answer (§8–§11).** The C/E/H coefficient
matrices carry a ~1e-5 relative node-error floor that does not shrink with mpsi, inherited from the
independently-traced surface geometry (ν, offset, r²) through the ψ-derivative channel that builds
g11/g12/g31. The spline interpolates it exactly, contributing ~ε/Δ² to the integrand's second
derivative. So the integrand is genuinely *not* the same function at every mpsi.

**Partial repairs recover little; fixing the source is what matters.** §11 caps the ψ-derivatives
of the already-noisy data and §13 corrects one component of e_ψ — the first buys 11–20% of the
steps, the second ~2%. Neither removes ε itself.

**§14 shows what removing ε is worth.** Across five ladders varying input, fill site and grid one
at a time, **clean geometry ⟺ flat step count**: where the surface data converges, quadrupling the
knots costs ~25% more steps; where it sits on a floor, 3.3×. An analytic Solovev equilibrium comes
out of the standard construction with `nu` node data flat at 19% relative and r1 = −0.65 — pure
white noise from a perfectly smooth input. And a clean case given the DIII-D axis-packed grid stays
flat (1.10×, 1.15× per doubling), so grid packing is not the driver and those near-axis steps are
noise-chasing rather than physics.

## 1. FastInterpolations microbenchmark (`microbench_mpsi.jl`)

`CubicSeriesInterpolant`, 676 ComplexF64 series (mirrors `fmats/kmats/gmats`, numpert=26),
non-uniform Vector grid, in-place eval, persistent `Ref(1)` hint, 16k queries:

| npts | mono ns/ev | stage ns/ev | no-hint mono | range grid | 3-sep stage | B/eval | MB |
|---|---|---|---|---|---|---|---|
| 33   | 880 | 928 | 890 | 889 | 883 | 0 | 1.4 |
| 129  | 898 | 913 | 915 | 925 | 893 | 0 | 5.6 |
| 513  | 898 | 906 | 930 | 890 | 896 | 0 | 22.2 |
| 1025 | 887 | 889 | 937 | 991 | 1020 | 0 | 44.4 |
| 2049 | 892 | 909 | 948 | 901 | 935 | 0 | 88.8 |

Flat in knot count; hint saves only ~5% here (payload dominates); zero allocations.
Single-series control (`q_spline` analogue): ~2 ns/eval hinted (flat), 5→13 ns unhinted
(the expected O(log n) — still negligible). First-touch lazy `permutedims` ≤ 11 ms one-time.
So `precompute_transpose!`/extra-hint "quick wins" are marginal — not applied.

## 2. End-to-end mpsi ladder (warm runs; `run_one.jl` + `parse_ladder.jl`)

| mpsi | knots | EL int (s) | nstep | ms/step | balloon (s) | et[1] |
|---|---|---|---|---|---|---|
| 128  | 129  | 9.3  | 1190 | 7.8 | 1.7  | 1.391 |
| 256  | 257  | 10.0 | 1753 | 5.7 | 3.7  | 0.376 |
| 512  | 513  | 18.5 | 3010 | 6.2 | 8.2  | 0.783 |
| 1024 | 1025 | 32.4 | 5683 | 5.7 | 18.7 | 0.803 |
| auto | 558  | 16.9 | 2542 | 6.6 | 17.4 | 0.801 |

- EL time ∝ nstep; ms/step flat → per-step cost is mpsi-independent.
- `et[1]` converges to ~0.80 only by mpsi≈1024 on the fixed log grid; the two-pass **auto grid
  reaches the converged value with 558 knots and half the EL time** — it is the right default.
- Ballooning scan is O(mpsi) as expected (out of scope here per discussion).

## 3. Where the steps go (`step_regions.jl`)

Steps binned by ψ (accepted steps, from `integration/psi`):

| region | 128 | 256 | 512 | 1024 | auto |
|---|---|---|---|---|---|
| ψ < 0.1 (axis packing)   | 553 | 964 | 1891 | 3806 | 581 |
| 0.985–0.9935 (edge pack) | 117 | 157 | 217  | 418  | 227 |
| q=2 surface ±            | 63  | 62  | 64   | 67   | 112 |
| q=3 surface ±            | 59  | 60  | 58   | 59   | 87  |

Singular-surface neighborhoods are **flat** (physics-controlled). Growth is where knots pack.

## 4. Step size is slaved to knot spacing (`step_vs_knot.jl`)

Within ψ<0.1: knots 73/145/289 (mpsi 128/256/512) → steps 553/964/1891, and
dψ/knotΔ p25/med/p75 = 0.06/0.12/0.21 **at every mpsi**. If steps were tolerance/physics-limited,
dψ would be mpsi-independent; instead it halves when knot spacing halves (~6.5 accepted steps
per knot interval). Two mechanisms were proposed here originally — C² knot discontinuities of the
cubic spline, and node-noise amplification on fine grids. §8–§10 confirm the second exists and
trace it to its source; §11 and §13 show that *partial* repairs recover only 11–20% and ~2%
respectively; **§14 shows that removing the noise at its source flattens the scaling entirely.**

## 5. Discriminators (mpsi=512)

- **`grid_type="uniform"`** (un-packs the axis): near-axis steps 1891 → **559**, total nstep
  3010 → **2237**, EL ≈ 17.7 s. Steps follow knot density, not axis physics. (et[1]=0.993 —
  uniform grid is physically under-resolved near axis; it's a mechanism probe, not a config
  recommendation.)
- **`eulerlagrange_tolerance = 1e-8`** (100× looser, same packing): nstep 3010 → **1629**
  (near-axis 1891 → 1057) with **identical physics** (et[1] = 0.78286 at both tolerances).
  The step size is genuinely error-controlled, but the error *magnitude* per unit ψ is set by
  knot-scale roughness in the spline coefficients — so dψ ∝ knotΔ at fixed tolerance (§4) and
  ∝ tol^α at fixed grid. This drop was originally read as matching the 9th-order expectation
  (100^{1/9} ≈ 1.7); **§6 measures α properly and it does not** — the effective order here is 7.3,
  outside Vern9's formal band, which is itself part of the mechanism argument.
  (The step count drops 1.88×; §6 shows EL *wall time* does not — there is a step-independent floor.)

## 6. Tolerance sweep (`parse_tolsweep.jl`, mpsi=512 fixed grid, 513 knots)

Five warm runs, one process each, single-threaded, everything but `eulerlagrange_tolerance` fixed.
Δ' deviations are relative to the 1e-12 point: `Δ'diag` = worst per-surface relative error on the
`singular/delta_prime_matrix` diagonal, `Δ'mat` = max elementwise deviation over max|Δ'|.

**Column semantics** (`GeneralizedPerturbedEquilibrium.jl:808-810`): `integration/nstep_total` is
the number of **accepted solver steps** and is the physically meaningful count; `integration/nstep`
and `integration/psi` are the **saved snapshots** (`save_interval = 3`, plus forced saves near
rationals and in the edge band). Rejected steps are never recorded anywhere. The tables below and
in §2–§4 originally conflated the two — "saved" is a proxy for step count, "accepted" is the count.

| tol | accepted | saved | ψ<0.1 (saved) | EL (s) | ms/step | run (s) | et[1] | Δ'diag rel | Δ'mat rel |
|---|---|---|---|---|---|---|---|---|---|
| 1e-6  | 1159 | 929  | 591  | 10.1 | 8.72 | 224.9 | 0.78255904 | 1.7e-02 | 7.5e-03 |
| 1e-7  | 1565 | 1240 | 807  | 12.2 | 7.80 | 207.8 | 0.78255903 | 3.3e-03 | 6.5e-04 |
| 1e-8  | 2116 | 1656 | 1090 | 14.7 | 6.95 | 216.0 | 0.78255906 | 6.9e-04 | 8.1e-05 |
| 1e-10 | 3977 | 3018 | 1893 | 18.9 | 4.75 | 221.8 | 0.78255905 | 6.2e-06 | 4.3e-06 |
| 1e-12 | 9620 | 7297 | 3732 | 32.9 | 3.42 | 234.6 | 0.78255905 | 0        | 0        |

- **`et[1]` is not a discriminator**: identical to 8 significant digits across six decades of
  tolerance. Any tolerance recommendation has to be made on Δ', not on the free-boundary energy.
- **Δ' is the discriminator** and it degrades fast below 1e-8. The worst surface is the q=5 row
  (|Δ'| ~ 1.3e3): −1460 at 1e-6, −1332 at 1e-7, −1321 at 1e-8, −1319.5 at 1e-10, converged at
  1e-12. **1e-8 is the sweet spot** — 1.8× fewer steps than 1e-10 at 7e-4 relative Δ' error.
- **Vern9 never achieves its formal order.** Local exponents of h ∝ tol^α from accepted steps:

  | decade | α | effective 1/α |
  |---|---|---|
  | 1e-6 → 1e-7   | 0.130 | 7.7 |
  | 1e-7 → 1e-8   | 0.131 | 7.6 |
  | 1e-8 → 1e-10  | 0.137 | 7.3 |
  | 1e-10 → 1e-12 | 0.192 | 5.2 |

  A 9th-order method on a smooth integrand should give α = 1/(p+1) = 0.100 (error-per-step) or
  1/p = 0.111 (error-per-unit-step). Measured α is outside both everywhere, and **degrades
  sharply at tight tolerance** — the classic signature of an error floor that is not truncation
  error. Vern7 over the same range gives α = 0.131, comfortably inside its own formal band
  (0.125–0.143): the 7th-order method behaves as advertised, the 9th-order one does not. The
  integrand's effective smoothness, not the method, is what caps the order at ≈ 7–8.
- **EL wall time is sub-linear in step count**: 1e-10 → 1e-8 cuts accepted steps 1.88× but EL time
  only 1.29× (18.9 → 14.7 s). A linear fit gives ~2.7 ms per accepted step plus a **~7 s
  tolerance-independent floor** in the phase (which also contains the parallel-FM BVP setup and
  the serial dense-ξ pass). The nstep → time proportionality of §2 holds across *grids*, not
  across tolerances at fixed grid.
- **Whole-run effect is small on this deck**: total warm run is ~220 s regardless of tolerance —
  EL integration is under 10% of it at mpsi=512 (KineticForces/NTV and PE dominate). Tolerance
  relaxation is worth ~4 s of ~220 s here; it matters where EL dominates (large mpsi, no NTV).

Note: this sweep ran on the current branch tree, which has moved since the §2–§5 ladder
(`e7bb7cb8`, before the on-demand-solution-derivatives merge). The 1e-10 point reproduces §5
within 0.3% on nstep (3018 vs 3010) and 4e-4 relative on et[1] (0.782559 vs 0.78286); conclusions
are unaffected, but do not mix numbers across the two tables.

## 7. Integrator order: Vern9 → Vern7 (mpsi=512, same grid)

Working-tree swap of all six `Vern9()` call sites inside `ForceFreeStates` (`EulerLagrange.jl:766`
and `Riccati.jl:1080,1443,1455,1515,1525`); the equilibrium-side Vern9 uses were left alone so the
EL phase is isolated. Δ' deviation is again relative to the Vern9 1e-12 point of §6. **The `src/`
edits were reverted after measurement — this branch touches no source.**

| method | tol | accepted | saved | EL (s) | ms/step | et[1] | Δ'diag rel |
|---|---|---|---|---|---|---|---|
| Vern9 | 1e-10 | 3977 | 3018 | 18.9 | 4.75 | 0.782559048 | 6.2e-06 |
| Vern7 | 1e-10 | 5355 | 4163 | 15.3 | 2.86 | 0.782559048 | 2.5e-04 |
| Vern9 | 1e-8  | 2116 | 1656 | 14.7 | 6.95 | 0.782559057 | 6.9e-04 |
| Vern7 | 1e-8  | 2927 | 2317 | 10.9 | 3.72 | 0.782559050 | 2.5e-04 |

- The §2 prediction holds: Vern7 takes ~40% more steps but each step is **1.7× cheaper**
  (10 stages vs 16), so EL wall time drops ~20% at 1e-10 and ~26% at 1e-8.
- **Vern7 at 1e-8 dominates Vern9 at 1e-8** on both axes — 10.9 s vs 14.7 s *and* 2.5e-4 vs 6.9e-4
  Δ' error. Against the current deck setting (Vern9 @ 1e-10) it is 42% faster for a Δ' error of
  2.5e-4. et[1] is unchanged to 8 digits everywhere.
- **Open question before acting on this**: Vern7's Δ' deviation is 2.47e-4 at 1e-10 and 2.50e-4 at
  1e-8 — i.e. *tolerance-independent*, so it is not step-truncation error. Something in the
  propagator/matching path carries a method-dependent offset that Vern9 does not have. That floor
  should be understood before Vern7 is adopted; it is small, but it is not a convergence error.

## 8. Mechanism: why more knots means more steps (`roughness.jl`)

§2–§7 established *that* dψ ∝ knot spacing. This section establishes *why*, because "the integrand
is the same smooth function, just sampled more finely" is the natural expectation and it is wrong.

Hypotheses written down before the runs (`PREDICTIONS.md` in the scratch dir):

- **H1 node-noise floor** — every flux surface's coefficients are built independently, so node
  values carry a fixed-amplitude error that does *not* shrink with mpsi. Interpolating it gives a
  contribution to f'' of order ε/Δ², growing as the grid refines. Predicts: roughness grows ~4×
  per mpsi doubling, and the knot-to-knot correlation of f'' goes negative.
- **H2 pure cubic-spline C² structure with exact data** — roughness comes only from interpolating
  an exactly-known function. Predicts: f'' converges to the physical curvature, correlation stays
  positive.

### The measurement

`matrices/ideal/F,K,G` are `ffit.fmats_lower/kmats/gmats` — the actual EL integrand splines —
evaluated at their own knots. Second divided differences approximate f''. Two statistics per
region: `r1`, the lag-1 autocorrelation of f'' knot to knot (**+1 = smooth, −2/3 = white noise**),
and median |f''|/|f|, the curvature per unit amplitude. `q(ψ)` on the same grid is the control.

| region | quantity | mpsi=256 | mpsi=512 | mpsi=1024 |
|---|---|---|---|---|
| ψ<0.1  | K: r1 / \|f''\|/\|f\| | −0.42 / 1.01e7 | −0.45 / 4.53e7 | −0.56 / 1.62e8 |
| ψ<0.1  | G: r1 / \|f''\|/\|f\| | +0.12 / 1.16e4 | −0.27 / 4.98e4 | −0.45 / 4.01e5 |
| ψ<0.1  | F: r1 / \|f''\|/\|f\| | +0.91 / 4.48e3 | +0.67 / 4.10e3 | −0.32 / 4.65e3 |
| 0.3–0.7 | F: r1 / \|f''\|/\|f\| | +0.91 / 1.730 | +0.96 / 1.712 | +0.97 / 1.705 |
| 0.3–0.7 | G: r1 / \|f''\|/\|f\| | +0.83 / 3.66 | +0.86 / 3.87 | −0.18 / 3.99 |
| ψ>0.95 | G: r1 / \|f''\|/\|f\| | +0.84 / 7.04e3 | +0.69 / 6.43e3 | −0.13 / 6.56e3 |
| all | **q control r1** | **+0.88…+0.94** | **+0.94…+0.97** | **+0.97** |

**H1 confirmed, H2 refuted.** Near-axis K grows 4.5× then 3.6× per doubling — the Δ⁻² of a
fixed-amplitude node error — while its r1 marches to −0.56, approaching the −2/3 of pure noise.
The q profile on the identical grid stays at r1 ≈ +0.95 throughout, so this is not an artefact of
the estimator or of the graded grid.

The two effects are cleanly separable in the table:

- **Where the grid is still coarse relative to the noise, the physics dominates and converges** —
  mid-plasma F is 1.730 → 1.712 → 1.705, textbook convergence. The user's intuition holds here.
- **Where the grid is fine, the noise dominates and diverges.** The noise floor spreads outward
  with mpsi: at 256 only K near axis is noise-dominated; at 512 G near axis has joined; at 1024
  F near axis, G mid-plasma and everything at the edge have gone negative.

Note the F row near axis: |f''|/|f| is *converged* at ~4.5e3 (that curvature is real, ψ→0 physics)
yet r1 still collapses to −0.32. Magnitude convergence and smoothness are independent — the
integrator responds to the second, which is why "the functions look the same" and "the integrator
needs more steps" are both true at once.

### Where the noise comes from — two knobs ruled out

Both candidate *adjustable* sources were tested at fixed mpsi=512 and both are null.

`etol`, the equilibrium field-line integration tolerance, over four decades:

| etol | accepted steps | et[1] |
|---|---|---|
| 1e-8  | 4038 | 0.781859964 |
| 1e-10 | 3977 | 0.782559048 |
| 1e-12 | 3930 | 0.782601093 |

**2.7% total across 10⁴ in tolerance** — monotonic in the predicted direction but negligible. This
refutes ranked follow-up #4 as originally stated: the floor is not set by an adjustable integration
tolerance, so tightening `etol` cannot buy back the step count.

`mtheta`, the poloidal resolution each surface's metric integrals use:

| mtheta | accepted steps | near-axis K \|f''\|/\|f\| | near-axis K r1 |
|---|---|---|---|
| 256  | 3977 | 4.525e7 | −0.450 |
| 512  | 3966 | 4.502e7 | −0.450 |
| 1024 | 3983 | 4.509e7 | −0.450 |

**Completely flat** — 4× the poloidal resolution changes the roughness in the third significant
figure and the step count by 0.3%. The noise is not poloidal quadrature error either.

The obvious remaining suspect was the input equilibrium — `TkMkr_D3Dlike_Hmode.geqdsk` is a
257×257 reconstruction of ψ(R,Z), a fixed-resolution source that no ψ refinement can resolve away.
**That is also wrong**: the analytic Solovev equilibrium (`eq_type = "sol"`, exact ψ(R,Z), same
stripped treatment, etol 1e-10) shows the same divergence, more strongly if anything —

| mpsi | accepted steps | near-axis K \|f''\|/\|f\| | mid-plasma K \|f''\|/\|f\| | mid-plasma K r1 |
|---|---|---|---|---|
| 256  | 880  | 2.60e2 | 6.42e1 | −0.575 |
| 512  | 1514 | 1.97e3 | 3.90e2 | −0.505 |
| 1024 | 2776 | 1.34e4 | 3.22e3 | −0.581 |

so the floor is generated *inside* the equilibrium → coefficient pipeline, not inherited from the
input data. §9 finds where.

## 9. Where the floor is generated (`roughness.jl` on the primitive matrices)

The residual of a local degree-4 fit over a 9-knot window measures the node-error amplitude
directly: smooth data leaves a residual falling like Δ⁵ (≈32× per mpsi doubling), data on a noise
floor leaves a residual that is flat in Δ. Mid-plasma (ψ = 0.3–0.7), where the physics has no
structure and the diagnosis is cleanest:

| matrix | built from | resid @512 | resid @1024 | r1 @512 | r1 @1024 |
|---|---|---|---|---|---|
| A | g22, g23, g33          | 5.96e-08 | **1.32e-08** | +0.972 | **+0.987** |
| B | g22, g23, g33          | 2.66e-08 | **1.96e-08** | —      | —      |
| D | g23, g33               | 5.92e-08 | **1.28e-08** | +0.972 | **+0.987** |
| C | + g31, jtheta·imat     | 8.90e-06 | 1.21e-05 | +0.694 | **−0.401** |
| E | + g31, jtheta·imat, q1 | 8.74e-06 | 1.17e-05 | +0.547 | **−0.400** |
| H | + g31, jtheta·imat     | 1.11e-05 | 1.41e-05 | +0.837 | **+0.022** |
| F | = F̃ − D†A⁻¹D          | 2.48e-07 | 1.96e-07 | +0.957 | +0.973 |
| K | = E − K†A⁻¹C           | 1.22e-05 | 1.56e-05 | +0.916 | +0.593 |
| G | = H − C†A⁻¹C           | 1.01e-05 | 1.70e-05 | +0.856 | −0.176 |

**The floor enters at C, E and H, at the ~1e-5 relative level, and it is not created by the
derived-matrix algebra.** A, B and D — the matrices built from g22/g23/g33 alone — are clean at
1e-8 *and still converging* (residual halving, r1 rising toward +1) exactly as smooth data should.
F, which is derived through the same `A⁻¹` path as K and G, stays clean at 2e-7 because its inputs
are clean. K and G are dirty only because E, C and H are.

Per `Fourfit.jl:439-447`, the three dirty matrices are precisely the ones that additionally involve
**g31**, **jtheta·imat**, and (for E) **q1 = dq/dψ**; the three clean ones use only g22/g23/g33.
That is the actionable pointer: whatever sets the ~1e-5 floor lives in the construction of those
metric/current quantities, not in the spline layer, not in the EL solver, and not in the input file.

### The causal chain, end to end

1. C/E/H node values carry a **relative error of ~1e-5 that does not shrink with mpsi** (§9).
2. The cubic spline interpolates that error exactly, so the interpolant acquires knot-scale
   wiggles whose contribution to f'' is ~ε/Δ² — growing 4× per mpsi doubling (§8).
3. Once ε/Δ² exceeds the physical curvature, the integrand the solver sees is dominated by
   knot-scale structure. This happens first where the grid packs hardest (near axis at mpsi=256),
   and spreads outward with refinement (edge and mid-plasma by mpsi=1024).
4. The adaptive controller prices exactly that structure, so the accepted step collapses onto the
   knot scale: dψ ∝ Δ, ~6 steps per knot interval, invariant in mpsi (§4).
5. Because the error being controlled is not smooth truncation error, Vern9 cannot realise its
   formal order — measured effective order ≈ 7–8, collapsing to ≈ 5 at tight tolerance (§6) —
   which is also why the cheaper Vern7 loses nothing (§7).

So the premise "it is the same smooth function, just sampled more finely" is true of the *physics*
and false of the *data*. Refining mpsi does not make the integrand a better approximation of a
smooth function below the 1e-5 floor; past that point it makes the interpolant measurably rougher
in exactly the derivative norms the error estimator reads.

### Stripped-deck control

The runs in this section drop the `[ForcingTerms]`/`[PerturbedEquilibrium]`/`[KineticForces]`
sections and set `local_stability_flag = false` (31 s/run instead of 220 s). The slaving is
unaffected — accepted steps 2309 / 3977 / 7638 for mpsi 256 / 512 / 1024, still tracking knot
count — so the mechanism is a property of the EL integration, not of the surrounding pipeline.

## 10. Source of the floor: ψ-derivatives of independently-traced surface geometry

`surface_roughness.jl` on the raw `splines/rzphi` node data (ψ-direction, median over four θ):

| quantity | region | resid @256 | resid @512 | resid @1024 | r1 @1024 |
|---|---|---|---|---|---|
| nu      | ψ<0.1   | 2.41e-05 | 2.11e-05 | 2.12e-05 | −0.744 |
| offset  | ψ<0.1   | 6.39e-04 | 5.37e-04 | 5.60e-04 | −0.671 |
| rcoords | ψ<0.1   | 1.92e-06 | 2.31e-06 | 2.36e-06 | −0.742 |
| nu      | 0.3–0.7 | 1.76e-06 | 1.51e-06 | 1.29e-06 | +0.786 |
| **jac** | 0.3–0.7 | 2.22e-07 | 4.29e-08 | **3.26e-09** | **+0.967** |

`nu`, `offset` and `rcoords` are **flat in Δ** — a genuine floor — while `jac` converges cleanly.
This maps exactly onto the clean/dirty split of §9. In `Fourfit.jl:84-90`, g22/g23/g33 are built
only from **θ-derivatives** (`partials[3]`, a fixed mtheta grid) and are clean; g11/g12/g31 are
built from **ψ-derivatives** (`partials[2]`) of exactly those three floored quantities, so their
error is ε/Δψ and grows as the grid refines.

The 1D profiles are **not** implicated: q, mu0p, 2piF and dVdpsi all have r1 ≥ +0.87 and residuals
that fall with refinement (mid-plasma q: 2.35e-7 → 5.50e-8 → 7.34e-9), so `q1`, `p1` and `jtheta`
are cleared.

Two candidate origins of ε were tested and **both are null**, so the floor is not ODE error control:

| knob | change | accepted steps @512 | @1024 |
|---|---|---|---|
| `etol` (reltol) | 1e-8 → 1e-12 | 4038 → 3930 | — |
| hard-coded `abstol` (`DirectEquilibrium.jl:294`) | 1e-8 → 1e-14 | 3977 → 4071 | 7638 → 7474 |

The `abstol=1e-8` literal looked like a smoking gun — `u0` starts at zeros and its components
become exactly ν/offset/r², so it should bind where `reltol` cannot — but tightening it changes
nothing. Whatever sets ε, it is not the field-line integration's error control.

## 11. Repairing the amplification works — and buys much less than expected

Since ε is fixed-amplitude in ψ, the amplification (not ε's origin) is the fixable part. Probe:
replace **only** the ψ-derivative channel (`fx1/fx2/fx3`) with derivatives of a cubic fit over
every k-th ψ node (k = mpsi÷256), leaving values, θ-derivatives, `jac` and the coordinate mapping
untouched. Working-tree probe, `src/` reverted afterwards.

**The repair does exactly what it was designed to do.** Mid-plasma at mpsi=1024:

| matrix | resid before | resid after | r1 before | r1 after |
|---|---|---|---|---|
| C | 1.21e-05 | **5.30e-07** | −0.401 | **+0.915** |
| E | 1.17e-05 | **5.21e-07** | −0.400 | **+0.881** |
| H | 1.41e-05 | **6.56e-07** | +0.022 | **+0.928** |
| K | 1.56e-05 | **2.58e-06** | +0.593 | **+0.948** |
| G | 1.70e-05 | **9.38e-07** | −0.176 | **+0.942** |

Near-axis likewise: C/E/H/G residuals fall 30–100×, K's |f''|/|f| drops 1.62e8 → 2.10e7, and every
r1 flips positive. The causal claim of §9–§10 — that the roughness enters through the ψ-derivative
channel — is **confirmed**.

**But the step count barely moves:**

| grid | accepted steps, stock | with repair | change | et[1] stock → repaired |
|---|---|---|---|---|
| mpsi=512  | 3977 | 3551 | **−11%** | 0.7825590 → 0.7822223 |
| mpsi=1024 | 7638 | 6081 | **−20%** | 0.8028405 → 0.7998018 |

Cleaning the coefficient matrices by 20–100× recovers only 11–20% of the steps. **So knot-scale
roughness in C/E/H is a contributing cause, not the dominant one** — §9's causal chain overstated
the link from matrix roughness to step count, and the summary and issue comment written from it
need correcting.

What the numbers now say:

- The benefit **grows with mpsi** (−11% at 512, −20% at 1024), exactly as an ε/Δψ amplification
  should. It is real, and it is worth more on finer grids.
- The remaining ~80% is not explained. Note that even after the repair, near-axis
  |f''|/|f| for K is 2.1e7 — a curvature scale of ~2e-4 in ψ, which is the *same order as the
  local knot spacing there* (median Δψ = 1.7e-4 at mpsi=1024). Near the axis the log grid packs
  to roughly the scale of genuine structure, so a large share of those steps may be legitimate
  resolution of real near-axis behaviour rather than noise-chasing. The §5 uniform-grid probe is
  consistent with this: un-packing the axis cut steps 1891 → 559 but left et[1] = 0.993, i.e.
  physically under-resolved.
- The probe also biases the answer slightly (et[1] moves in the 4th digit) because uniform
  every-k subsampling smooths hardest exactly where the log grid packs. A fixed-ψ-scale
  (rather than fixed-node-stride) smoothing width would be the correct design.

## 12. Two structural routes to grid-insensitive matrices (design, not yet implemented)

Post-hoc smoothing of the ψ-derivatives is rejected: it biases the answer (the §11 probe moved
et[1] in the 4th digit) and treats the symptom. Two structural routes remain.

### (a) Make the surface trace correlated in ψ

`equilibrium_solver` (`DirectEquilibrium.jl:494`) loops `for ipsi in (mpsi+1):-1:1` calling
`fieldline_int(psi_nodes[ipsi], …)` with **no state carried between surfaces**. Each surface:
Newton-solves its own start point from a crude guess, integrates θ ∈ [0, 2π] with its **own**
adaptive step locations, then (`:500-525`) builds a periodic cubic spline `ff_interp` on those
solver-chosen SFL-angle nodes and resamples onto the common `theta_nodes`.

That resampling error depends on where the solver happened to place its steps on that particular
surface, so it is **uncorrelated between neighbouring surfaces** — a white-in-ψ error of exactly
the kind §10 measures, and one that tightening `reltol`/`abstol` does not remove (both null, §10),
because it is interpolation error in the remap, not integration error.

Fixes, cheapest first:

- Evaluate the trace **at the common θ abscissae directly** (dense output plus a root-solve for the
  θ where the normalised SFL angle hits each uniform node) instead of splining surface-specific
  nodes and resampling. Every surface is then sampled at the same abscissae, so the remap error
  becomes a smooth function of ψ rather than white noise.
- Weaker variant: continuation — seed each surface's Newton start point and initial step from the
  previous surface, correlating at least the start-point error.

The per-surface endpoint normalisations (`y_out[end, ·]` in `ff_x_nodes` and `ff_fs_nodes[:,3]`)
are also per-surface scalars entering multiplicatively; they deserve the same scrutiny.

### (b) Get the ψ-derivative quantities from metric identities instead

**The codebase already contains a worked example of this.** `DirectEquilibrium.jl:624-626`
computes |∇ψ| for `modB` using **θ-derivatives only**:

```julia
w11 = (1.0 + fy[2]) * (2π)^2 * rfac * r / jacfac
w12 = -fy[1] * π * r / (rfac * jacfac)
delpsi_norm = sqrt(w11^2 + w12^2)
```

whereas `Fourfit.jl:84-90` builds the same class of quantity from `fx1/fx2/fx3` — the ψ-derivative
channel. Two routes to the same geometry, one immune to the ψ grid and one not, and the metric
construction takes the noisy one. This follows from the basis identities: ∇ψ ∝ (∂x/∂θ × ∂x/∂ζ)/J
needs only θ-derivatives, and in axisymmetry the ζ direction is analytic.

Route (b) is therefore: re-derive `metric.fs` channels 1, 5 and 6 (the g11/g31/g12 group) through
the θ-derivative identities, reusing the existing `delpsi_norm` form, so `partials[2]` disappears
from the metric construction. Where a genuine ψ-derivative is unavoidable (∂ν/∂ψ in g31, from the
straight-field-line defining relation), take it from the analytic 2D field representation
evaluated locally rather than by differencing traced surfaces.

**Acceptance test for either route** — the issue's own criterion made quantitative: run the mpsi
ladder (256/512/1024, stripped decks, 31 s each) and require accepted steps to stop tracking knot
count while et[1] and Δ' hold fixed. Diagnostics to confirm the repair took: `surface_roughness.jl`
residuals for nu/offset/rcoords should start falling with Δ instead of sitting flat, and
`GPEC_ROUGHNESS_MATS=A,C,E,H,F,K,G roughness.jl` should show C/E/H converging like A/B/D.

Expected payoff, stated honestly: §11 caps this whole channel at **~20% of the steps at mpsi=1024**
(growing with mpsi). The stronger reason to do it is correctness — coefficient matrices that are
insensitive to grid choice by construction — not speed.

## 13. Route (b) tried: it works, and it does not help — and the numbers say why

Route (b) implemented as a rank-1 constraint projection, not a smoothing. ∇ψ = (e_θ × e_ζ)/J needs
only θ-derivatives (the form already used for `modB` at `DirectEquilibrium.jl:624-626`), and
duality gives **e_ψ·∇ψ = 1 exactly**, so the noisy normal component of the ψ-derivative-built e_ψ
can be replaced by clean data with no free parameters and no bias.

**The identity checks out.** Instrumenting the stock code, mid-plasma `e_ψ·∇ψ` has mean
**1.0000000092** — exact to 8 digits — with pointwise spread **0.9978 … 1.0037**. So the two routes
to the same geometry disagree by up to 0.4% at individual nodes, and the projection removes exactly
that disagreement.

Four variants at mpsi=1024 (stripped deck), isolating each channel:

| variant | what it fixes | accepted steps | et[1] | mid-plasma C / G r1 |
|---|---|---|---|---|
| stock | — | 7638 | 0.8028405 | −0.401 / −0.176 |
| **(b)** duality projection on fx1/fx2 | normal part of e_ψ | **7495 (−1.9%)** | 0.8028194 | −0.401 / +0.261 |
| cap ∂ν/∂ψ only | g31 → C/E/H/K | **7524 (−1.5%)** | 0.7990120 | +0.915 / −0.130 |
| cap all three ψ-derivatives (§11) | everything incl. g11 → G | **6081 (−20%)** | 0.7998018 | +0.915 / +0.942 |

Read together these are conclusive:

- **Route (b) is correct and harmless but worth ~2%.** et[1] moves by 2.6e-5 relative — it is a
  consistency repair, not a bias, unlike the §11 smoothing which moved et[1] in the 4th digit.
- **∂ν/∂ψ is the carrier for C/E/H — and C/E/H are not what the step count cares about.** Capping
  ∂ν/∂ψ alone makes C, E, H and K clean (r1 +0.92/+0.88/+0.93/+0.94, residuals ~5e-7) and buys
  **1.5%**. g31 being the common ingredient of the three dirty matrices, established in §9, turns
  out to be a red herring for cost.
- **The steps care about G, through g11.** Only the variant that also cleans g11 — which is built
  from fx1/fx2 — reaches −20%, and it is the only one where G's r1 goes positive (+0.942).
- **And g11's noise is in the part duality cannot reach.** The projection fixes e_ψ along ∇ψ;
  G improves only slightly (r1 −0.176 → +0.261). What remains is the **tangential** component of
  e_ψ — the θ-parametrization slip between neighbouring surfaces.

**That tangential slip is precisely route (a)'s territory.** It is not geometry, it is a statement
about how each surface's θ parametrization is chosen — and §12(a) shows each surface builds that
parametrization independently, splining on its own solver-chosen SFL-angle nodes before resampling
to the common grid. Duality cannot constrain it because it is coordinate freedom, not metric
content; only making the parametrization consistent across surfaces can.

**Conclusion: (b) is not the fix; (a) is.** Route (b) is still worth keeping as a cheap exactness
improvement (it enforces a constraint currently violated by up to 0.4%), but the ~20% lives behind
the surface-to-surface θ mapping.

## 14. Phase 0 of route (a): geometry cleanliness — not grid packing — governs the step count

Five ladders, chosen to vary one thing at a time. "Geometry ε" is the ψ-direction `surface_roughness.jl`
residual for `nu` (mid-plasma unless noted); "steps" are accepted EL steps at mpsi 256/512/1024.

| case | input | fill site | grid | geometry ε across ladder | steps | ratios |
|---|---|---|---|---|---|---|
| DIII-D `efit` | EFIT g-file | A | packed | flat 1.8e-6 → 1.3e-6 | 2309/3977/7638 | 1.72 / 1.92 |
| DIII-D `efit_by_inversion` | EFIT + contours | B | packed | **rising** 1.3e-6 → 5.6e-6 | 2058/3629/7116 | 1.76 / 1.96 |
| Solovev `sol` | **analytic** | A | packed | **flat 0.19**, r1 = −0.65 | 880/1514/2776 | 1.72 / 1.83 |
| LAR `tj_analytic` | analytic | B | ldp | converging 1.2e-7 → 5.2e-9 | 591/672/739 | **1.14 / 1.10** |
| LAR + packed grid | analytic | B | **packed** | converging 5.9e-7 → 2.9e-8 | 753/826/951 | **1.10 / 1.15** |

Read in order, these isolate the cause:

- **The fill site alone does not decide cleanliness.** `efit_by_inversion` (Fill Site B) is *dirtier*
  than Fill Site A on the same equilibrium — its marching-squares contour extraction is its own
  per-surface noise source — and it scales just as badly.
- **The input alone does not decide it either.** Solovev is analytic and perfectly smooth, yet
  through Fill Site A its `nu` node data is flat at **19% relative** with r1 = −0.65 — pure white
  noise. With the input variable removed, Fill Site A's adaptive-abscissae trace-and-remap
  manufactures the noise by itself.
- **Grid packing is not the driver.** This was the one live confound: LAR was the only clean case
  and also the only one without axis packing. Giving LAR the DIII-D grid (`auto`, psilow = 1e-4)
  keeps its geometry clean and converging (near-axis `nu` 8.99e-6 → 9.95e-8, r1 +0.938 → +0.984)
  and its steps stay flat — **1.10× and 1.15× per doubling against 1.72×/1.92%**. et[1] is identical
  to 6-7 digits across that ladder.

**Conclusion: clean geometry ⟺ flat step count, across two fill sites, two inputs and two grids.**
Where the geometry converges, quadrupling the knots costs ~25% more steps; where it sits on a
floor, it costs 3.3×.

Two things this overturns:

1. **§11's speculation that a good share of the near-axis steps might be legitimate resolution of
   real near-axis structure is wrong.** LAR resolves the same packed near-axis region with flat
   step counts. Those steps are noise-chasing.
2. **The ~20% figure is a ceiling on that *partial* repair, not on fixing the source.** §11 capped
   the ψ-derivatives *of already-noisy data*; §13 corrected one component of e_ψ. Neither removed ε.
   The LAR ladders show what clean source data buys: flat scaling. Extrapolating the DIII-D 256-point
   baseline at LAR-like scaling puts mpsi=1024 near ~2600 instead of 7638.

**Phase 0 verdict: proceed.** The premise of route (a) is confirmed with the input, fill-site and
grid confounds each eliminated by measurement.

## 15. Phase 1 of route (a): the remap is the manufacturer, and the prize is ~50%

**Gate 1a — force dense saves.** `dtmax = 2π/2000` at `DirectEquilibrium.jl:294` shrinks the
spacing of `ff_x_nodes`, leaving tolerances and refinement untouched.

| mpsi | accepted steps | near-axis `nu` resid | near-axis r1 |
|---|---|---|---|
| 512 stock | 3977 | 2.11e-5 (flat) | −0.515 |
| 512 + dtmax | **2521 (−37%)** | 7.42e-7 | **+0.968** |
| 1024 stock | 7638 | 2.12e-5 (flat) | −0.744 |
| 1024 + dtmax | **3614 (−53%)** | 8.22e-8 | **+0.984** |

The geometry flips from a flat floor with white-noise correlation to **converging** data with
r1 → +0.98, and the per-doubling step ratio drops **1.92× → 1.43×**.

**The confound inside Gate 1a is settled.** `dtmax` improves integration accuracy *and* save
density together. The existing etol sweep separates them, at mpsi=512 mid-plasma:

| variant | `nu` resid | `offset` resid |
|---|---|---|
| etol 1e-8 | 1.544e-06 | 9.09e-06 |
| etol 1e-10 | 1.513e-06 | 7.93e-06 |
| etol 1e-12 | 1.769e-06 | 7.04e-06 |
| **etol 1e-10 + dtmax** | **6.36e-08** | **3.39e-07** |

Four decades of integration accuracy move nothing; forcing dense steps at the *same* accuracy
cleans by ~24×. **The error is set by the spacing of the resample abscissae, not by how accurately
each node is computed** — the remap-interpolation signature, and the reason `etol` and `abstol`
were both null.

**Gate 1b did not run.** `save_positions=(true,false) → (false,true)` breaks the periodic closure
— θ=0 is the unrefined initial condition while θ=2π is refined — and the `PeriodicBC` check
rejects it. Its premise was weak regardless (drift ~ reltol = 1e-10, orders below the observed
floor). Recorded as inconclusive, with a constraint for Phase 2: **whatever points are evaluated,
the periodic closure must be preserved.**

**Phase 1 verdict: proceed.** `dtmax` is a brute-force proxy — it forces 2000 steps per surface and
still leaves finite resample error. The Phase 2 fix evaluates *exactly* at the target abscissae, so
the resample error at the output nodes is zero rather than merely small, and it does not force
extra steps.

## 16. Route (a) implemented and measured (`route_a.patch`)

Implementation: `direct_fieldline_int` solves with `dense=true` and returns the solution;
`equilibrium_solver` then root-solves (Brent, bracketed by the monotone `y_out[:,5]`) for the η
where the normalised SFL angle hits each target node and evaluates there, so `ff_fs_nodes` is built
**directly on the uniform θ grid** and every surface is sampled at the same abscissae. The
arclength tracer returns `nothing` and falls back to the old path. Patch saved to
`route_a.patch`; `src/` is reverted (Phase 4 applies it on a branch off `develop`).

### Primary — accepted steps

| mpsi | stock | route (a) | change |
|---|---|---|---|
| 256  | 2309 | 1881 | −19% |
| 512  | 3977 | 2768 | −30% |
| 1024 | 7638 | **4403** | **−42%** |

Per-doubling ratio **1.72 / 1.92 → 1.47 / 1.59**. This clears the plan's "strong win" threshold
(≤5350 at mpsi=1024) and is well past the §11 ceiling that the partial repairs implied.

### Diagnostics

- **Geometry**: near-axis `offset` residual 6.39e-4 → **3.48e-6** at mpsi=256, and every `nu` /
  `offset` / `rcoords` r1 flips positive (e.g. near-axis `nu` −0.345 → **+0.938**). Across the
  ladder the residuals now **converge**: mid-plasma `nu` 2.65e-7 → 6.36e-8 → 6.57e-9.
- **Matrices**: mid-plasma at mpsi=1024, C 1.21e-5 → **1.86e-6** with r1 −0.401 → **+0.856**;
  G 1.70e-5 → 1.29e-6 with r1 −0.176 → **+0.947**. A is unchanged, as expected.
- **Physics preserved**: Δ′ diagonal at mpsi=1024 [8.58, −5.53, −16.09, −2368.85, 55.11] →
  [8.59, −5.51, −16.08, −2367.99, 55.62]; et[1] 0.802840 → 0.799037 (4.7e-3 relative, an accuracy
  change from sampling the geometry correctly). Singular surfaces and `psilim` identical.

### Two results that say the job is not finished

1. **The etol self-consistency prediction failed.** The plan predicted that once the remap error
   was gone, ε would become integration-limited and the `etol` sweep would stop being null. At
   mpsi=512 after route (a): **2857 / 2768 / 2817** accepted steps for etol 1e-8 / 1e-10 / 1e-12 —
   still flat. A residual ε remains that is not controlled by integration tolerance.
2. **The crude `dtmax` probe still beats it** — 3614 vs 4403 at mpsi=1024 (ratio 1.43 vs 1.59) —
   even though every diagnostic available says the two runs are the same: geometry residuals equal
   to 3-4 significant figures (mid `nu` 6.567e-9 vs 6.499e-9), matrix residuals equal (C 1.86e-6
   vs 1.84e-6), profiles equal, singular surfaces and `psilim` identical. Elementwise the two
   geometries differ by only ~1e-6 relative, against the ~1e-4 by which both differ from stock.

**Open question for the next session**: an 18% step difference with no measurable difference in the
integrand. Whatever it is sits outside the node values, the coefficient matrices, the profiles and
the surface positions — the leading suspect is the dense-output interpolant's behaviour *between*
solver steps (route (a) evaluates it; `dtmax` makes the steps small enough that it barely matters),
which the current diagnostics, all built on node values, cannot see. Closing that gap is worth
roughly another 18% on top of route (a)'s 42%.

## 17. Route (a) landed; Phase A diagnostics cancel the planned Phase B

Route (a) is now on `performance/consistent-surface-theta-parametrization` (PR #398), measured
against **current** develop (9491f893d — the earlier harness run used a stale local `develop` at
40f9a9d03, ~20 commits behind; refreshed, conclusions unchanged):

- `diiid_n1`: ODE steps (total) **4572 → 1974 (−56.8%)**, et[1] 0.10%, q0/q95/singular surfaces
  unchanged to 0.00%. Runtime 202.1 → 177.6 s.
- `solovev_n1`: et[1] moves 98% — **because the develop value was grid-dependent**. Same commit,
  only route (a) differing:

  | mpsi | et[1] develop | et[1] route (a) | steps develop | steps route (a) |
  |---|---|---|---|---|
  | 256  | 1.959e-02 | **1.462068e-02** | 1074 | 775 |
  | 512  | 5.399e-02 | **1.462087e-02** | 1558 | 806 |
  | 1024 | 8.890e-02 | **1.462084e-02** | 2689 | 891 |

  develop drifts 4.5× and is still moving; route (a) is converged to 6 significant figures and the
  step ratio is 1.04×/1.11× against 1.45×/1.73×. **For Solovev the issue's goal is met on both
  axes at once.**

### A3 — the kill-switch fires: there is no interpolant error left to remove

Route (a) geometry vs a near-exact trace (`dtmax = 2π/8000`), DIII-D mpsi=512, max relative
difference:

| quantity | ψ<0.05 (280 surfaces) | mid-plasma |
|---|---|---|
| nu      | 5.53e-10 | 1.31e-09 |
| offset  | 2.40e-08 | 7.38e-09 |
| rcoords | 7.30e-10 | 4.63e-09 |
| jac     | 9.92e-10 | 7.45e-10 |

The pre-registered prediction was that route (a) would differ from gold near the axis by **far more
than the integration tolerance**. It does not — it agrees *at* tolerance. **Phase B (reparametrise
onto the SFL angle with `tstops`) would fix an error that is not there, and is cancelled.**

### The step count is hypersensitive, so small step differences are not signal

Two traces differing only by `dtmax = 2π/8000` vs `2π/7900` — geometry identical to **5.3e-11** —
give 2343 vs 2370 steps (1.2%). And the gold trace takes 15% fewer steps than route (a) while
agreeing with it to ~1e-9. So `nstep` is not a smooth functional of equilibrium quality: it responds
to perturbations far below physical significance (et[1] agrees to 8-9 digits throughout). Route
(a)'s −57% is far outside this sensitivity and is real; the leftover 15–20% gaps are its tail.

### A1 — the residual belongs to the traced construction, not to the EFIT input

Same analytic equilibrium (`TJ_ANALYTIC_INPUT`), same grid, both with route (a) applied, only the
construction path differing:

| mpsi | traced (`tj_analytic_direct`) | inversion (`tj_analytic`) |
|---|---|---|
| 256  | 814  | 789 |
| 512  | 997 (1.23×) | 892 (1.13×) |
| 1024 | 1470 (1.47×) | 1056 (1.18×) |

With a perfectly smooth analytic input, the traced path still grows 1.47× while the inversion path
grows 1.18×. So the DIII-D residual is **not** specific to the EFIT file.

### …and it is not trace integration error either

On that same traced analytic case at mpsi=1024, tightening the trace tolerance 1000× is null:
`etol` 1e-10 → 1470 steps, 1e-13 → 1487 steps (+1.2%), et[1] identical to 8 digits.

(Method note: the first attempt at this test was a **no-op** — `examples/LAR_epsilon_scan` has no
`etol` key, so the `sed` matched nothing and all three runs used the default. Bit-identical results
gave it away. The numbers above are from runs with the key actually inserted.)

### Where that leaves the deep core

Geometry accurate to ~1e-9, insensitive to trace tolerance, yet still growing with mpsi on the
traced path but not the inversion path. What distinguishes them is not the *size* of the per-surface
error but its *character*: tracing each surface independently produces an error that is white in ψ
at whatever amplitude it has, and a packed grid amplifies white noise no matter how small. The
inversion path's quadrature error is smooth in ψ and so is not amplified.

If that is right, no amount of per-surface accuracy will flatten the traced path — only making the
error *correlated* across surfaces (continuation) or using the inversion construction will. That is
a hypothesis, not a measurement, and it is where the next session should start.

**Untouched**: A2 (the EL-side error-control question — the solves pass only `reltol`, so `abstol`
sits at the DiffEq default; plus the integration-direction and Frobenius-start questions).

## 18. A2 — the deep-core startup: three levers tested, none flattens the ladder

The hypothesis: near the axis the surfaces are nearly circular/large-aspect-ratio, harmonics
approach the cylindrical limit ξ_m ~ ψ^(|m|/2), so with mpert = 35 the state spans an enormous
range and the controller may be chasing components that are pure noise. Measured on the stored
solution the dynamic range across components is **~1e14** in the core with max|ξ| = 5.85e-4, so the
premise is real. All runs DIII-D, route (a) applied, stripped decks.

**E1 — absolute tolerance bracket** (the EL solves pass only `reltol`; `abstol` sits at the DiffEq
default 1e-6). At mpsi=512:

| abstol | total steps | core (ψ<0.05) | et[1] |
|---|---|---|---|
| 1e-2 (loose) | 2468 | 771 | 0.781587768 |
| 1e-6 (default) | 2768 | 1066 | 0.781587768 |
| 1e-12 (tight) | 3400 | 1615 | 0.781587768 |

**et[1] is identical to 9 digits across four decades** while core steps vary 2.1×, and 91% of the
variation is in ψ<0.05. So the solver *is* spending steps on components that contribute nothing —
the hypothesis is confirmed as a real waste.

**E2 — but it is not the mpsi mechanism.** The same comparison across the ladder:

| abstol | 256 | 512 | 1024 | ratios |
|---|---|---|---|---|
| 1e-6 | 1881 | 2768 | 4403 | 1.472 / 1.591 |
| 1e-2 | 1686 | 2468 | 3884 | **1.464 / 1.574** |

A level shift of ~11% at unchanged physics, and the scaling is untouched.

**E3 — harmonic count.** Cutting `delta_mlow`/`delta_mhigh` 8 → 2 (mpert 35 → 17; this *does*
change the physics, so it is a diagnostic only):

| harmonics | 256 | 512 | 1024 | ratios | core steps |
|---|---|---|---|---|---|
| mpert 35 | 1881 | 2768 | 4403 | 1.472 / 1.591 | 603 / 1066 / 2017 |
| mpert 17 | 1126 | 1526 | 2215 | 1.355 / 1.452 | 224 / 392 / **745** |

High-m harmonics *are* disproportionately expensive in the core — core steps fall 2.7× against a
2.0× fall in the total, and the core's share drops 46% → 34%. But halving the harmonics moves the
ratio only 1.591 → 1.452. **It does not flatten the ladder either**, which is consistent with the
earlier attempt at core harmonic truncation ending up slower than the full solve: the harmonics are
a cost multiplier, not the knot-count mechanism.

### What this leaves

Every lever tried — route (a) (−42%), `abstol` (−11%), harmonics (−50%) — shifts the *level* and
leaves the *scaling* at ~1.45–1.59, while the inversion path reaches 1.18 on the same analytic
equilibrium. The arithmetic that explains it: at mpsi=1024 the near-axis spacing is Δψ ~ 1e-6, so
the ε/Δψ² amplification is ~1e12. Route (a) cut ε from ~1e-4 to ~1e-9 — five orders, hence the big
level drop — but ε is still **white in ψ**, because each surface is still constructed independently.
Shrinking ε lowers the curve; only making ε *smooth in ψ* can change its slope.

### Actionable

1. ~~Free ~11% from an explicit `abstol`~~ — **retracted, see §19.** That figure was measured on
   *step count only*. Wall time does not follow, and on the Riccati path an `abstol` destroys Δ′.
2. **The scaling** needs correlated surface construction — continuation from surface to surface so
   the per-surface error is smooth rather than white — or the inversion construction. Not attempted.

## 19. abstol: tested thoroughly, and it must not ship

**Prior art (found in the history, not in `develop`).** `abstol` was implemented for exactly this
purpose in Jan 2026 — `377d85132 DCON - IMPROVEMENT - Add absolute tolerance to ODE solver to reduce
deep core steps`, issue #122 option (1), with a scale-aware `compute_atol_scale!` sampling solution
magnitudes at half-integer q. It was **never merged**: it lives only on
`origin/claude/issue-122-20260103-1833`. Issue #122 (still open) records the verdict — the maintainer
benchmarked it and concluded *"the user shouldn't be given an `abstol`. Not enough gain in the safe
range to warrant the possibility of being in the bad range."* Their table shows runtime *rising*
with looser atol (150 → 274 → 1547 → 36800 s) and et going to garbage (−775) at 1e6.

So there was no removal to explain: it was tried, measured, and rejected. This section re-tests it on
the current code with the observable that was missing then — **Δ′**.

Setup: DIII-D stripped deck, mpsi=512, route (a) applied. `integrator = "riccati"` (which is what
unlocks `SingularSurfaces/Delta_prime_matrix`) vs `integrator = "forward"`. Δ′ deviations are
relative to that integrator's own no-abstol run.

**Riccati chunks — Δ′ is destroyed, and et[1] hides it completely:**

| abstol | steps | run (s) | et[1] | Δ'diag rel |
|---|---|---|---|---|
| none (default 1e-6) | 1384 | 17.7 | 0.7841145216 | ref |
| 1e-4 | 907 | 17.0 | 0.7840998277 | **4.9e-03** |
| 1e-2 | 679 | 16.4 | 0.7840701367 | **2.54 (254%)** |
| 1    | 535 | 16.6 | 0.7845740708 | **4.68 (468%)** |

Steps fall 51% and **runtime barely moves (−7%)**, while Δ′ — which feeds the tearing calculations —
is wrong by 254% at abstol 1e-2. et[1] moves only 5e-5 relative, so **an et-only acceptance check
would have passed this**. That is the trap the Δ′ test was for.

**Forward integration — safe, but worthless:**

| abstol | steps | core | run (s) | et[1] |
|---|---|---|---|---|
| none (default 1e-6) | 2768 | 1066 | 12.8 | 0.7815877685 |
| 1e-4 | 2575 | 887 | 14.0 | 0.7815877684 |
| 1e-2 | 2468 | 771 | 12.4 | 0.7815877685 |
| 1    | 2282 | 642 | 13.3 | 0.7815877685 |

et[1] identical to 10 digits and core steps down 40%, but **wall time is flat within noise
(12.4–14.0 s)**. Consistent with §6's step-independent floor in that phase: the steps being removed
are cheap ones, so removing them buys nothing.

**Conclusion: no PR.** Riccati is unsafe (Δ′), forward is safe but gains nothing. This independently
reproduces the issue #122 verdict and adds the mechanism it was missing. Route (a)'s step reduction
was different in kind — it *was* confirmed by wall clock (harness runtime 202 → 178 s).

**Corollary for future work**: step count is not a proxy for cost here, and et[1] is not a proxy for
correctness. Any deep-core optimisation must be judged on wall time **and** Δ′.

## 20. Δ′ and wall time for PR #398 (the check that was missing)

The §17 evidence used `diiid_n1`/`solovev_n1`, which run `integrator = "forward"` and therefore
**emit no BVP Δ′ matrix at all** — so route (a) had never been checked against Δ′, the observable
that §19 showed can break while et[1] looks fine. Closed with the dedicated cases.

`diiid_n1_riccati`: Δ′ BVP diagonal **16.73%**, raw side-major 2.35%, delta_coil 1.26%, et[1] 0.10%,
singular surfaces/psi/q unchanged, ODE steps 1649 → 1358, runtime 202.5 → 192.8 s.

`gal_resistive_diiid` — what the tearing/matching path actually consumes:

| quantity | diff |
|---|---|
| gal PEST3 Δ diagonal | **0.01%** |
| ‖gal Δ′ matrix‖ | **0.03%** |
| gal D_I / α per surface | 0.00% |
| ‖gal inner-layer Δ‖ | 0.00% |
| ‖gal Δ_coil block‖ | 0.80% |
| ‖gal match cout‖ | 1.38% |

**The 16.73% is one element.** Δ′ BVP diagonal across an mpsi ladder on the riccati deck:

| mpsi | develop | route (a) |
|---|---|---|
| 256  | [9.7, −1.9, −13.1, **103024**, 336.0] | [9.7, −1.9, −13.1, **102768**, 336.1] |
| 512  | [7.2, −5.2, −16.6, **−1319.5**, 61.4] | [7.2, −5.2, −16.6, **−1293.2**, 61.5] |
| 1024 | [8.6, −5.5, −16.1, **−2368.8**, 55.1] | [8.6, −5.5, −16.1, **−2368.0**, 55.6] |
| drift vs prev mpsi | 174% → 79.5% | 174.7% → 83.1% |

The 4th entry (q=5) runs 1e5 → −1319 → −2369 and **changes sign — it is not converged in mpsi on
either version**, so it cannot discriminate between them. The other four entries agree between
versions at every grid; full-diagonal agreement is 0.25% / 2.0% / 0.03% at mpsi 256 / 512 / 1024.

**Route (a) does not improve Δ′ convergence either** (174.7%/83.1% vs 174.0%/79.5%). The unconverged
q=5 Δ′ element is pre-existing, is arguably more concerning than anything in #398, and deserves its
own investigation — it means BVP Δ′ at that surface is not trustworthy at these resolutions today.

Wall time improves on every case: forward `diiid_n1` 210.6 → 184.5 s, `diiid_n1_riccati`
202.5 → 192.8 s, `gal_resistive_diiid` 216.8 → 213.6 s. (Earlier a single identical "Runtime" row
appeared in both case reports — that was an aggregate; these are per-case and differ.)

## 21. Phase 0 of the alignment work: the hypothesis is refuted, in the opposite direction

The plan was to make the per-surface trace error *smooth in ψ* by aligning step breakpoints between
neighbouring surfaces (continuation). It was gated on a pre-registered check (`PREDICTIONS.md`):
does the traced construction leave geometry measurably **rougher in ψ** than the inversion
construction at the same mpsi? Only a smoothness difference licenses alignment — an amplitude
difference does not, since A3 showed the traced error is already at tolerance and the step count is
hypersensitive to perturbations far below physical significance.

Same analytic equilibrium (`TJ_ANALYTIC_INPUT`), same grid, both with route (a).

**Geometry, mid-plasma fit-residual:**

| quantity | traced 256/512/1024 | inversion 256/512/1024 |
|---|---|---|
| nu | 6.08e-10 / 6.67e-10 / 4.08e-10 | 6.04e-07 / 8.61e-08 / 2.84e-08 |
| jac | 2.62e-10 / 8.93e-10 / 5.06e-10 | 8.09e-07 / 1.13e-07 / 3.27e-08 |

**EL coefficient matrices, mid-plasma (r1 / residual):**

| matrix | traced @1024 | inversion @1024 |
|---|---|---|
| A | +0.978 / 5.84e-10 | +0.958 / 3.43e-08 |
| C | −0.422 / 1.20e-07 | −0.220 / **9.01e-06** |
| E | **+0.949** / 1.18e-07 | **+0.235** / 9.25e-06 |
| H | **+0.971** / 1.49e-07 | **+0.394** / 1.37e-05 |
| G | +0.966 / 4.79e-08 | +0.964 / 3.10e-07 |

**The traced construction is both smoother (higher r1 on A/E/H) and 100–1000× more accurate than
the inversion construction** — and it is the one whose step count scales *worse* (1.47 vs 1.18).
Neither pre-registered criterion is met; the hypothesis fails, and not narrowly.

**Therefore: do not build the alignment/continuation change.** There is nothing to align — the
traced path's coefficients are already the smoother and more accurate of the two. Whatever makes
the inversion path's step count scale better, it is not geometry smoothness.

The one narrow signature worth noting for the record: traced `rcoords` r1 in the **core** degrades
0.959 → 0.708 → **−0.304** across the ladder while inversion improves 0.984 → 0.992 → 0.996. That
is a real per-surface signature in exactly one quantity, in the region where the steps are — but it
sits alongside mid-plasma residuals that are 1000× *better* than inversion's, so it does not support
a general alignment scheme.

### This also weakens §14's A1 inference

A1 concluded "the residual belongs to the traced construction, not the input" from traced 1.47× vs
inversion 1.18× on the same analytic equilibrium. But those two runs give **et[1] = 0.4623 vs
0.4700 — a 1.7% difference**, so the two constructions are not solving quite the same problem to
high accuracy. Their step counts are therefore not a clean apples-to-apples comparison, and A1
supports "the constructions differ" more than it supports "the trace is at fault".

**Net state of the deep-core question:** after route (a), the remaining mpsi-scaling (1.59× on
DIII-D) has **no identified mechanism**. Ruled out by measurement: remap interpolation, EFIT input
resolution, trace tolerance (`etol`, `abstol`), poloidal resolution, EL error control (`abstol` on
either integrator), harmonic count, and now geometry/coefficient smoothness. The honest position is
that the cause is unknown, and the next step should be finding it rather than fixing a candidate.

## 22. Mechanism found and confirmed by intervention: C3 knot jumps, and grid decoupling fixes it

**The metric that finally discriminates** (`jumps.jl`): the solver never sees node values — it sees
the spline between them, and a cubic spline has a discontinuous third derivative at every knot. The
jump is J ~ dpsi*f'''' for perfect data but **J ~ eps/dpsi^3 for node error eps**, and the local
error crossing a C2 kink is ~J*h^4 regardless of method order, forcing h ~ (tol/J)^(1/4). Node
statistics measure eps; the solver feels eps/dpsi^3 — which weights the packed core (dpsi ~ 1e-6)
by ~1e18. That is why every node-smoothness metric failed to discriminate (§21) while the step
counts differed.

Measured on the a1 pair (same analytic equilibrium, both with route (a)):

| | traced | inversion |
|---|---|---|
| median core J, m512 → m1024 | 5.2e9 → **1.8e11** | 3.1e7 → **1.7e7** |
| model-predicted extra core steps | 868 → 1579 | 157 → 93 |
| observed step growth | 997 → 1470 | 892 → 1056 |

Consistency checks: A3's measured eps ~ 1e-9 at core dpsi ~ 1e-6 predicts J ~ 1e9 (matches);
white-noise-floor scaling predicts step ratio 2^(3/4) = 1.68/doubling (pre-route-(a) DIII-D was
1.72/1.92), clean-data scaling predicts ~2^(1/4) = 1.19 (inversion is 1.13/1.18); the dtmax probe,
uniform-grid probe and LAR flatness all fit the same arithmetic.

**The intervention** (pre-registered, `patch_decouple.py`): the EL coefficient splines do not need
the equilibrium's core-packed knots — the coefficients are near-cylindrical there. Build them on a
subset with core density capped at dpsi >= 0.05*psi below psi = 0.1 (513→329 and 1025→575 knots;
values at kept knots unchanged, only density changes).

| | baseline route (a) | decoupled grid |
|---|---|---|
| forward m512 steps / core | 2768 / 1066 | **2188 / 489** |
| forward m1024 steps / core | 4403 / 2017 | **3034 / 626** |
| ratio per doubling | 1.59 | **1.39** |
| forward m512 warm run | 12.8 s | **10.4 s (−19%)** |
| et[1] m512 / m1024 | 0.781587768 / 0.799036879 | 0.781587786 / 0.7990369 (**3e-8 rel**) |
| riccati m512 Δ′ diagonal | ref | **4e-7 rel, all 5 elements** |

Every pre-registered prediction fired: core steps collapsed into the predicted 500–900 band, the
ratio fell to ~1.39, equilibrium time was unchanged, and — far better than predicted — et[1] and
Δ′ are unchanged at the 1e-7–1e-8 level, because the kept node values are identical and the coarse
representation is adequate for near-cylindrical coefficients.

**This is the answer to issue #376's mpsi question in its final form**: the EL integrand inherits
the equilibrium grid's knots, and knot-packed regions amplify tolerance-level node error into huge
third-derivative jumps that slave the step size. Decouple the coefficient-spline grid from the
equilibrium grid and the coupling breaks — with bit-level physics, because decoupling changes only
the interpolation density, not the data.

Remaining 1.39 ratio: the cap only touched psi < 0.1; mid-plasma and edge-packed knots still carry
growing jumps. A production version could apply a matrix-curvature criterion over the whole domain
(the same measured-curvature machinery the auto grid uses), plausibly approaching the clean-data
~1.19 floor.

## 23. Curvature-based knot selection: tried, measured, and the fixed cap wins for the ideal path

Implementation (`phaseA_selection.patch`): removal-error greedy coarsening in `make_matrix` — drop a
knot when a cubic Lagrange through its kept neighbours reproduces its value within
`matrix_grid_tol · max|M|` for **every element of all 12 matrices** (top-K monitoring was tried
first and made no difference — the certificate was already honest), rational Δ′-stencil brackets
and the `RATIONAL_RES_RADIUS` windows mandatory-keep, `matrix_grid_tol` control field.

DIII-D m512, stripped decks, vs the uncapped route-(a) baseline (steps 2768, et[1] 0.781587768;
Riccati steps 1384, et[1] 0.7841145216):

| δ | knots | fwd steps | et[1] rel err | Riccati Δ′ diag rel (excl q=5) |
|---|---|---|---|---|
| 1e-5 | 158 | 1527 | 1.4e-3 | 5.3e-2 |
| 1e-6 | 358 | 2297 | 3.6e-5 | 2.2e-2 |
| 1e-7 | 504 | 2742 | 1.6e-9 | 3.4e-9 |
| 1e-8 | 513 | 2768 | 6e-10 | 0 |
| **fixed cap** | **329** | **2188** | **3e-8** | **4e-7** |

The physics plateau (et[1] ≤1e-6, Δ′ ≤1e-5) is reached only at δ = 1e-7 — where selection keeps
504/513 knots and buys nothing. At δ small enough to matter, physics breaks. **The fixed cap
strictly dominates** (fewer knots, fewer steps, better physics), so per the pre-registered
threshold the general selection is not landed for the ideal path.

(The mpsi ladder at δ=1e-5 did hit ratio 1.07/1.11 — below the clean floor — proving the *steps*
follow the knot count exactly as the mechanism says; the failure is purely that physics-safe
tolerances leave nothing to remove beyond what the cap already removes.)

**Why interpolation error is the wrong objective for the ideal path** — the lesson that matters for
Phase B: the cap removes 184 core knots that *fail* the removal test at 1e-7 (the near-axis K
matrix has real relative curvature ~1e4 there), yet physics is unchanged at 3e-8. The physics does
not need the coefficients resolved where the solution components are negligible (ξ_m ~ ψ^|m|/2).
Conversely, mid/edge knots that *pass* a loose removal test carry structure whose small
interpolation errors amplify ~40–140× into et[1] and Δ′. A correct general certificate would be
**solution-weighted** — coefficient accuracy demanded in proportion to the solution magnitude that
multiplies it — not raw interpolation error. Two further implementation lessons recorded for
Phase B: matrix-scale normalization (`max|M|` global) is distorted when one region dominates the
matrix norm (near-axis K is ~1e4× mid values), and the `ForceFreeStates` module shadows `Base.eps`
with a const `eps = 1e-10`, so never call `eps()` unqualified there.

**Phase A outcome:** the committed fixed cap *is* the ideal-path deliverable. The selection
machinery (patch preserved here) moves to Phase B, where the certificate must be built
solution-weighted from the start and where per-eval cost makes adaptive selection actually pay.

## 24. The cap grounded and generalization-tested (response to the over-fit concern)

The user challenged the fixed cap as a possibly unphysical rule tuned to one case. Three answers,
all by measurement:

**1. The auto-grid no-op is genuine, not a stacking artifact.** The selection experiment *replaced*
the cap (never stacked); the no-op test ran cap-only. Measured directly, the production auto grid's
core spacing sits 1.65–2.93× *above* the 0.05·ψ floor in every decade, so the cap removes nothing
there by construction.

**2. The rule has a physics basis.** Near the axis every component is a Frobenius power law in ψ;
power laws are scale-free, so log-uniform sampling (Δψ ≥ c·ψ) resolves them at constant relative
accuracy. Cubic interpolation of ψ^p on a log-uniform grid errs by ~(p·c)⁴/384, so c = 0.05
resolves even the steepest spectrum component (p = m_max/2 = 11) to ~2e-4 — with the physics far
below that because steep components carry vanishing solution amplitude. The genuinely ad-hoc piece
was the ψ<0.1 region bound; it is now min(0.1, innermost rational − RATIONAL_RES_RADIUS) with
rational windows never coarsened (commit 0c17097b5). Both guards verified as no-ops on all current
cases (DIII-D and Solovev m512 reproduce the prior step counts and knot sets exactly); they exist
for decks whose rationals reach the core (Solovev's q=2 sits at ψ=0.122 — only 22% outside the old
bound — and higher-n decks will cross it).

**3. It generalizes across everything in hand** — four equilibria, two construction paths, three
grid families:

| case | knots | steps | et[1] change |
|---|---|---|---|
| DIII-D efit, traced, log-family m1024 | 1025→575 | −31% | 3e-8 |
| tj_analytic_direct, traced, log-family m1024 | 1025→575 | −35% | 1.3e-9 |
| LAR, inversion, packed auto m1024 | 1025→575 | −8% | exact to 8 digits |
| Solovev, traced, ldp m512/m1024 | 505→468 | ~0 | ~4.5e-7 abs (3e-5 rel via the ±10.4→0.0146 cancellation) |

The LAR row is the sharpest anti-over-fit evidence: its geometry is *clean* — there is no DIII-D
noise structure to exploit — and the cap is still harmless while mildly helping. The Solovev
relative number is cancellation-amplified ~700×; in absolute energy terms it matches the other
cases.

## 25. Phase B0: the kinetic matrices are under-resolved in ψ, and the kernel cost is real

Solovev_kinetic_calculated ladder (mpsi 16/64/256; the shipped deck uses **mpsi = 16** — itself
evidence that kernel cost forces grid starvation). Measured on the h5 totals
(`EulerLagrangeMatrices/Kinetic/*`) with the preserved `Ideal/*` group as in-run control:

| family (total) | ε @m16 | ε @m64 | ε @m256 | r1 @m256 |
|---|---|---|---|---|
| A | 2.5e-4 | 4.2e-5 | 3.0e-6 | −0.08 |
| B | 6.8e-2 | 7.3e-3 | 5.8e-4 | −0.35 |
| C | 2.8e-2 | 2.4e-2 | 5.4e-3 | −0.37 |
| K | 3.3e-3 | 1.2e-3 | 4.2e-4 | −0.06 |
| (ideal A control) | 5.1e-8 | 2.2e-8 | 3.1e-8 | +0.27 |

**Fork resolution — mostly unresolved real structure, not a noise floor.** ε falls steadily with
refinement (B by ~12× per size step), so the dominant error is kinetic ψ-structure the grids do
not resolve; the negative r1 at m256 suggests a white component underneath, floor not yet reached
at these sizes (A is down to 3e-6 and still falling). Consequence: the shipped kinetic deck carries
**percent-level** total-matrix errors — certification is needed for correctness first, cost second.
J on the kinetic totals still explodes with refinement (C: 2.2e3 → 3.9e6 → 6.8e9 relative),
~10³–10⁴× above the same-family ideal values, so the conditioning concern stands alongside.

**Cost anchor**: from the threaded m64→m256 delta, kinetic-matrix formation + EL ≈ 1.3 s/surface
single-thread-equivalent (~80 ms threaded at 16). At mpsi=1024 that is ~20 serial minutes per run
of matrix formation alone — the budget certification competes against.

Observation flagged (not chased): `Kinetic/G` and `Ideal/G`/`Ideal/H` show identical statistics at
every size — plausibly small kinetic corrections on a large shared scale, but worth one look
before building G-based diagnostics.

## 26. Phase B1: batched certified kinetic grid — implemented, knob-off bit-identical

Implementation (cap branch): `certified_kinetic_grid` in `Kinetic.jl` — seed = the ideal
coefficient-spline knots (`ffit.matrix_xs`), batched certify-or-refine rounds, kernel driven over
arbitrary ψ lists (`CalculatedKineticMatrices.jl` `psis` kwarg; the threaded per-surface pattern
untouched), certificate = max-element residual of the spline-predicted **totals** (ideal part
cancels, so only increments are splined, but the tolerance scale is the total's — the user's rule
at increment-only cost), including the adjoint combination `kw₃ − kt₃`; spacing floors = the
Frobenius cap in the core, `RATIONAL_RES_SPACING` outside, and `RATIONAL_RES_SPACING/4` inside
rational windows (anti-aliasing). Control: `kinetic_grid_tol` (default 0 = off, today's behaviour).

Smoke (Solovev calculated, m64, tol 1e-3): knob-off **bit-identical** to the unpatched run;
knob-on: **65-knot seed → 93 knots, 168 kernel evaluations, 28 refined** — the certification
*added* knots at this coarse grid, matching both the pre-registered expectation (kinetic adds to
the ideal-optimal seed) and §25's under-resolution finding. et[1] certified-vs-plain agrees within
the 1e-3 tolerance, as designed.

## 27. Phase B2: tolerance sweep, coarse seeding, and the honest accounting

**Tolerance sweep** (Solovev calculated m64, certified knots/evals → et[1]):
3e-3: 85/158; 1e-3: 93/168; 3e-4: 100/178; 1e-4: 112/197 — et[1] plateau-flat at 1.767261 across
the sweep. The certified answer is internally converged; the residual ~1e-3 difference from the
m256 run is **equilibrium resolution, not kinetic-grid resolution** (the m64 equilibrium itself
differs), so cross-mpsi comparisons conflate the two. The clean accuracy statement is at fixed
equilibrium: certified vs full kinetic grid on the m256 deck agrees to **9.4e-5** on et[1].

**The seed policy is the cost story.** Seeding with the full capped ideal grid made certification
*more* expensive than blanket evaluation at fine grids (484 vs 255 evals at m256 — every certified
interval costs one midpoint evaluation). Fixed by decimating the seed to a coarse skeleton (~50
knots; endpoints and rational windows always retained): m256 now runs **150 evals vs 255 (−41%)**,
and the saving scales with grid size (~6× at m1024-class grids at ~1.3 s/surface serial-equivalent
kernel cost). At coarse decks the ledger goes the other way by design — m64 spends 168 evals vs 65
to *fix* under-resolution (65 → 93 knots), per §25.

**nuzero** (collisionless): certified m64 tracks plain m64 to 1.9e-5; both sit ~5e-4 from the m256
gold — again equilibrium-resolution-dominated at these sizes. The 1e-3 certificate did not trigger
extra resonance refinement on this case; the withheld-seed aliasing stress test has **not** been
run (rational windows are always seeded by construction) and is recorded as outstanding.

**Kinetic-case harness attribution**: vs develop the three `solovev_kinetic_*` cases move
substantially (et[1] 1.7–2.5%, Im up to 23%, NTV torque magnitude, and NTV ψ-quadrature
evaluations 840 → 60 — the torque quadrature was apparently spending 14× the evaluations
resolving geometry noise). Vs the #398 branch: **zero changed quantities (34/34 unchanged)** —
the deltas are inherited entirely from route (a)'s Solovev geometry fix, and this branch is
bit-neutral on the kinetic cases with the knob off.

Status: committed on `performance/decoupled-el-matrix-grid` (2225813fc), **default OFF**
(`kinetic_grid_tol = 0`), knob-off verified bit-identical. Open design questions for review are in
the morning-questions list (seed size constant, default-on policy, kinetic ground-truth case).

## 28. Morning decisions applied: PR split, physics-skeleton seed, and the DIII-D survival test

User decisions (morning after §27): (1) PR #398 must own the kinetic harness moves in its body —
done, with the ntv torque ~2500× move stated plainly and flagged for physics review + re-pinning;
(2) the certified kinetic grid is its own PR — #410, stacked on #408 (branch
`performance/certified-kinetic-grid`; #408's branch trimmed back to the cap commits);
(3+4) the DIII-D example, not Solovev, is the referee — it is fully kinetic-capable (bundled
kinetic h5, forward integrator, mpsi=0 auto grid) and the survival test simply had never been run;
(5) the constant-49 seed was wrong — the user correctly remembered the seed should be the
resonance structure. Reworked (35e4ae593): seed = rational windows ∪ located Ω_ℓ=0 resonance
surfaces (`kinetic_resonance_psi_nodes`, one source of truth with the NTV quadrature paneling)
∪ 4 interior points per inter-rational span (log-spaced in the axis span). Certification
exhaustion now warns loudly. Solovev smoke (m16, tol 1e-3): 15 seed → 74 knots, 133 evals, fully
certified; (6) Kinetic/G vs Ideal/G identical-statistics observation: leave as is.

**DIII-D kinetic-calculated survival test** (pre-registered acceptance in
`run_survival.sh`: et[1] ≤ 1e-4, NTV torque ≤ 1e-2, evals < auto grid, wall ≤ knob-off):
FIRST ATTEMPT KILLED THE DISK — the knob-off run wrote a 9.1 GB gpec.h5 within ~4 minutes and
died on ENOSPC (tmp filesystem hit 100%). The dominant dataset is [TBD from instrumented rerun];
this is itself a finding about the never-before-run configuration.

[RESULTS OF INSTRUMENTED RERUN — TBD]

### Pre-registered decision rule (written before the ladder/discriminator results were known)

The original et ≤ 1e-4 acceptance assumed a trustworthy knob-off baseline; the baseline turned out
to be the pathology itself (223,271 EL steps vs ~2,000 ideal; 17.5 GB of solution output). Revised
referee, fixed in advance:

- Run the certified ladder (tol 1e-3/3e-4/1e-4, max_rounds=10) AND the discriminator: knob-off
  full grid with rtol_xlmda tightened 1e-5 → 1e-6 (kernel node noise ε↓10×, so J ∝ ε drops 10×).
- If knob-off(rtol 1e-6) moves TOWARD the certified plateau: the full-grid baseline was
  noise-biased; certified is vindicated; the PR story is "the full grid needs a ~10× more
  expensive kernel to approach the answer the certificate gets at rtol 1e-5".
- If knob-off(rtol 1e-6) stays near et[1] = 1.0055−0.2594i: certification is smoothing real core
  structure (suspect: the Frobenius floor at ψ≈0.02, derived for ideal ξ ~ ψ^|m|/2, binding both
  uncertified intervals); the floor/seed needs rework and #410 does not ship until fixed.
- A ladder plateau alone proves internal consistency only (shared seed/floors/kernel tolerance),
  never correctness. Floor-certified intervals are certified WITHOUT a residual check — a known
  blind spot recorded here.
- Holds until resolved: no deck enabling, #410 stays draft, no stage-2 torque referee.

### Localization (zero-cost, from the two stage-1 h5 files)

Full-grid vs certified on the shared 288-point output grid, max-element relative residual per
family: A 8.5e-4, B 2.8e-3 — at certificate level. But C/K ~9.8% and G 23% in the deep core
(ψ<0.05; the whole core is one rational-free span, innermost rational at ψ=0.518), and f0 2.1%
at mid-radius ψ≈0.40. The worst core elements are high-m corner modes (34,34) where solution
amplitude ~ψ^|m|/2 is negligible; the mid-radius f0 deviation is in the physically live region.
G's core residuals have lag-1 correlation +0.95 (real steep structure, not noise); K/C core
r1 ≈ −0.2 (noise-like) with certified deviating ~7× above the data's own roughness.
**Candidate certificate-design gap**: f0/K/G are derived (post-Schur) — the 1e-3 certificate on
raw increments amplifies through the derivation (f0: 20×). If the discriminator vindicates the
certified answer this still needs addressing; if not, it is a prime suspect.

## Implications / ranked follow-ups

1. **Use the two-pass auto grid** (`mpsi=0`, `psi_accuracy`) — already the example default; it
   minimizes knots for a target accuracy, which this data shows is exactly what minimizes EL time.
1b. **`eulerlagrange_tolerance = 1e-8`** (the struct default) is the right per-deck value on the
   evidence of §6; the DIII-D decks' 1e-10 and the LAR decks' 1e-12 buy Δ' accuracy the case does
   not need. Any such deck edit is a separate branch and needs the regression harness.
2. **Integrator order** — measured (§7). Vern7 is ~20–26% faster in the EL phase and, at 1e-8,
   strictly better than Vern9 on both time and Δ' accuracy. Blocked on explaining Vern7's
   tolerance-independent 2.5e-4 Δ' offset; then a branch + regression harness.
3. **Fix the ~1e-5 node-error floor in C/E/H** (§9) — this is the root cause, and the only change
   that would genuinely decouple nstep from mpsi rather than rescale it. Start at
   `Fourfit.jl:439-447` and the construction of `g31`, `jtheta`/`imat` and `q1`, since A/B/D
   (g22/g23/g33 only) are clean at 1e-8 and converging. Failing that, fit C/E/H with smoothing
   rather than interpolating splines, so the floor is not differentiated.
4. ~~Node-noise floor via `etol`~~ — **tested and refuted** (§8): 10⁴ in `etol` moves the step
   count 2.7%. Likewise `mtheta` (4× → 0.3%) and the input equilibrium (analytic Solovev shows
   the same divergence). Do not spend time on these.
5. Spline-eval micro-optimizations (`precompute_transpose!`, extra hints, merged Series) are
   **not worth pursuing** for this issue — measured flat/marginal.
