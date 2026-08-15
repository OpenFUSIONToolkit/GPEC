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

**And the reason the step size follows the knot spacing (§8–§9):** the C/E/H coefficient matrices
carry a ~1e-5 relative node-error floor that does not shrink with mpsi. The spline interpolates it
exactly, contributing ~ε/Δ² to the integrand's second derivative — growing 4× per mpsi doubling
until it swamps the physical curvature, first where the grid packs hardest. The integrand is
therefore *not* the same smooth function at every mpsi: below the 1e-5 floor, refining the grid
makes the interpolant measurably rougher in exactly the norms the error estimator reads.

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
cubic spline, and node-noise amplification on fine grids. **§8–§9 test both: the second is right
and the first is not the driver.**

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
