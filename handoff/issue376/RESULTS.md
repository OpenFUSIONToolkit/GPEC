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
per knot interval). Consistent with (i) C² knot discontinuities of cubic splines that a 9th-order
error estimator can't step across cheaply, and (ii) node-noise amplification on fine grids
(the h⁻⁴ effect warned about in `src/Equilibrium/GridRefinement.jl`).

## 5. Discriminators (mpsi=512)

- **`grid_type="uniform"`** (un-packs the axis): near-axis steps 1891 → **559**, total nstep
  3010 → **2237**, EL ≈ 17.7 s. Steps follow knot density, not axis physics. (et[1]=0.993 —
  uniform grid is physically under-resolved near axis; it's a mechanism probe, not a config
  recommendation.)
- **`eulerlagrange_tolerance = 1e-8`** (100× looser, same packing): nstep 3010 → **1629**
  (near-axis 1891 → 1057) with **identical physics** (et[1] = 0.78286 at both tolerances).
  The ~1.85× drop matches the 9th-order expectation (100^{1/9} ≈ 1.7): the step size is genuinely
  error-controlled, but the error *magnitude* per unit ψ is set by knot-scale roughness in the
  spline coefficients — so dψ ∝ knotΔ at fixed tolerance (§4) and ∝ tol^{1/9} at fixed grid.
  (The step count halves; §6 shows EL *wall time* does not — there is a step-independent floor.)

## 6. Tolerance sweep (`parse_tolsweep.jl`, mpsi=512 fixed grid, 513 knots)

Five warm runs, one process each, single-threaded, everything but `eulerlagrange_tolerance` fixed.
Δ' deviations are relative to the 1e-12 point: `Δ'diag` = worst per-surface relative error on the
`singular/delta_prime_matrix` diagonal, `Δ'mat` = max elementwise deviation over max|Δ'|.

| tol | nstep | tried | ψ<0.1 | EL (s) | ms/step | run (s) | et[1] | Δ'diag rel | Δ'mat rel |
|---|---|---|---|---|---|---|---|---|---|
| 1e-6  | 929  | 1159 | 591  | 10.1 | 10.89 | 224.9 | 0.78255904 | 1.7e-02 | 7.5e-03 |
| 1e-7  | 1240 | 1565 | 807  | 12.2 | 9.84  | 207.8 | 0.78255903 | 3.3e-03 | 6.5e-04 |
| 1e-8  | 1656 | 2116 | 1090 | 14.7 | 8.85  | 216.0 | 0.78255906 | 6.9e-04 | 8.1e-05 |
| 1e-10 | 3018 | 3977 | 1893 | 18.9 | 6.26  | 221.8 | 0.78255905 | 6.2e-06 | 4.3e-06 |
| 1e-12 | 7297 | 9620 | 3732 | 32.9 | 4.51  | 234.6 | 0.78255905 | 0        | 0        |

- **`et[1]` is not a discriminator**: identical to 8 significant digits across six decades of
  tolerance. Any tolerance recommendation has to be made on Δ', not on the free-boundary energy.
- **Δ' is the discriminator** and it degrades fast below 1e-8. The worst surface is the q=5 row
  (|Δ'| ~ 1.3e3): −1460 at 1e-6, −1332 at 1e-7, −1321 at 1e-8, −1319.5 at 1e-10, converged at
  1e-12. **1e-8 is the sweet spot** — 1.8× fewer steps than 1e-10 at 7e-4 relative Δ' error.
- **Step count scales as tol^{−1/6.7}** over the full range (929 → 7297 for 10⁶ in tolerance),
  slightly steeper than the ideal 9th-order tol^{−1/9}, as expected when the error being
  controlled is knot-scale roughness rather than the smooth truncation term. The rejected-step
  fraction is flat (0.20–0.24), so this is not a step-control artefact.
- **EL wall time is sub-linear in nstep**: 1e-10 → 1e-8 cuts accepted steps 1.82× but EL time only
  1.29× (18.9 → 14.7 s). A linear fit gives ~2.7 ms per attempted step plus a **~7 s
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

## Implications / ranked follow-ups

1. **Use the two-pass auto grid** (`mpsi=0`, `psi_accuracy`) — already the example default; it
   minimizes knots for a target accuracy, which this data shows is exactly what minimizes EL time.
1b. **`eulerlagrange_tolerance = 1e-8`** (the struct default) is the right per-deck value on the
   evidence of §6; the DIII-D decks' 1e-10 and the LAR decks' 1e-12 buy Δ' accuracy the case does
   not need. Any such deck edit is a separate branch and needs the regression harness.
2. **Integrator order**: Vern9's 16 stages/step buy nothing when steps can't span a knot
   interval. A 5–7th order method (Tsit5/Vern6/Vern7) may cut RHS evals per accepted step ~2×
   at equal nstep. Cheap experiment, no physics change.
3. **Smoother coefficients**: fit F/K/G with smoothing (or higher-continuity) splines instead of
   pure interpolation to lift the knot-scale noise floor that pins the step size.
4. **Node-noise floor**: tightening the equilibrium field-line integration tolerance (`etol`)
   reduces the noise that fine grids amplify — may decouple nstep from mpsi at high mpsi.
5. Spline-eval micro-optimizations (`precompute_transpose!`, extra hints, merged Series) are
   **not worth pursuing** for this issue — measured flat/marginal.
