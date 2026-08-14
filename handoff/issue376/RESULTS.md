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
  Practical corollary: the default 1e-10 tolerance looks overly tight — 1e-8 halves EL cost on
  this case with no change in et[1].

## Implications / ranked follow-ups

1. **Use the two-pass auto grid** (`mpsi=0`, `psi_accuracy`) — already the example default; it
   minimizes knots for a target accuracy, which this data shows is exactly what minimizes EL time.
2. **Integrator order**: Vern9's 16 stages/step buy nothing when steps can't span a knot
   interval. A 5–7th order method (Tsit5/Vern6/Vern7) may cut RHS evals per accepted step ~2×
   at equal nstep. Cheap experiment, no physics change.
3. **Smoother coefficients**: fit F/K/G with smoothing (or higher-continuity) splines instead of
   pure interpolation to lift the knot-scale noise floor that pins the step size.
4. **Node-noise floor**: tightening the equilibrium field-line integration tolerance (`etol`)
   reduces the noise that fine grids amplify — may decouple nstep from mpsi at high mpsi.
5. Spline-eval micro-optimizations (`precompute_transpose!`, extra hints, merged Series) are
   **not worth pursuing** for this issue — measured flat/marginal.
