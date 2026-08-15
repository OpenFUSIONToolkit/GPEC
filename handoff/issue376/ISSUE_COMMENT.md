# DRAFT — proposed comment for issue #376 (not posted; awaiting user review)

---

## Answer: spline evaluation is not the bottleneck — adaptive step count is

I measured this end to end on `examples/DIIID-like_ideal_example` (ideal path, `kinetic_factor = 0`),
single-threaded, Vern9, `eulerlagrange_tolerance = 1e-10` unless stated otherwise. Short version:

**The hint-based spline lookup already does exactly what the issue hoped — evaluation cost is
flat in knot count. There is nothing left to win there.** The mpsi slowdown is entirely
**adaptive step-count growth**: the integrator's accepted step size is slaved to the *local knot
spacing*, so `nstep` scales with the number of knots in the regions where steps concentrate.

### 1. Spline evaluation is knot-count-independent

`CubicSeriesInterpolant`, 676 `ComplexF64` series (mirrors `fmats`/`kmats`/`gmats` at `numpert = 26`),
non-uniform `Vector` grid, in-place eval, persistent `Ref(1)` hint, 16k queries:

| npts | mono ns/ev | stage ns/ev | no-hint mono | range grid | 3-sep stage | B/eval |
|---|---|---|---|---|---|---|
| 33   | 880 | 928 | 890 | 889 | 883 | 0 |
| 129  | 898 | 913 | 915 | 925 | 893 | 0 |
| 513  | 898 | 906 | 930 | 890 | 896 | 0 |
| 1025 | 887 | 889 | 937 | 991 | 1020 | 0 |
| 2049 | 892 | 909 | 948 | 901 | 935 | 0 |

Flat from 33 to 2049 knots, zero allocations. The hint buys only ~5% here because the 676-series
payload dominates the lookup entirely. A single-series control (the `q_spline` analogue) runs
~2 ns/eval hinted and 5→13 ns unhinted across the same range — the expected O(log n), and
negligible. So `precompute_transpose!`, extra hints, and merged `SeriesInterpolant`s were measured
and **dropped**: there is no knot-length sensitivity left in the evaluation path to remove.

### 2. What actually grows: the number of ODE steps

mpsi ladder, warm runs (second `main()` call in one process, JIT excluded):

| mpsi | knots | EL integration (s) | nstep | ms/step | ballooning (s) | et[1] |
|---|---|---|---|---|---|---|
| 128  | 129  | 9.3  | 1190 | 7.8 | 1.7  | 1.391 |
| 256  | 257  | 10.0 | 1753 | 5.7 | 3.7  | 0.376 |
| 512  | 513  | 18.5 | 3010 | 6.2 | 8.2  | 0.783 |
| 1024 | 1025 | 32.4 | 5683 | 5.7 | 18.7 | 0.803 |
| auto | 558  | 16.9 | 2542 | 6.6 | 17.4 | 0.801 |

EL time ∝ nstep, and **ms/step is flat** — per-step cost is mpsi-independent (Npert-sized LAPACK
dominates it, not interpolation). The whole slowdown is the step count.

### 3. The steps pile up where the knots pack, not where the physics is

Accepted steps binned by ψ:

| region | 128 | 256 | 512 | 1024 | auto |
|---|---|---|---|---|---|
| ψ < 0.1 (axis packing)   | 553 | 964 | 1891 | 3806 | 581 |
| 0.985–0.9935 (edge pack) | 117 | 157 | 217  | 418  | 227 |
| q = 2 surface ±          | 63  | 62  | 64   | 67   | 112 |
| q = 3 surface ±          | 59  | 60  | 58   | 59   | 87  |

Singular-surface neighbourhoods are **mpsi-flat** — those steps are physics-controlled, as they
should be. All the growth is in the near-axis region where the log-family grid packs hardest.

Within ψ < 0.1, knot counts 73/145/289 (mpsi 128/256/512) give 553/964/1891 steps, and the ratio
dψ/knotΔ has p25/median/p75 = 0.06/0.12/0.21 **at every mpsi** — about 6.5 accepted steps per knot
interval, invariant. If steps were tolerance- or physics-limited, dψ would be mpsi-independent;
instead it halves whenever the knot spacing halves.

Two discriminators confirm the mechanism at mpsi = 512:

- **`grid_type = "uniform"`** (un-packs the axis): near-axis steps 1891 → **559**, total nstep
  3010 → **2237**. Steps follow knot density, not axis physics. (This is a mechanism probe, not a
  config recommendation — the uniform grid is physically under-resolved near axis, et[1] = 0.993.)
- **`eulerlagrange_tolerance = 1e-8`** (100× looser, same grid): nstep 3010 → **1629** with
  identical physics (et[1] = 0.78286 at both). The 1.85× drop matches the 9th-order expectation
  (100^(1/9) ≈ 1.7).

So the step size is genuinely error-controlled — but the error *magnitude* per unit ψ is set by
knot-scale roughness in the interpolated F/K/G coefficients. Hence dψ ∝ knotΔ at fixed tolerance
and dψ ∝ tol^(1/9) at fixed grid. Physically this is (i) the C² knot discontinuities of a cubic
spline, which a 9th-order error estimator cannot cheaply step across, plus (ii) node-noise
amplification on fine grids (the h⁻⁴ effect already warned about in `GridRefinement.jl`).

### 4. Tolerance sweep — how tight does `eulerlagrange_tolerance` actually need to be?

Five runs at mpsi = 512 (513 knots), everything but the tolerance held fixed. Δ' deviations are
relative to the 1e-12 point; `Δ'diag` is the worst per-surface relative error on the
`singular/delta_prime_matrix` diagonal.

| tol | nstep | attempted | ψ<0.1 steps | EL (s) | et[1] | Δ'diag rel |
|---|---|---|---|---|---|---|
| 1e-6  | 929  | 1159 | 591  | 10.1 | 0.78255904 | 1.7e-02 |
| 1e-7  | 1240 | 1565 | 807  | 12.2 | 0.78255903 | 3.3e-03 |
| 1e-8  | 1656 | 2116 | 1090 | 14.7 | 0.78255906 | 6.9e-04 |
| 1e-10 | 3018 | 3977 | 1893 | 18.9 | 0.78255905 | 6.2e-06 |
| 1e-12 | 7297 | 9620 | 3732 | 32.9 | 0.78255905 | 0 (ref)  |

Three things worth knowing:

- **et[1] is not a discriminator** — identical to 8 significant digits across six decades of
  tolerance. Δ' is what actually degrades, and it degrades fast below 1e-8 (the worst surface is
  the q = 5 row: −1460 at 1e-6, −1332 at 1e-7, −1321 at 1e-8, −1319.5 at 1e-10). **1e-8 — the
  `ForceFreeStatesControl` default — is the sweet spot**: 1.8× fewer steps than 1e-10 for 7e-4
  relative Δ' error.
- **Step count scales as tol^(−1/6.7)**, a little steeper than the ideal 9th-order tol^(−1/9),
  exactly as expected when the error being controlled is knot-scale roughness rather than the
  smooth truncation term. The rejected-step fraction is flat (0.20–0.24), so this is not a
  step-control artefact.
- **EL wall time is sub-linear in nstep**: 1e-10 → 1e-8 cuts accepted steps 1.82× but EL time only
  1.29×. There is a ~7 s tolerance-independent floor in that phase on this deck. And the whole
  warm run is ~220 s regardless of tolerance — at mpsi = 512 with the NTV section active, EL
  integration is under 10% of the run. Tolerance relaxation is real but second-order here; it
  matters where EL dominates (large mpsi, no kinetic post-processing).

### 5. Integrator order — Vern9 is the wrong tool here

If steps cannot span a knot interval anyway, Vern9's 16 stages per step are wasted. I swapped all
six `Vern9()` call sites inside `ForceFreeStates` for `Vern7()` (10 stages), left the
equilibrium-side ones alone, and reran at mpsi = 512 (edits reverted afterwards):

| method | tol | nstep | EL (s) | ms/step | et[1] | Δ'diag rel |
|---|---|---|---|---|---|---|
| Vern9 | 1e-10 | 3018 | 18.9 | 6.26 | 0.782559048 | 6.2e-06 |
| Vern7 | 1e-10 | 4163 | 15.3 | 3.67 | 0.782559048 | 2.5e-04 |
| Vern9 | 1e-8  | 1656 | 14.7 | 8.85 | 0.782559057 | 6.9e-04 |
| Vern7 | 1e-8  | 2317 | 10.9 | 4.70 | 0.782559050 | 2.5e-04 |

Vern7 takes ~40% more steps but each step is 1.7× cheaper, so EL wall time drops ~20% at 1e-10 and
~26% at 1e-8. **Vern7 at 1e-8 dominates Vern9 at 1e-8 on both axes** — faster *and* closer to the
converged Δ' — and is 42% faster than the deck's current Vern9 @ 1e-10.

One caveat before anyone acts on this: Vern7's Δ' deviation is 2.47e-4 at 1e-10 and 2.50e-4 at
1e-8, i.e. **tolerance-independent**, so it is not step-truncation error. Something in the
propagator/matching path carries a method-dependent offset that Vern9 does not. It is small, but
it should be understood rather than absorbed.

### Recommendations

1. **Use the two-pass auto grid** (`mpsi = 0` with `psi_accuracy`) — already the example default.
   It reaches the converged et[1] ≈ 0.80 with 558 knots and about half the EL time of a fixed
   1024-knot log grid. Minimising knots for a target accuracy is exactly what minimises EL time,
   so the auto grid is the direct fix for the symptom in this issue.
2. **Revisit the per-deck `eulerlagrange_tolerance`.** The struct default is `1e-8`; the DIII-D
   decks pin `1e-10` and the LAR decks `1e-12`. On the evidence of §4 those tighter values buy
   Δ' accuracy this case does not need, at 1.8–4× the step count. This is a per-deck edit on its
   own branch with a regression-harness run, not a change to the struct default.
3. **Switch the ForceFreeStates integrator to Vern7** — measured in §5, ~20–26% off the EL phase,
   with better Δ' than Vern9 at the same tolerance. Gated on explaining the tolerance-independent
   2.5e-4 Δ' offset first; then its own branch and a regression-harness run.
4. **Smoother coefficients** — fitting F/K/G with smoothing (or higher-continuity) splines instead
   of pure interpolation would lift the knot-scale noise floor that currently pins the step size.
   This is the only change that would actually decouple nstep from mpsi rather than rescale it.
5. **Node-noise floor** — tightening the equilibrium field-line integration tolerance (`etol`)
   reduces the noise that fine grids amplify; worth checking at high mpsi.

Scripts and full tables are in the handoff branch (`handoff/issue376/`). One bookkeeping note: the
mpsi ladder (§2–§3) was measured a few merges earlier than the tolerance sweep (§4); the shared
mpsi = 512 / 1e-10 point agrees to 0.3% in nstep between the two, but the tables should not be
mixed element-by-element.
