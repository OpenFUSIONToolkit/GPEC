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
- **`eulerlagrange_tolerance = 1e-8`** (100× looser, same grid): accepted steps 3977 → **2116**
  with identical physics (et[1] unchanged to 8 digits). The step size is genuinely error-controlled
  — but at an effective order well short of Vern9's, see §4.

So the step size is genuinely error-controlled — but the error *magnitude* per unit ψ is set by
knot-scale roughness in the interpolated F/K/G coefficients, not by the physics. §6 is about where
that roughness comes from, because "it's the same smooth function, just sampled more finely" is
the natural expectation and it turns out to be false.

### 4. Tolerance sweep — how tight does `eulerlagrange_tolerance` actually need to be?

Five runs at mpsi = 512 (513 knots), everything but the tolerance held fixed. Δ' deviations are
relative to the 1e-12 point; `Δ'diag` is the worst per-surface relative error on the
`singular/delta_prime_matrix` diagonal.

| tol | accepted steps | saved | EL (s) | et[1] | Δ'diag rel |
|---|---|---|---|---|---|
| 1e-6  | 1159 | 929  | 10.1 | 0.78255904 | 1.7e-02 |
| 1e-7  | 1565 | 1240 | 12.2 | 0.78255903 | 3.3e-03 |
| 1e-8  | 2116 | 1656 | 14.7 | 0.78255906 | 6.9e-04 |
| 1e-10 | 3977 | 3018 | 18.9 | 0.78255905 | 6.2e-06 |
| 1e-12 | 9620 | 7297 | 32.9 | 0.78255905 | 0 (ref)  |

(`integration/nstep_total` is accepted steps; `integration/nstep` is saved snapshots at
`save_interval = 3`. Rejected steps are not recorded anywhere.)

Three things worth knowing:

- **et[1] is not a discriminator** — identical to 8 significant digits across six decades of
  tolerance. Δ' is what actually degrades, and it degrades fast below 1e-8 (the worst surface is
  the q = 5 row: −1460 at 1e-6, −1332 at 1e-7, −1321 at 1e-8, −1319.5 at 1e-10). **1e-8 — the
  `ForceFreeStatesControl` default — is the sweet spot**: 1.8× fewer steps than 1e-10 for 7e-4
  relative Δ' error.
- **Vern9 never achieves its formal order.** Per-decade exponents of h ∝ tol^α: 0.130, 0.131,
  0.137, then **0.192** for the last two decades. A 9th-order method on a smooth integrand should
  give α = 1/(p+1) = 0.100, or 0.111 under error-per-unit-step. Measured α is outside both
  everywhere and degrades sharply at tight tolerance — an error floor that is not truncation
  error. Vern7 over the same range gives α = 0.131, comfortably inside its own formal band
  (0.125–0.143): the 7th-order method behaves as advertised, the 9th-order one does not. §6
  explains why.
- **EL wall time is sub-linear in step count**: 1e-10 → 1e-8 cuts accepted steps 1.88× but EL time only
  1.29×. There is a ~7 s tolerance-independent floor in that phase on this deck. And the whole
  warm run is ~220 s regardless of tolerance — at mpsi = 512 with the NTV section active, EL
  integration is under 10% of the run. Tolerance relaxation is real but second-order here; it
  matters where EL dominates (large mpsi, no kinetic post-processing).

### 5. Integrator order — Vern9 is the wrong tool here

If steps cannot span a knot interval anyway, Vern9's 16 stages per step are wasted. I swapped all
six `Vern9()` call sites inside `ForceFreeStates` for `Vern7()` (10 stages), left the
equilibrium-side ones alone, and reran at mpsi = 512 (edits reverted afterwards):

| method | tol | accepted steps | EL (s) | ms/step | et[1] | Δ'diag rel |
|---|---|---|---|---|---|---|
| Vern9 | 1e-10 | 3977 | 18.9 | 4.75 | 0.782559048 | 6.2e-06 |
| Vern7 | 1e-10 | 5355 | 15.3 | 2.86 | 0.782559048 | 2.5e-04 |
| Vern9 | 1e-8  | 2116 | 14.7 | 6.95 | 0.782559057 | 6.9e-04 |
| Vern7 | 1e-8  | 2927 | 10.9 | 3.72 | 0.782559050 | 2.5e-04 |

Vern7 takes ~40% more steps but each step is 1.7× cheaper, so EL wall time drops ~20% at 1e-10 and
~26% at 1e-8. **Vern7 at 1e-8 dominates Vern9 at 1e-8 on both axes** — faster *and* closer to the
converged Δ' — and is 42% faster than the deck's current Vern9 @ 1e-10.

One caveat before anyone acts on this: Vern7's Δ' deviation is 2.47e-4 at 1e-10 and 2.50e-4 at
1e-8, i.e. **tolerance-independent**, so it is not step-truncation error. Something in the
propagator/matching path carries a method-dependent offset that Vern9 does not. It is small, but
it should be understood rather than absorbed.

### 6. Why refining the grid makes the integrand *rougher* — the actual mechanism

The integrand is the same smooth function only down to the accuracy of the node data. Below that,
adding knots does not add information; it just interpolates the error more finely.

Diagnostic: second divided differences of `matrices/ideal/F,K,G` (the actual EL integrand splines)
at their own knots. Two statistics — `r1`, the knot-to-knot autocorrelation of f'' (**+1 = smooth,
−2/3 = white noise**), and |f''|/|f|. `q(ψ)` on the same grid is the control.

| region | quantity | mpsi=256 | mpsi=512 | mpsi=1024 |
|---|---|---|---|---|
| ψ<0.1 | K: r1 / \|f''\|/\|f\| | −0.42 / 1.01e7 | −0.45 / 4.53e7 | −0.56 / 1.62e8 |
| ψ<0.1 | F: r1 / \|f''\|/\|f\| | +0.91 / 4.48e3 | +0.67 / 4.10e3 | −0.32 / 4.65e3 |
| 0.3–0.7 | F: r1 / \|f''\|/\|f\| | +0.91 / 1.730 | +0.96 / 1.712 | +0.97 / 1.705 |
| 0.3–0.7 | G: r1 / \|f''\|/\|f\| | +0.83 / 3.66 | +0.86 / 3.87 | −0.18 / 3.99 |
| all | **q control r1** | **+0.88…+0.94** | **+0.94…+0.97** | **+0.97** |

Near-axis K grows 4.5× then 3.6× per doubling — the Δ⁻² of a fixed-amplitude node error — while
r1 marches toward the white-noise value. Meanwhile mid-plasma F converges beautifully
(1.730 → 1.712 → 1.705): where the grid is still coarse relative to the noise, the physics
dominates and your intuition holds exactly. The noise floor simply spreads outward with mpsi —
axis first, then edge, then mid-plasma.

**Three candidate sources tested and ruled out**, all at fixed mpsi = 512:

| knob | range tested | effect on accepted steps |
|---|---|---|
| `etol` (field-line integration tolerance) | 1e-8 → 1e-12 | 4038 → 3930 (2.7%) |
| `mtheta` (poloidal resolution) | 256 → 1024 | 3977 → 3983 (0.3%) |
| input equilibrium | 257×257 EFIT vs analytic Solovev | Solovev diverges the same, or worse |

So it is not the ODE tolerance, not the poloidal quadrature, and not the finite resolution of the
input reconstruction. It is generated inside the equilibrium → coefficient pipeline.

**Localised.** A local degree-4 fit residual over a 9-knot window measures the node-error amplitude
directly (smooth data → falls like Δ⁵; a floor → flat). Mid-plasma:

| matrix | built from | resid @512 | resid @1024 | r1 @512 | r1 @1024 |
|---|---|---|---|---|---|
| A, B, D | g22, g23, g33 | ~6e-08 | **~1.3e-08** | +0.972 | **+0.987** |
| C | + g31, jtheta·imat | 8.90e-06 | 1.21e-05 | +0.694 | **−0.401** |
| E | + g31, jtheta·imat, q1 | 8.74e-06 | 1.17e-05 | +0.547 | **−0.400** |
| H | + g31, jtheta·imat | 1.11e-05 | 1.41e-05 | +0.837 | **+0.022** |
| F = F̃ − D†A⁻¹D | clean inputs | 2.48e-07 | 1.96e-07 | +0.957 | +0.973 |
| K = E − K†A⁻¹C | dirty inputs | 1.22e-05 | 1.56e-05 | +0.916 | +0.593 |
| G = H − C†A⁻¹C | dirty inputs | 1.01e-05 | 1.70e-05 | +0.856 | −0.176 |

**The floor enters at C, E, H at the ~1e-5 relative level.** A/B/D are clean at 1e-8 *and still
converging*, exactly as smooth data should. The derived-matrix algebra is not the culprit either —
F goes through the same `A⁻¹` path and stays clean at 2e-7 because its inputs are clean; K and G
are dirty only because E, C and H are. Per `Fourfit.jl:439-447`, the three dirty matrices are
precisely the ones involving **g31**, **jtheta·imat** and (for E) **q1 = dq/dψ**; the clean ones
use only g22/g23/g33.

### So, the answer to the question in the title

1. C/E/H node values carry a relative error of ~1e-5 that **does not shrink with mpsi**.
2. The cubic spline interpolates that error exactly, contributing ~ε/Δ² to f'' — growing 4× per
   mpsi doubling.
3. Once ε/Δ² exceeds the physical curvature, the integrand is dominated by knot-scale structure —
   near axis first, spreading outward.
4. The adaptive controller prices that structure, so the step collapses onto the knot scale:
   dψ ∝ Δ, ~6 steps per knot interval, invariant in mpsi.
5. Because the controlled error is not smooth truncation error, Vern9 cannot reach its formal
   order — hence §4's exponents and §5's result that Vern7 loses nothing.

The premise "same smooth function, just sampled more finely" is true of the physics and false of
the data. Below the 1e-5 floor, refining the grid doesn't improve the integrand — it makes the
interpolant measurably rougher in exactly the derivative norms the error estimator reads.

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
4. **Fix the ~1e-5 node-error floor in C/E/H** — this is the root cause and the only change that
   would genuinely decouple nstep from mpsi rather than rescale it. `Fourfit.jl:439-447` and the
   construction of `g31`, `jtheta`/`imat` and `q1` are the place to start, since A/B/D are clean
   at 1e-8 and converging. Failing that, fit C/E/H with smoothing rather than interpolating
   splines so the floor is not differentiated.

One bookkeeping note: the
mpsi ladder (§2–§3) was measured a few merges earlier than the tolerance sweep (§4); the shared
mpsi = 512 / 1e-10 point agrees to 0.3% in nstep between the two, but the tables should not be
mixed element-by-element.
