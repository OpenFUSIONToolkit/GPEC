# York ground-truth: does anyone converge the drift-kinetic island solve?

Session note, 2026-07-19. Answers the "recommended first task" of the M1 launch
prompt: before more solver-robustness work on our resmax ~ 10⁻³ stall, establish
whether the York codes (kokuchou / DK-NTM / RDK-NTM) actually converge their
`Δ_neo`, and if so by what method / tolerance / domain. Sources read first-hand
this session: **L23** = Leigh 2023 thesis (`2023-Leigh-…`), **Diss19** =
Dudkovskaia 2019 PhD (`2019-Dudkovskaia-Modelling_NTMs…`), both in
`docs/resources/Drift_Kinetic_Island_References/`. This corroborates and sharpens
the `[CHECKED]` forensics already in `design/04-numerics.md §2, §5`.

## The one-line answer

**Nobody converges the self-consistent field.** In the E×B-dominated regime — the
physical regime — the electrostatic potential `Φ̂` does not converge for *any* York
code. What converges (and what York actually publishes) is the **output** `Δ_loc`
(the current-moment island-drive), which stabilises *even while the field residual
does not*. Our solve gating on the field residual `resmax ~ 10⁻³` is therefore
(a) measuring the wrong quantity and (b) demanding a tolerance ~100× tighter than
kokuchou ever met and ~1000× tighter than York's own stated criterion.

## kokuchou (L23) — the direct 4D `{p,ξ}` solve, our closest analogue

- **Method**: Picard iteration. `ĝ`, `Φ̂`, `Ū_∥,i` are recalculated in sequence
  from initial guesses `Φ̂ = 0`, `Ū_∥,i = pF̂′`, iterating to self-consistency
  (L23 §3.1.5, p. 78). Not a global Newton solve.
- **Convergence criterion**: array-averaged iterative residual
  `R(X_I) = |X_I − X_{I−1}| / (|X_I| + |X_{I−1}|)` (L23 Eq. 3.1.5) below a
  **tolerance of ε¹ ≈ 10%** — justified because the ion drift-kinetic equation
  (2.3.47) is itself only accurate to `O(ε^{3/2})` with `ε ≈ 0.1`, so a ~10%
  relative error is the natural floor (L23 §3.1.5, p. 78). L23 explicitly notes
  the array average *hides* individual points where `R` is large.
- **Result (L23 §6.1.1, p. 118)**: the 10% criterion on `Φ̂` and `Ū_∥`
  **"did not come to within (ε = 10%) relative error after the maximum of 4
  iterations for any of the runs."** Above `ŵ ≈ 10⁻³ r_s` (i.e. across the whole
  physical E×B regime) the **max iterative residual of `Φ̂` exceeds 100% per
  iteration** — the potential genuinely does not converge. The current-side term
  `Ū_∥` does much better (`< 30%`/iter, decreasing, localised to the island;
  far-field → the neoclassical constant).
- **But the output converges (L23 §6.2, Fig. 6.3, p. 118–119)**: **"∆loc is seen
  to converge stably"** across the 4 iterations *despite* the large `Φ̂` residual.
  `Δ_loc = 0` (the threshold `w_c`) at `ŵ ≈ 0.4–0.6 ρ̂_θi`. Toward `ŵ = 0.3 ρ̂_θi`
  even `Δ_loc` stops trending cleanly (L23 p. 118) — a low-`ŵ` reliability floor.
- **Why the field can't converge (L23 §3.1.6, p. 78–80)**: the physical response
  has a **drift-island separatrix layer of width `O(ρ̂_θi)`**, and the drift island
  is *shifted radially from the magnetic island* by `ρ̂_θi ω̂_D(y,σ,u)` — its
  location **varies with `y` and `u`**, so a single rectilinear `{p,ξ}` mesh cannot
  sit on it. When E×B dominates (`ν̂/(ŵL̂_q) < 1`, true over part of velocity space
  in *every* production run) the layer width `∝ Φ̂`, so **it moves between
  iterations** (design 04 §2, `[CHECKED: L23 Eqs. 6.1.1–6.1.2]`). The rectilinear
  mesh over-/under-resolves the rounded, moving layer → a locally-divergent region
  that pins the max residual. The L23 §2.6 amendment (`∂²ĝ/∂p² ∝ ρ̂²_θi`, not
  `ρ̂⁰`) makes those gradients *steeper* than the original DK-NTM, shrinking the
  reachable window further.
- **Cost/limits**: `nξ = 30`, `np = 145`, `ny = 49+40`, `nu = 24`; ~16.6 GB/u-point,
  0.4 hr/u-point on ARCHER2. Memory — not physics — set kokuchou's `ν_★ ≥ 5×10⁻³`
  floor and `ŵ ≲ 0.75 ρ̂_θi` ceiling (L23 §6.1.2). Max 4 Picard iterations was a
  resource limit, not a converged stop.

## DK-NTM (I19) and RDK-NTM (Diss19) — why *they* report converged `Δ`

- **DK-NTM (I19)**: same Picard-in-`Φ` structure, but **"did not have a strict
  definition of a converged solution (the final result over several iterations was
  assumed converged)"** (L23 §3.1.5, p. 78). So its "convergence" is looser than
  kokuchou's, not tighter.
- **RDK-NTM (Diss19)** sidesteps the moving-layer problem two ways:
  1. **Streamline (`S`) coordinate reduction** — `S` is "a generalised radial
     coordinate" whose contours *follow the drift-island streamlines* (Diss19 p. 1,
     Ch. III). The layer is then fixed in `S`, treated with **analytic matched
     layer solutions**, so there is no rectilinear-mesh-vs-rounded-layer mismatch.
     Valid only in the low-`ν_★` limit — which is exactly why kokuchou (finite
     `ν_★`) *cannot* use it and must pay the `{p,ξ}` rectilinear penalty.
  2. **Many headline `Δ` results are reported at the "0th iteration in Φ"**
     (`Φ = 0`, E×B coupling off) — e.g. Diss19 P4.9, Figs at p. 567/4556/6560/6590.
     The clean bootstrap `∝ 1/w` trend the earlier LOG cross-check cited
     (Figs 4.13–4.15) is largely a *pre-nonlinearity* result. Turning `Φ` on
     (their right-hand panels) is where the sensitivity appears.

So the field-vs-output split is universal in the lineage: the **field** does not
converge once the E×B nonlinearity is self-consistent; the **output `Δ`** is what
stabilises, and RDK-NTM buys extra robustness with analytic streamline layers +
`Φ = 0` reporting.

## Consequences for Islands (what this changes)

1. **We are gating on the wrong quantity.** Our `resmax` is the field residual.
   York gates on (or at least *reports*) the **output moment `Δ`**, which converges
   when the field does not. The immediate diagnostic — not yet run cleanly at
   physical parameters with the current fixed quadrature — is: **does our `Δ_neo`
   (and the current moment `⟨J̄_∥ cosξ⟩`) stabilise across resolution / continuation
   steps even though `resmax` floors at ~10⁻³?** That is the kokuchou `Δ_loc`
   behaviour and is the actual pass/fail test. (The earlier "`Δ_neo` doesn't
   resolution-converge" result was at plasma-scale `w=1` and pre-quadrature-fix —
   it must be re-taken at physical `ŵ ~ ρ̂_θi` with the spline `delta_moments`.)
2. **`resmax ~ 10⁻³` is already better than any York code.** kokuchou's `Φ̂`
   residual is `> 100%`/iter in this regime; its stated tolerance is 10%. Our
   ~10⁻³ (0.1%) field residual is ~100× below what York achieved and ~1000× below
   what York demanded. Treating the ~10⁻³ floor as a bug to eliminate is chasing a
   target the physics (the `O(ε^{3/2})` equation accuracy, the moving `O(ρ̂_θi)`
   layer) does not support.
3. **Two robustness levers are now *named by the ground truth*, not guessed**:
   - **Continuation in the E×B coupling** (Φ=0 → ramp): the linear neoclassical
     `Φ=0` problem should converge cleanly (it is RDK-NTM's 0th iteration); ramping
     the E×B coupling as a natural-continuation parameter and watching `Δ_neo`
     stabilise mirrors both RDK-NTM's reporting and kokuchou's Picard lag. This is
     the "continuation in the ExB coupling" the LOG named as the leading suspicion —
     the ground truth now supports it directly.
   - **Grid packed at the drift-island separatrix, not the magnetic island**
     (design 04 §1). Our island-centred mesh packs the wrong contour; the layer
     sits at `x` shifted by `ρ̂_θi ω̂_D(y,σ,u)` and *spread across velocity space*.
     A static single-location packing cannot resolve it — this is kokuchou's
     dominant accuracy limiter and a plausible source of our locally-divergent
     residual floor.

## The decision this raises (→ QUESTIONS Q7)

Changing the convergence **gate** (field `resmax` → output `Δ` stability) and its
**tolerance** (10⁻³ → the `O(ε^{3/2})`/few-% level York uses) is a
methodology/threshold decision. Per the module `[VERIFY]`/no-guess discipline
(thresholds are not to be set silently) this is escalated in `QUESTIONS.md` Q7,
not decided here. The ground-truth reading itself needs no sign-off — it is
sourced — but *adopting York's output-convergence posture as our definition of
done* does.
