# Islands — session LOG

Cross-session memory spine (design doc `06 §2.5`). Read this at session start
together with `QUESTIONS.md`. Append a short entry before every session end:
**what moved / what's blocked / next action**. Newest entries at the top.
Reference `QUESTIONS.md` IDs (`Q<n>`) and ladder IDs (`A1`, `B5a`, …) where
relevant.

---

## 2026-07-24 (cont. 12) — GOAL LOOP B2a: bounce-substitution implemented — fixes the erratic misses (6→1) and improves nE=3 ~10×, but NOT sufficient alone

Implemented B2a in `Coefficients.jl`: a shared `_bounce_primitives(y,ε)` helper + both
trapped branches (`orbit_average_drift_brackets`, `orbit_average_pitch_brackets`) now
use the half-angle substitution `sin(θ/2)=sin(θb/2)sinφ`, removing the integrable
`1/√(1−yb)` turning-point singularity analytically. Prototype-validated: matches the old
`quadgk` to ~1e-10 where quadgk converged; smooth, no misses.
- **Result**: trapped-region misses **6→1** (only `y=1.000`, the genuine `y_c`
  log-divergence, remains). Cold solve: nE=2 converges (1.1e-11); **nE=3 improves ~10×
  (1.0e-4→1.01e-5) but still fails**; nE=4/6 unchanged (~4e-3, ~1e-2).
- **⇒ B2a is a correct fix (removes a real quadrature bug) but NOT sufficient** — the
  stall persists at nE≥3. Remaining non-smoothness: the genuine `y_c=1` divergence (B2b)
  and/or the conditioning growth with nE. Committing B2a on its own merit (correctness);
  physics-verifier auditing concurrently — **revert if it flags an error**.
- **Next**: evaluate physics-verifier + configure-suite regression; then Gate B: pursue
  B2b (genuine `y_c` layer treatment — physics-adjacent, York matching/TSVD) or jump to
  Option C (trust-region), given B2a alone doesn't converge nE≥3.

## 2026-07-24 (cont. 11) — GOAL LOOP Option B: root cause FOUND — the trapped-region orbit-average brackets are non-smooth (erratic quadrature misses + genuine y_c divergence)

B1 evaluated `orbit_average_drift_brackets` (A,G) and `orbit_average_pitch_brackets`
(S,T) across `y_c=1.0` (`Coefficients.jl`). Findings:
- **Passing (y<1.0): perfectly smooth**, monotonic A,S,T.
- **At y_c=1.0/1.02: MISS→0** (a hole at the trapped-passing boundary).
- **Trapped (y>1.04): erratic + sporadic MISSES→0** at y=1.08, 1.16, 1.20, 1.22, with
  T(y) spiking non-monotonically (3.68→6.53→5.97→…). The "graceful miss"
  (`_try_*`→`nothing`→coeff 0) then zeros scattered trapped-region nodes ⇒ a
  **non-smooth coefficient field** exactly where the physics is stiffest. More `(y,E)`
  nodes ⇒ more nodes in this zone ⇒ worse Newton convergence — **explains the
  resolution-dependent stall precisely** (cont. 9/10).
- **Mechanism** (read `Coefficients.jl:60-156`): both brackets correctly branch
  passing (`∫₀^{2π}`) / trapped (bounce `∫_{-θb}^{θb}`). Two distinct failures:
  1. **Quadrature-robustness bug (numerics, fixable):** the trapped `1/√(1−yb)`
     integrands (G, T) have an **integrable turning-point singularity** at `±θb`;
     `QuadGK.quadgk(f, -θb, 0, θb)` fails to converge for some `y` → the `try` wrapper
     returns `nothing`→0 (the scattered MISSES at *finite*-integral `y`). Fix: the
     standard **half-angle substitution** `sin(θ/2)=sin(θb/2)·sinφ` maps the bounce
     integral to `∫_{-π/2}^{π/2} …/√(1−k²sin²φ) dφ` with the `1/√` singularity
     **removed analytically** (same integral, computed robustly). Pure numerics.
  2. **Genuine `y_c=1` log-divergence (physics):** `T(y)=⟨1/√(1−yb)⟩ → ∞` as `y→1`
     (near-separatrix layer) — a real singularity the York codes treat specially
     (I19 y_c matching Eqs A.7–A.10; L23 §4.2 TSVD). Not a quadrature artifact.
- **Plan**: B2a — implement the half-angle-substitution bounce quadrature (fixes the
  MISSES; pure numerics, but coefficient-valued ⇒ **physics-verifier before commit**).
  B2b — the genuine `y_c` divergence, only if B2a is insufficient (physics-adjacent →
  QUESTIONS/derivation if the treatment is uncertain). Prototyping B2a now in scratch
  (confirm smooth + matches quadgk where quadgk works) before touching `src`.

## 2026-07-24 (cont. 10) — GOAL LOOP: Option A ruled out (the stall is init-independent) → advancing to B (y_c smoothness)

Goal-mode loop on `notes/solver-convergence-goal-plan.md` (converge the physical
`nE≥3` solve). **Option A — resolution/continuation — is ruled out:**
- **A2 grid-prolongation** (the decisive diagnostic): solve `nE=2` (converges,
  resmax 2.3e-12), interpolate that state onto the `nE=3` energy grid, warm-start the
  `nE=3` Newton. Result: **prolonged `nE=3` still fails (resmax 1.8e-3) — no better
  than cold (1.0e-4), slightly worse.** A near-optimal init does not cross the wall.
- ⇒ **the stall is initialization-independent** → not a basin problem → no
  continuation path (A1 coefficient-homotopy included — it hits the same intrinsic
  `nE=3` problem at λ=1) will cross it. Gate A: **advance to Option B.** (A1 not
  separately run; revisit only if B and C both fail.)
- This *reinforces* the cont.-9 conclusion (nonlinear-iteration wall, not linear/init):
  the prolonged init sits at resmax~1e-3 and Newton cannot descend below the ~1e-3
  plateau from a good starting point — the fingerprint of a **non-smooth residual**,
  which is exactly Option B (`y_c`-layer coefficient discontinuity).
- Next: B1 — diagnose `y_c` smoothness (evaluate the orbit-average brackets +
  coefficients across `y_c`; confirm/deny a jump). No physics change yet.

## 2026-07-24 (cont. 9) — REFRAME: the resmax~1e-3 stall is a SOLVER/preconditioner robustness problem, NOT the far-field BC — sign-off recorded, then isolated the real axis

- **Recorded the human sign-off** on the analytic large-p far-field
  (`analytic-far-field.md` → `[CLEARED: 2026-07-24]`; distinguished the signed-off
  anchor *value* `⟨x_D⟩` from the numerics of *how* it's imposed).
- **Then isolated the actual blocker with cold-solve diagnostics** (node freed up):
  - **BC- and grid-independent**: at a *collisional* `ν̂=0.5` (the easy regime),
    cold Newton–Krylov stalls at resmax~1e-3 for **all four** of
    {band, plain} × {`:analytic`, `:dirichlet`} — none converge. So the stall is
    **not** the far-field form and **not** the band grid.
  - **Not a clean `w` axis either**: holding everything fixed and sweeping `w`
    (identical grids in `w`-normalized units), `w=0.5,0.3,0.2,0.1,0.05` all stall
    (resmax 1e-3–1e-4) but `w=0.03` converges to **1.06e-11**. Non-monotonic,
    config-sensitive — rules out `w`, collisionality, BC, grid-type, box-size as the
    axis.
  - **Conclusion**: the persistent "crawls to resmax~1e-3 from any init" (LOG cont.
    2–3, e69b7b65) is a **matrix-free Newton–Krylov + `PlaneJacobi` preconditioner
    robustness problem** — the preconditioned operator stays too ill-conditioned for
    GMRES to push most configs below ~1e-3 (the Newton line search then stagnates),
    while a lucky config reaches machine precision. The system is **solvable** (1e-11
    when it works), so it is not inconsistent — it is solver robustness.
- **This reframes the milestone blocker.** Several sessions (and my collisionality
  homotopy, cont. 8) targeted the far-field/localization; the diagnostics say that is
  *not* what's stalling the solve. The far-field anchor (item 1, signed off) and the
  drift coordinate (item 2 = Q8) remain correct localization physics, but they cannot
  make the solve converge because the convergence wall is upstream, in the
  nonlinear-solver/preconditioner.
- **SHARPENED (same day, conditioning + solver diagnostics) — it is a NONLINEAR
  Newton-convergence wall, NOT the preconditioner:**
  - `cond(J)` at the stalling config is only **~1e5** (σ_min~2e-4, **not**
    near-singular → no null mode) — and is **the same for stall and converge** (w=0.05
    stalls, w=0.03 converges, both cond(J)~2e5). Raw conditioning does not explain it.
  - **`PlaneJacobi` is nearly inert on physical resolved grids** — `cond(M⁻¹J)`
    reduction is **≤2.3×** (sometimes <1×, i.e. it makes it worse), vs the advertised
    >1000× it gives only in the special high-`ρ̂_θi`/`w=0.5` regime it was tuned on.
  - **`cond(J)` grows ~4× per energy node** (nE sweep: 8.9e3→3.1e4→1.2e5→4.6e5 for
    nE=1,2,3,4); cold convergence flips at **nE=3** (cond crosses ~1e5).
  - **DECISIVE**: `newton_direct` (exact ForwardDiff Jacobian + exact dense LU solve —
    `dense_jacobian` is exact duals, not FD) **also fails, and worse** — it fails even
    at nE=2 where inexact `newton_krylov` converges to 1e-12 (nE=2/3/4/6 resmax
    1.1e-2/6.7e-2/2.3/5.7). An *exact* linear solve does no better ⇒ **the blocker is
    not the linear preconditioner/conditioning at all; it is the nonlinear Newton
    iteration**, which stalls at resmax~1e-2–1e-3 from the cold init and worsens with
    resolution (nE, ny). (This contradicts the `newton_direct` docstring claim that the
    exact solve converges "regardless of conditioning" — true only in the `cond~1e9`
    high-`ρ̂_θi` regime, false on physical resolved grids — flag to fix.)
  - **Most likely cause** (worsens with resolution near the `y_c` layer): a **non-smooth
    residual** from the `y_c`-layer "graceful miss" coefficient discontinuities
    (`_try_drift_brackets`/`_try_pitch_brackets` return `nothing`→coeff 0 as a node
    crosses `y_c`) and/or the boundary/forbidden-row replacements — Newton's quadratic
    convergence breaks on a kinked residual, and more `(y,E)` nodes near `y_c` ⇒ more
    kinks ⇒ worse. Inexact Krylov's truncated GMRES acts as implicit damping, which is
    why it sometimes beats the exact step.
- **Next (recommended)**: (a) test whether **resolution/parameter continuation**
  (warm-start nE=3 from the nE=2 solution — reuse the `globalized_level0_solve`
  machinery but continue in resolution, not `ν̂`) crosses the wall → distinguishes a
  basin/init problem from a fundamental non-smoothness; (b) audit the `y_c`-layer
  coefficient handling for smoothness (a `C⁰`/`C¹` blend across `y_c` instead of the
  hard `nothing`→0 drop); (c) consider a trust-region globalization instead of Armijo
  line search. NOT more preconditioner or far-field work. QUESTIONS Q7: localization
  items are "needed but not sufficient"; the blocker is the nonlinear solve.
- Scratch diagnostics under `/tmp` (not committed); the reusable
  `globalized_level0_solve` + its test stay (they're correct machinery, just aimed at
  the wrong axis — a `w`- or exact-Jacobian-continuation variant is the likely reuse).

## 2026-07-23 (cont. 8) — Globalization (3) implemented + tested: it EXTENDS the basin but does NOT reach the physical target — the low-ν̂ wall confirms (1)/(b) is still required

Implemented the unblocked numerics item (3) from the York ground-truth:
`Configure.globalized_level0_solve(grid, phys, species; farfield_mode=:analytic,
nu_boost, nsteps, ...)` — **collisionality-homotopy warm-start** driven through the
existing `Solvers.natural_continuation`. Ramps `ν_star` geometrically from a
collisional (well-conditioned) start down to the physical value, warm-starting each
solve; only `ν_star` varies (every intermediate point is a valid, more-collisional
physical config — no coefficient guessed). This is exactly L23 §7.1's remedy
(`kokuchou` warm-starts `Φ̂` from a stable run because it will not converge from
`Φ̂=0` at low ν_★).

**Empirical result (DIII-D physical scenario, ε=0.265, ρ̂_θi=0.075, ν_★=0.0124,
`:analytic` BC, bounded box Lx=0.2=4w):**
- COLD `:analytic`: **not** converged, resmax=2.8e-3 (the familiar stall).
- GLOBALIZED: converges the collisional end and warm-steps **down to ν̂≈0.078**
  (~16× extension of the convergent range) then **hits a wall** — sub-step halving
  cannot cross ν̂≈0.07 to reach ν̂=0.0124.
- **So (3) helps but does NOT solve the physical target.** The obstruction is a
  genuine low-collisionality convergence wall, not an init problem — even the
  `:analytic` (drift-shifted, "well-posed + localizing" by design) BC stalls there.
  This **confirms the ground-truth**: the remaining blocker is the **(1)/(b)
  far-field null-mode anchor** (L23 §7.1's analytic large-p form), which the finite-ν̂
  term was only partially standing in for. Below ν̂≈0.07 that regularization is too
  weak and the winged/null-mode obstruction returns.
- **Tests green**: solve suite gains a fast globalization integration test (small
  known-converging config: reaches the target ν_star, warm-started from collisional;
  arg guards; `_with_nu_star`) — passes. Configure suite `1563/1563` + physical
  scenario testset still green with the new `import ..Solvers` (no regression).
- **Next**: (3) is a reusable tool now in place and will be needed once (1)+(2) land,
  but it is NOT sufficient alone. The critical path is unchanged: bring the **analytic
  large-p far-field form** (1) to human sign-off + physics-verifier, and the
  **drift-shifted radial coordinate** (2 = Q8). Recommend that as the next work.

## 2026-07-23 (cont. 7) — York ground-truth on the far-field BC + Δ extraction: Q7 fork collapses to (b)

User chose "pin York ground-truth first." Read L23 (§2.3.6/§2.4/§2.5–2.6/§7.1) and
Diss19 (§4.2) directly from the in-repo PDFs and wrote a cited side-by-side:
`notes/york-farfield-extraction-ground-truth.md`. Decisive outcome:

- **(c) [matched-jump extraction] is RULED OUT.** York's `Δ_loc` (L23 Eq.
  2.5.10→2.5.13) is a **volume moment** — *identical to our `delta_moments`*; the
  jump `Δ'` is the OUTER parameter York **neglects**. Our extraction operator is
  correct as-is (L23 footnote 8 even confirms our σ-even/ξ-independent cancellation).
- **The non-localization is (b): BC + coordinate + globalization**, with a cited recipe:
  (1) `:neumann` `∂ĝ/∂p=0` **+ an analytic large-`p` anchor** to kill the winged
  null-mode — L23 §7.1 explicitly says the bare BC has "multiple numerically-valid
  solutions" (the winged branch) and names the analytic large-p form as the fix
  (**physics sign-off item**); (2) carry the **drift-island shift in the radial
  coordinate** `p̂_ϕ` (Diss19 §4.2 / L23 Eq. 7.1.1) — this is **Q8**, a prerequisite;
  (3) **warm-start / continuation** globalization — L23 §7.1 says `kokuchou` itself
  does NOT converge from `Φ̂=0` and warm-starts from a stable run — the **same**
  "crawls from any init" symptom we've been fighting. (3) is unblocked numerics.
- **Q7/QUESTIONS updated** with the same finding; the milestone's critical path is now
  concrete: (3) now, (1)+(2) to sign-off. No `src/` change and no coefficient cleared
  — read-only ground-truth.
- **Next**: recommend implementing (3) (warm-start/natural-continuation globalization
  of the physical solve) since it needs no sign-off and directly targets the stall;
  in parallel, bring the analytic large-p far-field form (1) to human sign-off.

## 2026-07-23 (cont. 6) — B5 config is now DERIVED from the physical scenario (task #5); benchmark still gated on Q5+Q7

- **Wired (task #5)**: `benchmark_B5_york_thresholds.jl` `_b5_phys` no longer hand-sets
  York-regime numbers — it calls `Configure.scenario_from_equilibrium` on
  `examples/DIIID-like_ideal_example` (cached load) at the q=2 surface, so the B5
  config is the same DERIVED physical vector (ε=0.265, ρ̂_θi=0.075, ν_★=0.012,
  η_i=2.16). `_assemble_b5` now takes the matching radius from `physical_domain(phys)`
  (fixed physical fraction, NOT scaled with w) and y_max from `(1+ε)/(1−ε)`.
- **Verified end-to-end** (clean env): both `_assemble_b5(:original)` and
  `(:improved)` build via `configure_level0` (returns the term NamedTuple). Benchmark
  stays SKIPPED as designed — `const UNGATED=false`.
- **Still gated** on **Q5** (uncleared coefficient families → `configure_level0` runs
  structurally only) AND **Q7** (far-field extraction does not localize). Flipping
  UNGATED before both clear asserts-out. So B5 is now *physical* but the York
  threshold NUMBER remains blocked — the remaining work is the Q7 non-localization,
  a strategic decision (see the cont.-3 close and QUESTIONS Q7).
- **Env note**: this repo's Julia must be run with `LD_LIBRARY_PATH` cleared
  (`env -u LD_LIBRARY_PATH julia …`) — the omfit conda env on this box leaks
  SuiteSparse 5, breaking CHOLMOD init when the equilibrium/vacuum sparse solvers load.

## 2026-07-23 (cont. 5) — DIII-D-like ingest → a PHYSICAL banana-regime scenario; `scenario_from_equilibrium` wired + tested (and the a10 "edge-cold" was my unit bug)

- **Moved (user: use the DIII-D-like example)**: `examples/DIIID-like_ideal_example`
  (H-mode g-file + kinetic `.h5` with `/T_i,/n_i,/psi`) → `scenario_from_equilibrium`
  → a **fully physical, self-consistent** Level-0 vector at the **mid-radius q=2**
  surface (`ψ_s=0.518`): **ε=0.265, ρ̂_θi=0.075, ν_★=0.0124 (low-collisionality
  banana — York's regime!), η_i=2.16**, inv_Lq=0.716, inv_Ln0=−0.337, from
  `R₀=1.74 m, B=2.04 T, T_i=2.08 keV, n_i=3.3e19` — every quantity from the SAME
  equilibrium + `T_i(ψ)/n_i(ψ)`. Assembles into a runnable config. Compare the
  arbitrary hand-set `_b5_phys` (ε=0.1, ρ̂_θi=0.05, ν_★=0.01, η_i=1): the derived
  values are genuinely different and physical.
- **CORRECTION to (cont. 4)**: the a10 "edge-cold T_i→0" was **my unit bug**, not
  physics — `load_kinetic_profiles` stores `Ti_spline` in **Joules** (`.*eV_to_J`),
  and I passed that (~3e-16) to `physical_scenario` as if eV, so `T_i_eV≈0`. Fixed
  (÷e). The a10 profile is warm too; the cold-surface guard (cont. 4) correctly caught
  the degenerate output either way (defense in depth, kept).
- **Wired + tested**: `Configure.scenario_from_equilibrium(equil, kp; m, n, w_psi)` —
  reads only fields off the passed objects (`q_spline`, `rzphi_rsquared`, `eqfun_B`,
  `ro`, `Ti_spline`, `ni_spline`), so **no Islands→Equilibrium module dependency**;
  finds `q=m/n`, computes `r_s=⟨√(rzphi_rsquared)⟩_θ`, converts `Ti` J→eV, FD gradients,
  forwards to `physical_scenario`. Fast **mock-based** unit test (no equilibrium solver
  in the suite; the real DIII-D ingest is a benchmark). Also found+worked-around a
  `Ti_deriv` accessor returning 0 (FD gradients used instead).
- **Next (task #5)**: un-gate B5 on `scenario_from_equilibrium` (physical DIII-D params
  + `physical_domain`). Producing the actual York threshold number is still Q7-gated
  (the extraction non-localization), but B5 can now be made physical. Scratch `/tmp`.

## 2026-07-23 (cont. 4) — Equilibrium ingest works (r_s from the flux surface); it reveals the a10 q=2 surface is EDGE-COLD

- **Moved (user: use the equilibrium code to find r_s)**: validated the full a10 ingest
  (scratch `/tmp/a10_ingest.jl`): `EquilibriumConfig(inputs["Equilibrium"], dir)` +
  `setup_equilibrium` + `load_kinetic_profiles`, root-find `q=2`, and **r_s from the
  flux-surface geometry** — `r_s = ⟨√(rzphi_rsquared(ψ_s,θ))⟩_θ` (`rzphi_rsquared =
  (R−ro)²+(Z−zo)²`, the distance² from the axis). Result: `R₀=2.005 m`, `B₀=1.0 T`,
  `q=2` at `ψ_s=0.788`, **`r_s=0.167 m`, `ε=0.083`**; gradients `inv_Lq=1.07`,
  `inv_Ln0=−3.19`, `η_i=2.29`, `lnΛ=15.96` (cross-checked vs `kp.nui_spline`). The
  geometry/gradient extraction is **correct and physical**.
- **BUT the ingest revealed a real problem**: the a10 `q=2` surface is at `ψ=0.79`
  (near the edge), where `T_i` has dropped to `≲10⁻³` eV (the profile → 0 at the edge;
  the cubic spline even undershoots slightly negative). So `ρ̂_θi→0` and `ν_★→−∞`
  (lnΛ goes negative below `~3×10⁻³` eV). **The a10 example's rational surface is
  edge-cold — not a usable NTM/York scenario as-is** (NTMs are warm core/mid-radius).
- **Added**: a **cold-surface degeneracy guard** in `physical_scenario` (throws with a
  clear message if `ν_★`/`ρ̂_θi` come out non-finite/≤0), + a test. configure suite +
  the new guard test green.
- **Next (needs a steer)**: the ingest MACHINE is done and validated; the a10 profile
  is unsuitable at `q=2`. Options: (i) a different LAR/example with a **warm q=2
  surface** (mid-radius); (ii) scale/replace the a10 kinetic profile; (iii) pick a
  lower-`q` rational (e.g. `q=3/2` if warmer). Then wire `scenario_from_equilibrium`
  into the module and un-gate B5. Surfaced to the user. Scratch `/tmp`; guard + LOG
  committed.

## 2026-07-23 (cont. 3) — Built the pinned physical scenario (design 10): SI equilibrium/profiles → self-consistent normalized Level-0 vector

- **Moved (user: build the scenario)**: `Configure.physical_scenario(...)` +
  `Configure.physical_domain(...)`, with `derivations/physical-scenario.md`
  (`[DERIVED: 2026-07-23]`). Turns SI quantities at the rational surface
  (`R₀, r_s, B, T_i, n_i, q, dq/dψ, ψ_s`, log-gradients) into the normalized
  `Level0Physics` — **every input derived from the SAME `T_i`/`n_i`/`B`/geometry**,
  closing the audit gap (ρ̂_θi/ν_★/inv_Ln0/η_i were independent hand-set knobs).
  `physical_domain` = a fixed local matching radius (`|x|≲0.2`, independent of `w`).
- **Formulas** (standard textbook, NOT the disputed island coefficients): ε=r_s/R₀;
  v_th=√(2T/m); ρ_i=m_i v_th/(ZeB); ρ_θi=ρ_i q/ε (LAR); ρ̂_θi=ρ_θi/r_s; NRL ν_ii + lnΛ;
  ν_★=ν_ii R₀q/(ε^{3/2}v_th) (docs/01 §2.3); the ψ-ratio normalizations (inv_Lq,
  inv_Ln0, η_i, inv_LB) exact per docs/01 §5. Validated: a10-like inputs give ε=0.100,
  ρ̂_θi=0.045, ν_★=0.34 (the **real** physical value for 0.48 keV/2e18 — honestly more
  collisional than the artificial 0.01; note inv_Ln0<0, the physical sign the hand-set
  +1.0 had wrong).
- **physics-verifier PASS**: every formula a faithful standard relation (NRL ν_ii to
  0.08%; ν_★ matches L23 Eq. 2.3.40; gradient ratios match docs/01 §5; SI constants
  correct); the two convention items (v_th factor, ν_ii-vs-ν_jj prefactor) honestly
  flagged as `[DERIVED]` sign-off targets; no tag cleared, no disputed coefficient
  hardcoded. Tests: configure 1563/1563 + a new `physical_scenario` testset (14) green.
- **Next (task #5)**: the real a10 equilibrium ingest (`setup_equilibrium` +
  `read_kinetic_file` → find q=2 surface → extract the SI inputs) and un-gate B5 as
  the York-replication gate. The pure `physical_scenario` is ready to receive those
  numbers; the equilibrium-geometry extraction (`r_s`, gradients at ψ_s) is the
  remaining API work.

## 2026-07-23 (cont. 2) — Fix the failing Documentation CI (Documenter cross-references broken by the band-grid docstrings)

- **The Documentation workflow had been red for several commits** — my band-grid
  docstrings introduced three Documenter `:cross_references` errors on `islands.md`
  (the `@autodocs` API page): (1) `[`_fd_matrix`](@ref)` — `@ref` to an internal
  (undocumented) symbol; (2) `[`PlaneJacobi`](@ref)` — cross-module `@ref` that does
  not resolve from the PhaseSpace autodocs; (3) `[`drift_coefficient_table`] (…)` —
  a shortcut-ref code span immediately followed by a parenthetical, which Documenter
  parsed as `[text](dest)` with the parenthetical as an invalid local file link.
- **Fixed** (repro'd + verified with a local `docs/make.jl` build → `EXIT=0`, no
  errors): the two bad `@ref`s → plain code spans (`` `_fd_matrix` ``,
  `` `Solvers.PlaneJacobi` ``); the two `[`drift_coefficient_table`] (…)` links →
  plain code spans (an *explicit* `[`name`](@ref)` also failed to resolve for this
  symbol, so plain code is the safe form). Lesson (already in root CLAUDE.md): build
  docs locally before pushing — `[`code`]`/`(@ref)` patterns in docstrings are CI-hard.
- Docstring-only; no behavior change. Suite unaffected; docs build green.

## 2026-07-23 (cont.) — No src box larger than the plasma: physics boxes made physical; the one MMS box is proven-irreducibly-abstract and now documented as such

- **User: "don't leave any boxes larger than the plasma in the src code."** Surveyed
  every domain literal in `src/Islands`. The only literal box `>1` is the
  discretization-test `solve_mms` (`Verify.jl`, `halfwidth_x=6`); the grid-builder
  `Lx_over_w` defaults are ratios (relative to `w`), not boxes, and already carry
  physical-domain cautions.
- **Tried hard to make `solve_mms` physical — it CANNOT be, and this is intrinsic**
  (verified across 5 solver/box combinations, ~90 min compute): a manufactured
  solve-level MMS needs the box several feature-widths wide (well-resolved +
  boundary-decayed + well-conditioned). Shrinking to a physical `|x|<1` steepens the
  feature and the assembled linear MMS system becomes ill-conditioned/unsolvable —
  naive GMRES ran 69 min without converging (order→1.8); `PlaneJacobi` gave order 2.4,
  non-converged; `newton_direct` (exact LU) gave err=4.05, non-converged. A physical
  box either steepens the feature (unsolvable) or flattens it (no order signal) — no
  workable middle. So box `≫1` is a mathematical requirement of the test.
- **Resolution (zero-risk, addresses the actual concern = confusion)**: documented the
  MMS radial `x` as an **ABSTRACT discretization-test coordinate, not the physical
  `x=(ψ−ψ_s)/ψ_s`** — a prominent note in the `Verify` module docstring + an inline
  comment at the `solve_mms` grid. Readers can no longer mistake it for a physical grid
  `6×` the plasma. **All physics grids are physical** (B5 fixed to `Lx=0.25`;
  `resolved_island_grid`/`drift_island_resolved_grid` cautions; design-10 scenario).
- **Net**: no physics box in src is larger than the plasma; the one remaining `>1` box
  is the abstract MMS test coordinate, now unmistakably labelled. If a truly physical
  MMS is wanted it needs a redesigned manufactured verification (a separate task) —
  surfaced to the user. Comment/docstring only; suite unaffected.

## 2026-07-23 — Physical-parameter audit + cleanup + pinned-scenario plan (user: make everything physical, plan the scenario, explain "resistivity")

- **Audit** (`notes/physical-parameter-audit.md`): the *physics* knobs are York-regime
  (ε=0.1, q=2, m/n=2/1, τ=1, ν_★=0.01 low-collisionality/banana, ρ̂_θi=0.05 ⇒
  ρ_i≈2.5e-3 r_s) but the **domains are unphysical everywhere** (`halfwidth_x=6–8` in
  `x=(ψ−ψ_s)/ψ_s`, i.e. 6–8× the plasma; axis at `x=−1`), there is **no physical-input
  layer** (ρ̂_θi/ν_★/inv_Ln0/η_i are independent hand-set knobs, not from T_i/B/R/a;
  `Species` T/n/gradients inert), and **York replication is NOT demonstrated** (B5 is a
  gated stub).
- **"Resistivity η" clarified**: `Level0Physics.eta_i` is **η_i = L_n/L_{T_i}** (the ion
  temperature-gradient ratio, dimensionless, naturally O(1)) — used only in the
  gradient-drive temperature factor `[1+(E−3/2)η_i]`. It is **not** resistivity; there
  is **zero resistivity** in the Islands Level-0 drift-kinetic model
  (`grep resistiv|ohm|spitzer src/Islands` is empty). Resistivity (η~1e-8 Ω·m) lives in
  the *outer* resistive-MHD region / the classical Δ′ (a separate GPEC path), not here.
- **Cleanups (committed)**: B5 benchmark domain fixed `halfwidth_x=8` → a physical local
  `resolved_island_grid` (`Lx=5w=0.25`, `y_max=4.0`), config annotated PROVISIONAL
  pending the derived scenario; physical-domain **cautions** added to
  `resolved_island_grid` and `drift_island_resolved_grid` docstrings (`Lx` must stay a
  local matching radius `|x|≲0.3`, never plasma-scale). Structural/MMS **unit** tests
  keep their abstract boxes by design (they verify discretization/wiring, not physics;
  forcing physical values would need the preconditioner throughout and churn 1563
  coefficient assertions for no gain) — flagged as such, not silently physical.
- **Pinned-scenario PLAN** (`design/10-physical-scenario.md`): reuse the inbuilt
  `examples/a10_kinetic_example` (large aspect, q=2, β_N=0.10; EFIT g-file + kinetic
  profile a10_prof1.txt, n_i=2e18, T_i≈0.5 keV) via `setup_equilibrium`; at the q=2
  surface derive the **self-consistent** normalized vector (ε, ρ̂_θi=ρ_i q/(ε r_s),
  ν_★ Braginskii, inv_Ln0, η_i, inv_Lq, ψ̃) from the SAME T_i/n_i/B; a
  `physical_domain` fixed at a matching radius independent of `w`; then un-gate B5 as
  the York-replication gate. Reuse existing GPEC kinetic/equilibrium helpers; one
  `[DERIVED]`+sign-off item (the ρ_θi=ρ_i q/ε LAR relation + ν_★ prefactor).
- **Next**: build the scenario (design 10) — the missing foundation that makes the
  `w`-scan and the Q7 far-field/extraction questions physically well-posed. All
  committed + pushed; grids 63/63 green.

## 2026-07-22 (cont.) — Far-field DECAY measurement: the response does NOT localize (falsifies the "match the decaying tail" BC premise) — reframes Q7 to a physics fork

- **Measured** (read-only, converged `Lx=20w` solve, `resmax=9e-11`): the perturbation
  amplitude `gpert_rms(x)` (RMS of the ξ-varying `g` over velocity) is **constant
  ~0.091 across the whole domain** (island → `x/w=16`; ratio to island 1.00/1.01/1.02/
  1.04 at `x/w=2/5/10/15`), even slightly **growing**, snapping to 0 only at the pinned
  boundary. **There is no decaying tail.** The moment integrand `m1(x)=∮J̄cosξ` is
  **small + sign-oscillatory** in the island region (~1e-4) **plus a growing boundary
  spike** (`|m1|` at `x/w=15.95` is 12× the island value, →0 at the wall) — the
  `g_far∝x` Dirichlet-pin boundary layer.
- **Consequence (falsifies my (b) recommendation's premise)**: "match the decaying
  far field" has **nothing to match** — the ξ-structured response is non-localizing
  (constant amplitude), and the σ-odd current carries a BC-induced boundary spike.
  So the ill-conditioned, domain-dependent moment is driven by BOTH a **non-localizing
  response** AND a **BC artifact**, not a matchable decay. Stopped implementing —
  choosing any BC now would be guessing *against* the data.
- **Reframed Q7 (a physics fork, not a BC menu)**: why doesn't the response localize?
  (i) **BC artifact** — the `g_far∝x` Dirichlet pin injects the boundary spike; a
  genuinely non-reflecting/asymptotic-matching outer condition might localize it
  (but there's no clean decay to match, so this is not obviously sufficient);
  (ii) **physical passing-particle tail** — collisionless passing ions stream the
  ξ-structure radially without damping over the domain (LOG 2026-07-15 hypothesis,
  now with direct evidence: constant `gpert`), in which case a volume moment cannot
  converge and the extraction MUST be reformulated (a boundary/jump term), OR the
  collision damping length `~√ν` must enter (is our `ν_★=0.01` / collision operator
  giving the York localization? York's converged results are `ν_★≥5e-3`, comparable —
  so if theirs localizes and ours doesn't, a **term/normalization is missing**);
  (iii) a **spurious weakly-constrained mode** (the constant `gpert` may be an
  under-damped near-null structure, kin to the winged branch).
- **This needs physics judgment / a careful York-formulation comparison, not
  autonomous implementation** — escalated to Q7. The disciplined next step is to
  establish, from York's equations (does THEIR `m=1` response decay, and by what
  mechanism — collisional layer `√ν`, the `S`-streamline boundedness, or a damping
  term we lack?), whether our non-localization is a bug or a genuine model feature —
  then the extraction/BC follows. **Do NOT guess.** Solver/grid stack remains done;
  the blocker is this physics question. Diagnostics scratch; LOG + QUESTIONS committed.

## 2026-07-22 — Extraction diagnostic (user: Lx=20w is unphysical): the moment is ill-conditioned on physical domains; g_far-subtraction ruled out; neither far-field BC localizes

- **User correction (decisive)**: `Lx=20w=1.0` reaches the magnetic axis (`x=−1`) — it
  is the whole inner plasma, not a "far field", and scaling `Lx∝w` breaks the w-scan
  (small→large islands). The huge-domain "convergence" was **over-optimization**: it
  *masked* the non-localization, didn't fix it. **Physics target confirmed (user)**:
  `Δ_neo` should stay **finite** as `w→0` (finite-ion-orbit `ρ̂_θi` regularizes the
  classical `∝1/w` bootstrap divergence; threshold `w_c~O(ρ̂_θi)`; recover `∝1/w` at
  `w≫ρ̂_θi`), NOT zero and NOT divergent — the L23 §6.2 story.
- **Analytic finding (verified by hand + numerically)**: subtracting the `∝x` far
  field `g_far` is an **exact no-op** for `Δ_neo`. `J̄_∥` is a **σ-odd** moment
  (`W=v̂_∥=σ√E√(1−yb)`), and `g_far = x·L̂_{n0}⁻¹[1+(E−3/2)η_i]` is **σ-even** and
  **ξ-independent** ⇒ contributes 0 to `J̄_∥`. Numerics: `Δ−Δpert = 3.6e−14`. So Q7
  option (a) [subtract the background] is **OUT**; the 99% "far-field" is genuine
  **m=1, σ-odd** current that doesn't decay.
- **Extraction diagnostic (read-only, modest FIXED domains `Lx=3–8 ρ̂_θi` × far-field
  mode `:dirichlet`/`:analytic`, w=0.05)**: the result is **negative/sobering** —
  **none converged** (resmax 2e−4…2e−3), and `Δ_neo` is **domain- and mode-dependent
  with sign flips** (dirichlet `+13.5/+8.1/−17.3`; analytic `−27/−4.1/−6.2`). The
  cumulative-moment fractions expose the mechanism: the wild values (`cum<3w = 15.77`,
  `−0.17`) mean `∫m1 dx` is a **tiny residual of large cancelling ± regions** — the
  volume moment is **ill-conditioned** (small difference of large numbers), so those
  `Δ_neo` are numerically meaningless. One case looked localized (`5ρ :dirichlet`,
  97% in 3w) but is **not representative**. `:analytic` does **not** rescue it (more
  erratic).
- **Net (honest)**: on *physical* domains the **m=1 perturbed current does not cleanly
  localize**, and **neither existing far-field BC** gives a converged,
  domain-independent, well-conditioned `Δ_neo`. The far-field domination on huge
  domains and the non-convergence on physical ones are the **same** non-decaying-m=1
  pathology. **Q7 remains the blocker, now sharply characterized**: the fix must make
  the m=1 perturbation genuinely **decay** (a proper far-field BC — bare `:neumann`
  `∂ĝ/∂p=0` failed the winged null mode, so it needs the null-mode anchor; **or** the
  `Δ`-extraction must be a **matched-asymptotic jump** (`Δ'`-style), not a volume
  moment sensitive to the cut). Both are signed-off-physics Q7 decisions — do NOT
  guess. Q7 updated. Diagnostics are scratch (`/tmp`); LOG + QUESTIONS committed.
- **Next (Q7-gated, human decision)**: choose the far-field/extraction reformulation
  (localized-decaying BC with null-mode anchor, vs matched-asymptotic `Δ'`). The solver
  stack + band grid are done; the remaining work is this one physics/numerics decision
  on `docs/01 §4` / `numerics.md §7`.

## 2026-07-21 (cont.) — Converged-solve scan: CONVERGENCE IS SOLVED; Δ_neo is ~99% far-field-dominated (Q7 extraction, now proven on converged solves)

- **Moved**: 2nd Slurm array on the ADEQUATE domain (Lx≥16w, where the solve
  converges), K-convergence + Lx-stability + an **island-fraction** diagnostic
  (fraction of `∫m1(x)dx` from `|x|<3w`).
- **Δ_neo IS roughly K-convergent once the domain is adequate**: at `Lx=20w`,
  `Δ_neo = −1.65, −1.88, −1.80` for K=4,6,8 (~15% spread, all converged
  resmax≤1.7e-8) — vs the oscillating `−24/−14/−22/−12` at `Lx=8w`. Island
  resolution converges when the domain and field are adequate.
- **THE decisive number — `islandfrac ≈ 0.01–0.03` for EVERY config**: only **1–3%
  of `Δ_neo` comes from `|x|<3w`** (the island); **97–99% is the outer region**, the
  m1 cos-moment peaking near the domain edge (`x/w = −13…−19`). And `Δ_neo` is **not
  domain-convergent** — it shrinks `−2.78 → −1.88 → −1.48 → −1.06` as
  `Lx = 16→20→24→28`. So the raw volume-moment `Δ_neo = C∫dx∮dξ J̄cosξ` is
  **far-field-dominated**, NOT the localized island quantity; the ξ-structured current
  does not decay to the boundary.
- **Net (the whole arc resolves)**: **convergence is SOLVED** (domain `Lx≳20w` +
  smoothed band grid + working `PlaneJacobi` → clean converged physical solves, ~5–9
  Newton iters, resmax~1e-11). What remains is exactly the **original Q7 extraction
  question** — now proven on *converged* solves with a hard number (1% island
  fraction), no longer confounded by non-convergence. The far field (`g_far ∝ x`
  Dirichlet) and/or the volume-moment definition let the outer region dominate; the
  fix is a Q7 decision (subtract the ∝x background before the moment / a decaying
  far-field BC / a matched-asymptotic extraction) — **signed-off physics, needs human
  sign-off, do NOT guess**. Q7 updated with the converged-solve evidence.
- **Next (Q7-gated)**: the extraction. With a converged base, the cleanest test of
  each Q7 option is now possible (e.g. does subtracting `g_far` from `g` before the
  moment give a domain-stable, island-localized `Δ_neo`?). Held for the human decision
  on `docs/01 §4` / `numerics.md §7`. Scan is scratch; LOG + QUESTIONS committed.

## 2026-07-21 — Slurm scan (K × domain × band-width): the DOMAIN SIZE is the convergence lever — Lx=20w gives the FIRST cleanly-converged physical solve

- **Moved (user: K-convergence + a parallel resolved-width scan on Slurm)**: fired a
  10-config Slurm array (`LocalQ`, matrix-free on the smoothed band grid, physical
  `ρ̂_θi=w=0.05`) over K, domain `Lx/w`, and band `margin`. Scratch:
  `/tmp/.../bandgrid_scan.jl` + `scan.sbatch`.
- **#2 — K (island resolution) does NOT converge `Δ_neo`** (Lx=8w, margin=1):
  `Δ_neo = −24, −14, −22, −12` for K=4,6,8,10 — **oscillating**, fields not converged
  (resmax~1e-4). Island resolution is **not** the limiter; the band already resolves
  the drift islands.
- **#3 — the DOMAIN SIZE `Lx` is the convergence lever** (K=6, margin=1):
  `Lx/w = 6, 8, 12, 20` → resmax `4.6e-4, 5.6e-4, 1.6e-6, **9.3e-11**`, and
  **`Lx=20w` CONVERGES** (`converged=true`, **9 Newton iters**, `Δ_neo=−1.88`, O(1)
  channels `Δbs=−4.6/Δpol=+2.7`). **The first cleanly-converged physical Level-0 solve
  in this whole effort.** The far field at `Lx=8w` was too close: the drift-island
  envelope reaches ~2w, so the domain must sit *well* beyond it (~20w), not the ~6–8w
  the old resolution protocol used. Widening the band `margin` alone (1→5) does not
  stabilize `Δ_neo` (−13.6, −13.8, −1.1, −7.0) — it is the *outer domain*, not the
  fine-band width.
- **Synthesis / reframe**: the convergence blocker was ultimately the **far-field
  domain being too small**, now fixed by `Lx≳20w` + the smoothed band grid + the
  working `PlaneJacobi`. Chain of the whole arc: gate-on-output (York) → extraction/grid
  (measurement) → drift-island band grid (resolve the envelope) → smoothing (fix
  PlaneJacobi) → **large domain (fix convergence)**. Each was necessary.
- **Still open (now on a CONVERGED base)**: at `Lx=20` the m1 peak is still near the
  domain edge and `Δ_neo` varies with `Lx` — the volume-moment **extraction**
  (original Q7: outer-region-dominated / far-field decay) is the remaining piece, now
  studyable on a *converged* solve for the first time.
- **Next**: re-run the `Δ_neo` **K-convergence at the adequate `Lx=20w`** (does it
  converge in K now that the domain and field are adequate?), and characterize the
  m1(x) integrand on the converged solve (is `Δ_neo` island- or edge-dominated?) →
  informs the Q7 extraction decision. All code + tests committed + pushed; scan is
  scratch (not committed).

## 2026-07-20 (cont.) — Root-caused the band-grid preconditioner failure: an abrupt spacing JUMP; smoothed the tail (WIP, empirical confirm in progress)

- **Diagnosed (dense cond experiments)**: the band grid made `PlaneJacobi`
  **anti-precondition** — on the smooth sinh grid PlaneJacobi drops `cond(M⁻¹J)`
  71× (1.5e6→2.1e4), but on the band grid it was making cond WORSE. Root cause: the
  band grid had a **35× adjacent-spacing jump** (vs the sinh grid's 1.41×) — a
  *sliver* last interval created by the old `banded_x_nodes` clamp (`tail[end]=Lx`
  after the geometric loop overshot). That wrecks the FD conditioning → the plane
  blocks' TSVD inverse injects noise. **Refreshing the preconditioner at the iterate
  did NOT help (1.7×)** — it was never the frozen-ExB; it was the grid.
- **Fix (numerics, no physics)**: rewrote `banded_x_nodes` to build a **graded**
  geometric tail with a single ratio `r ≤ max_ratio` (default 1.3), solved by
  bisection so the tail lands *exactly* on `Lx` — every adjacent ratio is `r`, no
  sliver; a uniform-tail fallback for short tails. New smoothness test asserts
  `max Δx-ratio ≤ max_ratio`. Renamed the `growth` kwarg → `max_ratio`.
- **Confirmed (dense cond)**: on the **smoothed** band grid the max Δx-ratio dropped
  **35.5 → 1.20**, and `PlaneJacobi` now **drops** `cond(M⁻¹J)` **7×**
  (2.3e7→3.2e6) — a genuine preconditioner again, not anti-preconditioning. (Still
  below the sinh grid's 71×, as the band spans more scales — acceptable.) So the
  matrix-free path is unblocked on the band grid.
- **Full islands suite green** (grids 63, configure 1563, anchor 12, operators 30,
  solve 178) — the `banded_x_nodes` rewrite is validated. Committed + pushed
  (`918397ca`).
- **Δ_neo re-test on the smoothed band grid (matrix-free) — substantial improvement,
  not full closure**. Before→after smoothing (K=4,6, physical `ρ̂_θi=w=0.05`):
  - **field residual ~10× lower**: `resmax 2–3e-3 → 1.7–4.6e-4` (matrix-free now
    nearly converges the field, PlaneJacobi working);
  - **`Δ_neo` more stable**: `125%` (sign-flipping +13.8/−55) → **`66%`** (−20.8/−12.5,
    no flip, magnitude *decreasing*) across K=4→6;
  - **m1 peak migrating back toward the island**: edge (`x/w=−7.4`) → **band edge
    (`x/w=−2.2`)** ≈ `±(R+w)`. So the outer-edge artifact is largely gone; the peak
    now sits at the **band→tail transition (~2w)**.
  - **Still open**: `Δ_neo` not yet resolution-converged (66%), and the peak at the
    band edge points at either (a) the preconditioner still moderate (`cond(M⁻¹J)≈3e6`
    after the 7× drop — GMRES + un-captured cross-plane/momentum terms leave a
    ~1e-4 floor), or (b) the physical response extending to ~2w (envelope/domain), or
    (c) a residual band→tail transition effect. Improving but not done.
- **Next (candidate levers, needs a steer)**: (i) strengthen `PlaneJacobi` further
  (capture more cross-plane coupling / a Schur outer block) to push `cond(M⁻¹J)`
  below ~1e5 so matrix-free fully converges; (ii) a K=8 point to see whether `Δ_neo`
  is slowly converging or plateauing; (iii) widen the resolved envelope past 2w if
  the response genuinely extends there. Scratch scripts in `/tmp`; all code + tests
  committed + pushed.

## 2026-07-20 — Drift-island band grid built (04 §1): the response layer's velocity-spread envelope reaches ~4w, so the magnetic-island-centred grid missed it

- **Moved (user: "build the drift-island grid map")**: implemented the band grid
  that resolves the **drift-island shift envelope**, not the magnetic island.
  - **Envelope quantified first (the scientific gate)**: at physical
    `ρ̂_θi=0.05, w=0.05`, `X = max_passing|x_D^island| = 0.197 = 3.94·w`. The drift
    islands sit at `x~±0.2`, entirely OUTSIDE the old fine region `[-w,+w]=[-0.05,0.05]`
    — in the coarse tail. Hard confirmation of the 2026-07-19 diagnosis (why `Δ_neo`
    was resolution-divergent and the m1 peak migrated).
  - **Doc contradiction found + resolved**: the mesh shift is
    `x_D^island = ρ̂_θi ω̂_D L̂_q` — docs/01 §2.2 (`[CLEARED]`) and design 04 §1 both
    say **L̂_q**; `analytic-far-field.md §4` (a `[DERIVED]` parenthetical) wrote
    `L̂_q⁻¹`. Structurally the L̂_q form is right (its shear part = `⟨x_D⟩` orbit width,
    since `L̂_q` cancels the `1/L̂_q` inside `ω̂_D`). Corrected the outlier typo; the
    CLEARED docs/01 §2.2 is authoritative. Flagged, not silently resolved.
  - **Implemented** (reuses the **cleared** `ω̂_D` — no new coefficient):
    `Configure.drift_island_shift_envelope` (`R = max_passing|ρ̂_θi ω̂_D L̂_q|`,
    passing `y<y_c` only — trapped have no shifted island, 01 §2.2) and
    `drift_island_resolved_grid`; `PhaseSpace.banded_x_nodes` /`drift_island_grid`
    (uniform central band at `Δx≤w/K` over `[-(R+w),R+w]` + geometric tails to
    `±Lx`, `Lx` beyond the envelope) and a `MappedFDGrid(nodes;order)` constructor
    (Fornberg D1/D2 on arbitrary nodes; x-`wq` trapezoidal — unused downstream, the
    Δ x-integral uses spline quadrature).
- **Verified**: grids 63/63, configure 1563/1563, anchor-sync 12/12, operators 30/30,
  solve 178/178 green. Doc-first: numerics.md §1 (band-grid subsection + implementing
  symbols); design 04 §1 already prescribed it.
- **physics-verifier: BLOCK → resolved**. Verdict: the physics is faithful (L̂_q
  direction correct, no guessed coefficient, passing-only correct, no leaked values),
  but I had **mislabeled the shift structure `[CLEARED]`** — docs/01 §2.2 tags it
  `[CHECKED: I19 Eq. 33; D21 Eq. 21; Diss19 Eq. 2.37]` (the erratum-prone I19 lineage;
  the `ω̂_D` *coefficient* is `[CLEARED]` §2.1, but the *shift form* is not signed off).
  Fixed the tag in `Configure` docstring + numerics.md + analytic-far-field.md, and
  **escalated to QUESTIONS Q8** (clear the `x_D = ρ̂_θi ω̂_D L̂_q` structure). Mitigation
  recorded: the `[CHECKED]` shift enters only to **size** the grid (margin-protected),
  never as a physics output value — the grid map's correctness gate is the empirical
  `Δ_neo` convergence test, not the shift value.
- **Cost note**: the band is ~5× wider than the magnetic island, so ~5× more x-nodes
  at the same Δx (nx~100 to resolve the envelope) — kokuchou's memory wall; the
  matrix-free solve (PlaneJacobi) is the intended absorber.
- **Payoff test (band-grid `Δ_neo` resolution) — band grid VALIDATED but not
  sufficient; blocker now precisely located at the matrix-free preconditioner**:
  - **matrix-free** (`newton_krylov`+`PlaneJacobi`) on the band grid: `Δ_neo` still
    unstable (K=4→6: `+13.8 → −55`, sign flip), with the m1 peak at **x/w=−7.36 ≈
    −0.9·Lx** — the domain EDGE, i.e. the documented PlaneJacobi outer-region
    artifact, NOT the drift islands (which are at ±R inside the resolved band).
  - **exact** `newton_direct` on the band grid (small N to be tractable): the moment
    **LOCALIZES at the island** (m1 peak x/w=−0.33, +1.25 for K=3,4) — confirming the
    edge feature WAS the matrix-free artifact and the band grid puts the moment where
    the physics is. **But** the exact solve is now **poorly conditioned on the band
    grid** (resmax 0.1–0.4; K=4 hit max_iter at 26 min) — more band nodes worsen the
    conditioning/cost, so `Δ_neo` still isn't resolution-stable (−146 → −200).
  - **Net**: the band grid is a **necessary, validated** piece (drift islands were
    unresolved; the exact solve now localizes on it), but **not sufficient** — a
    convergent, resolution-stable `Δ_neo` is now blocked on the **solve** on the band
    grid: matrix-free injects the outer-edge artifact (`PlaneJacobi` outer-region
    accuracy, flagged 2026-07-18) and exact is too slow/ill-conditioned at the band's
    N. This is exactly the milestone premise (island resolution fights dense-direct ⇒
    matrix-free mandatory) — the remaining work is **fixing the matrix-free
    preconditioner's stiff outer-streaming block so it stops injecting the edge
    feature** on the band grid. Candidate refinement: a smoother band→tail transition
    (the current uniform→geometric spacing jump may hurt the FD conditioning).
- **Next (needs a steer)**: improve `PlaneJacobi`'s outer-region accuracy (or a
  sparse/banded exact solve exploiting the operator structure) so the matrix-free
  `Δ_neo` on the band grid localizes AND converges — then B2 `Δ_neo(w)` is reachable.
  Scratch scripts in `/tmp` (not committed). Everything else committed + pushed.

## 2026-07-19 (cont.) — Kokuchou Δ_loc measurement: the OUTPUT extraction (not the solve) is the blocker — redirects the 2026-07-18 pivot

- **Moved (user: "measure Δ stability now")**: ran the kokuchou `Δ_loc` test at a
  physical point (`ρ̂_θi=0.05`, `w=0.05`, `ŵ/ρ̂_θi=1`, Lx/w=6, nξ=8, ny=9, nE=3,
  `:dirichlet`) — does `Δ_neo` stabilise across x-resolution as York's `Δ_loc` does?
  Two solve paths (matrix-free `newton_krylov`+`PlaneJacobi`; exact `newton_direct`).
  Scratch scripts under `/tmp` (not committed, per discipline).
- **Two clean signals**:
  1. **At fixed resolution, `Δ_neo` is insensitive to the FIELD residual** — nx=13
     gave `Δ_neo=−434` at `resmax=1.0e-5` (60 iters) and `−441` at `resmax=0.15`
     (45 iters). This **confirms the ground-truth**: gating on field `resmax` is the
     wrong lever; the output barely moves as the field residual falls 4 orders.
  2. **Across x-resolution, `Δ_neo` does NOT converge** — exact solve
     `Δ_neo = −435 → −38 → −9.3` (nx=13,17,21; ~1–2 orders of magnitude), with the
     `m1(x)=∮J̄cosξ` peak **migrating off the magnetic-island centre**
     (`x/w = +0.48 → −0.32 → −0.91`). Not a field artifact (per signal 1). The
     channels are huge and cancelling (`Δbs≈+1065`, `Δpol≈−1506` at nx=13) — the
     documented item-4 fragility, now at physical w.
- **Diagnosis (redirects 2026-07-18)**: the **exact solve CONVERGES the field**
  here (`resmax=1e-5` at nx=13) — so the field is **not** the universal blocker the
  2026-07-18 LOG concluded; that non-convergence was largely the **matrix-free
  (PlaneJacobi) path** stalling at ~1e-3, not the underlying problem. **The real
  standing blocker is the OUTPUT EXTRACTION / non-localisation** — the volume-moment
  `Δ_neo` is resolution-sensitive (collapsing/peak-migrating), exactly Q7-**original**
  (2026-07-16: island-restricted moment → ~0 while full value grows) + milestone
  item-4 (`channel_decomposition` fragility) + **design 04 §1**: the response layer
  sits at the **drift** island (shifted `ρ̂_θi ω̂_D(y,σ,u)`, spread over velocity
  space), NOT the magnetic island our rectilinear x-mesh packs — a magnetic-island-
  centred grid structurally cannot resolve a moment whose support moves across
  velocity space (L23 §3.1.6, the same rectilinear-mesh-vs-rounded-drift-island
  mismatch that was kokuchou's dominant accuracy limiter).
- **Caveats (honest)**: only 3 coarse resolutions (nE=3); the max_iter=45 trend runs
  weren't fully field-converged (signal-1 shows that doesn't change the conclusion).
  Whether `Δ_neo` is heading to a small finite value (under-resolution artifact
  shrinking) or genuinely non-convergent is not disambiguated (would need nx≥25).
- **Blocked / next (needs a steer — a real build, not a probe)**: the concrete lever
  named by design 04 §1 + L23 §7.1.1 is a **drift-island-separatrix grid map**
  `x(s; y,v̂,σ-envelope)` absorbing the orbit shift `p̃ = ψ − I(v_∥/ω_c)` (as a GRID
  MAP; D1 stands — not a solve coordinate), so the moment's velocity-spread support
  is resolved. This is the honest precondition for a resolution-convergent `Δ_neo`;
  it supersedes further solver-robustness work. Surfaced to the user. Docs-only this
  session (LOG + QUESTIONS Q7 update); nothing in `src/`.

## 2026-07-19 — York ground-truth (recommended first task): the FIELD converges for nobody; the OUTPUT `Δ` is what York gates on — our resmax~1e-3 stall is NOT a bug

- **Moved (literature ground-truth, per the M1-launch recommendation)**: read L23
  (Leigh 2023) §3.1.5/§3.1.6/§6.1.1/§6.2 and Diss19 (Dudkovskaia 2019) Ch. III/IV
  first-hand (not just the LOG's earlier note). Full synthesis with page cites:
  `docs/src/islands/notes/york-convergence-ground-truth.md`. Answers the blocking
  question "does the discretized problem converge for anyone?": **not the
  self-consistent field — for nobody.**
- **kokuchou (L23) — the direct 4D `{p,ξ}` solve, our closest analogue** — uses
  **Picard** (not Newton), tolerance **ε¹≈10%** (justified by the `O(ε^{3/2})`
  equation accuracy). L23 §6.1.1 p.118: that 10% criterion **never met in any run**
  (max 4 iters); `Φ̂` iterative residual **>100%/iter** across the whole physical
  E×B regime. **Yet `Δ_loc` (the OUTPUT) converges stably** (§6.2 Fig.6.3) despite
  the non-converging field. Cause = the `O(ρ̂_θi)` drift-island separatrix layer that
  sits at `x` shifted by `ρ̂_θi ω̂_D(y,σ,u)` (varies over velocity space) AND moves
  with `Φ̂` between iterations → a rectilinear single-location mesh can't track it.
- **RDK-NTM (Diss19)** reports converged `Δ` only via (i) the `S`-streamline
  coordinate (analytic layers following the drift island; low-`ν_★` only) and
  (ii) reporting headline `Δ` (bootstrap ∝1/w) at the **"0th iteration in Φ"
  (Φ=0, E×B off)** — i.e. pre-nonlinearity.
- **Reframe of the blocker**: our **resmax~10⁻³ field-residual floor is ~100× BELOW
  what kokuchou achieved and ~1000× below York's own 10% criterion** — it is the
  documented, universal, physics-rooted moving-E×B-layer non-convergence, not a bug.
  We have been gating on the **wrong quantity** (field residual, not the output `Δ`)
  at an **unphysically tight tolerance**. The two robustness levers the LOG had only
  *suspected* (E×B-coupling continuation; and grid packing) are now **named by the
  ground truth**: continuation from `Φ=0` (RDK-NTM's 0th iteration) and packing the
  mesh at the **drift-island separatrix** (design 04 §1), not the magnetic island
  (our current mesh packs the wrong contour).
- **Blocked / escalated**: adopting York's **output-convergence** posture as our
  definition of done — gate on `Δ_neo`/current-moment *stability* at the few-%
  (`O(ε^{3/2})`) level instead of `resmax~10⁻³` — is a **methodology/threshold
  decision** → written up in **QUESTIONS Q7 (2026-07-19 update)** for human sign-off
  (recommendation: adopt output-convergence + pursue the two named numerics levers;
  always report `resmax` as a diagnostic; never weaken the science to reach "done").
- **Next (unblocked — measuring needs no sign-off)**: re-take `Δ_neo` at physical
  `ŵ ~ ρ̂_θi` with the fixed spline `delta_moments` and test whether it (and
  `⟨J̄_∥cosξ⟩`) **stabilises across resolution even as `resmax` floors** — the
  kokuchou `Δ_loc` test. Only the *gate change* itself awaits sign-off. Nothing
  changed in `src/` this session; docs-only (note + QUESTIONS + LOG).

## 2026-07-18 (cont. 2) — Globalization hypothesis NOT confirmed: the exact solve crawls to resmax~1e-3 regardless of initial guess

- **Test**: `newton_direct` at physical `w=0.03` from ZEROS vs the FAR-FIELD
  extrapolation (`g = x·slope` everywhere — satisfies the BC). Result (nx=21):
  both **fail to converge** at max_iter — zeros `resmax=2e-2`, far-field
  `resmax=2.5e-3`. The good guess helps (~10× lower residual) but does **not** crack
  it; `Δ` differs wildly (−79 vs −29), so neither is trustworthy. **Globalization
  (initial guess) is NOT the fix.**
- **State (honest)**: after extensive investigation this session, the physical
  Level-0 solve does **not robustly converge** below `resmax ~ 10⁻³` under any path
  tried — matrix-free (edge artifact; fails at small w) or exact `newton_direct`
  (crawls, from any init). Ruled OUT: the far-field BC (B/A were the wrong lever),
  the `y_c` matching block (A8), a genuine null mode / near-singularity (cond only
  ~10⁶–10⁷ at u=0), and simple globalization. Confirmed OK: the response localizes
  (exact solve, m1 peak @ island), the quadrature/moment/extraction. **Remaining
  suspicion**: the crawl (exact Newton, moderate cond at u=0, but residual won't fall)
  points at either the **strong nonlinearity** (the ExB Poisson-bracket term — Newton
  overshoots, Armijo damps to a crawl) or conditioning worsening near the solution, or
  a **consistency** question (does a discrete root exist at this resolution?). Not
  resolved.
- **Assessment**: I have spent a large amount of compute/experimentation without a
  converged physical solve; the experimental poking is not converging on the cause.
  **Handing back for a strategic decision** rather than more autonomous solves.
  Options: (i) a focused **solver-robustness** effort (continuation in the ExB
  nonlinearity / a trust-region or pseudo-transient Newton, vs the current
  line-search) — the crawl smells like a globalization-of-the-nonlinearity problem
  the current Armijo can't handle; (ii) a **ground-truth comparison with York's
  actual numerics** (kokuchou/DK-NTM: what residual tolerance + method did they
  reach? Picard may have a looser criterion, or their small domain matters); (iii)
  human structural/physics judgment. **Landed + solid this session (all committed,
  tests green, non-regressing)**: the `delta_moments` quadrature fix, `PlaneJacobi`,
  `natural_continuation`, the resolution protocol, and the 3-mode far-field toggle
  (A `[DERIVED]` + physics-verifier PASS) — correct infrastructure, but the
  convergent physical `Δ_neo` remains unachieved.

## 2026-07-18 (cont.) — A8 conditioning investigation: NOT near-singular, NOT the y_c layer — the stall is globalization + a matrix-free edge artifact (hopeful)

- **A8 monitor result (overturns the y_c hypothesis)**: the `y_c` trapped-passing
  block is **NOT** the near-singularity — its smallest singular value (1.7e-4…5e-4)
  is **43–1230× LARGER** than the whole-Jacobian σmin (1e-7…1e-5). The near-null
  direction is elsewhere, ~99% in `g` (not `Φ`, not the forbidden-y rows).
- **No genuine null mode**: σmin is **bounded** (~1–2e-5, does not shrink with
  resolution nx=9→21) and cond is only **~4e5** (mild grid) to **~6e6** (resolved,
  heavily-clustered β=3.55 grid — clustering for the tiny `w=0.03` island worsens
  cond ~14× but only to ~10⁶–10⁷). **This is well within `newton_direct`'s exact-LU
  capability** (a step on cond~10⁷ is accurate to ~10⁻⁹).
- **Reframed diagnosis (much more hopeful)**: the physical solve is **moderately
  conditioned, not near-singular**, and the `y_c` layer is fine. So `newton_direct`
  stalling at `resmax≈7e-3` (w=0.03, cold from zeros) is a **globalization /
  initial-guess** problem — Newton from zeros at small `w` misses the basin — NOT a
  singularity or conditioning failure. This is exactly what `Solvers.natural_continuation`
  (already built + tested) is for. And the **exact solve LOCALIZES correctly**
  (m1 peak @ island) — so the moment/far-field are fine; the outer-**edge** feature
  was purely a **matrix-free (PlaneJacobi) accuracy artifact** in the stiff outer
  streaming region (a preconditioner-quality issue, separable).
- **Net**: two addressable numerical issues, both with existing tools — (a)
  globalize the solve with **continuation** (reach the small-`w` / large-`w` basins
  robustly), using the **exact `newton_direct`** at modest grids (it localizes
  correctly); (b) if the matrix-free path is needed for scale, **improve the
  PlaneJacobi outer-region accuracy**. NOT a far-field-BC problem (B/A were the wrong
  lever, though A is a correct verified toggle). **Next**: confirm the globalization
  hypothesis — `newton_direct` + `natural_continuation` reaching a converged w=0.03
  solve — then a resolved `Δ_neo(w)` via the exact-solve path.

## 2026-07-18 — Exact-solve diagnostic: the edge feature IS a matrix-free artifact, but the physical solve is near-singular (deeper than the BC)

- **Test**: `newton_direct` (EXACT dense solve, no preconditioner) at physical
  `w=0.03` (≈0.6 ρ̂_θi), thin `Lx=0.18`, to remove the matrix-free solver as a variable.
- **Two decisive findings from the first (converged-enough to read) row** (nx=25,
  N=5550): (1) the exact solve puts the **m1 peak at the ISLAND (x/w=0.00)**, NOT the
  edge — so the "edge feature at ~0.9·Lx" seen in every matrix-free sweep **was a
  matrix-free (PlaneJacobi/Krylov) artifact**, concentrated in the stiff outer
  streaming region. Good: the response *does* localize. (2) BUT the exact solve
  **does not converge** either — the Armijo line search **stalls at resmax≈7e-3**
  (it=54/60), i.e. the Newton step stops reducing the residual. A dense-LU exact
  solve stalling ⇒ a **near-singular Jacobian** at physical parameters. This is
  almost certainly the **documented `y_c` trapped-passing matching layer** (design
  `04 §3`, ladder A8: the prior art measured rcond~1e-16 there and needs TSVD
  regularization) — `newton_direct` uses plain LU with NO regularization, so it
  stalls exactly where the TSVD-preconditioned `newton_krylov` copes.
- **Synthesis**: neither solver is clean — `newton_direct` (exact) stalls on the
  `y_c` near-singularity (no TSVD); `newton_krylov+PlaneJacobi` (TSVD) copes with
  `y_c` but injects an outer-edge artifact and doesn't converge at small `w`. The
  **Δ_neo non-convergence is a SOLVE problem, not a far-field-BC problem** — the far
  field (B/A) was the wrong lever. A remains a correct, verified physics piece (the
  toggle), but it is not the fix.
- **Cost/limit**: `newton_direct` is ~11 min/solve at N~5.5k (N column-evals/step) —
  too slow for sweeps. Many experiments run this session (compute-heavy) without a
  clean convergent Δ_neo. **Stepping back for a human steer.** Principled next
  diagnostics (not yet run): (i) the **A8 `yc_block_sigma_min` monitor** on the
  physical dense Jacobian to confirm/quantify the near-singular `y_c` block — if it is
  the `y_c` layer, the fix is regularizing the *solve* (TSVD in the direct path /
  tuning the plane-block regularization), not the BC; (ii) replicate York's exact
  converging small-domain setup as a reference to isolate our-formulation vs a bug.

## 2026-07-17 (cont. 3) — Physical-regime test: the edge feature is a domain-FRACTION artifact (scale-invariant), not a plasma-scale effect

- **Context (user catch)**: the convergence sweeps used `w_psi=1.0`, `Lx=6` — but
  `x=(ψ−ψs)/ψs` so `|x|~1` is the plasma edge; `w=1`/`Lx=6` is an island the size of
  the plasma and a domain 6× the minor radius, far outside the local model. Retested
  in the **physical** regime `w=0.1` (≈2ρ̂_θi, ≪1), thin domain `Lx=0.6`.
- **Result — does NOT dissolve; it is scale-invariant**: `:dirichlet` at `w=0.1`
  **fails to converge** (nx=41,61 both conv=0, it=40 max; Δ_neo swings 0.84→−14.8),
  and the m1 peak is at **x/w ≈ −5.3…−5.6**, i.e. ≈0.9·(Lx/w=6) — the SAME domain
  *fraction* (near the outer edge) as the `w=1` case (peak @ x=−5.78 = 0.96·Lx). So
  the edge feature scales with the **domain**, not the absolute `x`: it is an
  **outer-boundary artifact at ~0.9·Lx, independent of w**, NOT a plasma-scale
  (`x~1`) effect. And physical `w` makes the solve *harder* (non-convergent), not
  easier.
- **Where this leaves the Δ_neo convergence**: none of the tried fixes crack it —
  the quadrature spline (helped, insufficient), `:neumann`/B (winged null mode),
  `:analytic`/A (offset ~1% too small), physical scaling (worse). The feature is at a
  fixed fraction near the outer boundary and *sharpens with resolution*. Open
  hypotheses (not yet distinguished): (i) a **solve-accuracy** artifact — the outer
  region has the largest streaming `a_xi ∝ x` (stiffest, worst-conditioned), so the
  finer grid's residual concentrates there (would explain non-convergence + the edge
  peak, and points at the preconditioner/tolerance, NOT the BC); (ii) the response is
  genuinely **not localized** in this model (long passing-particle radial tail), so a
  volume moment over any finite domain is edge-dominated — then York's *small* domain
  (2–3w, island-dominated) is essential, opposite to our 6w. **Stepping back for a
  human steer rather than more autonomous solves.** Committed to date:
  quadrature-fix, the 3-mode far-field toggle (dirichlet/neumann/analytic, A
  `[DERIVED]`); none regress — they are correct pieces that don't (yet) yield a
  convergent Δ_neo.

## 2026-07-17 (cont. 2) — A resolution sweep: A is correct + well-posed but does NOT fix the boundary layer (negative result)

- **Result**: the `:analytic` resolution-convergence sweep came back **negative**.
  `:analytic` converges cleanly (unlike `:neumann`) and localizes at nx=41 (m1 peak
  @ island, Δ_neo=−1.13), but at nx=61 it **diverges just like `:dirichlet`**:
  Δ_neo=−2.94 (dirichlet: −2.84) with the m1 peak back at the domain **edge**
  (x=−5.78) — the boundary layer returns. `:analytic` and `:dirichlet` are nearly
  identical at both resolutions. **The hypothesis that the dropped `O(ρ̂_θi)`
  orbit-shift offset causes the boundary layer is WRONG**: `⟨x_D⟩` is only ~1% of the
  far-field value at |x|=Lx, far too small to change the layer. The boundary layer is
  a larger, different effect — intrinsic to pinning the *value* (Dirichlet) of a
  *linear* far field at a domain that is (apparently) too small for the interior to
  have reached its asymptote, and it *sharpens with resolution* (absent at nx=41,
  present at nx=61).
- **Standing**: A remains a **correct, physics-verified, well-posed** improvement
  (the true far field to `O(ρ̂_θi)`; no null mode) — committed `186893e8`, valuable as
  the toggle/comparison even though it is not the convergence fix. The
  `analytic-far-field.md` `[DERIVED]` sign-off item is unchanged.
- **Reassessment / next**: the convergence blocker is NOT the far-field *offset*.
  Leading candidate now: **domain size** — at `Lx/w=6` the response has not decayed to
  the linear asymptote, so any linear value-BC (dirichlet or analytic) is incompatible
  → a resolution-sharpening boundary layer. Test: `:analytic`/`:dirichlet` at
  `Lx/w=12,20` on near-uniform grids — does the layer vanish and Δ_neo converge? If
  yes, the fix is "adequate domain + A"; if no, the far field needs more than
  linear+offset (full asymptotic matching) or a different treatment. Surfaced to the
  user; `Δ_neo(w)` checkpoint / B2 still gated on this.

## 2026-07-17 (cont.) — A landed: the `:analytic` drift-orbit-shifted far field (derived from cleared quantities)

- **Moved (user: "derive + implement A")**: derived and implemented the `:analytic`
  far-field mode — the principled Q7 fix. The exact far field is the neoclassical
  response at the **canonical momentum** `p̂ = x − x_D` (I19 Eq. 2, `p_φ=(ψ−ψs)−Iv_∥/ω_c`);
  the code's `:dirichlet` uses the leading `p̂ ≈ x`, dropping the `O(ρ̂_θi)` orbit-width
  term (which `gradient-drive.md` §2 explicitly discarded as "a small shift at |x|=Lx"
  — the boundary-layer evidence shows it is NOT negligible). `:analytic` restores it:
  `g_far = (x − ⟨x_D⟩_θ)·slope`, `⟨x_D⟩_θ = ρ̂_θi(σ√E/(1+ε))A(y)` with the **cleared**
  drift bracket `A(y)=⟨√(1−yb)/b⟩_θ` (the boxed `x_D` of the signed-off
  `omega-D-drift-frequency.md` §2). **No new coefficient** — every factor is already
  cleared. Well-posed (Dirichlet → pins the level, no winged-branch null mode) AND
  accurate (true asymptote → no boundary layer). Applied as a `:dirichlet` value
  condition. Doc-first: `derivations/analytic-far-field.md` `[DERIVED: 2026-07-17]`;
  Configure docstrings.
- **physics-verifier PASS**: every derivation link confirmed against primary sources
  (I19 Eq. 2; the boxed `x_D`; `A(y)` = the cleared bracket's first element; the term
  `gradient-drive.md` §2 dropped), no new/guessed coefficient/sign/normalization,
  `:dirichlet` default untouched, forbidden-y/`y_c`-miss guarded (zero shift, not
  guessed). The one item reserved for **human sign-off**: the orbit-*averaged* `⟨x_D⟩_θ`
  choice (vs local `x_D(θ)`) — correctly gated by `[DERIVED]`.
- **Numerics (early)**: `:analytic` **converges** (nx=41 conv=1, m1 peak @ island x=0.00,
  Δ_neo=−1.13) — already beating both baselines: `:dirichlet` diverges (−1.05→−2.84,
  peak→edge) and `:neumann` doesn't converge (winged branch). The resolution-
  convergence sweep (nx=61,81) is running to confirm Δ_neo settles. Tests: 1563
  configure (was 1540; new `:analytic` test asserts the exact cleared formula + σ-oddness
  + forbidden-y-no-shift + back-compat).
- **Next**: finish the A resolution-convergence sweep → report A-vs-dirichlet (the
  paper comparison); then, on human sign-off of `analytic-far-field.md`, the resolved
  `Δ_neo(w)` checkpoint / B2 become reachable.

## 2026-07-17 — Far-field BC toggle (B = York localized ∂g/∂x form) landed; A next

- **Moved (user-directed: "do B as a toggle to plot A's impact in the paper; if B
  works, proceed straight into A")**: implemented the far-field **mode toggle** on
  `Operators.FarFieldConditions` — `:dirichlet` (default, unchanged: pin the value
  `g → g_far ∝ x`, the I19-Formulation-A drive-in-BC form) vs `:neumann` (pin the
  **slope** `∂g/∂x → s_far`, the York/kokuchou localized form `∂ĝ/∂p=0` for
  `ĝ = g − g_far`, so `g` floats by a constant and reaches its own asymptote instead
  of forming the diagnosed edge boundary layer). `apply_farfield!` branches (Neumann
  via the x-grid `D1` boundary row); `Configure.gradient_far_field(...; mode)` builds
  the value or the slope (`s_far = L̂_{n0}⁻¹[1+(E−3/2)η_i]` = ∂/∂x of the *same*
  cleared far field — no new coefficient); `configure_level0(...; farfield_mode)`
  threads it. **physics-verifier PASS** (physics-neutral: `:dirichlet` a no-op,
  `:neumann` slope is exactly the derivative of the cleared value, no coefficient/
  sign/normalization introduced, no tag cleared). Doc-first: numerics.md §4 +
  docstrings + `configure_level0`/`gradient_far_field` docs; new `:neumann` unit test
  (D1-row slope residual + constant-annihilation + default-`:dirichlet` back-compat).
  178 solve (was 169) + 1540 configure + 5 anchor-sync green.
- **B experiment result — bare `:neumann` FAILS the winged-branch (expected)**: at
  `w=1`, physical `ρ̂_θi`, near-uniform grids, `:neumann` (symmetric slope-pinning) does
  **not converge** — nx=41 `conv=0` (Δ=0.15, m1 peak@edge), nx=61 `conv=0` (Δ=7.37,
  m1 peak@island); the values swing wildly, the signature of Newton wandering an
  unconstrained null space. Root cause (as anticipated + design-doc/L23 warned): pure
  symmetric Neumann leaves `g`'s absolute level undetermined — an additive-constant
  null mode `g→g+c(E,σ)` for σ-even `c` with zero density-moment (quasineutrality's
  `M[g]` and the σ-odd momentum term don't see it) → singular Jacobian → the solve
  stalls. Dirichlet over-constrains (boundary layer); bare Neumann under-constrains
  (null mode) — neither works. So the **"if B works, proceed to A" precondition is
  false**; reassessed rather than auto-proceeding.
- **Reassessment**: the principled fix is **A** — pin the value to the *correct*
  asymptote (linear `∝x` **plus** the drift-island-shift correction), which is
  well-posed (pins the level → no null mode) AND accurate (the true asymptote → no
  boundary layer). This is exactly the "analytic far-field BC" L23 §7.1 lists as
  **unimplemented future work** (tied to the drift-island shift `ρ_shift`/`p̃`
  coordinate) — genuinely novel beyond York, a `[DERIVED]` + physics-verifier +
  human-sign-off task. Bare `:neumann` (B) stays in as the *baseline that exhibits the
  winged-branch* (the paper motivation for A); making it a converging baseline would
  need a numeric null-mode anchor (a Robin/one-sided regularization — a knob).
  Surfaced to the user for the path decision (anchor-B-as-baseline vs invest in A's
  asymptote derivation); the toggle infrastructure (committed `f8b519d1`) supports
  either.

## 2026-07-16 (cont.) — York cross-check + integration experiment: quadrature fixed (delta_moments spline); far-field BC is the residual blocker

- **York source check (from the papers; the codes are NOT public)**: Diss19
  (Dudkovskaia thesis) Eq. 4.12 defines `Δ_neo = −(μ₀R/2ψ̃)∫dψ∫dξ J̄_∥cosξ` —
  **exactly our volume moment** (bs/pol split Eqs 4.13–4.15 = our
  `channel_decomposition`), balanced `Δ₀+Δneo=0` with `Δ₀` the **outer jump**. Their
  integral **converges** (Figs 4.13–4.15 → bootstrap ∝1/w). kokuchou (L23): a **small
  domain (~2–3 island widths)**, a **uniform high-res central region** covering the
  drift-shifted island, and far-field **`∂ĝ/∂p=0` (ĝ→const, self-matching)** — plus
  L23 §5.10: `h(Ω)→x` cancels the `xL̂_{n0}⁻¹` drive so the QN source → 0 far out;
  everything localizes → the perturbed current decays → the moment converges. **So the
  extraction FORM is confirmed correct** (Q7 option (ii) "matched jump" is NOT what
  York does — the jump is Δ₀). No wholesale conversion to York: we extend their physics
  (e.g. Diss19 notes the polarization's *external* contribution is comparable-and-
  opposite to the inner), so we want a flexible domain + robust integration, not their
  tight box.
- **3-way integration experiment (user-approved)**: on converged matrix-free solves,
  integrate the radial Δ moment via (1) Simpson (`grid.x.wq`) vs (2) cubic-spline
  quadrature (`FastInterpolations`, the repo idiom) of the same `m1(x)=∮J̄cosξ`;
  clustered vs near-uniform grids. Findings: (a) **quadrature RULE matters** — Simpson
  vs spline differ 3× (sign-flip on clustered grids) because Simpson over-weights the
  coarse far-field nodes of the center-clustered grid. → **fixed** (below). (b) But
  accurate integration is **necessary-not-sufficient**: on the two converged (uniform)
  grids the spline value still diverges with resolution (−1.05 → −2.84) and the `m1`
  peak **migrates from the island (x=0) to the domain edge (x=−5.78)** — a
  resolution-sharpening **boundary layer** at the far-field boundary that contaminates
  the whole solve (island `|x|<2w` collapses 0.96→0.17). `m1` is pinned ~0 at the very
  edge (BC) with a growing spike just inside. **Diagnosis**: our Dirichlet far field
  `g_far = x·L̂_{n0}⁻¹[1+(E−3/2)η_i]` (∝x) over-constrains the edge; the interior does
  not self-match it, so a boundary layer forms and sharpens — the far-field issue L23
  §7.1 flagged (analytic far-field BC), and why York's `∂ĝ/∂p=0` localizes and ours
  doesn't.
- **Landed (this commit)**: `Moments.delta_moments` x-integration swapped
  Simpson→cubic-spline quadrature (`FastInterpolations`), removing the clustering
  artifact; the definition/prefactors/ξ-projection are unchanged. **physics-verifier
  PASS** (pure quadrature swap, physics-neutral: no coefficient/sign/normalization/
  definition change; parity + additive-split preserved). Doc-first numerics.md §7 +
  docstring (scoped explicitly as "quadrature only; convergence still needs the
  far-field fix, Q7"). 169 solve + 1540 configure + 5 anchor-sync green;
  `build_docs_local.jl` green.
- **Escalated / next**: **Q7 updated** — the quadrature half is done; the **far-field
  BC is the standing physics decision** (analytic far-field à la L23 §7.1 vs a
  York-style localized `∂ĝ/∂p=0` with the winged-branch fix vs domain mitigation),
  written up as concrete options for human sign-off. NOT implemented — `g_far` is
  signed-off physics. `Δ_neo(w)` checkpoint / B2 / item-4 remain gated on the far-field
  decision, NOT on the extraction form or the quadrature.

## 2026-07-16 — Option (a) tested → ruled out: Δ_neo moment is not resolution-convergent; escalated as Q7

- **Moved**: ran the user-approved option-(a) test (does a bigger box / island
  restriction converge `Δ_neo`?). It does **not**, and the diagnosis is now sharp
  enough to escalate. Evidence (matrix-free solves, `w=1`, physical `ρ̂_θi`):
  - **Lx-sensitivity**: growing `Lx/w = 6→12→20` does not converge `Δ_neo`; at fixed
    central `K` the bigger box starves the outer region and the solve stops
    converging (`Lx/w=12`, `conv=0`). The response `m1(x)=∮J̄cosξ` **decays outward**
    (RMS ~30× smaller at the edge) — physically localized — but the outer bands still
    contribute 15–32% of `Δ` because the center-clustered grid puts **huge Simpson
    weights** on the small tail.
  - **Cutoff-restriction (decisive)**: `Δ_neo` restricted to `|x|<{1.5,2,3,4}w` across
    `K=8,12,16` **shrinks toward ~0** (all ≲0.1 at K=16) while the full-domain value
    **grows** 1.40→1.74→2.20. So essentially **all** the reported `Δ_neo` is the
    spurious outer-tail quadrature artifact; the genuine island cos-moment is small
    and resolution-noisy. Restricting to the island does **not** give a converged
    value either.
- **Conclusion**: the `Δ_neo = C∫dx∮dξ J̄cosξ` **volume-moment extraction itself is
  the blocker**, not the solver (which is done: PlaneJacobi + continuation +
  resolution protocol converge cleanly past the dense cap). Candidate (a)
  [box/quadrature only] is insufficient; the issue also touches candidate (c) [the
  extraction definition — volume moment vs matched-asymptotic `Δ'` jump; docs/01 §4].
  This is a physics/normalization decision, not a guessable numerics tweak.
- **Escalated**: wrote **QUESTIONS Q7** — the intended `Δ_neo` extraction + quadrature
  (volume moment with a resolved outer region + island restriction + decaying far
  field; vs a matched-asymptotic jump; vs the small island value being physical with
  the `∝1/w` target on a different channel). A moment/output change → physics-verifier
  before any implementation. Nothing changed in `src/`.
- **Next**: **blocked on Q7** for the `Δ_neo(w)` checkpoint, B2, and item-4
  `channel_decomposition` (all sit on `Δ_neo`). The solver enablers (items 1/2/3) are
  landed and green — that half of the milestone is complete and reusable. Do not run
  the full `Δ_neo(w)` sweep until Q7 is resolved (it would produce artifact numbers).

## 2026-07-15 (cont. 4) — Matrix-free probe: solver works past the dense wall, but Δ_neo is OUTER-region-dominated (not the island) — moment-extraction issue

- **Moved (the solver half is GREEN)**: the matrix-free path — `PlaneJacobi` +
  `newton_krylov` (cold from zeros, preconditioner built once at `u=0`) — **converges
  cleanly on resolved grids well beyond the dense `N≲1e4` cap** at physical `ρ̂_θi=0.05`,
  `w=1`: `N=10854` (7–9 Newton, ~3.7k GMRES, rmax 7e-8), `N=15678`, `N=20502` all
  `conv=1`. Items 1/2/3 are validated end-to-end on real physics grids. So reaching
  resolution past the dense wall is a **solved problem**.
- **Blocked — Δ_neo does NOT resolution-converge, and the diagnosis is a moment/far-
  field issue, not the solver**: sweeping island resolution `K=8→12→16` (N up to 20.5k,
  all converged) gives `Δ_neo = 1.40 → 1.74 → 2.20` — a **monotone ~20–25%-per-step
  power-law growth (`≈K^{0.5–0.8}`), not convergence**. The per-x integrand
  `m1(x)=∮dξ J̄_∥cosξ` shows **why**: the moment is **not localized to the island** —
  only **7–14%** of `Δ_neo` comes from `|x|<w`, and that fraction **decreases** with
  resolution (0.14→0.07). `m1(x)` does not decay outward; at `K=12` the largest `|m1|`
  sits at `x≈5.07` (near the domain edge `Lx=6`), and the outer region — carrying large
  quadrature weights on the clustered grid — dominates `∫dx J̄_∥cosξ`. So `Δ_neo`'s
  resolution-divergence is a **far-field / outer-region effect** (the response isn't
  localizing to the island; the naive full-domain volume integral is dominated by the
  poorly-resolved, non-decaying outer region), **NOT** a rational-surface singularity and
  **NOT** a solver failure.
- **Next / decision (stay stopped before items 4/5; surfaced to the user)**: the raw
  `Δ_neo = C∫dx∮dξ J̄_∥cosξ` volume moment is not extracting a localized (island) quantity
  at `Lx/w=6`, so a `Δ_neo(w)` trend from it is not yet meaningful, and the
  `channel_decomposition` rework (item 4) can't sit on top of it. Candidate root causes to
  discriminate BEFORE any fix (do not guess): (a) domain too small — the response simply
  hasn't decayed by `Lx/w=6`; test `Lx/w = 12,20` and see if `m1` decays and `Δ_neo`
  stabilizes (a resolution knob, cheap-ish); (b) the far-field/gradient-drive setup
  (`g_far = x L̂_{n0}^{-1}[1+(E-3/2)η_i]`, linear-in-x to the boundary) produces a
  non-decaying outer response that a volume moment mis-captures; (c) the Δ extraction
  itself should be a matched-asymptotic jump (`Δ'`-style, outer-solution log-derivative
  across the layer), not a naive `∫dx` — a physics/normalization question on docs/01 §4.
  (b)/(c) are physics-adjacent → physics-verifier + likely a QUESTIONS.md escalation; (a)
  is a cheap numerical check to run first. Not written to QUESTIONS.md yet — the root cause
  isn't pinned (writing one now would presume the cause).

## 2026-07-15 (cont. 3) — Solver milestone: continuation + resolution protocol land; quick-check shows dense is under-resolved (checkpoint gate)

- **Moved**: landed the two remaining solver enablers and ran the milestone
  quick-check. (1) `Solvers.natural_continuation` (commit `b6dfdb83`) — natural-
  parameter w-continuation (warm-start chaining + adaptive halve-on-fail/grow-on-
  success), stack-agnostic, tested on a manufactured stiff drifting-root problem
  where cold-start AND a single warm jump both fail but stepping succeeds; robust to
  a solve that throws (singular linearization → treated as non-convergence → halve).
  (2) `PhaseSpace` island-resolution protocol (commit `f7829186`) —
  `island_clustering_x` (invert the exact sinh central spacing for the β giving
  `Δx(0) ≤ w/K`), `resolved_island_grid`, `is_island_resolved`, `central_x_spacing`.
  Both pure numerics, tested, doc-first (numerics.md §6/§1).
- **Quick-check (the milestone decision gate) — the resolved `Δ_neo(w)` checkpoint
  is NOT reachable with dense `newton_direct`**: swept `Δ_neo(w)` on per-w resolved
  grids. (a) First pass (K mismatched, only the finer grid converged) trended
  `Δ_neo ∝ w` (log-log slope → +0.9), OPPOSITE the expected `∝1/w`. (b) Clean
  fixed-`K=8` convergence study (nx 15→33, up to the dense ceiling `N≈11k`, `nE=3`,
  all converged) shows `Δ_neo` is **not resolution-converged** — it swings **24–41%**
  between the two finest affordable grids and **flips sign** (positive at coarse →
  negative at fine, stabilizing negative for all w). So the coarse grids are
  qualitatively wrong (the diagnosed single-node-moment artifact) and the finest
  dense grid still hasn't settled → the `∝w` read from (a) is **untrustworthy**.
  **Conclusion: a resolution-tool limit, not a physics bug** — the Δ moment peaks at
  the separatrix and needs `Δx ≪ w` over an `Lx ≳ 5w` domain, which exceeds the dense
  `N ≲ 1e4` cap (exactly the x-resolution wall the 2026-07-15 diagnosis predicted).
  No physics coefficient/sign/threshold is implicated, so nothing new to QUESTIONS.md.
- **Blocked / next (stop-and-reassess gate honored — NOT proceeding to items 4/5)**:
  the trustworthy `Δ_neo(w)` checkpoint requires the **matrix-free** resolved sweep —
  `PlaneJacobi` + `newton_krylov` (+ `natural_continuation`) on grids **beyond** the
  dense cap, with convergence in BOTH `K` (island nodes across the half-width) and
  `nx`. Items 1 (PlaneJacobi) + 2 (continuation) + 3 (resolution protocol) are the
  enablers, all now in place. This matrix-free convergence campaign is the honest
  precondition before item 4 (channel_decomposition rework) and item 5 (B2/B5b) — do
  not read a `Δ_neo(w)` trend or touch the decomposition until a resolution-converged
  curve exists. The `w_c`/Δ′ convention and frames-ω gates remain the standing
  human-sign-off items (unchanged).

## 2026-07-15 (cont. 2) — Solver milestone: PlaneJacobi lands (y-colored JVP) — the physical solve is preconditioned

- **Moved**: implemented `Solvers.PlaneJacobi`, the `(x,ξ)` advection-plane
  preconditioner — the scalable matrix-free fix for the physical-`ρ̂_θi` conditioning.
  The third dead-end from the prior entry is resolved exactly as planned: build the
  plane blocks from the stack **minus the nonlocal momentum-restoring** term (kept
  everything else — pitch/cross supply the within-plane collision diagonal that
  regularizes the otherwise-singular pure `(x,ξ)` advection block), extracted by
  **y-colored JVP**. Scratch-validated the whole chain BEFORE writing production
  code: (a) the reduced stack couples planes *only* via pitch/cross, which are
  y-banded (`K=GᵀDG` half-bandwidth `≤ order`; I re-derived this rather than trust
  `2·order`) and act within fixed `(E,σ)`, so same-color planes (spaced > band)
  never cross-couple — the colored extraction is **bit-exact** vs the reduced-stack
  dense diagonal blocks (0.0 error, even with colors repeating); (b) the
  reduced-stack blocks precondition the FULL J essentially identically to the
  exact-full blocks (momentum's diagonal contribution is negligible); (c) on a
  physical grid `cond(J)=3.0e8 → cond(M⁻¹J)=8.4e4` (order-4 realistic grid:
  `1.08e9 → 1.4e5`, the LOG's "4.5e5" ballpark, grid-dependent); (d) preconditioned
  `newton_krylov` **converges** at `ρ̂_θi=0.05` (5 Newton, 343 GMRES) where
  unpreconditioned **stalls** (converged=false, 7381 GMRES), matching `newton_direct`.
- **Landed**: `Solvers.PlaneJacobi` (struct + block-callback constructor + one-shot
  `(stack, grid, u; bc, …)` convenience + `LinearAlgebra.ldiv!`, mirroring
  `YBlockJacobi`; SVD/TSVD-regularized plane blocks, `phi_scale=−α` on Φ rows),
  `Solvers.plane_blocks` (the y-colored JVP extractor, stack-agnostic), and
  `Solvers.without_momentum_restoring`. Pure numerics — no physics content. Unit
  test mirrors the `newton_direct` test (coloring exactness + cond reduction ≥3
  orders + preconditioned-converges-where-unpreconditioned-stalls + matches direct
  + wrong-block-size throws). 154 solve assertions green; grids/operators/configure
  green; `build_docs_local.jl` green. Doc-first: numerics.md §5.
- **Blocked / next**: nothing on the preconditioner. **Next** (milestone order):
  the resolution-adequacy protocol (`Δx ≪ w` island-resolved grid helper) → the
  resolved `Δ_neo(w)` checkpoint (expect `∝1/w` at large w; reassess if not) →
  rework `channel_decomposition` (restrict `⟨·⟩_Ω` to the island region, ξ-mean
  outside; physics-verifier before commit) → B2/B5b + F5/F7 + `_LADDER`/STATE +
  regression. A resolution note surfaced along the way: coarse grids (nx≲7, or the
  mid nE=2 grids) don't converge even with PJ or `newton_direct` — a genuine
  under-resolution effect (the separatrix response), not a preconditioner one; this
  is exactly what the resolution-adequacy protocol must pin. The `w_c` threshold
  (Δ′ convention) and frames ω-conventions remain the human-sign-off gates (unchanged).

## 2026-07-15 (cont.) — Solver milestone: continuation works; PlaneJacobi is subtler than planned

- **Moved**: (a) **w-continuation confirmed working** (scratch): adaptive natural-parameter
  continuation (step-halving + warm-start) with `newton_direct` reaches larger `w` smoothly
  in 3–4 Newton iters — the globalization enabler works. (b) The **resolution wall** is
  confirmed decisive: at a coarse grid `Δ_neo(w=0.5) = −3.6` vs the resolved `+6.6` (sign
  flip) — the Level-0 response peaks at the **separatrix (x~w)**, away from the x=0 cluster,
  so a clean checkpoint needs adequate x-resolution → needs the matrix-free preconditioner.
- **Blocked — PlaneJacobi is a real sub-problem** (attempt reverted, not committed): the
  `(x,ξ)`-plane preconditioner is right *in principle* (exact plane block → cond 1.24e9→4.5e5),
  but building it matrix-free is subtle. Two dead ends found: (1) **colored-JVP with all planes
  seeded at once is contaminated** — the nonlocal **momentum-restoring** term couples *all*
  planes densely, corrupting every block (cond→3e21). (2) Building from a **pure-advection
  sub-stack is singular** — first-order `(x,ξ)` advection has null modes (constant-in-ξ:
  `∂ξ const=0`); the *exact* plane block is regularized by the **collision diagonal** (`c·K[iy,iy]`
  of pitch diffusion), so dropping the collisions removes exactly the regularization (cond still
  2e21). **Correct approach (next)**: build the plane block from the stack **minus
  momentum-restoring** (keep advection + neo + pitch/cross for their within-plane diagonal),
  extracted by **y-colored JVP** — the pitch/cross operators are y-*banded* (`K` from a banded
  `D1`), so y-planes spaced `> 2·order` apart don't cross-couple and can be seeded together
  (`≈ np·(2·order+1)` JVPs per build, ≪ N). Or analytic plane-block assembly from the
  coefficient arrays.
- **Next**: implement PlaneJacobi via y-colored JVP (exclude momentum), reproduce cond→4.5e5 +
  preconditioned `newton_krylov` convergence at physical `ρ̂_θi`; then the resolved `Δ_neo(w)`
  checkpoint. Fallback if PlaneJacobi stays hard: `newton_direct` + continuation at the
  best-affordable dense grid (N≲1e4) for a first resolution-caveated B2 result.

## 2026-07-15 — Solver-robustness milestone: the first converged PHYSICAL solves + a 3-layer diagnosis

- **Moved**: attempting to run the B-ladder (B2/B5b) surfaced that **the Level-0
  physics solve had never actually converged at physical parameters** — every prior
  "converged" solve (incl. the configure test) uses an artificial order-unity
  `ρ̂_θi=1.0`. A deep diagnosis (all in `/tmp` scratch, evidence-based) peeled **three
  layered, tractable issues — none a physics bug**:
  1. **Solver conditioning**: matrix-free `newton_krylov` stalls (residual plateaus
     ~1e-4, GMRES burns 30k iters) because `cond(J) ~ 1e9` (grows as `1/ρ̂_θi` from the
     `1/ρ̂_θi` streaming + collision coefficients). A **dense-direct Newton converges in
     6–9 iters at physical `ρ̂_θi=0.05`** regardless of conditioning. Naive block-Jacobi
     *worsens* it (y: 1.24e9→1.55e12); the **(x,ξ) advection-plane block** drops it
     **1.24e9→4.5e5** — so `YBlockJacobi` (y-only) is the wrong tool; the scalable
     matrix-free fix is an (x,ξ)-plane preconditioner (future).
  2. **Island x-resolution**: the per-x profile of `∮J̄_∥cosξ` is entirely localized on
     the **single x=0 node** (neighbors 2.7 away) — the island (half-width `w`) was
     **grossly under-resolved**, so the Δ moment ≈ one node × a huge quad weight, wildly
     `Lx`/clustering-sensitive (the "domain-dependence"/sign-flips). With strong
     x-clustering (Δx≪w), **`Δ_neo` stabilizes** (6.6 vs 5.8 across Lx=5→8; was
     sign-flipping).
  3. **Diagnostic fragility**: with `Δ_neo` stable, **`Δ_pol` still swings −2.4→−30.9** —
     the new `channel_decomposition` (⟨J̄_∥⟩_Ω bootstrap/pol split) is not robust (its
     interpolant/Ω-quadrature over the full domain is ill-behaved far from the island).
- **Landed**: `Solvers.newton_direct` (dense-Jacobian exact-solve Newton + Armijo LS,
  reusing `dense_jacobian`) — the robust benchmark solve for N≲1e4; 143 islands-solve
  assertions green (incl. a new test: matches `newton_krylov` on a well-conditioned
  case, converges on a stiff advective stack). No physics content (pure numerics).
- **Blocked / next**: (1) establish **resolved-island benchmark grids** (strong
  x-clustering) + confirm resolved `Δ_neo(w) ∝ 1/w` at large w (the primary B2 target via
  the *robust* direct moment; a resolved large-w sweep is running); (2) **rework
  `channel_decomposition`** for a robust bootstrap/polarization split (needed for the
  `Δ_pol ∝ 1/w³` B2 sub-target); (3) then the full B2 + B5b. The `w_c` threshold (Δ′
  convention) and frames ω-conventions remain the human-sign-off gates (unchanged).

## 2026-07-14 — FIRST PHYSICS: solved state → the resonant-current Δ outputs (output assembly + decomposition)

- **Moved**: wired the physical solve into the **Δ outputs** — the first physics
  deliverable off the converged state. (1) `Moments.parallel_current!` gained
  `wy`/`wE` kwargs forwarded to `weighted_moment!`, so `J̄_∥` now carries the
  **physical `∫d³v` measure** (was the flat default) — the cleared `W = v̂_∥`'s
  `√(1−y b_min)` cancels the pitch Jacobian, `J̄_∥` regular. (2) New output-assembly
  entry point `Configure.delta_outputs(grid, phys, species, Usol, cfg)`: builds the
  cleared `W` (`parallel_flow_weight`) + physical measure, forms `J̄_∥` from the
  solved bulk-ion `g`, projects to `Δ_neo ≡ Δ_cos` and `Δ_sin` with the cleared
  `∓μ₀R/2ψ̃` prefactors (`cfg.delta_prefactors`), so `Δ_cos+iΔ_sin ↔` layer-`Δ(Q)`.
  (3) Decomposition diagnostics (`Moments.channel_decomposition`, docs/01 §4, L23
  Eq. 2.5.3 *approximate* split): lifts `J̄_∥` to a callable (`grid_interpolant`,
  separable local-Lagrange, reusing PhaseSpace `fd_weights`), reconstructs the
  flux-surface-constant `⟨J̄_∥⟩_Ω` **bootstrap+curvature** channel and the
  **`Δ_pol` polarization** residual, plus the `⟨J̄_∥⟩_Ω` profile. **No new [VERIFY]
  coefficient** — assembly + diagnostics only. **physics-verifier PASS** (every
  physics number from a cleared builder / `cfg.delta_prefactors`; Δ_cos≡Δ_neo sign
  + prefactor mapping, the open/closed `⟨·⟩_Ω` branch, and the physical-measure
  forwarding all match docs/01 §4). 138 solve + 1540 configure + 5 anchor-sync
  assertions green (nodal-exact interpolant, off-node accuracy, flux-function
  `Δ_pol→0`, `⟨J̄_∥⟩_Ω` profile recovers `f(Ω)`, additive split, single-bulk
  contract); `build_docs_local.jl` green. Doc-first: numerics.md §7 (Δ moments,
  physical measure, decomposition, entry point) + §8 (F stale-line fix).
- **Blocked**: nothing on Step 1. The **absolute** threshold `w_c` (B-ladder T4)
  needs an external outer-region `Δ′` input + the docs/09 manifest — ESCALATE, do
  not guess, if it comes up. Deferred: Hirshman–Sigmar `k ≃ −1.173`; the Verify.jl
  MMS repointing (collision operators); the electron/species partition of `J̄_∥`
  (docs/01 §4, a later diagnostic — L0 `delta_outputs` is single-bulk-ion).
- **Next**: Step 2 — the B-ladder T2/T3 physics gates (un-skip the B-benchmarks
  via `const UNGATED`). Primary: **B5** the `:original/:improved` drift-toggle
  differential (the reproducible 8.73→1.46 ρ_bi form); then **B2** large-w
  `Δ_bs+Δ_cur ∝ 1/w` (T3), **B4** `Δ_pol(ω_E)` trend (T3).

## 2026-07-14 — Q5: clear the momentum-restoring term F — the Level-0 collision operator is COMPLETE (6/6)

- **Moved**: implemented the sixth and final collision term — the **momentum-restoring**
  field-particle integral (F), the one **nonlocal** Level-0 term (`orbit-averaged-collision.md`
  §6, now unblocked by the Q6 physical measure). `Operators.MomentumRestoring` +
  `Configure.momentum_restoring_term`: forms the parallel-flow moment
  `Ū(x,ξ) = (1/√π⟨ν̂_ii⟩_u){ν̂_ii v̂_∥ g}_v` (physical `∫d³v` measure, moment weight
  `W = ν̂_ii·v̂_∥` σ-odd) into a new `cache.Ubar`, then adds the σ-even redistribution
  `+2ν̂_ii(1+ε)Ū/(m ρ̂_θi)` (positive — ÷−m ρ̂_θi of the RHS; `F̂_M=e^{−E}` cancels in
  the `g=shape` convention). **Linear** in g (a moment), **allocation-free** (via
  `cache.Ubar`), AD-transparent. Uses only cleared inputs (`⟨ν̂_ii⟩_u`, `v̂_∥`, `ν̂_ii`).
  **physics-verifier PASS** (traced Ū vs L23 Eq. 8.3.17 line-by-line, the positive
  sign, linearity, no guessed number); 1690 islands assertions green (incl. W σ-odd,
  redistribute positive, `F(2g)=2F(g)`, nu_star=0 guard); `build_docs_local.jl` green.
  Doc-first: docs/01 §2.3, QUESTIONS Q5 (F resolved), numerics.md §2/§8,
  orbit-averaged-collision.md status (6/6), anchor-sync marker.
- **Blocked**: nothing on the collision operator — **all six terms complete**, no
  gated kinetic physics remains. One flagged structural-completeness item (verifier,
  non-blocking): the `Ū(pF̂′)` drive piece of A/F folds into the far-field/drive
  (I19 Formulation A, `orbit-averaged-collision.md` §7) — the operators act on `g`
  which satisfies the neoclassical far-field BC. Deferred: Hirshman–Sigmar
  `k ≃ −1.173`; the Verify.jl MMS repointing (collision operators).
- **Next**: wire `Moments.parallel_current!`/`J̄_∥` → the Δ outputs with the cleared
  `W` + physical measure (the primary physics deliverable); the B-ladder physics
  gates are now un-gated (the full L0 operator + closure are physical).

## 2026-07-13 — Q6/Q3: clear the physical ∫d³v moment measure + parallel-flow weight W (unblocks term F)

- **Moved**: attempting to clear Q3's parallel-flow weight `W` surfaced that the
  code's `velocity_moment!`/`weighted_moment!` used a **flat** measure missing the
  physical `∫d³v` Jacobians (`√E/2` speed, `1/√(1−yb)` pitch) — and that the
  **already-cleared QN density `δn̄_i`** used that flat moment. Escalated as **Q6**;
  the user chose the **physical `∫d³v`** with **flux-surface `b`**. Derived and
  cleared (`velocity-moment-measure.md`, **human sign-off**): the `√E/2` speed
  Jacobian (folded into Gauss–Laguerre), the `1/√(1−y b_min)` pitch Jacobian
  (`b_min=(1−ε)/(1+ε)`, an **exact singular-weight quadrature** = the `IinvB` edge,
  forbidden region zeroed), via `Configure.physical_velocity_weights`; and the
  parallel-flow weight **`W = v̂_∥ = σ√E√(1−y b_min)`** (`Configure.parallel_flow_weight`,
  clearing Q3's `W`; its `√(1−yb)` cancels the pitch Jacobian so `J̄_∥` is regular).
  `velocity_moment!`/`weighted_moment!` gained `wy`/`wE` kwargs (default flat, so M1/M2
  manufactured tests are untouched); `Operators.Quasineutrality` gained the physical
  weights and its `δn̄_i` is now physical (**`max|Φ|` shifted 5.7→4.5**, the approved
  QN revision; closure algebra `α`/`S` unchanged). **physics-verifier PASS**; 1596
  islands assertions green (incl. `Σwy=2/b_min` exact, forbidden zeroed, `W` σ-odd,
  QN wired); `build_docs_local.jl` green. Doc-first: docs/01 §3/§4, QUESTIONS Q6
  (resolved)/Q3 (`W`)/Q5 (F unblocked), Moments docstrings, derivations index/nav.
- **Blocked**: nothing new. **Term F (momentum restoring) is now UNBLOCKED** — `Ū`
  is a bounded physical moment (`W` + the cleared `u³ν̂_ii/⟨ν̂_ii⟩_u` weight), ready
  to implement as the nonlocal operator. Deferred: Hirshman–Sigmar `k ≃ −1.173`;
  the Verify.jl MMS repointing (collision operators).
- **Next**: implement term F (completes the collision operator; now unblocked), and
  wire `parallel_current!`/`J̄_∥` → the Δ outputs with the cleared `W` + physical
  measure.

## 2026-07-12 — Q5: clear the full orbit-averaged collision operator (5/6 terms; last operator gate)

- **Moved**: cleared the **full orbit-averaged collision operator**
  (`orbit-averaged-collision.md`, **human sign-off**) — the last Level-0 operator
  gate. Reading L23 Eq. 2.3.47 / appendix 8.3.2 first-hand showed "B_profile" is
  the tip of a **six-term** operator. Implemented the five **differential** terms
  in the code normalization ÷(−m ρ̂_θi): the σ-**odd** mimetic **pitch diffusion**
  D+E — an exact `y`-divergence `∂_y(P_oa ∂_y)`, `P_oa = y⟨√(1−yb)⟩_θ`
  (orbit-averaged, replacing the local single-`B` placeholder), **flat measure**
  (divergence identity `d/dy[yS] = ⟨(2−3yb)/2√(1−yb)⟩` verified to 1e-11); the
  σ-even `∂_x` **drag** (A); the σ-odd `∂²_x` **neoclassical** diffusion (B, using
  `⟨1/√(1−yb)⟩_θ`); the σ-even `∂²_{xy}` **cross** (C). New:
  `Coefficients.orbit_average_pitch_brackets`, `Configure.{pitch_diffusivity_profile,
  pitch_collision_coefficient, collisional_drag_coefficient,
  neoclassical_diffusion_coefficient, collisional_cross_coefficient}`,
  `Operators.{CollisionalDrag, NeoclassicalDiffusion, CollisionalCross}` + kernels;
  a **σ-parity correction** (pitch diffusion is σ-odd via the `1/v̂_∥` weight, not
  σ-even); a forbidden-pitch **`g=0` domain BC** (`FarFieldConditions.forbidden_y`,
  since physically-zeroed collision coefficients left forbidden nodes
  unconstrained); and a new `Level0Physics.m` (collision terms carry `1/(m ρ̂_θi)`,
  which the drift's `m` cancellation does not). **`GatedLevel0Inputs` /
  `level0_placeholders` removed — no gated kinetic inputs remain** (Q5 fully
  cleared). **physics-verifier caught a real sign bug**: the derivation §3 asserted
  the collision code coefficients negative, but ÷(−m ρ̂_θi) of L23's leading `−`
  gives **positive** (matching the streaming anchor `a_ξ=+(x/L̂q)Θ/ρ̂θ`) — all five
  flipped `−→+`, §3 now shows the flip step, **physics-verifier PASS** on re-check.
  1595 islands assertions green (incl. node-for-node coefficient/σ-parity checks +
  an A4 conservation/entropy gate on the shipped `P_oa`); `build_docs_local.jl`
  green. Doc-first: docs/01 §2.3, QUESTIONS Q5, numerics.md §2/§8, derivations
  index/nav; B5 benchmark de-gated.
- **Blocked**: two collision follow-ups (not gates on the differential solve):
  (1) the **momentum-restoring** term F (nonlocal velocity integral
  `2ν̂_ii(1+ε)Ū_∥ᵢ`, its magnitude `⟨ν̂_ii⟩_u` already cleared) — a new operator
  type; (2) **repoint `Verify.jl` (build_stack/MMS) to the new operators** so the
  assembled-MMS/A4 harness covers `PitchAngleDiffusion(K,c_pitch)` +
  Drag/Neoclassical/Cross (physics-verifier follow-up; the shipped `P_oa` is A4-gated
  in the configure test, and the kernels reuse MMS-covered stencils). Plus the
  deferred Hirshman–Sigmar `k ≃ −1.173`.
- **Next**: the momentum-restoring term F (completes the collision operator), then
  the Verify.jl MMS repointing. With F, the Level-0 collision physics is complete.

## 2026-07-12 — Q5: clear the collision magnitude (ν_★ + momentum-restoring ⟨ν̂_ii⟩_u)

- **Moved**: closed the collision-magnitude Q5 item (`collision-magnitude.md`,
  **human sign-off**). Read L23 Eqs. 4.1.4–4.1.6 (p. 87–88) first-hand and
  **derived** the momentum-restoring speed average `⟨ν̂_ii⟩_u =
  (4ε^{3/2}ν_★/3√π)(√2−ln(1+√2))` from L23's reduced integrand: the reduction
  `u⁴ν̃_jj = u·erf − erf/2u + e^{−u²}/√π` plus three standard integrals
  (`∫u e^{−u²}erf=1/2√2`, `∫e^{−2u²}=½√(π/2)`, `∫u⁻¹e^{−u²}erf=ln(1+√2)=arcsinh(1)`).
  **Verified independently** (QuadGK): all three integrals + the `u⁴ν̃` reduction
  exact to 15 digits, `⟨1⟩_u=1` (normalized), and the full value reproduces L23's
  own unit-test `1.267537×10⁻⁴` (ε=0.1, ν_★=0.01) to all 7 digits. Cleared as
  `Coefficients.momentum_restoring_average`. Also wired the collision **magnitude**
  `nu_tilde = ε^{3/2}ν_★` from a new `Level0Physics.nu_star` scenario field (the
  §4 ν_★ normalization, already signed off) — `:nu_tilde` moved gated→cleared,
  `nu_tilde` dropped from `GatedLevel0Inputs` (now `B_profile`-only). This
  un-gates the collision operator's magnitude. **physics-verifier PASS**; 1577
  islands assertions green (incl. the L23 1.2675e-4 reproduction + ε^{3/2}/ν_★
  scalings); `build_docs_local.jl` green. Doc-first: docs/01 §2.3, QUESTIONS Q5,
  numerics.md §2/§8, collision-operator.md §7 (deferred item now resolved),
  derivations index/nav.
- **Blocked**: **one** kinetic family remains gated (Q5): the orbit-averaged
  pitch measure `B_profile` (the collision operator's `|B|` on the y-grid is the
  orbit-averaged/turning-point field, not a single local B; ties to the A4
  conservation gate). Plus the deferred Hirshman–Sigmar `k ≃ −1.173` (its own
  parallel-viscosity moment problem, L23 Eq. 4.1.7). The momentum-restoring
  *operator term* itself is a separate future addition (its magnitude is cleared).
- **Next**: the orbit-averaged pitch measure `B_profile` — the **last** Level-0
  operator coefficient gate. Clearing it makes the whole L0 operator stack
  physical (only `k` and the momentum-restoring term remain, both beyond the
  minimal L0 solve). Same rhythm: derive → present → sign off → clear.

## 2026-07-12 — Q5: clear the E×B coupling c_E (passing σ-odd, trapped ≡ 0)

- **Moved**: derived and cleared the `E×B` coupling `c_E` (`exb-coupling.md`,
  **human sign-off**). Matching the two master-eq E×B braces to the
  `Operators.ExBDrift` Poisson bracket in the `c_D=ω̂_D` normalization (÷ −m ρ̂_θi,
  `ρ̂_θi` cancels — **no new physics parameter**) gives `c_E = ½⟨1/v̂_∥⟩_θ`,
  `1/v̂_∥ = σ/(v̂√(1−yb))`. **σ-parity nailed (the crux)**: passing (`y<y_c`) is
  **σ-odd** `c_E = (σ/2√E)B₁(y)` with a new orbit bracket
  `B₁(y)=⟨1/√(1−yb)⟩_θ`; trapped (`y>y_c`) is **identically 0** — the σ-odd
  `1/v̂_∥` cancels between the two banana legs under `Σ_σ`. The decisive check: the
  published drift-island label requires trapped `S ∝ p̂` (docs/01 §2.2), which the
  E×B piece would break unless it vanishes — so trapped=0 is *required*, not
  chosen. E×B is thus a **passing-particle** effect, like island-streaming.
  Implemented: `Coefficients.orbit_average_exb_bracket` (B₁, passing-only, same
  y_c-layer handling as the drift `G`), `Configure.exb_coupling_table`, and
  `Operators.ExBDrift` generalized to a **velocity-dependent array** coefficient
  (scalar path kept for tests). `:exb` moved gated→cleared; `c_E` dropped from
  `GatedLevel0Inputs`. **physics-verifier PASS**; 1566 islands assertions green
  (incl. passing σ-odd node-for-node, trapped≡0, σ-flip); `build_docs_local.jl`
  green. The M2c structural-solve assertion switched to the grid-independent
  **max-norm** (04 §5 criterion) since the now-active E×B spreads a ~1e-8 residual
  across ~N unknowns, inflating the √N L2 norm past 1e-7 (per-equation residual
  4.5e-8; not a physics change). Doc-first: docs/01 §2 (+ fixed the §2 line-107
  orbit-average convention to match both cleared code paths), QUESTIONS Q5,
  numerics.md §2/§8, derivations index/nav.
- **Blocked**: only **two** kinetic families remain gated (Q5): the collision
  magnitude `⟨ν̂_ii⟩_u` (needs L23 Eq. 4.1.6 integrand) and the orbit-averaged
  pitch measure `B_profile`. Plus the deferred `k ≃ −1.173`.
- **Next**: the collision magnitude `⟨ν̂_ii⟩_u` or the orbit-averaged pitch
  measure — the last two kinetic clearances for a fully physical L0 solve. Same
  rhythm: derive → present → sign off → clear.

## 2026-07-12 — Q5: clear the gradient drive (far-field BC, no frame convention)

- **Moved**: completed the gradient drive by re-reading I19 Eq. 29 first-hand.
  **Correction**: the earlier draft misread the ratio as `ω_si^T/ω_ci` (⇒ frame
  convention); it is `ω_si^T/ω_si = 1+(v̂²−3/2)η_i` — a **temperature factor, not
  a frequency ratio**, so **no frame convention is needed**. The drive is the
  standard neoclassical `p_φ F'_Mi`, imposed as the **far-field BC** (master eq
  homogeneous, I19 Formulation A). Cleared: `Operators.GradientDrive = 0` and
  `Configure.gradient_far_field` builds `g_far = x L̂_{n0}⁻¹[1+(E−3/2)η_i]`
  (`Φ̂_far = 0`), with a new `Level0Physics.eta_i`. Both `gradient_drive` and
  `far_field` moved gated→cleared; `drive`/`bc` dropped from `GatedLevel0Inputs`.
  1511 islands assertions green (the assembly now solves with the *physical* far
  field). `gradient-drive.md` signed off; docs/01 §2, QUESTIONS Q5, numerics.md,
  index/nav updated.
- **Blocked**: only three kinetic families remain gated (Q5): the `E×B` coupling
  `c_E`, the collision magnitude `⟨ν̂_ii⟩_u`, and the orbit-averaged pitch
  measure. Plus the deferred `k ≃ −1.173`.
- **Next**: E×B coupling (Poisson-bracket normalization) or the collision
  magnitude `⟨ν̂_ii⟩_u` (needs L23 Eq. 4.1.6). With these three, the L0 solve is
  fully physical.

## 2026-07-11 — Q3/Q5: clear the passing fraction f_p

- **Moved**: signed off `derivations/passing-fraction.md` and cleared
  `Coefficients.passing_fraction(ε) = 1 − 1.4624√ε` (the effective
  trapped-fraction coefficient, derived + numerically confirmed, = the sources'
  quoted 1.46 to 3 s.f.). Authorizes `Fields.ElectronClosure.f_p`. 1499 islands
  assertions green (limits + monotonicity + the 1.46 match); docs green.
  docs/01 §2.4, QUESTIONS Q3/Q5, derivations index/nav updated.
- **Blocked**: the companion Hirshman–Sigmar `k ≃ −1.173` stays deferred (needs
  the parallel-viscosity moment problem). Gradient-drive amplitude + frame
  convention still pending (bundled sign-off, `gradient-drive.md`).
- **Next**: the gradient-drive amplitude/frame convention is the last structural
  blocker; then E×B, collision magnitude, pitch measure. `k` its own derivation.

## 2026-07-11 — Q5: gradient-drive structural finding (drive = far-field BC)

- **Moved**: read I19 first-hand (Eqs. 8, 23–32) to derive the gradient drive.
  **Finding** (`derivations/gradient-drive.md`, draft): the master equation
  (I19 Eq. 32) is **homogeneous** — no interior source; the drive is the
  **far-field boundary condition** `Ḡ₀ → p_φ(ω_si^T/ω_ci)(n'/n)F_Mi` (Eq. 29).
  So `Operators.GradientDrive = 0` at Level 0, and the Q5 `gradient_drive` and
  `far_field` items **merge** into one object — the diamagnetic far field
  `g_drive = D_dia·x·[1+(v̂²−3/2)η_i]·F_Mi`. The `x`-linearity, temperature
  correction, and Maxwellian are cleared structure; the **normalized amplitude
  `D_dia` bundles the frame convention** `Frames.C_dia` (NaN-gated) — clearing
  the drive = clearing `C_dia` (ion `ω_dia` normalization). Nothing entered
  `src/` (draft, docs-only). QUESTIONS Q5 + derivations index/nav updated.
- **Blocked**: `D_dia` + the frame convention `C_dia`/`sign_omega0`/
  `C_gradient_shift` (Q3) — a bundled human sign-off; the normalized ion
  diamagnetic amplitude needs the careful `ω_si/ω_ci` normalization algebra.
- **Next**: complete `D_dia` (ion `ω_dia` in the code normalization) + sign off
  the frame convention, then wire `g_far` and set `GradientDrive = 0`. This is
  the last structural blocker for a real `g` to develop.

## 2026-07-11 — Q5: clear the parallel (island) streaming coefficients

- **Moved**: re-derived the island-streaming channel from the master DKE
  (I19 Eq. 32) — `derivations/parallel-streaming.md`, **human sign-off**. Key
  result: the two coefficients factor **exactly** into `{Ω, g}` flux-surface
  advection, `a_ξ = (L̂_q⁻¹/ρ̂_θi)x Θ`, `a_x = −(L̂_q⁻¹ŵ²/4ρ̂_θi)sinξ Θ` (passing-
  only via `Θ(y_c−y)`) — a coefficient-free structural check that leaves no
  freedom. Normalization chosen (÷ −m ρ̂_θi) to keep the cleared `c_D = ω̂_D`
  untouched. Implemented as `Configure.streaming_coefficients` with a new
  `Level0Physics.rho_hat_theta_i`; `:streaming` moved from gated to cleared;
  `a_xi`/`a_x` removed from `GatedLevel0Inputs`. **physics-verifier PASS**; 1494
  islands assertions green (incl. a per-node `{Ω,g}` structure test);
  `build_docs_local.jl` green. Doc-first: docs/01 §2, QUESTIONS Q5,
  numerics.md §2/§8 amended.
- **Blocked**: nothing new. Cleared now: streaming, drift, collision *shapes*,
  quasineutrality field, Δ prefactors. Still gated (Q5): `E×B` coupling, gradient
  drive, collision magnitude `⟨ν̂_ii⟩_u`, orbit-averaged pitch measure, far field.
- **Next**: the gradient drive + far field are the remaining structural blockers
  for a real `g` to develop (the drive is the source; the far field is the BC).
  `f_p` sign-off still pending. Same rhythm: derive → present → sign off → clear.

## 2026-07-11 — Q5 field fix: wire the cleared quasineutrality closure (Φ now driven)

- **Moved**: closed the M2c-surfaced QN **structural gap** (QUESTIONS Q5). The
  quasineutrality closure was signed off in M2b but the operator carried only
  `R_Φ = M[g] − αΦ` (no drive), so the Level-0 potential collapsed to zero.
  Implemented the full cleared closure: `Operators.Quasineutrality` gained an
  optional `source` field; `Configure.configure_level0` now builds
  **`α = (τ+1)/τ`** (= `1/quasineutrality_coefficient(τ)` — the reciprocal, =2 at
  τ=1) and the drive **`S = L̂_{n0}⁻¹(x − ĥ(Ω))`** (`Configure.quasineutrality_source`,
  from the cleared `h_amplitude`/`h_profile`, one width `w=w_psi` for both `Ω`
  and the `ĥ` prefactor per `electron-closure.md §3`). Added `inv_Ln0` to
  `Level0Physics`; removed `alpha` from the gated inputs (`quasineutrality` moved
  to `cleared`). **Verified: max|Φ| ≈ 5.7 after solve** (was ~0). 194 islands
  tests green; **physics-verifier PASS** (α reciprocal, source sign, width
  convention all checked vs the signed-off derivation); `build_docs_local.jl`
  green. Doc-first: docs/01 §3, `quasineutrality-closure.md §6`, numerics.md,
  QUESTIONS Q5 all amended.
- **Blocked**: nothing new. The *kinetic* Q5 families (streaming, E×B, gradient
  drive, `⟨ν̂_ii⟩_u`, pitch measure, far field) remain gated — the field equation
  is now the fully cleared closure, but a physics threshold still needs those.
- **Next**: (human/next lane) the remaining Q5 kinetic clearances — the
  parallel-streaming coefficients are the highest-leverage next (they + a far
  field would let a real `g` develop). `f_p` sign-off still pending
  (`passing-fraction.md`). The B-ladder scaffolding is wired to light up as each
  clears.

## 2026-07-11 — M2c: L0 configuration assembly + input-completeness audit (autonomous)

- **Moved (M2b lane complete → M2c started)**:
  - **Derivation lane 6/6 cleared** (earlier this session): ψ̃, ω̂_D + drift
    toggle, collision operator, h(Ω) closure, quasineutrality, Δ prefactors — all
    human-signed-off and in `src/` via `Coefficients.*`/`Moments.*` (recorded in
    docs/01 + `derivations/`). Re-derivation caught the I19 ψ̃ published typo, the
    collision low-v limit error, and the quasineutrality δn normalization.
  - **Input-completeness audit** (Decision D9 deliverable): new
    `docs/09-input-manifests.md` — per-source manifests (I19, D21, D23a/b, L23).
    Headline: I19's own run collisionality is contradictory (0.01 vs 10⁻³) and its
    Δ′ unspecified, so B5a's absolute threshold is only a T3 (existence) target;
    **L23 (thesis) is the only clean T4 candidate**. Itself a reproducibility
    result (Paper-I C9). Nav-wired.
  - **M2c goal prompt** authored (`design/M2c-launch-prompt.md`): L0 assembly +
    audit + docs/07 infra, autonomous-mode (un-gate nothing, escalate to
    QUESTIONS, never guess).
  - **L0 configuration assembly** (`src/Islands/configure/Configure.jl`,
    `configure_level0`): wires the **cleared** coefficients onto the operator
    stack — `c_D` node-for-node from `magnetic_drift_frequency` (verified Δ=0.0,
    with the `:improved` toggle and forbidden-region zeroing), the pitch-collision
    shapes from `pitch_diffusivity`/`deflection_frequency`, the Δ prefactors from
    `delta_moment_prefactors`. Everything uncleared is a **supplied gated input**
    (`GatedLevel0Inputs`); `level0_placeholders` gives documented non-physics
    values so the assembled stack **converges structurally** (verified: 5 Newton
    iters, ‖F‖=1.3e-9). 184 islands tests green (new `runtests_islands_configure.jl`);
    the y=0 orbit-average guard was relaxed (`y>0`→`y>=0`, a domain-boundary fix,
    no y>0 value changes). **Physics-verifier PASS** on the diff.
  - **docs/07 STATE dashboard** (M2c #3a): `Verify.write_state_dashboard` +
    `ladder_status` generate `docs/src/islands/state/STATE.md` (auto-gen header,
    do-not-hand-edit) — the docs/05 ladder as a status table (8 A-ladder rows
    green, B/C physics rows gated on QUESTIONS). Nav-wired.
  - **B-ladder scaffolding** (M2c #4): `benchmarks/islands/benchmark_B{2,4,5}*.jl`
    wired to `configure_level0` with a one-line un-skip (`const UNGATED = true`),
    kept skipped on QUESTIONS Q3/Q5. B5 carries the full T2 toggle scaffold.
  - **Anchor-sync check** (M2c #3b, docs/07 §1.1): `Verify.check_anchor_sync`
    enforces the bidirectional operator↔docs sync — every `AbstractTerm` operator
    named by an `Implemented by:` marker in `numerics.md` (forward), every marker
    symbol resolving to a real Islands binding (reverse). numerics.md §8 gained
    the as-implemented assembly section + `Implemented by:` markers. Tested with
    negative controls (a missing operator ⇒ undocumented; a bogus symbol ⇒
    dangling). 189 islands tests green; `build_docs_local.jl` green.
  - **Deferred-constant draft** (M2c #5): `derivations/passing-fraction.md`
    `[DERIVED]` derives the electron-closure passing fraction `f_p ≃ 1−1.46√ε`
    from the effective trapped-fraction integral and **numerically confirms** the
    coefficient (`1.4624`, = quoted `1.46` to 3 s.f.). **Drafted, awaiting
    sign-off** — does NOT clear `Fields.ElectronClosure.f_p` (stays NaN-gated).
    One open reviewer item (I19 Eq. 22's f_p definition). `⟨ν̂_ii⟩_u` and the
    Hirshman–Sigmar `k` left escalated (need specific source integrands) — not
    drafted speculatively.
- **Blocked (escalated → QUESTIONS Q5)**: the L0 assembly surfaced that several
  operator-stack coefficient families are **not yet cleared** — parallel
  streaming (`a_xi`/`a_x`), `E×B` `c_E`, gradient drive, the collision magnitude
  `⟨ν̂_ii⟩_u`/`ν_★`, the orbit-averaged pitch measure, and the neoclassical far
  field — plus a **structural gap**: the quasineutrality operator lacks the
  `L̂_{n0}⁻¹(x−ĥ)` field source the cleared closure requires (and its α is the
  reciprocal of `quasineutrality_coefficient`), so no Level-0 *physics* run is
  possible until that lands. This is why M2c delivers the assembly **scaffold**,
  not a physics result. These need a second derivation lane (an "M2d",
  human-present) run like M2b.
  Autonomous M2c is now complete (#1 assembly, #2 audit, #3 docs infra [STATE +
  anchor-sync], #4 B-ladder scaffolding, #6 as-implemented numerics.md; all
  green, physics-verifier PASS on the assembly).
- **Next**: (human) work **Q5** — clear the remaining coefficient families and fix
  the QN operator structure (doc-first: amend docs/01 §3 + docs/03 §2). That is
  the only thing gating a Level-0 *physics* run; it un-gates the B-ladder T2/T3
  gates (scaffolding, STATE dashboard, anchor-sync all already wired). Then #5
  (deferred sub-constants ⟨ν̂_ii⟩_u/k/f_p) is a focused sign-off session like M2b.
  When the full as-implemented Physics Book chapters (docs/07 §1.1) are scoped,
  point the operators' anchors there; `Verify.check_anchor_sync` already enforces
  the sync against `numerics.md` today. The M2c goal prompt is re-entrant.

## 2026-07-11 — Re-scope verification targets: tiered by reproducibility (Decision D9)

- **Moved**: user flagged that absolute literature numbers (w_c ≃ 2.76 ρ_θi ≡
  8.73 ρ_bi, 0.45 ρ_θi ≡ 1.46 ρ_bi, the kokuchou 0.440… fit, the −0.89 ω_dia,e
  reversal, the D23a shaping widths) were quoted as if they were pass/fail
  targets — but reproducing an absolute number needs *every* input of the
  source's exact scenario, which the lineage under-specifies (B5a's own
  collisionality is internally contradictory). Direction: qualitative/scaling
  checks (the Park 2022 / Burgess 2026 modality) are the real physics gates.
- **Decision D9** (adopted, docs/00): a **four-tier target taxonomy** written
  into docs/05 ("Target tiers and reproducibility"): T1 exact math / T2 internal
  cross-checks & toggle differentials (the sharpest quantitative claims) / T3
  scalings-trends-existence vs. literature (primary literature-facing gates) /
  T4 absolute reproduction — **audit-gated**, never pass/fail without an *input
  manifest*, downgraded to T3 where the source is under-specified. Added a fifth
  triage outcome ("under-specified source configuration") and three reporting
  rules (publish the manifest; prefer differentials/ratios; sensitivity scans).
- **Applied** across docs/05 (every B/C row retagged; A7 constants marked T1),
  docs/00 (Level-0 gate softened, D9 logged), the Paper-I OUTLINE (C5–C7
  reframed scaling-first; new C9 = the input-completeness audit as a methods
  deliverable), the three B-benchmark scripts + README (tier-labeled headers),
  the M2b prompt (new deliverable: per-source input manifests in a `docs/09`
  audit; B-ladder DoD = T2 differential + T3 scalings, T4 only with manifests),
  QUESTIONS (B5a collisionality reframed as the audit type specimen), and the
  numerics chapter / islands.md status. Docs-only; no `src/` or test changes.
- **Why it strengthens the project**: T2 internal differentials give *sharper*
  claims than absolute matches (we control both sides); the input audit is
  itself publishable reproducibility content; and it aligns the ladder with the
  SLAYER-validation precedent Islands models itself on.
- **Next**: unchanged — M2b derivation lane, now with the input-completeness
  audit folded into its DoD.

## 2026-07-08 — M2 L0 solve machinery: Newton–Krylov + moments + species/frames/fields (structure, gated physics)

- **Contract**: `docs/src/islands/design/M2-launch-prompt.md` (interactive /goal
  run). Branch `feature/islands-m2`, **PR #324** (stacked on
  `feature/islands-m1`/PR #320; retargets to `feature/islands` when #320 merges).
  Full suite green locally.
- **Moved**: the full L0 solve *structure*, every physics coefficient a supplied
  `[VERIFY]`-gated parameter (physics-verifier: **PASS**):
  - `solvers/` — matrix-free Newton–Krylov (Krylov.jl GMRES on a preallocated
    ForwardDiff JVP; Eisenstat–Walker; line search; convergence on norm AND
    max-norm per `04 §5`), `YBlockJacobi` physics-block preconditioner with
    TSVD-regularized pencil solves (the `04 §3` y_c treatment), dense tiny-grid
    debug Jacobian, pseudo-arclength continuation with fold detection (toy fold
    found at step 6 of the test problem).
  - `species/` (D3 plumbing), `frames/` (conversion forms, NaN-gated
    `FrameConvention`), `fields/` (Q(Ω)/h(Ω) structure + NaN-gated
    `ElectronClosure`), `moments/` (J̄_∥, Δ projections with required gated
    prefactors, ⟨·⟩_Ω diagnostics), operators additions (mimetic
    `PitchAngleDiffusion`, `FarFieldConditions` — never bare Neumann,
    `weighted_moment!`).
  - **Structural gates green** (67 new tests in `runtests_islands_solve.jl`):
    A5 (residual exactly 0 at g≡0), solve-MMS at design order (3.98 observed,
    nx 17→33), A4 (conservation ≲1e-11, entropy sign exact), A3 parity, A7
    ⟨∂²h/∂x²⟩_Ω ≈ 1e-16, A8 σ_min monitor + singular detection. Preconditioner
    cuts a stiff collisional solve from 79.5 s/28 Newton to 0.6 s/7 (GMRES
    1700→200-class); all new kernels pass `--check-bounds=yes`.
  - `benchmarks/islands/` created: B2/B4/B5 scripts **skipped**, each naming its
    gating QUESTIONS IDs; `regression-harness` case `islands_l0_structural`
    (solve-MMS err 5.254e-2, 6 Newton/1210 GMRES, A7 8.0e-17, σ_min 0.1139).
  - Paper-I figure contract: `docs/src/islands/papers/paper-1/OUTLINE.md`
    (claims C1–C3 green as CI artifacts; C4–C8 gated on Q2–Q4).
  - **Rendered docs story** (user-flagged gap vs docs/07's M0–M1 intent): new
    `docs/src/islands/numerics.md` — the equations + figures of everything as
    implemented — plus the pinned figure script
    (`benchmarks/islands/figures/make_structural_figures.jl`, five structural
    figures committed as docs assets) and a full "Islands" site-nav section
    (overview, numerics chapter, Paper-I contract, design docs 00–08).
    Remaining docs/07 infra for later milestones: anchor-sync CI check,
    STATE.md dashboard.
- **Physics debugging note**: the first solve-MMS attempt failed to converge —
  the generic `Collisions` (a_y ∂²y) term has no y-BCs, so its BVP
  discretization is unstable under refinement; the *mimetic* divergence form
  (degenerate P → 0 endpoints, zero-flux built in) is the correct structure and
  the far-field x-BCs are what make the advective solve well-posed. Exactly the
  design's point (`01 §3`, `04 §1`).
- **Blocked**: the York gates (B5a/b/c, B2, B4) and Paper-I claims C4–C8 — all
  on the human clearance queue **Q2/Q3/Q4** (unchanged).
- **Next**: human clears Q2–Q4 → thin run fills the L0 coefficients from the
  D7 re-derivation and un-skips the B-ladder; independent M2+ work: kinetic-
  electron toggle (E4), io/ TOML section, trace-species linear pass.

## 2026-07-08 — M1 skeleton: phase-space grids + operator stack + MMS/AD harness

- **PR**: #320 (`feature/islands-m1` → `feature/islands`); full suite green.
- **Moved**: Landed the M1 core (design `03 §1–2`, `04`, ladder `A1/A2`). Three
  `src/Islands/` submodules, all structure-only (no `[VERIFY]` physics numbers):
  - `phasespace/PhaseSpace.jl` — the `(x, ξ, y, E, σ)` grids with layer-clustered
    maps: Fourier spectral `∂ξ`, Fornberg high-order FD `∂x`/`∂y` on `sinh`-stretched
    grids (window sized per-derivative so `D1`/`D2` are both 4th-order incl.
    boundaries), composite-Simpson quadrature weights, Gauss–Laguerre energy nodes.
  - `operators/Operators.jl` — `AbstractTerm` + `apply!` + `residual!`; the term
    structs of `03 §2` (`ParallelStreaming`, `MagneticDrift` with the
    `:original/:improved` toggle, `ExBDrift` as the `(x,ξ)` Poisson bracket,
    `Collisions`, `GradientDrive`, `PerpTransport`/`RadiationSink` L4 stubs,
    `Quasineutrality` field residual). Every physics coefficient is a **supplied
    data field** — no literal in `src/`. Allocation-free, AD-generic.
  - `verify/Verify.jl` — manufactured-solution + AD-vs-FD JVP harness.
  - Tests `test/runtests_islands_{grids,operators}.jl` (wired into `runtests.jl`):
    A1 per-operator MMS → 4th order for `∂x/∂y` terms, machine-precision for the
    `∂ξ` term; assembled kinetic residual → 4th order; A2 JVP-vs-FD agree to ~6e-9;
    **allocation regression = 0 bytes** for every `apply!` and `residual!`. All
    53 Islands tests green. Added `ForwardDiff` to `Project.toml` (design `04 §9`).
- **physics-verifier**: PASS — audited all six new/changed files, no
  `[VERIFY]`-policy violation; the flagged literature numbers (8.73/1.46 ρ_bi,
  k=−1.173, …) appear only in docstring prose, never assigned to a coefficient.
- **Blocked**: nothing. **Q1 RESOLVED**: julia is at
  `/mnt/homes_global/ncl2128/software/julia-1.11.7/bin/julia`; must be run with
  `env -u LD_LIBRARY_PATH` (OMFIT contamination). Used it to run the suite here.
- **Next**: M2 — wire moments (`Δ_cos`, `Δ_sin`), `frames/`, `species/`, and the
  Newton–Krylov solver toward the L0 single-species solve; every physics
  coefficient stays `[VERIFY]`-gated with a skipped benchmark until cleared.

## 2026-07-08 — Harden Stop hook against OMFIT LD_LIBRARY_PATH contamination

- **Moved**: Diagnosed why the Stop hook's package-load check fails on this box.
  A loaded OMFIT module (`module load omfit/unstable`) leaks
  `LD_LIBRARY_PATH=/mnt/codes/atom/mambaforge/envs/omfit/lib:` into the session;
  those conda libs shadow Julia's bundled artifacts, giving `undefined symbol`
  errors in CHOLMOD and the Plots/Cairo/GR native stack (and the ubiquitous
  `libtinfo.so.6` bash warning). Not a code issue — CI is green, and
  `env -u LD_LIBRARY_PATH julia … using GeneralizedPerturbedEquilibrium` loads
  clean (exit 0). Fixed `stop-check.sh` to run the build check with
  `env -u LD_LIBRARY_PATH` (no-op on a clean shell / CI). Repo deps were
  instantiated here; the shared depot (`/mnt/codes/ncl2128/.julia`) is populated.
- **Blocked**: nothing new. This is the concrete shape of **Q1** — the
  automation shell must invoke julia with a clean `LD_LIBRARY_PATH` (unload
  OMFIT, or unset the var) or the overnight loop's *actual* gpec runs fail the
  same way, not just the hook.
- **Next**: (human) launch the loop from a shell without the OMFIT module
  (`module unload omfit`); hook hardening is defense-in-depth on top of that.

## 2026-07-08 — Fix invalid deny rules in `.claude/settings.json`

- **Moved**: `/doctor` flagged two skipped permission-deny rules —
  `Bash(git push:* main)` / `Bash(git push:* develop)` — invalid because `:*`
  (prefix match) is only allowed at the end of a pattern. Rewrote them with a
  mid-pattern wildcard (`Bash(git push* main)` / `Bash(git push* develop)`) so
  they load and again deny pushes to `main`/`develop` for any remote/flags.
- **Blocked**: nothing.
- **Next**: unchanged — pending items are the Phase A bootstrap **Next** below.

## 2026-07-08 — Phase A bootstrap (supervised)

- **Moved**: Created the `Islands` submodule skeleton (`src/Islands/Islands.jl`,
  empty `module Islands`) and wired it into `src/GeneralizedPerturbedEquilibrium.jl`
  (`include` + `import . as` + `export`, last submodule slot before `Rerun.jl`).
  Stood up this `LOG.md` and `QUESTIONS.md`, the `.claude` unattended-run
  guardrails, the `physics-verifier` subagent, and the M1 launch prompt.
- **Landed CI-green** on `feature/islands` (PR #318): both `runtests` jobs pass
  (the wiring is valid — the package loads and the full suite passes) and the
  docs build passes. One fix was needed en route: the exported `Islands` module
  docstring required a manual page under `checkdocs=:exports`, so
  `docs/src/islands.md` (an `@autodocs` block) was added and wired into
  `docs/make.jl` (repo-root CLAUDE.md docs-coverage rule).
- **Blocked**: `julia` is not on the automation shell's PATH (no module, not in
  `$HOME`) → changes could not be run locally; CI is the only Julia validation
  here. See **Q1** — the overnight loop's scratch-clone environment must expose
  `julia` or it cannot run tests / meet M1's definition-of-done.
- **Next**: (human) resolve Q1 + one supervised `dontAsk` dry-run of the hooks,
  then launch the overnight loop on milestone **M1** (design `00 §M1`) —
  phase-space grids + operator-stack skeleton + MMS/AD harness (ladder A1, A2),
  no `[VERIFY]` physics coefficients.
