# Plan — confirm A1 (MMS) + A5 (zero-drive null) on a stalling physical config

**Purpose.** Decide, with evidence, whether the ~1e-3 residual floor that blocks the
physical `nE≥3` solve (LOG cont. 9–14) is a **structural discretization inconsistency**
(the assembled physical operator+BC system has no exact root / is order-deficient) or a
genuine solver/nonlinearity issue. This is the step-back to the *bottom of the docs/05
ladder* the milestone skipped: A-rungs must be green on the **physical** config before
any physics reproduction (B1) or threshold (B5). Read with `LOG.md` cont. 14 and
`QUESTIONS.md` Q7.

**Definition of done.** A clear verdict + committed evidence:
- **Verdict S (structural):** A5 and/or physical-coefficient A1 FAIL on a stalling config
  ⇒ the discretization/BC assembly is defective; localize the offending rows/term. This
  is the good outcome — it turns a vague "solve stalls" into a concrete bug.
- **Verdict N (not structural):** A5 passes (machine-zero homogeneous solve) AND
  physical-coefficient A1 converges at design order ⇒ the discretization is sound; the
  ~1e-3 floor is genuine nonlinearity/consistency-of-the-driven-problem — route to the
  coker(J) probe (Step 3) to confirm whether the *driven* system is inconsistent.

**Guardrails.** Numerics/diagnostics only — no physics coefficient/sign/normalization
changes (any such change → physics-verifier + QUESTIONS). Never weaken a tolerance to
"pass." Append LOG.md and push before ending. Reuse the existing `Verify` harness; do
not reimplement MMS/null machinery.

## Environment / launch (learned the hard way this week)
- Run Julia as `env -u LD_LIBRARY_PATH julia --project=.` (omfit conda leaks SuiteSparse
  5 → CHOLMOD init failure otherwise).
- Launch long runs with the **Bash tool's `run_in_background: true`** — manual
  `nohup`/`setsid` redirects failed with exit 144 / missing output files.
- **Never `pkill -f <pattern>`** where `<pattern>` matches the scratch script name you
  are writing/launching — it kills the `cat`/`julia` writing it.
- Right-size: dense `dense_jacobian`/SVD only at `N ≲ 6000`. Check `free -g` and for
  competing `julia` before launching (the node is shared; OOM-kills happen).

## The pinned stalling config (use throughout)
Hand-set, the smallest robust staller (LOG cont. 13/14: krylov floors at ~1e-3):
```
phys = Configure.Level0Physics(; epsilon=0.1, inv_Lq=1.0, inv_LB=0.7, q_s=2.0,
    dq_dpsi=0.8, w_psi=0.05, mu0_R=3.0, inv_Ln0=1.0, rho_hat_theta_i=0.05, eta_i=0.5,
    nu_star=0.5, m=2.0, tau=1.0, variant=:original)
grid = PhaseSpace.resolved_island_grid(; w=0.05, nx=15, K=4, Lx_over_w=6.0, nxi=6,
    ny=9, nE=4, y_max=(1+ε)/(1−ε), y_c=1.0, clustering_y=0.8, order=4)   # N≈6570, STALLS
```
Also keep a **converging** control (same but `nE=3`, N≈4950, converges to 2.2e-9) so every
probe is run on stall-vs-converge and the *difference* is the signal.

## Reusable harness (`src/Islands/verify/Verify.jl`)
`manufactured_state(grid)`, `mms_assembled_error(grid)`, `mms_operator_error(grid,term)`,
`estimate_order(errors, refine)`, `solve_mms(...)`, `zero_drive_setup(...)`,
`yc_block_sigma_min(...)`, `build_stack(grid; coeffs)`. **Caveat:** the existing A1 tests
(`test/runtests_islands_operators.jl`) run MMS with **manufactured coefficients on an
abstract box** (`halfwidth_x=6`) — they prove the *machinery* is order-correct but do
**not** exercise the **physical** coefficients/grid/BCs. Step 2 closes exactly that gap.

## Step 1 — A5 zero-drive null on the physical config (fast, most decisive)
Hypothesis tested: is the homogeneous physical operator+BC assembly consistent and
non-singular (independent of the physical drive)?
1. Build the physical config, then **zero every drive**: the interior `GradientDrive`
   source (already 0), the far-field `g_far`/`Φ_far`, AND the quasineutrality source
   `S_Φ`. The clean lever: set **`inv_Ln0 = 0`** — both `g_far ∝ inv_Ln0` and
   `S_Φ ∝ inv_Ln0` vanish (verify this by reading `gradient_far_field` +
   `quasineutrality_source`), giving a purely homogeneous system whose exact solution is
   `U ≡ 0`. Cross-check with `Verify.zero_drive_setup` (use it if it already does this).
2. **A5.1:** `residual(U=0)` must be **machine zero** (≲1e-13). A nonzero residual at
   `U=0` with all drives off ⇒ a spurious constant term in the assembly — structural bug.
3. **A5.2:** from a *nonzero random* init, `newton_krylov` (well-resourced: max_iter=120,
   memory=400, PlaneJacobi) must drive to **machine zero** and `‖U‖→0`. If it floors at
   ~1e-3 for the homogeneous system too ⇒ the operator/BC is inconsistent or singular
   **independent of the physics drive** — the cleanest possible localization.
4. Run A5.1/A5.2 on BOTH the stalling (nE=4) and converging (nE=3) configs. **Gate:** any
   A5 failure on either ⇒ **Verdict S**, and the term/BC is structural — go to Step 4.

## Step 2 — A1 with PHYSICAL coefficients on the physical grid
Hypothesis tested: does the assembled residual using the **physical** coefficients
(`configure_level0`'s stack + bc), not manufactured ones, converge at design order?
1. Manufacture a smooth `U*` (`manufactured_state`) on the physical grid; apply the
   **physical** stack (from `configure_level0(grid, phys, species)`) to get the source
   `S* = F_phys(U*)`; solve `F_phys(u) = S*` and check `‖u − U*‖ ≲ tol` (`solve_mms`
   pattern, but with the physical stack — extend `solve_mms`/write a scratch driver if it
   only takes manufactured coeffs).
2. **Order:** refine the *radial/velocity* resolution (nx, ny, nE up) at fixed physical
   `phys`; `estimate_order` on `‖u−U*‖` must show the design order (≈4 for the interior;
   the `ξ` spectral part is separate). Compare stall-config resolution vs the converging one.
3. **Gate:** if physical-coefficient MMS does **not** converge at design order (esp. as
   `nE` grows past 2, where the stall onsets) ⇒ **Verdict S** — the physical discretization
   is order-deficient (candidate: the `y_c`-straddling nodes, the far-field row coupling,
   or the collision operator at physical magnitude). If it converges cleanly ⇒ the
   discretization is sound for smooth solutions → Step 3.

## Step 3 — coker(J) inconsistency probe at the stalled driven solve (the direct test)
Only if A5 + A1 pass (Verdict N so far): is the **driven** (physical-`inv_Ln0`) system
inconsistent — `F ∉ range(J)` at the floor?
1. Run the physical solve to its ~1e-3 floor; take `u_floor`, `F_floor`.
2. `J = dense_jacobian(f!, u_floor)` (N≈6570 ok); SVD `J = UΣVᵀ`.
3. Compute the projection of `F_floor` onto the **left-null-space** (columns of `U` with
   `σ ≈ 0` relative to `σ_max`): `‖U_nullᵀ F_floor‖ / ‖F_floor‖`. **Large (O(1))** ⇒
   `F_floor` is largely *outside* `range(J)` ⇒ `Jδ=−F` is inconsistent ⇒ no Newton step
   reduces it ⇒ the floor is a **genuine inconsistency of the driven discretized system**
   (confirms the LOG cont. 14 hypothesis). **Small** ⇒ `F` is reachable ⇒ the floor is a
   basin/globalization issue after all (re-open the solver angle, but now with proof it's
   not inconsistency).
4. If inconsistent: identify WHICH rows carry `F_null` (unflatten `U_null`/`F_floor`;
   which `(ix,iξ,iy,iE,iσ)` block dominates?) — that localizes the offending
   operator/BC (expected suspects: far-field row replacement × forbidden-`y` pinning ×
   the `y_c` layer).

## Decision tree / hand-off
- **A5 fails** → structural bug in the homogeneous assembly (a spurious source or a
  singular/near-singular operator+BC). Localize the term; fix under numerics rules
  (physics-verifier if any coefficient/sign touched). Highest-value outcome.
- **A5 passes, A1 (physical) fails** → the physical discretization is order-deficient;
  localize by which term/region breaks the order (bisect nE, ny, and the far-field/`y_c`
  handling). Likely a formulation/BC interaction.
- **A5 + A1 pass, coker(J) large** → the *driven* system is inconsistent → it is a
  formulation/BC over-determination (far-field + forbidden-`y` + operator); fix the BC
  formulation (this is where a York-faithful review of the BC/closure earns its keep).
- **A5 + A1 pass, coker(J) small** → not inconsistency; a real globalization problem —
  but now we *know* the discretization is sound, so a proper continuation/homotopy in the
  physics (not the failed Ψtc/LM) is justified, or the York warm-start recipe (L23 §7.1).

In every branch: append a LOG entry with the verdict + evidence, update this plan's
checklist, update QUESTIONS Q7, push `feature/islands`. Then the next milestone step is
**B1 (no-island neoclassical bootstrap vs Sauter/NEO)** — the first physics reproduction,
which the verdict here tells us whether to expect to converge.

## Progress checklist
- [ ] Confirm `inv_Ln0=0` zeros all drives (read `gradient_far_field` + `quasineutrality_source`); confirm `zero_drive_setup` semantics
- [ ] A5.1 residual(U=0) machine-zero — stall config + converging control
- [ ] A5.2 homogeneous solve → machine zero from random init — stall + control
- [ ] A1 physical-coefficient MMS recovers `U*` on the physical grid
- [ ] A1 order estimate at design order as nE/ny refine (stall vs control)
- [ ] (if Verdict N) coker(J) projection at the ~1e-3 floor; localize `F_null` rows
- [ ] Verdict recorded (S or N) in LOG + QUESTIONS; plan checklist closed; pushed
