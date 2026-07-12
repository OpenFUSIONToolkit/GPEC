# Islands — session LOG

Cross-session memory spine (design doc `06 §2.5`). Read this at session start
together with `QUESTIONS.md`. Append a short entry before every session end:
**what moved / what's blocked / next action**. Newest entries at the top.
Reference `QUESTIONS.md` IDs (`Q<n>`) and ladder IDs (`A1`, `B5a`, …) where
relevant.

---

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
