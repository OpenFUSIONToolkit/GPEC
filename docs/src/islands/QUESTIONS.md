# Islands — blocker queue (QUESTIONS)

Append-only non-blocking escalation queue (design doc `06 §2.2`). When blocked
on anything the CLAUDE.md forbids guessing — `[VERIFY]` clearances, physics
coefficients, signs, normalizations, convention/doc contradictions, or a tooling
prerequisite — write an entry here **and switch to the next unblocked task**.
Never stall waiting for a human; never resolve silently.

Each entry:
- **ID**: `Q<n>` (monotonic). Commits/PRs reference the IDs they were blocked on
  or unblocked by.
- **Status**: `OPEN` / `RESOLVED (by <who>, <date>)`.
- **Context**: what you were doing.
- **Question**: the specific thing a human must decide.
- **Options**: the alternatives considered.
- **Recommendation**: your best guess (not acted on until cleared).
- **Gated work**: what is blocked until this resolves.

The human's recurring job is clearing this queue (and `[VERIFY]` tags), not
supervising sessions.

---

## Q1 — Julia not on the automation shell PATH — RESOLVED (by Claude, 2026-07-08)

- **Context**: Phase A bootstrap. Verifying the `Islands` module skeleton loads
  (`using GeneralizedPerturbedEquilibrium`) and running the test suite requires
  `julia`, but it was not on the non-interactive shell's PATH (no `julia` module,
  none in `$HOME`; other users have installs under `/mnt/homes*/…/julia*`).
- **Question**: What is the canonical `julia` invocation for automation on this
  cluster, and will the overnight loop's scratch-clone environment expose it?
- **Resolution**: the `ncl2128`-owned install is at
  `/mnt/homes_global/ncl2128/software/julia-1.11.7/bin/julia` (option (b): an
  absolute path to a user-owned binary), and it is on this session's PATH. The
  M1 run used it to build and run `test/runtests.jl` locally. The **only caveat**
  is the OMFIT `LD_LIBRARY_PATH` contamination already documented in LOG
  (2026-07-08): the binary must be invoked with a clean loader path —
  `env -u LD_LIBRARY_PATH /mnt/homes_global/ncl2128/software/julia-1.11.7/bin/julia
  --project=. …` — or the conda libs shadow Julia's bundled artifacts. The Stop
  hook already applies `env -u LD_LIBRARY_PATH`; the overnight loop's launch
  script must do the same for its own gpec runs.
- **Gated work (now unblocked)**: local verification of every Julia change; the
  overnight loop's ability to run tests / meet its definition-of-done.

## Q2 — Ratify Decisions D7 and D8 — RESOLVED (by the user, 2026-07-08)

- **Resolution**: both ratified as written (option (a)); recorded in the
  `docs/00` Decision Log. D7 additionally carries the user's clearance-mode
  choice for Q3: **re-derivation first** — the L0 coefficient set is cleared by
  human sign-off of in-repo derivations, not literature transcriptions.

- **Context**: M2 setup. The L0 equation set and benchmark targets rest on two
  Decision-Log proposals (`docs/00`) that are dated 2026-07-07 and still marked
  "needs human ratification". Until ratified, M2 cannot fix the L0 physics form or
  pin its benchmark tolerances.
- **Question**: Ratify (or amend) **D7** — implement Level-0 physics from an
  *independent re-derivation* cross-checked against the L23-amended equation set,
  treating I19 Eq. (A.1) as printed as known-errata (L23 §2.6), with ω_E a scanned
  input from day one — and **D8** — pin the benchmark grid as the three-code
  triangle (DK-NTM / RDK-NTM / kokuchou) with B5a/b/c configs, superseding the
  single "York thresholds" gate item.
- **Options**: (a) ratify both as written; (b) ratify with amendments; (c) direct a
  different L0 derivation strategy.
- **Recommendation**: ratify both — they encode the project's core anti-guessing
  thesis (published O(1) coefficients in this lineage are demonstrably wrong) and
  the benchmark structure the ladder already assumes.
- **Gated work**: pinning any L0 physics coefficient; the York gates B5a/b/c; the
  Paper-I physics claims.

## Q3 — Clear the Level-0 coefficient set ([CHECKED] → human sign-off) — OPEN (mode decided)

- **Mode decision (user, 2026-07-08, via Q2/D7)**: option (b) — **re-derivation
  first**. The next milestone (M2b) produces independent derivations of each
  item below in `docs/src/islands/derivations/` (marked `[DERIVED]`, with a
  cross-check table against the `[CHECKED]` transcriptions and every
  discrepancy flagged); the human then signs off the *derivations*, which
  clears the corresponding coefficients. Items remain individually OPEN until
  that sign-off.

- **Context**: M2 builds the L0 solve machinery with every physics coefficient a
  parameterized `[VERIFY]` stub. Reaching the York gates needs these `[CHECKED]`
  (AI-transcribed, cited, but not human-signed-off) items cleared. `[CHECKED]` is
  not permission to hardcode (`docs/01` header). Each is implemented structurally;
  none has a value in `src/`.
- **Question**: Check each against its source PDF and sign off (record paper + Eq./p.
  in `docs/01`), or flag a discrepancy:
  - `ω̂_D` magnetic-drift frequency + the `:original`/`:improved` `L̂_B⁻¹` toggle
    `[CHECKED: I19 Eq. (32) def.; D21 Eqs. 15, B1; D21 Eq. A2, p. 16]` — drives the
    8.73 → 1.46 ρ_bi threshold shift.
  - Pitch-angle collision kernel `[CHECKED: I19 Eqs. (9)–(12); Diss19 Eqs. 2.25–2.30;
    WCHH96 Eq. (62)]` + the analytic velocity average
    `⟨ν̂_ii⟩_u = (4ε^{3/2}ν_★/3√π)(√2 − ln(1+√2))` `[CHECKED: L23 Eq. 4.1.6, p. 88]`
    + normalization `ν_★ = ν_jj Rq/(ε^{3/2}v_th)` `[CHECKED: L23 Eq. (2.3.40)]`.
  - Electron-closure constants `k ≃ −1.173` (Hirshman–Sigmar) and `f_p ≃ 1 − 1.46√ε`
    `[CHECKED: I19 Eq. (22); L23 Eqs. 2.5.5–2.5.8]`.
  - Quasineutrality closure `e_iΦ̂/T_i = [δn̄_i/n₀ + x − ĥ(Ω)]/(2 L̂_{n0})`
    `[CHECKED: I19 Eq. (A.11); L23 Eq. (2.4.14)]` and the Picard form
    `δΦ̂ = (δn̂_i − δn̂_e)/2` `[CHECKED: Diss19 Eq. 2.45]`.
- **Options**: (a) clear item-by-item after PDF check; (b) require an independent
  re-derivation first (couples to Q2/D7) before clearing.
- **Recommendation**: clear via re-derivation (per D7) rather than transcription
  alone — L23 §2.6 documents concrete errors in the published set.
- **Gated work**: populating any of these coefficients; the York/large-w/polarization
  gates (B5a/b/c, B2, B4); the A7 number-bearing identities (`k`, `f_p`, `⟨ν̂_ii⟩`).

## Q4 — Resolve open [VERIFY]s and acquire two missing sources — OPEN

- **Context**: Independent of the `[CHECKED]` clearances above, three genuinely
  open `[VERIFY]` items block pinning M2's moment normalization and B5a tolerance.
- **Question**:
  - `ψ̃` amplitude: is it `(w_ψ²/4)(q_s′/q_s)` (Diss19/D21/L23, dimensional analysis)
    or `(w_ψ²/4)(q_s/q_s′)` (one I19 extraction)? `[VERIFY: check I19 as printed —
    possible typo in the paper itself]` — sets the `Δ_cos/Δ_sin` prefactor.
  - B5a run collisionality: I19 §4.2 states `ν_★ = 0.01`; L23 p. 82 quotes DK-NTM at
    `ν_★ = 10⁻³`. **Reframed by Decision D9 (2026-07-11)**: this is no longer a
    tolerance to pin but the **type specimen of the input-completeness problem**
    — I19 is internally inconsistent about its own run, so its absolute w_c is a
    T4 (audit-gated) target, not a pass/fail number. It is resolved *in the M2b
    input manifest* (`docs/09`/audit), recording both values as the source's
    ambiguity; the primary B5 gates (T2 toggle differential, T3 existence/trend)
    do not depend on it.
  - ~~Acquire WCHH96 and Park PoP 29 (2022)~~ **Both resolved (2026-07-09)**:
    the user added WCHH96 (Wilson, Connor, Hastie & Hegna, PoP **3**, 248
    (1996), doi:10.1063/1.871830) to the island reference library, and Park
    2022 was found already in-repo in the general `docs/resources/` dir (the
    docs/08 island-subfolder map had missed it; map corrected). The user also
    flagged **Burgess 2026** (`docs/resources/2026-Burgess-…Two-Fluid Slab
    Layer.pdf`) as the methodological template: toroidal outer-region Δ′ +
    SLAYER regime-generalized linear layer — Islands extends that inner region
    from zero-width linear layers to finite-width islands (recorded in docs/08
    as **B26**). Remaining open in Q4: the `ψ̃` amplitude check and the B5a run
    collisionality.
- **Options**: (a) user resolves from the source PDFs; (b) an independent
  re-derivation pins `ψ̃` and the collisionality is recorded in the input manifest.
- **Recommendation**: resolve `ψ̃` by re-derivation (a clean dimensional check, M2b);
  record the B5a collisionality ambiguity in the input manifest rather than
  "resolving" it — per D9 the absolute B5a value is audit-gated, not a gate.
- **Gated work**: the `Δ_cos/Δ_sin` normalization (the `ψ̃` item); the T4 B5a/B2
  absolute comparisons (the manifest items) — the T2/T3 primary gates are not
  blocked by these.

**Tier note (Decision D9, 2026-07-11):** verification targets are now tiered by
reproducibility (docs/05 "Target tiers"). This changes what the Q3/Q4 clearances
*buy*: a cleared coefficient set un-gates the **primary** T2 (internal
differentials) and T3 (scalings/trends/existence) physics gates directly; the T4
absolute literature comparisons additionally require the M2b input-completeness
audit and are reported only with input manifests + sensitivity scans, never as
bare pass/fail. The derivation lane inherits this framing.

## Q5 — The remaining un-cleared Level-0 operator coefficient families (a second derivation lane) — OPEN

- **Context**: M2c assembled `Configure.configure_level0` — the Level-0
  named-configuration builder. Wiring the cleared coefficients onto the operator
  stack surfaced that the M2b derivation lane cleared the coefficient families
  that appear as *closures/moments* (`ω̂_D`, the collision `P`/`ν` shapes, the
  quasineutrality scalar, the `Δ` prefactors), but **several operator-stack
  coefficients are not yet a cleared family** and were left supplied/gated in
  `Configure.GatedLevel0Inputs`. `ω̂_D` (`MagneticDrift.c_D`), the pitch
  diffusivity shape, the deflection-frequency shape, and the `Δ` prefactors *are*
  wired from cleared `Coefficients.*`; the items below are not.
- **Question**: clear (by the D7 re-derivation-first route, `docs/derivations/`,
  human sign-off) each remaining Level-0 operator coefficient:
  - **Parallel-streaming** `a_xi`, `a_x` (`Operators.ParallelStreaming`):
    **RESOLVED (2026-07-11)** — re-derived (`parallel-streaming.md`, signed off)
    from I19 Eq. 32; the coefficients factor exactly into `{Ω, ·}` flux-surface
    advection (a coefficient-free structural check). Implemented as
    `Configure.streaming_coefficients` with a new `Level0Physics.rho_hat_theta_i`;
    normalization chosen to keep the cleared `c_D = ω̂_D` unchanged.
  - **`E×B` coupling** `c_E` (`Operators.ExBDrift`) — **RESOLVED (2026-07-12**,
    `exb-coupling.md`, signed off): matching the master-eq E×B terms to the
    `ExBDrift` Poisson bracket in the `c_D=ω̂_D` normalization (÷ −m ρ̂_θi, `ρ̂_θi`
    cancels) gives **`c_E = ½⟨1/v̂_∥⟩_θ`**, `1/v̂_∥ = σ/(v̂√(1−yb))`. The three
    wrinkles resolved: (i) `c_E` is velocity-dependent `(y,E,σ)` → `Operators.ExBDrift`
    generalized to accept an **array** coefficient (backward-compatible with the
    scalar test path); (ii) a new orbit bracket `B₁(y) = ⟨1/√(1−yb)⟩_θ`
    (`Coefficients.orbit_average_exb_bracket`, same machinery + y_c singularity as
    the drift `G`); (iii) the **σ-parity nailed** — passing (`y<y_c`) is **σ-odd**
    `c_E = (σ/2√E)B₁(y)`; trapped (`y>y_c`) is **identically zero** (the σ-odd
    `1/v̂_∥` cancels between the two banana legs under `Σ_σ`), which the
    drift-island label *requires* (trapped `S ∝ p̂`, docs/01 §2.2). So E×B is a
    **passing-particle** effect, like island-streaming. Implemented as
    `Configure.exb_coupling_table`; introduces no new physics parameter (only `ε`).
  - **Gradient drive** `drive` + **far field** `bc` — **RESOLVED (2026-07-11**,
    `gradient-drive.md`, signed off): reading I19 first-hand (Eqs. 28–32), the
    master equation is **homogeneous** (no interior source); the drive is the
    **far-field boundary condition** `Ḡ₀ → p_φ(ω_si^T/ω_si)(n'/n)F_Mi = p_φ F'_Mi`
    (Eq. 29 — the ratio is `ω_si^T/ω_si = 1+(v̂²−3/2)η_i`, a **temperature factor,
    not a frequency ratio**; an earlier reading of `ω_ci` was a misread, so **no
    frame convention is needed**). Implemented: `Operators.GradientDrive = 0` and
    `Configure.gradient_far_field` builds `g_far = x L̂_{n0}⁻¹[1+(E−3/2)η_i]`
    (`Φ̂_far = 0` at `ω_E = 0`), with a new `Level0Physics.eta_i`. The
    `gradient_drive` **and** `far_field` families are both cleared.
  - **Quasineutrality closure — RESOLVED (2026-07-11).** The cleared closure
    `Φ̂ = τ/(τ+1)[δn̄_i/n₀ + L̂_{n0}⁻¹(x−ĥ)]` (Q3, `quasineutrality-closure.md`,
    signed off) is now implemented: `Operators.Quasineutrality` carries a
    `source` field, and `Configure.configure_level0` builds the residual
    `R_Φ = M[g] − α Φ̂ + S` with `α = (τ+1)/τ` (= `1/quasineutrality_coefficient(τ)`)
    and `S = L̂_{n0}⁻¹(x−ĥ)` (`Configure.quasineutrality_source`, from the cleared
    `h_amplitude`/`h_profile`). The Level-0 potential is now driven (was trivially
    zero). docs/01 §3 records it; the derivation §6 was the authorization. **No
    longer gates a Level-0 physics run.**
  - **Collision magnitude** `nu_tilde`: **RESOLVED (2026-07-12**,
    `collision-magnitude.md`, signed off): read L23 Eqs. 4.1.4–4.1.6 (p. 87–88)
    first-hand and derived the momentum-restoring average
    `⟨ν̂_ii⟩_u = (4ε^{3/2}ν_★/3√π)(√2−ln(1+√2))` from L23's reduced integrand +
    three standard integrals (`∫u e^{−u²}erf=1/2√2`, `∫e^{−2u²}=½√(π/2)`,
    `∫u⁻¹e^{−u²}erf=ln(1+√2)`), reproducing L23's unit-test `1.267537×10⁻⁴` to 7
    digits — cleared as `Coefficients.momentum_restoring_average`. The collision
    **magnitude** `nu_tilde = ε^{3/2}ν_★` is wired from a new `Level0Physics.nu_star`
    scenario field (§4 normalization, already signed off); `:nu_tilde` moved
    gated→cleared, un-gating the collision operator's magnitude. The
    momentum-restoring *operator term* is a separate future addition.
  - **Passing fraction** `f_p ≃ 1 − 1.46√ε` (electron closure, Q3): **RESOLVED
    (2026-07-11)** — `derivations/passing-fraction.md` signed off; cleared as
    `Coefficients.passing_fraction(ε) = 1 − 1.4624√ε` (the effective
    trapped-fraction coefficient, = quoted `1.46` to 3 s.f.), authorizes
    `Fields.ElectronClosure.f_p`. The Hirshman–Sigmar `k ≃ −1.173` remains
    escalated (needs the parallel-viscosity moment problem).
  - **Orbit-averaged pitch measure** `B_profile`: **RESOLVED (2026-07-12**,
    `orbit-averaged-collision.md`, signed off). Reading L23 Eq. 2.3.47 / appendix
    8.3.2 first-hand showed `B_profile` is the tip of the **full** orbit-averaged
    collision operator (six terms). The pitch diffusivity is the orbit-averaged
    `⟨√(1−yb)⟩_θ` (not a local `B`), giving the exact `y`-divergence
    `∂_y(y⟨√(1−yb)⟩_θ ∂_y)` with flat measure; and the pitch diffusion is **σ-odd**
    (the `1/v̂_∥` weight) — a correction to the former σ-even placeholder. Cleared &
    implemented: the five **differential** terms (D+E pitch, A drag, B neoclassical,
    C cross) via `Coefficients.orbit_average_pitch_brackets` +
    `Configure.*_coefficient` + `Operators.{PitchAngleDiffusion, CollisionalDrag,
    NeoclassicalDiffusion, CollisionalCross}`, plus the forbidden-pitch `g=0` domain
    BC and the new `Level0Physics.m`. `B_profile` and `GatedLevel0Inputs` are
    **removed** — **no gated kinetic inputs remain**. The sixth term (momentum-restoring
    F, nonlocal) is a pending operator addition (see next item).
  - **Momentum-restoring term F** (`2ν̂_ii(1+ε)Ū_∥ᵢ(ĝ+pF̂′)F̂_M`, the collision
    operator's field-particle piece) — **BLOCKED on a gated normalization
    (2026-07-12)**, escalated rather than guessed. The *structure* is clear
    (L23 Eq. 2.3.47/8.3.17): a nonlocal operator that (i) forms the parallel-flow
    moment `Ū_∥ᵢ(x,ξ) = (1/√π⟨ν̂_ii⟩_u) Σ_σ σ ∫du u³ν̂_ii ∫dy ĝ` (b²→1), then (ii)
    redistributes `2ν̂_ii(1+ε)Ū F̂_M`, σ-even, sign positive (÷−m ρ̂_θi of the RHS,
    like the other five). Its magnitude `⟨ν̂_ii⟩_u` is **cleared**
    (`collision-magnitude.md`). **The blocker:** `Ū` is exactly the **parallel-flow
    velocity moment** whose weight `W(y,E,σ)` (the `v̂_∥`-structure, `∫du u³ν̂_ii`)
    is **`[VERIFY]`-gated (Q3)** — `Moments.parallel_current!` / `Operators.jl:185`
    already flag the field-particle piece as Q3-gated. Compounding: the code's
    energy grid is **Gauss–Laguerre** (`Σ wₑ f ≈ ∫ f e^{−E}dE`, the Maxwellian
    folded in), but `Ū`'s moment is a **plain** `∫du u³ν̂_ii` — so the `g↔F̂_M`
    convention (is `g` stored with the Maxwellian in or out?) must be pinned before
    the moment measure is unambiguous. **Question:** clear the parallel-flow
    velocity-moment weight `W` and the `g↔F̂_M` energy-measure convention (Q3), then
    F follows. **Recommendation:** do NOT guess the measure; clear Q3's `W` first
    (it also un-gates the `J̄_∥` output moment). Nothing entered `src/` for F.
    — **UNBLOCKED (2026-07-13)**: Q6 cleared the physical `∫d³v` measure and the
    parallel-flow weight `W = v̂_∥` (`velocity-moment-measure.md`), so `Ū` is now a
    bounded physical moment (`W` + the cleared `u³ν̂_ii/⟨ν̂_ii⟩_u` weight) with no
    remaining gated normalization.
    — **RESOLVED (2026-07-14)**: F implemented as `Operators.MomentumRestoring` +
    `Configure.momentum_restoring_term` — the one nonlocal Level-0 term:
    `Ū(x,ξ) = (1/√π⟨ν̂_ii⟩_u){ν̂_ii v̂_∥ g}_v` (physical measure) redistributed as
    `+2ν̂_ii(1+ε)Ū/(m ρ̂_θi)` (σ-even, positive; `F̂_M=e^{−E}` cancels in the `g=shape`
    convention). Linear in `g`, allocation-free. **The full Level-0 collision
    operator (all six terms) is now complete** — no gated kinetic physics remains.
  - **Neoclassical far field** `bc` (`Operators.FarFieldConditions`): the
    no-island `g_far`/`Φ_far` (never bare Neumann — L23 §5.3), gated physics
    already flagged under Q3.
- **Options**: (a) a focused second derivation lane (an "M2d") clearing these
  item-by-item like M2b, human-present; (b) clear the highest-leverage first
  (streaming + the QN structural fix un-gate a genuine physics residual).
- **Recommendation**: (a) — run it exactly like the M2b lane (re-derive →
  physics-verifier → sign-off → clear into `Coefficients.*` / operator structure).
  The QN structural gap is the highest priority: without the `(x−ĥ)` source the
  Level-0 quasineutrality field is trivially `Φ = 0`, so no Level-0 *physics* run
  is possible until it lands (the M2c assembly runs *structurally* on placeholders
  only). **This is why M2c delivers the assembly scaffold, not a physics result.**
- **Gated work**: any Level-0 *physics* solve (as opposed to the structural
  convergence check); the B-ladder T2/T3 physics gates; the L23/B5c T4 attempt.

## Q6 — The velocity-moment measure convention (blocks the parallel-flow weight W, term F, and re-opens the QN density moment) — RESOLVED (by the user, 2026-07-13)

- **Resolution**: option (b) — the **physical `∫d³v`** measure, with the
  **flux-surface `b`** (`b_min = (1−ε)/(1+ε)`, full trapped range). Derived and
  cleared in `derivations/velocity-moment-measure.md` (signed off 2026-07-13):
  `Operators.velocity_moment!`/`weighted_moment!` gained the physical measure
  (`√E/2` speed Jacobian via Gauss–Laguerre, `1/√(1−y b_min)` pitch Jacobian as an
  exact singular-weight quadrature — the `IinvB` edge, forbidden region zeroed),
  built by `Configure.physical_velocity_weights`. The parallel-flow weight
  `W = v̂_∥ = σ√E√(1−y b_min)` is cleared (`Configure.parallel_flow_weight`,
  resolving Q3's `W`). The QN density moment `δn̄_i = M[g]` is re-implemented with
  the physical measure (wired into `Operators.Quasineutrality`; the closure
  algebra `α`/`S` unchanged; `max|Φ|` shifted 5.7→4.5 as approved). This also
  **unblocks term F** (Q5) — `Ū` is the same physical moment + the cleared
  `u³ν̂_ii/⟨ν̂_ii⟩_u` weight, no new gated normalization.

## Q6 (original) — The velocity-moment measure convention — [see resolution above]

- **Context**: Clearing Q3's parallel-flow velocity weight `W(y,E,σ)` (used by
  `Moments.parallel_current!` for `J̄_∥`, and by term F's `Ū_∥ᵢ`) surfaced that the
  code's velocity-moment machinery (`Operators.velocity_moment!` / `weighted_moment!`)
  uses a **flat** measure — `Σ_σ ∫dy [Simpson, flat] ∫dE [Gauss–Laguerre, e^{−E}]` —
  with the Maxwellian carried by the E-grid (`g = shape`, confirmed:
  `Configure.gradient_far_field` docstring) but **no physical d³v Jacobians**. The
  physical velocity-volume integral is (L23 Eq. 8.4.1/8.4.2)
  `{ĝ}_v = πB Σ_σ ∫dv̂ v̂² ∫dy/√(1−yb) ĝ` — i.e. a `√E/2` speed Jacobian and a
  **`1/√(1−yb)` pitch Jacobian** (whose `y→b⁻¹` singularity L23 §8.4.1 handles by a
  numeric+analytic split, the `IinvB` function). The current `velocity_moment!`
  (flat) lacks both.
- **What is determinable (not a guess)**: for the **parallel flow** the physical
  weight `v̂_∥ = σ√E√(1−yb)` *cancels* the `1/√(1−yb)` pitch Jacobian, so
  `J̄_∥ = πB Σ_σ σ ∫dy ∫dE (E/2) e^{−E} g` ⇒ **`W = const·σ·(E/2)`, flat in y,
  σ-odd, ∝E** (the `const = πB Z` folds into the cleared `Δ`-prefactor `μ₀R/2ψ̃`).
  Term F's `Ū` follows the same structure (with the extra `u³ν̂_ii`/`⟨ν̂_ii⟩_u`).
- **The blocker it exposes**: the **density** moment `δn̄_i = ∫ĝ d³v` has *no*
  such cancellation — it needs the `1/√(1−yb)` pitch Jacobian and the `√E/2` speed
  Jacobian. But the **already-cleared** QN closure (`quasineutrality-closure.md`,
  signed off 2026-07-11) uses the **flat** `velocity_moment!` as `δn̄_i` "directly."
  So the flat convention is either (a) a *deliberate simplification* (then `W` is
  the flat-convention `v̂_∥` weight `σ√E√(1−yb)`, and `δn̄_i` stays flat — physically
  approximate), or (b) the flat `velocity_moment!` is an **incomplete** stand-in for
  the physical `∫d³v` (then it needs the `√E/2` + `1/√(1−yb)`+`IinvB` Jacobians
  added — which **revises the cleared QN `δn̄_i`** and gives `W = σ·E/2`).
- **Question**: which moment convention does Islands intend — (a) the flat
  simplified moment (self-consistent, physically approximate, `W = σ√E√(1−yb)`,
  QN unchanged), or (b) the **physical `∫d³v`** with the `v²`/`1/√(1−yb)`/`IinvB`
  Jacobians (`W = σ·E/2`, and the cleared QN `δn̄_i` re-implemented physically)?
- **Recommendation**: (b) — the physical `∫d³v` is unambiguous (L23 Eq. 8.4.1) and
  the flat convention mis-weights the velocity moments (the density/polarization and
  the `Δ` outputs depend on it). This is a **foundational moment-machinery
  clearance** (a small module: the physical `velocity_moment!`/`weighted_moment!`
  measure + the `IinvB` singular-pitch split), not a single coefficient, and it
  **revises signed-off QN work** — hence a human decision, not a guess.
- **Gated work**: the parallel-flow weight `W` (Q3) and hence `J̄_∥`/the `Δ`
  outputs; the momentum-restoring term F (Q5); the physical fidelity of the cleared
  QN `δn̄_i`. Nothing entered `src/`.

## Q7 — The Δ_neo output moment is not resolution-convergent (outer-tail-dominated) — OPEN

- **Context**: With the matrix-free resolved solve now working past the dense cap
  (`PlaneJacobi` + `newton_krylov` + `natural_continuation` + the resolution
  protocol; all converge cleanly on `N` up to ~20k at physical `ρ̂_θi=0.05`), the
  first resolved `Δ_neo(w)` checkpoint was attempted and **`Δ_neo` does not
  resolution-converge**. At `w=1`, sweeping island resolution `K=8→12→16` (nodes
  across the half-width) gives `Δ_neo = 1.40 → 1.74 → 2.20` — a monotone power-law
  growth, not convergence. Diagnostics (per-x integrand `m1(x)=∮dξ J̄_∥cosξ`,
  radial-band breakdown, cutoff-restricted moments) show the cause is **not** the
  solver and **not** a rational-surface singularity:
  - the physical response `m1(x)` **decays outward** (RMS falls ~30× from the
    island to the edge), i.e. it localizes;
  - but the moment `Δ_neo = C∫dx∮dξ J̄_∥cosξ` is **dominated by the outer region**
    `|x|≫w`, where the center-clustered grid carries **large Simpson weights** on
    the small (and sign-oscillating) tail — a quadrature artifact that **grows with
    resolution** as the outer region is starved of nodes;
  - restricting the integral to the island (`|x| < 1.5–4w`) does **not** rescue it:
    those values are small and **trend toward ~0** as `K` increases (all ≲0.1 at
    K=16), while the full value keeps growing — so essentially **all** of the
    reported `Δ_neo` is the spurious outer tail, and the genuine island-region
    cos-moment is small and resolution-noisy.
  - Growing the box (`Lx/w = 6→12→20`) does not help: it further starves the outer
    region (the solve stops converging at `Lx/w=12` for fixed central `K`).
- **Question**: What is the intended `Δ_neo` extraction, and how should it be
  quadratured?
  - (i) a **volume moment** `C∫dx∮dξ J̄_∥cosξ` — if so, the outer region must be
    resolved for the quadrature (the center-clustered grid cannot be, at fixed
    budget) and/or the integral restricted to where the response lives, and the far
    field must genuinely decay; **or**
  - (ii) a **matched-asymptotic jump** (`Δ'`-style: the jump in the outer solution's
    logarithmic derivative across the layer), for which a naive volume integral is
    the wrong operator; **or**
  - (iii) is the **small island-region value the physical `Δ_neo`** (the outer tail
    is pure quadrature artifact to be removed), with the `∝1/w` B2 target then
    applying to a specific channel (e.g. `Δ_bs`) rather than the raw moment?
- **Options**: (a) keep the volume moment but fix the quadrature/grid (resolve the
  outer region + a principled island restriction + a decaying far field); (b)
  reformulate the Δ extraction as a matched-asymptotic jump; (c) accept the small
  island value as physical and re-target the B2 scaling onto the correct channel.
- **Recommendation**: do **not** guess — the extraction definition and its
  normalization are a physics decision on `docs/01 §4` / `numerics.md §7`. My best
  guess is (a) as the immediate step to get a resolution-stable number (a proper
  outer quadrature + island restriction), but the physical meaning must be checked
  against the tearing-`Δ` definition (matched asymptotics) before any `Δ_neo(w)`
  trend is trusted. This is a moment/output change → **physics-verifier** before any
  implementation, per the module policy.
- **Gated work**: the resolved `Δ_neo(w)` checkpoint; the B2 `Δ_neo ∝ 1/w` gate;
  the `channel_decomposition` rework (it sits on top of `Δ_neo`); the B5b toggle
  differential. The solver stack (PlaneJacobi / continuation / resolution protocol)
  is **not** blocked — it is done and green. Nothing was changed in `src/` for this;
  it is a definition/normalization decision.

### Update (2026-07-16) — sharpened by the York cross-check + the integration experiment

The question is now **narrower and concrete**. Two of the three original options are
effectively resolved by evidence:

- **York source check (papers; codes not public)**: Diss19 Eq. 4.12 defines `Δ_neo`
  as **exactly our volume moment**, balanced `Δ₀+Δneo=0` (`Δ₀` = the outer jump), and
  it **converges** for them (Figs 4.13–4.15 → bootstrap ∝1/w). ⇒ **Option (ii)
  [reformulate as a matched jump] is OFF the table** — the jump is `Δ₀`, and the
  volume moment is the correct, York-faithful `Δneo`. **Option (iii) [value is just
  small] is unlikely** — York gets finite O(1) plottable `Δneo(w)`.
- **Integration experiment** isolated two independent causes:
  1. **Quadrature rule** — Simpson on the center-clustered grid over-weights the
     coarse far-field nodes (Simpson vs accurate spline differ 3×, sign-flip on
     clustered grids). **RESOLVED**: `Moments.delta_moments` now uses cubic-spline
     quadrature (physics-verifier PASS, physics-neutral). This was necessary but **not
     sufficient**.
  2. **Far-field BC** — on converged, outer-resolved grids the spline `Δneo` still
     diverges with resolution (−1.05 → −2.84) and the `m1(x)=∮J̄cosξ` peak migrates
     from the island to the domain edge: our Dirichlet `g_far = x·L̂_{n0}⁻¹[1+(E−3/2)η_i]`
     (∝x) over-constrains the boundary, the interior does not self-match it, and a
     resolution-sharpening **boundary layer** forms that contaminates the whole solve.

- **The remaining decision (the real Q7)**: how to impose the Level-0 **far-field /
  outer boundary** so the perturbed response **localizes** (the resonant current tail
  decays), without the spurious "winged" branch our design rejected bare Neumann for
  (doc 01 §3; L23 §5.3/§7.1). Concretely, the design chose `g_far ∝ x` as "the
  analytic far-field," but the experiment shows that specific form does not let the
  solution reach its own asymptote. Options:
  - **(A) Analytic far-field, corrected** — implement the *actual* asymptotic
    interior solution as the boundary state (what L23 §7.1 "analytic far-field BC"
    future-work intends), so the interior self-matches and no boundary layer forms.
    Requires the analytic large-`|x|` form of `ĝ` (the neoclassical + drift-island
    asymptote), a derivation/sign-off item.
  - **(B) York-style localized BC** — `∂ĝ/∂p=0` (ĝ→const), matching kokuchou, with an
    explicit remedy for the winged-branch non-uniqueness (e.g. an anchored constant /
    a solvability constraint) so it stays single-valued.
  - **(C) Decomposition / domain mitigation** — solve for and integrate the *localized*
    perturbed current (subtract the ∝x background analytically before the moment) and/or
    keep a modest domain so the boundary layer is pushed out of the physical region.
- **Recommendation**: (A) is the most physically faithful and matches L23's own
  recommendation; (B) is the direct York analogue but re-imports the pathology our
  design avoided; (C) is a lighter mitigation that may suffice if the ∝x part
  subtracts cleanly. This is a **modeling decision on signed-off physics (`g_far`)** →
  human sign-off + physics-verifier before implementation; do not guess. The extraction
  form and the quadrature are **not** in question anymore.
- **Gated work (unchanged)**: the resolution-convergent `Δ_neo(w)` checkpoint; B2
  (∝1/w); the `channel_decomposition` rework; B5b. The solver stack, the resolution
  protocol, the extraction form, and now the moment quadrature are all done/green.

### Update (2026-07-18) — the far-field was ruled out; the real blocker is the physical-SOLVE non-convergence

The far-field investigation this Q7 triggered is **complete, and it was the wrong
lever** (full detail: LOG 2026-07-17/18). A 3-mode far-field toggle now exists
(`:dirichlet` / `:neumann` / `:analytic`):

- **B (`:neumann`, York-localized slope BC)** — fails: an additive-constant null
  mode (the "winged branch"), the solve does not converge.
- **A (`:analytic`, drift-orbit-shifted `g_far = (x−⟨x_D⟩_θ)·slope`)** — derived from
  **already-cleared** quantities (`derivations/analytic-far-field.md`,
  `[DERIVED: 2026-07-17]`), **physics-verifier PASS**, well-posed (no null mode) and
  more accurate than `:dirichlet`. **But it does NOT fix the `Δ_neo` convergence** —
  the orbit-shift offset is ~1% of the far-field, too small to matter.

**What actually blocks a resolution-convergent `Δ_neo` is the physical Level-0 solve
itself not converging** (below `resmax ~ 10⁻³`), NOT the extraction or the far field.
Ruled out this session (LOG 2026-07-18): the far-field BC, the `y_c` matching block
(A8: it is 43–1230× better-conditioned than the near-null direction), a genuine
null mode / near-singularity (`cond ~ 10⁶–10⁷` at `u=0`), and initial-guess /
globalization. Confirmed fine: the response localizes (exact `newton_direct`, m1
peak @ island) and the quadrature/moment (the spline `delta_moments` fix landed).
Remaining, unconfirmed suspicion: the strong **ExB nonlinearity** overwhelming the
Armijo line search (Newton crawls), or a discrete-**consistency** question.

**So the standing open items are now TWO, and neither is "which far-field is right for
the moment":**

1. **(New, the actual blocker) Physical-solve robustness** — make the Level-0 solve
   converge at physical parameters. Candidate directions (LOG 2026-07-18): a
   solver-robustness effort (continuation in the ExB coupling / trust-region or
   pseudo-transient Newton vs the current Armijo line search); and/or a **ground-truth
   comparison with York's actual numerics** (kokuchou/DK-NTM — what residual
   tolerance + method did they reach, on their small 2–3w domain?) to establish
   whether the discretized problem converges *for anyone*. **Recommended first: the
   York ground-truth reference** — without it, solver work is guessing.
2. **(Standing) A's `[DERIVED]` sign-off** — the orbit-averaged `⟨x_D⟩_θ` far-field
   (`analytic-far-field.md`) still needs human sign-off (a correct improvement + the
   paper's "our vs York" toggle), independent of the convergence blocker.

### Update (2026-07-19) — York ground-truth read: the FIELD converges for nobody; the OUTPUT `Δ` is what York gates on. New methodology decision.

Read L23 (Leigh 2023) §3.1.5/§3.1.6/§6.1.1/§6.2 and Diss19 (Dudkovskaia 2019)
Ch. III/IV first-hand this session (full synthesis with page cites:
`notes/york-convergence-ground-truth.md`). The result reframes blocker item 1:

- **kokuchou (L23), the direct 4D `{p,ξ}` solve — our closest analogue — never
  converges its self-consistent field.** It uses Picard iteration with a stated
  tolerance of **ε¹ ≈ 10%** (justified by the `O(ε^{3/2})` accuracy of the
  drift-kinetic equation). L23 §6.1.1 (p. 118): that 10% criterion **"did not come
  to within (ε = 10%) relative error after the maximum of 4 iterations for any of
  the runs"**; the `Φ̂` iterative residual is **> 100%/iteration** across the whole
  physical E×B regime (`ŵ ≳ 10⁻³ r_s`). **But the OUTPUT `Δ_loc` converges stably**
  (§6.2, Fig. 6.3) *despite* the non-converging field. The physical cause is the
  `O(ρ̂_θi)` drift-island separatrix layer that (a) sits at an `x` shifted by
  `ρ̂_θi ω̂_D(y,σ,u)` *varying across velocity space* and (b) *moves with `Φ̂`
  between iterations* — a rectilinear single-location mesh cannot track it
  (`[CHECKED: L23 §3.1.6, Eqs. 6.1.1–6.1.2]`, already in design 04 §2/§5).
- **RDK-NTM (Diss19)** reports converged `Δ` only because it (i) works in the
  `S`-streamline coordinate (analytic matched layers that *follow* the drift
  island; valid only at low `ν_★`) and (ii) reports many headline `Δ`
  (e.g. bootstrap `∝ 1/w`) at the **"0th iteration in Φ" (Φ = 0, E×B off)**.

**So our resmax ~ 10⁻³ field-residual floor is NOT a bug** — it is ~100× *below*
what kokuchou achieved and ~1000× below York's own 10% criterion, and it reflects
the documented, universal, physics-rooted moving-E×B-layer non-convergence. We have
been gating on the wrong quantity at an unphysically tight tolerance.

- **Question (new, methodology/threshold — do not decide silently)**: should
  Islands adopt York's **output-convergence** posture as its definition of done —
  i.e. gate on the **stability of the output moment `Δ_neo` (and `⟨J̄_∥ cosξ⟩`)
  across resolution / continuation** at the `O(ε^{3/2})`≈few-% level, rather than on
  the global field residual `resmax` at `10⁻³`? This is a change to the convergence
  gate *and* its tolerance, both of which the no-guess discipline reserves for human
  sign-off.
- **Options**: (a) adopt output-convergence (gate on `Δ` stability; report `resmax`
  as a diagnostic, not a gate) — matches York and the `O(ε^{3/2})` physics;
  (b) keep the `resmax` gate but relax the tolerance to the physically-justified
  few-% level; (c) keep the strict `resmax ~ 10⁻³` gate and invest in making the
  field itself converge (drift-island-separatrix mesh packing + E×B continuation)
  — more faithful/ambitious than York, but no source demonstrates it is achievable.
- **Recommendation**: (a) as the definition of done, **with** the (c) numerics
  (E×B-coupling continuation from `Φ = 0`; grid packed at the drift-island
  separatrix per design 04 §1) pursued to make the *output* robust — because the
  ground truth names exactly those two levers. Report `resmax` always, gate on `Δ`.
  Do **not** weaken the science to reach "done"; adopt the field's own,
  physics-justified success metric.
- **Gated work**: the resolution/continuation-convergent `Δ_neo(w)` checkpoint; B2
  (`∝ 1/w`); B5b. **Immediate unblocked next step (no sign-off needed to *measure*)**:
  re-take `Δ_neo` at physical `ŵ ~ ρ̂_θi` with the fixed spline `delta_moments` and
  check whether it (and the current moment) *stabilises* across resolution even as
  `resmax` floors — the kokuchou `Δ_loc` test. Only the *gate change* needs sign-off.

### Update (2026-07-19, measured) — the kokuchou test: output IS field-residual-insensitive (gate confirmed wrong) BUT does NOT resolution-converge → the blocker is the extraction/grid, not the solve

Ran the kokuchou `Δ_loc` test (LOG 2026-07-19 cont.; physical `ρ̂_θi=0.05`, `w=0.05`):

- **The gate question is answered by evidence**: at fixed resolution `Δ_neo` is
  **insensitive to the field residual** (`−434` at `resmax=1e-5` vs `−441` at
  `resmax=0.15`). So **option (a) [gate on the output, not `resmax`] is empirically
  the right posture** — the recommendation stands and is now measured, not just
  argued. (Still a threshold/methodology sign-off item.)
- **But a NEW finding supersedes the "which solver" framing**: `Δ_neo` does **not
  resolution-converge** — the exact solve gives `−435 → −38 → −9.3` across x-res with
  the moment peak migrating off the magnetic-island centre. And crucially the **exact
  solve converges the FIELD fine** (`resmax=1e-5`) — so the 2026-07-18 "physical-solve
  non-convergence" blocker was largely a **matrix-free-preconditioner** artifact, not
  the real problem. **The real blocker is the output EXTRACTION / non-localisation**
  (this is Q7-*original*, 2026-07-16, resurfacing) + the **grid**: the response layer
  sits at the drift island (shifted `ρ̂_θi ω̂_D(y,σ,u)`, velocity-spread), not the
  magnetic island our rectilinear x-mesh packs (design 04 §1; L23 §3.1.6).
- **New gated decision (a build, needs a steer)**: implement the **drift-island-
  separatrix grid map** `x(s; y,v̂,σ-envelope)` (design 04 §1 / L23 §7.1.1 `p̃`; a GRID
  MAP only — D1 stands) so the moment's velocity-spread support is resolved, THEN
  re-test `Δ_neo` resolution-convergence. This supersedes solver-robustness work as
  the precondition for a trustworthy `Δ_neo(w)`.

## Q8 — Clear the drift-island shift structure `x_D = ρ̂_θi ω̂_D L̂_q` (docs/01 §2.2, `[CHECKED]` → sign-off) — OPEN

- **Context (2026-07-20)**: the drift-island **band grid** (`04 §1`; `LOG 2026-07-20`)
  sizes its uniform central region from the drift-island shift envelope
  `R = max_passing |x_D^island|`, `x_D^island = ρ̂_θi ω̂_D L̂_q`. The `ω̂_D`
  *coefficient* is `[CLEARED]` (§2.1, signed off 2026-07-11), but the **shift
  structure** `x_D = ρ̂_θi ω̂_D L̂_q` (docs/01 §2.2, line 180) is only
  `[CHECKED: I19 Eq. (33); D21 Eq. 21; Diss19 Eq. 2.37]` — machine-checked + cited,
  **not** human-signed-off. The physics-verifier flagged an initial mislabel of this
  as `[CLEARED]` (now corrected to `[CHECKED]` in `Configure.drift_island_shift_envelope`,
  `numerics.md`, `analytic-far-field.md`). The I19-Eq.-33 lineage is exactly the
  erratum-prone set CLAUDE.md rule 6 / L23 §2.6 warns about, so the tag grade is
  load-bearing.
- **Question**: is the drift-island shift structure `x_D = ρ̂_θi ω̂_D L̂_q` (the `L̂_q`
  form, passing particles, `S = (ŵ²/4L̂_q)[2(p̂−ρ̂_θi ω̂_D L̂_q)²/ŵ² − cosξ]Θ(y_c−y)`)
  correct as printed, or does it carry an L23-§2.6-class erratum? Clear it (record
  paper + Eq. in docs/01 §2.2), or flag the discrepancy.
- **Mitigation already in place (why this does not gate the grid build)**: the
  `[CHECKED]` shift enters **only to size the grid** — the band spans `[-(R+w), R+w]`
  with a `+w` margin and geometric far tails, so an O(1) error in the shift form
  mis-sizes the fine region but still resolves a wide neighbourhood with the far field
  beyond it. It is **never** baked into a physics output. The grid map's *correctness*
  gate is empirical: the `Δ_neo` resolution-convergence test on the band grid (the
  payoff test), not the shift value.
- **Options**: (a) clear via first-hand PDF re-derivation (D7 route), like the M2b
  lane; (b) clear by transcription check against I19 Eq. 33 / D21 Eq. 21 / Diss19
  Eq. 2.37 (weaker — rule 6 cautions against for this lineage).
- **Recommendation**: (a). Until then the band grid stands on the `[CHECKED]` tag,
  validated empirically by `Δ_neo` convergence; no output depends on the shift value.
- **Gated work**: promoting the drift-island grid map from "empirically validated" to
  "physics-cleared"; any *output* that would use `x_D^island` as a value (none today).
