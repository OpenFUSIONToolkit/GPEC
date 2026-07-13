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
  - **Collision magnitude** `nu_tilde`: the `⟨ν̂_ii⟩_u`/`ν_★` normalization scaling
    the cleared `ν_{jj}(v̂)` shape — the deferred sub-constant already tracked in
    Q3 (`⟨ν̂_ii⟩_u = (4ε^{3/2}ν_★/3√π)(√2−ln(1+√2))`, L23 Eq. 4.1.6). Still
    escalated (needs the L23 integrand read in detail).
  - **Passing fraction** `f_p ≃ 1 − 1.46√ε` (electron closure, Q3): **RESOLVED
    (2026-07-11)** — `derivations/passing-fraction.md` signed off; cleared as
    `Coefficients.passing_fraction(ε) = 1 − 1.4624√ε` (the effective
    trapped-fraction coefficient, = quoted `1.46` to 3 s.f.), authorizes
    `Fields.ElectronClosure.f_p`. The Hirshman–Sigmar `k ≃ −1.173` remains
    escalated (needs the parallel-viscosity moment problem).
  - **Orbit-averaged pitch measure** `B_profile`: the collision operator's `|B|`
    on the `y`-grid is the *orbit-averaged* field (turning-point structure), not a
    single local `B`; the cleared `pitch_diffusivity(λ,B)` is the local building
    block. Clear the orbit-averaged measure form.
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
