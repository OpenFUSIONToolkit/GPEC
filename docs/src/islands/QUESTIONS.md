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

## Q2 — Ratify Decisions D7 and D8 — OPEN

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

## Q3 — Clear the Level-0 coefficient set ([CHECKED] → human sign-off) — OPEN

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
    `ν_★ = 10⁻³`. Resolve before pinning the B5a tolerance.
  - Acquire **WCHH96** (analytic electron closure / large-w limits, B2) and **Park
    PoP 29 (2022)** (SLAYER Q-convention map, D1) into the `docs/08` reference
    library — both are currently cited only via transcription.
- **Options**: (a) user resolves from the source PDFs; (b) an independent
  re-derivation pins `ψ̃` and the collisionality is read from the reproduced run.
- **Recommendation**: resolve `ψ̃` by re-derivation (it is a clean dimensional check);
  read the B5a collisionality from whichever run the D8 triangle pins as canonical.
- **Gated work**: the `Δ_cos/Δ_sin` normalization; the B5a threshold tolerance; the
  B2 large-w and D1 SLAYER comparisons.
