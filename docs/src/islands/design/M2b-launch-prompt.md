# M2b milestone contract (Islands) — Level-0 physics via the derivation lane

> The milestone contract for Islands M2b (design doc `06 §2.1`). Set it as the
> `/goal` completion condition when working M2b. Prerequisites are in place:
> M1 (PR #320) and M2 (PR #324) are merged — the full L0 solve *structure*
> exists with every physics coefficient gated — and the user has ratified
> **D7/D8** and chosen the **re-derivation-first** clearance mode (QUESTIONS
> Q2 resolved, Q3 mode pinned, 2026-07-08).

You are working milestone **M2b** of the Islands module (`src/Islands/`):
filling in the Level-0 physics that M2 deliberately gated, by the route the
project's own thesis demands — **independent re-derivation, human sign-off,
then implementation** — ending with the B-ladder physics benchmarks running.

## Read first (every session)

`src/Islands/CLAUDE.md` (the [VERIFY]/[DERIVED]/[CHECKED] policy — rules 3 and
4 are the spine of this milestone), `docs/src/islands/LOG.md` and
`QUESTIONS.md` (Q3 items and mode, Q4), then design docs
`docs/src/islands/design/{00-roadmap,01-physics-level0,03-architecture,04-numerics,05-verification}.md`
— **especially docs/05 "Target tiers and reproducibility" and Decision D9
(00): the physics gates are scalings/differentials, absolute numbers are
audit-gated** — and the as-implemented chapter `docs/src/islands/numerics.md`.
The source PDFs are in the `docs/08` reference library. The repo-root
`CLAUDE.md` governs
GPEC-wide conventions.

## Goal (set this as your `/goal` completion condition)

**M2b is done when all of the following hold:**

1. **The derivation set exists** in `docs/src/islands/derivations/`, one
   chapter per Q3 item, each marked `[DERIVED: date]` and structured as:
   stated starting point (the drift-kinetic equation and orderings O1–O9,
   cited to `01 §2`), explicit assumptions, the derivation with no skipped
   sign-bearing steps, a boxed final coefficient form, and a **cross-check
   table** against the `[CHECKED]` transcriptions (I19/Diss19/D21/L23,
   honoring the L23 §2.6 amendments). Items:
   - (a) the orbit-averaged drift frequency `ω̂_D` including the
     `:original`/`:improved` `L̂_B⁻¹` treatment;
   - (b) the pitch-angle collision kernel, the `ν_★` normalization, and the
     analytic `⟨ν̂_ii⟩_u` velocity average;
   - (c) the flattened-electron closure: the `h(Ω)` prefactor, the coupled
     flow relation, and the `k`/`f_p` constants;
   - (d) the quasineutrality closure coefficient;
   - (e) the `ψ̃` island amplitude — deriving it settles the Q4
     `q_s′/q_s` vs `q_s/q_s′` question by dimensional necessity;
   - (f) the `Δ_cos`/`Δ_sin` prefactors, including pinning the sin-moment
     normalization (a `[DERIVED]` pin per `01 §4`).
   **Every discrepancy against a transcription is flagged in the table and in
   `QUESTIONS.md` — never resolved silently** (policy rule 2).
2. **Human sign-off, item by item.** Present each derivation to the user (they
   are working interactively; ask). A signed-off derivation clears the
   corresponding coefficients: record the clearance in `docs/01` (source,
   equation, date, derivation link) per policy rule 3. **You never sign off
   your own derivation.**
3. **Implementation fill-in for signed-off items only**: a single Level-0
   coefficient/configuration builder in `src/Islands` (the named-configuration
   mechanism of `03 §2` — e.g. `:dkntm_original`, `:rdkntm_improved`) that
   populates the M2 gated parameters, each value annotated with its derivation
   anchor. Un-gate progressively: partial sign-off ⇒ partial fill-in; anything
   unsigned stays NaN/supplied.
4. **The input-completeness audit** (Decision D9, docs/05 "Target tiers"): a
   per-source *input manifest* (using the docs/05 template) for each threshold
   source — I19, D21, D23a/b, L23 — written into a new
   `docs/src/islands/design/09-input-manifests.md` (or a docs/05 appendix;
   executor's choice), each required input either cited to where the paper
   states it or recorded as "unspecified → assumption + sensitivity scan
   needed". This audit is what decides, per target, whether a T4 absolute
   comparison is even attemptable or the target stays T3. It is itself a
   Paper-I methods deliverable (OUTLINE claim C9). B5a's collisionality
   contradiction is the type specimen — resolve it *in the manifest*, not by
   picking a number.
5. **The B-ladder starts running — tiered (D9)**: for cleared configurations,
   un-skip the `benchmarks/islands/` scripts. The DoD is the **primary-tier
   gates running with archived artifacts**: the T2 drift-model **toggle
   differential** (:original → :improved w_c ratio, measured within Islands),
   and the T3 **scaling/existence/trend** sweeps (1/w and 1/w³, layer widths
   ∝ ν^{1/2}, ω_E² with a sign reversal existing, threshold existence at
   w_c ~ O(ρ_θi), dw_c/dν_★ > 0) with fitted exponents/signs. **T4 absolute
   comparisons are attempted only for manifests the audit completes, and
   reported only with the manifest + an input-sensitivity scan** (docs/05
   reporting rules 6–8) — never as bare pass/fail. Follow the docs/05 reporting
   rules (grid-convergence archived; half-widths with both ρ_θi and ρ_bi
   stated); triage disagreements per docs/05 rule 3 (now including
   "under-specified source configuration"). B5 absolute *agreement* is never
   this milestone's precondition.
6. Q4 source status: WCHH96 and Park 2022 are both in the library (resolved
   2026-07-09; docs/08). Nothing to acquire — proceed with the in-repo sources.
7. The full suite passes; the Physics Book chapter
   (`docs/src/islands/numerics.md` or a new physics chapter) is updated **in
   the same PR** for every equation that becomes as-implemented (docs/07
   policy), with figures regenerated via the pinned script where outputs
   change.
8. A PR is open onto `feature/islands`, and `physics-verifier` has passed on
   every physics-adjacent commit — its job here is checking **provenance**:
   derivations marked `[DERIVED]`, transcriptions never silently promoted,
   no unsigned coefficient in `src/`.

## The hard rules (non-negotiable, sharpened for this milestone)

- **Never present a derivation as a literature transcription or vice versa**
  (policy rule 4). The provenance tag is part of the result.
- **Never assign an unsigned coefficient in `src/`** — sign-off happens in the
  conversation and is recorded in `docs/01` before the value lands in code.
- **Never tune a derived coefficient to make a benchmark pass** (policy rule
  2). If a benchmark disagrees, the triage path is docs/05 rule 3, in the open.
- If a derivation stalls (a step you cannot justify), flag it in
  `QUESTIONS.md` with the specific step and move to the next item — a partial
  derivation set with honest gaps beats a complete one with a glossed step.

## Working discipline

- Derivation chapters are Documenter pages: one-sentence-per-line source, LaTeX
  `math` blocks, cross-check tables as Markdown tables; wire new pages into the
  Islands nav section in `docs/make.jl` and verify with a local docs build
  (`env -u LD_LIBRARY_PATH …/julia --project=. build_docs_local.jl`).
- Run every new kernel once under `--check-bounds=yes`; keep `apply!` paths
  allocation-free (regression-tested); julia via
  `env -u LD_LIBRARY_PATH /mnt/homes_global/ncl2128/software/julia-1.11.7/bin/julia --project=. …`.
- Commit granularly to a milestone sub-branch (`ISLANDS - <TAG> - …`,
  referencing Q3 items and ladder IDs); update the `islands_l0_structural`
  regression case if tracked numbers legitimately move (with justification —
  never re-baseline to reach green); append a `LOG.md` entry per session.
- The user is present: surface each completed derivation for sign-off rather
  than batching everything to the end.

## Definition of NOT done

Any coefficient in `src/` without a recorded human sign-off; a derivation
presented without its cross-check table; a discrepancy resolved silently; a
benchmark "passing" via tuned coefficients or weakened tolerance; a **T4
absolute comparison reported without its input manifest and sensitivity scan**
(Decision D9); the suite red; docs not updated with the as-implemented
equations; or the tree dirty. Conversely: a T4 absolute *disagreement* after
honest triage — including "the source is under-specified" — is a reportable
result, not a failure. **Do not chase absolute agreement**; the primary gates
are the T2 differentials and T3 scalings, and pursuing an absolute number past
what the derivations and input manifests support is itself out of bounds.
