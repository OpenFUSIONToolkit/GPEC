# Plan — B1: no-island neoclassical bootstrap & flows vs Sauter/NEO

**Purpose.** The first *physics* rung of the docs/05 ladder and the "reproduce the
basics before the new physics" step-back: show that the Islands neoclassical machinery
(collision operator, orbit-averages, electron closure, flow moments) reproduces the
**standard neoclassical bootstrap current `J_bs` and parallel flows** in the *no-island*
limit, matching **Sauter et al. PoP 6, 2834 (1999)** across collisionality `ν_★`
(banana → plateau → collisional), single- and multi-species. This validates the
neoclassical core **independently of the fragile driven-island solve** (the A1/A5
diagnostic proved that solve's system is consistent but its convergence is fragile; the
no-island limit should be linear and well-conditioned, sidestepping it).

Read with `docs/05 §B1`, `QUESTIONS.md` Q3, `docs/01 §2.3–2.5`, and the A1/A5 LOG entries
(2026-07-25) that motivate B1.

## Definition of done
1. Islands reproduces the neoclassical **bootstrap coefficient `L31`** (density-gradient
   response) and the **full `J_bs(ν_★)`** vs Sauter across banana/plateau/collisional, to
   **stated per-regime tolerances** (banana tightest — Sauter is most accurate there),
   single-species.
2. The **trapped-fraction dependence** (`L31` vs `f_t`, scanning `ε`) matches Sauter.
3. **Multi-species** (`Z_eff`/impurity) bootstrap modification matches Sauter multi-species
   (Islands is multi-species-first, D3).
4. Captured durably: a **regression case** (`J_bs`, `L31` at a few `ν_★`), a **Physics
   Book B1 section**, and a **regenerable figure** (`J_bs` vs `ν_★` with Sauter overlay).
5. An **executable cross-check against the TokaMaker (Redl) bootstrap** on the same
   `(f_t, ν_★, Z)` inputs — a code-to-code diff, not just a paper comparison.

## Guardrails
- **Ground truth is Sauter's analytic formulas** (a validated fit to accurate
  calculations, good to a few % across regimes); NEO is a bonus cross-check, not required.
- **[VERIFY] discipline** even though Sauter is standard: transcribe `L31/L32/L34/σ_neo`,
  `f_t`, and the `ν_★` definition with `[CHECKED: Sauter 1999, Eq. …]` and machine-check;
  never guess a coefficient/exponent. Physics-verifier before committing any
  physics-adjacent change. Doc-first. LOG + push each session.
- **Reuse** the existing collision operator, orbit-average brackets (B2a-fixed),
  `passing_fraction`, `parallel_current!`/`weighted_moment!`; do not reinvent neoclassical
  machinery. Grep first.
- **Environment/launch** (learned this month): `env -u LD_LIBRARY_PATH julia --project=.`;
  launch long runs via the **Bash tool's `run_in_background`** (manual nohup/setsid failed
  with exit 144); never `pkill` a pattern matching the scratch script name; right-size
  (dense LA only at `N ≲ 6000`); check node load first.

## Known state / what exists (from the 2026-07-31 inventory)
- **No in-repo bootstrap/Sauter/NCLASS** anywhere in GPEC (Equilibrium, ForceFreeStates,
  PerturbedEquilibrium, KineticForces). ⇒ Sauter analytic is the reference; there is no
  code cross-check to lean on.
- **Sauter (1999) is NOT in the reference library** (docs/08 lists it "to acquire"). This
  is a **prerequisite** — see Phase 0.
- **Q3 coefficient set largely cleared** via M2b re-derivations (`ω̂_D`, collision kernel +
  `⟨ν̂_ii⟩`, electron closure `k=−1.173` / `f_p=1−1.46√ε`, quasineutrality) — the
  ingredients are signed off. Confirm each B1 needs is `[CLEARED]`, not `[CHECKED]`.
- **The bootstrap is dominantly the ELECTRON flow, which Islands treats ANALYTICALLY**
  (L23 §2.5, Eqs. 2.5.5–2.5.8: `k_neo`, `f_p`, `∂h/∂x`, `η`). ⇒ a large part of B1 needs
  **no drift-kinetic solve at all** — the cheapest, first target.
- Existing flow/current machinery: `Moments.parallel_current!`, `channel_split`
  (`J_bs = ⟨J̄_∥⟩_Ω`), `Coefficients.passing_fraction`. These are island-perturbation
  oriented; B1 needs the **equilibrium/no-island** neoclassical response (the Q3
  "neoclassical no-island" solution referenced in `Operators.jl:650,659`).

## Phase 0 — References, `ν_★` convention, scope (acquisition RESOLVED via TokaMaker)
- **Reference = OpenFUSIONToolkit / TokaMaker bootstrap code (Sauter + Redl).** The user
  (2026-07-31) pointed to the open-source **TokaMaker** (same GitHub org as this repo,
  `OpenFUSIONToolkit`), which implements the bootstrap current including the **Redl
  formula** update to Sauter — **Redl, Angioni, Belli, Sauter et al., "A new set of
  analytical formulae for the bootstrap current and neoclassical conductivity", PoP 28,
  022502 (2021)**. Redl (2021) *is* the modern, corrected Sauter model (fixes the known
  Sauter-Angioni issues) — **use Redl as the primary ground truth**, Sauter as the
  historical cross-reference. NEO/NCLASS: **dropped** (user has neither).
- **0.1** Get the exact formulas **from the TokaMaker source** (open source — no PDF
  blocker): the `L31/L32/L34`, `α`, neoclassical conductivity `σ_neo`, trapped fraction
  `f_t`, and the collisionality definitions as coded. Transcribe into a new
  `derivations/redl-sauter-bootstrap-reference.md` with `[CHECKED: Redl 2021 Eq. N /
  TokaMaker <file>:<lines>]`; machine-check against the Redl (2021) paper if obtainable.
  Because it is *code*, we get an executable reference (port the formulas to Julia and
  diff against TokaMaker's outputs where feasible).
- **0.2 `ν_★` convention alignment (CRITICAL).** Redl/TokaMaker `ν_★e`/`ν_★i` vs the
  Islands `ν_★ = ν_jj Rq/(ε^{3/2} v_th)` (L23 Eq. 2.3.40) may differ by O(1) factors and
  species/√2 conventions. Pin the exact map **before** any `ν_★`-axis comparison — a
  mismatch here silently ruins the benchmark. Document it in the reference doc.
- **0.3 Pin the primary comparable:** `L31` and `J_bs(ν_★)` (density-gradient bootstrap)
  single-species first; `L32`/`L34`/`α` and multi-species as extensions.

## Phase 1 — Analytic neoclassical constants (cheapest; foundational; likely near-green)
- **1.1** Confirm `f_p(ε)=1−1.46√ε` / `f_t`, and `k_neo=−1.173` (Hirshman–Sigmar) against
  Sauter/HS — these feed Sauter's formula. Likely already tested (A7 identities); confirm
  green and cite.
- **1.2** Confirm the `ν_★` normalization end-to-end (0.3) with a numeric spot-check at one
  `(ε, T, n, B)`.

## Phase 2 — Analytic electron bootstrap (NO SOLVE) vs Sauter `L31` — the cheap win
- **2.1** In the no-island limit the flattened-electron closure gives an **analytic**
  parallel electron flow (L23 Eqs. 2.5.5–2.5.8 with `h→x`, `∂h/∂x→1`, `ω_E=0`). Assemble
  the bootstrap electron current `J_bs,e = ⟨J_∥,e⟩` from it — pure algebra over the cleared
  `f_p`, `k_neo`, `η`. **No drift-kinetic solve.**
- **2.2** Compare `J_bs,e` / the implied `L31` vs Sauter across `f_t` and `ν_★`. This
  validates the **electron closure** (the dominant bootstrap piece) at near-zero cost. A
  mismatch here localizes a closure/convention bug before any solver work.
- **Gate 2:** electron `L31` matches Sauter in the banana limit to the stated tolerance ⇒
  proceed to the ion solve; else fix the closure/convention first (physics-verifier).

## Phase 3 — The no-island **ion** neoclassical solve (the core solver work)
- **3.1** Define the no-island neoclassical ion problem: the drift-kinetic equation with
  only the `∇p`/`∇T` drive, **no island** (`Ω`-independent / `w→0`), the cleared collision
  operator (Lorentz + momentum-restoring) and parallel streaming. This is the standard
  Hinton–Hazeltine/Sauter neoclassical problem — **linear, no E×B self-consistency
  nonlinearity**.
- **3.2 Realization decision:** (a) reduce the Level-0 machinery (drop island geometry, set
  the drive to the neoclassical gradient, keep collision+streaming+orbit-averages), or
  (b) a dedicated minimal neoclassical solver reusing the same coefficient builders.
  **Recommend (a)** if a clean `w→0`/no-island config exists; else (b). Either way reuse
  `Coefficients.*` and the collision operator.
- **3.3 Verify it is WELL-CONDITIONED and converges cleanly across `ν_★`** — this is the
  key premise (it should sidestep the driven-island fragility). If it does *not* converge
  cleanly, STOP and surface it: it would mean the collision/streaming core (not the island
  drive) has a conditioning problem — a more fundamental finding.
- **3.4** Compute the ion neoclassical parallel flow `u_∥i` and its bootstrap contribution;
  cross-check the ion flow far-field against L23 Eq. 5.2.1 (`O(ε^{3/2})` analytic result) —
  an internal consistency check independent of Sauter.

## Phase 4 — Full `J_bs(ν_★)` vs Sauter (single species) — the B1 headline
- **4.1** Assemble `J_bs = ⟨J_∥,e + J_∥,i⟩` and scan `ν_★` across banana → plateau →
  collisional; overlay Sauter. Per-regime tolerances (state them; banana tightest).
- **4.2** Scan `ε` → `L31(f_t)` vs Sauter. `L32`/`L34`/`α` if in scope.

## Phase 5 — Multi-species
- **5.1** `Z_eff`/impurity bootstrap modification vs Sauter multi-species (+ NEO if
  available). Exercises the species-list machinery (D3). Trace-species via the linear
  post-pass (docs/02 §1.2).

## Phase 6 — Closeout
- Regression case (`regression-harness/`): `J_bs`, `L31` at a few `ν_★` (single + one
  multi-species). Physics Book **B1 section** (`docs/src/islands/`) with the as-implemented
  bootstrap expressions + anchors. Regenerable **figure** (`benchmarks/islands/figures/`):
  `J_bs`/`L31` vs `ν_★` with Sauter overlay. `regression-guardian`. LOG + push + PR.

## Risks / decisions to flag to the user
- ~~Sauter PDF acquisition~~ **RESOLVED** — reference is the open-source TokaMaker (Redl)
  bootstrap code (user, 2026-07-31); get formulas from source, no PDF needed.
- ~~NEO availability~~ **DROPPED** — user has neither NEO nor NCLASS; TokaMaker/Redl is the
  reference.
- **`ν_★` convention** (0.2) — now the single biggest silent-error risk; pin it explicitly
  against TokaMaker's coded definition.
- **Realization of the no-island solve** (3.2) — reduce Level-0 vs dedicated solver; a
  design choice worth a quick check of whether a clean `w→0` path exists.
- **Scope sizing:** the minimal, high-value first deliverable is **Phase 2 (analytic
  electron `L31` vs Sauter)** — it validates the dominant bootstrap piece with no solve and
  no new machinery. Recommend doing Phase 0–2 first, reporting, then deciding on Phase 3+.

## Progress checklist
- [x] 0.1 Redl formulas transcribed from TokaMaker `bootstrap.py:576-795` + Julia port (`redl-sauter-bootstrap-reference.md`)
- [x] 0.2 `ν_★` map pinned: Islands ion ν_★ ≈ 0.77·Redl ν_i★; ν_e★/ν_i★ ≈ 1.41 (lnΛ-match for precision)
- [x] 1.1 `f_T = 1.4624√ε` confirmed vs Sauter circular `1.46√ε` (3–4 digits); `k_HS≃−1.173` present in Fields
- [ ] 2.1–2.2 analytic electron `L31` — needs the York→L31 normalization `[DERIVED]` map (ref §10) + physics-verifier, THEN compare vs Redl `F31(f_T)` (Gate 2)
- [ ] 3.1–3.3 no-island ion neoclassical solve built + confirmed well-conditioned across `ν_★`
- [ ] 3.4 ion flow vs L23 Eq. 5.2.1 internal check
- [ ] 4.1–4.2 full `J_bs(ν_★)` + `L31(f_t)` vs Sauter (single species)
- [ ] 5.1 multi-species vs Sauter
- [ ] 6 regression case + Physics Book B1 + figure + guardian + PR
