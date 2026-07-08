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
