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

## Q1 — Julia not on the automation shell PATH — OPEN

- **Context**: Phase A bootstrap. Verifying the `Islands` module skeleton loads
  (`using GeneralizedPerturbedEquilibrium`) and running the test suite requires
  `julia`, but it is not on the non-interactive shell's PATH (no `julia` module,
  none in `$HOME`; other users have installs under `/mnt/homes*/…/julia*`).
- **Question**: What is the canonical `julia` invocation for automation on this
  cluster, and will the overnight loop's scratch-clone environment expose it?
  The unattended loop **cannot run `julia --project` tests without it** — this is
  a hard prerequisite for Phase B, not just a local convenience.
- **Options**: (a) a `module load <julia>` line prepended in the loop's shell
  rc / launch script; (b) an absolute path to a `juliaup`/`julia` binary the
  user owns; (c) install juliaup under `ncl2128` and pin 1.11.
- **Recommendation**: user provides the module/path; bake it into the tmux
  launch script (and the Stop hook, which runs the fast test subset). Until
  then, CI (`test.yaml`) is the only validation of Julia changes.
- **Gated work**: local verification of every Julia change; the Phase-B overnight
  loop's ability to run tests / meet its definition-of-done.
