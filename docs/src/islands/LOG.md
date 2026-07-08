# Islands — session LOG

Cross-session memory spine (design doc `06 §2.5`). Read this at session start
together with `QUESTIONS.md`. Append a short entry before every session end:
**what moved / what's blocked / next action**. Newest entries at the top.
Reference `QUESTIONS.md` IDs (`Q<n>`) and ladder IDs (`A1`, `B5a`, …) where
relevant.

---

## 2026-07-08 — Phase A bootstrap (supervised)

- **Moved**: Created the `Islands` submodule skeleton (`src/Islands/Islands.jl`,
  empty `module Islands`) and wired it into `src/GeneralizedPerturbedEquilibrium.jl`
  (`include` + `import . as` + `export`, last submodule slot before `Rerun.jl`).
  Stood up this `LOG.md` and `QUESTIONS.md`, the `.claude` unattended-run
  guardrails, the `physics-verifier` subagent, and the M1 launch prompt.
- **Blocked**: `julia` is not on the automation shell's PATH (no module, not in
  `$HOME`) → local `using GeneralizedPerturbedEquilibrium` + test run could not
  be executed here; relying on CI (test.yaml) to validate the wiring on push.
  See **Q1** — the overnight loop's scratch-clone environment must expose
  `julia` or it cannot run tests.
- **Next**: land Phase A on `feature/islands`, confirm CI green (tests + docs),
  then the first overnight run works milestone **M1** (design `00 §M1`) —
  phase-space grids + operator-stack skeleton + MMS/AD harness (ladder A1, A2),
  no `[VERIFY]` physics coefficients.
