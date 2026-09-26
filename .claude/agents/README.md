# Agent Team

This repo ships a small team of specialized Claude Code subagents in `.claude/agents/`. They are **stateless reviewers**: each runs in its own context, is handed a specific deliverable, and returns its findings to the main session — they do not talk to each other. **The main session is the integrator.** You drive them; you do not delegate the whole task and walk away.

**Invoke an agent by name** for the matching job — e.g. *"review this with the fortran-physics-reviewer"* or *"run the regression-guardian against develop"*. Do **not** "consult all the agents" reflexively: that burns the token budget and produces noise. Pick the agent whose job matches the change.

## Roster

| Agent | Model | Role — invoke when… |
|---|---|---|
| `fortran-physics-reviewer` | opus | A physics kernel, numerical method, derivative, integral, or quadrature was written/changed. Audits fidelity to the reference papers and the Fortran GPEC source. Carries project memory (correspondence map + per-domain audit checklists for KineticForces and InnerLayer), so its reviews compound — let it record findings. |
| `clean-code-reviewer` | opus | A logical chunk of code is ready and you want readability/maintainability review for a fusion physicist audience (naming, magic numbers, docstrings, structure). |
| `julia-performance-optimizer` | opus | A specific function/hotspot is slow or perf-sensitive (type stability, allocations, hot-loop work in ODE/kinetic/resistive-layer paths). |
| `fast-interpolations-optimizer` | opus | Code uses FastInterpolations.jl and you want allocation-free / optimal-search-type review. |
| `regression-guardian` | sonnet | **Before merging any substantive change** (mandatory per the Regression Harness policy — see `docs/development/regression-harness.md`), or when you need to know whether a tracked numerical quantity moved. Runs the harness, reports the table, flags non-OK rows, proposes new cases. |

## Recommended review pipeline for a substantive change

Run sequentially, reading each agent's findings before launching the next:

1. **`fortran-physics-reviewer`** — physics fidelity first; a fast-but-wrong result is worthless.
2. **`clean-code-reviewer`** — readability and maintainability.
3. **`julia-performance-optimizer`** and/or **`fast-interpolations-optimizer`** — only if the change is performance-relevant.
4. **`regression-guardian`** — always, last, before merge. Confirms the numbers didn't silently move.

**These agents are diff-scoped, and that leaves one thing uncovered.** Each reviews what changed, so a
hot path nobody is editing is invisible to all of them — permanently, no matter how expensive it
becomes. `julia-performance-optimizer` in particular is handed a named function and told not to
profile the suite, so it cannot find a hotspot outside the diff and should not be described as though
it had. Whole-program cost is answered only by profiling a full run, which is a developer-initiated
check rather than anything the pipeline does on its own. Two moments are worth spending it on: when a
module first lands in a hot path, and when someone's sense is that runtime has changed a lot. When
asked to do performance work, do not assume the diff is the scope — say what the whole-run cost
picture is, or say that it is unmeasured.

Not every change needs all four. A docs-only change needs none; a pure perf refactor still needs the physics reviewer (to confirm no numerical change) and the regression-guardian.

## Budget

Every consultation is bounded — see **Subagent Consultations** in `/CLAUDE.md` (≤30 tool uses, ≤10 min, one concrete deliverable, never re-launch a runaway). The agent bodies now self-enforce this, but state the budget in your prompt anyway and always hand the agent the specific file paths to act on.
