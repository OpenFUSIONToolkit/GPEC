# 06 — Autonomy, tooling, and GPEC-repo integration

Audience: (a) the human setting up the environment once, (b) every Claude Code /
Fable session working this project. Sections marked **[HUMAN SETUP]** are
one-time actions; everything else is standing guidance for agent sessions.

---

## 1. Living inside the GPEC repo (supersedes docs/03 §5 where in-repo)

ISLET develops inside the OpenFUSIONToolkit GPEC repository as a **subdirectory
Julia package** (e.g. `islet/` with its own `Project.toml`, depending on GPEC):

- **The interfaces become function calls.** docs/03 §5 specified file-based
  interfaces to Δ′ and SLAYER because a standalone repo needed them. In-repo,
  everything ISLET consumes arrives with the Tearing module work (PR #238,
  `feature/tearing-growthrates`): SLAYER Δ(Q) (Fitzpatrick Riccati layer model,
  `src/Tearing/InnerLayer/SLAYER/`), GGJ under the same `InnerLayer` interface,
  the dispersion/root-finding layer (`src/Tearing/Dispersion/`), and the
  outer-region Δ′ exposed as the full 2m×2m matrix (`delta_prime_raw` /
  `pest3_decompose` from ForceFreeStates, fed by the new `Riccati.jl` ideal
  solver) — all as direct Julia calls, no digitization, no format drift.
  Equilibrium representations are reused, not re-ingested. This is the single
  biggest win of colocating. **Sequencing:** #238 lands before ISLET M0; if
  ISLET starts earlier, branch from `feature/tearing-growthrates` (docs/00
  milestone sequencing) so the interfaces are in hand from the first commit.
  On `develop` today the old `src/InnerLayer/SLAYER/Slayer.jl` is still a
  placeholder — do not code against `develop`'s module layout; #238 also moves
  GGJ under `src/Tearing/`. `src/KineticForces` (PENTRC/NTV) exists on
  `develop` and is the natural Level-4 torque-balance counterpart.
- **Namespace discipline**: ISLET imports GPEC; GPEC never imports ISLET until
  ISLET is stable enough to be a documented feature. One-way dependency,
  enforced by CI (a test that greps GPEC sources for `using .Islet`-type
  references).
- **CLAUDE.md layering**: Claude Code loads nested CLAUDE.md files; the repo
  root keeps GPEC-wide conventions (including the existing merge-conflict
  synthesis policy), and `islet/CLAUDE.md` (this project's, from this bundle)
  applies within the subtree. On conflict, the more specific file governs for
  work inside `islet/`; flag genuine contradictions to the human rather than
  resolving silently.
- **CI**: ISLET tests/benchmarks run as separate CI jobs so a red ISLET ladder
  never blocks unrelated GPEC merges (and vice versa), but the SLAYER/DCON
  cross-checks run in *both* suites once they exist — they protect the
  interface from both sides.
- **Branch discipline for autonomous work**: `main` protected; all agent work
  on `islet/<milestone>-<topic>` branches via PR; parallel milestones (e.g.
  M3–M4 alongside M7, per docs/00) in separate git worktrees so concurrent
  sessions never share a working tree.

## 2. The operating model: autonomy without babysitting

The goal is milestone-sized unattended runs with human attention concentrated
at a few high-leverage points. Three mechanisms make this safe:

### 2.1 Machine-checkable definition of done

Every autonomous run is launched against a milestone (docs/00) whose exit
criterion is a set of ladder IDs (docs/05) going green plus CI passing. "Done"
is never a judgment call. Launch prompts follow this template:

> Work milestone M<N> per docs/00-roadmap.md. Definition of done: ladder IDs
> <list> green with convergence artifacts archived, full test suite passing,
> PR opened with the docs/05 reporting requirements. If blocked, follow the
> escalation protocol (docs/06 §2.2) and continue with the next parallelizable
> task. Do not weaken a benchmark tolerance or re-baseline a target to reach
> done; that is a blocker, not a fix.

### 2.2 Non-blocking escalation: `islet/QUESTIONS.md`

Autonomous runs must never stall waiting for a human, and must never guess on
the things CLAUDE.md forbids guessing (coefficients, signs, normalizations,
[VERIFY] clearances). The resolution: an append-only `QUESTIONS.md` queue.
When blocked, the agent writes an entry — context, the specific question,
options considered, its recommendation, and what work is gated — then switches
to the next unblocked task. The human's recurring job is clearing this queue
(and [VERIFY] tags), not supervising sessions. Entries get IDs; commits/PRs
reference the IDs they were blocked on or unblocked by.

### 2.3 Guardrails in tooling, not vigilance

**[HUMAN SETUP]** in `islet/.claude/settings.json` (project-scoped, checked in):

- `permissions`: allow the routine loop (edit within repo, `julia --project`
  test/benchmark commands, git branch/commit/push, `gh pr` on this repo); deny
  destructive git (force-push, `push` to main), package registry publishes,
  and credential paths (extend the existing personal deny rules — keep those
  global, add repo-specific ones here so collaborators inherit them).
- Truly unattended runs (`--dangerously-skip-permissions` or equivalent
  auto modes) only inside a container/devcontainer or on a cluster node with a
  scratch clone — never on a laptop checkout with credentials in reach. The
  existing tmux-on-cluster workflow is the natural home for long runs: one
  tmux window per worktree per milestone.
- **Hooks** (checked in with the project): PostToolUse on Edit/Write → run the
  fast test subset for the touched module (seconds, not the ladder); a Stop
  hook that blocks session completion if the working tree has uncommitted
  changes or failing fast tests; PreToolUse on Bash → deny-pattern for
  destructive commands as a second layer under permissions.
- **Checkpointing** stays on (default) so exploratory refactors are cheaply
  reversible.

Consult https://code.claude.com/docs/en/ (docs map) when configuring — hook
names, permission syntax, and flags change; verify against the installed
version rather than this document.

### 2.4 GitHub Actions layer

**[HUMAN SETUP]**, building on the prior GPEC GitHub-App exploration (org
permission constraints noted there still apply — may need org-owner action):

- `anthropics/claude-code-action` for PR review on `islet/**` paths, with a
  review prompt pointing at CLAUDE.md and docs/: check [VERIFY] policy
  compliance, no regime-branches in operators, ladder IDs addressed.
- Issue-triggered autonomous work: label `claude-task` on an issue containing
  a milestone-template prompt → action runs headless (`claude -p`) in a
  container, opens a PR. This is the "assign work from a phone" channel.
- Scheduled (cron) workflows: nightly fast-ladder regression + allocation
  regression; weekly full ladder on `main`. Failures open issues automatically
  (which can themselves be `claude-task`-labeled — closing the loop on
  maintenance without human dispatch).
- Secrets: `ANTHROPIC_API_KEY` as a repo/org secret. Local-config gotcha from
  prior experience: if Claude Code ignores a correctly set key and shows the
  login menu, check `~/.claude.json` for a stale `customApiKeyResponses.rejected`
  entry.

### 2.5 Session-level habits (standing guidance for agents)

- Start milestone work in plan mode; write the plan against docs/00 before
  editing. Persist the session objective (goal/directive mechanisms) so long
  sessions don't drift from the milestone definition of done.
- Delegate to subagents (§4) for review/verification passes so the main
  context stays on implementation; summarize subagent findings into the PR.
- Commit granularly with ladder/QUESTIONS references; a session that ends
  without a pushed branch and status note has failed its exit criteria.
- Post-session, append a short entry to `islet/LOG.md`: what moved, what's
  blocked, next action. This file is the cross-session memory spine — read it
  at session start along with QUESTIONS.md.

## 3. Get Physics Done (GPD) — assessment and setup

**What it is**: open-source (Apache-2.0) agentic physics-research command pack
from Physical Superintelligence PBC, released March 2026; installs into Claude
Code (also Codex/Gemini/OpenCode) via `npx -y get-physics-done`, adding a
command ladder (`/gpd:help` → `start` → `tour` → `new-project` /
`map-research` → `resume-work`) plus an autopilot mode for directed autonomous
research. It targets exactly this project's genre — long-horizon problems
needing rigorous verification, structured research memory, multi-step
analytical work, and manuscript preparation — with a stated bias toward rigor
over agreeability. Plasma physics is among its supported subfields.
Repo: https://github.com/psi-oss/get-physics-done

**Recommendation: install and trial it, scoped to a specific lane.** The
division of labor:

- **GPD lane — derivation and verification work**: clearing [VERIFY] tags by
  independent re-derivation (its verification discipline is exactly the
  [VERIFY] workflow's counterpart); deriving the companion analytic limits the
  ladder needs (e.g. docs/05 C6 resonant-EP limit); literature mapping
  (`map-research`) for the polarization-current sign genealogy before touching
  B4; eventually manuscript drafting. GPD derivations feed
  `docs/derivations/` as `[DERIVED]` artifacts per CLAUDE.md — they *propose*
  [VERIFY] clearances; a human still signs off.
- **Native lane — implementation**: code, numerics, CI, benchmarks stay under
  this bundle's CLAUDE.md + docs. GPD is a research harness, not a
  software-engineering harness; don't let two workflow systems fight over the
  same task.

**[HUMAN SETUP]** cautions: install project-local first (not global), read
what it injects before granting — a command pack is prompt-layer software and
should be audited and version-pinned like any dependency. Confirm its
project-artifact directories (`GPD/`, `~/.gpd`) are gitignored or deliberately
committed, and that its instructions don't contradict CLAUDE.md (if they do,
CLAUDE.md wins inside `islet/`; note conflicts in QUESTIONS.md).

Its sibling GSD (general-purpose "Get Shit Done" workflow, which GPD is
modeled on) is an optional trial for milestone execution on the native lane;
adopt only if the plain milestone-prompt + hooks + ladder setup proves
insufficient — more workflow machinery is not automatically better.

## 4. Skills and subagents

### 4.1 Public skills **[HUMAN SETUP]**

Baseline installs from Anthropic's official marketplace
(`/plugin marketplace add anthropics/claude-plugins-official`, and
`anthropics/skills` as a second marketplace):

- **skill-creator** — the important one. It scaffolds skills interactively and
  runs eval loops (test cases in `evals/evals.json`, isolated subagent runs,
  graded assertions). All custom skills below get built and *evaluated* with
  it rather than hand-written.
- **code-simplifier** — Anthropic's internal cleanup pass (behavior-preserving
  simplification); run it at the end of implementation sessions to counter
  agent-accumulated complexity.
- Document skills (pdf/docx/pptx/xlsx) as needed for reports; low priority.

Community directories (skills.sh, claudeskills.info, curated lists) are worth
a periodic browse, but the ecosystem is flooded and physics-specific offerings
are thin: expect nothing that knows Julia plasma physics. Adopt sparingly,
pin versions, and audit anything that runs scripts. A community Julia skill,
if a well-maintained one exists at setup time, can seed §4.2's `julia-conventions`
skill; verify currency (Julia ecosystem skills go stale fast) and strip
anything conflicting with project conventions.

### 4.2 Custom project skills (the real leverage; build with skill-creator)

These live in-repo (`.claude/skills/` at the appropriate level) so every
session and CI agent inherits them. They exist to make *subagents and fresh
sessions* cheap to orient — progressive disclosure of exactly the context that
would otherwise be re-explained:

1. **gpec-map** (repo root): GPEC/OFT architecture — where DCON Δ′, SLAYER,
   equilibrium representations, and coordinate machinery live; module naming;
   how to run each test suite; SFL coordinate conventions (Hamada/Boozer/PEST)
   and Jacobian gotchas. Mostly distilled from existing GPEC docs + the
   maintainer's head; this skill is the highest-value few hours of human
   dictation in the whole setup.
2. **julia-conventions** (repo root): project Julia idioms — Revise workflow,
   per-thread preallocation patterns, allocation-test policy, `@inbounds`
   policy, OrdinaryDiffEq-vs-QuadGK division of labor, Interpolations boundary
   conditions. Encodes the lessons already learned so no session relearns them.
3. **islet-conventions** (`islet/`): the load-bearing distillation of docs/01–05
   — half-width convention, frames module rule, [VERIFY]/[DERIVED] workflow,
   operator-stack rules, escalation protocol. Keeps subagents aligned without
   loading the full docs.
4. **benchmark-ladder** (`islet/`): how to run `verify/` and `benchmarks/`,
   where reference data lives, how to read convergence artifacts, what
   re-baselining requires. Paired with the ladder from day one.
5. **paper-figures** (later): publication figure conventions (Makie/matplotlib
   styles, the editable-SVG lessons), once results exist.

Maintain skills with the same [VERIFY]-grade discipline as docs: they are
normative context, and a stale skill is worse than none. skill-creator's eval
loop is the regression test.

### 4.3 Project subagents

Defined in-repo so they're versioned. Minimal set:

- **physics-verifier** (read-only tools): audits diffs against docs/01
  conventions and the [VERIFY] policy; adversarial by instruction ("find the
  sign error" posture). Runs before every PR.
- **numerics-reviewer** (read-only): convergence-artifact and
  allocation-regression review; checks that "passing" benchmarks meet the
  docs/05 reporting rules.
- **literature-scout** (read + web): given a [VERIFY] tag, retrieves the source
  (arXiv/DOI), extracts the exact equation context, and drafts the clearance
  proposal for human sign-off. Pairs with an arXiv/paper-search MCP server if
  one is connected **[HUMAN SETUP — optional]**; audit any third-party MCP
  server before connecting, same rules as command packs.

## 5. Human attention budget (what "not babysitting" costs instead)

Steady state, the human's recurring surface is: (1) the QUESTIONS.md queue,
(2) [VERIFY]/[DERIVED] sign-offs, (3) PR review of milestone branches (with
the action + subagents having pre-reviewed), (4) gate sign-offs and Decision
Log entries at level boundaries. Everything else — implementation, tests,
regression triage, benchmark bookkeeping, nightly maintenance — is delegated.
If any other category starts consuming attention, that's a tooling bug: fix
the hook/skill/prompt, don't absorb the load manually.

## 6. Setup checklist (condensed)

**[HUMAN SETUP]**, in order:
1. Create `islet/` subpackage skeleton in the GPEC repo; drop this bundle in;
   commit docs before any code.
2. Project settings: permissions allow/deny, hooks, checked into `islet/.claude/`.
3. Branch protection on main; worktree convention documented in LOG.md.
4. GitHub Actions: claude-code-action PR review; `claude-task` issue workflow;
   nightly/weekly ladder crons; API key secret.
5. Plugin marketplaces + skill-creator + code-simplifier.
6. Build gpec-map and julia-conventions skills (human-dictated, skill-creator
   evaluated); islet-conventions and benchmark-ladder skills alongside M0–M1.
7. Define the three subagents.
8. GPD: project-local install, audit, trial on one [VERIFY] clearance and one
   derivation task; decide lane adoption after two weeks of use.
9. First autonomous run: M1 (operator-stack skeleton + MMS harness) with the
   §2.1 template — deliberately low-physics-risk to shake out the tooling.
