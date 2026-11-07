# How to work this project

Read this file in full before doing anything else. It describes how this
project is executed. The scientific content is in the other files.

You are almost certainly on a linux cluster with no Zotero, no Drive, and no
literature search. Everything you need is inside the project directory, one
level above this repo. Do not go looking for context elsewhere.

## Read first, in this order

1. `../../../CLAUDE.md` and `../../../PLAN.md` - the project: the two repos,
   the paper skeleton, the key results and their figures, the milestone
   ladder, the budget.
2. `README.md` (this directory) - the code-side goal, scope boundary and stop
   rules.
3. `physics.md` - the cleared equations. **Authoritative.** A human signed this
   off before this branch existed.
4. `../../../JOURNAL.md` - where work last stopped, and the parked list.
5. `../../../QUESTIONS.md` - what is open and what has been answered.

**Never re-derive what physics.md states.** If physics.md is silent, ambiguous,
or contradicts the code, that is a Decisions card - not a derivation
exercise. Re-deriving is how an agent generates months of real work that
converges on nothing.

`../../../refs/` holds the source PDFs, digitized reference curves, the
BibTeX file and the claims table from the tracker's literature pass. Consult
the *named section* of a named paper to check a sign, a limit or a
definition. Never read a paper end to end; that is what physics.md already
did. The claims table says which statements were verified from full text and
which were not; do not promote an unverified claim into the paper.

## The loop, per milestone

1. **Produce the figure first**, from current behavior, before changing
   physics. A milestone whose plot cannot be drawn from today's code is not
   yet a milestone - split it.
2. Make the change described in the milestone. Nothing else. The "not in this
   milestone" list is binding.
3. Run the milestone's physics checks from PLAN.md - the limits, scalings,
   signs, symmetries and convergence the new piece must show - and record
   each as pass / fail / not settled with its panel. A failed check means
   the milestone is not green, whatever the key figure looks like. Then
   regenerate the figure and compare against the key result's *qualitative*
   acceptance criterion.
4. Write `reports/M<k>.html` (one per attempt, republished in place), in the
   order below, and publish it.
5. Update `../../../JOURNAL.md`: one entry per finding, tagged with the paper
   section or key result it serves; anything that serves none goes to the
   parked list, not into PLAN.md.
6. Commit and push `feature/<slug>` in `gpec/`, the project repo (PLAN, JOURNAL,
   QUESTIONS, reports' figures), and the paper repo once it exists. If the
   report has an open Decisions card that gates the next milestone, stop and
   wait for the answer; otherwise go on to the next milestone. One milestone,
   one PR, target under ~400 lines of diff.

Start each milestone, in the same session or a new one, by re-reading
JOURNAL.md and the answers block in QUESTIONS.md: state lives in those files,
not in the conversation, so nothing depends on what context compaction keeps.
Use `git worktree`
off `gpec/` for trial branches and delete them when done; keep
`feature/<slug>` clean.

## The report is the deliverable, and the only place Nik is asked anything

Nik does not read the code. The report is how he learns what the code now
does, what it found and what is proposed, well enough to decide. In his words:
"I want to understand what's being done, the result, and the proposed fix all
right next to the relevant question." He reads one report per attempt, in
under an hour, without the terminal and without the journal; everything he
needs is in that one file. Self-contained HTML (no CDN, no external images -
inline the SVG or embed a data: URI), light/dark aware, printable as a
fallback.

**One report per attempt.** Never a separate explainer page: when Nik needs
more context, fold it into the report and republish it at the same link.

Sections, in order:

- **Status strip** - one row: milestone state (green / amber / red, as the
  skeleton defines them), attempts used of budget, tests (pass/fail), wall
  time of the key run, number of open decisions. Plain words only.
- **Lede** - one paragraph: what the milestone is for, the result in brief,
  and a map of which section holds which card and which cards gate the next
  milestone.
- **§1 Orientation** - the question; the physics in one paragraph; a
  schematic of the idea; the D2 workflow diagram with the changed node
  marked; a diagram of any numerical scheme built (grids, tables, fill
  rules); the physics-checks table from PLAN.md (expected, seen, pass / fail /
  not settled), each row linking down to the section that holds its evidence.
- **One section per decision**, in this order inside the section: what is
  being done (schematic); the result (figures and numbers); why (the
  diagnostics); the proposed fix, drawn as a figure (new versus old); what
  the figures do and do not settle; then the Decisions card at the end of the
  section. A milestone with no decision has one Results section built the
  same way, without a card.
- **Paper** - what changed in PLAN.md: abstract diff, key-results diff,
  figures promoted, and the coherence check from the template. Any change to
  the abstract claims or the key-results list is a Decisions card.
- **Next** - the following milestone, or the blocker.

There is no shared Figures section above the cards: every figure a card rests
on sits in that card's section. If one card needs evidence from two topics,
split it into two cards; the limit of three still holds. One "Copy answers"
button follows the last card.

**Cards are short, because the section did the explaining:** what was found,
citing figures by number; why it is Nik's call; the options, each with its
consequence and what green would mean; the recommendation and how sure it
is; a radio button per option and a free-text box.

**Definitions, journal style.** Define every acronym, symbol and code name in
running text at first use, as a paper does ("the finite orbit width (FOW)
limit", "forb, the new orbit method"), and use it bare afterwards. Every
report starts fresh, even if an earlier report defined the term. No glossary,
no symbols table, no per-card definitions block: Nik finds definitions with
Ctrl+F, so the first hit of every term must be its definition - check this
before publishing, including the status strip and the lede. A caption may
define the terms its figure introduces.

**Numbers are measured, not remembered.** Every timing, error or count in a
report names the job or run it came from (`results/<run>/`, the Slurm job,
the log line). A number found wrong later is corrected explicitly in the
next report, with the wrong value, the right one and where each came from.

**Delivery.** Publish each report as a private claude.ai Artifact when the
session has the Artifact tool, and republish every revision in place at the
same URL; record the URL in the report's meta line and in JOURNAL.md. The git
copy under `reports/` is the record. Without the Artifact tool, Nik opens
the file from a pulled checkout.

Rules for questions:

- **A question exists only as a Decisions card.** QUESTIONS.md is the ledger
  that records cards and their answers; it never holds a question that has
  not appeared as a card. Never ask Nik interactively in the terminal: a
  prompt there has no context he can see.
- **Three cards per milestone at most.** A fourth means the milestone is
  scoped wrong: stop, and make the one card "how to split this milestone",
  with the proposed split as the recommended option.
- **A card is answerable from its section alone.** No reference to "the
  earlier run" without the figure or number itself, no option whose
  consequence is not stated.
- **Every card has a recommendation.** "Either is fine" is not a card; make
  the choice and record it in the journal.
- **Deferred work stays in the plan.** An answer that defers work out of
  scope still gets a dated wrap-up task in PLAN.md (usually under the graduate
  milestone) so it is not lost - e.g. "open a GPEC issue on the kernel after
  the PR is ready".

Assume the report is read a week later by someone who has not followed the
work. If it needs the journal or the terminal to be understood, it has
failed.

**Milestone 0 is a report with no implementation.** Background, the plan, the
pipeline diagram, a plot of current behavior, the first PLAN.md, and the two
shared figure helpers below. It ships before any physics change, so the plan
can be corrected while that is still free. **Milestone 1 is diagnostic-only.**

## Figures

Schematics, workflow diagrams, grids and before/after views of a fix matter
as much as result plots; Nik reuses them in talks and long papers, so make
many and make each one slide- and paper-ready. Follow the "Figures" rules in
`~/.claude/CLAUDE.md` where it exists; in short:

- True print size: 3.4 in wide by default, 5.0 or 7.0 in when needed, with
  the same 10 pt fonts at every width.
- Bold serif text from the matplotlibrc `paper_bold` style; never override
  its fonts, sizes or weights.
- Short legends and panel letters; the explanation goes in the caption.
- Few colorblind-safe colors, one fixed color per physical category,
  consistent across the whole project; series also distinguished by dash and
  marker.
- Every figure saved as PDF and 300 dpi PNG (and SVG for the report).

At M0, create two shared helpers in `reports/` and use them for every figure
after: `style.py` (the project palette, the three widths, a panel-letter
helper, and `save()` writing PDF, PNG and SVG) and `d2render.sh` (bold serif;
SVG, PDF and PNG; about 12 nodes at most; one node shape; color only on
changed or failing nodes).

## Cluster hygiene

- **Worktrees:** one snapshot worktree per project, at
  `/mnt/scratch/<user>/wt/<slug>`. Never a shared path: a shared `wt/tests`
  belonged to another repo and failed with "reference is not a tree".
- **Unreadable caches:** when a struct change breaks cached state, write a
  new `results/<run>/` directory. Never overwrite or delete the old one;
  deleting results needs a card.
- **Non-interactive commands:** shell aliases such as `cp -i` hang an
  unattended session. Use `/bin/cp -f … </dev/null` and
  `timeout 60 git push … </dev/null`, and the same for anything that can
  prompt.

## The paper

PLAN.md is the paper skeleton. Its key results are hypotheses, each with the
figure that proves it; the figure scripts live in `../../../paper/figure_scripts/`
and write PDFs to the top level of the paper repo; each script names the
`gpec.h5` and the commit it reads. A figure enters the paper only after it
appeared in a report and the journal records Nik's approval. Internal reports
may carry every plot and quirk; the paper carries promoted figures only.

The paper draft is its own milestone, second to last: Methods first, then
hypothesis-driven Results sections, Conclusion, Introduction, then the
coherence check and the AI-assisted review, in that order, as the template's
comment block prescribes. Do not write paper prose before that milestone
beyond the abstract and the Methods outline.

## What you may not do without a Decisions card

- Merge to `develop`.
- Change `physics.md`.
- Change the key-results list or the abstract claims in `PLAN.md`.
- Delete anything in `../../../results/`.

## Stop rules

These are deliberately the opposite of "never stall, switch to the next
unblocked task". That protocol produces months of lateral motion.

- The budget is in `README.md` and `PLAN.md`. It is a real limit, not a target.
- **A milestone that is not green after two attempts (each ends in a report):
  write to QUESTIONS.md and
  STOP.** Do not start the next milestone. Do not open a new line of attack.
- **Blocked on anything a human must decide: write the Decisions card, record
  it in QUESTIONS.md, write the report, and STOP that milestone.** Do not work
  around it, and do not ask in the terminal.
- **An external benchmark that has not arrived is not a blocker.** Its
  request milestone exits at "package sent"; work on everything that does not
  consume it, and when the consuming milestone comes up without the data, use
  the fallback curve in refs/ and say so in the report.
- **A question unanswered for a whole milestone is itself a stop.** Say so in
  the report and wait.
- If the same blocker recurs across two milestones, say plainly that the
  project's premise may be wrong. That sentence is more valuable than more
  commits.

## Graduate - the final milestone, non-optional

The packet never merges. The last milestone:

1. Moves `physics.md` into `docs/src/` as the real feature documentation,
   edited to house style.
2. Turns each key-result figure into a regression-harness case so the figures
   regenerate on demand.
3. Folds the reports into one consolidated docs page, or leaves them
   superseded by the paper draft.
4. **Deletes `docs/workorders/<slug>/` entirely**, this file included. The
   merge PR into develop contains no packet files.
5. Tags the project repo `graduated`; it already carries the history of
   PLAN, JOURNAL, QUESTIONS and the reports, so nothing is copied into the
   paper repo.

## How to write

Write with scientific restraint in every report, journal entry, question and
paper paragraph. State the fact and let the reader judge its weight.

- No emphatic intensifiers on a claim: exactly, precisely, genuinely, truly,
  simply, crucially, critically, importantly, notably, by far, the single most.
- No editorial asides about a finding. A consequence is written as a
  consequence, not as a remark about how interesting it is.
- No unsourced superlatives or novelty claims.
- Plain verbs: "is" over "amounts to", "shows" over "reveals", "does not
  reproduce" over "fails to capture".
- Hedges are expected and correct - "appears to", "unverified", "in the case
  run". Confidence you have not earned is not.
- Real subscripts in anything typeset; never underscore notation in prose.

It matters most where a decision section says what its figures show and
what they do not settle: state both, without arguing for a conclusion.
