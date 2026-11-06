
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
4. Write `reports/M<k>.html` from the skeleton: orientation, then one
   section per decision with its card at the end, then Paper and Next.
5. Update `../../../JOURNAL.md`: one entry per finding, tagged with the paper
   section or key result it serves; anything that serves none goes to the
   parked list, not into PLAN.md.
6. Commit the packet and the paper repo and push both. Stop only if the
   report has a gating Decisions card; otherwise continue to the next
   milestone. One milestone,
   one PR, target under ~400 lines of diff.

Start each milestone, in the same session or a new one, by re-reading
JOURNAL.md and the answers in QUESTIONS.md. Use `git worktree`
off `gpec/` for trial branches and delete them when done; keep
`feature/ntv-orbit` clean.

## The report is the deliverable, and the only place Nik is asked anything

The code exists to produce the figure; the figure exists to be understood.
Nik reads one report per milestone, in under an hour, at the end of his day,
without the terminal and without the journal. Everything he needs to judge
the milestone and make its decisions is in that one file. Self-contained
HTML (no CDN, no external images - inline the SVG or embed a data: URI),
greyscale-safe, printable as a fallback, opened from the pulled checkout
(GitHub shows the source, not the page).

Nik does not read the code. The report is how he learns what the code now
does, what it found and what is proposed, well enough to decide. Write it
as a guided explanation, not a results dump followed by a quiz. Nik
(2026-11-06): "I want to understand what's being done, the result, and the
proposed fix all right next to the relevant question so I can think about
it and answer it with the appropriate context right there in front of me."

Sections, in order:

- **Status strip** - one row at the top: milestone state (green / amber /
  red, as the skeleton defines them), attempts used of budget, tests
  (pass/fail counts), cost (wall time of the key run this milestone and last),
  number of decisions below. Plain words only: readable in five seconds.
- **Lede** - what the milestone is for, in plain language, defining the
  first acronyms and code names it needs; then "in brief", the result in
  three or four sentences; then a short map: which section holds which
  decision card, and which cards gate the next attempt or milestone.
- **1. What this milestone computes** - the orientation every later section
  relies on: the question (two sentences), the physics in one paragraph,
  a schematic of the idea, a workflow diagram (D2) of where the change sits
  in the pipeline with the new node marked, and diagrams of any numerical
  scheme the milestone builds (grids, tables, fill rules). Then the physics
  checks from PLAN.md as a short table, one row each (expected, seen,
  pass / fail / not settled), each row linking down to the section that
  holds its evidence.
- **2 ... k. One section per decision**, in reading order, each built as:
  what is being done (schematic or workflow), the result (the figure and
  the numbers), why (diagnostic figures), the proposed fix (a figure of the
  fix itself, e.g. the new grid or the new workflow beside the old), open
  questions, and then **the Decisions card at the end of that section**.
  Every figure a card rests on sits in that card's section, never in a
  shared Figures section above. A finding with no decision gets a section
  only if a check rests on it; otherwise it goes in Reading, Paper or Next.
  If one card needs evidence from two topics, split it into two cards (the
  limit of three still holds).
- **Each card** is short because its section did the explaining: what was
  found (one or two sentences, pointing at figures by number); why the
  agent cannot decide it alone; the options, each with its consequence in
  one line and, for an attempt, what green means; the recommendation and
  how sure; a radio button per option plus a free-text box. One "Copy
  answers" button after the last card assembles the QUESTIONS.md block.
  No decisions: a one-line section says so.
- **Cost** - only when the milestone touched performance or a run exceeded
  its previous wall time by more than a factor of two, and then inside the
  section whose decision it affects.
- **Paper** - what changed in PLAN.md: abstract diff, key-results diff,
  figures promoted from "evidence in a report" to "promoted to the paper",
  and the coherence check from the template (abstract states the final
  story; introduction promises only what is delivered; methods suffice for
  the results; results support the conclusion; every figure earns its place).
  Any change to the abstract claims or the key-results list is a Decisions
  card, not a unilateral edit.
- **Next** - the following milestone, or the blocker.

One report per milestone attempt. Never a separate "explainer" page: if Nik
asks for more context, fold it into the report and republish it at the same
link.

**Definitions, journal style.** Define every acronym, symbol and code name
in running text at its first occurrence in the report, the way a journal
paper does: "the finite orbit width (FOW) limit", "the orbit width
Δ<sub>b</sub>, the ψ range the orbit spans", "`forb`, the new orbit
method". Afterwards use it bare. Every report starts fresh, even if an
earlier report defined the term. No symbols table or glossary block: Nik
finds a definition with Ctrl+F on its first occurrence, so the first hit
of every term must be its definition (check this before publishing, and
keep the status strip and nav free of undefined terms). Captions may
define a term the figure introduces.

**Figures and schematics: as many as help, ready for slides and papers.**
Schematics of the approach, workflow diagrams (D2), grids and before/after
views of a fix are as important as result plots, and Nik reuses them in
talks, so every figure follows his global figure rules (`~/.claude/CLAUDE.md`,
"Figures"): design at true print size (3.4 in single column by default,
5.0 or 7.0 in when needed, same 10 pt fonts), bold serif text, short
legends, panel letters, explanations in the caption, few colors. In
matplotlib, `reports/ntvstyle.py` holds the project palette (trapped,
passing, lost, unclosed; surface vs orbit path), the widths, `panel()` and
`save()` (PDF, 300 dpi PNG, SVG for the HTML); never override the rc fonts,
sizes or weights. Render D2 with `reports/d2render.sh` (bold serif, SVG,
PDF, PNG); keep diagrams to ~12 nodes, one node shape, dark edges, and
color only the changed or failing nodes. Figures made before 2026-11-06
predate these rules and are not redone.

Rules for questions:

- **A question exists only as a Decisions card.** QUESTIONS.md is the ledger
  that records cards and their answers; it never holds a question that has
  not appeared as a card. Never ask Nik anything interactively in the
  terminal: sessions run unattended, and a prompt there has no context he
  can see.
- **Three cards per milestone at most.** A fourth means the milestone is
  scoped wrong: stop, and make the one card "how to split this milestone",
  with the agent's proposed split as the recommended option.
- **A card is answerable from its section.** No acronym or code name
  undefined upstream, no reference to "the earlier run" without the figure
  or number itself, no option whose consequence is not stated.
- **Every card has a recommendation.** "Either is fine" is not a card; make
  the choice and record it in the journal.

Assume the report is read a week later, by someone who has not followed the
work. If it needs the journal or the terminal to be understood, it has failed.

**Milestone 0 is a report with no implementation.** Background, the plan, the
pipeline diagram, a plot of current behavior, and the first PLAN.md. It ships
before any physics change, so the plan can be corrected while that is still
free. **Milestone 1 is diagnostic-only.**

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
- **A milestone that is not green after two attempts (each ends in a report): write to QUESTIONS.md and
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
4. **Deletes `docs/workorders/ntv-orbit/` entirely**, this file included. The
   merge PR into develop contains no packet files.
5. Copies `PLAN.md`, `JOURNAL.md` and `QUESTIONS.md` into `paper/notes/` so the
   paper repo carries its own history. The project directory stays.

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

The Reading section of a report is where this matters most: say what the
figure shows and what it does not settle, without arguing for a conclusion.
