---
description: Draft the Unreleased section of CHANGELOG.md from merged pull requests
argument-hint: "[--since <ref|date>] [--write]"
allowed-tools: Bash(gh:*), Bash(git:*), Read, Edit
---

Draft release notes from merged pull requests and place them in the `## Unreleased`
section of `CHANGELOG.md`.

Arguments: `$ARGUMENTS`

## Determining the range

Use `--since` if given (a git ref or an ISO date). Otherwise use the most recent
version tag:

```bash
git tag -l 'v*' --sort=-v:refname | head -1
```

The repository may have no version tags yet. If there is none and no `--since` was
given, **stop and ask** which starting point to use rather than defaulting to the
whole history.

Convert the starting point to a date, then list merged PRs:

```bash
gh pr list --state merged --base develop --limit 200 \
  --search "merged:>=<date>" \
  --json number,title,mergedAt,body,labels,author
```

## Building each entry

For every PR, parse the `## Release note` block from its body:

- **Audience** — `developers` entries are omitted from the notes entirely.
- **Numerical impact** — text after this label, minus the `_(harness @ sha)_` stamp.
- **Migration** — what a user must change.
- The prose sentences below the fields are the entry text.

Parse the title as `Area[.Submodule] - TAG[!] - Summary`.

Place each entry:

- **Any title with `!`** goes in `Changed results & breaking changes`, regardless of
  its tag. Give the numerical impact and the migration text, not just the summary.
  This section is the reason the mark exists — never let a marked change appear only
  under its tag.
- Otherwise map the tag to its section: `FEATURE` → New capabilities, `BUGFIX` → Bug
  fixes, `PERF` → Performance, `API` → Interface & format changes, `DEPRECATION` →
  Deprecations, `DOCS` → Documentation.
- `MINOR`, `REFACTOR`, and `TEST` are omitted unless marked with `!`.

Within a section, group by Area, ordering areas by entry count. Write in the past
tense, in terms a GPEC user would recognize, and end each entry with the PR number
in parentheses.

## Gaps

Some PRs will have no parseable block. **List them explicitly at the end of your
report as unprocessed, with their numbers and titles.** Do not invent an entry from
a title or a diff — an unreported change is a visible gap, whereas a fabricated one
is indistinguishable from a real entry and corrupts the record.

Also report any PR whose block says `Audience: users` but whose prose is empty.

## Output

Print the drafted sections for review. Only edit `CHANGELOG.md` if `--write` was
passed; leave sections with no entries in place and empty. Never invent a version
number or move entries out of `## Unreleased` — releasing is a human decision.
