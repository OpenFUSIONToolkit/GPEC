#!/usr/bin/env python3
# PEP 723 inline metadata, so the script runs anywhere without setting up an
# environment: `uv run ci/conventions/check_subject.py --title "..."`.
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Validate pull request titles, release-note blocks, and commit subjects.

Two things block on this: the `pr-conventions` CI job, which checks a pull request
title, its release-note block and the freshness of its harness stamp; and the
`naming-table-in-sync` pre-commit hook, which checks the Area table in the docs
against `src/`. Commit subjects follow the same grammar but are not enforced --
`--commit-msg` and `--stdin-report` exist for checking them by hand or from a
locally installed hook.

The grammar and its rationale are documented in `docs/development/naming.md`.

Valid Areas are derived from `src/` on disk rather than hardcoded, so adding or
removing a module updates the vocabulary automatically.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

# Tags that name a release-note section.
RELEASE_NOTE_TAGS = ("FEATURE", "BUGFIX", "PERF", "API", "DEPRECATION", "DOCS")

# Tags whose changes never appear in release notes.
EXCLUDED_TAGS = ("MINOR", "REFACTOR", "TEST")

ALL_TAGS = RELEASE_NOTE_TAGS + EXCLUDED_TAGS

# `!` marks a user-visible change; it is incoherent on tags that cannot move
# results or break an interface. An author reaching for one of these has
# mis-tagged the change.
NO_BANG_TAGS = frozenset({"MINOR", "TEST", "DOCS"})

# Areas that are not Julia modules. Closed set.
NON_MODULE_AREAS = (
    "Benchmarks",
    "Build",
    "CI",
    "Docs",
    "Examples",
    "Regression",
    "Repo",
    "Test",
)

# The package entry point is not an Area; a change there is `Repo`.
PACKAGE_ENTRY = "GeneralizedPerturbedEquilibrium.jl"

# Spellings used before this convention, mapped to their replacement. Purely to
# make the error message instructive; none of these are accepted.
LEGACY_TAGS = {
    "IMPROVEMENT": "FEATURE if user-visible, else PERF, REFACTOR or MINOR",
    "CLEANUP": "REFACTOR if structural, else MINOR",
    "FIX": "BUGFIX",
    "NEW": "FEATURE",
    "OPTIMIZATION": "PERF",
    "PERFORMANCE": "PERF",
    "TESTS": "TEST",
    "VALIDATION": "TEST",
    "WIP": "MINOR",
    "CONFIG": "MINOR",
    "UPDATE": "MINOR",
    "PLANNING": "MINOR",
    "REVIEW": "MINOR",
    "MMINOR": "MINOR",
}

LEGACY_AREAS = {
    "EQUIL": "Equilibrium",
    "VAC": "Vacuum",
    "FFS": "ForceFreeStates",
    "PE": "PerturbedEquilibrium",
    "GPEC": "Repo",
    "ALL": "Repo",
    "MULTIPLE": "Repo",
    "AGENTS": "Docs",
    "H5": "HDF5Schema",
    "BENCH": "Benchmarks",
    "EXAMPLE": "Examples",
    "GALERKIN": "ForceFreeStates.Galerkin",
    "GGJ": "InnerLayer.GGJ",
    "SLAYER": "InnerLayer.SLAYER",
    "RICCATI": "ForceFreeStates",
    "SINGULARCOUPLING": "PerturbedEquilibrium",
    "COILS": "ForcingTerms",
    "FOURIER": "Utilities",
}

# Subjects git or a rebase generates for us are not the author's to format.
EXEMPT_SUBJECT = re.compile(r"^(Merge |Revert |fixup!|squash!|amend!)")

SUBJECT = re.compile(r"^(?P<areas>\S+) - (?P<tag>[A-Za-z]+)(?P<bang>!?) - (?P<summary>\S.*)$")

RELEASE_NOTE_HEADING = re.compile(r"^##\s+Release note\s*$", re.MULTILINE)
FIELD = r"^\s*[-*]\s*\*\*{label}:\*\*\s*(?P<value>.*?)\s*$"
HARNESS_STAMP = re.compile(r"_\(harness @ (?P<sha>[0-9a-fA-F]{7,40})\)_")

# Template text left unedited. Requires a closing bracket so that prose such as
# "q95 moved (<1%)" is not mistaken for a placeholder.
PLACEHOLDER = re.compile(r"<[^>]{3,}>|\bTODO\b|\bFIXME\b", re.IGNORECASE)


def repo_root() -> Path:
    """Locate the repository root, preferring the script's own location."""
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "src").is_dir():
        return candidate
    return Path.cwd()


def module_areas(root: Path) -> list[str]:
    """Derive module and submodule Areas from `src/` on disk.

    A subdirectory qualifies as a submodule Area only if it holds Julia sources,
    which keeps data directories such as `ForcingTerms/coil_geometries` out of
    the vocabulary. File-based submodules are deliberately not Areas: the parent
    module is enough for triage and the summary carries the detail.
    """
    src = root / "src"
    if not src.is_dir():
        return []

    areas: list[str] = []
    for module in sorted(p for p in src.iterdir() if p.is_dir()):
        areas.append(module.name)
        for sub in sorted(p for p in module.iterdir() if p.is_dir()):
            if any(sub.glob("*.jl")):
                areas.append(f"{module.name}.{sub.name}")

    # Top-level sources such as Rerun.jl and HDF5Schema.jl are components in
    # their own right and are committed against directly.
    for source in sorted(src.glob("*.jl")):
        if source.name != PACKAGE_ENTRY:
            areas.append(source.stem)
    return areas


def valid_areas(root: Path) -> list[str]:
    return module_areas(root) + list(NON_MODULE_AREAS)


def suggest(token: str, candidates: list[str], legacy: dict[str, str]) -> str:
    """Return a ' — use X' hint for a rejected token, or an empty string."""
    replacement = legacy.get(token.upper())
    if replacement is None:
        # Catches spellings that differ only in case, e.g. DOCS for Docs.
        folded = {c.casefold(): c for c in candidates}
        replacement = folded.get(token.casefold())
    if replacement is None:
        close = difflib.get_close_matches(token, candidates, n=1, cutoff=0.7)
        replacement = close[0] if close else None
    return f" — use {replacement}" if replacement else ""


def check_subject(subject: str, areas: list[str]) -> list[str]:
    """Return a list of violations for one subject line; empty means valid."""
    subject = subject.strip()
    if not subject or EXEMPT_SUBJECT.match(subject):
        return []

    match = SUBJECT.match(subject)
    if not match:
        return ["does not match `Area - TAG - Summary` (single-token Area, single-word TAG, ' - ' separators)"]

    errors: list[str] = []

    tag = match.group("tag")
    if tag not in ALL_TAGS:
        errors.append(f"unknown TAG {tag!r}{suggest(tag, list(ALL_TAGS), LEGACY_TAGS)}")
    elif match.group("bang") and tag in NO_BANG_TAGS:
        errors.append(f"{tag}! is not allowed — {tag} changes cannot move results or break an interface")

    # A change spanning exactly two areas may name both as `A/B`.
    parts = match.group("areas").split("/")
    if len(parts) > 2:
        errors.append("more than two Areas — use `Repo` for a change spanning three or more")
    for part in parts:
        if part not in areas:
            errors.append(f"unknown Area {part!r}{suggest(part, areas, LEGACY_AREAS)}")

    return errors


def format_vocabulary(areas: list[str]) -> str:
    modules = [a for a in areas if a not in NON_MODULE_AREAS]
    return (
        "\nValid TAGs:\n"
        f"  release-note: {', '.join(RELEASE_NOTE_TAGS)}\n"
        f"  excluded:     {', '.join(EXCLUDED_TAGS)}\n"
        f"  `!` marks a user-visible change; not allowed on {', '.join(sorted(NO_BANG_TAGS))}\n"
        "\nValid Areas:\n"
        f"  modules: {', '.join(modules)}\n"
        f"  other:   {', '.join(NON_MODULE_AREAS)}\n"
        "\nSee docs/development/naming.md\n"
    )


def report_subject(subject: str, errors: list[str], areas: list[str]) -> None:
    print(f"Invalid subject: {subject}", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    print(format_vocabulary(areas), file=sys.stderr)


def field_value(body: str, label: str) -> str | None:
    match = re.search(FIELD.format(label=label), body, re.MULTILINE)
    return match.group("value") if match else None


def check_body(body: str, title: str | None, touches_src: bool) -> list[str]:
    """Validate the `## Release note` block of a PR body."""
    if not RELEASE_NOTE_HEADING.search(body):
        return ["missing a `## Release note` section (see .github/PULL_REQUEST_TEMPLATE.md)"]

    errors: list[str] = []
    values: dict[str, str] = {}
    for label in ("Audience", "Numerical impact", "Migration"):
        value = field_value(body, label)
        if value is None:
            errors.append(f"`## Release note` has no **{label}:** line")
        elif not value or PLACEHOLDER.search(value):
            errors.append(f"**{label}:** is empty or still holds template text")
        else:
            values[label] = value

    audience = values.get("Audience")
    if audience and audience not in ("users", "developers"):
        errors.append(f"**Audience:** must be `users` or `developers`, not {audience!r}")

    impact = values.get("Numerical impact")
    if impact and touches_src and not HARNESS_STAMP.search(impact):
        errors.append(
            "**Numerical impact:** must carry the commit the harness ran at, "
            "e.g. `none _(harness @ a3bdfd21)_`, because this PR changes files under src/"
        )

    if title:
        match = SUBJECT.match(title.strip())
        if match and match.group("bang"):
            migration = values.get("Migration", "")
            if migration.lower().startswith("none"):
                errors.append("title carries `!`, so **Migration:** cannot be `none` — say what a user must change")

    return errors


def check_table(path: Path, areas: list[str]) -> list[str]:
    """Assert the Area table in naming.md lists exactly the modules on disk."""
    if not path.is_file():
        return [f"{path} does not exist"]

    # Only rows pointing at a top-level entry of src/ are compared; submodule
    # rows such as `src/InnerLayer/GGJ/` are documented but not required here.
    top_level = re.compile(r"`src/([^/`]+?)(?:/|\.jl)`")

    documented = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        match = top_level.fullmatch(cells[2])
        if match:
            documented.add(match.group(1))

    on_disk = {a for a in areas if a not in NON_MODULE_AREAS and "." not in a}
    errors = []
    for missing in sorted(on_disk - documented):
        errors.append(f"module {missing!r} exists in src/ but is not in the Area table")
    for extra in sorted(documented - on_disk):
        errors.append(f"Area table lists {extra!r}, which is not a directory in src/")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--commit-msg", metavar="FILE", help="validate the subject line of a commit message file")
    parser.add_argument("--title", metavar="TEXT", help="validate a PR title")
    parser.add_argument("--body", metavar="FILE", help="validate the release-note block of a PR body file")
    parser.add_argument("--touches-src", action="store_true", help="with --body, require the harness stamp")
    parser.add_argument("--harness-sha", metavar="FILE", help="print the harness commit stamped in a PR body")
    parser.add_argument("--check-table", metavar="FILE", help="check an Area table against src/ on disk")
    parser.add_argument("--stdin-report", action="store_true", help="validate subjects piped one per line")
    args = parser.parse_args()

    root = repo_root()
    areas = valid_areas(root)
    failed = False

    if args.harness_sha:
        match = HARNESS_STAMP.search(Path(args.harness_sha).read_text(encoding="utf-8"))
        print(match.group("sha") if match else "")
        return 0

    if args.commit_msg:
        lines = Path(args.commit_msg).read_text(encoding="utf-8").splitlines()
        subject = next((line for line in lines if line.strip() and not line.startswith("#")), "")
        errors = check_subject(subject, areas)
        if errors:
            report_subject(subject, errors, areas)
            failed = True

    if args.title:
        errors = check_subject(args.title, areas)
        if errors:
            report_subject(args.title, errors, areas)
            failed = True

    if args.body:
        errors = check_body(Path(args.body).read_text(encoding="utf-8"), args.title, args.touches_src)
        if errors:
            print("Invalid `## Release note` block:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            print("\nSee docs/development/naming.md\n", file=sys.stderr)
            failed = True

    if args.check_table:
        errors = check_table(Path(args.check_table), areas)
        if errors:
            print(f"Area table in {args.check_table} is out of sync with src/:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            failed = True

    if args.stdin_report:
        total = bad = 0
        for line in sys.stdin:
            if not line.strip():
                continue
            total += 1
            errors = check_subject(line, areas)
            if errors:
                bad += 1
                print(f"{line.strip()}\n    {'; '.join(errors)}")
        print(f"\n{total - bad}/{total} subjects valid, {bad} rejected")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
