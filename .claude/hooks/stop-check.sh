#!/usr/bin/env bash
# Stop hook for unattended Islands runs.
# `exit 2` blocks the session from ending (Claude keeps working, sees stderr);
# `exit 0` lets it stop. Purpose: never end a night with a dirty tree or a
# broken build.
set -uo pipefail

payload="$(cat)"

# Avoid an infinite loop: if we are already continuing because of this hook,
# let the session stop.
active="$(printf '%s' "$payload" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("stop_hook_active", False))' 2>/dev/null || echo False)"
if [ "$active" = "True" ]; then
  exit 0
fi

root="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || echo .)}"
cd "$root" 2>/dev/null || exit 0

# 1. Working tree must be clean (commit granularly, per the protocol).
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
  echo "BLOCKED stop: uncommitted changes present. Commit (to feature/islands) or stash before ending; then write a LOG.md entry." >&2
  exit 2
fi

# 2. Build/tests must pass. Requires julia on PATH (see QUESTIONS.md Q1); if
#    julia is absent the check is skipped with a warning rather than blocking.
# Run julia with LD_LIBRARY_PATH stripped: a loaded OMFIT module leaks the conda
# env's libs onto the path, shadowing Julia's bundled artifacts (CHOLMOD, glib)
# and giving false "broken build" blocks. Julia uses its own RPATH, so unsetting
# is a no-op on a clean shell / CI.
if command -v julia >/dev/null 2>&1; then
  # Once M1 lands test/runtests_islands_*.jl, prefer running just those:
  #   julia --project=. test/runtests.jl test/runtests_islands_grids.jl ...
  # Until then, a package-load smoke check.
  if ! env -u LD_LIBRARY_PATH julia --project=. -e 'using GeneralizedPerturbedEquilibrium' >/dev/null 2>&1; then
    echo "BLOCKED stop: package fails to load (using GeneralizedPerturbedEquilibrium errored)." >&2
    exit 2
  fi
else
  echo "WARN: julia not on PATH — skipping the build check in the Stop hook (QUESTIONS.md Q1)." >&2
fi

exit 0
