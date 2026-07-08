#!/usr/bin/env bash
# PreToolUse(Bash) guard for unattended Islands runs.
# Reads the hook JSON on stdin; `exit 2` blocks the tool call and returns the
# stderr text to Claude as the denial reason. `exit 0` allows it.
# Defense-in-depth under `dontAsk` (permissions.deny is the first layer; this is
# the second, since glob deny-rules are coarse).
set -euo pipefail

payload="$(cat)"
cmd="$(printf '%s' "$payload" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("tool_input",{}).get("command",""))' 2>/dev/null || true)"

# Never force-push.
if printf '%s' "$cmd" | grep -Eq 'git[[:space:]]+push([[:space:]]|.)*(--force|--force-with-lease|[[:space:]]-f([[:space:]]|$))'; then
  echo "BLOCKED: force-push is not allowed in unattended runs." >&2
  exit 2
fi

# Never push to protected branches — pushes go only to feature/islands*.
if printf '%s' "$cmd" | grep -Eq 'git[[:space:]]+push'; then
  if printf '%s' "$cmd" | grep -Eq '(^|[[:space:]/])(main|develop)([[:space:]]|$|:)'; then
    echo "BLOCKED: push to main/develop is not allowed; push only to feature/islands*." >&2
    exit 2
  fi
fi

# Refuse obviously destructive filesystem ops on absolute/home roots.
if printf '%s' "$cmd" | grep -Eq 'rm[[:space:]]+(-[a-zA-Z]*r[a-zA-Z]*[[:space:]]+)(/|~|\$HOME|/mnt/homes)'; then
  echo "BLOCKED: refusing recursive rm on an absolute/home path." >&2
  exit 2
fi

exit 0
