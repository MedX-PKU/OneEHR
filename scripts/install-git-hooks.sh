#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
HOOK_DIR="$(git rev-parse --git-path hooks)"
PRE_PUSH_HOOK="$HOOK_DIR/pre-push"

mkdir -p "$HOOK_DIR"

cat >"$PRE_PUSH_HOOK" <<'HOOK'
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
exec "$ROOT_DIR/scripts/check-ci.sh"
HOOK

chmod +x "$PRE_PUSH_HOOK"

printf 'Installed pre-push hook: %s\n' "$PRE_PUSH_HOOK"
printf 'It runs scripts/check-ci.sh before every git push.\n'
