#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

find_tool() {
  local name="$1"
  if [[ -x ".venv/bin/$name" ]]; then
    printf '.venv/bin/%s\n' "$name"
    return 0
  fi

  if command -v "$name" >/dev/null 2>&1; then
    command -v "$name"
    return 0
  fi

  printf 'Required tool not found: %s\n' "$name" >&2
  printf 'Install development dependencies with: uv pip install -e ".[test]"\n' >&2
  return 1
}

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

RUFF="$(find_tool ruff)"
PYTEST="$(find_tool pytest)"

run "$RUFF" check oneehr/ tests/
run "$RUFF" format --check oneehr/ tests/
run "$PYTEST" tests/ -v --tb=short

if [[ "${ONEEHR_CHECK_DOCS:-0}" == "1" ]]; then
  UV="$(find_tool uv)"
  run "$UV" run --group docs mkdocs build
fi
