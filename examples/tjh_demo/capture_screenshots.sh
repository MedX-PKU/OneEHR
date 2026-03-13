#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_NAME="tjh_tutorial_showcase"
BASELINE_RUN="tjh_tutorial_fast"
LOG_ROOT="$ROOT_DIR/logs"
OUT_DIR="$ROOT_DIR/.cache/tjh_demo/screenshots"
PORT="${PORT:-8765}"
SCREEN_WIDTH="${SCREEN_WIDTH:-2560}"
SCREEN_HEIGHT="${SCREEN_HEIGHT:-1600}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --baseline-run)
      BASELINE_RUN="$2"
      shift 2
      ;;
    --log-root)
      LOG_ROOT="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$OUT_DIR/$RUN_NAME"
SCREEN_DIR="$OUT_DIR/$RUN_NAME"
rm -f "$SCREEN_DIR"/*.png "$SCREEN_DIR"/manifest.json

export LOG_ROOT RUN_NAME SCREEN_DIR BASELINE_RUN SCREEN_WIDTH SCREEN_HEIGHT

CASE_INFO_JSON="$(python - <<'PY'
import json
import os
from pathlib import Path
from urllib.parse import quote

root = Path(os.environ["LOG_ROOT"])
run_name = os.environ["RUN_NAME"]
cases_index = root / run_name / "cases" / "index.json"
case_id = ""
case_route = ""
if cases_index.exists():
    payload = json.loads(cases_index.read_text(encoding="utf-8"))
    records = payload.get("records") or []
    if records:
        case_id = str(records[0]["case_id"])
        case_route = quote(case_id, safe="")
print(json.dumps({"case_id": case_id, "case_route": case_route}))
PY
)"
export CASE_INFO_JSON
CASE_ROUTE="$(python - <<'PY'
import json
import os

print(json.loads(os.environ["CASE_INFO_JSON"])["case_route"])
PY
)"
CASE_ID="$(python - <<'PY'
import json
import os

print(json.loads(os.environ["CASE_INFO_JSON"])["case_id"])
PY
)"
export CASE_ID

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

uv run oneehr webui serve --root "$LOG_ROOT" --host 127.0.0.1 --port "$PORT" >/tmp/oneehr_tjh_demo_webui.log 2>&1 &
SERVER_PID=$!
SERVER_READY=0
for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${PORT}/api/v1/health" >/dev/null 2>&1; then
    SERVER_READY=1
    break
  fi
  sleep 1
done

if [[ "$SERVER_READY" -ne 1 ]]; then
  echo "Web UI server failed to become ready on port ${PORT}" >&2
  exit 1
fi

npx --yes playwright install chromium >/dev/null 2>&1

shot() {
  local url="$1"
  local path="$2"
  npx --yes playwright screenshot \
    --browser chromium \
    --viewport-size "${SCREEN_WIDTH},${SCREEN_HEIGHT}" \
    --wait-for-selector ".page-stack, .run-shell" \
    --wait-for-timeout 4000 \
    "$url" \
    "$path" >/dev/null
}

shot "http://127.0.0.1:${PORT}/" "$SCREEN_DIR/01-runs.png"
shot "http://127.0.0.1:${PORT}/runs/${RUN_NAME}" "$SCREEN_DIR/02-overview.png"
shot "http://127.0.0.1:${PORT}/runs/${RUN_NAME}/analysis/test_audit" "$SCREEN_DIR/03-test-audit.png"
shot "http://127.0.0.1:${PORT}/runs/${RUN_NAME}/analysis/prediction_audit" "$SCREEN_DIR/04-prediction-audit.png"
shot "http://127.0.0.1:${PORT}/runs/${RUN_NAME}/comparison" "$SCREEN_DIR/05-comparison.png"
if [[ -n "$CASE_ROUTE" ]]; then
  shot "http://127.0.0.1:${PORT}/runs/${RUN_NAME}/cases/${CASE_ROUTE}" "$SCREEN_DIR/06-case-detail.png"
fi
shot "http://127.0.0.1:${PORT}/runs/${RUN_NAME}/agents" "$SCREEN_DIR/07-agents.png"

python - <<'PY'
import json
import os
from pathlib import Path

screen_dir = Path(os.environ["SCREEN_DIR"])
payload = {
    "run_name": os.environ["RUN_NAME"],
    "baseline_run": os.environ["BASELINE_RUN"],
    "case_id": os.environ["CASE_ID"],
    "viewport": {"width": int(os.environ.get("SCREEN_WIDTH", "2560")), "height": int(os.environ.get("SCREEN_HEIGHT", "1600"))},
    "screenshots": [
        {"name": path.name, "path": str(path)}
        for path in sorted(screen_dir.glob("*.png"))
    ],
}
(screen_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
