#!/usr/bin/env bash
# Developer UI launcher (double-click friendly). Requires uv; uses repo .venv.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="$REPO_ROOT/.venv"
RENGA="$VENV/bin/renga"

pause_window() {
  echo ""
  echo "Press Enter to close this window..."
  read -r _ </dev/tty 2>/dev/null || read -r _ || sleep 15
}

fail() {
  echo ""
  echo "ERROR: $*" >&2
  pause_window
  exit 1
}

trap 'echo ""; echo "Script failed at line $LINENO." >&2; pause_window; exit 1' ERR

command -v uv >/dev/null 2>&1 || fail "uv is required. Install from https://docs.astral.sh/uv/"

DEV_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --no-open) DEV_ARGS+=(--no-open) ;;
    --no-reload-api) DEV_ARGS+=(--no-reload-api) ;;
    -h|--help)
      echo "Usage: ./scripts/start-ui-dev.sh [--no-open] [--no-reload-api]"
      echo ""
      echo "  --no-reload-api   Skip Python hot-reload (faster API startup)"
      pause_window
      exit 0
      ;;
    *) fail "Unknown option: $arg" ;;
  esac
done

if [[ ! -x "$RENGA" ]]; then
  echo "==> Setting up UI environment (uv sync --extra ui)..."
  uv sync --extra ui || fail "uv sync failed"
fi

[[ -f "$REPO_ROOT/renga.local.toml" ]] || "$RENGA" init --only-config

export PYTHONUNBUFFERED=1

echo "==> Dev UI (renga ui dev)..."
echo "    Vite:  http://127.0.0.1:5173/"
echo "    API:   http://127.0.0.1:8765/api/v1/  (auto-reload; use --no-reload-api to disable)"
echo ""
set +e
"$RENGA" ui dev --skip-sync "${DEV_ARGS[@]}"
EXIT_CODE=$?
set -e
pause_window
exit "$EXIT_CODE"
